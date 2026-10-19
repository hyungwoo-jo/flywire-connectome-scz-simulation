#!/usr/bin/env python3
"""Step 68: do interpretability methods recover the true wiring? Per-KC attribution (observational ridge, interventional
perturbation) and population decompositions (PCA, ICA, NMF, sparse autoencoder) scored against the connectome's
receptor-level PN->KC weights, in five hemispheres, with strength-preserving null wirings. Verdicts in
docs/PREREGISTRATION_68.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p55',Path(__file__).with_name('55_memory_vs_leak.py'))
p55=_u.module_from_spec(spec);sys.modules['p55']=p55;spec.loader.exec_module(p55)
p34,p37,cal=p55.p34,p55.p37,p55.cal;robust=cal.robust
spec=_u.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'))
p26=_u.module_from_spec(spec);spec.loader.exec_module(p26)

NAMES=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']
SEED0=680000;N_TRIAL=20;CONCS=(1.,.3,.1);N_MIX=6600;EPS=.02;MIN_ACTIVE=20;N_NULL=10
RIDGE=1e-3;SAE_LAMS=(.03,.01,.1);SAE_EPOCHS=100;SAE_BATCH=256;SAE_LR=1e-2  # amended before real runs, see prereg
_D={}


# ---------- interpretability methods (numpy) ----------

def auc(score,pos):
    """AUC of `score` separating pos (True) from neg; ties averaged."""
    pos=np.asarray(pos,bool);r=stats.rankdata(score);n1=pos.sum();n0=len(pos)-n1
    if n1==0 or n0==0:return np.nan
    return float((r[pos].sum()-n1*(n1+1)/2)/(n1*n0))


def ridge(X,Y,lam):
    """X: trials x features, Y: trials x targets. Coefficients on standardized X (features x targets)."""
    Xs=(X-X.mean(0))/np.where(X.std(0)>0,X.std(0),1);Yc=Y-Y.mean(0)
    return np.linalg.solve(Xs.T@Xs+lam*np.eye(X.shape[1]),Xs.T@Yc)


def pca_atoms(Y,K):
    """Y: units x samples. Top-K left singular vectors of the sample-centered data (unit space)."""
    Yc=Y-Y.mean(1,keepdims=True);U,s,_=np.linalg.svd(Yc,full_matrices=False);return U[:,:K],s[:K]


def ica_atoms(Y,K,rng,iters=500,tol=1e-6):
    """Symmetric FastICA (logcosh) after PCA whitening; returns mixing columns in unit space."""
    Yc=Y-Y.mean(1,keepdims=True);U,s,Vt=np.linalg.svd(Yc,full_matrices=False);n=Y.shape[1]
    Z=np.sqrt(n)*Vt[:K];W=np.linalg.qr(rng.normal(size=(K,K)))[0]
    for _ in range(iters):
        G=np.tanh(W@Z);Wn=G@Z.T/n-np.diag((1-G**2).mean(1))@W
        u,_,vt=np.linalg.svd(Wn);Wn=u@vt
        done=np.max(np.abs(np.abs(np.diag(Wn@W.T))-1))<tol;W=Wn
        if done:break
    return U[:,:K]*s[:K]@W.T/np.sqrt(n)


def nmf_atoms(Y,K,rng,iters=500):
    Y=np.maximum(Y,0);sc=np.sqrt(Y.mean()/K)
    W=np.abs(rng.normal(size=(Y.shape[0],K)))*sc+1e-6;H=np.abs(rng.normal(size=(K,Y.shape[1])))*sc+1e-6
    for _ in range(iters):
        H*=(W.T@Y)/(W.T@W@H+1e-12);W*=(Y@H.T)/(W@(H@H.T)+1e-12)
    return W


def sae_atoms(Y,L,lam,rng,epochs=SAE_EPOCHS,batch=SAE_BATCH,lr=SAE_LR,center=False):
    """ReLU sparse autoencoder, unit-norm decoder rows; loss mean||x-xh||^2 + lam*mean||h||_1 on data scaled to
    mean squared sample norm 1 (so lam is scale-free). Y: units x samples."""
    X=Y.T.astype(np.float64)
    if center:X=X-X.mean(0)  # post hoc fix: uncentered KC data collapsed every latent
    X=X/np.sqrt(np.mean(np.sum(X**2,1)));n,d=X.shape
    Wd=rng.normal(size=(L,d));Wd/=np.linalg.norm(Wd,axis=1,keepdims=True);We=Wd.T.copy();be=np.zeros(L);bd=X.mean(0)
    P=[We,be,Wd,bd];M=[np.zeros_like(p) for p in P];V=[np.zeros_like(p) for p in P];t=0;b1,b2=.9,.999
    for _ in range(epochs):
        for ix in np.array_split(rng.permutation(n),max(1,n//batch)):
            x=X[ix];B=len(ix);pre=x@We+be;h=np.maximum(pre,0);e=h@Wd+bd-x
            gWd=2*h.T@e/B;gbd=2*e.mean(0);dh=(2*e@Wd.T+lam*np.sign(h))/B*(pre>0)
            G=[x.T@dh,dh.sum(0),gWd,gbd];t+=1
            for i,(p,g) in enumerate(zip(P,G)):
                M[i]=b1*M[i]+(1-b1)*g;V[i]=b2*V[i]+(1-b2)*g*g
                p-=lr*(M[i]/(1-b1**t))/(np.sqrt(V[i]/(1-b2**t))+1e-8)
            Wd/=np.linalg.norm(Wd,axis=1,keepdims=True)+1e-12
    h=np.maximum(X@We+be,0);dead=float(np.mean(h.max(0)<=0));mse=float(np.mean(np.sum((h@Wd+bd-X)**2,1)))
    return Wd.T,dict(dead=dead,mse=mse,l0=float((h>0).sum(1).mean()))


def recovery(true,learned):
    """Mean over true atoms (columns) of max |cosine| with learned atoms; also per-atom values."""
    a=true/np.maximum(np.linalg.norm(true,axis=0),1e-12);b=learned/np.maximum(np.linalg.norm(learned,axis=0),1e-12)
    per=np.abs(a.T@b).max(1);return float(per.mean()),per


def coherence(true):
    a=true/np.maximum(np.linalg.norm(true,axis=0),1e-12);c=np.abs(a.T@a);np.fill_diagonal(c,np.nan)
    return float(np.nanmean(c)),np.nanmax(c,1)


# ---------- circuit, stimuli ----------

def receptor_weights(m,c,receptors):
    """W_true (KC x receptor) summing mapped PN columns per receptor; receptor order = sorted unique mapped."""
    rec=c['mapped'].receptor.to_numpy();uniq=[r for r in receptors if r in set(rec)];m=m.tocsc()
    W=np.column_stack([np.asarray(m[:,np.flatnonzero(rec==r)].sum(1)).ravel() for r in uniq]);return W,uniq


def stimuli(regime,delta,sfr,rng):
    """Noisy baseline-subtracted ORN rates (24 x trials)."""
    if regime=='N':
        cols=np.concatenate([cc*delta for cc in CONCS],1);lab=np.repeat(np.arange(cols.shape[1]),N_TRIAL)
    else:
        n=delta.shape[1];cols=np.zeros((delta.shape[0],N_MIX))
        for t in range(N_MIX):
            k=rng.integers(1,4);o=rng.choice(n,k,replace=False);cols[:,t]=(np.exp(rng.uniform(np.log(.1),0,k))*delta[:,o]).sum(1)
        lab=np.arange(N_MIX)
    return cal.sample_orn(cols,sfr,lab,.5,rng)


def receptor_drive(orn,uniq,receptors):
    return (cal.olsen(orn)/cal.OLSEN['rmax'])[[receptors.index(r) for r in uniq]]


def solve(E,c,theta):return p55.respond(E,c,theta)


def build(name,graph):
    """graph: 'real', 'flat' or ('null', i). Returns c, m, theta, W_true, receptor list."""
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];c=dict(d['c']);c['kk']=None
    m,mask=cal.partial_matrix(c)
    if graph=='flat':m=p26.equalize_rows(m)
    elif graph!='real':
        rng=np.random.default_rng(SEED0+50000+1000*NAMES.index(name)+graph[1])
        target,_=robust.strength_preserving_null(c['pre'][mask],c['post'][mask],c['w'][mask],rng);m,_=cal.partial_matrix(c,target)
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    W,uniq=receptor_weights(m,c,receptors);return c,m.tocsr(),theta,W,uniq


def decompose(Y,W,K,rng,lams):
    rows=[];per={}
    for meth in ('PCA','ICA','NMF'):
        A=pca_atoms(Y,K)[0] if meth=='PCA' else ica_atoms(Y,K,rng) if meth=='ICA' else nmf_atoms(Y,K,rng)
        R,pa=recovery(W,A);rows.append(dict(method=meth,lam=np.nan,R=R,n_atoms=K));per[meth]=pa
    for lam in lams:
        A,info=sae_atoms(Y,2*K,lam,rng);R,pa=recovery(W,A);rows.append(dict(method='SAE',lam=lam,R=R,n_atoms=2*K,**info))
        if lam==SAE_LAMS[0]:per['SAE']=pa
    for n in sorted({K,2*K}):rows.append(dict(method=f'chance_{n}',lam=np.nan,R=recovery(W,rng.normal(size=(W.shape[0],n)))[0],n_atoms=n))
    return rows,per


def _run(job):
    name,graph,regimes=job;receptors,delta,sfr,_=_D['inputs'];c,m,theta,W,uniq=build(name,graph);K=len(uniq)
    has=W.sum(1)>0;Wk=W[has];gname=graph if isinstance(graph,str) else f'null_{graph[1]}'
    coh,maxcoh=coherence(Wk);dec=[];kcs=[];atoms=[]
    for reg in regimes:
        rng=np.random.default_rng(SEED0+1000*NAMES.index(name)+(0 if reg=='N' else 1))  # same stimuli across graphs
        X=receptor_drive(stimuli(reg,delta,sfr,rng),uniq,receptors)
        pn_full=np.zeros((len(c['mapped']),X.shape[1]));rec=c['mapped'].receptor.to_numpy()
        for j,r in enumerate(uniq):pn_full[rec==r]=X[j]
        E=m@pn_full;k=solve(E,c,theta);Y=k[has]
        mrng=np.random.default_rng(SEED0+7+1000*NAMES.index(name)+(0 if reg=='N' else 1))
        rows,per=decompose(Y,Wk,K,mrng,SAE_LAMS if (graph=='real') else SAE_LAMS[:1])
        dec+=[dict(dataset=name,graph=gname,regime=reg,coherence=coh,**r) for r in rows]
        if graph=='real' and reg=='M':
            atoms=[dict(dataset=name,receptor=u,R_SAE=float(per['SAE'][j]),R_ICA=float(per['ICA'][j]),max_coh=float(maxcoh[j])) for j,u in enumerate(uniq)]
        if graph in ('real','flat'):
            act=(k>1e-8);idx=np.flatnonzero(has&(act.sum(1)>=MIN_ACTIVE)&((W>0).sum(1)<K))
            B=ridge(X.T,k[idx].T,RIDGE*X.shape[1])
            G=np.zeros((len(idx),K))
            for j in range(K):
                kp=solve(E+EPS*W[:,j][:,None],c,theta)[idx];a=act[idx]
                G[:,j]=((kp-k[idx])/EPS*a).sum(1)/a.sum(1)
            for q,i in enumerate(idx):
                pos=W[i]>0;b=B[:,q]
                kcs.append(dict(dataset=name,graph=gname,regime=reg,kc=int(i),n_inputs=int(pos.sum()),strength=float(W[i].sum()),
                                n_active=int(act[i].sum()),auc_obs=auc(b,pos),auc_int=auc(G[q],pos),
                                n_est=int((b>.25*b.max()).sum()) if b.max()>0 else 0))
    return dec,kcs,atoms


def _run_sae(job):
    """Post hoc: centered SAE only (main lambda), same stimuli and graphs as _run."""
    name,graph,regimes=job;receptors,delta,sfr,_=_D['inputs'];c,m,theta,W,uniq=build(name,graph);K=len(uniq)
    has=W.sum(1)>0;Wk=W[has];gname=graph if isinstance(graph,str) else f'null_{graph[1]}';rows=[];atoms=[]
    for reg in regimes:
        rng=np.random.default_rng(SEED0+1000*NAMES.index(name)+(0 if reg=='N' else 1))
        X=receptor_drive(stimuli(reg,delta,sfr,rng),uniq,receptors);pn=np.zeros((len(c['mapped']),X.shape[1]));rec=c['mapped'].receptor.to_numpy()
        for j,r in enumerate(uniq):pn[rec==r]=X[j]
        Y=solve(m@pn,c,theta)[has];mrng=np.random.default_rng(SEED0+7+1000*NAMES.index(name)+(0 if reg=='N' else 1))
        for lam in (SAE_LAMS if graph=='real' else SAE_LAMS[:1]):
            A,info=sae_atoms(Y,2*K,lam,mrng,center=True);R,pa=recovery(Wk,A)
            rows.append(dict(dataset=name,graph=gname,regime=reg,method='SAE_centered',lam=lam,R=R,n_atoms=2*K,**info))
            if graph=='real' and reg=='M' and lam==SAE_LAMS[0]:atoms=[dict(dataset=name,receptor=u,R_SAEc=float(pa[j])) for j,u in enumerate(uniq)]
    return rows,atoms


def posthoc(root,out):
    p55.p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);receptors,delta,sfr,calib=inputs;_D.update(sets=sets,inputs=(receptors,delta,np.asarray(sfr),calib))
    jobs=[(n,'real',('N','M')) for n in NAMES]+[(n,'flat',('M',)) for n in NAMES]+[(n,('null',i),('M',)) for n in NAMES for i in range(N_NULL)]
    with Pool(12) as pool:res=pool.map(_run_sae,jobs,chunksize=1)
    f=pd.DataFrame([r for x in res for r in x[0]]);f.to_csv(out/'decomposition_sae_centered_posthoc.csv',index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(out/'per_atom_sae_centered_posthoc.csv',index=False);print(f.round(3).to_string())


def verdicts(dec,kc):
    med=kc[kc.graph=='real'].groupby(['dataset','regime'])[['auc_obs','auc_int']].median()
    I1=bool((med.auc_int>=.95).all())
    obsN=med.xs('N',level='regime').auc_obs;obsM=med.xs('M',level='regime').auc_obs
    I2=bool(((obsN<obsM).sum()>=4) and ((obsN<.9).sum()>=4))
    r=dec[(dec.graph=='real')&(dec.regime=='M')]
    R=lambda meth:r[(r.method==meth)&((r.lam==SAE_LAMS[0])|r.lam.isna())].set_index('dataset').R
    I3=bool(((R('SAE')>R('PCA'))&(R('ICA')>R('PCA'))).sum()>=4)
    nl=dec[dec.graph.str.startswith('null')&(dec.regime=='M')];coh_null=nl.groupby('dataset').coherence.median()
    sae_null=nl[nl.method=='SAE'].groupby('dataset').R.median();coh_real=r.groupby('dataset').coherence.first()
    I4=bool(((coh_real>coh_null).sum()>=4) and ((R('SAE')<sae_null).sum()>=4))
    k5=kc[(kc.graph=='real')&(kc.regime=='N')];rho={n:float(stats.spearmanr(q.auc_obs,q.strength).statistic) for n,q in k5.groupby('dataset')}
    pos=sum(v>.1 for v in rho.values());neg=sum(v<-.1 for v in rho.values())
    I5=dict(consistent=bool(pos>=4 or neg>=4),direction='positive' if pos>=4 else 'negative' if neg>=4 else 'none',rho=rho)
    detail=dict(median_auc=med.round(4).reset_index().to_dict(orient='records'),coherence_real=coh_real.to_dict(),coherence_null_median=coh_null.to_dict(),
                R_real={m:R(m).to_dict() for m in ('PCA','ICA','NMF','SAE')},R_SAE_null_median=sae_null.to_dict())
    return dict(I1=I1,I2=I2,I3=I3,I4=I4,I5=I5),detail


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/interpretability_ground_truth';out.mkdir(parents=True,exist_ok=True)
    if '--sae-posthoc' in sys.argv:return posthoc(root,out)
    p55.p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);receptors,delta,sfr,calib=inputs;_D.update(sets=sets,inputs=(receptors,delta,np.asarray(sfr),calib))
    jobs=[(n,'real',('N','M')) for n in NAMES]+[(n,'flat',('M',)) for n in NAMES]+[(n,('null',i),('M',)) for n in NAMES for i in range(N_NULL)]
    with Pool(12) as pool:res=pool.map(_run,jobs,chunksize=1)
    dec=pd.DataFrame([r for x in res for r in x[0]]);kc=pd.DataFrame([r for x in res for r in x[1]]);at=pd.DataFrame([r for x in res for r in x[2]])
    dec.to_csv(out/'decomposition.csv',index=False);kc.to_csv(out/'per_kc.csv',index=False);at.to_csv(out/'per_atom.csv',index=False)
    v,detail=verdicts(dec,kc)
    rep=dict(atom_rho={n:dict(R_SAE=float(stats.spearmanr(q.R_SAE,q.max_coh).statistic),R_ICA=float(stats.spearmanr(q.R_ICA,q.max_coh).statistic)) for n,q in at.groupby('dataset')},
             n_est_rho={f'{n}_{g}_{rg}':float(stats.spearmanr(q.n_est,q.n_inputs).statistic) for (n,g,rg),q in kc.groupby(['dataset','graph','regime'])},
             flat=dec[dec.graph=='flat'].round(4).to_dict(orient='records'),
             flat_auc=kc[kc.graph=='flat'].groupby('dataset')[['auc_obs','auc_int']].median().round(4).to_dict(orient='index'),
             sae_lambda=dec[(dec.graph=='real')&(dec.method=='SAE')].round(4).to_dict(orient='records'))
    report=dict(verdicts=v,detail=detail,report_only=rep)
    (out/'report.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
    pd.set_option('display.width',220);print(dec[dec.graph=='real'].round(3).to_string());print(json.dumps(v,indent=1,default=float));print(json.dumps(detail,indent=1,default=float))


if __name__=='__main__':main()
