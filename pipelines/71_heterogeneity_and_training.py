#!/usr/bin/env python3
"""Step 71: (A) heterogeneity dial m_alpha = diag((mean s / s_i)^alpha) m over identification, step 26 d', memory,
learned-value interference and NMF recovery; (B) gradient-trained PN->KC log weights (fixed connectivity) with a
softmax odor readout, from real, equalized and amplified starts. Verdicts in docs/PREREGISTRATION_71.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402


def _load(name,file):
    spec=_u.spec_from_file_location(name,Path(__file__).with_name(file));mod=_u.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p69=_load('p69','69_wiring_flatness.py');p68=_load('p68','68_interpretability_ground_truth.py')
p55,p34,p37,cal,p26=p69.p55,p69.p34,p69.p37,p69.cal,p69.p26;p25=p55.p25
NAMES=p69.NAMES;ALPHAS=(-.5,0.,.25,.5,.75,1.);STARTS=(0.,1.,-.5)
SEED0=710000;N_MIX_A=3300;STEPS=1500;PER_ODOR=4;LR=1e-2;RECAL_EVERY=50;CHECK=(0,250,500,1000,1500)
_D={}


def dial(m,alpha):
    """Row scaling by (mean s / s_i)^alpha for KCs with input; alpha 0 = real, 1 = equalize_rows."""
    s=np.asarray(m.sum(1)).ravel();t=s[s>0].mean();f=np.where(s>0,(t/np.where(s>0,s,1))**alpha,0.)
    return (sparse.diags(f)@m).tocsr()


def cv(m):
    s=np.asarray(m.sum(1)).ravel();s=s[s>0];return float(s.std()/s.mean())


def setup(name):
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];c=dict(d['c']);c['kk']=None
    g=p34.make_g(c,d['w'],receptors,delta,sfr,calib)
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(p55.SEED_TEST)))
    return c,g,x_test


def calibrate(m,c):
    receptors,delta,_,calib=_D['inputs'];return cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)


def nmf_recovery(m,c,theta,name):
    receptors,delta,sfr,_=_D['inputs'];W,uniq=p68.receptor_weights(m,c,receptors);has=W.sum(1)>0
    p68.N_MIX=N_MIX_A;rng=np.random.default_rng(p68.SEED0+1000*p68.NAMES.index(name)+1)
    X=p68.receptor_drive(p68.stimuli('M',delta,sfr,rng),uniq,receptors);pn=np.zeros((len(c['mapped']),X.shape[1]));rec=c['mapped'].receptor.to_numpy()
    for j,r in enumerate(uniq):pn[rec==r]=X[j]
    Y=p55.respond(m@pn,c,theta)[has];return p68.recovery(W[has],p68.nmf_atoms(Y,len(uniq),np.random.default_rng(SEED0+1)))[0]


def interference(m,g,theta):
    rows,_=p25.evaluate(m,dict(g,theta=theta));return float(np.mean([r['leak_learned']-r['leak_control'] for r in rows]))


def _run_a(job):
    name,alpha=job;c,g,x_test=setup(name);m=dial(g['m'],alpha);theta=calibrate(m,c)
    ident,mem=p69.performance(m,c,g,theta,x_test);root=_D['root']
    receptors,delta,sfr,calib=_D['inputs']
    keep=g['keep'];n=g['n'];disc=p26.disc;labels=np.repeat(np.arange(n),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,n)
    x=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,keep],sfr,labels,.5,np.random.default_rng(disc.NOISE_SEED)))
    ev=disc.evaluate(p55.respond(m@x,c,theta),labels,train,{'ALL':np.arange(n)})
    kt=p55.respond(m@g['x_full'],c,theta)
    return dict(dataset=name,alpha=alpha,cv=cv(m),ident=ident,mem=mem,dprime26=float(ev['ALL_dprime']),acc26=float(ev['ALL_accuracy']),
                specific=interference(m,g,theta),R_NMF=nmf_recovery(m,c,theta,name),active_kc=int(((kt>1e-8).sum(1)>0).sum()))


# ---------- B: gradient training ----------

def loss_grad(w,ij,shape,x,y,R,b,c,theta):
    """Cross-entropy of softmax(R k + b); gradients wrt synapse weights w (APL inhibition held fixed), R, b."""
    M=sparse.csr_matrix((w,ij),shape=shape);k=p55.respond(M@x,c,theta);z=R@k+b[:,None];z-=z.max(0)
    p=np.exp(z);p/=p.sum(0);B=x.shape[1];loss=float(-np.log(p[y,np.arange(B)]+1e-12).mean())
    dz=p.copy();dz[y,np.arange(B)]-=1;dz/=B
    dE=(R.T@dz)*((k>1e-12)&(k<1-1e-12));G=dE@x.T
    return loss,G[ij[0],ij[1]],dz@k.T,dz.sum(1)


def _adam(state,key,p,g,t,b1=.9,b2=.999):
    m,v=state.setdefault(key,[np.zeros_like(p),np.zeros_like(p)]);m*=b1;m+=(1-b1)*g;v*=b2;v+=(1-b2)*g*g
    return p-LR*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+1e-8)


def _run_b(job):
    name,alpha0=job;c,g,x_test=setup(name);receptors,delta,sfr,_=_D['inputs']
    real=g['m'].tocoo();total=float(real.data.sum());m0=dial(g['m'],alpha0).tocoo()
    ij=(m0.row,m0.col);shape=m0.shape;w=m0.data*total/m0.data.sum();u=np.log(w);s_init=np.asarray(g['m'].sum(1)).ravel()
    M=lambda w:sparse.csr_matrix((w,ij),shape=shape)
    theta=calibrate(M(w),c);i0,_=p69.performance(M(w),c,g,theta,x_test)
    n=g['n'];R=np.zeros((n,shape[0]));b=np.zeros(n);st={};curve=[];rng=np.random.default_rng(SEED0+100*NAMES.index(name)+10*STARTS.index(alpha0))
    for t in range(1,STEPS+1):
        y=np.repeat(np.arange(n),PER_ODOR);x=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,y,.5,rng))
        loss,gw,gR,gb=loss_grad(w,ij,shape,x,y,R,b,c,theta)
        u=_adam(st,'u',u,gw*w,t);R=_adam(st,'R',R,gR,t);b=_adam(st,'b',b,gb,t)
        w=np.exp(u);w*=total/w.sum();u=np.log(w)
        if t%RECAL_EVERY==0:theta=calibrate(M(w),c)
        if t in CHECK or t==1:curve.append(dict(step=t,loss=loss,cv=cv(M(w))))
    mf=M(w);theta=calibrate(mf,c);i1,m1=p69.performance(mf,c,g,theta,x_test);s_fin=np.asarray(mf.sum(1)).ravel();has=s_init>0
    return dict(dataset=name,start=alpha0,cv0=cv(M(m0.data)),cv_final=cv(mf),ident0=i0,ident_final=i1,mem_final=m1,
                specific_final=interference(mf,g,theta),rho_real_strength=float(stats.spearmanr(s_init[has],s_fin[has]).statistic),
                curve=json.dumps(curve))


def verdicts_a(f):
    rho={k:{n:float(stats.spearmanr(q.alpha,q[k]).statistic) for n,q in f.groupby('dataset')} for k in ('ident','specific','R_NMF','dprime26','mem','acc26')}
    cnt=lambda k,s:sum((v<=-.9) if s<0 else (v>=.9) for v in rho[k].values())
    v=dict(A1=cnt('ident',-1)>=4,A2=cnt('specific',-1)>=4,A3=cnt('R_NMF',1)>=4,A4=cnt('dprime26',1)>=4)
    a5={}
    for n,q in f.groupby('dataset'):
        b0=q[q.alpha==0].iloc[0];ok=q[(q.ident>=.99*b0.ident)&(q.specific<=.5*b0.specific)&(q.alpha!=0)]
        a5[n]=ok.alpha.tolist()
    v['A5_alphas']=a5;return v,rho


def verdicts_b(f):
    by=lambda s:f[f.start==s].set_index('dataset')
    ok0=sum(all(by(s).loc[n,'ident_final']>by(s).loc[n,'ident0'] for s in STARTS) for n in NAMES)
    B1=int((by(1.).cv_final>=.2).sum());r,fl=by(0.),by(1.)
    B2=int(((r.cv_final-fl.cv_final).abs()<.5*(r.cv0-fl.cv0).abs()).sum())
    d=np.sign(r.cv_final-r.cv0);B3='increase' if (d>0).sum()>=4 else 'decrease' if (d<0).sum()>=4 else 'mixed'
    return dict(B0=bool(ok0>=4),B0_count=int(ok0),B1=bool(B1>=4),B1_count=B1,B2=bool(B2>=4),B2_count=B2,B3=B3)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/heterogeneity_training';out.mkdir(parents=True,exist_ok=True)
    part=sys.argv[sys.argv.index('--part')+1] if '--part' in sys.argv else 'A'
    p55.p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);r,dl,sf,cb=inputs;_D.update(sets=sets,inputs=(r,dl,np.asarray(sf),cb),root=root)
    if part=='A':
        with Pool(12) as pool:f=pd.DataFrame(pool.map(_run_a,[(n,a) for n in NAMES for a in ALPHAS],chunksize=1))
        f.to_csv(out/'dial.csv',index=False);v,rho=verdicts_a(f)
        (out/'report_A.json').write_text(json.dumps(dict(verdicts=v,rho=rho,rows=f.round(5).to_dict(orient='records')),indent=2,default=float)+'\n')
        pd.set_option('display.width',220);print(f.round(4).to_string());print(json.dumps(v,indent=1,default=float))
    else:
        with Pool(15) as pool:f=pd.DataFrame(pool.map(_run_b,[(n,s) for n in NAMES for s in STARTS],chunksize=1))
        f.to_csv(out/'training.csv',index=False);v=verdicts_b(f)
        (out/'report_B.json').write_text(json.dumps(dict(verdicts=v,rows=f.drop(columns='curve').round(5).to_dict(orient='records')),indent=2,default=float)+'\n')
        pd.set_option('display.width',220);print(f.drop(columns='curve').round(4).to_string());print(json.dumps(v,indent=1))


if __name__=='__main__':main()
