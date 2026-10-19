#!/usr/bin/env python3
"""Step 72: train PN->KC log weights (step 71 B setup) on cross-entropy + lambda * D, where D is a differentiable
proxy of learned-value interference: excess spontaneous activity in odor-responsive KCs. Adds a shuffled-label
drift control. Verdicts in docs/PREREGISTRATION_72.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p71',Path(__file__).with_name('71_heterogeneity_and_training.py'))
p71=_u.module_from_spec(spec);sys.modules['p71']=p71;spec.loader.exec_module(p71)
p69,p55,p37,cal,p25=p71.p69,p71.p55,p71.p37,p71.cal,p71.p25
NAMES=p71.NAMES;LAMS=(0.,.01,.03,.1,.3,1.,3.,10.);STARTS=(0.,1.)
SEED_SP=720000;SEED_SHUF=721000;N_SP=64;CHECK=(250,500,1000,1500);EPS=1e-12


def proxy(kt,kn):
    """D = (sum t n / sum t) / mean n - 1 with t, n the per-KC means of odor and spontaneous responses."""
    t=kt.mean(1);n=kn.mean(1);T=t.sum()+EPS;nm=n.mean()+EPS;return float((t@n)/(T*nm)-1)


def penalty_grad(w,ij,shape,x_odor,x_sp,c,theta):
    """D and dD/dw with APL inhibition held fixed (same approximation as p71.loss_grad)."""
    M=sparse.csr_matrix((w,ij),shape=shape);kt=p55.respond(M@x_odor,c,theta);kn=p55.respond(M@x_sp,c,theta)
    t=kt.mean(1);n=kn.mean(1);T=t.sum()+EPS;nm=n.mean()+EPS;P=t@n;D=P/(T*nm)-1
    dt=n/(T*nm)-P/(T*T*nm);dn=t/(T*nm)-P/(T*nm*nm)/len(n)
    gate=lambda k:(k>1e-12)&(k<1-1e-12)
    G=((dt[:,None]*gate(kt))/kt.shape[1])@x_odor.T+((dn[:,None]*gate(kn))/kn.shape[1])@x_sp.T
    return float(D),G[ij[0],ij[1]]


def evaluate(M,c,g,theta,x_test):
    ident,mem=p69.performance(M,c,g,theta,x_test)
    D=proxy(p55.respond(M@g['x_full'],c,theta),p55.respond(M@g['x_noise'],c,theta))
    return ident,mem,p71.interference(M,g,theta),D


def _run_gate(job):
    name,alpha=job;c,g,_=p71.setup(name);m=p71.dial(g['m'],alpha);theta=p71.calibrate(m,c)
    D=proxy(p55.respond(m@g['x_full'],c,theta),p55.respond(m@g['x_noise'],c,theta))
    return dict(dataset=name,alpha=alpha,D=D)


def _run(job):
    name,alpha0,lam,shuffle=job;c,g,x_test=p71.setup(name);receptors,delta,sfr,_=p71._D['inputs']
    real=g['m'].tocoo();total=float(real.data.sum());m0=p71.dial(g['m'],alpha0).tocoo()
    ij=(m0.row,m0.col);shape=m0.shape;w=m0.data*total/m0.data.sum();u=np.log(w)
    M=lambda w:sparse.csr_matrix((w,ij),shape=shape)
    n=g['n'];R=np.zeros((n,shape[0]));b=np.zeros(n);st={};curve=[]
    i=NAMES.index(name);rng=np.random.default_rng(p71.SEED0+100*i+10*p71.STARTS.index(alpha0))
    rsp=np.random.default_rng(SEED_SP+100*i+10*STARTS.index(alpha0)+LAMS.index(lam));rsh=np.random.default_rng(SEED_SHUF+i)
    zero=np.zeros((delta.shape[0],1));theta=p71.calibrate(M(w),c);cv0=p71.cv(M(w))
    for t in range(1,p71.STEPS+1):
        y=np.repeat(np.arange(n),p71.PER_ODOR);x=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,y,.5,rng))
        yl=rsh.integers(0,n,len(y)) if shuffle else y
        loss,gw,gR,gb=p71.loss_grad(w,ij,shape,x,yl,R,b,c,theta);D=np.nan
        if lam>0:
            xs=cal.pn_drive(c,receptors,cal.sample_orn(zero,sfr,np.zeros(N_SP,int),.5,rsp))
            D,gd=penalty_grad(w,ij,shape,g['x_full'],xs,c,theta);gw=gw+lam*gd
        u=p71._adam(st,'u',u,gw*w,t);R=p71._adam(st,'R',R,gR,t);b=p71._adam(st,'b',b,gb,t)
        w=np.exp(u);w*=total/w.sum();u=np.log(w)
        if t%p71.RECAL_EVERY==0:theta=p71.calibrate(M(w),c)
        if t in CHECK or t==1:curve.append(dict(step=t,loss=loss,D=D,cv=p71.cv(M(w))))
    mf=M(w);theta=p71.calibrate(mf,c);ident,mem,spec_,Df=evaluate(mf,c,g,theta,x_test)
    return dict(dataset=name,start=alpha0,lam=lam,shuffle=shuffle,cv0=cv0,cv_final=p71.cv(mf),
                cv250=next(r['cv'] for r in curve if r['step']==250),ident_final=ident,mem_final=mem,
                specific_final=spec_,D_final=Df,curve=json.dumps(curve))


def lam_star(lams,cvs,target):
    """log-linear interpolation of lambda where cv crosses target; None if target is outside the range."""
    x=np.log(np.asarray(lams));y=np.asarray(cvs)
    for k in range(len(x)-1):
        lo,hi=sorted((y[k],y[k+1]))
        if lo<=target<=hi:
            f=0. if y[k+1]==y[k] else (target-y[k])/(y[k+1]-y[k]);return float(np.exp(x[k]+f*(x[k+1]-x[k])))
    return None


def verdicts(f,real_cv,cvcol='cv_final'):
    main=f[~f.shuffle];r=main[main.start==0.];fl=main[main.start==1.];v={}
    rho=lambda col:{n:float(stats.spearmanr(q.lam,q[col]).statistic) for n,q in r.groupby('dataset')}
    v['C1_rho']=rho('specific_final');v['C1']=sum(x<=-.9 for x in v['C1_rho'].values())>=4
    v['C2_rho']=rho(cvcol);v['C2']=sum(x<=-.9 for x in v['C2_rho'].values())>=4
    ls={}
    for n,q in r[r.lam>0].sort_values('lam').groupby('dataset'):ls[n]=lam_star(q.lam,q[cvcol],real_cv[n])
    ok=[x for x in ls.values() if x is not None];v['lam_star']=ls
    v['C3']=bool(len(ok)>=4 and max(ok)/min(ok)<=3)
    c4={}
    for n in NAMES:
        if ls[n] is None:c4[n]=False;continue
        lg=min((x for x in LAMS if x>0),key=lambda x:abs(np.log(x/ls[n])))
        a=r[(r.dataset==n)&(r.lam==lg)].iloc[0];b=fl[(fl.dataset==n)&(fl.lam==lg)].iloc[0]
        c4[n]=bool(abs(a[cvcol]-b[cvcol])<.5*abs(a.cv0-b.cv0))
    v['C4_by']=c4;v['C4']=sum(c4.values())>=4
    return v


def drift(f):
    s=f[f.shuffle].set_index('dataset');t=f[(~f.shuffle)&(f.start==0.)&(f.lam==0.)].set_index('dataset')
    by={n:dict(task=float(t.loc[n,'cv_final']-t.loc[n,'cv0']),shuffled=float(s.loc[n,'cv_final']-s.loc[n,'cv0']),
               task250=float(t.loc[n,'cv250']-t.loc[n,'cv0']),shuffled250=float(s.loc[n,'cv250']-s.loc[n,'cv0'])) for n in NAMES}
    flag=sum(d['shuffled']>=.5*d['task'] for d in by.values())>=3
    return dict(D1_contaminated=bool(flag),by=by)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/joint_objective';out.mkdir(parents=True,exist_ok=True)
    part=sys.argv[sys.argv.index('--part')+1] if '--part' in sys.argv else 'gate'
    p55.p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);r,dl,sf,cb=inputs;p71._D.update(sets=sets,inputs=(r,dl,np.asarray(sf),cb),root=root)
    dial=pd.read_csv(root/'qc_reports/heterogeneity_training/dial.csv')
    if part=='gate':
        with Pool(12) as pool:f=pd.DataFrame(pool.map(_run_gate,[(n,a) for n in NAMES for a in p71.ALPHAS],chunksize=1))
        f=f.merge(dial[['dataset','alpha','specific','cv']],on=['dataset','alpha']);f.to_csv(out/'gate.csv',index=False)
        rho={n:float(stats.spearmanr(q.D,q.specific).statistic) for n,q in f.groupby('dataset')}
        v=dict(C0=sum(x>=.9 for x in rho.values())>=4,rho=rho);(out/'report_gate.json').write_text(json.dumps(v,indent=2)+'\n')
        print(f.round(4).to_string());print(json.dumps(v,indent=1))
    else:
        jobs=[(n,s,l,False) for n in NAMES for s in STARTS for l in LAMS]+[(n,0.,0.,True) for n in NAMES]
        jobs.sort(key=lambda j:-j[2])
        with Pool(15) as pool:f=pd.DataFrame(pool.map(_run,jobs,chunksize=1))
        f.to_csv(out/'training.csv',index=False)
        real_cv=dial[dial.alpha==0].set_index('dataset').cv.to_dict();dr=drift(f)
        v=dict(main=verdicts(f,real_cv),drift=dr)
        if dr['D1_contaminated']:v['at_step250']=verdicts(f,real_cv,'cv250')
        t71=pd.read_csv(root/'qc_reports/heterogeneity_training/training.csv')
        rep=f[(~f.shuffle)&(f.lam==0.)].merge(t71[['dataset','start','cv_final']],on=['dataset','start'],suffixes=('','_71'))
        v['reproduces_71']=bool(np.allclose(rep.cv_final,rep.cv_final_71,rtol=1e-6))
        (out/'report.json').write_text(json.dumps(dict(verdicts=v,rows=f.drop(columns='curve').round(5).to_dict(orient='records')),indent=2,default=float)+'\n')
        pd.set_option('display.width',220);print(f.drop(columns='curve').round(4).to_string());print(json.dumps(v,indent=1,default=float))


if __name__=='__main__':main()
