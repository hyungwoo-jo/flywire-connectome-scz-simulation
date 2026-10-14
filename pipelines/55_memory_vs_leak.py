#!/usr/bin/env python3
"""Step 55: do the compensations that rescue memory performance (Abdelrahman, Vasilaki, Lin 2021) also remove the learned-value
leak? Five hemispheres x four conditions (original, flattened strength, KC-wise threshold homeostasis, KC-wise input-gain
homeostasis). Memory task in their format: all 73 evaluation odors get random valence; approach / avoid MBONs read all KCs
uniformly; learning depresses KC->MBON weights (w -> w exp(-5 k)) on clean odor responses; accuracy on noisy full-strength
trials. Leak: step-43 H (MBON11 weights) and step-25 simulated leak. Verdicts in docs/PREREGISTRATION_55.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402


def load(name,file):
    spec=_u.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=_u.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p43=load('p43','43_leakage_theory.py');p37=p43.p37;p34=p43.p34;p25=p43.p25;p24=p43.p24;cal=p43.cal
p29=load('p29','29_input_compensation.py')
NAMES=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']
CONDITIONS=('original','flat','theta_homeo','gain_homeo','theta_homeo_s','gain_homeo_s')   # _s: sparsity re-matched to 10% (deviation, see results doc)
N_DRAW=10;N_SMALL=20;ITERS=300;LR_THETA=.5;LR_GAIN=.05
SEED_VAL=550000;SEED_TEST=550100
_D={}


def respond(E,c,theta):
    return cal.base.response(E,c['inh'],c['drive'],theta,1.,gain=cal.APL_GAIN)[0]


def homeostasis(E,c,theta0,tune,a0,kind):
    """Tune per-KC threshold or input gain so each tuned KC's mean activity over calibration odors approaches a0."""
    th=np.full(E.shape[0],theta0);lg=np.zeros(E.shape[0])
    for _ in range(ITERS):
        k=respond(E*np.exp(lg)[:,None],c,th[:,None]);mean=k.mean(1)
        if kind=='theta':th[tune]=np.maximum(th[tune]+LR_THETA*(mean[tune]-a0),0.)
        else:lg[tune]=np.clip(lg[tune]+LR_GAIN*(a0-mean[tune])/a0,np.log(.01),np.log(100.))
    mean=respond(E*np.exp(lg)[:,None],c,th[:,None]).mean(1)
    return th,np.exp(lg),float(mean[tune].std()/mean[tune].mean())


def offset_to_target(E,c,th,target=.10):
    """Common shift added to per-KC thresholds so that a fraction `target` of KC-odor pairs is active."""
    lo,hi=-float(th.max()),float(E.max())
    for _ in range(40):
        dlt=(lo+hi)/2
        if np.mean(respond(E,c,(th+dlt)[:,None])>1e-8)>target:lo=dlt
        else:hi=dlt
    return (lo+hi)/2


def memory(kt,kte,lab,n_odor,n_learn):
    """Abdelrahman-style valence task; returns accuracy for each of N_DRAW random valence draws."""
    acc=[]
    for r in range(N_DRAW):
        rng=np.random.default_rng(SEED_VAL+r);odors=rng.choice(n_odor,n_learn,replace=False);val=rng.integers(0,2,n_learn)
        wa=np.ones(kt.shape[0]);wv=np.ones(kt.shape[0])
        for o,v in zip(odors,val):
            if v:wa=wa*np.exp(-p25.ETA*kt[:,o])
            else:wv=wv*np.exp(-p25.ETA*kt[:,o])
        vmap=dict(zip(odors,val));sel=np.isin(lab,odors);k=kte[:,sel];truth=np.array([vmap[o] for o in lab[sel]])
        diff=wv@k-wa@k;choice=np.where(diff>0,1.,np.where(diff<0,0.,.5))
        acc.append(float(np.mean(np.where(choice==.5,.5,choice==truth))))
    return acc


def _run(job):
    name,kind=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    xcal=cal.pn_drive(c,receptors,delta[:,calib]);theta0=cal.calibrate(m@xcal,c['inh'],c['drive'],.10)
    s=np.asarray(m.sum(1)).ravel();tune=s>0;a0=float(respond(m@xcal,c,theta0)[tune].mean())
    theta=None;cv=np.nan
    if kind=='flat':m=p29.compensate(m,1.)
    elif kind.startswith('theta_homeo'):
        th,_,cv=homeostasis(m@xcal,c,theta0,tune,a0,'theta')
        if kind.endswith('_s'):th=np.maximum(th+offset_to_target(m@xcal,c,th),0.)
        theta=th[:,None]
    elif kind.startswith('gain_homeo'):
        _,gain,cv=homeostasis(m@xcal,c,theta0,tune,a0,'gain');m=sparse.diags(gain)@m
        theta=cal.calibrate(m@xcal,c['inh'],c['drive'],.10) if kind.endswith('_s') else theta0
    rows=p43.evaluate(name,c,g['w'],g,m_override=m,label=kind,theta=theta)
    if theta is None:theta=cal.calibrate(m@xcal,c['inh'],c['drive'],.10)
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(SEED_TEST)))
    kt=respond(m@g['x_full'],c,theta);kte=respond(m@x_test,c,theta);kn=respond(m@g['x_noise'],c,theta)
    aa=memory(kt,kte,g['lab'],g['n'],g['n']);a20=memory(kt,kte,g['lab'],g['n'],N_SMALL)
    return dict(dataset=name,condition=kind,acc_all=float(np.mean(aa)),acc_20=float(np.mean(a20)),acc_all_draws=json.dumps(aa),acc_20_draws=json.dumps(a20),
                H=float(np.nanmean([r['H'] for r in rows])),H_median=float(np.nanmedian([r['H'] for r in rows])),
                leak=float(np.mean([r['leak_learned'] for r in rows])),specific=float(np.mean([r['specific'] for r in rows])),
                active_odor=float((kt>1e-8).mean()),active_noise=float((kn>1e-8).mean()),
                silent_kc=float(((kt>1e-8).sum(1)==0)[tune].mean()),homeo_cv=cv)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/memory_vs_leak';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    with Pool(10) as pool:f=pd.DataFrame(pool.map(_run,[(n,k) for n in NAMES for k in CONDITIONS],chunksize=1))
    f.to_csv(out/'per_condition.csv',index=False)
    diffs=[]
    for n in NAMES:
        base={k:np.array(json.loads(f[(f.dataset==n)&(f.condition=='original')][k].iloc[0])) for k in ('acc_all_draws','acc_20_draws')}
        for cond in CONDITIONS[1:]:
            r=f[(f.dataset==n)&(f.condition==cond)].iloc[0]
            for k in base:
                dd=np.array(json.loads(r[k]))-base[k];diffs.append(dict(dataset=n,condition=cond,task=k[:-6],diff=float(dd.mean()),se=float(dd.std(ddof=1)/np.sqrt(len(dd)))))
    pd.DataFrame(diffs).to_csv(out/'accuracy_diff_vs_original.csv',index=False)
    piv=lambda col:f.pivot(index='dataset',columns='condition',values=col)
    A,H=piv('acc_all'),piv('H')
    pre=['original','flat','theta_homeo','gain_homeo'];rho={n:float(stats.spearmanr(A.loc[n,pre],H.loc[n,pre]).statistic) for n in NAMES}
    sm=['original','flat','theta_homeo_s','gain_homeo_s'];rho_s={n:float(stats.spearmanr(A.loc[n,sm],H.loc[n,sm]).statistic) for n in NAMES}
    v_s=dict(P1=bool(((A.theta_homeo_s>A.original).sum()>=4) and ((A.gain_homeo_s>A.original).sum()>=4)),P2a=bool((H.theta_homeo_s<H.original).all()),
             P2b='odor_homeostasis_removes_leak' if (H.theta_homeo_s<=2).sum()>=4 else 'odor_homeostasis_leaves_leak',
             P3='aligned' if np.mean(list(rho_s.values()))<=-.6 else 'dissociated')
    v=dict(P1=bool(((A.theta_homeo>A.original).sum()>=4) and ((A.gain_homeo>A.original).sum()>=4)),
           P2a=bool((H.theta_homeo<H.original).all()),
           P2b='odor_homeostasis_removes_leak' if (H.theta_homeo<=2).sum()>=4 else 'odor_homeostasis_leaves_leak',
           P3='aligned' if np.mean(list(rho.values()))<=-.6 else 'dissociated')
    report=dict(verdicts=v,verdicts_sparsity_matched=v_s,rho_acc_H=rho,rho_mean=float(np.mean(list(rho.values()))),rho_acc_H_sparsity_matched=rho_s,rows=f.round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',220);print(f.round(4).to_string());print(json.dumps(dict(prereg=v,sparsity_matched=v_s),indent=2),rho,rho_s)


if __name__=='__main__':main()
