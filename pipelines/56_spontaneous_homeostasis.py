#!/usr/bin/env python3
"""Step 56: KC-wise thresholds set from each KC's own spontaneous input distribution.
theta_i(lambda) = (1 - lambda) theta_0 + lambda (mu_i + z sigma_i); mu_i, sigma_i from 1,000 calibration noise trials,
z common and set for 10% active KCs on calibration odors. lambda in {0, .25, .5, .75, 1}; step 55's sparsity-matched
threshold homeostasis as reference. Leak (step 43 H, step 25 simulation) and memory (step 55 task).
Verdicts in docs/PREREGISTRATION_56.md."""
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
p43,p34,p25,p24,cal=p55.p43,p55.p34,p55.p25,p55.p24,p55.cal
NAMES=p55.NAMES;LAMBDAS=(0.,.25,.5,.75,1.)
_D=p55._D


def z_for_target(E,c,base,spread,target=.10):
    """Common z so that theta = base + z * spread gives a fraction `target` of active KC-odor pairs."""
    lo,hi=-50.,50.
    for _ in range(50):
        z=(lo+hi)/2
        if np.mean(p55.respond(E,c,(base+z*spread)[:,None])>1e-8)>target:lo=z
        else:hi=z
    return (lo+hi)/2


def _run(job):
    name,kind,lam=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    xcal=cal.pn_drive(c,receptors,delta[:,calib]);E=m@xcal;theta0=cal.calibrate(E,c['inh'],c['drive'],.10)
    s=np.asarray(m.sum(1)).ravel();tune=s>0
    Ec=np.asarray(m@g['x_cal']);mu,sd=Ec.mean(1),Ec.std(1);z=np.nan
    if kind=='spont':
        base=(1-lam)*theta0+lam*mu;spread=lam*sd
        if lam==0:th=np.full(len(s),theta0)
        else:z=z_for_target(E,c,base,spread);th=np.maximum(base+z*spread,0.)
    else:                                                     # step 55 reference: odor-activity homeostasis, sparsity matched
        a0=float(p55.respond(E,c,theta0)[tune].mean());th,_,_=p55.homeostasis(E,c,theta0,tune,a0,'theta')
        th=np.maximum(th+p55.offset_to_target(E,c,th),0.)
    theta=th[:,None];rows=p43.evaluate(name,c,g['w'],g,theta=theta,label=f'{kind}_{lam}')
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(p55.SEED_TEST)))
    kt=p55.respond(m@g['x_full'],c,theta);kte=p55.respond(m@x_test,c,theta);kn=p55.respond(m@g['x_noise'],c,theta)
    pn=(kn>1e-8).mean(1)[tune]
    aa=p55.memory(kt,kte,g['lab'],g['n'],g['n']);a20=p55.memory(kt,kte,g['lab'],g['n'],p55.N_SMALL)
    return dict(dataset=name,condition=kind,lam=lam,z=z,acc_all=float(np.mean(aa)),acc_20=float(np.mean(a20)),acc_all_draws=json.dumps(aa),acc_20_draws=json.dumps(a20),
                H=float(np.nanmean([r['H'] for r in rows])),H_median=float(np.nanmedian([r['H'] for r in rows])),
                leak=float(np.mean([r['leak_learned'] for r in rows])),leak_control=float(np.mean([r['leak_control'] for r in rows])),
                specific=float(np.mean([r['specific'] for r in rows])),active_odor=float((kt>1e-8).mean()),active_noise=float((kn>1e-8).mean()),
                noise_p_cv=float(pn.std()/pn.mean()) if pn.mean()>0 else np.nan,noise_kc_share=float((pn>0).mean()))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/spontaneous_homeostasis';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);_D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    jobs=[(n,'spont',l) for n in NAMES for l in LAMBDAS]+[(n,'odor_homeo',np.nan) for n in NAMES]
    with Pool(10) as pool:f=pd.DataFrame(pool.map(_run,jobs,chunksize=1))
    diffs=[]
    for n in NAMES:
        b=f[(f.dataset==n)&(f.condition=='spont')&(f.lam==0)].iloc[0]
        for _,r in f[f.dataset==n].iterrows():
            for k in ('acc_all_draws','acc_20_draws'):
                dd=np.array(json.loads(r[k]))-np.array(json.loads(b[k]))
                diffs.append(dict(dataset=n,condition=r.condition,lam=r.lam,task=k[:-6],diff=float(dd.mean()),se=float(dd.std(ddof=1)/np.sqrt(len(dd))) if dd.any() else 0.))
    dfd=pd.DataFrame(diffs);f.to_csv(out/'per_condition.csv',index=False);dfd.to_csv(out/'accuracy_diff_vs_original.csv',index=False)
    sp=f[f.condition=='spont'];one=sp[sp.lam==1].set_index('dataset')
    d20=dfd[(dfd.condition=='spont')&(dfd.lam==1)&(dfd.task=='acc_20')].set_index('dataset')
    mono={n:float(stats.spearmanr(sp[sp.dataset==n].lam,sp[sp.dataset==n].specific).statistic) for n in NAMES}
    v=dict(P1=bool((one.specific<=.003).all()),P2=bool((one.H<=2).sum()>=4),P3=bool(((d20['diff']>-d20['se'])).sum()>=4),
           P4=bool(sum(x<=-.9 for x in mono.values())>=4))
    report=dict(verdicts=v,monotonic_rho=mono,rows=f.drop(columns=['acc_all_draws','acc_20_draws']).round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',240)
    print(f.drop(columns=['acc_all_draws','acc_20_draws']).round(4).to_string())
    dfd['diff']*=100;dfd['se']*=100;print(dfd.round(2).to_string());print(json.dumps(v,indent=2),mono)


if __name__=='__main__':main()
