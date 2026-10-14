#!/usr/bin/env python3
"""Step 57: mix odor-activity threshold homeostasis (step 55, sparsity matched) with spontaneous-input thresholds (step 56,
lambda 1). theta(beta) = (1 - beta) theta_odor + beta theta_spont + common shift for 10% active KCs. Exploratory: total-activity
homeostasis over a mix of odor (rho) and spontaneous (1 - rho) trials. Leak and memory as in step 56.
Verdicts in docs/PREREGISTRATION_57.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p56',Path(__file__).with_name('56_spontaneous_homeostasis.py'))
p56=_u.module_from_spec(spec);sys.modules['p56']=p56;spec.loader.exec_module(p56)
p55=p56.p55;p43,p34,p25,p24,cal=p55.p43,p55.p34,p55.p25,p55.p24,p55.cal
NAMES=p55.NAMES;BETAS=(0.,.2,.4,.6,.8,1.);RHOS=(.1,.01)
_D=p55._D


def mixed_homeostasis(Eo,En,c,theta0,tune,rho):
    """Per-KC thresholds so that rho * mean odor activity + (1 - rho) * mean spontaneous activity reaches the population mean
    at theta0; step scaled so rho = 1 reproduces step 55's rule."""
    act=lambda th:rho*p55.respond(Eo,c,th[:,None]).mean(1)+(1-rho)*p55.respond(En,c,th[:,None]).mean(1)
    a_odor=float(p55.respond(Eo,c,theta0)[tune].mean());th=np.full(Eo.shape[0],theta0);a0=float(act(th)[tune].mean())
    for _ in range(p55.ITERS):
        mean=act(th);th[tune]=np.maximum(th[tune]+p55.LR_THETA*a_odor*(mean[tune]-a0)/a0,0.)
    mean=act(th);return th,float(mean[tune].std()/mean[tune].mean())


def _run(job):
    name,kind,par=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    xcal=cal.pn_drive(c,receptors,delta[:,calib]);E=m@xcal;theta0=cal.calibrate(E,c['inh'],c['drive'],.10)
    s=np.asarray(m.sum(1)).ravel();tune=s>0;En=np.asarray(m@g['x_cal']);cv=np.nan
    if kind=='original':th=np.full(len(s),theta0)
    elif kind=='mix':
        a0=float(p55.respond(E,c,theta0)[tune].mean());to,_,_=p55.homeostasis(E,c,theta0,tune,a0,'theta')
        to=np.maximum(to+p55.offset_to_target(E,c,to),0.)
        base=En.mean(1);z=p56.z_for_target(E,c,base,En.std(1));ts=np.maximum(base+z*En.std(1),0.)
        th=(1-par)*to+par*ts;th=np.maximum(th+p55.offset_to_target(E,c,th),0.)
    else:
        th,cv=mixed_homeostasis(E,En,c,theta0,tune,par);th=np.maximum(th+p55.offset_to_target(E,c,th),0.)
    theta=th[:,None];rows=p43.evaluate(name,c,g['w'],g,theta=theta,label=f'{kind}_{par}')
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(p55.SEED_TEST)))
    kt=p55.respond(m@g['x_full'],c,theta);kte=p55.respond(m@x_test,c,theta);kn=p55.respond(m@g['x_noise'],c,theta)
    aa=p55.memory(kt,kte,g['lab'],g['n'],g['n']);a20=p55.memory(kt,kte,g['lab'],g['n'],p55.N_SMALL)
    return dict(dataset=name,condition=kind,par=par,acc_all=float(np.mean(aa)),acc_20=float(np.mean(a20)),acc_all_draws=json.dumps(aa),acc_20_draws=json.dumps(a20),
                H=float(np.nanmean([r['H'] for r in rows])),H_median=float(np.nanmedian([r['H'] for r in rows])),
                leak=float(np.mean([r['leak_learned'] for r in rows])),leak_control=float(np.mean([r['leak_control'] for r in rows])),
                specific=float(np.mean([r['specific'] for r in rows])),active_odor=float((kt>1e-8).mean()),active_noise=float((kn>1e-8).mean()),homeo_cv=cv)


def both_goals(sp,d20):
    """Both goals: specific leak <= 0.3% and 20-odor accuracy not below original (diff > -SE)."""
    return (sp<=.003)&(d20['diff']>-d20['se'])


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/mixed_homeostasis';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);_D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    jobs=[(n,'original',np.nan) for n in NAMES]+[(n,'mix',b) for n in NAMES for b in BETAS]+[(n,'total_homeo',r) for n in NAMES for r in RHOS]
    with Pool(14) as pool:f=pd.DataFrame(pool.map(_run,jobs,chunksize=1))
    diffs=[]
    for n in NAMES:
        b=f[(f.dataset==n)&(f.condition=='original')].iloc[0]
        for _,r in f[(f.dataset==n)&(f.condition!='original')].iterrows():
            for k in ('acc_all_draws','acc_20_draws'):
                dd=np.array(json.loads(r[k]))-np.array(json.loads(b[k]))
                diffs.append(dict(dataset=n,condition=r.condition,par=r.par,task=k[:-6],diff=float(dd.mean()),se=float(dd.std(ddof=1)/np.sqrt(len(dd)))))
    dfd=pd.DataFrame(diffs);f.to_csv(out/'per_condition.csv',index=False);dfd.to_csv(out/'accuracy_diff_vs_original.csv',index=False)
    mx=f[f.condition=='mix'];d20=dfd[(dfd.condition=='mix')&(dfd.task=='acc_20')]
    ok={}
    for b in BETAS:
        sp=mx[mx.par==b].set_index('dataset').specific;dd=d20[d20.par==b].set_index('dataset')
        ok[b]=both_goals(sp,dd.loc[sp.index]).to_dict()
    mid=BETAS[1:-1]
    rho_leak={n:float(stats.spearmanr(mx[mx.dataset==n].par,mx[mx.dataset==n].specific).statistic) for n in NAMES}
    rho_mem={n:float(stats.spearmanr(mx[mx.dataset==n].par,mx[mx.dataset==n].acc_20).statistic) for n in NAMES}
    v=dict(P1=bool(any(sum(ok[b].values())>=4 for b in mid)),
           P2=bool(all(any(ok[b][n] for b in mid) for n in ('BANC_right','BANC_left'))),
           P3a=bool(sum(x<=-.9 for x in rho_leak.values())>=4),P3b=bool(sum(x<=-.9 for x in rho_mem.values())>=4))
    report=dict(verdicts=v,both_goals={str(b):ok[b] for b in BETAS},rho_leak=rho_leak,rho_memory=rho_mem,
                rows=f.drop(columns=['acc_all_draws','acc_20_draws']).round(5).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',240)
    print(f.drop(columns=['acc_all_draws','acc_20_draws']).round(4).to_string())
    dfd['diff']*=100;dfd['se']*=100;print(dfd[dfd.task=='acc_20'].round(2).to_string());print(json.dumps(v,indent=2),ok,rho_leak,rho_mem)


if __name__=='__main__':main()
