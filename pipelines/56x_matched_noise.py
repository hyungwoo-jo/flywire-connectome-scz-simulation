#!/usr/bin/env python3
"""Step 56 post hoc: is the leak removal at lambda = 1 just lower noise-trial activity? Original circuit with the common
threshold raised until the noise-trial active fraction equals that of the lambda = 1 circuit; leak, H, odor activity, memory."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p56',Path(__file__).with_name('56_spontaneous_homeostasis.py'))
p56=_u.module_from_spec(spec);sys.modules['p56']=p56;spec.loader.exec_module(p56)
p55,p43,p34,p25,p24,cal=p56.p55,p56.p43,p56.p34,p56.p25,p56.p24,p56.cal


def _run(job):
    name,target=job;d=p55._D['sets'][name];receptors,delta,sfr,calib=p55._D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    En=m@g['x_noise'];lo,hi=0.,float(En.max())
    for _ in range(50):                                      # raise common theta until noise-active fraction <= target
        th=(lo+hi)/2
        if np.mean(p55.respond(En,c,th)>1e-8)>target:lo=th
        else:hi=th
    theta=hi;rows=p43.evaluate(name,c,g['w'],g,theta=theta,label='matched_noise')
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(p55.SEED_TEST)))
    kt=p55.respond(m@g['x_full'],c,theta);kte=p55.respond(m@x_test,c,theta)
    return dict(dataset=name,target_noise_active=target,theta=theta,active_odor=float((kt>1e-8).mean()),
                acc_all=float(np.mean(p55.memory(kt,kte,g['lab'],g['n'],g['n']))),acc_20=float(np.mean(p55.memory(kt,kte,g['lab'],g['n'],p55.N_SMALL))),
                H=float(np.nanmean([r['H'] for r in rows])),leak=float(np.mean([r['leak_learned'] for r in rows])),specific=float(np.mean([r['specific'] for r in rows])))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/spontaneous_homeostasis'
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    f=pd.read_csv(out/'per_condition.csv');tg=f[(f.condition=='spont')&(f.lam==1)].set_index('dataset').active_noise
    sets,inputs=p55.p37.build_all(root);p55._D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    with Pool(5) as pool:r=pd.DataFrame(pool.map(_run,[(n,float(tg[n])) for n in p56.NAMES]))
    r.to_csv(out/'posthoc_matched_noise.csv',index=False);pd.set_option('display.width',200);print(r.round(5).to_string())


if __name__=='__main__':main()
