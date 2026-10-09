#!/usr/bin/env python3
"""Is value leakage odor-specific, or a product of learning on high-input KCs? Input-strength-matched control, five hemispheres."""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p37=load('p37','37_kc_recurrence.py');p34=p37.p34;p25=p37.p25;p24=p37.p24;cal=p37.cal

SEED0=380000
BOOT=380300
_DATA={}


def matched_set(active,strength,deciles,rng):
    """For each active KC pick a distinct inactive KC from the same input-strength decile (nearest decile if exhausted)."""
    pool={d:list(rng.permutation(np.flatnonzero((deciles==d)&~active))) for d in range(10)}
    out=[]
    for i in np.flatnonzero(active):
        d=deciles[i]
        for step in range(10):
            cands=[x for x in (d-step,d+step) if 0<=x<10 and pool[x]]
            if cands:out.append(pool[cands[0]].pop());break
    return np.array(out),np.flatnonzero(active)[:len(out)]


def _run(name):
    d=_DATA['sets'][name];receptors,delta,sfr,calib=_DATA['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    kt,_=p24.kc(m,g['x_full'],c,theta);kn,_=p24.kc(m,g['x_noise'],c,theta);kw,_=p24.kc(m,g['x_weak'],c,theta)
    crit=float(np.percentile(p24.kc(m,g['x_cal'],c,theta)[0].sum(0),95));rep=kn.sum(0)>crit;w=g['w']
    strength=np.asarray(m.sum(1)).ravel();has=strength>0
    dec=np.full(len(strength),-1);dec[has]=np.minimum((pd.Series(strength[has]).rank(pct=True).to_numpy()*10).astype(int),9)
    rows=[]
    for a in range(g['n']):
        learn=kt[:,a];act=learn>1e-8
        lcrit=.5*np.median(p25.value_signal(w,learn,kw[:,g['lab']==a]))
        rng=np.random.default_rng(p25.SEEDS['control']+a);ctl=np.zeros_like(learn)
        ctl[rng.choice(len(learn),int(act.sum()),replace=False)]=rng.permutation(learn[act])
        tgt,src=matched_set(act&has,strength,dec,np.random.default_rng(SEED0+a));mat=np.zeros_like(learn);mat[tgt]=learn[src]
        # active KCs without PN input (none expected) keep no matched partner
        leak=lambda pattern:float(np.mean(rep&(p25.value_signal(w,pattern,kn)>lcrit)))
        rows.append(dict(dataset=name,odor=a,n_active=int(act.sum()),n_matched=len(tgt),leak_learned=leak(learn),leak_random=leak(ctl),leak_matched=leak(mat),
                         strength_active_mean=float(strength[act].mean()),strength_matched_mean=float(strength[tgt].mean()) if len(tgt) else np.nan,
                         strength_random_mean=float(strength[ctl>0].mean())))
    return rows


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/matched_control';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_DATA.update(sets=sets,inputs=inputs)
    names=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']
    with Pool(5) as pool:rows=[r for rs in pool.map(_run,names) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'per_odor.csv',index=False)
    res={}
    for name,q in f.groupby('dataset'):
        sr=(q.leak_learned-q.leak_random).to_numpy();sm=(q.leak_learned-q.leak_matched).to_numpy();rng=np.random.default_rng(BOOT)
        bm=[sm[rng.integers(len(sm),size=len(sm))].mean() for _ in range(1000)]
        res[name]=dict(leak_learned=float(q.leak_learned.mean()),leak_random=float(q.leak_random.mean()),leak_matched=float(q.leak_matched.mean()),
                       S_rand=float(sr.mean()),S_match=float(sm.mean()),S_match_ci=[float(x) for x in np.percentile(bm,[2.5,97.5])],
                       ratio=float(sm.mean()/sr.mean()) if sr.mean()>0 else None,matched_fraction=float((q.n_matched/q.n_active).mean()),
                       strength_active=float(q.strength_active_mean.mean()),strength_matched=float(q.strength_matched_mean.mean()),strength_random=float(q.strength_random_mean.mean()))
    rs=[r['ratio'] for r in res.values()]
    if all(r['S_match_ci'][0]>0 and r['ratio']>=.5 for r in res.values()):v='odor_specific_component_large'
    elif all(x is not None and x<=.2 for x in rs):v='mostly_hub_learning'
    else:v='both_contribute'
    report=dict(verdict=v,results=res)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
