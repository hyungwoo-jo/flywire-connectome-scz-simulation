#!/usr/bin/env python3
"""Closed-form leakage index: overlap between the learned KC pattern and noise-driven KC activity, checked against simulation."""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p38=load('p38','38_matched_control.py');p37=p38.p37;p34=p38.p34;p25=p38.p25;p24=p38.p24;p26=p34.p26;cal=p38.cal
ETA=p25.ETA
_D={}


def index_terms(w,learn,m):
    """L_hat = sum w m d / sum w m with d = 1-exp(-eta*learn); H = L_hat / (sum w d / sum w)."""
    d=1-np.exp(-ETA*learn);lhat=float((w*m*d).sum()/(w*m).sum()) if (w*m).sum()>0 else 0.
    base=float((w*d).sum()/w.sum());return lhat,(lhat/base if base>0 else np.nan),base


def evaluate(name,c,w,g,m_override=None,label='original',theta=None):
    receptors,delta,sfr,calib=_D['inputs']
    m=m_override if m_override is not None else g['m']
    if theta is None:theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    gg=dict(g,m=m,theta=theta);rows,_=p25.evaluate(m,gg)
    kt,_=p24.kc(m,g['x_full'],c,theta);kn,_=p24.kc(m,g['x_noise'],c,theta);mn=kn.mean(1)
    out=[]
    for a,r in enumerate(rows):
        lhat,H,base=index_terms(w,kt[:,a],mn)
        out.append(dict(dataset=name,circuit=label,odor=a,leak_learned=r['leak_learned'],leak_control=r['leak_control'],
                        specific=r['leak_learned']-r['leak_control'],L_hat=lhat,H=H,L_base=base))
    return out


def _run(name):
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib)
    rows=evaluate(name,c,g['w'],g)
    for a in (.5,1.):  # input compensation levels from step 29
        mc=load('p29','29_input_compensation.py').compensate(g['m'],a)
        rows+=evaluate(name,c,g['w'],g,m_override=mc,label=f'input_comp_{a}')
    return rows


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/leakage_theory';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs)
    with Pool(5) as pool:rows=[r for rs in pool.map(_run,['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'per_odor.csv',index=False)
    agg=f.groupby(['dataset','circuit']).agg(leak=('leak_learned','mean'),specific=('specific','mean'),L_hat=('L_hat','mean'),H=('H','mean')).reset_index()
    agg.to_csv(out/'per_condition.csv',index=False)
    within=[]
    for (ds,cir),q in f.groupby(['dataset','circuit']):
        within.append(dict(dataset=ds,circuit=cir,rho_Lhat_leak=float(stats.spearmanr(q.L_hat,q.leak_learned).statistic),rho_H_specific=float(stats.spearmanr(q.H,q.specific).statistic)))
    w=pd.DataFrame(within);w.to_csv(out/'within_condition_rho.csv',index=False)
    across=dict(rho_Lhat_leak=float(stats.spearmanr(agg.L_hat,agg.leak).statistic),rho_H_specific=float(stats.spearmanr(agg.H,agg.specific).statistic),n_conditions=len(agg))
    report=dict(across_conditions=across,within_condition_median=dict(rho_Lhat_leak=float(w.rho_Lhat_leak.median()),rho_H_specific=float(w.rho_H_specific.median())),
                per_condition=agg.round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',200);print(agg.round(4).to_string());print(w.round(3).to_string());print(json.dumps(across,indent=2))


if __name__=='__main__':main()
