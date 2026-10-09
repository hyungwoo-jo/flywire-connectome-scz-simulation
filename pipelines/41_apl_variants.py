#!/usr/bin/env python3
"""Experiment-2 predictions under APL variants: global feedback, self-inhibition (Amin et al. 2020) and tonic inhibition."""
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


p39=load('p39','39_claw_manipulation.py');p37=p39.p37;p34=p39.p34;cal=p39.cal
GAIN=4.
VARIANTS=dict(global_feedback=(0.,0.),self_inhibition_50=(.5,0.),self_inhibition_90=(.9,0.),tonic_low=(0.,.05),tonic_high=(0.,.15))
NAMES=['BANC_right','FlyWire_right','hemibrain_right']
_D={}


def respond(E,inh,drive,theta,s,lam,tonic):
    """k = clip((E - theta - G s inh[(1-lam) a + tonic]) / (1 + G s lam inh), 0, 1); a = drive . k solved by bisection."""
    def k_of(a):return np.clip((E-theta-GAIN*s*inh[:,None]*((1-lam)*a+tonic))/(1+GAIN*s*lam*inh[:,None]),0,1)
    lo=np.zeros(E.shape[1]);hi=drive@np.clip(E-theta,0,1)
    for _ in range(40):
        a=(lo+hi)/2;up=(drive@k_of(a))>a;lo=np.where(up,a,lo);hi=np.where(up,hi,a)
    return k_of((lo+hi)/2)


def calibrate(e,inh,drive,lam,tonic):
    lo,hi=0.,float(e.max())*2
    for _ in range(40):
        th=(lo+hi)/2
        if np.mean(respond(e,inh,drive,th,1.,lam,tonic)>1e-8)>.10:lo=th
        else:hi=th
    return (lo+hi)/2


def _run(name):
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m'];inh,drive=c['inh'],c['drive']
    rows=[]
    for vname,(lam,tonic) in VARIANTS.items():
        th=calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),inh,drive,lam,tonic)
        r=dict(dataset=name,variant=vname,lam=lam,tonic=tonic,theta=th)
        for s,lab in ((1.,'on'),(0.,'off')):
            ko=respond(m@g['x_full'],inh,drive,th,s,lam,tonic);kn=respond(m@g['x_noise'],inh,drive,th,s,lam,tonic)
            r[f'odor_frac_{lab}']=float((ko>1e-8).mean());r[f'noise_frac_{lab}']=float((kn>1e-8).mean())
            r[f'odor_amp_{lab}']=float(ko[ko>1e-8].mean());r[f'noise_amp_{lab}']=float(kn[kn>1e-8].mean()) if (kn>1e-8).any() else 0.
        for q in ('odor_frac','noise_frac','odor_amp','noise_amp'):r[f'{q}_ratio']=r[f'{q}_off']/r[f'{q}_on'] if r[f'{q}_on']>0 else np.nan
        rows.append(r)
    return rows


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/apl_variants';out.mkdir(parents=True,exist_ok=True)
    p39.p24.SEEDS.update(noise=p39.p25.SEEDS['noise'],signal=p39.p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs)
    with Pool(3) as pool:rows=[r for rs in pool.map(_run,NAMES) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False);pd.set_option('display.width',200)
    print(f[['dataset','variant','odor_frac_ratio','noise_frac_ratio','odor_amp_ratio','noise_amp_ratio','noise_frac_on']].round(3).to_string())


if __name__=='__main__':main()
