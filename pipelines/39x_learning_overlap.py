#!/usr/bin/env python3
"""Post hoc (not pre-registered): learning-based confusion after claw manipulation, using the step-21 generalization index."""
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


p39=load('p39','39_claw_manipulation.py');gen=load('gen21','21_learning_generalization.py')
p37=p39.p37;p34=p39.p34;cal=p39.cal
_D={}


def _run(job):
    i,name=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib)
    theta=cal.calibrate(g['m']@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    w=np.ones(len(c['inh']))/len(c['inh']);rows=[]
    for cond,cc in (('control',c),('more_claws',p39.add_claws(c,np.random.default_rng(390001+i))[0]),('one_claw',p39.remove_claws(c,np.random.default_rng(391001+i))[0])):
        m,_=cal.partial_matrix(cc);k,_=p37.respond(m@g['x_full'],cc,theta)
        ok=k.sum(0)>0;gi=gen.generalization(k[:,ok],w,5.);off=~np.eye(ok.sum(),dtype=bool)
        rows.append(dict(dataset=name,condition=cond,mean_GI=float(np.nanmean(gi[off])),odors_with_response=int(ok.sum())))
    return rows


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/claw_manipulation'
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs)
    with Pool(5) as pool:rows=[r for rs in pool.map(_run,list(enumerate(p39.NAMES))) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'posthoc_learning_generalization.csv',index=False);print(f.round(4).to_string())


if __name__=='__main__':main()
