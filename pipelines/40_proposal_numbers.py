#!/usr/bin/env python3
"""Model-derived quantities for the experiment proposal: claw count vs breadth/spontaneous activity, and APL block effects."""
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


p39=load('p39','39_claw_manipulation.py');p37=p39.p37;p34=p39.p34;cal=p39.cal;base=cal.base
_D={}


def _run(name):
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    cnt,_=p39.counts_of(c);claws=np.bincount(c['post'][cnt>=p39.CLAW_SYN],minlength=len(c['inh']))
    out=dict(dataset=name)
    for s,lab in ((1.,'apl_on'),(0.,'apl_off')):
        kt,_=base.response(m@g['x_full'],c['inh'],c['drive'],theta,s,gain=cal.APL_GAIN)
        kn,_=base.response(m@g['x_noise'],c['inh'],c['drive'],theta,s,gain=cal.APL_GAIN)
        out[f'odor_active_{lab}']=float((kt>1e-8).mean());out[f'noise_active_{lab}']=float((kn>1e-8).mean())
        if s==1.:
            part=(kt>1e-8).mean(1);spont=(kn>1e-8).mean(1);resp=part>0
            out['rho_claws_breadth']=float(stats.spearmanr(claws[resp],part[resp]).statistic)
            out['rho_claws_spont']=float(stats.spearmanr(claws,spont).statistic)
            q=pd.DataFrame(dict(claws=np.clip(claws,0,8),breadth=part,spont=spont)).groupby('claws').agg(n=('breadth','size'),breadth=('breadth','mean'),spont=('spont','mean'))
            out['by_claws']=q.round(4).reset_index().to_dict(orient='records')
    out['apl_block_odor_ratio']=out['odor_active_apl_off']/out['odor_active_apl_on']
    out['apl_block_noise_ratio']=out['noise_active_apl_off']/out['noise_active_apl_on']
    return out


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/proposal_numbers';out.mkdir(parents=True,exist_ok=True)
    p39.p24.SEEDS.update(noise=p39.p25.SEEDS['noise'],signal=p39.p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs)
    with Pool(5) as pool:res=pool.map(_run,p39.NAMES)
    (out/'report.json').write_text(json.dumps(res,indent=2)+'\n')
    for r in res:print(r['dataset'],{k:round(v,3) for k,v in r.items() if isinstance(v,float)});print('  by claws',[(x['claws'],x['n'],x['breadth'],x['spont']) for x in r['by_claws']])


if __name__=='__main__':main()
