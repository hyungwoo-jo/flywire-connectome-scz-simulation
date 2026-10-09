#!/usr/bin/env python3
"""Is KC-specific APL inhibition a compensation? Correlation with PN input and leakage under uniform APL."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

spec=importlib.util.spec_from_file_location('p25',Path(__file__).with_name('25_learned_value_leakage.py'))
p25=importlib.util.module_from_spec(spec);spec.loader.exec_module(p25)
p24=p25.p24;p23=p24.p23;cal=p24.cal


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/apl_compensation';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    res={}
    for side in p23.SIDES:
        g=p24.setup(root,side);g['w']=p25.mbon_weights(root,side,len(g['c']['inh']));c=g['c']
        pn_total=np.bincount(c['post'],weights=c['w'],minlength=len(c['inh']))
        has=pn_total>0;r=stats.spearmanr(pn_total[has],c['inh'][has])
        out_side=dict(rho_pn_apl=float(r.statistic),p_pn_apl=float(r.pvalue),apl_cv=float(c['inh'].std()/c['inh'].mean()))
        for label,inh in (('kc_specific',c['inh']),('uniform',np.ones_like(c['inh']))):
            cc=dict(c,inh=inh)
            theta=cal.calibrate(g['m']@cal.pn_drive(cc,receptors,delta[:,calib]),inh,c['drive'],.10)
            lr,pp=p25.evaluate(g['m'],dict(g,c=cc,theta=theta))
            out_side[label]=dict(theta=float(theta),leak_learned=float(np.mean([x['leak_learned'] for x in lr])),
                                 leak_control=float(np.mean([x['leak_control'] for x in lr])),noise_FA=pp['noise_FA'])
        out_side['ratio']=out_side['uniform']['leak_learned']/out_side['kc_specific']['leak_learned']
        res[side]=out_side
    R,L=res['right'],res['left']
    va='inhibitory_compensation_in_wiring' if all(x['rho_pn_apl']>0 and x['p_pn_apl']<.05 for x in (R,L)) else 'not_established'
    if all(x['ratio']>=1.2 for x in (R,L)):vb='apl_specificity_reduces_leakage'
    elif all(.9<=x['ratio']<=1.1 for x in (R,L)):vb='little_effect'
    else:vb='partial'
    report=dict(H33a=va,H33b=vb,results=res)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
