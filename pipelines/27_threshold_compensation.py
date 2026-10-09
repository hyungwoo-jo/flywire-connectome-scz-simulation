#!/usr/bin/env python3
"""Graded per-KC threshold compensation (equalize average KC activity) vs learned value leakage and discrimination."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'))
p26=importlib.util.module_from_spec(spec);spec.loader.exec_module(p26)
p25=p26.p25;p24=p25.p24;p23=p24.p23;cal=p24.cal;base=cal.base

ALPHAS=(0.,.25,.5,.75,1.)
QUANTILE=90
TARGET=.10


def homeostatic_thresholds(e_cal):
    """90th percentile of each KC's excitation over calibration odors; inputless KCs get the mean."""
    th=np.percentile(e_cal,QUANTILE,axis=1);has=e_cal.max(1)>0
    th[~has]=th[has].mean()
    return th


def calibrate_scale(e_cal,inh,drive,shape):
    lo,hi=0.,float(e_cal.max()/max(shape.min(),1e-12))*2
    for _ in range(45):
        s=(lo+hi)/2;k,_=base.response(e_cal,inh,drive,(s*shape)[:,None],1.,gain=cal.APL_GAIN)
        if np.mean(k>1e-8)>TARGET:lo=s
        else:hi=s
    return (lo+hi)/2


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/threshold_compensation';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    rows=[]
    for side in p23.SIDES:
        g=p24.setup(root,side);g['w']=p25.mbon_weights(root,side,len(g['c']['inh']));c=g['c']
        e_cal=g['m']@cal.pn_drive(c,receptors,delta[:,calib]);th_h=homeostatic_thresholds(e_cal);tbar=th_h.mean()
        for a in ALPHAS:
            shape=(1-a)*tbar+a*th_h;s=calibrate_scale(e_cal,c['inh'],c['drive'],shape);theta=(s*shape)[:,None]
            lr,pp=p25.evaluate(g['m'],dict(g,theta=theta))
            kt,_=p24.kc(g['m'],g['x_full'],c,theta);part=(kt>1e-8).sum(1)
            rows.append(dict(side=side,alpha=a,scale=float(s),theta_mean=float(theta.mean()),theta_cv=float(theta.std()/theta.mean()),
                             leak_learned=float(np.mean([r['leak_learned'] for r in lr])),leak_control=float(np.mean([r['leak_control'] for r in lr])),
                             noise_FA=pp['noise_FA'],strength_rho=pp['rho'],dprime=float(p26.discrimination(g['m'],g,theta,root)),
                             max_participation=int(part.max()),participating=int((part>0).sum()),participation_gini=p26.gini(part[part>0]),
                             eval_active=float((kt>1e-8).mean())))
            print(rows[-1],flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    res={}
    for side in p23.SIDES:
        q=f[f.side==side].sort_values('alpha')
        rho=float(stats.spearmanr(q.alpha,q.leak_learned).statistic);ratio=float(q.leak_learned.iloc[-1]/q.leak_learned.iloc[0])
        res[side]=dict(rho=rho,ratio=ratio,dprime_gain=float(q.dprime.iloc[-1]-q.dprime.iloc[0]),
                       alpha0_matches_original=bool(np.isclose(q.leak_learned.iloc[0],json.loads((root/'qc_reports/learned_value_leakage/report.json').read_text())['results'][side]['leak_learned'])))
    R,L=res['right'],res['left']
    if all(x['rho']==-1 and x['ratio']<=.5 for x in (R,L)):va='monotone_reduction'
    elif all(x['ratio']<=.5 for x in (R,L)):va='reduction_not_monotone'
    else:va='not_supported'
    vb='compensation_improves_discrimination' if all(x['dprime_gain']>0 for x in (R,L)) else 'not_supported'
    report=dict(H27a=va,H27b=vb,results=res,alphas=ALPHAS,quantile=QUANTILE)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for side,col in (('right','#D55E00'),('left','#0072B2')):
        q=f[f.side==side].sort_values('alpha')
        axes[0].plot(q.alpha,q.leak_learned*100,'-o',color=col,label=f'{side} learned');axes[0].plot(q.alpha,q.leak_control*100,':s',color=col,label=f'{side} random control')
        axes[1].plot(q.alpha,q.dprime,'-o',color=col,label=side);axes[2].plot(q.alpha,q.max_participation,'-o',color=col,label=side)
    axes[0].set_ylabel('value leakage (% noise trials)');axes[1].set_ylabel("discrimination d'");axes[2].set_ylabel('max odors per KC (of 73)')
    for ax in axes:ax.set_xlabel('compensation strength alpha');ax.legend(fontsize=7)
    fig.suptitle('27: per-KC threshold compensation');fig.tight_layout();fig.savefig(out/'threshold_compensation.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
