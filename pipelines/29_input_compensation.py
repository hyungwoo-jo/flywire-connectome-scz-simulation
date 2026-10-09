#!/usr/bin/env python3
"""Graded input-strength compensation (weights x (mean/S_i)^alpha) vs learned value leakage and discrimination."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'))
p26=importlib.util.module_from_spec(spec);spec.loader.exec_module(p26)
p25=p26.p25;p24=p25.p24;p23=p24.p23;cal=p24.cal

ALPHAS=(0.,.25,.5,.75,1.)


def compensate(m,alpha):
    s=np.asarray(m.sum(axis=1)).ravel();sbar=s[s>0].mean()
    scale=np.where(s>0,(sbar/np.where(s>0,s,1))**alpha,0.)
    scale[s==0]=0.
    return sparse.diags(scale)@m


def interior_optimum(values,alphas,mode):
    v=np.asarray(values);i=int(np.argmin(v) if mode=='min' else np.argmax(v))
    return alphas[i] in (.25,.5,.75),float(v[i]),float(v[-1])


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/input_compensation';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    rows=[]
    for side in p23.SIDES:
        g=p24.setup(root,side);g['w']=p25.mbon_weights(root,side,len(g['c']['inh']));c=g['c']
        for a in ALPHAS:
            m=compensate(g['m'],a)
            theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
            lr,pp=p25.evaluate(m,dict(g,theta=theta))
            kt,_=p24.kc(m,g['x_full'],c,theta);part=(kt>1e-8).sum(1)
            rows.append(dict(side=side,alpha=a,theta=float(theta),leak_learned=float(np.mean([r['leak_learned'] for r in lr])),
                             leak_control=float(np.mean([r['leak_control'] for r in lr])),noise_FA=pp['noise_FA'],strength_rho=pp['rho'],
                             dprime=float(p26.discrimination(m,g,theta,root)),max_participation=int(part.max()),participating=int((part>0).sum())))
            print(rows[-1],flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    ref26=json.loads((root/'qc_reports/hub_normalization/report.json').read_text())['results']
    res={}
    for side in p23.SIDES:
        q=f[f.side==side].sort_values('alpha')
        li,lmin,l1=interior_optimum(q.leak_learned,ALPHAS,'min');di,dmax,d1=interior_optimum(q.dprime,ALPHAS,'max')
        res[side]=dict(leak_interior_min=li,leak_min=lmin,leak_alpha1=l1,leak_condition=bool(li and lmin<=.9*l1),
                       dprime_interior_max=di,dprime_max=dmax,dprime_alpha1=d1,dprime_condition=bool(di and dmax>=d1+.3),
                       alpha_best_leak=float(q.alpha.iloc[int(np.argmin(q.leak_learned))]),alpha_best_dprime=float(q.alpha.iloc[int(np.argmax(q.dprime))]),
                       alpha1_matches_26=bool(np.isclose(l1,ref26[side]['equalized']['leak_learned'])),
                       alpha0_matches_25=bool(np.isclose(float(q.leak_learned.iloc[0]),ref26[side]['original']['leak_learned'])))
    R,L=res['right'],res['left']
    if all(x['leak_condition'] and x['dprime_condition'] for x in (R,L)):v='partial_compensation_optimal'
    elif all(x['leak_condition'] for x in (R,L)):v='partial_optimal_for_leakage_only'
    elif all(x['alpha_best_leak']==1. and x['alpha_best_dprime']==1. for x in (R,L)):v='full_compensation_optimal'
    else:v='inconsistent'
    report=dict(verdict=v,results=res,alphas=ALPHAS)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for side,col in (('right','#D55E00'),('left','#0072B2')):
        q=f[f.side==side].sort_values('alpha')
        axes[0].plot(q.alpha,q.leak_learned*100,'-o',color=col,label=f'{side} learned');axes[0].plot(q.alpha,q.leak_control*100,':s',color=col,label=f'{side} control')
        axes[1].plot(q.alpha,q.dprime,'-o',color=col,label=side)
    axes[0].set_ylabel('value leakage (%)');axes[1].set_ylabel("d'")
    for ax in axes:ax.set_xlabel('input compensation alpha');ax.legend(fontsize=7)
    fig.suptitle('29: input-strength compensation');fig.tight_layout();fig.savefig(out/'input_compensation.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
