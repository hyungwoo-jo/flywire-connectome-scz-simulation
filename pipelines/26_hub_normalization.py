#!/usr/bin/env python3
"""Causal test of the hub-KC mechanism: equalize each KC's total PN input and recompute value leakage."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p25',Path(__file__).with_name('25_learned_value_leakage.py'))
p25=importlib.util.module_from_spec(spec);spec.loader.exec_module(p25)
p24=p25.p24;p23=p24.p23;cal=p24.cal;disc=p24.disc;base=cal.base

N_BOOT=1000


def equalize_rows(m):
    """Scale each KC row to the mean nonzero row sum; zero rows stay zero."""
    s=np.asarray(m.sum(axis=1)).ravel();target=s[s>0].mean()
    scale=np.where(s>0,target/np.where(s>0,s,1),0.)
    return sparse.diags(scale)@m


def gini(x):
    x=np.sort(np.asarray(x,float));n=len(x)
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum())) if x.sum()>0 else 0.


def leak_summary(rows):
    d=np.array([r['leak_learned'] for r in rows]);c=np.array([r['leak_control'] for r in rows])
    return d,c


def discrimination(m,g,theta,root):
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));n=len(keep)
    labels=np.repeat(np.arange(n),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,n)
    x=cal.pn_drive(g['c'],receptors,cal.sample_orn(delta[:,keep],sfr.to_numpy(),labels,.5,np.random.default_rng(disc.NOISE_SEED)))
    k,_=base.response(m@x,g['c']['inh'],g['c']['drive'],theta,1.,gain=cal.APL_GAIN)
    return disc.evaluate(k,labels,train,{'ALL':np.arange(n)})['ALL_dprime']


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/hub_normalization';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    res={};per=[]
    for side in p23.SIDES:
        g=p24.setup(root,side);g['w']=p25.mbon_weights(root,side,len(g['c']['inh']))
        mn=equalize_rows(g['m'])
        e_cal=mn@cal.pn_drive(g['c'],receptors,delta[:,calib]);theta_n=cal.calibrate(e_cal,g['c']['inh'],g['c']['drive'],.10)
        out_side={}
        for label,m,theta in (('original',g['m'],g['theta']),('equalized',mn,theta_n)):
            gg=dict(g,theta=theta);rows,p=p25.evaluate(m,gg)
            kt,_=p24.kc(m,g['x_full'],g['c'],theta);part=(kt>1e-8).sum(1)
            d,c=leak_summary(rows);per+=[dict(side=side,circuit=label,**r) for r in rows]
            out_side[label]=dict(theta=float(theta),leak_learned=float(d.mean()),leak_control=float(c.mean()),noise_FA=p['noise_FA'],
                                 strength_rho=p['rho'],participation_gini=gini(part[part>0]) if (part>0).any() else 0.,
                                 kc_participating=int((part>0).sum()),max_participation=int(part.max()),
                                 eval_active=float((kt>1e-8).mean()),dprime=float(discrimination(m,g,theta,root)),_d=d)
        do,dn=out_side['original'].pop('_d'),out_side['equalized'].pop('_d')
        rng=np.random.default_rng(p25.SEEDS['boot']);boot=[]
        for _ in range(N_BOOT):
            i=rng.integers(len(do),size=len(do));boot.append(do[i].mean()-dn[i].mean())
        R=out_side['equalized']['leak_learned']/out_side['original']['leak_learned']
        res[side]=dict(**out_side,ratio=float(R),diff_ci=[float(x) for x in np.percentile(boot,[2.5,97.5])])
    pd.DataFrame(per).to_csv(out/'per_odor.csv',index=False)
    Rr,Rl=res['right']['ratio'],res['left']['ratio']
    if Rr<=.5 and Rl<=.5 and res['right']['diff_ci'][0]>0 and res['left']['diff_ci'][0]>0:v='hub_mechanism_supported'
    elif Rr>.8 and Rl>.8:v='not_explained_by_hubs'
    else:v='partial'
    report=dict(verdict=v,results=res)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pf=pd.DataFrame(per)
    fig,ax=plt.subplots(figsize=(6,4))
    for i,side in enumerate(('right','left')):
        for j,circ in enumerate(('original','equalized')):
            q=pf[(pf.side==side)&(pf.circuit==circ)]
            ax.bar(i*3+j,q.leak_learned.mean()*100,yerr=q.leak_learned.std()/np.sqrt(len(q))*100,color=['#D55E00','#009E73'][j],label=circ if i==0 else None)
    ax.set_xticks([.5,3.5]);ax.set_xticklabels(['right','left']);ax.set_ylabel('learned value leakage (% of noise trials)');ax.legend()
    ax.set_title('26: equalizing KC input removes hubs');fig.tight_layout();fig.savefig(out/'hub_normalization.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
