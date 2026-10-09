#!/usr/bin/env python3
"""Robustness of learned value leakage when unmapped PNs receive imputed (tuning-shuffled) Hallem-like input."""
import json
import importlib.util
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'))
p26=importlib.util.module_from_spec(spec);spec.loader.exec_module(p26)
p25=p26.p25;p24=p25.p24;p23=p24.p23;cal=p24.cal;disc=p24.disc;rep=p23.rep

N_IMPUTE=20
SEED0=280000
_G={}


def olsen_partial_sum(orn,n_real):
    """Olsen 2010 transform; lateral term sums only the real Hallem channels."""
    orn=np.maximum(orn,0);p=cal.OLSEN
    s=p['m']*orn[:n_real].sum(axis=0,keepdims=True)/p['divisor']
    return orn**1.5/(orn**1.5+p['sigma']**1.5+s**1.5)


def imputed_channels(pns,delta,sfr,rng):
    """One shuffled Hallem channel per unmapped glomerulus type."""
    types=sorted(pns[pns.receptor.fillna('')==''].glomerulus.unique())
    src=rng.integers(delta.shape[0],size=len(types))
    rows=np.stack([delta[s,rng.permutation(delta.shape[1])] for s in src]) if types else np.zeros((0,delta.shape[1]))
    return types,rows,sfr[src]


def build(side,seed):
    g=_G[side];c=g['c'];pns=c['pns'].copy();receptors=g['receptors']
    if seed is None:
        mapped=c['mapped'];chan=list(receptors);d_ext=g['delta'];s_ext=g['sfr']
    else:
        types,rows,s_imp=imputed_channels(pns,g['delta'],g['sfr'],np.random.default_rng(seed))
        pns['receptor']=[r if r else f'imp_{t}' for r,t in zip(pns.receptor.fillna(''),pns.glomerulus)]
        mapped=pns.reset_index(drop=True);chan=list(receptors)+[f'imp_{t}' for t in types]
        d_ext=np.vstack([g['delta'],rows]);s_ext=np.r_[g['sfr'],s_imp]
    cc=dict(c,mapped=mapped);m,_=cal.partial_matrix(cc);idx=np.array([chan.index(r) for r in mapped.receptor])
    drive=lambda orn:olsen_partial_sum(orn,len(receptors))[idx]
    noisy=lambda dd,lab,sd:drive(cal.sample_orn(dd,s_ext,lab,.5,np.random.default_rng(sd)))
    keep=g['keep'];n=len(keep);lab=np.repeat(np.arange(n),p25.p24.TRIALS);zero=np.zeros((len(chan),1))
    theta=cal.calibrate(m@drive(d_ext[:,g['calib']]),c['inh'],c['drive'],.10)
    gg=dict(c=cc,m=m,theta=theta,n=n,lab=lab,w=g['w'],x_full=drive(d_ext[:,keep]),
            x_cal=noisy(zero,np.zeros(1000,int),p23.SEEDS['cal']),x_noise=noisy(zero,np.zeros(p24.N_NOISE,int),p25.SEEDS['noise']),
            x_weak=noisy(.1*d_ext[:,keep],lab,p25.SEEDS['signal']))
    return gg


def _run(job):
    side,seed=job;gg=build(side,seed);rows,pp=p25.evaluate(gg['m'],gg)
    kt,_=p24.kc(gg['m'],gg['x_full'],gg['c'],gg['theta']);part=(kt>1e-8).sum(1);tot=np.asarray(gg['m'].sum(1)).ravel()
    ll=np.mean([r['leak_learned'] for r in rows]);lc=np.mean([r['leak_control'] for r in rows])
    return dict(side=side,seed=seed if seed is not None else -1,circuit='partial' if seed is None else 'imputed_full',
                pns_with_input=len(gg['c']['mapped']),theta=float(gg['theta']),leak_learned=float(ll),leak_control=float(lc),specific=float(ll-lc),
                noise_FA=pp['noise_FA'],strength_rho=pp['rho'],input_cv=float(tot[tot>0].std()/tot[tot>0].mean()),
                max_participation=int(part.max()),participation_gini=p26.gini(part[part>0]),eval_active=float((kt>1e-8).mean()))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/full_circuit_imputation';out.mkdir(parents=True,exist_ok=True)
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ))
    for side in p23.SIDES:
        c=rep.load_side_circuit(root,side,p23.SIDES[side]['apl'],receptors)
        _G[side]=dict(c=c,receptors=receptors,delta=delta,sfr=sfr.to_numpy(),calib=calib,keep=keep,w=p25.mbon_weights(root,side,len(c['inh'])))
    jobs=[(s,None) for s in p23.SIDES]+[(s,SEED0+i) for s in p23.SIDES for i in range(1,N_IMPUTE+1)]
    with Pool(10) as pool:f=pd.DataFrame(pool.map(_run,jobs))
    f.to_csv(out/'metrics.csv',index=False)
    res={}
    for side in p23.SIDES:
        q=f[f.side==side];sp=float(q[q.circuit=='partial'].specific.iloc[0]);im=q[q.circuit=='imputed_full']
        res[side]=dict(S_partial=sp,S_full_median=float(im.specific.median()),S_full_range=[float(im.specific.min()),float(im.specific.max())],
                       positive_imputations=int((im.specific>0).sum()),ratio=float(im.specific.median()/sp),
                       partial_matches_25=bool(np.isclose(float(q[q.circuit=='partial'].leak_learned.iloc[0]),json.loads((root/'qc_reports/learned_value_leakage/report.json').read_text())['results'][side]['leak_learned'])),
                       input_cv_partial=float(q[q.circuit=='partial'].input_cv.iloc[0]),input_cv_full_median=float(im.input_cv.median()),
                       max_participation_partial=int(q[q.circuit=='partial'].max_participation.iloc[0]),max_participation_full_median=float(im.max_participation.median()),
                       strength_rho_full_median=float(im.strength_rho.median()),leak_learned_full_median=float(im.leak_learned.median()),leak_control_full_median=float(im.leak_control.median()))
    R,L=res['right'],res['left']
    if all(x['ratio']>=.5 and x['positive_imputations']>=16 for x in (R,L)):v='robust'
    elif all(x['ratio']<.25 for x in (R,L)):v='mostly_partial_circuit_artifact'
    else:v='partially_robust'
    report=dict(verdict=v,results=res,imputations=N_IMPUTE,seeds=[SEED0+1,SEED0+N_IMPUTE])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6,4))
    for i,(side,col) in enumerate((('right','#D55E00'),('left','#0072B2'))):
        q=f[f.side==side];im=q[q.circuit=='imputed_full']
        ax.scatter(np.full(len(im),i)+np.random.default_rng(0).uniform(-.1,.1,len(im)),im.specific*100,color=col,alpha=.6,s=15,label=f'{side} imputed full (20)')
        ax.scatter([i],[q[q.circuit=='partial'].specific.iloc[0]*100],marker='*',s=150,color='k',label='partial circuit' if i==0 else None)
    ax.axhline(0,color='grey',lw=.8);ax.set_xticks([0,1]);ax.set_xticklabels(['right','left']);ax.set_ylabel('odor-specific leakage (%p)');ax.legend(fontsize=7)
    ax.set_title('28: full circuit with imputed inputs');fig.tight_layout();fig.savefig(out/'full_circuit_imputation.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
