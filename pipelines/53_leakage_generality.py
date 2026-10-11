#!/usr/bin/env python3
"""Step 53: is the learned-value leakage (step 43, hub factor H) a general property of expansion-layer circuits?

Synthetic expansion circuits, no fly wiring: N_IN input channels -> N cells; cell i samples K_i channels (mean K, CV_K,
negative-binomial-like via gamma-Poisson, K_i >= 1) with lognormal connection weights (CV_w).
Inputs: every channel has spontaneous rate 1; noise trials multiply it by (1 + SIGMA_N * N(0,1)), clipped at 0;
stimuli add a lognormal pattern (sigma 1, mean A) on a random 30% of channels.
Cells: k = clip((E - theta) / theta, 0, 1), one global theta giving 10% active cells on stimuli (as in the fly model);
SIGMA_N set per circuit so that 0.3% of cells are active on noise trials (FlyWire right value).
Learning stimulus A: d = 1 - exp(-5 k_A). H = (sum m d / sum m) / mean d, m = mean noise-trial activity; uniform readout.

Conditions: K in {3, 7, 15}, CV_K in {0, 0.5}, CV_w in {0, 0.35, 0.7}; plus for each, (a) equalized total input
strength per cell, (b) homeostatic per-cell thresholds (each cell active on 10% of stimuli). 10 seeds x 40 stimuli.
Measures (fixed before running):
1. Spearman correlation across uncompensated conditions between CV of total input strength and median H
   (prediction: > 0.8).
2. Median H with equalized strength and with homeostatic thresholds (prediction: close to 1, below 2).
3. Fly-like condition (K 3, CV_K 0.5, CV_w 0.7) reported next to the measured fly H (5-12, step 43)."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
N_IN,N,N_STIM,N_NOISE,SEEDS=50,2000,40,400,10
ACT_STIM,ACT_NOISE,ETA,A=.10,.003,5.,2.


def circuit(K,cvk,cvw,rng):
    if cvk>0:
        lam=rng.gamma(1/cvk**2,K*cvk**2,N);k=np.maximum(rng.poisson(lam),1)
    else:k=np.full(N,K)
    k=np.minimum(k,N_IN);W=np.zeros((N,N_IN))
    s2=np.log(1+cvw**2)
    for i in range(N):
        W[i,rng.choice(N_IN,k[i],replace=False)]=rng.lognormal(-s2/2,np.sqrt(s2),k[i]) if cvw>0 else 1.
    return W


def act(E,theta):
    return np.clip((E-theta)/theta,0,1)


def calibrate_theta(Es,frac,per_cell=False):
    return np.quantile(Es,1-frac,axis=1)[:,None] if per_cell else np.quantile(Es,1-frac)


def noise_inputs(sig,rng,n):
    return np.clip(1+sig*rng.standard_normal((N_IN,n)),0,None)


def run(W,rng,mode):
    if mode=='equalized':W=W/W.sum(1,keepdims=True)*W.sum(1).mean()
    X=np.ones((N_IN,N_STIM))
    for j in range(N_STIM):
        on=rng.choice(N_IN,int(.3*N_IN),replace=False);X[on,j]+=rng.lognormal(np.log(A)-.5,1.,len(on))
    Es=W@X;theta=calibrate_theta(Es,ACT_STIM,per_cell=mode=='homeostatic')
    z=rng.standard_normal((N_IN,N_NOISE))
    lo,hi=0.,5.
    for _ in range(40):                                     # bisection on noise amplitude
        sig=(lo+hi)/2;f=(act(W@np.clip(1+sig*z,0,None),theta)>0).mean()
        lo,hi=(sig,hi) if f<ACT_NOISE else (lo,sig)
    m=act(W@np.clip(1+sig*z,0,None),theta).mean(1);ks=act(Es,theta)
    H=[]
    for j in range(N_STIM):
        d=1-np.exp(-ETA*ks[:,j])
        if d.mean()>0 and m.sum()>0:H.append(float((m*d).sum()/m.sum()/d.mean()))
    st=W.sum(1)
    return dict(H=float(np.median(H)),strength_cv=float(st.std()/st.mean()),noise_sigma=float(sig),
                noise_active=float((act(W@np.clip(1+sig*z,0,None),theta)>0).mean()))


def main():
    out=ROOT/'qc_reports/leakage_generality';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for (K,cvk,cvw),mode in itertools.product(itertools.product((3,7,15),(0.,.5),(0.,.35,.7)),('none','equalized','homeostatic')):
        for s in range(SEEDS):
            rng=np.random.default_rng(530000+s);W=circuit(K,cvk,cvw,rng)
            rows.append(dict(K=K,cv_k=cvk,cv_w=cvw,mode=mode,seed=s,**run(W,rng,mode)))
    f=pd.DataFrame(rows);f.to_csv(out/'per_seed.csv',index=False)
    g=f.groupby(['K','cv_k','cv_w','mode'])[['H','strength_cv','noise_active']].median().reset_index()
    g.to_csv(out/'per_condition.csv',index=False)
    u=g[g['mode']=='none'];rho,p=stats.spearmanr(u.strength_cv,u.H)
    report=dict(spearman_strength_cv_vs_H=dict(rho=float(rho),p=float(p),n=len(u)),
                median_H={md:float(g[g['mode']==md].H.median()) for md in ('none','equalized','homeostatic')},
                max_H={md:float(g[g['mode']==md].H.max()) for md in ('none','equalized','homeostatic')},
                fly_like=g[(g.K==3)&(g.cv_k==.5)&(g.cv_w==.7)].round(3).to_dict(orient='records'),
                fly_measured_H='5-12 (step 43)')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',200);print(g.round(3).to_string());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
