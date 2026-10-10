#!/usr/bin/env python3
"""Shared setup for the downstream tracks (47-49): FlyWire right calibrated model, MBON readout and learning compartments,
comparators and integrators, KC responses to odors, mixtures, weak odors and noise trials."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def _load(name,file):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,ROOT/'pipelines'/file)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p46=_load('p46','46_leak_to_behavior_and_and_detectors.py');p45=p46.p45;p38=p46.p38;p34=p45.p34;cal=p45.cal;p24=p45.p24
ETA=p45.ETA
SIGN=p45.SIGN


def setup(n_noise=500):
    odors,sfr,names=cal.load_hallem(ROOT);rec=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index));sf=sfr.to_numpy()
    a,e=p34.load_flywire(ROOT);c,_,_=p34.side_circuit(a,e,'right',rec,ROOT)
    kc_ids=sorted(a[(a.cell_class=='Kenyon_Cell')&(a.side=='right')].id)
    m,_=cal.partial_matrix(c);theta=cal.calibrate(m@cal.pn_drive(c,rec,delta[:,calib]),c['inh'],c['drive'],.10)
    keep,W,comp,sign,mtypes=p45.mbon_setup(a,e,kc_ids)
    comps=p46.comparators(a,e,keep,comp)
    integ=pd.read_csv(ROOT/'qc_reports/acc_like_integrators/pair_MBON_BODY.csv').id.astype(str).tolist()
    ev=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=10))
    resp=lambda x:p24.kc(m,x,c,theta)[0]
    zero=np.zeros((24,1))
    kn=resp(cal.pn_drive(c,rec,cal.sample_orn(zero,sf,np.zeros(n_noise,int),.5,np.random.default_rng(p46.SEED_NOISE))))
    kcal=resp(cal.pn_drive(c,rec,cal.sample_orn(zero,sf,np.zeros(1000,int),.5,np.random.default_rng(230001))))
    rep=kn.sum(0)>np.percentile(kcal.sum(0),95)
    kfull=resp(cal.pn_drive(c,rec,delta[:,ev]))
    kweak={int(o):resp(cal.pn_drive(c,rec,cal.sample_orn(.1*delta[:,[o]],sf,np.zeros(20,int),.5,np.random.default_rng(p46.SEED_WEAK+o)))) for o in ev}
    strength=np.asarray(m.sum(1)).ravel();has=strength>0
    dec=np.full(len(strength),-1);dec[has]=np.minimum((pd.Series(strength[has]).rank(pct=True).to_numpy()*10).astype(int),9)
    return dict(a=a,e=e,c=c,m=m,theta=theta,rec=rec,delta=delta,names=names,keep=keep,W=W,comp=comp,sign=sign,mtypes=mtypes,
                comps=comps,integ=integ,ev=ev,resp=resp,kn=kn,rep=rep,kfull=kfull,kweak=kweak,strength=strength,has=has,dec=dec)


def signed_matrix(e,a,pre_ids,post_ids,normalize_by_post_total=False):
    """post x pre matrix of synapse counts signed by the presynaptic neuron's predicted transmitter."""
    q=e[e.pre.isin(pre_ids)&e.post.isin(post_ids)]
    mat=q.pivot_table(index='post',columns='pre',values='count',aggfunc='sum',fill_value=0).reindex(index=post_ids,columns=pre_ids,fill_value=0).to_numpy(float)
    sgn=a.set_index('id').reindex(pre_ids).top_nt.map(SIGN).fillna(0.).to_numpy()
    mat=mat*sgn[None,:]
    if normalize_by_post_total:
        tot=e[e.post.isin(post_ids)].groupby('post')['count'].sum().reindex(post_ids).fillna(1.).to_numpy()
        mat=mat/tot[:,None]
    return mat


def learned_dW(W,comp,pattern,which='aversive'):
    Wl=W.copy();Wl[comp==which]*=np.exp(-ETA*pattern)[None,:];return Wl-W


def control_patterns(kA,S,odor_seed):
    act=kA>1e-8;rng=np.random.default_rng(460000+odor_seed)
    rnd=np.zeros_like(kA);rnd[rng.choice(len(kA),int(act.sum()),replace=False)]=rng.permutation(kA[act])
    tgt,src=p38.matched_set(act&S['has'],S['strength'],S['dec'],np.random.default_rng(470000+odor_seed));mat=np.zeros_like(kA);mat[tgt]=kA[src]
    return rnd,mat
