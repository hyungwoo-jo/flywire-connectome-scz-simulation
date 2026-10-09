#!/usr/bin/env python3
"""Does KC->MBON learning alone make the learned odor's value signal appear on noise trials?"""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p24',Path(__file__).with_name('24_excitability_expectation.py'))
p24=importlib.util.module_from_spec(spec);spec.loader.exec_module(p24)
p23=p24.p23;cal=p24.cal;robust=cal.robust

MBON11=dict(right='720575941586084230',left='720575941575561885')
ETA=5.
SEEDS=dict(noise=250002,signal=250003,control=250200,boot=250300)
N_BOOT=1000
_G={}


def mbon_weights(root,side,n_kc):
    m=pd.read_feather(root/'data/banc_888_meta.feather');e=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    kc=sorted(m[(m.cell_class=='kenyon_cell')&(m.side==side)].root_888.astype(str));idx={k:i for i,k in enumerate(kc)}
    ed=e[(e.post==MBON11[side])&e.pre.isin(kc)]
    w=np.zeros(n_kc);np.add.at(w,ed.pre.map(idx).to_numpy(int),ed['count'].to_numpy(float))
    return w/w.sum()


def value_signal(w,learn,k):
    """Relative MBON drop L(k) after learning pattern `learn`; 0 where naive output is 0."""
    before=w@k;after=(w*np.exp(-ETA*learn))@k
    return np.where(before>0,1-after/np.where(before>0,before,1),0.)


def leakage(g,kt,kn,kw,crit,w):
    rep=kn.sum(0)>crit;rows=[]
    for a in range(g['n']):
        learn=kt[:,a];act=np.flatnonzero(learn>1e-8)
        lw=value_signal(w,learn,kw[:,g['lab']==a]);lcrit=.5*np.median(lw)
        rng=np.random.default_rng(SEEDS['control']+a);ctl=np.zeros_like(learn)
        ctl[rng.choice(len(learn),len(act),replace=False)]=rng.permutation(learn[act])
        le=value_signal(w,learn,kn);lc=value_signal(w,ctl,kn)
        rows.append(dict(odor=a,n_active=len(act),L_crit=float(lcrit),
                         leak_learned=float(np.mean(rep&(le>lcrit))),leak_control=float(np.mean(rep&(lc>lcrit))),
                         noise_L_learned=float(le[rep].mean()) if rep.any() else 0.,noise_L_control=float(lc[rep].mean()) if rep.any() else 0.))
    return rows


def evaluate(m,g):
    c=g['c'];th=g['theta']
    kt,_=p24.kc(m,g['x_full'],c,th);kn,_=p24.kc(m,g['x_noise'],c,th);kw,_=p24.kc(m,g['x_weak'],c,th)
    crit=float(np.percentile(p24.kc(m,g['x_cal'],c,th)[0].sum(0),95))
    rows=leakage(g,kt,kn,kw,crit,g['w'])
    # P25: KC input strength from mapped PNs vs noise activation probability.
    strength=np.asarray(np.abs(m).sum(axis=1)).ravel();pact=(kn>1e-8).mean(1)
    rho=stats.spearmanr(strength,pact)
    return rows,dict(rho=float(rho.statistic),p=float(rho.pvalue),noise_FA=float(np.mean(kn.sum(0)>crit)))


def _run(job):
    side,graph=job;g=_G[side];c=g['c']
    pm,qm,wm=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
    target,_=robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(p23.SIDES[side]['null_seed0']+graph))
    m,_=cal.partial_matrix(c,target);rows,_=evaluate(m,g)
    d=np.mean([r['leak_learned']-r['leak_control'] for r in rows])
    return dict(side=side,graph=graph,mean_delta=float(d))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/learned_value_leakage';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=SEEDS['noise'],signal=SEEDS['signal'])
    res={};per=[]
    for side in p23.SIDES:
        g=p24.setup(root,side);g['w']=mbon_weights(root,side,len(g['c']['inh']));_G[side]=g
        rows,p25=evaluate(g['m'],g);per+=[dict(side=side,**r) for r in rows]
        d=np.array([r['leak_learned']-r['leak_control'] for r in rows]);rng=np.random.default_rng(SEEDS['boot'])
        boot=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(N_BOOT)]
        res[side]=dict(mean_delta=float(d.mean()),ci=[float(x) for x in np.percentile(boot,[2.5,97.5])],
                       leak_learned=float(np.mean([r['leak_learned'] for r in rows])),leak_control=float(np.mean([r['leak_control'] for r in rows])),
                       noise_L_learned=float(np.mean([r['noise_L_learned'] for r in rows])),noise_L_control=float(np.mean([r['noise_L_control'] for r in rows])),
                       P25=p25,mbon11=MBON11[side],mbon_kc_inputs=int(np.count_nonzero(g['w'])))
    pd.DataFrame(per).to_csv(out/'per_odor.csv',index=False)
    with Pool(args.workers) as pool:nulls=pd.DataFrame(pool.map(_run,[(s,i) for s in p23.SIDES for i in range(1,args.nulls+1)]))
    nulls.to_csv(out/'nulls.csv',index=False)
    for side in p23.SIDES:
        nl=nulls[nulls.side==side].mean_delta;res[side]['wiring_z']=float((res[side]['mean_delta']-nl.mean())/nl.std(ddof=1));res[side]['null_mean_delta']=float(nl.mean())
    R,L=res['right'],res['left']
    v=dict(H25='supported' if (R['ci'][0]>0 and L['ci'][0]>0) else 'not_supported',
           P25='supported' if all(x['P25']['rho']>0 and x['P25']['p']<.05 for x in (R,L)) else 'not_supported',
           wiring='wiring_specific' if (abs(R['wiring_z'])>=1.96 and abs(L['wiring_z'])>=1.96 and np.sign(R['wiring_z'])==np.sign(L['wiring_z'])) else 'no_wiring_specificity')
    report=dict(verdicts=v,results=res,eta=ETA,seeds=SEEDS,nulls=args.nulls)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pf=pd.DataFrame(per)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for side,col in (('right','#D55E00'),('left','#0072B2')):
        q=pf[pf.side==side];axes[0].scatter(q.leak_control*100,q.leak_learned*100,s=12,color=col,alpha=.7,label=side)
        axes[1].hist(nulls[nulls.side==side].mean_delta*100,bins=20,alpha=.4,color=col,label=f'{side} nulls');axes[1].axvline(res[side]['mean_delta']*100,color=col,lw=2,label=f'{side} BANC')
    lim=max(pf.leak_learned.max(),pf.leak_control.max())*100;axes[0].plot([0,lim],[0,lim],'k:',lw=1)
    axes[0].set_xlabel('leakage, random-KC learning (%)');axes[0].set_ylabel('leakage, odor learning (%)');axes[0].legend(fontsize=7)
    axes[1].set_xlabel('mean leakage difference (%)');axes[1].legend(fontsize=7)
    fig.suptitle('25: learned value signal on noise trials');fig.tight_layout();fig.savefig(out/'learned_value_leakage.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
