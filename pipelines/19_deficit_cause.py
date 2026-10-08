#!/usr/bin/env python3
"""Separate KC-subtype allocation from finer overconvergence as the source of the discrimination deficit."""
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


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


disc=load('disc18','18_community_discrimination.py')
com=load('com17','17_pn_community.py')
cal=disc.cal;robust=cal.robust;base=cal.base

N0_SEED0=180000
N1_SEED0=190000
PERMUTATIONS=2000
PERM_SEED=191000
_G={}


def subtype_target_null(pre,post,weights,kc_type,rng):
    """Swap targets only among edges whose target KCs share a subtype."""
    target=post.copy()
    for t in np.unique(kc_type[post]):
        ix=np.flatnonzero(kc_type[post]==t)
        try:
            changed,_=robust.strength_preserving_null(pre[ix],post[ix],weights[ix],rng)
        except ValueError as exc:
            if str(exc)!='No eligible equal-weight swaps':raise
            changed=post[ix].copy()
        target[ix]=changed
    assert np.array_equal(kc_type[target],kc_type[post])
    return target


def structure_score(target,g):
    """Mean co-convergence Z over Hallem-glomerulus pairs, against the 17 null reference."""
    post=g['full_post'].copy();post[g['mask']]=target
    obs=com.co_convergence(g['pre'],post,g['pn_type'],g['n_types'],g['n_kc'])
    z=com.zscores(obs,g['mu'],g['sd'])
    h=g['hallem_idx'];sub=z[np.ix_(h,h)]
    return float(sub[np.triu_indices(len(h),1)].mean())


def _run(job):
    kind,graph=job;g=_G;c=g['c'];pre_m,post_m,w_m=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
    if kind=='real':target=post_m
    elif kind=='N0':target,_=robust.strength_preserving_null(pre_m,post_m,w_m,np.random.default_rng(N0_SEED0+graph))
    else:target=subtype_target_null(pre_m,post_m,w_m,g['kc_type'],np.random.default_rng(N1_SEED0+graph))
    m,_=cal.partial_matrix(c,None if kind=='real' else target)
    rows=[];s=structure_score(target,g)
    for cond in g['conds']:
        k,res=base.response(m@cond['x'],c['inh'],c['drive'],cond['theta'],1.,gain=cal.APL_GAIN)
        rows.append(dict(kind=kind,graph=graph,window=cond['window'],target=cond['target'],residual=res,S=s,
                         **disc.evaluate(k,g['labels'],g['train'],{'ALL':np.arange(g['n'])})))
    return rows


def zof(real,nulls):
    return float((real-nulls.mean())/nulls.std(ddof=1))


def centered_spearman(s,d,kind,rng):
    s=s.copy();d=d.copy()
    for k in np.unique(kind):
        s[kind==k]-=s[kind==k].mean();d[kind==k]-=d[kind==k].mean()
    rho=stats.spearmanr(s,d).statistic
    perm=np.array([stats.spearmanr(s,_permute_within(d,kind,rng)).statistic for _ in range(PERMUTATIONS)])
    return float(rho),float((1+(perm<=rho).sum())/(PERMUTATIONS+1))


def _permute_within(d,kind,rng):
    out=d.copy()
    for k in np.unique(kind):
        ix=np.flatnonzero(kind==k);out[ix]=d[rng.permutation(ix)]
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/deficit_cause';out.mkdir(parents=True,exist_ok=True)
    m16=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=cal.load_partial_circuit(root,receptors);_,mask=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));d=delta[:,keep];n=len(keep)
    labels=np.repeat(np.arange(n),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,n)
    conds=[]
    for wi,window in enumerate((.5,.25)):
        orn=cal.sample_orn(d,sfr.to_numpy(),labels,window,np.random.default_rng(disc.NOISE_SEED+wi));x=cal.pn_drive(c,receptors,orn)
        for target in ('0.10','0.05'):conds.append(dict(window=window,target=float(target),theta=m16['targets'][target]['theta'],x=x))
    # KC subtype labels in load_circuit order (sorted root_888).
    meta=pd.read_feather(root/'data/banc_888_meta.feather');kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].copy()
    kc=kc.set_index(kc.root_888.astype(str)).sort_index();kc_type=kc.cell_sub_class.fillna(kc.cell_type).fillna('unknown').to_numpy(str)
    pns=c['pns'];types=sorted(pns.glomerulus.unique());tix={t:i for i,t in enumerate(types)}
    ref=np.load(root/'qc_reports/pn_community/nulls.npz');nl=ref['nulls'].astype(float)
    hallem_idx=np.array(sorted(tix[gl] for gl in c['mapped'].glomerulus.unique()))
    _G.update(c=c,mask=mask,conds=conds,labels=labels,train=train,n=n,kc_type=kc_type,pre=c['pre'],full_post=c['post'],
              pn_type=pns.sort_values('pn_index').glomerulus.map(tix).to_numpy(),n_types=len(types),n_kc=len(c['inh']),
              mu=nl.mean(0),sd=nl.std(0,ddof=1),hallem_idx=hallem_idx)
    jobs=[('real',0)]+[(k,i) for k in ('N0','N1') for i in range(1,args.nulls+1)]
    with Pool(args.workers) as pool:rows=[r for rs in pool.map(_run,jobs) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    summary=[];rng=np.random.default_rng(PERM_SEED)
    for (window,target),g in f.groupby(['window','target']):
        real=g[g.kind=='real'].iloc[0];r=dict(window=window,target=target,primary=bool(window==.5 and target==.10))
        for metric in ('ALL_dprime','ALL_accuracy'):
            for k in ('N0','N1'):
                nulls=g[g.kind==k][metric].to_numpy();r[f'{metric}_z_{k}']=zof(real[metric],nulls);r[f'{metric}_rank_{k}']=float((nulls>=real[metric]).mean())
        nn=g[g.kind!='real']
        r['rho_centered'],r['rho_p']=centered_spearman(nn.S.to_numpy(),nn.ALL_dprime.to_numpy(),nn.kind.to_numpy(),rng)
        r['rho_pooled']=float(stats.spearmanr(nn.S,nn.ALL_dprime).statistic)
        summary.append(r)
    s=pd.DataFrame(summary);s.to_csv(out/'summary.csv',index=False)
    real_S=float(f[f.kind=='real'].S.iloc[0])
    S_stats={k:dict(mean=float(f[f.kind==k].S.mean()),p05=float(f[f.kind==k].S.quantile(.05)),p95=float(f[f.kind==k].S.quantile(.95))) for k in ('N0','N1')}
    p=s[s.primary].iloc[0];z0,z1=p['ALL_dprime_z_N0'],p['ALL_dprime_z_N1']
    if abs(z1)<1.96 and z1-z0>1:p1='explained_by_subtype_allocation'
    elif z1<=-1.96:p1='not_explained_by_subtype_allocation'
    else:p1='undetermined'
    p2='association' if (p['rho_centered']<0 and p['rho_p']<.05) else 'no_association'
    report=dict(nulls=args.nulls,seeds=dict(N0=[N0_SEED0+1,N0_SEED0+args.nulls],N1=[N1_SEED0+1,N1_SEED0+args.nulls],permutation=PERM_SEED),
        real_S=real_S,null_S=S_stats,P1=dict(z0=z0,z1=z1,verdict=p1),P2=dict(rho_centered=p['rho_centered'],p=p['rho_p'],verdict=p2),
        max_residual=float(f.residual.max()),summary=s.to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pr=f[(f.window==.5)&(f.target==.10)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for k,col in (('N0','#999999'),('N1','#0072B2')):
        axes[0].hist(pr[pr.kind==k].ALL_dprime,bins=20,alpha=.6,color=col,label=f'{k} ({args.nulls})')
        axes[1].scatter(pr[pr.kind==k].S,pr[pr.kind==k].ALL_dprime,s=12,alpha=.6,color=col,label=k)
    rr=pr[pr.kind=='real'].iloc[0]
    axes[0].axvline(rr.ALL_dprime,color='#D55E00',lw=2,label='BANC wiring');axes[0].set_xlabel("mean pairwise d'");axes[0].legend(fontsize=8)
    axes[1].scatter([rr.S],[rr.ALL_dprime],s=60,color='#D55E00',marker='*',label='BANC wiring');axes[1].set_xlabel('S: mean co-convergence Z (21 glomeruli)');axes[1].set_ylabel("mean pairwise d'");axes[1].legend(fontsize=8)
    fig.suptitle('19: N0 strength-preserving vs N1 KC-subtype-preserving nulls (0.5 s, 10% KC)')
    fig.tight_layout();fig.savefig(out/'deficit_cause.png',dpi=180);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('real_S','null_S','P1','P2')},indent=2));print(s.T.to_string())


if __name__=='__main__':main()
