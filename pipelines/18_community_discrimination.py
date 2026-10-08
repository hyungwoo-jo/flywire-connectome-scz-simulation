#!/usr/bin/env python3
"""Real vs strength-preserving nulls on calibrated Hallem input; generic (H1) and community-split (H2) discrimination."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('cal',Path(__file__).with_name('16_hallem_calibration.py'))
cal=importlib.util.module_from_spec(spec);spec.loader.exec_module(cal)
robust=cal.robust;base=cal.base

COMMUNITIES={'C_A':['DM6','VC4','VM5d'],'C_B':['DM2','DM4','VM2','VM3']}
MIN_PN_HZ=10.
TRAIN,TEST=10,20
NULL_SEED0=180000
NOISE_SEED=181000
DPRIME_CAP=10.
_G={}


def pair_dprime(g,labels_test,odors):
    """g: test trials x centroids (dot products). Mean pairwise d' among the given odor indices."""
    vals=[]
    for i,a in enumerate(odors):
        ta=labels_test==a
        for b in odors[i+1:]:
            tb=labels_test==b
            pa=g[ta,a]-g[ta,b];pb=g[tb,a]-g[tb,b]
            diff=abs(pa.mean()-pb.mean());sd=np.sqrt((pa.var(ddof=1)+pb.var(ddof=1))/2)
            # Pre-registered: only zero-variance pairs are set to the cap.
            vals.append((DPRIME_CAP if diff>0 else 0.) if sd==0 else diff/sd)
    return float(np.mean(vals))


def accuracy(k_test,centroids,labels_test,odors):
    sel=np.isin(labels_test,odors);x=k_test[sel];c=centroids[odors]
    d=np.maximum((x*x).sum(1)[:,None]+(c*c).sum(1)[None,:]-2*x@c.T,0)
    ties=np.isclose(d,d.min(1)[:,None],rtol=1e-10,atol=1e-12)
    col={o:i for i,o in enumerate(odors)};truth=np.array([col[l] for l in labels_test[sel]])
    return float((ties[np.arange(len(x)),truth]/ties.sum(1)).mean())


def evaluate(k,labels,train,groups):
    n=labels.max()+1
    centroids=np.stack([k[:,train&(labels==o)].mean(1) for o in range(n)])
    kt=k[:,~train].T;lt=labels[~train];g=kt@centroids.T
    out={}
    for name,odors in groups.items():
        out[f'{name}_dprime']=pair_dprime(g,lt,odors);out[f'{name}_accuracy']=accuracy(kt,centroids,lt,odors)
    out['active_fraction']=float(np.mean(k>1e-8))
    return out


def _run(graph):
    g=_G;c=g['c']
    if graph:
        mask=g['mask'];target,_=robust.strength_preserving_null(c['pre'][mask],c['post'][mask],c['w'][mask],np.random.default_rng(NULL_SEED0+graph))
        m,_=cal.partial_matrix(c,target)
    else:m=g['m']
    rows=[]
    for cond in g['conds']:
        k,res=base.response(m@cond['x'],c['inh'],c['drive'],cond['theta'],1.,gain=cal.APL_GAIN)
        rows.append(dict(graph=graph,window=cond['window'],target=cond['target'],residual=res,**evaluate(k,g['labels'],g['train'],g['groups'])))
    return rows


def zstats(frame,metric):
    real=frame[frame.graph==0][metric].iloc[0];nulls=frame[frame.graph>0][metric].to_numpy()
    return float((real-nulls.mean())/nulls.std(ddof=1)),nulls


def d_test(frame,name,metric='dprime'):
    hi,lo=f'{name}_HIGH_{metric}',f'{name}_LOW_{metric}'
    zh,nh=zstats(frame,hi);zl,nl=zstats(frame,lo);d=zh-zl
    pseudo=[]
    for i in range(len(nh)):
        oh=np.delete(nh,i);ol=np.delete(nl,i)
        pseudo.append((nh[i]-oh.mean())/oh.std(ddof=1)-(nl[i]-ol.mean())/ol.std(ddof=1))
    pseudo=np.array(pseudo)
    return dict(z_high=zh,z_low=zl,D=d,p_one_sided=float((1+(pseudo>=d).sum())/(len(pseudo)+1)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/community_discrimination';out.mkdir(parents=True,exist_ok=True)
    manifest16=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=cal.load_partial_circuit(root,receptors);m,mask=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    pn_clean=cal.olsen(delta)
    keep=np.flatnonzero((~calib)&(pn_clean.max(0)>=MIN_PN_HZ))
    d=delta[:,keep];n=len(keep)
    labels=np.repeat(np.arange(n),TRAIN+TEST);train=np.tile(np.arange(TRAIN+TEST)<TRAIN,n)
    # Input-only community share per odor (noise-free mapped PN rates).
    x_clean=cal.pn_drive(c,receptors,d);glom=c['mapped'].glomerulus.to_numpy()
    groups={'ALL':np.arange(n)};shares={}
    for name,members in COMMUNITIES.items():
        share=x_clean[np.isin(glom,members)].sum(0)/x_clean.sum(0);shares[name]=share
        med=np.median(share)
        groups[f'{name}_HIGH']=np.flatnonzero(share>med);groups[f'{name}_LOW']=np.flatnonzero(share<=med)
    conds=[]
    for wi,window in enumerate((.5,.25)):
        orn=cal.sample_orn(d,sfr.to_numpy(),labels,window,np.random.default_rng(NOISE_SEED+wi))
        x=cal.pn_drive(c,receptors,orn)
        for target in ('0.10','0.05'):
            conds.append(dict(window=window,target=float(target),theta=manifest16['targets'][target]['theta'],x=x))
    _G.update(c=c,m=m,mask=mask,conds=conds,labels=labels,train=train,groups=groups)
    with Pool(args.workers) as pool:
        rows=[r for rs in pool.map(_run,range(args.nulls+1)) for r in rs]
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    results=[]
    for (window,target),f in frame.groupby(['window','target']):
        r=dict(window=window,target=target,primary=bool(window==.5 and target==.10))
        for metric in ('dprime','accuracy'):
            z,_=zstats(f,f'ALL_{metric}');r[f'H1_z_all_{metric}']=z
            r[f'real_ALL_{metric}']=float(f[f.graph==0][f'ALL_{metric}'].iloc[0])
            for name in COMMUNITIES:
                for kk,v in d_test(f,name,metric).items():r[f'{name}_{metric}_{kk}']=v
        r['real_active_fraction']=float(f[f.graph==0].active_fraction.iloc[0])
        results.append(r)
    res=pd.DataFrame(results);res.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(dict(odor_key=odors.index[keep],name=names.to_numpy()[keep],**{f'{k}_share':v for k,v in shares.items()},
        **{f'{k}_group':np.where(np.isin(np.arange(n),groups[f'{k}_HIGH']),'HIGH','LOW') for k in COMMUNITIES})).to_csv(out/'odor_groups.csv',index=False)
    report=dict(nulls=args.nulls,null_seeds=[NULL_SEED0+1,NULL_SEED0+args.nulls],noise_seeds=[NOISE_SEED,NOISE_SEED+1],
        evaluation_odors_used=n,excluded_low_input=int((~calib).sum()-n),min_pn_hz=MIN_PN_HZ,communities=COMMUNITIES,
        group_sizes={k:len(v) for k,v in groups.items()},max_residual=float(frame.residual.max()),
        results=res.to_dict(orient='records'),
        status=dict(H1='pre-registered',H2='exploratory definitions C_A, C_B (see PREREGISTRATION_16_18.md addendum 1)'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    prim=frame[(frame.window==.5)&(frame.target==.10)]
    for ax,col in zip(axes,['ALL_dprime','C_A_HIGH_dprime','C_B_HIGH_dprime']):
        ax.hist(prim[prim.graph>0][col],bins=20,color='#999');ax.axvline(prim[prim.graph==0][col].iloc[0],color='#D55E00',lw=2,label='BANC wiring')
        low=col.replace('HIGH','LOW')
        if low!=col:ax.hist(prim[prim.graph>0][low],bins=20,color='#0072B2',alpha=.4,label='LOW group nulls');ax.axvline(prim[prim.graph==0][low].iloc[0],color='#0072B2',lw=2,ls='--',label='BANC LOW')
        ax.set_title(col);ax.set_xlabel("mean pairwise d'");ax.legend(fontsize=7)
    fig.suptitle(f'Primary condition (0.5 s window, 10% KC); grey/blue = {args.nulls} nulls')
    fig.tight_layout();fig.savefig(out/'community_discrimination.png',dpi=180);plt.close(fig)
    print(res.T.to_string())


if __name__=='__main__':main()
