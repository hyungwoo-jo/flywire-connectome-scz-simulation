#!/usr/bin/env python3
"""Multiple inputs, parameter sensitivity and exact input-strength nulls."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('baseline',Path(__file__).with_name('06_input_discrimination.py'))
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)


def strength_preserving_null(pre, post, weights, rng, multiple=5):
    target=post.copy();occupied=set(zip(pre,target))
    groups=[]
    for weight in np.unique(weights):
        ix=np.flatnonzero(weights==weight)
        if len(np.unique(pre[ix]))>1 and len(np.unique(post[ix]))>1:
            groups.append(ix)
    if not groups:
        raise ValueError('No eligible equal-weight swaps')
    probabilities=np.array([len(ix) for ix in groups],dtype=float);probabilities/=probabilities.sum()
    required=multiple*len(pre);done=attempts=0
    while done<required and attempts<required*100:
        attempts+=1;ix=groups[rng.choice(len(groups),p=probabilities)];i,j=rng.choice(ix,2,replace=False)
        u,v,x,y=int(pre[i]),int(target[i]),int(pre[j]),int(target[j])
        if u==x or v==y or (u,y) in occupied or (x,v) in occupied:continue
        occupied.remove((u,v));occupied.remove((x,y));occupied.add((u,y));occupied.add((x,v))
        target[i],target[j]=y,v;done+=1
    if done!=required:raise RuntimeError('Swap budget not met')
    np.testing.assert_array_equal(np.bincount(post),np.bincount(target))
    np.testing.assert_allclose(np.bincount(post,weights=weights),np.bincount(target,weights=weights),rtol=1e-12,atol=1e-12)
    return target,dict(swaps=done,changed_fraction=float(np.mean(target!=post)),
                       changed_weight_fraction=float(weights[target!=post].sum()/weights.sum()),
                       eligible_edges=int(sum(map(len,groups))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=int,default=8);parser.add_argument('--nulls',type=int,default=10);parser.add_argument('--trials',type=int,default=40);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/discrimination_robustness';out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'qc_reports/input_discrimination/manifest.json').read_text());theta=previous['model']['theta']
    for name,expected in previous['data_sha256'].items():
        actual=hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Input changed since baseline: '+name)
    pre,post,w,inh,drive,info=baseline.load_circuit(root);n=len(inh);n_pn=info['pn']
    inputs=[];input_metrics=[]
    for pair in range(args.pairs):
        for oi,overlap in enumerate((.25,.6,.9)):
            for ni,noise in enumerate((.1,.4)):
                seed=20000+pair*1000+oi*100+ni;x,y,actual=baseline.patterns(n_pn,overlap,noise,seed,args.trials)
                inputs.append((pair,overlap,noise,seed,x,y))
                input_metrics.append(dict(pair=pair,input_overlap=overlap,noise=noise,seed=seed,actual_overlap=actual,**baseline.metrics(x,y)))
    rows=[];null_details=[]
    for graph in range(args.nulls+1):
        if graph:
            target,detail=strength_preserving_null(pre,post,w,np.random.default_rng(30000+graph));detail['seed']=30000+graph;null_details.append(detail)
        else:target=post
        matrix=sparse.csr_matrix((w,(target,pre)),shape=(n,n_pn))
        settings=[(t,g) for t in (.8,1.,1.2) for g in (2.,4.,8.)] if graph==0 else [(1.,4.)]
        for pair,overlap,noise,seed,x,y in inputs:
            e=matrix@x
            for theta_factor,gain in settings:
                for strength in (1.,0.):
                    k,residual=baseline.response(e,inh,drive,theta*theta_factor,strength,gain=gain)
                    rows.append(dict(graph=graph,pair=pair,input_overlap=overlap,noise=noise,input_seed=seed,
                                     theta_factor=theta_factor,apl_gain=gain,apl_strength=strength,residual=residual,**baseline.metrics(k,y)))
        print(f'graph {graph}/{args.nulls} complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False);pd.DataFrame(input_metrics).to_csv(out/'pn_input_baseline.csv',index=False)
    keys=['graph','pair','input_overlap','noise','theta_factor','apl_gain']
    a=f[f.apl_strength==1].set_index(keys);b=f[f.apl_strength==0].set_index(keys)
    fields=['accuracy','active_fraction','centroid_cosine','between_within_ratio','mean_activity']
    delta=(b[fields]-a[fields]).add_prefix('delta_').reset_index();delta.to_csv(out/'paired_effects.csv',index=False)
    summary=delta.groupby(['graph','input_overlap','noise','theta_factor','apl_gain']).agg({f'delta_{k}':['mean','min','max'] for k in fields})
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    manifest=dict(circuit=info,data_sha256=previous['data_sha256'],baseline_theta=theta,
                  pairs=args.pairs,trials_per_class=args.trials,conditions=len(f),nulls=null_details,
                  input_seed_formula='20000+pair*1000+overlap_index*100+noise_index',
                  actual_settings=dict(theta_factors=[.8,1,1.2],apl_gains=[2,4,8]),null_settings=dict(theta_factors=[1],apl_gains=[4]),
                  max_residual=float(f.residual.max()),python=previous['python'],numpy=np.__version__,pandas=pd.__version__,
                  limitations=['Input-strength nulls restrict mixing to equal synapse counts; no uniform-null guarantee.',
                               'One noise sample set per pattern pair/overlap; noise levels use independently sampled patterns.',
                               'Scalar APL, phenomenological coefficients, synthetic inputs and one specimen.',
                               'No learning, disease model, statistical significance or biological validation.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sub=delta[(delta.input_overlap==.9)&(delta.noise==.4)]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    real=sub[(sub.graph==0)&(sub.theta_factor==1)&(sub.apl_gain==4)]
    null=sub[sub.graph>0].groupby('pair').delta_accuracy.agg(['min','max']).reindex(real.pair)
    axes[0].fill_between(real.pair,null['min']*100,null['max']*100,alpha=.2,label='Null range')
    axes[0].plot(real.pair,real.delta_accuracy*100,'o-',label='Empirical');axes[0].axhline(0,color='black',lw=.7)
    axes[0].set_title('Accuracy change: APL removal');axes[0].set_xlabel('Input pair');axes[0].set_ylabel('Percentage points');axes[0].legend()
    for ax,metric in zip(axes[1:],('delta_accuracy','delta_centroid_cosine')):
        heat=sub[sub.graph==0].pivot_table(index='theta_factor',columns='apl_gain',values=metric,aggfunc='mean').sort_index()
        im=ax.imshow(heat,cmap='coolwarm',aspect='auto');ax.set_xticks(range(3),heat.columns);ax.set_yticks(range(3),heat.index)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{heat.iloc[i,j]:.3f}',ha='center',va='center')
        ax.set_xlabel('APL gain');ax.set_ylabel('Threshold factor');ax.set_title('Mean '+metric);fig.colorbar(im,ax=ax)
    fig.suptitle('High overlap / high noise; changes are removal minus normal APL');fig.tight_layout();fig.savefig(out/'robustness.png',dpi=180)
    print('conditions',len(f),'max residual',manifest['max_residual'])


if __name__=='__main__':main()
