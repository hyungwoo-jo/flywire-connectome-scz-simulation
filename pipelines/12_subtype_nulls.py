#!/usr/bin/env python3
"""Preserve KC subtype composition as well as synaptic strengths in output nulls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('compare',Path(__file__).with_name('10_readout_comparison.py'))
compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
learning=compare.learning;base=compare.base;rewire=learning.robust.strength_preserving_null


def subtype_null(pre,post,weights,types,rng,multiple=5):
    target=post.copy();details=[]
    for subtype in sorted(set(types[pre])):
        ix=np.flatnonzero(types[pre]==subtype)
        try:
            changed,detail=rewire(pre[ix],post[ix],weights[ix],rng,multiple=multiple)
        except ValueError as exc:
            if str(exc)!='No eligible equal-weight swaps':raise
            changed=post[ix].copy();detail=dict(swaps=0,changed_fraction=0.,changed_weight_fraction=0.,eligible_edges=0)
        target[ix]=changed;details.append(dict(subtype=str(subtype),edges=len(ix),**detail))
        np.testing.assert_array_equal(np.bincount(post[ix]),np.bincount(target[ix]))
        np.testing.assert_allclose(np.bincount(post[ix],weights=weights[ix]),np.bincount(target[ix],weights=weights[ix]),atol=1e-10)
    return target,dict(groups=details,changed_fraction=float(np.mean(target!=post)),changed_weight_fraction=float(weights[target!=post].sum()/weights.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=int,default=8);p.add_argument('--nulls',type=int,default=10);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/subtype_nulls';out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((root/'qc_reports/associative_readout/manifest.json').read_text());theta=prior['baseline_theta']
    for name,h in prior['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed')
    pi,ki,w,inh,drive,info=base.load_circuit(root);n=len(inh);npn=info['pn'];pn=sparse.csr_matrix((w,(ki,pi)),shape=(n,npn))
    pre,post,count,totals,main,output_info=learning.load_output(root)
    m=pd.read_feather(root/'data/banc_888_meta.feather');kc=m[(m.cell_class=='kenyon_cell')&(m.side=='right')].copy();kc['id']=kc.root_888.astype(str);kc=kc.set_index('id').sort_index()
    types=kc.cell_sub_class.fillna(kc.cell_type).fillna('unknown').to_numpy(dtype=str)
    outputs=[];details=[]
    for kind in ('empirical','strength_only','subtype_strength'):
        for graph in range(1 if kind=='empirical' else args.nulls):
            if kind=='empirical':target=post;detail={};seed=None
            elif kind=='strength_only':seed=50001+graph;target,detail=rewire(pre,post,count,np.random.default_rng(seed))
            else:seed=180001+graph;target,detail=subtype_null(pre,post,count,types,np.random.default_rng(seed))
            detail=dict(kind=kind,graph=graph,seed=seed,**detail);details.append(detail)
            output=sparse.csr_matrix((count/totals[target],(target,pre)),shape=(len(totals),n)).toarray()[main:main+1]
            outputs.append((kind,graph,output))
        print('null construction',kind,'complete',flush=True)
    rows=[];max_residual=0.
    for pair in range(args.pairs):
        # Exactly reuse phase-10 validation patterns to isolate the null definition.
        seed=80000+pair*1000+201;x,y,_=base.patterns(npn,.9,.4,seed,80)
        selected=np.zeros(len(y),dtype=bool)
        for c in (0,1):selected[np.flatnonzero(y==c)[:40]]=True
        ktrain,res=base.response(pn@x[:,selected],inh,drive,theta,1);max_residual=max(max_residual,res);yt=y[selected]
        labels=np.repeat([0,1],100);bases=compare.presence.pattern_bases(npn,.9,seed);eval_seed=90000+pair*100+1
        x=np.maximum(0,bases[:,labels]+np.random.default_rng(eval_seed).normal(0,.4,(npn,len(labels))))
        ktest,res=base.response(pn@x,inh,drive,theta,1);max_residual=max(max_residual,res)
        for kind,graph,before in outputs:
            for gate in (0.,1.):
                after=learning.learn(before,ktrain[:,yt==0].mean(axis=1),20,gate=gate)
                train_scores=compare.read_scores(before,after,ktrain);test_scores=compare.read_scores(before,after,ktest)
                for mode,score in test_scores.items():
                    threshold=learning.decision_threshold(train_scores[mode],yt)
                    rows.append(dict(kind=kind,graph=graph,pair=pair,learning_gate=gate,readout=mode,
                        accuracy=float(np.mean((score>threshold)==(labels==0))),auc_A=learning.auc(score,labels),
                        threshold=threshold,pattern_seed=seed,evaluation_seed=eval_seed))
        print(f'pair {pair+1}/{args.pairs} complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    f.groupby(['kind','graph','learning_gate','readout'])[['accuracy','auc_A']].mean().reset_index().to_csv(out/'graph_means.csv',index=False)
    real_indices=np.flatnonzero(post==main);composition=pd.DataFrame(dict(subtype=types[pre[real_indices]],count=count[real_indices])).groupby('subtype')['count'].agg(['size','sum']);composition.to_csv(out/'main_input_composition.csv')
    manifest=dict(data_sha256=prior['data_sha256'],pairs=args.pairs,nulls_per_kind=args.nulls,rows=len(f),main_mbon=output_info['main_id'],
        subtype_column='cell_sub_class; fallback cell_type, then unknown',kc_type_counts=pd.Series(types).value_counts().to_dict(),
        construction=details,max_residual=max_residual,
        definition='Equal-weight target swaps restricted to presynaptic KC subtype; MBON indegree and input strength preserved per subtype.',
        constraints='KC outdegree/weight multiset, MBON indegree/strength, KC subtype to MBON cell-type weighted block composition.',
        evaluation='Same phase-10 validation patterns/labels/noise, amplitude 1, noise .4, normal APL, learning eta 20. No PN presence gate in this assay.',
        limitations=['Broad annotated KC subtype, not receptor, synapse location or compartment geometry.',
                     'Restricted swaps, no uniform graph sampling claim.',
                     'Results already inspected before defining subtype control; exploratory.',
                     'Eight patterns, one specimen, main MBON and privileged fractional readout.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4));modes=('raw','activity_normalized','counterfactual_fraction')
    means=f[f.learning_gate==1].groupby(['kind','graph','readout'])[['accuracy','auc_A']].mean().reset_index()
    for ax,metric in zip(axes,('accuracy','auc_A')):
        for idx,kind in enumerate(('empirical','strength_only','subtype_strength')):
            group=means[means.kind==kind]
            for j,mode in enumerate(modes):
                vals=group[group.readout==mode][metric].to_numpy();pos=j+(idx-1)*.18
                ax.scatter(np.full(len(vals),pos),vals,alpha=.6,s=22,label=kind if j==0 else None)
        ax.set_xticks(range(3),modes,rotation=15);ax.set_ylim(.4,1.0);ax.set_title(metric);ax.legend(fontsize=8)
    fig.suptitle('Main MBON; graph means across 8 identical input pairs')
    fig.tight_layout();fig.savefig(out/'subtype_nulls.png',dpi=180)
    print('rows',len(f),'max residual',max_residual)


if __name__=='__main__':main()
