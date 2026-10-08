#!/usr/bin/env python3
"""Anatomically constrained KC-MBON LTD assay with held-out inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('robust',Path(__file__).with_name('07_discrimination_robustness.py'))
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
base=robust.baseline


def load_output(root):
    m=pd.read_feather(root/'data/banc_888_meta.feather');e=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    if m.root_888.duplicated().any():raise ValueError('Ambiguous annotation')
    kc=sorted(m[(m.cell_class=='kenyon_cell')&(m.side=='right')].root_888.astype(str))
    mb=m[(m.side=='right')&m.cell_type.fillna('').str.startswith('MBON')].copy();mb['id']=mb.root_888.astype(str)
    edges=e[e.pre.isin(kc)&e.post.isin(mb.id)].copy()
    eligible=edges.groupby('post').pre.nunique();ids=sorted(eligible[eligible>=10].index)
    edges=edges[edges.post.isin(ids)];lookup={k:i for i,k in enumerate(kc)};out={k:i for i,k in enumerate(ids)}
    pre=edges.pre.map(lookup).to_numpy(dtype=int);post=edges.post.map(out).to_numpy(dtype=int);count=edges['count'].to_numpy(dtype=float)
    totals=np.bincount(post,weights=count,minlength=len(ids));main=int(np.argmax(totals))
    labels=mb.set_index('id').loc[ids,['cell_type','neurotransmitter_verified']].fillna('unknown')
    info=dict(ids=ids,types=labels.cell_type.tolist(),transmitters=labels.neurotransmitter_verified.tolist(),
              main_id=ids[main],main_type=labels.cell_type.iloc[main],main_selection='largest total KC input count, before outcome',
              neurons=len(ids),edges=len(edges),synapses=int(count.sum()),kc_with_output=int(len(np.unique(pre))),
              min_distinct_kc_inputs=10)
    return pre,post,count,totals,main,info


def learn(weights, activity, eta, gate=1.0):
    if eta<0 or gate<0:raise ValueError('Nonnegative learning coefficients required')
    return weights*np.exp(-eta*gate*activity[None,:])


def auc(scores,labels):
    pos=labels==0;n=int(pos.sum());q=len(labels)-n
    return float((rankdata(scores)[pos].sum()-n*(n+1)/2)/(n*q))


def decision_threshold(scores,labels):
    values=np.unique(scores);thresholds=np.r_[values[0]-1e-12,(values[:-1]+values[1:])/2,values[-1]+1e-12]
    accuracy=np.mean((scores[:,None]>thresholds[None,:])==(labels[:,None]==0),axis=0)
    # Deterministic tie handling; never choose thresholds from held-out data.
    return float(thresholds[np.flatnonzero(accuracy==accuracy.max())[len(np.flatnonzero(accuracy==accuracy.max()))//2]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=int,default=8);parser.add_argument('--nulls',type=int,default=5);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/associative_readout';out.mkdir(parents=True,exist_ok=True)
    old=json.loads((root/'qc_reports/input_discrimination/manifest.json').read_text());theta=old['model']['theta']
    for name,h in old['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Baseline data changed')
    pi,ki,w,inh,drive,circuit=base.load_circuit(root);n=len(inh);npn=circuit['pn']
    pn=sparse.csr_matrix((w,(ki,pi)),shape=(n,npn))
    pre,post,counts,totals,main,info=load_output(root)
    cache=[]
    for pair in range(args.pairs):
        for oi,overlap in enumerate((.25,.6,.9)):
            for ni,noise in enumerate((.1,.4)):
                seed=40000+pair*1000+oi*100+ni;x,y,actual=base.patterns(npn,overlap,noise,seed,80)
                train=np.zeros(len(y),dtype=bool)
                for c in (0,1):train[np.flatnonzero(y==c)[:40]]=True
                excitation=pn@x
                representations={s:base.response(excitation,inh,drive,theta,s)[0] for s in (1.,0.)}
                cache.append((pair,overlap,noise,seed,y,train,representations))
    rows=[];nulls=[]
    for graph in range(args.nulls+1):
        if graph:
            target,detail=robust.strength_preserving_null(pre,post,counts,np.random.default_rng(50000+graph));detail['seed']=50000+graph;nulls.append(detail)
        else:target=post
        weights=sparse.csr_matrix((counts/totals[target],(target,pre)),shape=(len(totals),n)).toarray()
        for pair,overlap,noise,seed,y,train,representations in cache:
            for train_apl in (1.,0.):
                ktrain=representations[train_apl][:,train];yt=y[train]
                for eta in (5.,20.):
                    for condition in ('no_gate','train_A','train_B'):
                        c=1 if condition=='train_B' else 0
                        mean=ktrain[:,yt==c].mean(axis=1)
                        learned=learn(weights,mean,eta,gate=float(condition!='no_gate'))
                        training_scores=-learned@ktrain
                        thresholds=[decision_threshold(training_scores[u],yt) for u in range(len(totals))]
                        for test_apl in (1.,0.):
                            k=representations[test_apl];scores=-learned@k
                            for unit in range(len(totals)):
                                threshold=thresholds[unit];test=scores[unit,~train];labels=y[~train]
                                rows.append(dict(graph=graph,pair=pair,input_overlap=overlap,noise=noise,input_seed=seed,
                                    train_apl=train_apl,test_apl=test_apl,eta=eta,condition=condition,
                                    mbon_id=info['ids'][unit],mbon_type=info['types'][unit],is_main=unit==main,
                                    auc_A=auc(test,labels),accuracy_A=float(np.mean((test>threshold)==(labels==0))),
                                    A_hit_fraction=float(np.mean(test[labels==0]>threshold)),B_assigned_A_fraction=float(np.mean(test[labels==1]>threshold)),
                                    zero_input_assigned_A=float(0>threshold),threshold=threshold,
                                    mean_response_A=float(-test[labels==0].mean()),mean_response_B=float(-test[labels==1].mean()),
                                    retained_weight_fraction=float(learned[unit].sum()/weights[unit].sum())))
        print(f'output graph {graph}/{args.nulls} complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    main_frame=f[f.is_main];main_frame.to_csv(out/'main_mbon_metrics.csv',index=False)
    summary=main_frame.groupby(['graph','input_overlap','noise','train_apl','test_apl','eta','condition'])[['auc_A','accuracy_A','zero_input_assigned_A','retained_weight_fraction']].agg(['mean','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'main_summary.csv',index=False)
    manifest=dict(circuit=circuit,output_circuit=info,data_sha256=old['data_sha256'],baseline_theta=theta,pairs=args.pairs,
        training_trials_per_class=40,evaluation_trials_per_class=40,learning_rule='W_after = W_before * exp(-eta * gate * mean_training_KC)',
        readout='negative MBON rate; A/B threshold fitted under training APL and held fixed at test; higher score denotes A',
        dopamine='abstract gate, no DAN anatomy, receptor kinetics or concentrations modeled',
        input_seeds='40000+pair*1000+overlap_index*100+noise_index',nulls=nulls,conditions=len(f),
        no_activity='exact zero PN input implies zero KC/MBON response; report forced-choice A assignment, not hallucination',
        fixed_scaling='Each empirical MBON row divided by its baseline total count; no renormalization after learning',
        interpretation='Single MBON synthetic assay, not a behavior/valence model. All units inspected; main chosen by anatomy.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sub=main_frame[(main_frame.train_apl==1)&(main_frame.test_apl==1)&(main_frame.eta==20)&(main_frame.noise==.4)]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,('auc_A','accuracy_A','retained_weight_fraction')):
        for condition,color in zip(('no_gate','train_A','train_B'),('#777777','#0072B2','#D55E00')):
            real=sub[(sub.graph==0)&(sub.condition==condition)].groupby('input_overlap')[metric].mean()
            null=sub[(sub.graph>0)&(sub.condition==condition)].groupby(['graph','input_overlap'])[metric].mean().groupby('input_overlap').agg(['min','max'])
            ax.fill_between(null.index,null['min'],null['max'],alpha=.15,color=color);ax.plot(real.index,real,'o-',label=condition,color=color)
        ax.set_title(metric);ax.set_xlabel('Input overlap');ax.set_ylim(0,1.02);ax.legend(fontsize=8)
    fig.suptitle('Main MBON assay: empirical line / input-strength null range; noise .4, eta 20')
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=180)
    print('rows',len(f),'main',info['main_id'],info['main_type'])


if __name__=='__main__':main()
