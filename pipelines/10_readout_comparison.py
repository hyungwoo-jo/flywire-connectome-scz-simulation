#!/usr/bin/env python3
"""Compare raw rate, current-activity normalization and counterfactual plasticity."""
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

spec=importlib.util.spec_from_file_location('presence',Path(__file__).with_name('09_signal_presence.py'))
presence=importlib.util.module_from_spec(spec);spec.loader.exec_module(presence)
learning=presence.learning;base=presence.base


def read_scores(before,after,k):
    original=(before@k).ravel();current=(after@k).ravel();population=k.mean(axis=0)
    fractional=np.divide(original-current,original,out=np.zeros_like(current),where=original>1e-12)
    normalized=np.divide(-current,population,out=np.zeros_like(current),where=population>1e-12)
    return {'raw':-current,'activity_normalized':normalized,'counterfactual_fraction':fractional}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=int,default=8);parser.add_argument('--nulls',type=int,default=10);parser.add_argument('--trials',type=int,default=100);parser.add_argument('--validation',action='store_true');args=parser.parse_args()
    if args.pairs<1 or args.nulls<0 or args.trials<2:parser.error('positive pairs/trials and nonnegative nulls required')
    root=Path(__file__).resolve().parents[1];out=root/('qc_reports/readout_comparison_validation' if args.validation else 'qc_reports/readout_comparison');out.mkdir(parents=True,exist_ok=True)
    pattern_seed_base=80000 if args.validation else 40000
    evaluation_seed_base=90000 if args.validation else 60000
    previous=json.loads((root/'qc_reports/associative_readout/manifest.json').read_text());theta=previous['baseline_theta']
    for name,h in previous['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Baseline changed')
    pi,ki,w,inh,drive,info=base.load_circuit(root);n=len(inh);npn=info['pn'];pn=sparse.csr_matrix((w,(ki,pi)),shape=(n,npn))
    pre,post,count,totals,main,output_info=learning.load_output(root);outputs=[];nulls=[]
    for graph in range(args.nulls+1):
        if graph:
            target,detail=learning.robust.strength_preserving_null(pre,post,count,np.random.default_rng(50000+graph));detail['seed']=50000+graph;nulls.append(detail)
        else:target=post
        outputs.append(sparse.csr_matrix((count/totals[target],(target,pre)),shape=(len(totals),n)).toarray()[main:main+1])
    gates={noise:presence.presence_threshold(noise,npn,61000+i) for i,noise in enumerate((.1,.4))}
    rows=[]
    for pair in range(args.pairs):
        for ni,noise in enumerate((.1,.4)):
            pattern_seed=pattern_seed_base+pair*1000+200+ni;x,y,_=base.patterns(npn,.9,noise,pattern_seed,80)
            train=np.zeros(len(y),dtype=bool)
            for c in (0,1):train[np.flatnonzero(y==c)[:40]]=True
            ktrain,train_residual=base.response(pn@x[:,train],inh,drive,theta,1);yt=y[train]
            trained=[]
            for graph,before in enumerate(outputs):
                for gate in (0.,1.):
                    after=learning.learn(before,ktrain[:,yt==0].mean(axis=1),20,gate=gate)
                    thresholds={mode:learning.decision_threshold(s,yt) for mode,s in read_scores(before,after,ktrain).items()}
                    trained.append((graph,gate,before,after,thresholds))
            labels=np.repeat([0,1],args.trials);bases=presence.pattern_bases(npn,.9,pattern_seed)
            eval_seed=evaluation_seed_base+pair*100+ni;z=np.random.default_rng(eval_seed).normal(0,noise,(npn,len(labels)))
            for amplitude in (0.,.1,.25,.5,1.):
                x=np.maximum(0,amplitude*bases[:,labels]+z);has_signal=x.sum(axis=0)>gates[noise]
                for test_apl in (1.,0.):
                    k,residual=base.response(pn@x,inh,drive,theta,test_apl)
                    for graph,gate,before,after,thresholds in trained:
                        for mode,scores in read_scores(before,after,k).items():
                            t=thresholds[mode];assigned=scores>t;correct=assigned==(labels==0)
                            # A rejection is not silently counted as a correct B decision.
                            gated_correct=correct&has_signal
                            for c in (0,1):
                                selected=labels==c
                                rows.append(dict(graph=graph,pair=pair,noise=noise,amplitude=amplitude,test_apl=test_apl,
                                    learning_gate=gate,readout=mode,source_pattern='A' if c==0 else 'B',
                                    naive_assigned_A=float(assigned[selected].mean()),gated_assigned_A=float((assigned&has_signal)[selected].mean()),
                                    signal_detected_fraction=float(has_signal[selected].mean()),
                                    accuracy=float(correct.mean()) if amplitude>0 else np.nan,
                                    accuracy_rejections_wrong=float(gated_correct.mean()) if amplitude>0 else np.nan,
                                    auc_A=learning.auc(scores,labels) if amplitude>0 else np.nan,
                                    threshold=t,exact_zero_assigned_A=float(0>t),residual=residual,train_residual=train_residual,
                                    pattern_seed=pattern_seed,evaluation_seed=eval_seed))
        print(f'pair {pair+1}/{args.pairs} complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    cols=['naive_assigned_A','gated_assigned_A','signal_detected_fraction','accuracy','accuracy_rejections_wrong','auc_A']
    summary=f.groupby(['graph','noise','amplitude','test_apl','learning_gate','readout','source_pattern'])[cols].agg(['mean','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    manifest=dict(data_sha256=previous['data_sha256'],pairs=args.pairs,trials_per_pattern=args.trials,rows=len(f),
        fresh_pattern_validation=args.validation,pattern_seed_base=pattern_seed_base,evaluation_seed_base=evaluation_seed_base,
        circuit=info,main_mbon=output_info['main_id'],theta=theta,eta=20,train_apl=1,train_amplitude=1,
        score_definitions={'raw':'-W_after dot k','activity_normalized':'-(W_after dot k)/mean(k)','counterfactual_fraction':'((W_before-W_after) dot k)/(W_before dot k)'},
        denominator_zero='score set to 0 when reference <=1e-12',
        counterfactual='Requires unlearned weights evaluated on the SAME current input; not a biological readout.',
        thresholds='Fit on original amplitude-1 training trials separately for each readout; fixed at evaluation.',
        presence='Same independent noise-only PN sum calibration as 09, 95th percentile; identical evaluation noise reused.',
        nulls=nulls,max_residual=float(f.residual.max()),max_train_residual=float(f.train_residual.max()),
        interpretation='Synthetic readout comparison; no signal identity at amplitude 0, no A/B accuracy/AUC assigned there.',
        limitations=['Readouts have different available information; counterfactual uses privileged reference.',
                     'Population normalization also adds information unavailable to a lone MBON.',
                     'No biological implementation, measured odor/behavior or disease inference.',
                     'Output nulls and eight patterns are exploratory, not population samples.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    sub=f[(f.graph==0)&(f.learning_gate==1)&(f.noise==.4)&(f.test_apl==1)]
    for mode,color in zip(('raw','activity_normalized','counterfactual_fraction'),('#777777','#0072B2','#D55E00')):
        s=sub[(sub.readout==mode)&(sub.source_pattern=='A')].groupby('amplitude').mean(numeric_only=True)
        axes[0].plot(s.index,s.gated_assigned_A,'o-',color=color,label=mode)
        s=sub[(sub.readout==mode)&(sub.source_pattern=='B')].groupby('amplitude').mean(numeric_only=True)
        axes[1].plot(s.index,s.gated_assigned_A,'o-',color=color,label=mode)
        axes[2].plot(s.index,s.accuracy_rejections_wrong,'o-',color=color,label=mode)
    for ax,title in zip(axes,('A recognized with presence gate','B assigned A with presence gate','Identification accuracy (reject = wrong)')):
        ax.set_title(title);ax.set_xlabel('Input amplitude');ax.set_ylim(0,1.02);ax.legend(fontsize=7)
    fig.suptitle('Same learned circuit / different readouts; high overlap, noise .4, normal APL')
    fig.tight_layout();fig.savefig(out/'readout_comparison.png',dpi=180)
    print('rows',len(f),'max residual',manifest['max_residual'])


if __name__=='__main__':main()
