#!/usr/bin/env python3
"""Separate signal presence from forced A/B classification; synthetic assay."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('learning',Path(__file__).with_name('08_associative_readout.py'))
learning=importlib.util.module_from_spec(spec);spec.loader.exec_module(learning)
base=learning.base


def presence_threshold(noise,n,seed,samples=2000,quantile=.95):
    x=np.maximum(0,np.random.default_rng(seed).normal(0,noise,(n,samples)))
    return float(np.quantile(x.sum(axis=0),quantile))


def pattern_bases(n,overlap,seed):
    order=np.random.default_rng(seed).permutation(n);size=round(.2*n);shared=round(size*overlap)
    a=np.zeros(n);b=np.zeros(n);a[order[:size]]=1;b[np.r_[order[:shared],order[size:size+size-shared]]]=1
    return np.stack([a,b],axis=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=int,default=8);parser.add_argument('--trials',type=int,default=100);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/signal_presence';out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'qc_reports/associative_readout/manifest.json').read_text());theta=previous['baseline_theta']
    import hashlib
    for name,h in previous['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Baseline changed')
    pre,post,w,inh,drive,circuit=base.load_circuit(root);n=len(inh);npn=circuit['pn'];pn=sparse.csr_matrix((w,(post,pre)),shape=(n,npn))
    kpre,mpost,count,totals,main,info=learning.load_output(root)
    output=sparse.csr_matrix((count/totals[mpost],(mpost,kpre)),shape=(len(totals),n)).toarray()[main:main+1]
    gates={noise:presence_threshold(noise,npn,61000+i) for i,noise in enumerate((.1,.4))};rows=[]
    for pair in range(args.pairs):
        for ni,noise in enumerate((.1,.4)):
            pattern_seed=40000+pair*1000+200+ni
            x,y,_=base.patterns(npn,.9,noise,pattern_seed,80)
            train=np.zeros(len(y),dtype=bool)
            for c in (0,1):train[np.flatnonzero(y==c)[:40]]=True
            k,_=base.response(pn@x[:,train],inh,drive,theta,1)
            trained=learning.learn(output,k[:,y[train]==0].mean(axis=1),20)
            threshold=learning.decision_threshold((-trained@k).ravel(),y[train])
            bases=pattern_bases(npn,.9,pattern_seed);labels=np.repeat([0,1],args.trials)
            eval_seed=60000+pair*100+ni
            z=np.random.default_rng(eval_seed).normal(0,noise,(npn,len(labels)))
            for amplitude in (0.,.1,.25,.5,1.):
                test_input=np.maximum(0,amplitude*bases[:,labels]+z)
                has_signal=test_input.sum(axis=0)>gates[noise]
                for test_apl in (1.,0.):
                    k,residual=base.response(pn@test_input,inh,drive,theta,test_apl)
                    assigned=(-trained@k).ravel()>threshold
                    for c in (0,1):
                        selected=labels==c
                        rows.append(dict(pair=pair,noise=noise,test_apl=test_apl,amplitude=amplitude,source_pattern='A' if c==0 else 'B',
                            naive_assigned_A=float(assigned[selected].mean()),gated_assigned_A=float((assigned&has_signal)[selected].mean()),
                            signal_detected_fraction=float(has_signal[selected].mean()),
                            exact_zero_naive_A=float(0>threshold),exact_zero_gated_A=0.,
                            residual=residual,pattern_seed=pattern_seed,eval_seed=eval_seed,readout_threshold=threshold,pn_presence_threshold=gates[noise]))
        print(f'pair {pair+1}/{args.pairs} complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    summary=f.groupby(['noise','test_apl','amplitude','source_pattern'])[['naive_assigned_A','gated_assigned_A','signal_detected_fraction']].agg(['mean','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    report=dict(data_sha256=previous['data_sha256'],pairs=args.pairs,evaluation_trials_per_pattern=args.trials,
        input_overlap=.9,train_apl=1,eta=20,mbon=info['main_id'],theta=theta,
        presence_criterion='PN activity sum > 95th percentile of 2000 independent noise-only trials',
        presence_thresholds=gates,calibration_seeds=[61000,61001],
        learning='Same patterns and training subset as pipeline 08; new evaluation noise. Readout threshold held fixed.',
        amplitude_comparison='Identical evaluation noise across amplitudes; model trained only at amplitude 1.',
        max_residual=float(f.residual.max()),
        limitations=['External evidence gate, not actual sensory/behavior circuitry.',
                     'Noise distribution and amplitude are assumed, not measured.',
                     'A gate changes the label decision, not the memory or MBON activity.',
                     'No output rewiring nulls in this focused decision-rule assay.',
                     'Noise-only A assignment is a model decision error, not hallucination.'])
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    sub=f[(f.noise==.4)&(f.test_apl==1)]
    for label,color in (('A','#0072B2'),('B','#D55E00')):
        s=sub[sub.source_pattern==label].groupby('amplitude').mean(numeric_only=True)
        axes[0].plot(s.index,s.naive_assigned_A,'o-',color=color,label='Input '+label)
        axes[1].plot(s.index,s.gated_assigned_A,'o-',color=color,label='Input '+label)
        axes[2].plot(s.index,s.signal_detected_fraction,'o-',color=color,label='Input '+label)
    for ax,title in zip(axes,('Naive A assignment','A assignment with presence gate','Signal detected')):
        ax.set_title(title);ax.set_xlabel('Input amplitude');ax.set_ylim(0,1.02);ax.legend()
    fig.suptitle('Independent evaluation noise; high overlap, noise .4, normal test APL; 8-pair means')
    fig.tight_layout();fig.savefig(out/'presence_tradeoff.png',dpi=180)
    print('rows',len(f),'max residual',report['max_residual'])


if __name__=='__main__':main()
