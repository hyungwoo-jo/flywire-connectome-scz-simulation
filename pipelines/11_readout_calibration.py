#!/usr/bin/env python3
"""Disentangle threshold calibration from scalar-score separation."""
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
base=compare.base;learning=compare.learning;presence=compare.presence


def fit_calibration(scores,labels,feature,context=False):
    fallback=learning.decision_threshold(scores,labels)
    if not context:return dict(cuts=[],thresholds=[fallback])
    cuts=np.quantile(feature,[1/3,2/3]);bins=np.searchsorted(cuts,feature,side='right');thresholds=[]
    for i in range(3):
        selected=bins==i
        thresholds.append(learning.decision_threshold(scores[selected],labels[selected])
                          if np.unique(labels[selected]).size==2 else fallback)
    return dict(cuts=cuts.tolist(),thresholds=thresholds)


def margins(scores,feature,calibration):
    bins=np.searchsorted(calibration['cuts'],feature,side='right')
    return scores-np.asarray(calibration['thresholds'])[bins]


def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=int,default=8);p.add_argument('--trials',type=int,default=100);args=p.parse_args()
    if args.pairs<1 or args.trials<2:p.error('positive pairs and at least two trials required')
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/readout_calibration';out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'qc_reports/associative_readout/manifest.json').read_text());theta=previous['baseline_theta']
    for name,h in previous['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed')
    pre,post,w,inh,drive,info=base.load_circuit(root);n=len(inh);npn=info['pn'];pn=sparse.csr_matrix((w,(post,pre)),shape=(n,npn))
    kp,mp,count,totals,main,output_info=learning.load_output(root)
    before=sparse.csr_matrix((count/totals[mp],(mp,kp)),shape=(len(totals),n)).toarray()[main:main+1]
    gates={noise:presence.presence_threshold(noise,npn,61000+i) for i,noise in enumerate((.1,.4))}
    rows=[];calibrations=[];maximum_residual=0
    for pair in range(args.pairs):
        for ni,noise in enumerate((.1,.4)):
            pattern_seed=100000+pair*1000+200+ni; x,y,_=base.patterns(npn,.9,noise,pattern_seed,80)
            chosen=np.zeros(len(y),dtype=bool)
            for c in (0,1):chosen[np.flatnonzero(y==c)[:40]]=True
            ktrain,residual=base.response(pn@x[:,chosen],inh,drive,theta,1);maximum_residual=max(maximum_residual,residual)
            bases=presence.pattern_bases(npn,.9,pattern_seed)
            # Single and mixed intensity calibration use equal sample counts and the SAME noise draws.
            cal_labels=np.tile(np.repeat([0,1],40),4);amplitudes=np.repeat([.1,.25,.5,1.],80)
            cal_seed=120000+pair*100+ni;z=np.random.default_rng(cal_seed).normal(0,noise,(npn,len(cal_labels)))
            x_single=np.maximum(0,bases[:,cal_labels]+z);x_mixed=np.maximum(0,bases[:,cal_labels]*amplitudes+z)
            ksingle,r1=base.response(pn@x_single,inh,drive,theta,1);kmixed,r2=base.response(pn@x_mixed,inh,drive,theta,1);maximum_residual=max(maximum_residual,r1,r2)
            fits=[]
            for gate in (0.,1.):
                after=learning.learn(before,ktrain[:,y[chosen]==0].mean(axis=1),20,gate=gate)
                singles=compare.read_scores(before,after,ksingle);mixed=compare.read_scores(before,after,kmixed)
                for mode in singles:
                    policies={
                        'single_intensity':fit_calibration(singles[mode],cal_labels,x_single.sum(axis=0)),
                        'mixed_global':fit_calibration(mixed[mode],cal_labels,x_mixed.sum(axis=0)),
                        'mixed_context':fit_calibration(mixed[mode],cal_labels,x_mixed.sum(axis=0),context=True)}
                    for policy,fit in policies.items():
                        fits.append((gate,mode,policy,before,after,fit))
                        calibrations.append(dict(pair=pair,noise=noise,learning_gate=gate,readout=mode,policy=policy,**fit))
            labels=np.repeat([0,1],args.trials);eval_seed=140000+pair*100+ni;z=np.random.default_rng(eval_seed).normal(0,noise,(npn,len(labels)))
            for amplitude in (0.,.1,.25,.5,1.):
                x=np.maximum(0,bases[:,labels]*amplitude+z);feature=x.sum(axis=0);present=feature>gates[noise]
                k,residual=base.response(pn@x,inh,drive,theta,1);maximum_residual=max(maximum_residual,residual)
                cached={gate:compare.read_scores(before,learning.learn(before,ktrain[:,y[chosen]==0].mean(axis=1),20,gate=gate),k) for gate in (0.,1.)}
                for gate,mode,policy,_,_,fit in fits:
                    scores=cached[gate][mode];margin=margins(scores,feature,fit);assigned=margin>0;correct=assigned==(labels==0)
                    rows.append(dict(pair=pair,noise=noise,amplitude=amplitude,learning_gate=gate,readout=mode,policy=policy,
                        accuracy=float(correct.mean()) if amplitude>0 else np.nan,
                        accuracy_rejections_wrong=float(np.mean(correct&present)) if amplitude>0 else np.nan,
                        raw_score_auc=learning.auc(scores,labels) if amplitude>0 else np.nan,
                        margin_auc=learning.auc(margin,labels) if amplitude>0 else np.nan,
                        signal_detected_fraction=float(present.mean()),
                        A_assigned_A=float(np.mean(assigned[labels==0]&present[labels==0])) if amplitude>0 else np.nan,
                        B_assigned_A=float(np.mean(assigned[labels==1]&present[labels==1])) if amplitude>0 else np.nan,
                        blank_assigned_A=float(np.mean(assigned&present)) if amplitude==0 else np.nan,
                        pattern_seed=pattern_seed,calibration_seed=cal_seed,evaluation_seed=eval_seed,residual=residual))
        print(f'pair {pair+1}/{args.pairs} complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    metrics=['accuracy','accuracy_rejections_wrong','raw_score_auc','margin_auc','signal_detected_fraction','A_assigned_A','B_assigned_A','blank_assigned_A']
    s=f.groupby(['noise','amplitude','learning_gate','readout','policy'])[metrics].agg(['mean','min','max']);s.columns=['_'.join(c) for c in s.columns];s.reset_index().to_csv(out/'summary.csv',index=False)
    (out/'calibrations.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    manifest=dict(data_sha256=previous['data_sha256'],pairs=args.pairs,trials_per_pattern=args.trials,rows=len(f),main_mbon=output_info['main_id'],theta=theta,eta=20,
        calibration_trials_per_class=160,learning_trials_per_class=40,calibration_amplitudes=[.1,.25,.5,1],test_amplitudes=[0,.1,.25,.5,1],
        calibration_design='Equal calibration sample counts; single and mixed use identical noise draws; learning/calibration/evaluation independent noise.',
        context='Three quantile bins of current PN sum, fitted on calibration only; no true test amplitude supplied.',
        threshold_objective='Balanced A/B accuracy on calibration; not blank rejection or gated accuracy.',
        noise='0.1/0.4, known independent presence threshold from 09',max_residual=maximum_residual,
        pattern_seed_base=100000,calibration_seed_base=120000,evaluation_seed_base=140000,
        limitations=['External labeled calibration, not a neural learning rule.',
                     'Context policy adds current PN information; margin AUC may change.',
                     'No new output rewiring nulls in this focused calibration assay.',
                     'Counterfactual fraction needs privileged unlearned weights.',
                     'Eight synthetic pattern pairs, one specimen and main MBON.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    sub=f[(f.noise==.4)&(f.learning_gate==1)&(f.readout=='counterfactual_fraction')]
    for policy,color in zip(('single_intensity','mixed_global','mixed_context'),('#777777','#0072B2','#D55E00')):
        t=sub.groupby('policy').get_group(policy).groupby('amplitude').mean(numeric_only=True)
        axes[0].plot(t.index,t.accuracy_rejections_wrong,'o-',color=color,label=policy)
        axes[1].plot(t.index,t.A_assigned_A,'o-',color=color,label=policy)
        axes[2].plot(t.index,t.B_assigned_A,'o-',color=color,label=policy)
    for ax,title in zip(axes,('Identification accuracy (reject = wrong)','A recognized','B assigned A')):
        ax.set_title(title);ax.set_xlabel('Input amplitude');ax.set_ylim(0,1.02);ax.legend(fontsize=7)
    fig.suptitle('Fresh patterns; fixed learned weights, different calibration policies; noise .4')
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=180)
    print('rows',len(f),'max residual',maximum_residual)


if __name__=='__main__':main()
