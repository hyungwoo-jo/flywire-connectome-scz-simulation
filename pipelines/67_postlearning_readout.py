#!/usr/bin/env python3
"""Information-access audit: post-learning MBON outputs versus privileged ratios."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats

spec=importlib.util.spec_from_file_location('p65',Path(__file__).with_name('65_kc_imaging_chan2026.py'))
p65=importlib.util.module_from_spec(spec);sys.modules['p65']=p65;spec.loader.exec_module(p65)
p55=p65.p55;p25=p55.p25;cal=p65.cal
SEEDS=dict(train=671000,test=672000,noise_cal=673000,noise_test=674000,control=675000)


def fit_threshold(scores,is_a):
    values=np.unique(scores)
    if not is_a.any() or is_a.all():raise ValueError('Two classes required')
    candidates=np.r_[np.nextafter(values[0],-np.inf),(values[:-1]+values[1:])/2,np.nextafter(values[-1],np.inf)]
    predicted=scores[:,None]>candidates[None,:]
    accuracy=.5*(predicted[is_a].mean(0)+(~predicted[~is_a]).mean(0))
    best=np.flatnonzero(np.isclose(accuracy,accuracy.max(),rtol=0,atol=1e-14))
    return float(candidates[best[len(best)//2]])


def actual_scores(output,pn_total):
    return dict(raw_after=-output,pn_normalized_after=-output/np.maximum(pn_total,1e-12))


def privileged_score(before_output,after_output):
    change=before_output-after_output
    change=np.where(np.isclose(before_output,after_output,rtol=1e-12,atol=1e-15),0.,change)
    return np.divide(change,before_output,out=np.zeros_like(after_output),where=before_output>1e-12)


def matched_profile(profile,weights,types,rng):
    permuted=profile.copy();eligible=np.zeros(len(weights),bool)
    for subtype in sorted(set(types)):
        for weight in np.unique(weights[(types==subtype)&(weights>0)]):
            idx=np.flatnonzero((types==subtype)&(weights==weight))
            if len(idx)>1:permuted[idx]=rng.permutation(profile[idx]);eligible[idx]=True
    expected=weights@np.exp(-p25.ETA*profile);obtained=weights@np.exp(-p25.ETA*permuted)
    np.testing.assert_allclose(obtained,expected,rtol=1e-12,atol=1e-12)
    return permuted,float(weights[eligible].sum()/weights.sum())


def auc(pos,neg):
    values=np.r_[pos,neg];ranks=stats.rankdata(values)
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def prepare_inputs(c,receptors,delta,sfr,keep):
    d=delta[:,keep];n=len(keep);result={}
    for phase,trials,key in [('train',10,'train'),('test',20,'test')]:
        labels=np.repeat(np.arange(n),trials)
        for amplitude in (1.,.1):
            # Draw order remains fixed and is shared across graph/learning conditions.
            seed=SEEDS[key]+(0 if amplitude==1 else 1)
            x=cal.pn_drive(c,receptors,cal.sample_orn(amplitude*d,sfr,labels,.5,np.random.default_rng(seed)))
            result[(phase,amplitude)]=(x,labels)
    for phase,key in [('noise_cal','noise_cal'),('noise_test','noise_test')]:
        x=cal.pn_drive(c,receptors,cal.sample_orn(np.zeros((len(receptors),1)),sfr,np.zeros(1000,int),.5,np.random.default_rng(SEEDS[key])))
        result[(phase,0.)]=(x,np.zeros(1000,int))
    return result


def plot_summary(frame,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=[('raw_after','mbon_gate'),('raw_after','pn_gate'),('pn_normalized_after','pn_gate'),('privileged_fraction','pn_gate')]
    labels=['MBON only', 'MBON + PN gate', 'MBON/PN + PN gate', 'Privileged + PN gate']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for side,color in [('right','#D55E00'),('left','#0072B2')]:
        for learning,marker in [('learned_A','o'),('matched_shuffle','x')]:
            for col,(amplitude,metric) in enumerate([(1.,'AB_accuracy'),(.1,'AB_accuracy'),(.1,'noise_to_A')]):
                values=[]
                for readout,gate in methods:
                    q=frame[(frame.side==side)&(frame.condition=='original')&(frame.learning==learning)&(frame.readout==readout)&(frame.gate==gate)&(frame.amplitude==amplitude)]
                    values.append(float(q[metric].mean())*100)
                axes[col].plot(range(4),values,marker=marker,color=color,linestyle='-' if learning=='learned_A' else ':',label=f'{side}, {learning}')
    for ax,title in zip(axes,['Full-strength A/B accuracy','Weak A/B accuracy; rejections count wrong','Noise assigned to A']):
        ax.set_title(title);ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_ylabel('%');ax.legend(fontsize=7)
    axes[0].set_ylim(0,102);axes[1].set_ylim(0,102);axes[2].set_ylim(0,6)
    fig.suptitle('Post-learning readout audit; original circuit, one specimen, 73 learned odors')
    fig.tight_layout();fig.savefig(out/'postlearning_readout.png',dpi=180);plt.close(fig)


def fixed_criterion_summary(summary,out):
    keys=['side','condition','readout','gate','amplitude']
    learned=summary[summary.learning=='learned_A'].set_index(keys)
    control=summary[summary.learning=='matched_shuffle'].set_index(keys)
    result=pd.DataFrame(index=learned.index)
    result['noise_A_learned_minus_control_adaptive']=learned.noise_to_A-control.noise_to_A
    result['noise_A_learned_minus_control_fixed']=learned.fixed_noise_to_A-control.fixed_noise_to_A
    result['AB_accuracy_learned_minus_control_adaptive']=learned.AB_accuracy-control.AB_accuracy
    result['AB_accuracy_learned_minus_control_fixed']=learned.fixed_AB_accuracy-control.fixed_AB_accuracy
    result.reset_index().to_csv(out/'fixed_criterion_contrasts.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,side in zip(axes,['right','left']):
        g=summary[(summary.side==side)&(summary.condition=='original')&(summary.readout=='raw_after')&(summary.gate=='mbon_gate')&(summary.amplitude==1)].set_index('learning')
        order=['learned_A','matched_shuffle','none'];g=g.loc[order]
        ax.plot(range(3),100*g.noise_to_A,'-o',label='Recalibrated per condition')
        ax.plot(range(3),100*g.fixed_noise_to_A,'-s',label='Same learned-A criteria')
        ax.set_xticks(range(3),order,rotation=15);ax.set_ylabel('Noise assigned to A (%)');ax.set_title(side);ax.legend(fontsize=8)
    fig.suptitle('Fixed-criterion intervention; control readouts are not re-optimized')
    fig.tight_layout();fig.savefig(out/'fixed_criterion.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation',action='store_true');args=parser.parse_args()
    if args.validation:
        for key in ('train','test','noise_cal','noise_test'):SEEDS[key]+=10000
    root=Path(__file__).resolve().parents[1];out=root/('qc_reports/postlearning_readout_validation' if args.validation else 'qc_reports/postlearning_readout');out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    for name,digest in manifest['source']['receptor_csv_sha256'].items():
        if hashlib.sha256((root/'data/door/receptors'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Receptor data changed')
    hashes=json.loads((root/'qc_reports/input_discrimination/manifest.json').read_text())['data_sha256']
    for name,digest in hashes.items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Connectome changed')
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T
    calib=cal.split_odors(list(odors.index));keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=p25.p24.disc.MIN_PN_HZ))
    n=len(keep);meta=pd.read_feather(root/'data/banc_888_meta.feather');rows=[];controls=[];anchors=[]
    for side in ('right','left'):
        g=p25.p24.setup(root,side);c=g['c'];c['kk']=None;m=g['m'];w=p25.mbon_weights(root,side,len(c['inh']))
        if n!=g['n']:raise ValueError('Odor layout changed')
        inputs=prepare_inputs(c,receptors,delta,sfr.to_numpy(),keep)
        kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side==side)].copy();kc['id']=kc.root_888.astype(str);kc=kc.set_index('id').sort_index()
        types=kc.cell_sub_class.fillna(kc.cell_type).fillna('unknown').to_numpy(str)
        for condition in ('original','total_homeo'):
            if condition=='original':theta=g['theta']
            else:
                E=m@cal.pn_drive(c,receptors,delta[:,calib]);p65.p63._D['m']=m
                theta=p65.p63.thresholds(condition,E,np.asarray(m@g['x_cal']),c,g['theta'])[:,None]
            clean=p55.respond(m@g['x_full'],c,theta)
            K={key:p55.respond(m@value[0],c,theta) for key,value in inputs.items()}
            pn={key:value[0].sum(0) for key,value in inputs.items()}
            before={key:w@value for key,value in K.items()}
            sums=clean.sum(0);distance=np.abs(np.log(np.maximum(sums,1e-12))[:,None]-np.log(np.maximum(sums,1e-12))[None,:]);np.fill_diagonal(distance,np.inf)
            partners=distance.argmin(1);pn_gate=float(np.quantile(pn[('noise_cal',0.)],.95))
            profiles={'learned_A':[],'matched_shuffle':[],'none':[]}
            for a in range(n):
                shuffled,eligible=matched_profile(clean[:,a],w,types,np.random.default_rng(SEEDS['control']+a))
                profiles['learned_A'].append(clean[:,a]);profiles['matched_shuffle'].append(shuffled);profiles['none'].append(np.zeros(len(w)))
                controls.append(dict(side=side,condition=condition,odor=a,partner=int(partners[a]),input_activity_ratio=float(sums[a]/max(sums[partners[a]],1e-12)),
                    eligible_weight_fraction=eligible,weight_drop=float(w@(1-np.exp(-p25.ETA*clean[:,a])))))
            frozen={}
            for learning_kind,profile in profiles.items():
                after_weights=np.exp(-p25.ETA*np.stack(profile))*w[None,:]
                outputs={key:after_weights@value for key,value in K.items()}
                for a in range(n):
                    b=int(partners[a]);train_parts=[];class_parts=[]
                    scores_by_phase={}
                    for key in K:
                        scores=actual_scores(outputs[key][a],pn[key]);scores['privileged_fraction']=privileged_score(before[key],outputs[key][a]);scores_by_phase[key]=scores
                    for readout in ('raw_after','pn_normalized_after','privileged_fraction'):
                        for amp in (1.,.1):
                            lab=inputs[('train',amp)][1];sel=(lab==a)|(lab==b)
                            train_parts.append(scores_by_phase[('train',amp)][readout][sel]);class_parts.append(lab[sel]==a)
                        threshold=fit_threshold(np.concatenate(train_parts),np.concatenate(class_parts));train_parts=[];class_parts=[]
                        noise_score=scores_by_phase[('noise_test',0.)][readout]
                        mbon_gate=float(np.quantile(outputs[('noise_cal',0.)][a],.95))
                        if learning_kind=='learned_A':frozen[(a,readout)]=(threshold,mbon_gate)
                        for amp in (1.,.1):
                            key=('test',amp);labels=inputs[key][1];score=scores_by_phase[key][readout];is_a=labels==a;is_b=labels==b
                            predicts_a=score>threshold;noise_predicts_a=noise_score>threshold
                            for gate in ('none','mbon_gate','pn_gate'):
                                if gate=='none':signal_detect=np.ones(len(score),bool);noise_detect=np.ones(len(noise_score),bool)
                                elif gate=='mbon_gate':signal_detect=outputs[key][a]>mbon_gate;noise_detect=outputs[('noise_test',0.)][a]>mbon_gate
                                else:signal_detect=pn[key]>pn_gate;noise_detect=pn[('noise_test',0.)]>pn_gate
                                fixed_cut,fixed_gate=frozen[(a,readout)]
                                if gate=='mbon_gate':
                                    fixed_detect=outputs[key][a]>fixed_gate;fixed_noise_detect=outputs[('noise_test',0.)][a]>fixed_gate
                                else:fixed_detect=signal_detect;fixed_noise_detect=noise_detect
                                fixed_a=score>fixed_cut;fixed_noise_a=noise_score>fixed_cut
                                rows.append(dict(side=side,condition=condition,learning=learning_kind,odor=a,partner=b,readout=readout,gate=gate,amplitude=amp,
                                    A_hit=float(np.mean((predicts_a&signal_detect)[is_a])),B_to_A=float(np.mean((predicts_a&signal_detect)[is_b])),
                                    AB_accuracy=.5*(float(np.mean((predicts_a&signal_detect)[is_a]))+float(np.mean((~predicts_a&signal_detect)[is_b]))),
                                    A_detect=float(signal_detect[is_a].mean()),B_detect=float(signal_detect[is_b].mean()),
                                    other_to_A=float(np.mean((predicts_a&signal_detect)[~is_a])),noise_to_A=float(np.mean(noise_predicts_a&noise_detect)),
                                    noise_detect=float(noise_detect.mean()),
                                    fixed_A_hit=float(np.mean((fixed_a&fixed_detect)[is_a])),fixed_B_to_A=float(np.mean((fixed_a&fixed_detect)[is_b])),
                                    fixed_AB_accuracy=.5*(float(np.mean((fixed_a&fixed_detect)[is_a]))+float(np.mean((~fixed_a&fixed_detect)[is_b]))),
                                    fixed_noise_to_A=float(np.mean(fixed_noise_a&fixed_noise_detect)),fixed_noise_detect=float(fixed_noise_detect.mean()),A_B_auc=auc(score[is_a],score[is_b]),threshold=threshold))
                print(side,condition,learning_kind,'complete',flush=True)
            anchors.append(dict(side=side,condition=condition,n_odors=n,clean_active_fraction=float(np.mean(clean>1e-8)),pn_gate=pn_gate))
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False);pd.DataFrame(controls).to_csv(out/'controls.csv',index=False)
    groups=['side','condition','learning','readout','gate','amplitude'];metrics=['A_hit','B_to_A','AB_accuracy','A_detect','B_detect','other_to_A','noise_to_A','noise_detect','A_B_auc','fixed_A_hit','fixed_B_to_A','fixed_AB_accuracy','fixed_noise_to_A','fixed_noise_detect']
    summary=f.groupby(groups)[metrics].mean().reset_index();summary.to_csv(out/'summary.csv',index=False)
    report=dict(date='2026-10-18',plan='docs/POSTLEARNING_READOUT_PLAN.md',seeds=SEEDS,n_odors=n,eta=p25.ETA,signal_rows=len(f),anchors=anchors,data_sha256=hashes,
        fixed_criterion_plan='docs/FIXED_READOUT_CRITERION_PLAN.md',interpretation='Abstract circuit information-access audit only. No solvent input, calcium observation claim or behavior inference.',
        access={'raw_after':'post-learning MBON output only','pn_normalized_after':'post-learning MBON and additional PN total channel','privileged_fraction':'same-input pre-learning output is privileged diagnostic information'},
        limitations=['One animal, two hemispheres; noise repeats and shared odor trials are not biological replicates.',
            'Readout calibrated separately after each learning condition using labeled A/B examples; adaptation is assumed.',
            'Strong and weak amplitudes mixed during calibration; test amplitude is not provided to predictor.',
            'B is matched by total clean KC activity, not all feature distributions.',
            'Same-weight/subtype shuffle is restricted and may leave singleton inputs unchanged.',
            'Noise metrics repeated in both amplitude rows use the same trials, not independent observations.',
            'No wiring-specificity claim, identified downstream circuit or physical fluorescence calibration.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot_summary(summary,out)
    fixed_criterion_summary(summary,out)
    print('signal rows',len(f))

if __name__=='__main__':main()
