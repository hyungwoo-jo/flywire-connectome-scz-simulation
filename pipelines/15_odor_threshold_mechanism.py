#!/usr/bin/env python3
"""Separate threshold, APL feedback and gain effects in the observed subset."""
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

spec=importlib.util.spec_from_file_location('odor',Path(__file__).with_name('14_observed_odor_geometry.py'))
odor=importlib.util.module_from_spec(spec);spec.loader.exec_module(odor)
base=odor.base


def trial_scores(x, labels, train):
    classes=np.unique(labels)
    if not np.array_equal(classes,np.arange(len(classes))):raise ValueError('Labels must be contiguous')
    means=np.stack([x[:,train & (labels==c)].mean(axis=1) for c in classes])
    test=x[:,~train].T
    distances=np.maximum((test*test).sum(axis=1)[:,None]+(means*means).sum(axis=1)[None,:]-2*test@means.T,0)
    ties=np.isclose(distances,distances.min(axis=1)[:,None],rtol=1e-10,atol=1e-12)
    scores=ties[np.arange(len(test)),labels[~train]]/ties.sum(axis=1)
    silent=np.max(test,axis=1)<=1e-8
    return scores,silent


def error_components(scores,silent):
    return dict(accuracy=float(scores.mean()),test_silent_fraction=float(silent.mean()),
        silent_error_mass=float(np.mean((1-scores)*silent)),active_error_mass=float(np.mean((1-scores)*(~silent))),
        accuracy_if_active=float(scores[~silent].mean()) if (~silent).any() else None)


def zero_error_bound(zero, labels):
    """Oracle lower error bound when all zero-vector trials share one output."""
    if not zero.any():return 0.
    counts=np.bincount(labels[zero])
    return float((zero.sum()-counts.max())/len(labels))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/odor_threshold_mechanism';out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((root/'qc_reports/observed_odor_geometry/manifest.json').read_text())
    for item in prior['door_source']['files']:
        if hashlib.sha256((root/'data/door'/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('DoOR input changed')
    for name,h in prior['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Circuit input changed')
    channels=prior['channels'];keys=[o['odor_key'] for o in prior['odors']]
    data=pd.read_csv(root/'data/door/door_response_matrix.csv',sep=';',index_col=0)
    values=data.loc[keys,channels].to_numpy(float).T;sfr=data.loc['SFR',channels].to_numpy(float)
    mapping=pd.read_csv(root/'qc_reports/odor_input_audit/pn_mapping_audit.csv',dtype={'root_888':str})
    selected=mapping[(mapping.status=='mapped')&mapping.response_channel.isin(channels)].set_index('root_888')
    meta=pd.read_feather(root/'data/banc_888_meta.feather');edges=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].root_888.astype(str)
    pn=meta[(meta.cell_class=='antennal_lobe_projection_neuron')&(meta.neurotransmitter_verified=='acetylcholine')].root_888.astype(str)
    pn_ids=sorted(edges[edges.pre.isin(pn)&edges.post.isin(kc)].pre.unique())
    chosen=[i for i,p in enumerate(pn_ids) if p in selected.index]
    channel_index=np.array([channels.index(selected.loc[pn_ids[i],'response_channel']) for i in chosen])
    pi,ki,w,inh,drive,info=base.load_circuit(root);mask=np.isin(pi,chosen)
    matrix=sparse.csr_matrix((w[mask],(ki[mask],pi[mask])),shape=(len(inh),info['pn']))[:,chosen]
    labels=np.repeat(np.arange(18),30);train=np.tile(np.arange(30)<10,18)
    centers=odor.transfer(values,sfr,'positive_delta');rows=[];per_odor=[];residual=0.
    for repeat in range(4):
        for ni,noise in enumerate((.02,.1)):
            seed=210000+repeat*1000+ni
            x=np.clip(centers[:,labels]+np.random.default_rng(seed).normal(0,noise,(20,len(labels))),0,1)[channel_index]
            blank_seed=220000+repeat*1000+ni
            blank=np.clip(np.random.default_rng(blank_seed).normal(0,noise,(20,200)),0,1)[channel_index]
            excitation=matrix@x;blank_excitation=matrix@blank
            for gain in (1.,4.):
                for factor in (0.,.25,.5,1.):
                    for apl in (0.,1.):
                        theta=prior['theta']*factor
                        k,res=base.response(gain*excitation,inh,drive,theta,apl);residual=max(residual,res)
                        kb,res=base.response(gain*blank_excitation,inh,drive,theta,apl);residual=max(residual,res)
                        scores,silent=trial_scores(k,labels,train)
                        exact_zero=np.max(k[:,~train],axis=0)==0
                        bound=zero_error_bound(exact_zero,labels[~train])
                        settings=dict(noise_repeat=repeat,noise=noise,input_seed=seed,blank_seed=blank_seed,input_gain=gain,theta_factor=factor,apl_strength=apl)
                        rows.append(dict(**settings,**error_components(scores,silent),zero_vector_error_lower_bound=bound,exact_zero_test_fraction=float(exact_zero.mean()),active_fraction=float(np.mean(k>1e-8)),
                            saturated_fraction=float(np.mean(k>=1-1e-8)),noise_only_response_fraction=float(np.mean(np.max(kb,axis=0)>1e-8)),
                            noise_only_active_fraction=float(np.mean(kb>1e-8)),noise_only_mean_activity=float(kb.mean())))
                        testlabels=labels[~train]
                        for c in range(18):
                            use=testlabels==c
                            per_odor.append(dict(**settings,odor_key=keys[c],name=prior['odors'][c]['name'],**error_components(scores[use],silent[use])))
        print('noise repetition',repeat+1,'complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    pf=pd.DataFrame(per_odor);pf.to_csv(out/'per_odor.csv',index=False)
    cols=['zero_vector_error_lower_bound','exact_zero_test_fraction','accuracy','test_silent_fraction','silent_error_mass','active_error_mass','active_fraction','saturated_fraction',
          'noise_only_response_fraction','noise_only_active_fraction','noise_only_mean_activity']
    means=f.groupby(['noise','input_gain','theta_factor','apl_strength'])[cols].mean().reset_index();means.to_csv(out/'means.csv',index=False)
    pf.groupby(['noise','input_gain','theta_factor','apl_strength','odor_key','name'])[['accuracy','test_silent_fraction','silent_error_mass','active_error_mass']].mean().reset_index().to_csv(out/'per_odor_means.csv',index=False)
    # Verify unchanged conditions reproduce the previous experiment exactly.
    old=pd.read_csv(root/'qc_reports/observed_odor_geometry/metrics.csv')
    old=old[(old.graph==0)&(old.transfer=='positive_delta')&old.input_gain.isin([1,4])]
    anchor=f[(f.theta_factor==1)&(f.apl_strength==1)].merge(old,on=['noise_repeat','noise','input_gain'],suffixes=('_new','_old'))
    assert len(anchor)==16
    max_difference=float(np.max(np.abs(anchor.accuracy_new-anchor.accuracy_old)))
    if max_difference>1e-12:raise ValueError('Prior anchor changed')
    manifest=dict(source_manifest='qc_reports/observed_odor_geometry/manifest.json',door_source=prior['door_source'],data_sha256=prior['data_sha256'],
        channels=channels,odors=prior['odors'],selected_pns=len(chosen),observed_edges=int(mask.sum()),rows=len(f),per_odor_rows=len(pf),
        theta=prior['theta'],theta_factors=[0,.25,.5,1],input_gains=[1,4],apl_strengths=[0,1],apl_gain=4,noise=[.02,.1],repeats=4,
        blank_trials_per_condition=200,max_residual=residual,prior_anchor_max_accuracy_difference=max_difference,
        transfer='positive_delta only: clip(OSN consensus-SFR,0,infinity) then additive synthetic channel noise clipped to [0,1].',
        classifier='Within-condition train10/test20 per odor, 18-way nearest Euclidean centroid, uniform expected tie score.',
        decomposition='1-accuracy = silent_error_mass + active_error_mass; masses normalized to all test trials.',
        zero_bound='Oracle min errors from exact zero vectors = (zero trials minus largest odor count among them) / all test trials. Uses test labels only for a descriptive bound, never prediction or calibration.',
        blank='Positive-delta no-odor baseline is zero; rectified Gaussian noise only, same draws across gain/theta/APL. Exact zero also remains structurally zero for all tested settings.',
        limitations=['Observed-channel partial circuit; direct OSN to PN transfer, synthetic noise, one specimen.',
            'No biological calibration, learned MBON, molecular mechanisms or disease conclusions.',
            'No rewired graphs in this focused mechanism assay; cannot infer detailed wiring specificity.',
            'Noise-only response means any nonzero KC, not percept, presence decision, false identification or hallucination.',
            'Classifier retrained in each setting; does not model an organism adapting its decision instantly.',
            'No independent biological replicates or inferential statistics.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for row,noise in enumerate((.02,.1)):
        for gain in (1,4):
            for apl in (0,1):
                g=means[(means.noise==noise)&(means.input_gain==gain)&(means.apl_strength==apl)].sort_values('theta_factor')
                for col,metric in enumerate(('accuracy','noise_only_response_fraction')):
                    axes[row,col].plot(g.theta_factor,100*g[metric],marker='o',linestyle='-' if apl else '--',label=f'gain={gain}, APL={apl}')
        for col in (0,1):
            axes[row,col].set_title(f'noise={noise}; '+('18-odor accuracy' if col==0 else 'Any KC response to noise only'))
            axes[row,col].set_ylim(-2,102);axes[row,col].set_xlabel('KC threshold / original threshold');axes[row,col].set_ylabel('%');axes[row,col].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'threshold_mechanism.png',dpi=180)
    print('rows',len(f),'per odor',len(pf),'prior anchor difference',max_difference,'max residual',residual)

if __name__=='__main__':main()
