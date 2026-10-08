#!/usr/bin/env python3
"""Explore an explicitly restricted PN-KC circuit with complete DoOR inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('robust',Path(__file__).with_name('07_discrimination_robustness.py'))
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
base=robust.baseline


def transfer(values, sfr, mode):
    if mode=='absolute':return values.copy()
    if mode=='positive_delta':return np.maximum(values-sfr[:,None],0)
    raise ValueError(mode)


def classify(x, labels, train):
    classes=np.unique(labels)
    means=np.stack([x[:,train & (labels==c)].mean(axis=1) for c in classes])
    test=x[:,~train].T
    distance=np.maximum((test*test).sum(axis=1)[:,None]+(means*means).sum(axis=1)[None,:]-2*test@means.T,0)
    ties=np.isclose(distance,distance.min(axis=1)[:,None],rtol=1e-10,atol=1e-12)
    expected=ties[np.arange(len(test)),labels[~train]]/ties.sum(axis=1)
    return dict(accuracy=float(expected.mean()),active_fraction=float(np.mean(x>1e-8)),
        silent_trial_fraction=float(np.mean(np.max(x,axis=0)<=1e-8)),
        tied_test_fraction=float(np.mean(ties.sum(axis=1)>1)),mean_activity=float(x.mean()))


def plot_results(frame,out):
    means=frame.groupby(['transfer','noise','input_gain','graph'])[['accuracy','active_fraction','silent_trial_fraction']].mean().reset_index()
    means.to_csv(out/'graph_means.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for row,mode in enumerate(('absolute','positive_delta')):
        for col,noise in enumerate((.02,.1)):
            ax=axes[row,col];g=means[(means.transfer==mode)&(means.noise==noise)]
            real=g[g.graph==0].sort_values('input_gain')
            null=g[g.graph>0].groupby('input_gain').accuracy.agg(['min','max'])
            ax.fill_between(null.index,null['min']*100,null['max']*100,alpha=.25,label=f'{frame.graph.nunique()-1} null graph means')
            ax.plot(real.input_gain,real.accuracy*100,'-o',label='Empirical graph mean')
            raw=frame[(frame.transfer==mode)&(frame.noise==noise)&(frame.graph==0)]
            ax.scatter(raw.input_gain,raw.accuracy*100,s=15,alpha=.5,label=f'{frame.noise_repeat.nunique()} noise repetitions')
            ax.set_title(f'{mode}; synthetic noise SD={noise}');ax.set_ylim(50,102);ax.set_xticks([1,2,4]);ax.set_xlabel('Input gain');ax.set_ylabel('18-odor accuracy (%)');ax.legend(fontsize=7)
    fig.suptitle('Observed-channel partial circuit; fixed theta, normal APL')
    fig.tight_layout();fig.savefig(out/'odor_geometry.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nulls',type=int,default=10);parser.add_argument('--repeats',type=int,default=4);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/observed_odor_geometry';out.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/'data/door/SOURCE.json').read_text())
    for item in source['files']:
        if hashlib.sha256((root/'data/door'/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('DoOR input changed')
    prior=json.loads((root/'qc_reports/input_discrimination/manifest.json').read_text())
    for name,h in prior['data_sha256'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()!=h:raise ValueError('Circuit input changed')
    subsets=json.loads((root/'qc_reports/odor_input_audit/completeness_subsets.json').read_text())
    channels=next(s['channels'] for s in subsets if s['channel_count']==20)
    r=pd.read_csv(root/'data/door/door_response_matrix.csv',sep=';',index_col=0)
    complete=r.drop(index='SFR')[channels].dropna();sfr=r.loc['SFR',channels].to_numpy(float)
    assert np.isfinite(sfr).all() and len(complete)==18
    mapping=pd.read_csv(root/'qc_reports/odor_input_audit/pn_mapping_audit.csv',dtype={'root_888':str})
    selected=mapping[(mapping.status=='mapped')&mapping.response_channel.isin(channels)].set_index('root_888')
    meta=pd.read_feather(root/'data/banc_888_meta.feather')
    pi,ki,w,inh,drive,info=base.load_circuit(root)
    # Reconstruct the same sorted PN index used by load_circuit.
    edges=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].root_888.astype(str)
    pn_candidates=meta[(meta.cell_class=='antennal_lobe_projection_neuron')&(meta.neurotransmitter_verified=='acetylcholine')].root_888.astype(str)
    pn_ids=sorted(edges[edges.pre.isin(pn_candidates)&edges.post.isin(kc)].pre.unique())
    pn_selected=[i for i,p in enumerate(pn_ids) if p in selected.index]
    channel_index=np.array([channels.index(selected.loc[pn_ids[i],'response_channel']) for i in pn_selected])
    mask=np.isin(pi,pn_selected);pre=pi[mask];post=ki[mask];weights=w[mask]
    # Nulls only exchange targets of observed-PN edges: subset input strengths stay fixed.
    matrices=[];details=[]
    for graph in range(args.nulls+1):
        if graph:
            target,detail=robust.strength_preserving_null(pre,post,weights,np.random.default_rng(200000+graph))
            details.append(dict(graph=graph,seed=200000+graph,**detail))
        else:target=post
        matrices.append(sparse.csr_matrix((weights,(target,pre)),shape=(len(inh),info['pn']))[:,pn_selected])
    labels=np.repeat(np.arange(len(complete)),30);train=np.tile(np.arange(30)<10,len(complete))
    values=complete.to_numpy(float).T;rows=[];input_rows=[];blank_rows=[];max_res=0.
    for mode in ('absolute','positive_delta'):
        centers=transfer(values,sfr,mode)
        for repeat in range(args.repeats):
            for ni,noise in enumerate((.02,.1)):
                seed=210000+repeat*1000+ni
                # Perturb receptor channels, then share each draw across all matching PNs.
                noisy=np.clip(centers[:,labels]+np.random.default_rng(seed).normal(0,noise,(20,len(labels))),0,1)
                for gain in (1.,2.,4.):
                    x=gain*noisy[channel_index]
                    input_rows.append(dict(transfer=mode,noise_repeat=repeat,noise=noise,input_gain=gain,representation='channels',**classify(gain*noisy,labels,train)))
                    input_rows.append(dict(transfer=mode,noise_repeat=repeat,noise=noise,input_gain=gain,representation='mapped_PNs',**classify(x,labels,train)))
                    for graph,matrix in enumerate(matrices):
                        k,res=base.response(matrix@x,inh,drive,prior['model']['theta'],1.)
                        max_res=max(max_res,res)
                        rows.append(dict(transfer=mode,noise_repeat=repeat,noise=noise,input_gain=gain,graph=graph,input_seed=seed,**classify(k,labels,train)))
                        blank=transfer(sfr[:,None],sfr,mode)[channel_index]*gain
                        kb,res=base.response(matrix@blank,inh,drive,prior['model']['theta'],1.)
                        max_res=max(max_res,res)
                        blank_rows.append(dict(transfer=mode,noise_repeat=repeat,noise=noise,input_gain=gain,graph=graph,active_fraction=float(np.mean(kb>1e-8)),mean_activity=float(kb.mean())))
        print('transfer',mode,'complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    plot_results(frame,out)
    pd.DataFrame(input_rows).to_csv(out/'input_baselines.csv',index=False)
    pd.DataFrame(blank_rows).to_csv(out/'blank_baseline.csv',index=False)
    names=pd.read_csv(root/'data/door/odor.csv',sep=';',index_col=0)
    lookup=names.drop_duplicates('InChIKey').set_index('InChIKey')['Name']
    selected_odors=pd.DataFrame({'odor_key':complete.index,'name':[lookup.get(k,'') for k in complete.index]})
    selected_odors.to_csv(out/'selected_odors.csv',index=False)
    manifest=dict(door_source=source,data_sha256=prior['data_sha256'],channels=channels,odors=selected_odors.to_dict('records'),
        selected_pns=len(pn_selected),full_pns=info['pn'],observed_edges=len(pre),full_edges=info['pn_kc_edges'],kc=len(inh),
        theta=prior['model']['theta'],apl_gain=4,apl_strength=1,input_gains=[1,2,4],noise=[.02,.1],
        train_per_odor=10,test_per_odor=20,noise_repeats=args.repeats,null_count=args.nulls,chance_accuracy=1/18,rows=len(rows),max_residual=max_res,nulls=details,
        definition='Observed-channel partial circuit; edges from unobserved PN inputs are excluded, not imputed as measured zero.',
        calibration='Synthetic baseline theta retained; gains are fixed sensitivity settings, no outcome tuning.',
        noise_definition='Independent additive Gaussian perturbation per receptor channel and trial, clipped to [0,1], then duplicated to matching PNs. Same draws across graphs/gains/transfers.',
        classifier='18-way Euclidean nearest training centroid; exact/near ties scored as uniform expected accuracy.',
        limitations=['Direct OSN consensus to PN substitution, not measured PN physiology.',
            'Positive-delta transfer discards inhibitory deviations; absolute transfer retains relative low responses but not biophysical units.',
            'Only 20 channels, 18 completeness-selected odors and one connectome specimen.',
            'Synthetic noise is not empirical trial variability; repeated trials are not biological replicates.',
            'Existing synthetic theta may silence this partial circuit; gains are phenomenological.',
            'Constrained nulls, exploratory selection, no significance or uniform random sampling claim.',
            'No learning, downstream MBON, molecular mechanisms, disease or behavioral inference.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('selected_pns','observed_edges','rows','max_residual')},indent=2))

if __name__=='__main__':main()
