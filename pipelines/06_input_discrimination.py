#!/usr/bin/env python3
"""Directed BANC v888 PN/KC/APL rate-model exploration, not a disease model."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_circuit(root):
    m = pd.read_feather(root / 'data/banc_888_meta.feather')
    e = pd.read_feather(root / 'data/banc_888_edgelist_simple_v2.feather')
    # The edge release uses root_888, not the mutable current root_id.
    if m.root_888.duplicated().any():
        raise ValueError('Ambiguous v888 annotations')
    m = m.set_index(m.root_888.astype(str))
    kc = m[(m.cell_class == 'kenyon_cell') & (m.side == 'right')].index.tolist()
    apl = m[(m.cell_type == 'APL') & (m.side == 'right')].index.tolist()
    if len(apl) != 1:
        raise ValueError('Expected one annotated right APL')
    pn = m[(m.cell_class == 'antennal_lobe_projection_neuron') &
           (m.neurotransmitter_verified == 'acetylcholine')].index
    edges = e[e.pre.isin(pn) & e.post.isin(kc)].copy()
    pn = sorted(edges.pre.unique()); kc = sorted(kc)
    pi = {x:i for i,x in enumerate(pn)}; ki = {x:i for i,x in enumerate(kc)}
    pre = edges.pre.map(pi).to_numpy(dtype=int)
    post = edges.post.map(ki).to_numpy(dtype=int)
    counts = edges['count'].to_numpy(dtype=float)
    inh = e[e.pre.isin(apl) & e.post.isin(kc)].groupby('post')['count'].sum().reindex(kc, fill_value=0).to_numpy(dtype=float)
    drive = e[e.pre.isin(kc) & e.post.isin(apl)].groupby('pre')['count'].sum().reindex(kc, fill_value=0).to_numpy(dtype=float)
    if not inh.any() or not drive.any():
        raise ValueError('APL feedback edges missing')
    # Fixed empirical scaling: counts are relative weights, not conductances.
    scale = np.median(np.bincount(post, weights=counts, minlength=len(kc)))
    inh /= inh.mean(); drive /= drive.sum()
    info = dict(dataset='BANC v888',id_column='root_888',hemisphere='right',pn=len(pn),kc=len(kc),apl=apl,
                pn_kc_edges=len(edges),pn_kc_synapses=int(counts.sum()),
                apl_kc_edges=int(np.count_nonzero(inh)),kc_apl_edges=int(np.count_nonzero(drive)),pn_weight_scale=float(scale),
                current_root_id_duplicate_rows=int(m.root_id.duplicated().sum()),
                pn_verified_transmitters=m.loc[pn].neurotransmitter_verified.fillna('unknown').value_counts().to_dict(),
                kc_verified_transmitters=m.loc[kc].neurotransmitter_verified.fillna('unknown').value_counts().to_dict(),
                apl_verified_transmitter=m.loc[apl].neurotransmitter_verified.tolist())
    return pre, post, counts / scale, inh, drive, info


def response(excitation, inh, drive, theta, strength, gain=4.0):
    """Solve a = drive @ clip(E-theta-gain*strength*inh*a,0,1)."""
    lo = np.zeros(excitation.shape[1])
    hi = drive @ np.clip(excitation-theta,0,1)
    for _ in range(35):
        a=(lo+hi)/2
        k=np.clip(excitation-theta-gain*strength*inh[:,None]*a,0,1)
        above=(drive @ k)>a
        lo=np.where(above,a,lo); hi=np.where(above,hi,a)
    a=(lo+hi)/2
    k=np.clip(excitation-theta-gain*strength*inh[:,None]*a,0,1)
    return k, float(np.max(np.abs(drive @ k-a)))


def rewire(pre, post, rng, multiple=5):
    target=post.copy(); occupied=set(zip(pre,target)); done=attempts=0; required=multiple*len(pre)
    while done<required and attempts<required*100:
        attempts+=1; i,j=rng.integers(len(pre),size=2)
        u,v,x,y=int(pre[i]),int(target[i]),int(pre[j]),int(target[j])
        if u==x or v==y or (u,y) in occupied or (x,v) in occupied:
            continue
        occupied.remove((u,v));occupied.remove((x,y));occupied.add((u,y));occupied.add((x,v))
        target[i],target[j]=y,v;done+=1
    if done != required:
        raise RuntimeError('Insufficient null swaps')
    assert np.array_equal(np.bincount(post),np.bincount(target))
    return target, dict(swaps=done, changed_fraction=float(np.mean(target!=post)))


def patterns(n, overlap, noise, seed, trials):
    rng=np.random.default_rng(seed);size=max(2,round(.2*n));shared=round(size*overlap)
    order=rng.permutation(n);a=order[:size];b=np.r_[order[:shared],order[size:size+size-shared]]
    base=np.zeros((n,2));base[a,0]=1;base[b,1]=1
    labels=np.repeat([0,1],trials)
    x=np.maximum(0,base[:,labels]+rng.normal(0,noise,(n,len(labels))))
    return x, labels, float(shared/size)


def metrics(k, labels):
    # First half of each class trains centroids; the other half tests them.
    train=np.zeros(len(labels),dtype=bool)
    for c in (0,1):
        ix=np.flatnonzero(labels==c);train[ix[:len(ix)//2]]=True
    means=np.stack([k[:,train & (labels==c)].mean(axis=1) for c in (0,1)])
    test=k[:,~train].T
    distances=((test[:,None,:]-means[None,:,:])**2).sum(axis=2)
    cosine=float(means[0]@means[1]/max(np.linalg.norm(means[0])*np.linalg.norm(means[1]),1e-15))
    within=float(np.mean([np.mean(np.sum((k[:,labels==c].T-means[c])**2,axis=1)) for c in (0,1)]))
    between=float(np.sum((means[0]-means[1])**2))
    return dict(active_fraction=float(np.mean(k>1e-8)),mean_activity=float(k.mean()),centroid_cosine=cosine,
                between_within_ratio=between/max(within,1e-15),accuracy=float(np.mean(distances.argmin(axis=1)==labels[~train])),
                silent_trial_fraction=float(np.mean(np.max(k,axis=0)<=1e-8)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nulls',type=int,default=10);parser.add_argument('--trials',type=int,default=40)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=root/'qc_reports/input_discrimination';out.mkdir(parents=True,exist_ok=True)
    pre,post,w,inh,drive,info=load_circuit(root);n=len(inh);n_pn=info['pn']
    matrix=sparse.csr_matrix((w,(post,pre)),shape=(n,n_pn))
    cal=np.maximum(0,np.random.default_rng(9000).normal(0,.1,(n_pn,100)))
    rng=np.random.default_rng(9001)
    for c in range(cal.shape[1]):cal[rng.choice(n_pn,round(.2*n_pn),replace=False),c]+=1
    ecal=matrix@cal;lo,hi=0.,float(ecal.max())
    # Fit only the baseline sparseness, never a discrimination outcome.
    for _ in range(30):
        theta=(lo+hi)/2;k,_=response(ecal,inh,drive,theta,1)
        if np.mean(k>1e-8)>.10:lo=theta
        else:hi=theta
    theta=(lo+hi)/2
    inputs=[]
    for oi,overlap in enumerate((.25,.6,.9)):
        for ni,noise in enumerate((.1,.4)):
            x,y,actual=patterns(n_pn,overlap,noise,10000+oi*100+ni,args.trials)
            inputs.append((overlap,actual,noise,x,y))
    pd.DataFrame([dict(input_overlap=o,actual_overlap=a,noise=noise,**metrics(x,y))
                  for o,a,noise,x,y in inputs]).to_csv(out/'pn_input_baseline.csv',index=False)
    rows=[];null_details=[]
    for graph in range(args.nulls+1):
        if graph:
            target,detail=rewire(pre,post,np.random.default_rng(11000+graph));detail['seed']=11000+graph;null_details.append(detail)
            mat=sparse.csr_matrix((w,(target,pre)),shape=(n,n_pn))
        else:mat=matrix
        for overlap,actual,noise,x,y in inputs:
            excitation=mat@x
            for strength in (1.,.75,.5,.25,0.):
                k,residual=response(excitation,inh,drive,theta,strength)
                rows.append(dict(graph=graph,input_overlap=overlap,actual_overlap=actual,noise=noise,apl_strength=strength,residual=residual,**metrics(k,y)))
        print(f'graph {graph}/{args.nulls} complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    hashes={name:hashlib.sha256((root/'data'/name).read_bytes()).hexdigest() for name in ('banc_888_meta.feather','banc_888_edgelist_simple_v2.feather')}
    report=dict(circuit=info,model=dict(theta=theta,apl_gain=4,baseline_target_active_fraction=.1,calibration_active_fraction=float(np.mean(response(ecal,inh,drive,theta,1)[0]>1e-8)),
                                     equations='k=clip(W_PN_KC*x-theta-4*s*w_APL_KC*a,0,1); a=w_KC_APL dot k',solver='35 monotone bisection iterations',normalization='fixed across strengths and graphs'),
                nulls=null_details,trials_per_class=args.trials,max_residual=float(frame.residual.max()),data_sha256=hashes,
                python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for row,noise in enumerate((.1,.4)):
        for col,metric in enumerate(('active_fraction','centroid_cosine','accuracy')):
            ax=axes[row,col]
            for overlap,color in zip((.25,.6,.9),('#0072B2','#D55E00','#009E73')):
                sub=frame[(frame.noise==noise)&(frame.input_overlap==overlap)]
                real=sub[sub.graph==0].sort_values('apl_strength');null=sub[sub.graph>0].groupby('apl_strength')[metric].agg(['min','max']).sort_index()
                ax.fill_between(null.index,null['min'],null['max'],color=color,alpha=.13)
                ax.plot(real.apl_strength,real[metric],'-o',color=color,label=f'overlap {overlap}')
            ax.set_title(f'{metric}; noise={noise}');ax.set_xlabel('APL strength');ax.legend(fontsize=8)
            if metric in ('accuracy','centroid_cosine'):ax.set_ylim(0,1.02)
    fig.suptitle('Directed BANC PN/KC/APL model; shade = degree-preserving PN-KC null range')
    fig.tight_layout();fig.savefig(out/'discrimination.png',dpi=180)
    print(json.dumps(report['model'],indent=2));print('max residual',report['max_residual'])


if __name__=='__main__':main()
