#!/usr/bin/env python3
"""Graph-only test for overconvergent PN-type communities (Zheng 2022 style) in BANC."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('robust',Path(__file__).with_name('07_discrimination_robustness.py'))
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
base=robust.baseline

Z_EDGE=3.29
SEED0=170000
_G={}


def co_convergence(pre,post,pn_type,n_types,n_kc):
    b=np.zeros((n_kc,n_types),dtype=np.float64)
    b[post,pn_type[pre]]=1.
    return b.T@b


def components(adj,min_size=3):
    n=len(adj);seen=np.zeros(n,bool);out=[]
    for s in range(n):
        if seen[s]:continue
        stack=[s];comp=[];seen[s]=True
        while stack:
            u=stack.pop();comp.append(u)
            for v in np.flatnonzero(adj[u]):
                if not seen[v]:seen[v]=True;stack.append(v)
        if len(comp)>=min_size:out.append(sorted(comp))
    return sorted(out,key=len,reverse=True)


def zscores(obs,mu,sd):
    z=np.where(sd>0,(obs-mu)/np.where(sd>0,sd,1),0.);np.fill_diagonal(z,0);return z


def best_cluster(z,k=6,min_size=3):
    """Exploratory: average-linkage clusters of Z-row profiles; best mean within-cluster Z."""
    c=np.corrcoef(z);c=np.nan_to_num(c);d=1-c;np.fill_diagonal(d,0);d=(d+d.T)/2
    lab=fcluster(linkage(squareform(np.clip(d,0,None),checks=False),'average'),k,criterion='maxclust')
    best=(-np.inf,[])
    for g in np.unique(lab):
        ix=np.flatnonzero(lab==g)
        if len(ix)<min_size:continue
        sub=z[np.ix_(ix,ix)];score=sub[np.triu_indices(len(ix),1)].mean()
        if score>best[0]:best=(float(score),ix.tolist())
    return best


def _null(seed):
    g=_G;target,_=robust.strength_preserving_null(g['pre'],g['post'],g['w'],np.random.default_rng(seed))
    return co_convergence(g['pre'],target,g['pn_type'],g['n_types'],g['n_kc'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=1000);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/pn_community';out.mkdir(parents=True,exist_ok=True)
    pre,post,w,inh,drive,info=base.load_circuit(root)
    pns=pd.read_csv(root/'qc_reports/hallem_calibration/pn_mapping.csv',dtype={'root_888':str})
    types=sorted(pns.glomerulus.unique());tix={t:i for i,t in enumerate(types)}
    pn_type=pns.sort_values('pn_index').glomerulus.map(tix).to_numpy()
    n_kc=len(inh)
    _G.update(pre=pre,post=post,w=w,pn_type=pn_type,n_types=len(types),n_kc=n_kc)
    observed=co_convergence(pre,post,pn_type,len(types),n_kc)
    with Pool(args.workers) as pool:
        nulls=np.stack(pool.map(_null,range(SEED0+1,SEED0+args.nulls+1),chunksize=10))
    mu=nulls.mean(axis=0);sd=nulls.std(axis=0,ddof=1)
    z=zscores(observed,mu,sd)
    adj=z>Z_EDGE;comps=components(adj)
    np.savez_compressed(out/'nulls.npz',nulls=nulls.astype(np.float32),observed=observed)
    # Calibration check added after the pre-registered rule: each null as pseudo-observed.
    iu=np.triu_indices(len(types),1);n_edges=int((z[iu]>Z_EDGE).sum());cl_score,cl_ix=best_cluster(z)
    pseudo=[];n=len(nulls);tot=nulls.sum(axis=0);tot2=(nulls**2).sum(axis=0)
    for k in range(n):
        m_k=(tot-nulls[k])/(n-1);v_k=np.maximum((tot2-nulls[k]**2-(n-1)*m_k**2)/(n-2),0)
        zk=zscores(nulls[k],m_k,np.sqrt(v_k))
        pseudo.append(((zk[iu]>Z_EDGE).sum(),best_cluster(zk)[0],len(components(zk>Z_EDGE)[0]) if components(zk>Z_EDGE) else 0))
    pseudo=np.array(pseudo,dtype=float)
    calib=dict(observed_edges=n_edges,null_edges_mean=float(pseudo[:,0].mean()),null_edges_p95=float(np.percentile(pseudo[:,0],95)),
        edges_p=float((1+(pseudo[:,0]>=n_edges).sum())/(n+1)),
        observed_largest_component=len(comps[0]) if comps else 0,null_largest_component_median=float(np.median(pseudo[:,2])),
        exploratory_cluster=[types[i] for i in cl_ix],exploratory_cluster_mean_z=cl_score,
        null_cluster_mean_z_p95=float(np.percentile(pseudo[:,1],95)),cluster_p=float((1+(pseudo[:,1]>=cl_score).sum())/(n+1)))
    core=[types[i] for i in comps[0]] if comps else []
    zf=pd.DataFrame(z,index=types,columns=types);zf.to_csv(out/'pair_z.csv')
    pd.DataFrame(observed,index=types,columns=types).to_csv(out/'pair_observed.csv')
    # Descriptive: KC subtype distribution of output synapses from core vs other PNs.
    meta=pd.read_feather(root/'data/banc_888_meta.feather')
    kc=sorted(meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].root_888.astype(str))
    sub=meta.set_index(meta.root_888.astype(str)).loc[kc].cell_sub_class.fillna('unannotated').to_numpy()
    is_core=np.isin(pn_type[pre],[tix[t] for t in core])
    dist=pd.DataFrame(dict(group=np.where(is_core,'core','other'),kc_subclass=sub[post],w=w)).groupby(['group','kc_subclass']).w.sum().unstack(fill_value=0)
    dist=dist.div(dist.sum(axis=1),axis=0);dist.to_csv(out/'core_kc_subclass_share.csv')
    # Per-type summary
    pos=(z>Z_EDGE).sum(axis=1)
    summary=pd.DataFrame(dict(type=types,pn_count=np.bincount(pn_type,minlength=len(types)),
        partners_z_gt_edge=pos,mean_z=z.sum(axis=1)/(len(types)-1),in_core=[t in core for t in types])).sort_values('mean_z',ascending=False)
    summary.to_csv(out/'type_summary.csv',index=False)
    hallem=pns[pns.receptor.fillna('')!=''].glomerulus.unique()
    report=dict(nulls=args.nulls,seeds=[SEED0+1,SEED0+args.nulls],z_edge=Z_EDGE,types=len(types),
        components=[[types[i] for i in c] for c in comps],core_community=core,replicated=len(core)>=3,
        core_hallem_glomeruli=sorted(set(core)&set(hallem)),core_kc_subclass_share=dist.round(4).to_dict(orient='index'),
        null='strength_preserving_null on all PN->KC edges (PN out-degree, KC in-degree and input weight sum preserved)',
        calibration_check=calib)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    order=np.argsort([-(t in core) for t in types],kind='stable')
    fig,ax=plt.subplots(figsize=(11,10));lim=max(5,np.percentile(np.abs(z),99))
    im=ax.imshow(z[np.ix_(order,order)],cmap='RdBu_r',vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(types)));ax.set_xticklabels([types[i] for i in order],rotation=90,fontsize=5)
    ax.set_yticks(range(len(types)));ax.set_yticklabels([types[i] for i in order],fontsize=5)
    ax.set_title(f'PN-type co-convergence Z vs {args.nulls} strength-preserving nulls (core first, n={len(core)})')
    fig.colorbar(im,ax=ax,shrink=.6);fig.tight_layout();fig.savefig(out/'pair_z.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(core_size=len(core),core_hallem=len(report['core_hallem_glomeruli']),calibration_check=calib),indent=2))


if __name__=='__main__':main()
