#!/usr/bin/env python3
"""Search FlyWire for 'ACC-like' integrators: central neurons with enriched input from MBONs (value), ascending/endocrine
neurons (body state, nociception) and fan-shaped body neurons, and their output to DANs and descending neurons."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MIN_INPUT=200      # candidate must receive at least this many synapses
MIN_CLASS_SYN=20   # and at least this many from each of the three classes
ENRICH=3.0         # and each class fraction at least 3x its brain-wide share
N_PERM=200
SEED=440000


def load_edges():
    spec=importlib.util.spec_from_file_location('p34',ROOT/'pipelines/34_flywire_replication.py')
    m=importlib.util.module_from_spec(spec);sys.modules['p34']=m;spec.loader.exec_module(m)
    return m.load_flywire(ROOT)


def classes(a):
    ct=a.cell_type.fillna('');cc=a.cell_class.fillna('');sc=a.super_class.fillna('')
    lab=np.full(len(a),'other',dtype=object)
    lab[(sc.isin(['ascending','endocrine'])).to_numpy()]='BODY'
    lab[ct.str.match(r'^FB\d').to_numpy()]='FB'
    lab[(cc=='MBON').to_numpy()]='MBON'
    return pd.Series(lab,index=a.id)


PAIRS=(('MBON','BODY'),('MBON','FB'),('BODY','FB'))


def converge(inp,lab,total,base,need=('MBON','BODY','FB')):
    """inp: edges with pre,post,count; returns per-post synapse counts by class and the convergence flag for the classes in `need`."""
    x=inp.assign(c=inp.pre.map(lab).fillna('other')).pivot_table(index='post',columns='c',values='count',aggfunc='sum',fill_value=0)
    for k in ('MBON','BODY','FB'):
        if k not in x:x[k]=0
    x=x.join(total.rename('total'),how='left')
    flag=(x.total>=MIN_INPUT)
    for k in need:flag&=(x[k]>=MIN_CLASS_SYN)&((x[k]/x.total)>=ENRICH*base[k])
    return x,flag


def main():
    out=ROOT/'qc_reports/acc_like_integrators';out.mkdir(parents=True,exist_ok=True)
    a,e=load_edges();lab=classes(a)
    central=set(a[a.super_class=='central'].id)-set(a[a.cell_class=='Kenyon_Cell'].id)
    inp=e[e.post.isin(central)]
    total=inp.groupby('post')['count'].sum()
    syn_by_class=inp.assign(c=inp.pre.map(lab).fillna('other')).groupby('c')['count'].sum()
    base={k:float(syn_by_class.get(k,0)/syn_by_class.sum()) for k in ('MBON','BODY','FB')}
    x,flag=converge(inp,lab,total,base);obs=int(flag.sum())
    # Null: permute class labels among presynaptic neurons within output-synapse deciles.
    outdeg=e.groupby('pre')['count'].sum();pres=outdeg.index.to_numpy();dec=pd.qcut(outdeg.rank(method='first'),10,labels=False).to_numpy()
    lab_pres=lab.reindex(pres).fillna('other').to_numpy();rng=np.random.default_rng(SEED);null=[]
    for _ in range(N_PERM):
        perm=lab_pres.copy()
        for d in range(10):
            ix=np.flatnonzero(dec==d);perm[ix]=perm[rng.permutation(ix)]
        _,fl=converge(inp,pd.Series(perm,index=pres),total,base);null.append(int(fl.sum()))
    null=np.array(null)
    cand=x[flag].copy();ann=a.set_index('id')
    cand['cell_type']=ann.reindex(cand.index).cell_type;cand['cell_class']=ann.reindex(cand.index).cell_class;cand['side']=ann.reindex(cand.index).side
    cand['top_nt']=ann.reindex(cand.index).top_nt
    for k in ('MBON','BODY','FB'):cand[f'frac_{k}']=cand[k]/cand.total
    # Outputs of candidates.
    olab=pd.Series('other',index=a.id);olab[a[a.cell_class=='DAN'].id]='DAN';olab[a[a.super_class=='descending'].id]='DN';olab[a[a.cell_class=='MBON'].id]='MBON'
    o=e[e.pre.isin(cand.index)].assign(c=lambda d:d.post.map(olab).fillna('other')).pivot_table(index='pre',columns='c',values='count',aggfunc='sum',fill_value=0)
    for k in ('DAN','DN','MBON'):
        if k not in o:o[k]=0
    o['out_total']=o.sum(1);cand=cand.join(o[['DAN','DN','MBON','out_total']].add_prefix('to_'),how='left').fillna({'to_DAN':0,'to_DN':0,'to_MBON':0,'to_out_total':0})
    for k in ('DAN','DN','MBON'):cand[f'out_frac_{k}']=cand[f'to_{k}']/cand.to_out_total.replace(0,np.nan)
    cand['loop_like']=(cand.out_frac_DAN>=.05)&(cand.out_frac_DN>=.05)
    cand=cand.sort_values('total',ascending=False);cand.to_csv(out/'candidates.csv')
    by_type=cand.groupby(cand.cell_type.fillna('unknown')).agg(n=('total','size'),frac_MBON=('frac_MBON','mean'),frac_BODY=('frac_BODY','mean'),frac_FB=('frac_FB','mean'),
        out_DAN=('out_frac_DAN','mean'),out_DN=('out_frac_DN','mean'),loop_like=('loop_like','sum')).sort_values('n',ascending=False)
    by_type.to_csv(out/'candidates_by_type.csv')
    pair_res={}
    for pr in PAIRS:
        _,fl=converge(inp,lab,total,base,need=pr);ob=int(fl.sum());nl=[]
        rng2=np.random.default_rng(SEED+1)
        for _ in range(N_PERM):
            perm=lab_pres.copy()
            for d in range(10):
                ix=np.flatnonzero(dec==d);perm[ix]=perm[rng2.permutation(ix)]
            nl.append(int(converge(inp,pd.Series(perm,index=pres),total,base,need=pr)[1].sum()))
        nl=np.array(nl);ids=fl[fl].index
        oo=e[e.pre.isin(ids)].assign(c=lambda d:d.post.map(olab).fillna('other')).groupby('c')['count'].sum()
        # two-step: do these integrators feed neurons that integrate the remaining class?
        pair_res['+'.join(pr)]=dict(observed=ob,null_mean=float(nl.mean()),null_p95=float(np.percentile(nl,95)),p=float((1+(nl>=ob).sum())/(N_PERM+1)),
            classes=ann.reindex(ids).cell_class.fillna('unknown').value_counts().head(8).to_dict(),
            types=ann.reindex(ids).cell_type.fillna('unknown').value_counts().head(12).to_dict(),
            output_share={k:float(v/oo.sum()) for k,v in oo.items()} if len(oo) else {})
        pd.Series(list(ids),name='id').to_csv(out/f"pair_{'_'.join(pr)}.csv",index=False)
    report=dict(pairwise_exploratory=pair_res,criteria=dict(min_input=MIN_INPUT,min_class_syn=MIN_CLASS_SYN,enrichment=ENRICH),baseline_share=base,
                observed=obs,null_mean=float(null.mean()),null_p95=float(np.percentile(null,95)),p=float((1+(null>=obs).sum())/(N_PERM+1)),
                candidates_by_class=cand.cell_class.fillna('unknown').value_counts().to_dict(),loop_like=int(cand.loop_like.sum()),
                top_types=by_type.head(20).round(3).reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:report[k] for k in ('baseline_share','observed','null_mean','null_p95','p','candidates_by_class','loop_like','pairwise_exploratory')},indent=2,default=str))
    pd.set_option('display.width',200);print(by_type.head(25).round(3).to_string())


if __name__=='__main__':main()
