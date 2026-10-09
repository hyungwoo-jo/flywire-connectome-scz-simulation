#!/usr/bin/env python3
"""Is compensation built into BANC wiring? Synapses per PN connection vs number of PN connections per KC."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_PERM=1000
SEED0=320000


def kc_table(meta,edges,side):
    kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side==side)].copy();kc['id']=kc.root_888.astype(str)
    pn=meta[(meta.cell_class=='antennal_lobe_projection_neuron')&(meta.neurotransmitter_verified=='acetylcholine')].root_888.astype(str)
    e=edges[edges.pre.isin(pn)&edges.post.isin(kc.id)][['pre','post','count']].copy()
    sub=kc.set_index('id').cell_sub_class.fillna(kc.set_index('id').cell_type).fillna('unknown')
    return e,sub


def slope(post,counts):
    """OLS slope of log(mean synapses per connection) on log(n connections) across KCs."""
    df=pd.DataFrame(dict(post=post,c=counts)).groupby('post').c.agg(['size','mean'])
    x=np.log(df['size'].to_numpy(float));y=np.log(df['mean'].to_numpy(float))
    return float(np.polyfit(x,y,1)[0]),df


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/wiring_compensation';out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_feather(root/'data/banc_888_meta.feather');edges=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    res={};tables=[]
    for si,side in enumerate(('right','left')):
        e,sub=kc_table(meta,edges,side);post=e.post.to_numpy();counts=e['count'].to_numpy(float)
        b,df=slope(post,counts);df=df.join(sub.rename('subtype'));df['side']=side;tables.append(df.reset_index())
        rng=np.random.default_rng(SEED0+si*10000);null=np.array([slope(post,rng.permutation(counts))[0] for _ in range(N_PERM)])
        p=float((1+(null<=b).sum())/(N_PERM+1))
        bysub={}
        for t in ('KCg','KCab',"KCa'b'"):
            ids=set(sub[sub==t].index);mask=np.isin(post,list(ids))
            if mask.sum()>50:bysub[t]=slope(post[mask],counts[mask])[0]
        S=df['size']*df['mean']
        res[side]=dict(kcs=len(df),connections=len(e),b=b,alpha_eff=-b,null_mean=float(null.mean()),null_p05=float(np.percentile(null,5)),p=p,
                       n_median=float(df['size'].median()),n_range=[int(df['size'].min()),int(df['size'].max())],
                       synapses_per_connection_median=float(df['mean'].median()),total_input_cv=float(S.std()/S.mean()),
                       slope_total_on_n=float(np.polyfit(np.log(df['size']),np.log(S),1)[0]),by_subtype_b=bysub)
    pd.concat(tables).to_csv(out/'per_kc.csv',index=False)
    R,L=res['right'],res['left']
    v='compensation_in_wiring' if (R['b']<0 and L['b']<0 and R['p']<.05 and L['p']<.05) else 'not_established'
    report=dict(verdict=v,results=res,permutations=N_PERM)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    t=pd.concat(tables);fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,side in zip(axes,('right','left')):
        q=t[t.side==side];ax.scatter(q['size']+np.random.default_rng(0).uniform(-.2,.2,len(q)),q['mean'],s=4,alpha=.3)
        xs=np.linspace(q['size'].min(),q['size'].max(),50);c=np.polyfit(np.log(q['size']),np.log(q['mean']),1)
        ax.plot(xs,np.exp(c[1])*xs**c[0],'r-',label=f"b = {c[0]:.2f}");ax.set_xlabel('PN connections per KC');ax.set_title(side);ax.legend()
    axes[0].set_ylabel('mean synapses per connection');fig.suptitle('32: wiring-level compensation');fig.tight_layout();fig.savefig(out/'wiring_compensation.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
