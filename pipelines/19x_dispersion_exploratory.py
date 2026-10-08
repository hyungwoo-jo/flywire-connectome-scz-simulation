#!/usr/bin/env python3
"""Exploratory (not pre-registered): dispersion of PN-type co-convergence Z for real, N0 and N1 graphs."""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p19',root/'pipelines/19_deficit_cause.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
cal=p.cal;odors,sfr,names=cal.load_hallem(root);c=cal.load_partial_circuit(root,list(odors.columns));_,mask=cal.partial_matrix(c)
meta=pd.read_feather(root/'data/banc_888_meta.feather');kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].copy()
kc=kc.set_index(kc.root_888.astype(str)).sort_index();kc_type=kc.cell_sub_class.fillna(kc.cell_type).fillna('unknown').to_numpy(str)
pns=c['pns'];types=sorted(pns.glomerulus.unique());tix={t:i for i,t in enumerate(types)}
nl=np.load(root/'qc_reports/pn_community/nulls.npz')['nulls'].astype(float)
G=dict(pre=c['pre'],full_post=c['post'],mask=mask,pn_type=pns.sort_values('pn_index').glomerulus.map(tix).to_numpy(),n_types=len(types),n_kc=len(c['inh']),mu=nl.mean(0),sd=nl.std(0,ddof=1),hallem_idx=np.array(sorted(tix[g] for g in c['mapped'].glomerulus.unique())))
pm,qm,wm=c['pre'][mask],c['post'][mask],c['w'][mask]
def disp(job):
    k,i=job
    if k=='real':t=qm
    elif k=='N0':t,_=p.robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(p.N0_SEED0+i))
    else:t=p.subtype_target_null(pm,qm,wm,kc_type,np.random.default_rng(p.N1_SEED0+i))
    post=G['full_post'].copy();post[G['mask']]=t
    z=p.com.zscores(p.com.co_convergence(G['pre'],post,G['pn_type'],G['n_types'],G['n_kc']),G['mu'],G['sd'])
    h=G['hallem_idx'];v=z[np.ix_(h,h)][np.triu_indices(len(h),1)]
    return dict(kind=k,graph=i,z_sd=v.std(),z_max=v.max(),n_pos=(v>3.29).sum(),n_neg=(v<-3.29).sum())
if __name__=="__main__":
  with Pool(10) as pool:r=pd.DataFrame(pool.map(disp,[('real',0)]+[(k,i) for k in ('N0','N1') for i in range(1,101)]))
  m=pd.read_csv(root/'qc_reports/deficit_cause/metrics.csv');m=m[(m.window==.5)&(m.target==.1)]
  r=r.merge(m[['kind','graph','ALL_dprime']],on=['kind','graph'])
  print(r[r.kind=='real'].to_string())
  print(r[r.kind!='real'].groupby('kind')[['z_sd','z_max','n_pos','n_neg']].describe(percentiles=[.05,.95]).T.round(2).to_string())
  from scipy import stats
  nn=r[r.kind!='real'].copy()
  for col in ['z_sd','n_pos','n_neg']:
      x=nn[col]-nn.groupby('kind')[col].transform('mean');y=nn.ALL_dprime-nn.groupby('kind').ALL_dprime.transform('mean')
      print(col,'centered spearman',round(stats.spearmanr(x,y).statistic,3),round(stats.spearmanr(x,y).pvalue,3))
  r.to_csv(root/'qc_reports/deficit_cause/exploratory_dispersion.csv',index=False)
