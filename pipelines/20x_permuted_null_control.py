#!/usr/bin/env python3
"""Post hoc control (not pre-registered): apply the step-20 label permutations to N0 random wiring.

If random wiring loses as much d' under permutation as the real wiring does, the step-20 drop
reflects receptor/output-strength misalignment rather than the real pair structure."""
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p20',Path(__file__).with_name('20_label_permutation.py'))
p20=importlib.util.module_from_spec(spec);spec.loader.exec_module(p20)
disc=p20.disc;cal=p20.cal;base=cal.base;robust=cal.robust
_G={}


def _run(job):
    graph,perm=job;g=_G;c=g['c']
    pm,qm,wm=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
    target,_=robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(disc.NULL_SEED0+graph))
    m,_=cal.partial_matrix(c,target)
    idx=g['identity'] if perm==0 else p20.receptor_rows(c['mapped'],g['receptors'],g['mappings'][perm])
    cond=g['cond'];k,res=base.response(m@cond['pn'][idx],c['inh'],c['drive'],cond['theta'],1.,gain=cal.APL_GAIN)
    return dict(graph=graph,perm=perm,residual=res,**disc.evaluate(k,g['labels'],g['train'],{'ALL':np.arange(g['n'])}))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/label_permutation'
    m16=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=cal.load_partial_circuit(root,receptors);_,mask=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));d=delta[:,keep];n=len(keep)
    labels=np.repeat(np.arange(n),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,n)
    orn=cal.sample_orn(d,sfr.to_numpy(),labels,.5,np.random.default_rng(disc.NOISE_SEED))
    cond=dict(pn=cal.olsen(orn)/cal.OLSEN['rmax'],theta=m16['targets']['0.10']['theta'])
    mapped=c['mapped'];gloms=sorted(mapped.glomerulus.unique());pn_count=mapped.glomerulus.value_counts().to_dict()
    mappings={i:p20.class_permutation(gloms,pn_count,np.random.default_rng(p20.PERM_SEED0+i)) for i in range(1,101)}
    _G.update(c=c,mask=mask,cond=cond,labels=labels,train=train,n=n,receptors=receptors,mappings=mappings,
              identity=p20.receptor_rows(mapped,receptors,{g:g for g in gloms}))
    # Graph i is paired with permutation i; perm 0 reproduces step 18's N0 values.
    with Pool(10) as pool:rows=pool.map(_run,[(i,i) for i in range(1,101)]+[(i,0) for i in range(1,101)])
    f=pd.DataFrame(rows);f.to_csv(out/'posthoc_null_permuted.csv',index=False)
    p20m=pd.read_csv(out/'metrics.csv');p20m=p20m[(p20m.window==.5)&(p20m.target==.1)]
    true_n0=f[f.perm==0].ALL_dprime;perm_n0=f[f.perm>0].ALL_dprime
    real_true=float(p20m[p20m.perm==0].ALL_dprime.iloc[0]);real_perm=p20m[p20m.perm>0].ALL_dprime
    res=dict(N0_true_median=float(true_n0.median()),N0_permuted_median=float(perm_n0.median()),
             real_true=real_true,real_permuted_median=float(real_perm.median()),
             drop_N0=float(true_n0.median()-perm_n0.median()),drop_real=float(real_true-real_perm.median()),
             real_permuted_vs_N0_permuted_mwu_p_less=float(__import__('scipy.stats',fromlist=['x']).mannwhitneyu(real_perm,perm_n0,alternative='less').pvalue),
             real_true_rank_vs_N0_true=float((true_n0>=real_true).mean()),max_residual=float(f.residual.max()))
    (out/'posthoc_null_permuted.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))


if __name__=='__main__':main()
