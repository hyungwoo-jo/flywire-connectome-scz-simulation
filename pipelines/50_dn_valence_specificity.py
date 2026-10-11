#!/usr/bin/env python3
"""Step 50: is the MDN result of step 49 specific to aversive learning, and does appetitive learning go the other way?

Same transfer as step 49 (MBON -> intermediate central neurons -> DNs, plus direct, signed, input-normalized).
Learning is linear in the compartment mask, so per odor u_o = mean over weak-A trials of (W * (exp(-ETA kA) - 1)) @ k_weak
is computed once; aversive change = T @ (u_o * [comp == aversive]), appetitive likewise.
Per DN, signed effect s_d = mean over odors of ref_d / |ref| (ref = DN change for that odor).

Measures (fixed before running):
1. MDN aversive score = mean s_d over the 4 MDNs. Null: permute the compartment labels over the 53 MBONs, keeping the
   numbers of aversive and appetitive compartments (1000 permutations). One-sided p (null >= observed).
2. MDN rank: percentile of the MDN score among all type-mean scores (types with >= 2 cells) under real labels.
3. Valence symmetry: Pearson correlation between aversive and appetitive s over all DNs (prediction: negative), and
   the MDN appetitive score (prediction: negative). Null for the MDN appetitive score: same permutation.
4. Reported, not tested: s for named DNs with known function, MDN (backward walking, Bidaye 2014), DNp09 (forward
   walking, Bidaye 2020), DNa02 and DNa01 (steering), DNp01 (giant fiber escape)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parent))
import downstream_common as dc  # noqa: E402

p49=dc._load('p49','49_descending_paths.py')
N_PERM=1000
NAMED=('MDN','DNp09','DNa02','DNa01','DNp01')


def effects(T,U,mask):
    R=(U*mask[None,:])@T.T                                  # odors x DNs
    n=np.linalg.norm(R,axis=1,keepdims=True);ok=n[:,0]>0
    return (R[ok]/n[ok]).mean(0)


def main():
    out=dc.ROOT/'qc_reports/dn_valence';out.mkdir(parents=True,exist_ok=True)
    S=dc.setup();a,e=S['a'],S['e'];keep=S['keep'];W=S['W'];comp=np.asarray(S['comp'])
    dns=a[a.super_class=='descending'].id.tolist()
    inter=sorted(set(e[e.pre.isin(keep)].post)&set(a[a.super_class=='central'].id)-set(keep))
    T=p49.norm_signed(e,a,inter,dns)@p49.norm_signed(e,a,keep,inter)+p49.norm_signed(e,a,keep,dns)
    T=T.toarray() if sparse.issparse(T) else np.asarray(T)
    U=np.stack([((W*(np.exp(-dc.ETA*S['kfull'][:,j])-1)[None,:])@S['kweak'][int(o)]).mean(1) for j,o in enumerate(S['ev'])])
    ctype=a.set_index('id').reindex(dns).cell_type.fillna('').to_numpy()
    mdn=ctype=='MDN'
    av=effects(T,U,(comp=='aversive').astype(float));ap=effects(T,U,(comp=='appetitive').astype(float))
    rng=np.random.default_rng(500001);null_av=np.empty(N_PERM);null_ap=np.empty(N_PERM)
    for i in range(N_PERM):
        pc=rng.permutation(comp)
        null_av[i]=effects(T,U,(pc=='aversive').astype(float))[mdn].mean()
        null_ap[i]=effects(T,U,(pc=='appetitive').astype(float))[mdn].mean()
    f=pd.DataFrame(dict(id=dns,cell_type=ctype,aversive=av,appetitive=ap))
    ft=f[f.cell_type!=''].groupby('cell_type').agg(n=('id','size'),aversive=('aversive','mean'),appetitive=('appetitive','mean'))
    ft=ft[ft.n>=2]
    obs_av,obs_ap=float(av[mdn].mean()),float(ap[mdn].mean())
    report=dict(
        dns=len(dns),odors=int(len(U)),permutations=N_PERM,
        mdn_aversive=dict(score=obs_av,null_mean=float(null_av.mean()),null_p95=float(np.percentile(null_av,95)),p=float((null_av>=obs_av).mean()),
                          type_percentile=float((ft.aversive<obs_av).mean())),
        mdn_appetitive=dict(score=obs_ap,null_mean=float(null_ap.mean()),null_p5=float(np.percentile(null_ap,5)),p_low=float((null_ap<=obs_ap).mean()),
                            type_percentile=float((ft.appetitive<obs_ap).mean())),
        valence_correlation=float(np.corrcoef(av,ap)[0,1]),
        named=ft.reindex(list(NAMED)).round(4).reset_index().to_dict(orient='records'),
        top_aversive=ft.sort_values('aversive',ascending=False).head(10).round(4).reset_index().to_dict(orient='records'),
        top_appetitive=ft.sort_values('appetitive',ascending=False).head(10).round(4).reset_index().to_dict(orient='records'))
    f.to_csv(out/'dn_effects.csv',index=False);ft.to_csv(out/'type_effects.csv')
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,indent=2,default=str))


if __name__=='__main__':main()
