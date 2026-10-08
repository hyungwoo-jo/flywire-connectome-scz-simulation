#!/usr/bin/env python3
"""Keep the real wiring; permute receptor-to-glomerulus assignment within PN-count classes."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('disc18',Path(__file__).with_name('18_community_discrimination.py'))
disc=importlib.util.module_from_spec(spec);spec.loader.exec_module(disc)
cal=disc.cal;base=cal.base

PERM_SEED0=200000
MANTEL_SEED=201000
MANTEL_N=10000
_G={}


def class_permutation(glomeruli,pn_count,rng):
    """Map each glomerulus to another with the same PN count; never the identity overall."""
    while True:
        mapping={}
        for k in sorted(set(pn_count.values())):
            members=sorted(g for g in glomeruli if pn_count[g]==k)
            for a,b in zip(members,rng.permutation(members)):mapping[a]=b
        if any(a!=b for a,b in mapping.items()):return mapping


def receptor_rows(mapped,receptors,mapping):
    """Receptor index per mapped PN after relabeling its glomerulus."""
    rec=dict(zip(mapped.glomerulus,mapped.receptor))
    return np.array([receptors.index(rec[mapping[g]]) for g in mapped.glomerulus])


def _run(job):
    g=_G;c=g['c'];rows=[]
    idx=g['identity'] if job==0 else receptor_rows(c['mapped'],g['receptors'],g['mappings'][job])
    for cond in g['conds']:
        x=cond['pn'][idx]
        k,res=base.response(g['m']@x,c['inh'],c['drive'],cond['theta'],1.,gain=cal.APL_GAIN)
        rows.append(dict(perm=job,window=cond['window'],target=cond['target'],residual=res,
                         **disc.evaluate(k,g['labels'],g['train'],{'ALL':np.arange(g['n'])})))
    return rows


def mantel(sim,z,rng,n):
    iu=np.triu_indices(len(sim),1);obs=stats.spearmanr(sim[iu],z[iu]).statistic
    null=np.empty(n)
    for i in range(n):
        p=rng.permutation(len(sim));null[i]=stats.spearmanr(sim[np.ix_(p,p)][iu],z[iu]).statistic
    return float(obs),float((1+(np.abs(null)>=abs(obs)).sum())/(n+1))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--perms',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/label_permutation';out.mkdir(parents=True,exist_ok=True)
    m16=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=cal.load_partial_circuit(root,receptors);m,_=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));d=delta[:,keep];n=len(keep)
    labels=np.repeat(np.arange(n),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,n)
    conds=[]
    for wi,window in enumerate((.5,.25)):
        orn=cal.sample_orn(d,sfr.to_numpy(),labels,window,np.random.default_rng(disc.NOISE_SEED+wi))
        pn=cal.olsen(orn)/cal.OLSEN['rmax']
        for target in ('0.10','0.05'):conds.append(dict(window=window,target=float(target),theta=m16['targets'][target]['theta'],pn=pn))
    mapped=c['mapped'];gloms=sorted(mapped.glomerulus.unique());pn_count=mapped.glomerulus.value_counts().to_dict()
    mappings={i:class_permutation(gloms,pn_count,np.random.default_rng(PERM_SEED0+i)) for i in range(1,args.perms+1)}
    identity=receptor_rows(mapped,receptors,{g:g for g in gloms})
    _G.update(c=c,m=m,conds=conds,labels=labels,train=train,n=n,receptors=receptors,mappings=mappings,identity=identity)
    with Pool(args.workers) as pool:rows=[r for rs in pool.map(_run,range(args.perms+1)) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    ref=pd.read_csv(root/'qc_reports/community_discrimination/metrics.csv')
    summary=[]
    for (window,target),g in f.groupby(['window','target']):
        real=g[g.perm==0].iloc[0];perm=g[g.perm>0];n0=ref[(ref.window==window)&(ref.target==target)&(ref.graph>0)]
        r18=ref[(ref.window==window)&(ref.target==target)&(ref.graph==0)].iloc[0]
        r=dict(window=window,target=target,primary=bool(window==.5 and target==.10),identity_matches_18=bool(np.isclose(real.ALL_dprime,r18.ALL_dprime)))
        for metric in ('ALL_dprime','ALL_accuracy'):
            r[f'{metric}_real']=float(real[metric]);r[f'{metric}_perm_median']=float(perm[metric].median());r[f'{metric}_N0_median']=float(n0[metric].median())
            r[f'{metric}_z_perm']=float((real[metric]-perm[metric].mean())/perm[metric].std(ddof=1))
            r[f'{metric}_rank_perm']=float((perm[metric]>=real[metric]).mean())
            r[f'{metric}_mwu_p_perm_lt_N0']=float(stats.mannwhitneyu(perm[metric],n0[metric],alternative='less').pvalue)
        summary.append(r)
    s=pd.DataFrame(summary);s.to_csv(out/'summary.csv',index=False)
    # P3: tuning similarity vs co-convergence Z (input-only, all 110 odors).
    pn_clean=cal.olsen(delta);rec=dict(zip(mapped.glomerulus,mapped.receptor))
    prof=np.stack([pn_clean[receptors.index(rec[g])] for g in gloms]);sim=np.corrcoef(prof)
    z=pd.read_csv(root/'qc_reports/pn_community/pair_z.csv',index_col=0).loc[gloms,gloms].to_numpy()
    rho,p3=mantel(sim,z,np.random.default_rng(MANTEL_SEED),MANTEL_N)
    iu=np.triu_indices(len(gloms),1)
    pd.DataFrame(dict(a=np.array(gloms)[iu[0]],b=np.array(gloms)[iu[1]],tuning_r=sim[iu],co_convergence_z=z[iu])).to_csv(out/'tuning_vs_convergence.csv',index=False)
    p=s[s.primary].iloc[0];zp=p['ALL_dprime_z_perm']
    v1='alignment_contributes' if zp<=-1.96 else ('alignment_favourable' if zp>=1.96 else 'no_alignment_effect')
    v2='structure_itself_detrimental' if p['ALL_dprime_mwu_p_perm_lt_N0']<.05 else 'not_established'
    report=dict(perms=args.perms,perm_seeds=[PERM_SEED0+1,PERM_SEED0+args.perms],pn_count_classes={str(k):sorted(g for g in gloms if pn_count[g]==k) for k in sorted(set(pn_count.values()))},
        P1=dict(z_perm=zp,rank=p['ALL_dprime_rank_perm'],verdict=v1),P2=dict(p=p['ALL_dprime_mwu_p_perm_lt_N0'],perm_median=p['ALL_dprime_perm_median'],N0_median=p['ALL_dprime_N0_median'],verdict=v2),
        P3=dict(spearman=rho,mantel_p=p3,permutations=MANTEL_N,verdict='associated' if p3<.05 else 'not_associated'),
        max_residual=float(f.residual.max()),summary=s.to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pr=f[(f.window==.5)&(f.target==.10)];n0=ref[(ref.window==.5)&(ref.target==.10)&(ref.graph>0)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    axes[0].hist(n0.ALL_dprime,bins=20,alpha=.5,color='#999999',label='N0 random wiring, true labels')
    axes[0].hist(pr[pr.perm>0].ALL_dprime,bins=20,alpha=.6,color='#009E73',label='Real wiring, permuted labels')
    axes[0].axvline(pr[pr.perm==0].ALL_dprime.iloc[0],color='#D55E00',lw=2,label='Real wiring, true labels')
    axes[0].set_xlabel("mean pairwise d'");axes[0].legend(fontsize=7)
    axes[1].scatter(sim[iu],z[iu],s=12,color='#0072B2');axes[1].set_xlabel('Receptor tuning similarity (r)');axes[1].set_ylabel('Co-convergence Z (step 17)')
    axes[1].set_title(f'Spearman {rho:.2f}, Mantel p={p3:.3f}',fontsize=9)
    fig.suptitle('20: label permutation (0.5 s window, 10% KC)');fig.tight_layout();fig.savefig(out/'label_permutation.png',dpi=180);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('P1','P2','P3')},indent=2));print(s.T.to_string())


if __name__=='__main__':main()
