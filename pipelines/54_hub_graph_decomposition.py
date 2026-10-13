#!/usr/bin/env python3
"""Step 54: graph decomposition of the hub factor H (step 43), five hemispheres.

H = E_w[m d] / (E_w[m] E_w[d]) = 1 + Cov_w(m, d) / (E_w[m] E_w[d]), w = MBON11 weights, m = mean noise-trial KC activity,
d = 1 - exp(-eta k_A). Hypothesis: m and d both rise with KC input strength (first-order node property), so most of H comes
from strength; which PNs a KC samples (composition, higher-order) is a small part.
1. Share explained by strength: replace m and d by their w-weighted means within KC strength deciles -> H_s;
   share = (H_s - 1) / (H - 1), median over odors; for s_raw (row sums of the mapped PN->KC matrix) and s_spont (KC input
   at the mean noise-trial PN drive). Chance: strength shuffled across KCs (100 times).
2. 2 x 2 rewiring: {real, rewired composition} x {raw, flattened strength (step 29, alpha 1)}. Rewiring = step-07
   strength-preserving swaps on mapped PN edges (KC in-strength, in-degree and PN out-strength kept), 20 per hemisphere.
   Theta recalibrated per condition; H and simulated leak (step 25) per condition.
Verdicts in docs/PREREGISTRATION_54.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402


def load(name,file):
    spec=_u.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=_u.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p43=load('p43','43_leakage_theory.py');p37=p43.p37;p34=p43.p34;p25=p43.p25;p24=p43.p24;cal=p43.cal;robust=cal.robust
p29=load('p29','29_input_compensation.py')
NAMES=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']
N_NULL=20;N_PERM=100;BINS=10;SEED0=540000;PERM_SEED=540100
_D={}


def hfac(w,m,d):
    """Hub factor; identical to step 43's L_hat / L_base."""
    wm=(w*m).sum();wd=(w*d).sum()
    return float((w*m*d).sum()*w.sum()/(wm*wd)) if wm>0 and wd>0 else np.nan


def decile_bins(s):
    return np.minimum((stats.rankdata(s,method='ordinal')/len(s)*BINS).astype(int),BINS-1) if len(s) else s


def bin_means(x,bins,w):
    """w-weighted mean of x within each bin, broadcast back; x is (KC,) or (KC, odors)."""
    out=np.zeros_like(x,dtype=float)
    for b in np.unique(bins):
        ix=bins==b;out[ix]=(w[ix,None]*x[ix]).sum(0)/w[ix].sum() if x.ndim>1 else (w[ix]*x[ix]).sum()/w[ix].sum()
    return out


def shares(w,m,D,s):
    """Per-odor H, H_s and share = (H_s - 1) / (H - 1) for strength vector s (KCs with w > 0 only)."""
    bins=decile_bins(s);mh=bin_means(m,bins,w);Dh=bin_means(D,bins,w)
    H=np.array([hfac(w,m,D[:,a]) for a in range(D.shape[1])]);Hs=np.array([hfac(w,mh,Dh[:,a]) for a in range(D.shape[1])])
    return H,Hs,(Hs-1)/(H-1)


def decomposition(name):
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m'];w=g['w']
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    kt,_=p24.kc(m,g['x_full'],c,theta);kn,_=p24.kc(m,g['x_noise'],c,theta);mn=kn.mean(1);D=1-np.exp(-p25.ETA*kt)
    s_raw=np.asarray(m.sum(1)).ravel();s_sp=np.asarray(m@g['x_noise'].mean(1)).ravel()
    sel=w>0;ws,ms,Ds=w[sel],mn[sel],D[sel]
    out=dict(dataset=name,n_kc=len(w),n_read=int(sel.sum()))
    per=[]
    for lab,s in (('raw',s_raw),('spont',s_sp)):
        H,Hs,sh=shares(ws,ms,Ds,s[sel]);rng=np.random.default_rng(PERM_SEED)
        perm=[float(np.median(shares(ws,ms,Ds,rng.permutation(s[sel]))[2])) for _ in range(N_PERM)]
        out[f'share_{lab}']=float(np.median(sh));out[f'share_{lab}_perm95']=float(np.percentile(perm,95))
        out[f'share_{lab}_perm_median']=float(np.median(perm))
        per+=[dict(dataset=name,strength=lab,odor=a,H=float(H[a]),H_s=float(Hs[a]),share=float(sh[a])) for a in range(len(H))]
        has=s>0
        out[f'cv_{lab}']=float(s[has].std()/s[has].mean());out[f'cv_{lab}_read']=float(s[sel].std()/s[sel].mean())
    top=s_sp>=np.quantile(s_sp[s_sp>0],.9)
    out.update(H_median=float(np.median(H)),noise_share_top10_spont=float(mn[top].sum()/mn.sum()),
               rho_raw_spont=float(stats.spearmanr(s_raw,s_sp).statistic))
    return out,per


def rewiring(job):
    name,kind,i=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    if kind.startswith('rewired'):
        mask=g['mask'];seed=SEED0+NAMES.index(name)*100+i
        target,_=robust.strength_preserving_null(c['pre'][mask],c['post'][mask],c['w'][mask],np.random.default_rng(seed))
        m,_=cal.partial_matrix(c,target)
    if kind.endswith('flat'):m=p29.compensate(m,1.)
    rows=p43.evaluate(name,c,g['w'],g,m_override=m,label=kind)
    return dict(dataset=name,condition=kind,null=i,H=float(np.nanmean([r['H'] for r in rows])),
                leak=float(np.mean([r['leak_learned'] for r in rows])),specific=float(np.mean([r['specific'] for r in rows])))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/hub_graph_decomposition';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    jobs=[(n,'original',0) for n in NAMES]+[(n,'flat',0) for n in NAMES]
    jobs+=[(n,k,i) for n in NAMES for k in ('rewired','rewired_flat') for i in range(1,N_NULL+1)]
    with Pool(10) as pool:
        dec=pool.map(decomposition,NAMES);rw=pd.DataFrame(pool.map(rewiring,jobs,chunksize=1))
    dt=pd.DataFrame([x[0] for x in dec]);per=pd.DataFrame([r for x in dec for r in x[1]])
    dt.to_csv(out/'decomposition.csv',index=False);per.to_csv(out/'per_odor_share.csv',index=False);rw.to_csv(out/'rewiring.csv',index=False)
    s=rw.groupby(['dataset','condition'])[['H','leak','specific']].median().unstack('condition')
    orig=s['H']['original'];ratio=dict(rewired=(s['H']['rewired']/orig).to_dict(),flat=(s['H']['flat']/orig).to_dict(),
                                       rewired_flat_H=s['H']['rewired_flat'].to_dict())
    v=dict(P1=bool(((dt.share_spont>=.5)&(dt.share_spont>dt.share_spont_perm95)).all()),
           P2='supported' if min(ratio['rewired'].values())>=.7 else ('composition_major_somewhere' if min(ratio['rewired'].values())<=.4 else 'not_supported'),
           P3=bool(max(ratio['rewired_flat_H'].values())<=2))
    report=dict(verdicts=v,decomposition=dt.round(4).to_dict(orient='records'),H_ratios=ratio,
                medians=rw.groupby(['dataset','condition'])[['H','leak','specific']].median().round(4).reset_index().to_dict(orient='records'),n_null=N_NULL,bins=BINS)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',220);print(dt.round(3).to_string());print(s.round(4).to_string());print(json.dumps(v,indent=2))


if __name__=='__main__':main()
