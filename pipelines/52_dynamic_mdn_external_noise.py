#!/usr/bin/env python3
"""Step 52: does the step-51 MDN effect survive variability from inputs outside the model?

Step 51 measured the noise-trial learned change in MDN against variability that came only from the olfactory pathway.
Each unit also receives input from neurons outside the model (MDN: about 40%). Here each unit i gets an extra
trial-to-trial input xi_i = kappa * s_i * (f_out_i / f_in_i) * N(0, 1), where s_i is the SD over naive noise trials of the
unit's in-model input (step-51 network, no external noise), f_in_i / f_out_i are the fractions of its total input
synapses from inside / outside the model (ratio capped at 20). kappa = 1 means outside synapses are as variable per
synapse as inside ones and fully correlated (a conservative, large noise); sqrt scaling would be smaller.
The same draw is used for the naive and learned run of a trial (paired). theta is re-calibrated with the noise on.

Measures (fixed before running), g = 0.9, q in {0.5, 0.95}, kappa in {0.5, 1, 2, 4}, same 30 odors and trials as 51:
- z: mean (learned - naive) MDN rate over reported noise trials, sign-aligned to the weak-A effect, / SD of naive MDN
  rate over all noise trials with external noise. Median over odors. Controls: random and matched learning.
- flip: fraction of reported noise trials with any MDN switching silent/active.
- Break-even: the smallest kappa at which median learned z < 0.5 (reported, not tested)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import downstream_common as dc  # noqa: E402

p49=dc._load('p49','49_descending_paths.py');p51=dc._load('p51','51_dynamic_mdn.py')
G=.9;QS=(.5,.95);KAPPAS=(.5,1.,2.,4.);RATIO_CAP=20.


def run(net,m,theta,ext):
    drive=net.Wm@torch.from_numpy(m.astype(np.float64))+torch.from_numpy(ext)
    th=torch.from_numpy(theta)[:,None];r=torch.zeros_like(drive)
    for _ in range(p51.STEPS):r=r+p51.DT*(-r+torch.relu(net.Wrr@r+drive-th))
    return r.numpy(),(net.Wrr@r+drive).numpy()


def main():
    out=dc.ROOT/'qc_reports/dynamic_mdn_external';out.mkdir(parents=True,exist_ok=True)
    S=dc.setup();a,e=S['a'],S['e'];keep=S['keep'];W=S['W'];comp=np.asarray(S['comp'])
    dns=a[a.super_class=='descending'].id.tolist()
    inter=sorted(set(e[e.pre.isin(keep)].post)&set(a[a.super_class=='central'].id)-set(keep))
    units=inter+dns;inside=set(units)|set(keep)
    q_=e[e.post.isin(set(units))];tot=q_.groupby('post')['count'].sum()
    fin=(q_[q_.pre.isin(inside)].groupby('post')['count'].sum()/tot).reindex(units).fillna(0.).to_numpy()
    ratio=np.minimum((1-fin)/np.maximum(fin,1e-9),RATIO_CAP)
    net=p51.Net(p49.norm_signed(e,a,units,units),p49.norm_signed(e,a,keep,units),G)
    mdn=np.flatnonzero(a.set_index('id').reindex(units).cell_type.to_numpy()=='MDN')
    kn,rep=S['kn'],S['rep'];idx_rep=np.flatnonzero(rep);kcal=kn[:,:200]
    odors=np.sort(np.random.default_rng(510001).choice(len(S['ev']),p51.N_ODORS,replace=False))
    rng=np.random.default_rng(520001);xi=rng.standard_normal((len(units),kn.shape[1]))
    rows=[];meta=[]
    for q in QS:
        theta=np.zeros(len(units))
        for _ in range(3):
            _,u=run(net,W@kcal,theta,np.zeros((len(units),kcal.shape[1])));theta=np.quantile(u,q,axis=1)
        _,u=run(net,W@kn,theta,np.zeros((len(units),kn.shape[1])));s_in=u.std(1)
        for kappa in KAPPAS:
            ext=kappa*(s_in*ratio)[:,None]*xi
            th=theta.copy()
            for _ in range(3):
                _,u=run(net,W@kcal,th,ext[:,:200]);th=np.quantile(u,q,axis=1)
            rn,_=run(net,W@kn,th,ext);sd=float(rn[mdn].mean(0).std())
            r0=rn[:,idx_rep];er=ext[:,idx_rep];base=r0[mdn].mean(0)
            meta.append(dict(q=q,kappa=kappa,mdn_sd=sd,mdn_active=float((rn[mdn]>0).any(0).mean()),mdn_ext_over_in=float(ratio[mdn].mean())))
            for j in odors:
                o=int(S['ev'][j]);kA=S['kfull'][:,j];kw=S['kweak'][o];rnd,mat=dc.control_patterns(kA,S,o)
                zw=np.zeros((len(units),kw.shape[1]));w0,_=run(net,W@kw,th,zw)
                for val in ('aversive','appetitive'):
                    w1,_=run(net,(W+dc.learned_dW(W,comp,kA,val))@kw,th,zw)
                    ref=float((w1[mdn].mean(0)-w0[mdn].mean(0)).mean());sg=np.sign(ref) if ref!=0 else 1.
                    for lab,pat in (('learned',kA),('random',rnd),('matched',mat)):
                        r1,_=run(net,(W+dc.learned_dW(W,comp,pat,val))@kn[:,idx_rep],th,er)
                        d=r1[mdn].mean(0)-base
                        rows.append(dict(q=q,kappa=kappa,odor=o,valence=val,control=lab,
                                         z=float(np.mean(d*sg)/sd) if sd>0 else np.nan,flip=float(((r1[mdn]>0)!=(r0[mdn]>0)).any(0).mean())))
            print(meta[-1],flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'per_odor.csv',index=False)
    s=f.groupby(['q','kappa','valence','control'])[['z','flip']].median().round(4)
    report=dict(g=G,odors=[int(S['ev'][j]) for j in odors],regimes=meta,summary=s.reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.set_option('display.width',200);print(s.unstack('control').to_string())


if __name__=='__main__':main()
