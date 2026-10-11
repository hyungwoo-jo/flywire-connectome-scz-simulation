#!/usr/bin/env python3
"""Step 51: is the false (noise-trial) learned value signal large enough to change descending-neuron activity in a
nonlinear, recurrent rate network?

Network: units = central neurons receiving MBON input (step 49 intermediates) + all DNs. Connections among units and
from the 53 MBONs: synapse count / receiving neuron's total input, signed by predicted transmitter. Recurrent weights are
scaled by gain g (row sums of |w| <= 1, so g < 1 keeps the network contractive).
Dynamics: tau dr/dt = -r + relu(g Wrr r + Wm m - theta), tau = 20 ms, Euler dt = 10 ms, 40 steps (400 ms); MBON rates
m = W k are held constant (odor step). theta_i: set so each unit is active on a fraction (1 - q) of naive noise trials in
the nonlinear network itself (3 rounds of re-calibration on 200 naive noise trials).

Learning as in steps 49-50 (aversive: PPL1, appetitive: PAM compartments; controls: random and input-matched KCs).
Measures (fixed before running), per regime (q in {0.5, 0.95}, g in {0.5, 0.9}), 30 evaluation odors (seed 510001):
- MDN = mean rate of the 4 MDNs. ref = learned - naive MDN rate on weak-A trials (mean of 20).
- z_noise: (learned - naive) MDN rate on noise trials reported present, divided by the SD of naive MDN rate over
  noise trials. Reported as mean |z| and mean signed z (sign aligned to ref). |z| >= 1 = comparable to natural variability.
- ratio: mean over reported noise trials of (learned - naive) * sign(ref) / |ref|.
- flip: fraction of reported noise trials in which any MDN switches between silent and active.
- per-odor values summarized by the median over odors; all for learned, random and matched learning; time course of the learned z at 50, 100, 200, 400 ms."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parent))
import downstream_common as dc  # noqa: E402

p49=dc._load('p49','49_descending_paths.py')
REGIMES=[(q,g) for q in (.5,.95) for g in (.5,.9)]
N_ODORS=30;STEPS=40;DT=.5;REC_STEPS=(5,10,20,40)          # dt/tau = 0.5 -> 10 ms steps
torch.set_num_threads(16)


def to_t(mat):
    c=sparse.csr_matrix(mat,dtype=np.float64)
    return torch.sparse_csr_tensor(torch.from_numpy(c.indptr.astype(np.int64)),torch.from_numpy(c.indices.astype(np.int64)),
                                   torch.from_numpy(c.data),size=c.shape)


class Net:
    def __init__(self,Wrr,Wm,g):
        self.Wrr=to_t(Wrr*g);self.Wm=to_t(Wm)

    def run(self,m,theta,record=False):
        drive=self.Wm@torch.from_numpy(m.astype(np.float64))
        th=torch.from_numpy(theta.astype(np.float64))[:,None]
        r=torch.zeros_like(drive);trace={}
        for s in range(1,STEPS+1):
            r=r+DT*(-r+torch.relu(self.Wrr@r+drive-th))
            if record and s in REC_STEPS:trace[s*10]=r.numpy().copy()
        u=(self.Wrr@r+drive).numpy()
        return (r.numpy(),u,trace) if record else (r.numpy(),u)


def main():
    out=dc.ROOT/'qc_reports/dynamic_mdn';out.mkdir(parents=True,exist_ok=True)
    S=dc.setup();a,e=S['a'],S['e'];keep=S['keep'];W=S['W'];comp=np.asarray(S['comp'])
    dns=a[a.super_class=='descending'].id.tolist()
    inter=sorted(set(e[e.pre.isin(keep)].post)&set(a[a.super_class=='central'].id)-set(keep))
    units=inter+dns
    Wrr=p49.norm_signed(e,a,units,units);Wm=p49.norm_signed(e,a,keep,units)
    mdn=np.flatnonzero(a.set_index('id').reindex(units).cell_type.to_numpy()=='MDN')
    kn,rep=S['kn'],S['rep'];krep=kn[:,rep];kcal=kn[:,:200]
    odors=np.sort(np.random.default_rng(510001).choice(len(S['ev']),N_ODORS,replace=False))
    rows=[];trows=[];meta=[]
    for q,g in REGIMES:
        t0=time.time();net=Net(Wrr,Wm,g)
        theta=np.full(len(units),0.)
        for _ in range(3):
            _,u=net.run(W@kcal,theta);theta=np.quantile(u,q,axis=1)
        r_cal,_=net.run(W@kcal,theta)
        r0,_=net.run(W@krep,theta);base=r0[mdn].mean(0);rn,_=net.run(W@kn,theta);sd=float(rn[mdn].mean(0).std())
        meta.append(dict(q=q,g=g,unit_active=float((r_cal>0).mean()),mdn_active=float((r_cal[mdn]>0).any(0).mean()),
                         mdn_sd=sd,mdn_mean=float(rn[mdn].mean()),reported=int(rep.sum())))
        for j in odors:
            o=int(S['ev'][j]);kA=S['kfull'][:,j];kw=S['kweak'][o];rnd,mat=dc.control_patterns(kA,S,o)
            w0,_=net.run(W@kw,theta)
            for val in ('aversive','appetitive'):
                dWA=dc.learned_dW(W,comp,kA,val);w1,_=net.run((W+dWA)@kw,theta)
                ref=float((w1[mdn].mean(0)-w0[mdn].mean(0)).mean());sg=np.sign(ref) if ref!=0 else 1.
                for lab,pat in (('learned',kA),('random',rnd),('matched',mat)):
                    rec=lab=='learned'
                    res=net.run((W+dc.learned_dW(W,comp,pat,val))@krep,theta,record=rec)
                    r1=res[0];d=r1[mdn].mean(0)-base
                    flip=((r1[mdn]>0)!=(r0[mdn]>0)).any(0)
                    rows.append(dict(q=q,g=g,odor=o,valence=val,control=lab,ref=ref,
                                     z_abs=float(np.mean(np.abs(d))/sd) if sd>0 else np.nan,
                                     z_signed=float(np.mean(d*sg)/sd) if sd>0 else np.nan,
                                     ratio=float(np.mean(d*sg)/abs(ref)) if ref!=0 else np.nan,flip=float(flip.mean())))
                    if rec:
                        _,_,tr0=net.run(W@krep,theta,record=True)
                        for t,x in res[2].items():
                            trows.append(dict(q=q,g=g,odor=o,valence=val,t_ms=t,z_abs=float(np.mean(np.abs(x[mdn].mean(0)-tr0[t][mdn].mean(0)))/sd) if sd>0 else np.nan))
        print(f'regime q={q} g={g} done in {time.time()-t0:.0f}s',meta[-1],flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'per_odor.csv',index=False)
    tf=pd.DataFrame(trows);tf.to_csv(out/'time_course.csv',index=False)
    s=f.groupby(['q','g','valence','control'])[['z_abs','z_signed','ratio','flip']].median().round(4)
    ts=tf.groupby(['q','g','valence','t_ms']).z_abs.median().unstack().round(4)
    report=dict(units=len(units),intermediates=len(inter),dns=len(dns),odors=[int(S['ev'][j]) for j in odors],regimes=meta,
                summary=s.reset_index().to_dict(orient='records'),time_course=ts.reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.set_option('display.width',200);print(s.to_string());print(ts.to_string())


if __name__=='__main__':main()
