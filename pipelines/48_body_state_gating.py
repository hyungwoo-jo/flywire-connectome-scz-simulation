#!/usr/bin/env python3
"""Track 2: does body-state input change how often learned (false) value signals are expressed in behavior-selection neurons?

Integrators i (step 44): rate r_i = relu(V_i + B_i(a) - theta_i), V_i = signed MBON drive (synapse counts x MBON activity),
B_i(a) = a x unit x signed ascending/endocrine synapse counts onto i (tonic). unit = mean MBON drive per synapse on noise
trials, so a = 1 means body-state synapses are as active as an average MBON synapse on noise trials.
theta_i: 95th percentile of V_i + B_i(1) over noise trials with naive weights (5% firing at a = 1 before learning).

After aversive learning of odor A (or the same amount on random / input-matched KCs), for each noise trial reported as
odor present, the rate change vector dr = r_learned - r_naive is compared with the weak-A rate change (same a).
Measures (fixed before running), for a in {0, 0.5, 1, 2, 4, 8}:
- leak(a): fraction of noise trials (reported present) with cos(dr, dr_A) > 0.5 and |dr| > 0.5 |dr_A|.
- naive_fire(a): fraction of noise trials with any integrator firing before learning.
- specificity(a): leak_learned / leak_random."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import downstream_common as dc  # noqa: E402

A_LEVELS=(0.,.5,1.,2.,4.,8.)


def main():
    out=dc.ROOT/'qc_reports/body_state_gating';out.mkdir(parents=True,exist_ok=True)
    S=dc.setup();a,e=S['a'],S['e'];integ=S['integ']
    Mv=dc.signed_matrix(e,a,S['keep'],integ)
    body_ids=a[a.super_class.isin(['ascending','endocrine'])].id.tolist()
    Mb=dc.signed_matrix(e,a,body_ids,integ).sum(1)        # signed body synapse count per integrator
    W=S['W'];kn=S['kn'];rep=S['rep']
    mbon_noise=W@kn;unit=float(mbon_noise.mean())
    V0=Mv@mbon_noise                                       # integrators x trials, naive
    theta=np.percentile(V0+(Mb*unit)[:,None],95,axis=1)
    rows=[];naive=[]
    for lv in A_LEVELS:
        B=(lv*unit*Mb)[:,None];r0=np.maximum(V0+B-theta[:,None],0)
        naive.append(dict(a=lv,naive_fire=float((r0>0).any(0).mean()),naive_fire_reported=float((r0>0).any(0)[rep].mean())))
    for j,o in enumerate(S['ev']):
        kA=S['kfull'][:,j];rnd,mat=dc.control_patterns(kA,S,int(o));kw=S['kweak'][int(o)]
        Vw0=Mv@(W@kw)
        for lab,pat in (('learned',kA),('random',rnd),('matched',mat)):
            dW=dc.learned_dW(W,S['comp'],pat);Vl=Mv@((W+dW)@kn);Vwl=Mv@((W+dc.learned_dW(W,S['comp'],kA))@kw)
            for lv in A_LEVELS:
                B=(lv*unit*Mb)[:,None]
                dr=np.maximum(Vl+B-theta[:,None],0)-np.maximum(V0+B-theta[:,None],0)
                refA=(np.maximum(Vwl+B-theta[:,None],0)-np.maximum(Vw0+B-theta[:,None],0)).mean(1);nr=np.linalg.norm(refA)
                if nr==0:leak=0.
                else:
                    nd=np.linalg.norm(dr,axis=0);cos=(refA@dr)/(nr*np.maximum(nd,1e-12))
                    leak=float(np.mean(rep&(cos>.5)&(nd>.5*nr)))
                rows.append(dict(odor=int(o),control=lab,a=lv,leak=leak,ref_norm=float(nr)))
    f=pd.DataFrame(rows);f.to_csv(out/'per_odor.csv',index=False)
    s=f.groupby(['a','control']).leak.mean().unstack()
    s['specificity_vs_random']=s.learned/s.random.replace(0,np.nan);s['specificity_vs_matched']=s.learned/s.matched.replace(0,np.nan)
    s=s.join(pd.DataFrame(naive).set_index('a'))
    s.to_csv(out/'summary.csv')
    report=dict(unit=unit,integrators=len(integ),body_synapses=dict(zip(integ,Mb.tolist())),theta=dict(zip(integ,theta.tolist())),
                summary=s.round(5).reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.set_option('display.width',200);print(s.round(4).to_string());print('body synapses (signed):',np.round(Mb,0).tolist())


if __name__=='__main__':main()
