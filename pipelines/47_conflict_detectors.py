#!/usr/bin/env python3
"""Track 1: can candidate neurons act as conflict detectors with a single fixed threshold?

Setup: odor A learned aversively, B appetitively (step 45). For each neuron i (comparators + integrators), the learned change
of its MBON-driven input for A, B and A+B. Orientation o_i = sign of the median A+B change over training pairs.
Score s = o_i * change. A fixed threshold per neuron is chosen on 30 training pairs (maximizing balanced accuracy for
'mixture vs single memory'), then evaluated on 30 held-out pairs.

Measures (fixed before running):
- Held-out AUC of s for mixture trials vs single-memory trials, and held-out balanced accuracy of the fixed threshold.
- 'Detector' = held-out AUC >= 0.8 and balanced accuracy >= 0.75.
- Type: 'activation' if mixture drives the input up (disinhibition, detectable as firing), 'suppression' if down.
- Null: compartment labels (aversive/appetitive) of the 53 MBONs permuted 100 times (counts kept); number of detectors."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0,str(Path(__file__).resolve().parent))
import downstream_common as dc  # noqa: E402

N_PAIRS=60
N_TRAIN=30
N_NULL=100
SEED=470100


def auc(pos,neg):
    r=rankdata(np.r_[pos,neg]);n1=len(pos);return float((r[:n1].sum()-n1*(n1+1)/2)/(n1*len(neg)))


def changes(S,M,comp,pairs,kmix):
    W=S['W'];out=[]
    for (x,y),kAB in zip(pairs,kmix):
        kA,kB=S['kfull'][:,x],S['kfull'][:,y]
        Wl=W.copy();Wl[comp=='aversive']*=np.exp(-dc.ETA*kA)[None,:];Wl[comp=='appetitive']*=np.exp(-dc.ETA*kB)[None,:];dW=Wl-W
        out.append((M@(dW@kA),M@(dW@kB),M@(dW@kAB)))
    return np.array(out)  # pairs x 3 x neurons


def score(ch):
    tr,te=ch[:N_TRAIN],ch[N_TRAIN:];orient=np.sign(np.median(tr[:,2,:],axis=0));orient[orient==0]=1
    res=[]
    for i in range(ch.shape[2]):
        s_tr_mix=orient[i]*tr[:,2,i];s_tr_single=orient[i]*np.r_[tr[:,0,i],tr[:,1,i]]
        cand=np.unique(np.r_[s_tr_mix,s_tr_single]);best,th=-1,0.
        for t in cand:
            ba=.5*(np.mean(s_tr_mix>=t)+np.mean(s_tr_single<t))
            if ba>best:best,th=ba,t
        s_te_mix=orient[i]*te[:,2,i];s_te_single=orient[i]*np.r_[te[:,0,i],te[:,1,i]]
        ba_te=.5*(np.mean(s_te_mix>=th)+np.mean(s_te_single<th))
        res.append(dict(auc=auc(s_te_mix,s_te_single),bal_acc=float(ba_te),orientation='activation' if orient[i]>0 else 'suppression'))
    return res


def main():
    out=dc.ROOT/'qc_reports/conflict_detectors';out.mkdir(parents=True,exist_ok=True)
    S=dc.setup();a,e=S['a'],S['e']
    targets=S['comps']+S['integ'];M=dc.signed_matrix(e,a,S['keep'],targets)
    rng=np.random.default_rng(SEED);pairs=[tuple(rng.choice(S['kfull'].shape[1],2,replace=False)) for _ in range(N_PAIRS)]
    kmix=[S['resp'](dc.cal.pn_drive(S['c'],S['rec'],(S['delta'][:,S['ev'][x]]+S['delta'][:,S['ev'][y]])[:,None]))[:,0] for x,y in pairs]
    ch=changes(S,M,S['comp'],pairs,kmix);res=score(ch)
    ann=a.set_index('id');f=pd.DataFrame(res);f['id']=targets;f['group']=['comparator']*len(S['comps'])+['integrator']*len(S['integ'])
    f['cell_type']=ann.reindex(targets).cell_type.to_numpy();f['side']=ann.reindex(targets).side.to_numpy()
    f['detector']=(f.auc>=.8)&(f.bal_acc>=.75);f=f.sort_values('auc',ascending=False);f.to_csv(out/'per_neuron.csv',index=False)
    obs=int(f.detector.sum());nulls=[];null_top=[];null_top5=[]
    nrng=np.random.default_rng(SEED+1)
    for _ in range(N_NULL):
        pc=nrng.permutation(S['comp']);r=pd.DataFrame(score(changes(S,M,pc,pairs,kmix)));nulls.append(int(((r.auc>=.8)&(r.bal_acc>=.75)).sum()));null_top.append(float(r.auc.max()));null_top5.append(float(np.sort(r.auc)[-5:].mean()))
    nulls=np.array(nulls)
    top5=float(f.auc.head(5).mean());report=dict(top_auc=float(f.auc.max()),null_top_auc_mean=float(np.mean(null_top)),null_top_auc_p95=float(np.percentile(null_top,95)),p_top=float((1+(np.array(null_top)>=f.auc.max()).sum())/(N_NULL+1)),
                top5_auc=top5,null_top5_mean=float(np.mean(null_top5)),null_top5_p95=float(np.percentile(null_top5,95)),p_top5=float((1+(np.array(null_top5)>=top5).sum())/(N_NULL+1)),
                neurons=len(targets),detectors=obs,null_mean=float(nulls.mean()),null_p95=float(np.percentile(nulls,95)),p=float((1+(nulls>=obs).sum())/(N_NULL+1)),
                detector_types=f[f.detector].cell_type.value_counts().to_dict(),detector_orientation=f[f.detector].orientation.value_counts().to_dict(),
                top=f.head(15).round(3).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:report[k] for k in ('neurons','detectors','null_mean','p','top_auc','null_top_auc_mean','null_top_auc_p95','p_top','top5_auc','null_top5_mean','null_top5_p95','p_top5')},indent=2))
    pd.set_option('display.width',200);print(f.head(20).round(3).to_string())


if __name__=='__main__':main()
