#!/usr/bin/env python3
"""Conflict simulation on FlyWire (right): odor A learned aversively (PPL1 compartments), odor B appetitively (PAM compartments);
present A, B and A+B, and propagate learned MBON changes to the value+body-state integrators found in step 44.

Measures (fixed before running):
- For each integrator i and odor pair, d_X = learned change of its MBON-driven input for stimulus X (A, B, A+B).
- Opposed: sign(d_A) != sign(d_B) (the neuron reads the two memories with opposite sign).
- Cancellation: opposed and |d_AB| < 0.5 * max(|d_A|, |d_B|).
- Conflict-amplifying: |d_AB| > 1.2 * max(|d_A|, |d_B|) (mixture produces a larger learned signal than either memory alone)."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p34=load('p34','34_flywire_replication.py');cal=p34.cal;p24=p34.p24
ROOT=Path(__file__).resolve().parents[1]
ETA=5.
N_PAIRS=60
SEED=450000
SIGN={'acetylcholine':1.,'gaba':-1.,'glutamate':-1.}
MIN_KC=50


def mbon_setup(a,e,kc_ids):
    ki={k:i for i,k in enumerate(kc_ids)}
    mb=a[(a.cell_class=='MBON')].copy()
    q=e[e.pre.isin(ki)&e.post.isin(mb.id)]
    nkc=q.groupby('post').pre.nunique();keep=sorted(nkc[nkc>=MIN_KC].index)
    W=np.zeros((len(keep),len(kc_ids)))
    for r,m in enumerate(keep):
        s=q[q.post==m];np.add.at(W[r],s.pre.map(ki).to_numpy(int),s['count'].to_numpy(float))
    W/=W.sum(1,keepdims=True)
    dan=a[a.cell_class=='DAN'];ppl1=set(dan[dan.cell_type.fillna('').str.startswith('PPL1')].id);pam=set(dan[dan.cell_type.fillna('').str.startswith('PAM')].id)
    d=e[e.post.isin(keep)&(e.pre.isin(ppl1)|e.pre.isin(pam))].assign(k=lambda x:np.where(x.pre.isin(ppl1),'PPL1','PAM')).pivot_table(index='post',columns='k',values='count',aggfunc='sum',fill_value=0)
    d=d.reindex(keep,fill_value=0)
    for k in ('PPL1','PAM'):
        if k not in d:d[k]=0
    comp=np.where(d.PPL1>d.PAM,'aversive',np.where(d.PAM>d.PPL1,'appetitive','none'))
    ann=a.set_index('id').reindex(keep)
    sign=ann.top_nt.map(SIGN).fillna(0.).to_numpy()
    return keep,W,comp,sign,ann.cell_type.fillna('').to_numpy()


def main():
    out=ROOT/'qc_reports/conflict_simulation';out.mkdir(parents=True,exist_ok=True)
    odors,sfr,names=cal.load_hallem(ROOT);rec=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    a,e=p34.load_flywire(ROOT);c,_,_=p34.side_circuit(a,e,'right',rec,ROOT)
    kc_ids=sorted(a[(a.cell_class=='Kenyon_Cell')&(a.side=='right')].id)
    m,_=cal.partial_matrix(c);theta=cal.calibrate(m@cal.pn_drive(c,rec,delta[:,calib]),c['inh'],c['drive'],.10)
    keep,W,comp,sign,mtypes=mbon_setup(a,e,kc_ids)
    integ=pd.read_csv(ROOT/'qc_reports/acc_like_integrators/pair_MBON_BODY.csv').id.astype(str).tolist()
    mi=e[e.pre.isin(keep)&e.post.isin(integ)].pivot_table(index='post',columns='pre',values='count',aggfunc='sum',fill_value=0).reindex(index=integ,columns=keep,fill_value=0)
    M=mi.to_numpy(float)*sign[None,:]   # signed MBON -> integrator synapse counts
    ev=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=10));rng=np.random.default_rng(SEED)
    pairs=[tuple(rng.choice(ev,2,replace=False)) for _ in range(N_PAIRS)]
    kc=lambda d:p24.kc(m,cal.pn_drive(c,rec,d[:,None]),c,theta)[0][:,0]
    rows=[]
    for A,B in pairs:
        kA,kB=kc(delta[:,A]),kc(delta[:,B]);kAB=kc(delta[:,A]+delta[:,B])
        Wl=W.copy()
        Wl[comp=='aversive']*=np.exp(-ETA*kA)[None,:]     # odor A paired with punishment
        Wl[comp=='appetitive']*=np.exp(-ETA*kB)[None,:]   # odor B paired with reward
        dM={X:(Wl-W)@k for X,k in (('A',kA),('B',kB),('AB',kAB))}  # learned change of MBON drive
        dI={X:M@v for X,v in dM.items()}
        for i,iid in enumerate(integ):
            dA,dB,dAB=dI['A'][i],dI['B'][i],dI['AB'][i];mx=max(abs(dA),abs(dB))
            rows.append(dict(odor_A=int(A),odor_B=int(B),integrator=iid,d_A=dA,d_B=dB,d_AB=dAB,
                             opposed=bool(np.sign(dA)!=np.sign(dB) and mx>0),cancel=bool(np.sign(dA)!=np.sign(dB) and mx>0 and abs(dAB)<.5*mx),
                             amplify=bool(mx>0 and abs(dAB)>1.2*mx)))
        rows.append(dict(odor_A=int(A),odor_B=int(B),integrator='__MBON_aversive_net__',d_A=float(dM['A'][comp=='aversive'].sum()),d_B=float(dM['B'][comp=='aversive'].sum()),d_AB=float(dM['AB'][comp=='aversive'].sum())))
    f=pd.DataFrame(rows);f.to_csv(out/'per_pair.csv',index=False)
    g=f[~f.integrator.str.startswith('__')].groupby('integrator').agg(opposed=('opposed','mean'),cancel=('cancel','mean'),amplify=('amplify','mean'),
        mean_abs_A=('d_A',lambda x:np.abs(x).mean()),mean_abs_B=('d_B',lambda x:np.abs(x).mean()),mean_abs_AB=('d_AB',lambda x:np.abs(x).mean()))
    ann=a.set_index('id');g['cell_type']=ann.reindex(g.index).cell_type;g['side']=ann.reindex(g.index).side
    g['mbon_inputs']=(mi!=0).sum(1).reindex(g.index);g['mbon_syn']=mi.sum(1).reindex(g.index)
    g.to_csv(out/'per_integrator.csv')
    report=dict(mbons=len(keep),mbon_compartments=pd.Series(comp).value_counts().to_dict(),mbon_signs=pd.Series(sign).value_counts().to_dict(),
                integrators=len(integ),pairs=N_PAIRS,eta=ETA,per_integrator=g.round(4).reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.set_option('display.width',220);print(json.dumps({k:report[k] for k in ('mbons','mbon_compartments','mbon_signs','integrators')},indent=2));print(g.round(4).to_string())


if __name__=='__main__':main()
