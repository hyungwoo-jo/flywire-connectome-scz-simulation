#!/usr/bin/env python3
"""Exploratory follow-up to 45: find 'comparator' neurons that receive from both aversive- and appetitive-compartment MBONs,
and run the same A(aversive)/B(appetitive)/A+B conflict simulation on them."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p45=load('p45','45_conflict_simulation.py');p34=p45.p34;cal=p45.cal;p24=p45.p24;ROOT=p45.ROOT
MIN_MBON_PER_SIDE=2
MIN_SYN_PER_SIDE=50


def main():
    out=ROOT/'qc_reports/conflict_simulation';odors,sfr,names=cal.load_hallem(ROOT);rec=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    a,e=p34.load_flywire(ROOT);c,_,_=p34.side_circuit(a,e,'right',rec,ROOT)
    kc_ids=sorted(a[(a.cell_class=='Kenyon_Cell')&(a.side=='right')].id)
    m,_=cal.partial_matrix(c);theta=cal.calibrate(m@cal.pn_drive(c,rec,delta[:,calib]),c['inh'],c['drive'],.10)
    keep,W,comp,sign,mtypes=p45.mbon_setup(a,e,kc_ids)
    central=set(a[a.super_class=='central'].id)-set(keep)
    q=e[e.pre.isin(keep)&e.post.isin(central)].copy();cm=dict(zip(keep,comp));q['comp']=q.pre.map(cm)
    st=q.groupby(['post','comp']).agg(n=('pre','nunique'),syn=('count','sum')).unstack(fill_value=0)
    ok=(st[('n','aversive')]>=MIN_MBON_PER_SIDE)&(st[('n','appetitive')]>=MIN_MBON_PER_SIDE)&(st[('syn','aversive')]>=MIN_SYN_PER_SIDE)&(st[('syn','appetitive')]>=MIN_SYN_PER_SIDE)
    comps=list(st[ok].index)
    mi=q[q.post.isin(comps)].pivot_table(index='post',columns='pre',values='count',aggfunc='sum',fill_value=0).reindex(index=comps,columns=keep,fill_value=0)
    M=mi.to_numpy(float)*sign[None,:]
    ev=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=10));rng=np.random.default_rng(p45.SEED)
    pairs=[tuple(rng.choice(ev,2,replace=False)) for _ in range(p45.N_PAIRS)]
    kc=lambda d:p24.kc(m,cal.pn_drive(c,rec,d[:,None]),c,theta)[0][:,0]
    rows=[]
    for A,B in pairs:
        kA,kB=kc(delta[:,A]),kc(delta[:,B]);kAB=kc(delta[:,A]+delta[:,B]);Wl=W.copy()
        Wl[comp=='aversive']*=np.exp(-p45.ETA*kA)[None,:];Wl[comp=='appetitive']*=np.exp(-p45.ETA*kB)[None,:]
        dI={X:M@((Wl-W)@k) for X,k in (('A',kA),('B',kB),('AB',kAB))}
        for i,iid in enumerate(comps):
            dA,dB,dAB=dI['A'][i],dI['B'][i],dI['AB'][i];mx=max(abs(dA),abs(dB))
            rows.append(dict(integrator=iid,d_A=dA,d_B=dB,d_AB=dAB,opposed=bool(np.sign(dA)!=np.sign(dB) and mx>0),
                             cancel=bool(np.sign(dA)!=np.sign(dB) and mx>0 and abs(dAB)<.5*mx),amplify=bool(mx>0 and abs(dAB)>1.2*mx)))
    f=pd.DataFrame(rows);g=f.groupby('integrator').agg(opposed=('opposed','mean'),cancel=('cancel','mean'),amplify=('amplify','mean'),
        mean_d_A=('d_A','mean'),mean_d_B=('d_B','mean'),mean_d_AB=('d_AB','mean'))
    ann=a.set_index('id');g['cell_type']=ann.reindex(g.index).cell_type;g['side']=ann.reindex(g.index).side
    g['n_aversive_mbons']=st.loc[g.index,('n','aversive')].to_numpy();g['n_appetitive_mbons']=st.loc[g.index,('n','appetitive')].to_numpy()
    # outputs of comparators
    olab=pd.Series('other',index=a.id);olab[a[a.cell_class=='DAN'].id]='DAN';olab[a[a.super_class=='descending'].id]='DN';olab[a[a.cell_class=='MBON'].id]='MBON';olab[a[a.cell_class=='CX'].id]='CX'
    o=e[e.pre.isin(comps)].assign(cl=lambda d:d.post.map(olab).fillna('other')).groupby('cl')['count'].sum()
    g=g.sort_values('opposed',ascending=False);g.to_csv(out/'comparators.csv')
    rep=dict(comparators=len(comps),types=ann.reindex(comps).cell_type.fillna('unknown').value_counts().head(15).to_dict(),
             opposed_mean=float(g.opposed.mean()),cancel_mean=float(g.cancel.mean()),amplify_mean=float(g.amplify.mean()),
             output_share={k:float(v/o.sum()) for k,v in o.items()},strong_cancelers=int((g.cancel>=.5).sum()),opposed_majority=int((g.opposed>=.5).sum()))
    (out/'comparators_report.json').write_text(json.dumps(rep,indent=2,default=str)+'\n')
    pd.set_option('display.width',220);print(json.dumps(rep,indent=2,default=str));print(g.head(20).round(3).to_string())


if __name__=='__main__':main()
