#!/usr/bin/env python3
"""Two digital tests downstream of the MB (FlyWire right).

Part A (leak to behavior side): after aversive learning of odor A, do noise trials produce a downstream pattern that
resembles the weak-A pattern in comparators (45x) and value+body integrators (44)? Controls: the same learning amount
placed on random KCs, and on input-strength-matched KCs (as in 38).
  - A noise trial 'leaks' if the KC detector reports odor present, and the downstream change vector has cosine > 0.5
    with the weak-A change vector and norm > 0.5 x the weak-A norm.

Part B (AND-type conflict detectors): with aversive A and appetitive B, a neuron is AND-capable for a pair if the two
memories drive it with the same sign and the weaker one is at least half the stronger; a threshold between the larger
single response and the mixture response then makes it fire only for the mixture. Report neurons that are AND-capable
in at least half of the pairs."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p45=load('p45','45_conflict_simulation.py');p38=load('p38','38_matched_control.py')
p34=p45.p34;cal=p45.cal;p24=p45.p24;ROOT=p45.ROOT
N_NOISE=500
SEED_NOISE=250002
SEED_WEAK=250003
CTRL_SEED=460000


def comparators(a,e,keep,comp):
    central=set(a[a.super_class=='central'].id)-set(keep);q=e[e.pre.isin(keep)&e.post.isin(central)].copy();q['comp']=q.pre.map(dict(zip(keep,comp)))
    st=q.groupby(['post','comp']).agg(n=('pre','nunique'),syn=('count','sum')).unstack(fill_value=0)
    ok=(st[('n','aversive')]>=2)&(st[('n','appetitive')]>=2)&(st[('syn','aversive')]>=50)&(st[('syn','appetitive')]>=50)
    return list(st[ok].index)


def signed(e,keep,targets,sign):
    mi=e[e.pre.isin(keep)&e.post.isin(targets)].pivot_table(index='post',columns='pre',values='count',aggfunc='sum',fill_value=0).reindex(index=targets,columns=keep,fill_value=0)
    return mi.to_numpy(float)*sign[None,:]


def leak_rate(dW,M,kn,rep,ref):
    """Fraction of noise trials (reported present) whose downstream change resembles the reference (weak-A) pattern."""
    D=M@(dW@kn);nr=np.linalg.norm(ref)
    if nr==0:return 0.
    cos=(ref@D)/(nr*np.maximum(np.linalg.norm(D,axis=0),1e-12));mag=np.linalg.norm(D,axis=0)/nr
    return float(np.mean(rep&(cos>.5)&(mag>.5)))


def main():
    out=ROOT/'qc_reports/leak_to_behavior';out.mkdir(parents=True,exist_ok=True)
    odors,sfr,names=cal.load_hallem(ROOT);rec=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index));sf=sfr.to_numpy()
    a,e=p34.load_flywire(ROOT);c,_,_=p34.side_circuit(a,e,'right',rec,ROOT)
    kc_ids=sorted(a[(a.cell_class=='Kenyon_Cell')&(a.side=='right')].id)
    m,_=cal.partial_matrix(c);theta=cal.calibrate(m@cal.pn_drive(c,rec,delta[:,calib]),c['inh'],c['drive'],.10)
    keep,W,comp,sign,_=p45.mbon_setup(a,e,kc_ids);av=comp=='aversive';ap=comp=='appetitive'
    comps=comparators(a,e,keep,comp);integ=pd.read_csv(ROOT/'qc_reports/acc_like_integrators/pair_MBON_BODY.csv').id.astype(str).tolist()
    Mc=signed(e,keep,comps,sign);Mi=signed(e,keep,integ,sign)
    ev=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=10))
    resp=lambda x:p24.kc(m,x,c,theta)[0]
    zero=np.zeros((24,1));kn=resp(cal.pn_drive(c,rec,cal.sample_orn(zero,sf,np.zeros(N_NOISE,int),.5,np.random.default_rng(SEED_NOISE))))
    kcal=resp(cal.pn_drive(c,rec,cal.sample_orn(zero,sf,np.zeros(1000,int),.5,np.random.default_rng(230001))))
    rep=kn.sum(0)>np.percentile(kcal.sum(0),95)
    kfull=resp(cal.pn_drive(c,rec,delta[:,ev]))
    strength=np.asarray(m.sum(1)).ravel();has=strength>0
    dec=np.full(len(strength),-1);dec[has]=np.minimum((pd.Series(strength[has]).rank(pct=True).to_numpy()*10).astype(int),9)
    # Part A
    rowsA=[]
    for j,o in enumerate(ev):
        kA=kfull[:,j];act=kA>1e-8
        kw=resp(cal.pn_drive(c,rec,cal.sample_orn(.1*delta[:,[o]],sf,np.zeros(20,int),.5,np.random.default_rng(SEED_WEAK+o))))
        def dW_of(pattern):
            Wl=W.copy();Wl[av]*=np.exp(-p45.ETA*pattern)[None,:];return Wl-W
        rng=np.random.default_rng(CTRL_SEED+o);rnd=np.zeros_like(kA);rnd[rng.choice(len(kA),int(act.sum()),replace=False)]=rng.permutation(kA[act])
        tgt,src=p38.matched_set(act&has,strength,dec,np.random.default_rng(CTRL_SEED+10000+o));mat=np.zeros_like(kA);mat[tgt]=kA[src]
        row=dict(odor=int(o))
        for lab,pat in (('learned',kA),('random',rnd),('matched',mat)):
            dW=dW_of(pat)
            for tname,M in (('comparators',Mc),('integrators',Mi)):
                ref=(M@(dW_of(kA)@kw)).mean(1)  # weak-A reference pattern under the true learning
                row[f'{tname}_{lab}']=leak_rate(dW,M,kn,rep,ref)
        rowsA.append(row)
    fa=pd.DataFrame(rowsA);fa.to_csv(out/'partA_per_odor.csv',index=False)
    A={t:{lab:float(fa[f'{t}_{lab}'].mean()) for lab in ('learned','random','matched')} for t in ('comparators','integrators')}
    # Part B
    rng=np.random.default_rng(p45.SEED);pairs=[tuple(rng.choice(len(ev),2,replace=False)) for _ in range(p45.N_PAIRS)]
    andcap=np.zeros(len(comps)+len(integ));names_=comps+integ;Mall=np.vstack([Mc,Mi])
    for x,y in pairs:
        kA,kB=kfull[:,x],kfull[:,y];kAB=resp(cal.pn_drive(c,rec,(delta[:,ev[x]]+delta[:,ev[y]])[:,None]))[:,0]
        Wl=W.copy();Wl[av]*=np.exp(-p45.ETA*kA)[None,:];Wl[ap]*=np.exp(-p45.ETA*kB)[None,:];dW=Wl-W
        dA,dB,dAB=Mall@(dW@kA),Mall@(dW@kB),Mall@(dW@kAB)
        mx=np.maximum(abs(dA),abs(dB));mn=np.minimum(abs(dA),abs(dB))
        andcap+=(np.sign(dA)==np.sign(dB))&(mx>0)&(mn>=.5*mx)&(abs(dAB)>mx)
    andcap/=len(pairs);ann=a.set_index('id')
    fb=pd.DataFrame(dict(id=names_,group=['comparator']*len(comps)+['integrator']*len(integ),and_capable=andcap,
                         cell_type=ann.reindex(names_).cell_type.to_numpy(),side=ann.reindex(names_).side.to_numpy())).sort_values('and_capable',ascending=False)
    fb.to_csv(out/'partB_and_capability.csv',index=False)
    report=dict(partA_leak_rates=A,partA_odors=len(ev),noise_reported_present=float(rep.mean()),
                partB=dict(neurons=len(names_),and_capable_majority=int((andcap>=.5).sum()),and_capable_any=int((andcap>0).sum()),
                           top=fb.head(10).round(3).to_dict(orient='records')))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str))


if __name__=='__main__':main()
