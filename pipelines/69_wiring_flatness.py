#!/usr/bin/env python3
"""Step 69: is the evolved PN->KC wiring in a flat region? Performance loss under log-normal synaptic perturbation
(odor identification, valence memory) for real, strength-preserving null and input-equalized wiring in five
hemispheres. Verdicts in docs/PREREGISTRATION_69.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p55',Path(__file__).with_name('55_memory_vs_leak.py'))
p55=_u.module_from_spec(spec);sys.modules['p55']=p55;spec.loader.exec_module(p55)
p34,p37,p24,cal=p55.p34,p55.p37,p55.p24,p55.cal;robust=cal.robust
spec=_u.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'))
p26=_u.module_from_spec(spec);spec.loader.exec_module(p26)

NAMES=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']
SEED0=690000;SIGMAS=(.1,.3,.5);N_DRAW=10;N_NULL=10;MAIN_SIGMA=.3
_D={}


def perturb(m,sigma,rng):
    """Independent mean-one log-normal factor on every nonzero synapse."""
    m=m.tocsr(copy=True);m.data=m.data*np.exp(sigma*rng.normal(size=m.nnz)-sigma**2/2);return m


def gini(x):return p26.gini(x)


def performance(m,c,g,theta,x_test):
    kt=p55.respond(m@g['x_full'],c,theta);kte=p55.respond(m@x_test,c,theta)
    nrm=np.linalg.norm(kt,axis=0);nrm[nrm==0]=1
    ident=float(np.mean(p24.identify(kte,kt/nrm)==g['lab']));mem=float(np.mean(p55.memory(kt,kte,g['lab'],g['n'],p55.N_SMALL)))
    return ident,mem


def _run(job):
    name,graph=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    if graph=='flat':m=p26.equalize_rows(m)
    elif graph!='real':
        rng=np.random.default_rng(SEED0+50000+1000*NAMES.index(name)+graph[1]);mk=g['mask']
        target,_=robust.strength_preserving_null(c['pre'][mk],c['post'][mk],c['w'][mk],rng);m,_=cal.partial_matrix(c,target)
    m=m.tocsr();xcal=cal.pn_drive(c,receptors,delta[:,calib]);theta=cal.calibrate(m@xcal,c['inh'],c['drive'],.10)
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(p55.SEED_TEST)))
    i0,m0=performance(m,c,g,theta,x_test);s=np.asarray(m.sum(1)).ravel();s=s[s>0]
    gname=graph if isinstance(graph,str) else f'null_{graph[1]}'
    base=dict(dataset=name,graph=gname,ident0=i0,mem0=m0,cv=float(s.std()/s.mean()),gini=gini(s));rows=[]
    for si,sigma in enumerate(SIGMAS):
        for k in range(N_DRAW):
            mp=perturb(m,sigma,np.random.default_rng(SEED0+100*si+k))  # same draws across graphs
            fi,fm=performance(mp,c,g,theta,x_test);ri,rm=performance(mp,c,g,cal.calibrate(mp@xcal,c['inh'],c['drive'],.10),x_test)
            rows.append(dict(base,sigma=sigma,draw=k,ident_fixed=fi,mem_fixed=fm,ident_recal=ri,mem_recal=rm))
    return rows


def summarize(f):
    g=f.groupby(['dataset','graph','sigma']).agg(ident0=('ident0','first'),mem0=('mem0','first'),cv=('cv','first'),gini=('gini','first'),
        ident_fixed=('ident_fixed','mean'),mem_fixed=('mem_fixed','mean'),ident_recal=('ident_recal','mean'),mem_recal=('mem_recal','mean')).reset_index()
    for t in ('ident','mem'):
        g[f'S_{t}']=g[f'{t}0']-g[f'{t}_fixed'];g[f'Srel_{t}']=g[f'S_{t}']/g[f'{t}0'];g[f'Srecal_{t}']=g[f'{t}0']-g[f'{t}_recal']
    return g


def verdicts(s):
    q=s[s.sigma==MAIN_SIGMA];out={};detail={}
    real=q[q.graph=='real'].set_index('dataset');flat=q[q.graph=='flat'].set_index('dataset');nl=q[q.graph.str.startswith('null')]
    for t,key in (('ident','F1'),('mem','F2')):
        med=nl.groupby('dataset')[f'S_{t}'].median();r=real[f'S_{t}'].reindex(med.index)
        lo,hi=int((r<med).sum()),int((r>med).sum());out[key]='flatter' if lo>=4 else 'sharper' if hi>=4 else 'no difference'
        fl=flat[f'S_{t}'].reindex(real.index);a,b=int((fl<real[f'S_{t}']).sum()),int((fl>real[f'S_{t}']).sum())
        out[f'F3_{t}']='equalizing flattens' if a>=4 else 'equalizing sharpens' if b>=4 else 'no difference'
        detail[t]=dict(real=r.to_dict(),null_median=med.to_dict(),flat=fl.to_dict())
    rho={}
    for t in ('ident','mem'):
        rho[t]={n:float(stats.spearmanr(qq[f'{t}0'],qq[f'S_{t}']).statistic) for n,qq in q[q.graph!='flat'].groupby('dataset')}
    F4={}
    for t,r in rho.items():
        pos=sum(v>.3 for v in r.values());neg=sum(v<-.3 for v in r.values())
        F4[t]='positive' if pos>=4 else 'negative' if neg>=4 else 'none'
    out['F4']=F4;detail['F4_rho']=rho;return out,detail


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/wiring_flatness';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_D.update(sets=sets,inputs=inputs)
    jobs=[(n,'real') for n in NAMES]+[(n,'flat') for n in NAMES]+[(n,('null',i)) for n in NAMES for i in range(N_NULL)]
    with Pool(12) as pool:res=pool.map(_run,jobs,chunksize=1)
    f=pd.DataFrame([r for x in res for r in x]);f.to_csv(out/'per_draw.csv',index=False);s=summarize(f);s.to_csv(out/'summary.csv',index=False)
    v,detail=verdicts(s)
    rep=dict(verdicts=v,detail=detail,report_only=dict(
        P0_real_vs_null={n:dict(ident_real=float(q[q.graph=='real'].ident0.iloc[0]),ident_null_median=float(q[q.graph.str.startswith('null')].ident0.median()),
                                mem_real=float(q[q.graph=='real'].mem0.iloc[0]),mem_null_median=float(q[q.graph.str.startswith('null')].mem0.median()))
                         for n,q in s[s.sigma==MAIN_SIGMA].groupby('dataset')},
        cv_gini=s[s.sigma==MAIN_SIGMA].groupby(['dataset',s.graph.where(~s.graph.str.startswith('null'),'null')])[['cv','gini']].mean().round(4).reset_index().to_dict(orient='records')))
    (out/'report.json').write_text(json.dumps(rep,indent=2,default=float)+'\n')
    pd.set_option('display.width',240);print(s[~s.graph.str.startswith('null')].round(4).to_string());print(json.dumps(rep,indent=1,default=float))


if __name__=='__main__':main()
