#!/usr/bin/env python3
"""Replicate steps 17, 18 and 21 in the independent left mushroom body of BANC."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


disc=load('disc18','18_community_discrimination.py')
com=load('com17','17_pn_community.py')
gen=load('gen21','21_learning_generalization.py')
cal=disc.cal;base=cal.base;robust=cal.robust

LEFT_APL='720575941734526507'
SEED_STRUCT=220000
SEED_NULL=222000
Z_EDGE=3.29
_G={}


def load_side_circuit(root,side,apl_id,receptors):
    """Same construction as 06 load_circuit and 16 mapping, for a given hemisphere and APL."""
    m=pd.read_feather(root/'data/banc_888_meta.feather');e=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    m=m.set_index(m.root_888.astype(str))
    kc=sorted(m[(m.cell_class=='kenyon_cell')&(m.side==side)].index)
    pn_all=m[(m.cell_class=='antennal_lobe_projection_neuron')&(m.neurotransmitter_verified=='acetylcholine')].index
    edges=e[e.pre.isin(pn_all)&e.post.isin(kc)]
    pn=sorted(edges.pre.unique());pi={x:i for i,x in enumerate(pn)};ki={x:i for i,x in enumerate(kc)}
    pre=edges.pre.map(pi).to_numpy(int);post=edges.post.map(ki).to_numpy(int);counts=edges['count'].to_numpy(float)
    inh=e[(e.pre==apl_id)&e.post.isin(kc)].groupby('post')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    drive=e[e.pre.isin(kc)&(e.post==apl_id)].groupby('pre')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    scale=np.median(np.bincount(post,weights=counts,minlength=len(kc)));inh/=inh.mean();drive/=drive.sum()
    glom=cal.receptor_glomeruli(root,receptors);rows=[]
    for i,p in enumerate(pn):
        ct=str(m.loc[p,'cell_type']);prefix=ct.split('_')[0];fafb=str(m.loc[p,'fafb_cell_type']).split('_')[0]
        status='multiple_glomeruli' if '+' in prefix else ('annotation_prefix_conflict' if fafb not in ('nan','None') and fafb!=prefix else 'ok')
        rows.append(dict(pn_index=i,root_888=p,cell_type=ct,glomerulus=prefix,status=status,
                         receptor=glom[prefix] if (prefix in glom and status=='ok') else ''))
    pns=pd.DataFrame(rows)
    info=dict(side=side,apl=apl_id,pn=len(pn),kc=len(kc),pn_kc_edges=len(edges),apl_kc_edges=int(np.count_nonzero(inh)),kc_apl_edges=int(np.count_nonzero(drive)))
    return dict(pre=pre,post=post,w=counts/scale,inh=inh,drive=drive,info=info,pns=pns,mapped=pns[pns.receptor!=''].reset_index(drop=True))


def _struct_null(seed):
    g=_G;t,_=robust.strength_preserving_null(g['pre'],g['post'],g['w'],np.random.default_rng(seed))
    return com.co_convergence(g['pre'],t,g['pn_type'],g['n_types'],g['n_kc'])


def _func(graph):
    g=_G;c=g['c']
    if graph:
        pm,qm,wm=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
        target,_=robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(SEED_NULL+graph))
        m,_=cal.partial_matrix(c,target)
    else:m=g['m']
    rows=[]
    for cond in g['conds']:
        k,res=base.response(m@cond['x'],c['inh'],c['drive'],cond['theta'],1.,gain=cal.APL_GAIN)
        rows.append(dict(kind='disc',graph=graph,window=cond['window'],target=cond['target'],residual=res,
                         **disc.evaluate(k,g['labels'],g['train'],{'ALL':np.arange(g['n'])})))
    k,res=base.response(m@g['x_clean'],c['inh'],c['drive'],g['theta10'],1.,gain=cal.APL_GAIN)
    w=np.ones(len(c['inh']))/len(c['inh'])
    for eta in gen.ETAS:
        rows.append(dict(kind='gen',graph=graph,eta=eta,residual=res,**gen.metrics(gen.generalization(k,w,eta),g['cls'],g['sim'])))
    return rows


def zof(real,nulls):return float((real-np.mean(nulls))/np.std(nulls,ddof=1))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--struct-nulls',type=int,default=1000);ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/left_replication';out.mkdir(parents=True,exist_ok=True)
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=load_side_circuit(root,'left',LEFT_APL,receptors);c['pns'].to_csv(out/'pn_mapping.csv',index=False)
    m,mask=cal.partial_matrix(c);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    # Calibration on the same 37 calibration odors, left circuit.
    e_cal=m@cal.pn_drive(c,receptors,delta[:,calib])
    thetas={t:cal.calibrate(e_cal,c['inh'],c['drive'],t) for t in (.10,.05)}
    checks={f'{t:.2f}':dict(theta=th,evaluation=cal.checks(m@cal.pn_drive(c,receptors,delta[:,~calib]),c['inh'],c['drive'],th)) for t,th in thetas.items()}
    # R1 structure (all PN->KC edges).
    types=sorted(c['pns'].glomerulus.unique());tix={t:i for i,t in enumerate(types)}
    pn_type=c['pns'].sort_values('pn_index').glomerulus.map(tix).to_numpy()
    _G.update(pre=c['pre'],post=c['post'],w=c['w'],pn_type=pn_type,n_types=len(types),n_kc=len(c['inh']))
    observed=com.co_convergence(c['pre'],c['post'],pn_type,len(types),len(c['inh']))
    with Pool(args.workers) as pool:nulls=np.stack(pool.map(_struct_null,range(SEED_STRUCT+1,SEED_STRUCT+args.struct_nulls+1),chunksize=10))
    mu=nulls.mean(0);sd=nulls.std(0,ddof=1);z=com.zscores(observed,mu,sd);iu=np.triu_indices(len(types),1)
    n_edges=int((z[iu]>Z_EDGE).sum());n=len(nulls);tot=nulls.sum(0);tot2=(nulls**2).sum(0);pseudo=[]
    for k in range(n):
        mk=(tot-nulls[k])/(n-1);vk=np.maximum((tot2-nulls[k]**2-(n-1)*mk**2)/(n-2),0)
        pseudo.append(int((com.zscores(nulls[k],mk,np.sqrt(vk))[iu]>Z_EDGE).sum()))
    pseudo=np.array(pseudo);zf=pd.DataFrame(z,index=types,columns=types);zf.to_csv(out/'pair_z.csv')
    top=pd.DataFrame(dict(a=np.array(types)[iu[0]],b=np.array(types)[iu[1]],z=z[iu])).sort_values('z',ascending=False)
    dm=float(zf.loc['DM1','DM4']) if {'DM1','DM4'}<=set(types) else None
    R1=dict(observed_edges=n_edges,null_edges_mean=float(pseudo.mean()),null_edges_p95=float(np.percentile(pseudo,95)),
            p=float((1+(pseudo>=n_edges).sum())/(n+1)),DM1_DM4_z=dm,top_pairs=top.head(15).round(2).to_dict(orient='records'))
    R1['verdict']='replicated' if R1['p']<.05 else 'not_replicated'
    R1['strongest_pair_verdict']='replicated' if (dm is not None and dm>Z_EDGE) else 'not_replicated'
    # R2/R3 functional.
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));d=delta[:,keep];nk=len(keep)
    labels=np.repeat(np.arange(nk),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,nk)
    conds=[]
    for wi,window in enumerate((.5,.25)):
        orn=cal.sample_orn(d,sfr.to_numpy(),labels,window,np.random.default_rng(disc.NOISE_SEED+wi));x=cal.pn_drive(c,receptors,orn)
        for t in (.10,.05):conds.append(dict(window=window,target=t,theta=thetas[t],x=x))
    classes=pd.read_csv(root/'data/door/odor.csv',sep=';',index_col=0).drop_duplicates('InChIKey').set_index('InChIKey')['Class'].reindex(odors.index).to_numpy(str)
    ev=np.flatnonzero(~calib);cnt=pd.Series(classes[ev]).value_counts();gk=np.array([i for i in ev if cnt[classes[i]]>=gen.MIN_CLASS])
    _G.update(c=c,m=m,mask=mask,conds=conds,labels=labels,train=train,n=nk,x_clean=cal.pn_drive(c,receptors,delta[:,gk]),
              theta10=thetas[.10],cls=classes[gk],sim=np.corrcoef(cal.olsen(delta[:,gk]).T))
    with Pool(args.workers) as pool:rows=[r for rs in pool.map(_func,range(args.nulls+1)) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    dsum=[]
    for (window,target),g in f[f.kind=='disc'].groupby(['window','target']):
        real=g[g.graph==0].iloc[0];nl=g[g.graph>0]
        dsum.append(dict(window=window,target=target,dprime_real=float(real.ALL_dprime),dprime_null_median=float(nl.ALL_dprime.median()),
                         dprime_z=zof(real.ALL_dprime,nl.ALL_dprime),dprime_rank=float((nl.ALL_dprime>=real.ALL_dprime).mean()),
                         accuracy_z=zof(real.ALL_accuracy,nl.ALL_accuracy)))
    gsum=[]
    for eta,g in f[f.kind=='gen'].groupby('eta'):
        real=g[g.graph==0].iloc[0];nl=g[g.graph>0];r=dict(eta=eta)
        for mt in ('CI','CI_ester','fidelity','mean_GI'):r[f'{mt}_real']=float(real[mt]);r[f'{mt}_z']=zof(real[mt],nl[mt]);r[f'{mt}_rank']=float((nl[mt]>=real[mt]).mean())
        gsum.append(r)
    dsum=pd.DataFrame(dsum);gsum=pd.DataFrame(gsum);dsum.to_csv(out/'discrimination_summary.csv',index=False);gsum.to_csv(out/'generalization_summary.csv',index=False)
    z2=float(dsum[(dsum.window==.5)&(dsum.target==.10)].dprime_z.iloc[0]);z3=float(gsum[gsum.eta==5.].CI_ester_z.iloc[0])
    R2=dict(z=z2,verdict='replicated' if z2<=-1.96 else ('same_direction_ns' if z2<0 else 'not_replicated'))
    R3=dict(z=z3,CI_z=float(gsum[gsum.eta==5.].CI_z.iloc[0]),verdict='replicated' if z3>=1.96 else ('same_direction_ns' if z3>0 else 'not_replicated'))
    report=dict(circuit=c['info'],mapped_pns=len(c['mapped']),mapped_glomeruli=c['mapped'].glomerulus.value_counts().sort_index().to_dict(),
                calibration=checks,R1=R1,R2=R2,R3=R3,discrimination=dsum.to_dict(orient='records'),generalization=gsum.to_dict(orient='records'),
                odors_generalization=len(gk),max_residual=float(f.residual.max()))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(circuit=c['info'],mapped_pns=len(c['mapped']),calibration=checks,R1={k:R1[k] for k in R1 if k!='top_pairs'},R2=R2,R3=R3),indent=2))
    print(top.head(12).to_string(index=False));print(dsum.to_string());print(gsum.T.to_string())


if __name__=='__main__':main()
