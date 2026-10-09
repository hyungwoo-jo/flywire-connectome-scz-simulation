#!/usr/bin/env python3
"""Cross-individual replication in FlyWire v783: calibration, overconvergence, value leakage, hub causality, wiring compensation."""
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


p26=load('p26','26_hub_normalization.py');p32=load('p32','32_wiring_compensation.py');com=load('com17','17_pn_community.py')
p25=p26.p25;p24=p25.p24;p23=p24.p23;cal=p24.cal;disc=p24.disc;robust=cal.robust

STRUCT_SEED0=340000
PERM_SEED0=345000
N_STRUCT=1000
N_PERM=1000
Z_EDGE=3.29
_G={}


def load_flywire(root):
    a=pd.read_csv(root/'data/flywire_v783_neuron_annotations.tsv',sep='\t',low_memory=False)
    a['id']=a.root_id.astype(str)
    e=pd.read_feather(root/'data/flywire/proofread_connections_783.feather')
    cols={c.lower():c for c in e.columns}
    pre=cols.get('pre_pt_root_id',cols.get('pre_root_id'));post=cols.get('post_pt_root_id',cols.get('post_root_id'));cnt=cols.get('syn_count',cols.get('count'))
    e=e.groupby([pre,post],as_index=False)[cnt].sum().rename(columns={pre:'pre',post:'post',cnt:'count'})
    e['pre']=e.pre.astype(str);e['post']=e.post.astype(str)
    return a,e


def side_circuit(a,e,side,receptors,root):
    kc=sorted(a[(a.cell_class=='Kenyon_Cell')&(a.side==side)].id)
    apl=a[(a.cell_type=='APL')&(a.side==side)].id.tolist()
    if len(apl)!=1:raise ValueError('Expected one APL')
    apl=apl[0]
    pn_all=a[(a.cell_class=='ALPN')&(a.top_nt=='acetylcholine')]
    edges=e[e.pre.isin(pn_all.id)&e.post.isin(kc)]
    pn=sorted(edges.pre.unique());pi={x:i for i,x in enumerate(pn)};ki={x:i for i,x in enumerate(kc)}
    pre=edges.pre.map(pi).to_numpy(int);post=edges.post.map(ki).to_numpy(int);counts=edges['count'].to_numpy(float)
    inh=e[(e.pre==apl)&e.post.isin(kc)].groupby('post')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    drive=e[e.pre.isin(kc)&(e.post==apl)].groupby('pre')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    scale=np.median(np.bincount(post,weights=counts,minlength=len(kc)));inh/=inh.mean();drive/=drive.sum()
    glom=cal.receptor_glomeruli(root,receptors);ann=pn_all.set_index('id');rows=[]
    for i,p in enumerate(pn):
        ct=str(ann.loc[p,'cell_type']);prefix=ct.split('_')[0];multi=('+' in prefix) or ann.loc[p,'cell_sub_class']=='multiglomerular'
        rows.append(dict(pn_index=i,root_id=p,cell_type=ct,glomerulus=prefix,status='multiple_glomeruli' if multi else 'ok',
                         receptor=glom[prefix] if (prefix in glom and not multi) else ''))
    pns=pd.DataFrame(rows)
    mb=a[(a.cell_type=='MBON11')&(a.side==side)].id.tolist()[0]
    q=e[(e.post==mb)&e.post.notna()&e.pre.isin(kc)];w=np.zeros(len(kc));np.add.at(w,q.pre.map(ki).to_numpy(int),q['count'].to_numpy(float))
    info=dict(side=side,apl=apl,mbon11=mb,pn=len(pn),kc=len(kc),pn_kc_edges=len(edges),min_edge_count=float(e['count'].min()),
              apl_kc_edges=int(np.count_nonzero(inh)),kc_apl_edges=int(np.count_nonzero(drive)),mbon11_kc_inputs=int(np.count_nonzero(w)))
    return dict(pre=pre,post=post,w=counts/scale,inh=inh,drive=drive,info=info,pns=pns,mapped=pns[pns.receptor!=''].reset_index(drop=True)),w/w.sum(),edges


def make_g(c,w_mbon,receptors,delta,sfr,calib):
    m,mask=cal.partial_matrix(c)
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));n=len(keep);lab=np.repeat(np.arange(n),p24.TRIALS);zero=np.zeros((24,1))
    drv=lambda dd,lb,sd:cal.pn_drive(c,receptors,cal.sample_orn(dd,sfr,lb,.5,np.random.default_rng(sd)))
    return dict(c=c,m=m,mask=mask,n=n,lab=lab,w=w_mbon,x_full=cal.pn_drive(c,receptors,delta[:,keep]),
                x_cal=drv(zero,np.zeros(1000,int),p23.SEEDS['cal']),x_noise=drv(zero,np.zeros(p24.N_NOISE,int),p25.SEEDS['noise']),
                x_weak=drv(.1*delta[:,keep],lab,p25.SEEDS['signal']),keep=keep)


def leak(g,m,theta,boot_seed):
    rows,pp=p25.evaluate(m,dict(g,theta=theta));d=np.array([r['leak_learned']-r['leak_control'] for r in rows])
    rng=np.random.default_rng(boot_seed);boot=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(1000)]
    return dict(leak_learned=float(np.mean([r['leak_learned'] for r in rows])),leak_control=float(np.mean([r['leak_control'] for r in rows])),
                specific=float(d.mean()),ci=[float(x) for x in np.percentile(boot,[2.5,97.5])],noise_FA=pp['noise_FA'],strength_rho=pp['rho'])


def _struct_null(seed):
    g=_G;t,_=robust.strength_preserving_null(g['pre'],g['post'],g['w'],np.random.default_rng(seed))
    return com.co_convergence(g['pre'],t,g['pn_type'],g['n_types'],g['n_kc'])


def structure(c,side_index):
    types=sorted(c['pns'].glomerulus.unique());tix={t:i for i,t in enumerate(types)};pn_type=c['pns'].sort_values('pn_index').glomerulus.map(tix).to_numpy()
    _G.update(pre=c['pre'],post=c['post'],w=c['w'],pn_type=pn_type,n_types=len(types),n_kc=len(c['inh']))
    obs=com.co_convergence(c['pre'],c['post'],pn_type,len(types),len(c['inh']))
    s0=STRUCT_SEED0+side_index*10000
    with Pool(10) as pool:nulls=np.stack(pool.map(_struct_null,range(s0+1,s0+N_STRUCT+1),chunksize=10))
    mu=nulls.mean(0);sd=nulls.std(0,ddof=1);z=com.zscores(obs,mu,sd);iu=np.triu_indices(len(types),1);n_edges=int((z[iu]>Z_EDGE).sum())
    n=len(nulls);tot=nulls.sum(0);tot2=(nulls**2).sum(0);pseudo=[]
    for k in range(n):
        mk=(tot-nulls[k])/(n-1);vk=np.maximum((tot2-nulls[k]**2-(n-1)*mk**2)/(n-2),0)
        pseudo.append(int((com.zscores(nulls[k],mk,np.sqrt(vk))[iu]>Z_EDGE).sum()))
    pseudo=np.array(pseudo);zf=pd.DataFrame(z,index=types,columns=types)
    top=pd.DataFrame(dict(a=np.array(types)[iu[0]],b=np.array(types)[iu[1]],z=z[iu])).sort_values('z',ascending=False).head(12)
    return dict(observed_edges=n_edges,null_mean=float(pseudo.mean()),null_p95=float(np.percentile(pseudo,95)),p=float((1+(pseudo>=n_edges).sum())/(n+1)),
                DM1_DM4_z=float(zf.loc['DM1','DM4']) if {'DM1','DM4'}<=set(types) else None,top=top.round(2).to_dict(orient='records')),zf


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/flywire_replication';out.mkdir(parents=True,exist_ok=True)
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index));sf=sfr.to_numpy()
    a,e=load_flywire(root);res={}
    for si,side in enumerate(('right','left')):
        c,w_mbon,edges=side_circuit(a,e,side,receptors,root);c['pns'].to_csv(out/f'pn_mapping_{side}.csv',index=False)
        g=make_g(c,w_mbon,receptors,delta,sf,calib);m=g['m']
        theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
        f1=cal.checks(m@cal.pn_drive(c,receptors,delta[:,~calib]),c['inh'],c['drive'],theta)
        f2,zf=structure(c,si);zf.to_csv(out/f'pair_z_{side}.csv')
        f3=leak(g,m,theta,p25.SEEDS['boot'])
        mn=p26.equalize_rows(m);thn=cal.calibrate(mn@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10);f4=leak(g,mn,thn,p25.SEEDS['boot'])
        post=edges.post.to_numpy();cnts=edges['count'].to_numpy(float);b,_=p32.slope(post,cnts)
        rng=np.random.default_rng(PERM_SEED0+si*10000);null=np.array([p32.slope(post,rng.permutation(cnts))[0] for _ in range(N_PERM)])
        f5=dict(b=b,null_mean=float(null.mean()),p_less=float((1+(null<=b).sum())/(N_PERM+1)),p_greater=float((1+(null>=b).sum())/(N_PERM+1)))
        res[side]=dict(circuit=c['info'],mapped_pns=len(c['mapped']),mapped_glomeruli=int(c['mapped'].glomerulus.nunique()),theta=float(theta),
                       F1=dict(**f1,passed=bool(.03<=f1['active_mean']<=.20 and f1['active_no_apl']>f1['active_mean'])),
                       F2=dict(**f2,replicated=bool(f2['p']<.05)),F3=dict(**f3,replicated=bool(f3['ci'][0]>0)),
                       F4=dict(original=f3['leak_learned'],equalized=f4['leak_learned'],ratio=float(f4['leak_learned']/f3['leak_learned']) if f3['leak_learned']>0 else None,
                               equalized_control=f4['leak_control'],replicated=bool(f3['leak_learned']>0 and f4['leak_learned']/f3['leak_learned']<=.5)),
                       F5=dict(**f5,compensation=bool(f5['p_less']<.05)))
        print(side,json.dumps({k:v for k,v in res[side].items() if k!='F2'},indent=1),flush=True)
    verdicts={k:('replicated' if all(res[s][k].get('replicated',res[s][k].get('passed',False)) for s in res) else 'not_replicated') for k in ('F1','F2','F3','F4')}
    verdicts['F5']='compensation_in_flywire' if all(res[s]['F5']['compensation'] for s in res) else 'no_compensation_in_flywire'
    report=dict(verdicts=verdicts,results=res,source='zenodo 10676866 proofread_connections_783.feather')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(verdicts,indent=2))


if __name__=='__main__':main()
