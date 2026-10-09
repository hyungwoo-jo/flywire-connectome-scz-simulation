#!/usr/bin/env python3
"""hemibrain replication of the core model results, and the compensation regression under three definitions in all three connectomes."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod  # picklable for multiprocessing
    spec.loader.exec_module(mod);return mod


p34=load('p34','34_flywire_replication.py');p32=p34.p32;p26=p34.p26;cal=p34.cal;p25=p34.p25

HB_PN=r'(_(ad|l)PN$)|(^M_(ad|l)PN)'
N_PERM=1000
PERM_SEED0=365000
MIN_SYN=5


def hemibrain_tables(root):
    d=root/'data/hemibrain/exported-traced-adjacencies-v1.2'
    n=pd.read_csv(d/'traced-neurons.csv');n['id']=n.bodyId.astype(str);n['type']=n.type.fillna('')
    tot=pd.read_csv(d/'traced-total-connections.csv').rename(columns={'bodyId_pre':'pre','bodyId_post':'post','weight':'count'})
    roi=pd.read_csv(d/'traced-roi-connections.csv').rename(columns={'bodyId_pre':'pre','bodyId_post':'post','weight':'count'})
    for f in (tot,roi):f['pre']=f.pre.astype(str);f['post']=f.post.astype(str)
    return n,tot,roi


def hemibrain_circuit(n,e,receptors,root):
    kc=sorted(n[n.type.str.startswith('KC')].id)
    apl=n[(n.type=='APL')].id.tolist();mb=n[(n.type=='MBON11')&n.instance.fillna('').str.endswith('_R')].id.tolist()
    if len(apl)!=1 or len(mb)!=1:raise ValueError('APL/MBON11 lookup failed')
    apl,mb=apl[0],mb[0]
    pn_all=n[n.type.str.contains(HB_PN,regex=True)]
    edges=e[e.pre.isin(pn_all.id)&e.post.isin(kc)]
    pn=sorted(edges.pre.unique());pi={x:i for i,x in enumerate(pn)};ki={x:i for i,x in enumerate(kc)}
    pre=edges.pre.map(pi).to_numpy(int);post=edges.post.map(ki).to_numpy(int);counts=edges['count'].to_numpy(float)
    inh=e[(e.pre==apl)&e.post.isin(kc)].groupby('post')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    drive=e[e.pre.isin(kc)&(e.post==apl)].groupby('pre')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    scale=np.median(np.bincount(post,weights=counts,minlength=len(kc)));inh/=inh.mean();drive/=drive.sum()
    glom=cal.receptor_glomeruli(root,receptors);ann=pn_all.set_index('id');rows=[]
    for i,p in enumerate(pn):
        t=ann.loc[p,'type'];prefix=t.split('_')[0];multi=t.startswith('M_') or '+' in prefix
        rows.append(dict(pn_index=i,root_id=p,cell_type=t,glomerulus=prefix,status='multiple_glomeruli' if multi else 'ok',
                         receptor=glom[prefix] if (prefix in glom and not multi) else ''))
    pns=pd.DataFrame(rows)
    q=e[(e.post==mb)&e.pre.isin(kc)];w=np.zeros(len(kc));np.add.at(w,q.pre.map(ki).to_numpy(int),q['count'].to_numpy(float))
    info=dict(side='right',apl=apl,mbon11=mb,pn=len(pn),kc=len(kc),pn_kc_edges=len(edges),apl_kc_edges=int(np.count_nonzero(inh)),
              kc_apl_edges=int(np.count_nonzero(drive)),mbon11_kc_inputs=int(np.count_nonzero(w)))
    return dict(pre=pre,post=post,w=counts/scale,inh=inh,drive=drive,info=info,pns=pns,mapped=pns[pns.receptor!=''].reset_index(drop=True)),w/w.sum()


def regression(edges,seed):
    post=edges.post.to_numpy();cnt=edges['count'].to_numpy(float)
    b,df=p32.slope(post,cnt);rng=np.random.default_rng(seed)
    null=np.array([p32.slope(post,rng.permutation(cnt))[0] for _ in range(N_PERM)])
    return dict(kcs=len(df),connections=len(edges),n_mean=float(df['size'].mean()),b=float(b),null_mean=float(null.mean()),
                p_less=float((1+(null<=b).sum())/(N_PERM+1)),p_greater=float((1+(null>=b).sum())/(N_PERM+1)),compensation=bool((1+(null<=b).sum())/(N_PERM+1)<.05))


def compensation_all(root,hb_n,hb_tot,hb_roi):
    out=[];k=0
    def add(ds,side,defn,edges):
        nonlocal k;k+=1;out.append(dict(dataset=ds,side=side,definition=defn,**regression(edges,PERM_SEED0+k)))
    # hemibrain
    kc=set(hb_n[hb_n.type.str.startswith('KC')].id);pn=set(hb_n[hb_n.type.str.contains(HB_PN,regex=True)].id)
    e=hb_tot[hb_tot.pre.isin(pn)&hb_tot.post.isin(kc)];add('hemibrain','right','D1',e);add('hemibrain','right','D2',e[e['count']>=MIN_SYN])
    r=hb_roi[(hb_roi.roi=='CA(R)')&hb_roi.pre.isin(pn)&hb_roi.post.isin(kc)];add('hemibrain','right','D3',r[r['count']>=MIN_SYN])
    # BANC
    meta=pd.read_feather(root/'data/banc_888_meta.feather');be=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    for side in ('right','left'):
        e,_=p32.kc_table(meta,be,side);add('BANC',side,'D1',e);add('BANC',side,'D2',e[e['count']>=MIN_SYN])
    # FlyWire
    a=pd.read_csv(root/'data/flywire_v783_neuron_annotations.tsv',sep='\t',low_memory=False);a['id']=a.root_id.astype(str)
    raw=pd.read_feather(root/'data/flywire/proofread_connections_783.feather',columns=['pre_pt_root_id','post_pt_root_id','neuropil','syn_count'])
    raw['pre']=raw.pre_pt_root_id.astype(str);raw['post']=raw.post_pt_root_id.astype(str)
    pnf=set(a[(a.cell_class=='ALPN')&(a.top_nt=='acetylcholine')].id)
    for side,npl in (('right','MB_CA_R'),('left','MB_CA_L')):
        kcs=set(a[(a.cell_class=='Kenyon_Cell')&(a.side==side)].id);sub=raw[raw.pre.isin(pnf)&raw.post.isin(kcs)]
        tot=sub.groupby(['pre','post'],as_index=False).syn_count.sum().rename(columns={'syn_count':'count'})
        ca=sub[sub.neuropil==npl].groupby(['pre','post'],as_index=False).syn_count.sum().rename(columns={'syn_count':'count'})
        add('FlyWire',side,'D1',tot);add('FlyWire',side,'D2',tot[tot['count']>=MIN_SYN]);add('FlyWire',side,'D3',ca[ca['count']>=MIN_SYN])
    return pd.DataFrame(out)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/hemibrain_replication';out.mkdir(parents=True,exist_ok=True)
    hb_n,hb_tot,hb_roi=hemibrain_tables(root)
    comp=compensation_all(root,hb_n,hb_tot,hb_roi);comp.to_csv(out/'compensation_regressions.csv',index=False)
    print(comp.round(3).to_string(),flush=True)
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    c,w_mbon=hemibrain_circuit(hb_n,hb_tot,receptors,root);c['pns'].to_csv(out/'pn_mapping.csv',index=False)
    g=p34.make_g(c,w_mbon,receptors,delta,sfr.to_numpy(),calib);m=g['m']
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    h1=cal.checks(m@cal.pn_drive(c,receptors,delta[:,~calib]),c['inh'],c['drive'],theta)
    p34.STRUCT_SEED0=360000;h2,zf=p34.structure(c,0);zf.to_csv(out/'pair_z.csv')
    h3=p34.leak(g,m,theta,p25.SEEDS['boot'])
    mn=p26.equalize_rows(m);thn=cal.calibrate(mn@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10);h4l=p34.leak(g,mn,thn,p25.SEEDS['boot'])
    ratio=float(h4l['leak_learned']/h3['leak_learned']) if h3['leak_learned']>0 else None
    model=dict(circuit=c['info'],mapped_pns=len(c['mapped']),mapped_glomeruli=int(c['mapped'].glomerulus.nunique()),theta=float(theta),
               H1=dict(**h1,passed=bool(.03<=h1['active_mean']<=.20 and h1['active_no_apl']>h1['active_mean'])),
               H2=dict(**h2,replicated=bool(h2['p']<.05)),H3=dict(**h3,replicated=bool(h3['ci'][0]>0)),
               H4=dict(original=h3['leak_learned'],equalized=h4l['leak_learned'],equalized_control=h4l['leak_control'],ratio=ratio,replicated=bool(ratio is not None and ratio<=.5)))
    hb=comp[comp.dataset=='hemibrain'];others=comp[comp.dataset!='hemibrain']
    hb_comp=bool(hb.compensation.any())
    if not hb_comp:vc='literature_not_reproduced_with_our_method'
    else:
        defs=hb[hb.compensation].definition.tolist();oth=others[others.definition.isin(defs)]
        vc='definition_difference' if oth.compensation.any() else 'dataset_difference'
    report=dict(compensation_verdict=vc,hemibrain_compensation=hb_comp,model=model,pn_rule=HB_PN,min_syn=MIN_SYN)
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(dict(compensation_verdict=vc,H1=model['H1'],H2={k:model['H2'][k] for k in ('observed_edges','null_mean','p','DM1_DM4_z','replicated')},
                          H3=model['H3'],H4=model['H4'],circuit=model['circuit'],mapped=(model['mapped_pns'],model['mapped_glomeruli'])),indent=2,default=str))


if __name__=='__main__':main()
