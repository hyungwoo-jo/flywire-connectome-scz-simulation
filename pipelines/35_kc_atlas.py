#!/usr/bin/env python3
"""Descriptive KC atlas across BANC, FlyWire and hemibrain: subtypes, input sources, claw proxy, KC-KC and outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CLAW_MIN_SYN=5   # Li et al. 2020 used a five-synapse threshold for uPN-to-KC connections
CATS=('PN','KC','APL','DPM','DAN','MBON','visual','other')
HB_PN=r'_(ad|l|v|lv|il)PN|^M_.*PN'
LEFT_APL_BANC='720575941734526507'


def major(sub):
    s=str(sub)
    if s.startswith("KCa'b'") or s.startswith('KCapbp'):return "a'b'"
    if s.startswith('KCab'):return 'ab'
    if s.startswith('KCg'):return 'g'
    return 'unknown'


def visual_subtype(sub):
    return str(sub) in ('KCg-d','KCab-p')


def banc(root):
    m=pd.read_feather(root/'data/banc_888_meta.feather');m['id']=m.root_888.astype(str)
    e=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')[['pre','post','count']]
    ct=m.cell_type.fillna('');cat=np.full(len(m),'other',dtype=object)
    cat[m.super_class.fillna('').str.contains('visual').to_numpy()]='visual'
    cat[(m.cell_class=='mushroom_body_dopaminergic_neuron').to_numpy()]='DAN'
    cat[ct.str.startswith('MBON').to_numpy()]='MBON';cat[(ct=='DPM').to_numpy()]='DPM';cat[(ct=='APL').to_numpy()]='APL'
    cat[(m.id==LEFT_APL_BANC).to_numpy()]='APL'  # unannotated left APL inferred in step 22
    cat[(m.cell_class=='antennal_lobe_projection_neuron').to_numpy()]='PN';cat[(m.cell_class=='kenyon_cell').to_numpy()]='KC'
    m['cat']=cat;kc=m[(m.cat=='KC')&m.side.isin(['left','right'])].copy();kc['subtype']=kc.cell_type.where(kc.cell_type.fillna('').str.startswith('KC'),kc.cell_sub_class).fillna('unknown')
    return m[['id','cat']],kc[['id','side','subtype']],e,None


def flywire(root):
    a=pd.read_csv(root/'data/flywire_v783_neuron_annotations.tsv',sep='\t',low_memory=False);a['id']=a.root_id.astype(str)
    ct=a.cell_type.fillna('');cc=a.cell_class.fillna('');cat=np.full(len(a),'other',dtype=object)
    cat[(a.super_class=='visual_projection').to_numpy()]='visual';cat[(cc=='DAN').to_numpy()]='DAN';cat[(cc=='MBON').to_numpy()]='MBON'
    cat[(ct=='DPM').to_numpy()]='DPM';cat[(ct=='APL').to_numpy()]='APL';cat[(cc=='ALPN').to_numpy()]='PN';cat[(cc=='Kenyon_Cell').to_numpy()]='KC'
    a['cat']=cat;kc=a[a.cat=='KC'].copy();kc['subtype']=kc.cell_type.fillna('unknown')
    raw=pd.read_feather(root/'data/flywire/proofread_connections_783.feather',columns=['pre_pt_root_id','post_pt_root_id','neuropil','syn_count'])
    raw['pre']=raw.pre_pt_root_id.astype(str);raw['post']=raw.post_pt_root_id.astype(str)
    e=raw.groupby(['pre','post'],as_index=False).syn_count.sum().rename(columns={'syn_count':'count'})
    calyx=raw[raw.neuropil.isin(['MB_CA_R','MB_CA_L'])].groupby(['pre','post'],as_index=False).syn_count.sum().rename(columns={'syn_count':'count'})
    return a[['id','cat']],kc[['id','side','subtype']],e,calyx


def hemibrain(root):
    d=root/'data/hemibrain/exported-traced-adjacencies-v1.2'
    n=pd.read_csv(d/'traced-neurons.csv');n['id']=n.bodyId.astype(str);t=n.type.fillna('')
    cat=np.full(len(n),'other',dtype=object)
    cat[t.str.match(r'^(PAM|PPL)').to_numpy()]='DAN';cat[t.str.startswith('MBON').to_numpy()]='MBON';cat[(t=='DPM').to_numpy()]='DPM'
    cat[(t=='APL').to_numpy()]='APL';cat[t.str.contains(HB_PN,regex=True).to_numpy()]='PN';cat[t.str.startswith('KC').to_numpy()]='KC'
    n['cat']=cat;kc=n[n.cat=='KC'].copy();kc['side']='right';kc['subtype']=kc.type
    e=pd.read_csv(d/'traced-total-connections.csv').rename(columns={'bodyId_pre':'pre','bodyId_post':'post','weight':'count'})
    e['pre']=e.pre.astype(str);e['post']=e.post.astype(str)
    r=pd.read_csv(d/'traced-roi-connections.csv');r=r[r.roi=='CA(R)'].rename(columns={'bodyId_pre':'pre','bodyId_post':'post','weight':'count'})
    r['pre']=r.pre.astype(str);r['post']=r.post.astype(str)
    return n[['id','cat']],kc[['id','side','subtype']],e,r


def per_kc(cats,kc,e,label):
    catmap=cats.set_index('id').cat
    inn=e[e.post.isin(kc.id)].copy();inn['src']=inn.pre.map(catmap).fillna('other')
    out=e[e.pre.isin(kc.id)].copy();out['dst']=out.post.map(catmap).fillna('other')
    ins=inn.pivot_table(index='post',columns='src',values='count',aggfunc='sum',fill_value=0).reindex(columns=list(CATS),fill_value=0)
    outs=out.pivot_table(index='pre',columns='dst',values='count',aggfunc='sum',fill_value=0).reindex(columns=list(CATS),fill_value=0)
    pn=inn[inn.src=='PN'];claws=pn[pn['count']>=CLAW_MIN_SYN].groupby('post').size()
    f=kc.set_index('id').copy();f['dataset']=label;f['major']=f.subtype.map(major);f['visual_subtype']=f.subtype.map(visual_subtype)
    f=f.join(ins.add_prefix('in_')).join(outs.add_prefix('out_')).fillna(0)
    f['pn_partners_5syn']=claws.reindex(f.index).fillna(0);f['pn_partners_any']=pn.groupby('post').size().reindex(f.index).fillna(0)
    f['in_total']=f[[f'in_{c}' for c in CATS]].sum(1);f['out_total']=f[[f'out_{c}' for c in CATS]].sum(1)
    return f.reset_index().rename(columns={'index':'kc'})


def summarize(f):
    rows=[]
    for (ds,side,maj),g in f.groupby(['dataset','side','major']):
        r=dict(dataset=ds,side=side,major=maj,n=len(g),visual_subtype_n=int(g.visual_subtype.sum()),
               claws_5syn_mean=float(g.pn_partners_5syn.mean()),pn_partners_any_mean=float(g.pn_partners_any.mean()),
               in_total_median=float(g.in_total.median()),out_total_median=float(g.out_total.median()))
        tin=g[[f'in_{c}' for c in CATS]].sum();tout=g[[f'out_{c}' for c in CATS]].sum()
        for c in CATS:r[f'in_share_{c}']=float(tin[f'in_{c}']/tin.sum());r[f'out_share_{c}']=float(tout[f'out_{c}']/tout.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def calyx_composition(cats,kc,calyx,label):
    if calyx is None:return None
    catmap=cats.set_index('id').cat;c=calyx[calyx.post.isin(kc.id)].copy();c['src']=c.pre.map(catmap).fillna('other')
    s=c.groupby('src')['count'].sum();s=s/s.sum()
    return dict(dataset=label,**{f'calyx_in_share_{k}':float(s.get(k,0.)) for k in CATS})


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/kc_atlas';out.mkdir(parents=True,exist_ok=True)
    frames=[];calyx_rows=[]
    for label,loader in (('BANC',banc),('FlyWire',flywire),('hemibrain',hemibrain)):
        cats,kc,e,cal=loader(root);f=per_kc(cats,kc,e,label);frames.append(f)
        c=calyx_composition(cats,kc,cal,label)
        if c:calyx_rows.append(c)
        print(label,len(kc),flush=True)
    f=pd.concat(frames,ignore_index=True);f.to_csv(out/'per_kc.csv.gz',index=False)
    s=summarize(f);s.to_csv(out/'summary_by_class.csv',index=False)
    overall=[]
    for (ds,side),g in f.groupby(['dataset','side']):
        tin=g[[f'in_{c}' for c in CATS]].sum();tout=g[[f'out_{c}' for c in CATS]].sum()
        overall.append(dict(dataset=ds,side=side,n=len(g),claws_5syn_mean=float(g.pn_partners_5syn.mean()),claws_5syn_median=float(g.pn_partners_5syn.median()),
                            **{f'in_share_{c}':float(tin[f'in_{c}']/tin.sum()) for c in CATS},**{f'out_share_{c}':float(tout[f'out_{c}']/tout.sum()) for c in CATS}))
    o=pd.DataFrame(overall);o.to_csv(out/'summary_overall.csv',index=False)
    report=dict(claw_threshold=CLAW_MIN_SYN,overall=o.round(4).to_dict(orient='records'),calyx=calyx_rows,
                literature=dict(source='Li et al. 2020 eLife 9:e62576 (hemibrain)',kcs_right=1927,claws_mean=5.6,
                                calyx_input_shares=dict(uPN=.636,KC=.119,APL=.095,MB_C1=.046)))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',250)
    print(o.round(3).to_string());print(pd.DataFrame(calyx_rows).round(3).to_string())
    print(s[['dataset','side','major','n','visual_subtype_n','claws_5syn_mean','in_share_PN','in_share_KC','in_share_APL','in_share_DAN','out_share_MBON','out_share_KC','out_share_APL']].round(3).to_string())


if __name__=='__main__':main()
