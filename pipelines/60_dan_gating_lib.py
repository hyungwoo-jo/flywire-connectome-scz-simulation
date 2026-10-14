"""Step 60 helpers: PPL101 (gamma1pedc DAN) -> KC synapse counts and KC subtype classes per hemisphere, in the KC order
used by the circuit builders (sorted KC ids per side)."""
import numpy as np
import pandas as pd


def kc_class(t):
    t=str(t)
    if t.startswith('KCg'):return 'g'
    if t.startswith("KCa'b'") or t.startswith('KCapbp'):return 'apbp'
    if t.startswith('KCab'):return 'ab'
    return 'other'


def _gate(edges,dans,kc):
    """Synapses from each candidate DAN onto the side's KCs; keep the DAN with most KC synapses."""
    ki={k:i for i,k in enumerate(kc)};best=None
    for d in dans:
        q=edges[(edges.pre==d)&edges.post.isin(ki)]
        g=np.zeros(len(kc));np.add.at(g,q.post.map(ki).to_numpy(int),q['count'].to_numpy(float))
        if best is None or g.sum()>best[1].sum():best=(d,g)
    return best


def hemibrain(root):
    d=root/'data/hemibrain/exported-traced-adjacencies-v1.2'
    n=pd.read_csv(d/'traced-neurons.csv');n['id']=n.bodyId.astype(str);n['type']=n.type.fillna('')
    e=pd.read_csv(d/'traced-total-connections.csv').rename(columns={'bodyId_pre':'pre','bodyId_post':'post','weight':'count'})
    e['pre']=e.pre.astype(str);e['post']=e.post.astype(str)
    kcs=n[n.type.str.startswith('KC')].sort_values('id');kc=kcs.id.tolist()
    dan,g=_gate(e,n[n.type=='PPL101'].id.tolist(),kc)
    return {'hemibrain_right':dict(gate=g,dan=dan,cls=kcs.type.map(kc_class).to_numpy())}


def flywire(root,a,e):
    out={}
    for side in ('right','left'):
        kcs=a[(a.cell_class=='Kenyon_Cell')&(a.side==side)].sort_values('id');kc=kcs.id.tolist()
        dan,g=_gate(e,a[a.cell_type=='PPL101'].id.tolist(),kc)
        out[f'FlyWire_{side}']=dict(gate=g,dan=dan,cls=kcs.cell_type.map(kc_class).to_numpy())
    return out


def banc(root):
    m=pd.read_feather(root/'data/banc_888_meta.feather');m['id']=m.root_888.astype(str)
    e=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')[['pre','post','count']]
    e['pre']=e.pre.astype(str);e['post']=e.post.astype(str);out={}
    for side in ('right','left'):
        kcs=m[(m.cell_class=='kenyon_cell')&(m.side==side)].sort_values('id');kc=kcs.id.tolist()
        dan,g=_gate(e,m[m.cell_type=='PPL101'].id.tolist(),kc)
        out[f'BANC_{side}']=dict(gate=g,dan=dan,cls=kcs.cell_sub_class.map(kc_class).to_numpy())
    return out


def malecns(root):
    d=root/'data/malecns';a=pd.read_feather(d/'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    a['id']=a.bodyId.astype(str);a['type']=a['type'].fillna('')
    dans=a[a.type=='PPL101'];kcall=a[a.type.str.startswith('KC')]
    w=pd.read_feather(d/'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
    w=w[w.body_pre.isin(set(dans.bodyId))&w.body_post.isin(set(kcall.bodyId))]
    e=pd.DataFrame(dict(pre=w.body_pre.astype(str),post=w.body_post.astype(str),count=w.weight.to_numpy()));out={}
    for side,s in (('right','R'),('left','L')):
        kcs=kcall[kcall.somaSide==s].sort_values('id');kc=kcs.id.tolist()
        dan,g=_gate(e,dans.id.tolist(),kc)
        out[f'MaleCNS_{side}']=dict(gate=g,dan=dan,cls=kcs.type.map(kc_class).to_numpy())
    return out
