#!/usr/bin/env python3
"""Hallem 2006 input, Olsen 2010 ORN->PN transform, fixed KC-threshold calibration."""
import argparse
import glob
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

spec=importlib.util.spec_from_file_location('robust',Path(__file__).with_name('07_discrimination_robustness.py'))
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
base=robust.baseline

COMMIT='db323a496577c4b4a72b5c2fcd1859e07521ffb5'
OLSEN=dict(rmax=165.,sigma=12.,m=10.63,divisor=190.)
# Or33b is co-expressed in DM3 and DM5; those glomeruli take Or47a and Or85a.
EXCLUDED_FROM_GLOMERULUS=('Or33b',)
APL_GAIN=4.
SPLIT_SEED=16000
N_CAL=37


def load_hallem(root):
    cols={}
    for f in sorted(glob.glob(str(root/'data/door/receptors/*.csv'))):
        d=pd.read_csv(f,sep=';',index_col=0)
        if 'Hallem.2006.EN' in d.columns:
            cols[Path(f).stem]=d.set_index('InChIKey')['Hallem.2006.EN']
    h=pd.DataFrame(cols).dropna(how='all')
    if h.shape!=(111,24) or h.isna().any().any():raise ValueError('Unexpected Hallem matrix')
    sfr=h.loc['SFR'].astype(float);odors=h.drop(index='SFR').astype(float)
    names=pd.read_csv(root/'data/door/odor.csv',sep=';',index_col=0).drop_duplicates('InChIKey').set_index('InChIKey')['Name']
    return odors,sfr,names.reindex(odors.index)


def olsen(orn):
    """orn: receptors x trials, baseline-subtracted rates clipped at 0."""
    orn=np.maximum(orn,0);p=OLSEN
    s=p['m']*orn.sum(axis=0,keepdims=True)/p['divisor']
    return p['rmax']*orn**1.5/(orn**1.5+p['sigma']**1.5+s**1.5)


def sample_orn(delta,sfr,labels,window,rng):
    """Poisson spike-count noise in a fixed window; returns baseline-subtracted rates."""
    rate=np.maximum(delta[:,labels]+sfr[:,None],0)
    return rng.poisson(rate*window)/window-sfr[:,None]


def receptor_glomeruli(root,receptors):
    m=pd.read_csv(root/'data/door/door_mappings.csv',sep=';',index_col=0)
    m=m[m.receptor.isin(receptors)&~m.receptor.isin(EXCLUDED_FROM_GLOMERULUS)]
    if m.glomerulus.duplicated().any() or m.receptor.duplicated().any():raise ValueError('Non-unique receptor mapping')
    return dict(zip(m.glomerulus,m.receptor))


def load_partial_circuit(root,receptors):
    """Full circuit plus the PN subset mapped to Hallem receptors."""
    pre,post,w,inh,drive,info=base.load_circuit(root)
    meta=pd.read_feather(root/'data/banc_888_meta.feather');edges=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].root_888.astype(str)
    cand=meta[(meta.cell_class=='antennal_lobe_projection_neuron')&(meta.neurotransmitter_verified=='acetylcholine')].root_888.astype(str)
    pn_ids=sorted(edges[edges.pre.isin(cand)&edges.post.isin(kc)].pre.unique())
    audit=pd.read_csv(root/'qc_reports/odor_input_audit/pn_mapping_audit.csv',dtype={'root_888':str}).set_index('root_888')
    glom=receptor_glomeruli(root,receptors)
    rows=[]
    for i,p in enumerate(pn_ids):
        a=audit.loc[p];ok=a.glomerulus in glom and a.status not in ('multiple_glomeruli','annotation_prefix_conflict')
        rows.append(dict(pn_index=i,root_888=p,cell_type=a.cell_type,glomerulus=a.glomerulus,status=a.status,receptor=glom[a.glomerulus] if ok else ''))
    pns=pd.DataFrame(rows)
    return dict(pre=pre,post=post,w=w,inh=inh,drive=drive,info=info,pns=pns,mapped=pns[pns.receptor!=''].reset_index(drop=True))


def partial_matrix(c,target=None):
    sel=c['mapped'].pn_index.to_numpy();mask=np.isin(c['pre'],sel)
    post=c['post'][mask] if target is None else target
    m=sparse.csr_matrix((c['w'][mask],(post,c['pre'][mask])),shape=(len(c['inh']),c['info']['pn']))
    return m[:,sel],mask


def pn_drive(c,receptors,orn):
    """Receptor-level ORN (24 x trials) -> mapped PN input (PN/Rmax)."""
    pn=olsen(orn)/OLSEN['rmax']
    idx=[receptors.index(r) for r in c['mapped'].receptor]
    return pn[idx]


def split_odors(keys):
    order=np.random.default_rng(SPLIT_SEED).permutation(len(keys))
    cal=np.zeros(len(keys),bool);cal[order[:N_CAL]]=True
    return cal


def calibrate(excitation,inh,drive,target):
    lo,hi=0.,float(excitation.max())
    for _ in range(40):
        theta=(lo+hi)/2;k,_=base.response(excitation,inh,drive,theta,1.,gain=APL_GAIN)
        if np.mean(k>1e-8)>target:lo=theta
        else:hi=theta
    return (lo+hi)/2


def checks(excitation,inh,drive,theta):
    k,r1=base.response(excitation,inh,drive,theta,1.,gain=APL_GAIN);k0,r0=base.response(excitation,inh,drive,theta,0.,gain=APL_GAIN)
    per=np.mean(k>1e-8,axis=0)
    return dict(active_mean=float(per.mean()),active_median=float(np.median(per)),active_no_apl=float(np.mean(k0>1e-8)),
                silent_odor_fraction=float(np.mean(per==0)),residual=max(r1,r0))


def main():
    ap=argparse.ArgumentParser();ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/hallem_calibration';out.mkdir(parents=True,exist_ok=True)
    odors,sfr,names=load_hallem(root);receptors=list(odors.columns)
    c=load_partial_circuit(root,receptors);m,mask=partial_matrix(c)
    delta=odors.to_numpy().T;cal=split_odors(list(odors.index))
    x=pn_drive(c,receptors,delta);e=m@x
    result={}
    for target in (.10,.05):
        theta=calibrate(e[:,cal],c['inh'],c['drive'],target)
        result[f'{target:.2f}']=dict(theta=theta,calibration=checks(e[:,cal],c['inh'],c['drive'],theta),evaluation=checks(e[:,~cal],c['inh'],c['drive'],theta))
        ev=result[f'{target:.2f}']
        ev['pass_apl_increase']=bool(ev['evaluation']['active_no_apl']>ev['evaluation']['active_mean'])
        ev['pass_eval_range']=bool(.03<=ev['evaluation']['active_mean']<=.20)
    split=pd.DataFrame(dict(odor_key=odors.index,name=names.to_numpy(),set=np.where(cal,'calibration','evaluation'),
        max_pn_hz=(olsen(delta)).max(axis=0),max_orn_delta=delta.max(axis=0)))
    split.to_csv(out/'odor_split.csv',index=False);c['pns'].to_csv(out/'pn_mapping.csv',index=False)
    raw=sorted(glob.glob(str(root/'data/door/receptors/*.csv')))
    manifest=dict(source=dict(repository='ropensci/DoOR.data',commit=COMMIT,dataset='Hallem.2006.EN',
        receptor_csv_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in raw}),
        olsen=OLSEN,olsen_reference='Olsen, Bhandawat, Wilson 2010 Neuron 66:287, Eq. 2 and Eq. 6; m fitted in VM7',
        excluded_from_glomerulus=list(EXCLUDED_FROM_GLOMERULUS),mapped_pns=len(c['mapped']),
        mapped_glomeruli=sorted(c['mapped'].glomerulus.unique()),observed_edges=int(mask.sum()),full_edges=c['info']['pn_kc_edges'],
        split_seed=SPLIT_SEED,n_calibration=int(cal.sum()),n_evaluation=int((~cal).sum()),apl_gain=APL_GAIN,targets=result)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(mapped_pns=manifest['mapped_pns'],glomeruli=len(manifest['mapped_glomeruli']),edges=manifest['observed_edges'],targets=result),indent=2))


if __name__=='__main__':main()
