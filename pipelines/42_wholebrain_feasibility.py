#!/usr/bin/env python3
"""Feasibility of the Shiu et al. 2024 whole-brain LIF model for mushroom-body odor coding.
Three input modes (ORN with background, ORN odor-only, PN-level Olsen rates) x KC thresholds; reports KC response
fraction on blank and odor trials and the odor specificity of KC and PN patterns (calibration odors only)."""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


eng=load('eng','42_wholebrain_engine.py');cal=load('cal16','16_hallem_calibration.py')
N_ODORS=8
THRESH=(-45.,-40.,-35.,-30.,-25.)
_S={}


def setup():
    root=eng.ROOT;odors,sfr,names=cal.load_hallem(root);rec=list(odors.columns);calib=cal.split_odors(list(odors.index))
    glom=cal.receptor_glomeruli(root,rec);a,n=eng.annotations()
    orn=a[a.cell_class=='olfactory'].copy();orn['glom']=orn.cell_type.fillna('').str.replace('ORN_','',regex=False)
    pn=a[(a.cell_class=='ALPN')&(a.top_nt=='acetylcholine')].copy();pn['glom']=pn.cell_type.fillna('').str.split('_').str[0]
    pn=pn[pn.glom.isin(glom)&(pn.cell_sub_class!='multiglomerular')]
    allpn=a[(a.cell_class=='ALPN')&(a.side=='right')].idx.to_numpy()
    _S.update(rec=rec,glom=glom,sfr=sfr.to_numpy(),D=odors.to_numpy().T,odors=list(np.flatnonzero(calib)[:N_ODORS]),n=n,
              orn=orn,pn=pn,allpn=allpn,kc=a[(a.cell_class=='Kenyon_Cell')&(a.side=='right')].idx.to_numpy(),kca=a[a.cell_class=='Kenyon_Cell'].idx.to_numpy())


def orn_rates(col,background):
    s=_S;sfr_med=float(np.median(s['sfr']));r=np.full(len(s['orn']),sfr_med if background else 0.)
    for k,g in enumerate(s['orn'].glom):
        if g in s['glom']:
            i=s['rec'].index(s['glom'][g]);r[k]=max((s['sfr'][i] if background else 0.)+(col[i] if col is not None else 0.),0.)
    return r


def pn_rates(col):
    s=_S;o=cal.olsen(col[:,None])[:,0];return np.array([o[s['rec'].index(s['glom'][g])] for g in s['pn'].glom])


def job(arg):
    mode,vth=arg;s=_S
    targets=s['orn'].idx.to_numpy() if mode!='pn_level' else s['pn'].idx.to_numpy()
    sim=eng.Sim(s['n'],s['kca'],vth,targets)
    if mode=='pn_level':blank_r=np.zeros(len(targets));rates=[pn_rates(s['D'][:,o]) for o in s['odors']]
    else:bg=mode=='orn_background';blank_r=orn_rates(None,bg);rates=[orn_rates(s['D'][:,o],bg) for o in s['odors']]
    b=sim.trial(blank_r,500,1);K=[];PN=[]
    for i,r in enumerate(rates):
        c=sim.trial(r,500,10+i);K.append((c[s['kc']]>=2).astype(float));PN.append(c[s['allpn']].astype(float))
    K=np.array(K);PN=np.array(PN);iu=np.triu_indices(len(K),1)
    inp=np.corrcoef(np.array(rates))[iu]
    return dict(mode=mode,v_th_kc=vth,blank_kc_frac=float((b[s['kc']]>=2).mean()),odor_kc_frac=float(K.mean()),
                kc_pattern_corr=float(np.nanmean(np.corrcoef(K)[iu])),pn_rate_corr=float(np.nanmean(np.corrcoef(PN)[iu])),input_corr=float(np.mean(inp)))


def main():
    out=eng.ROOT/'qc_reports/wholebrain_feasibility';out.mkdir(parents=True,exist_ok=True);setup()
    jobs=[(m,v) for m in ('orn_background','orn_odor_only','pn_level') for v in THRESH]
    with Pool(8) as pool:rows=pool.map(job,jobs)
    (out/'report.json').write_text(json.dumps(dict(odors=[int(o) for o in _S['odors']],results=rows),indent=2)+'\n')
    for r in rows:print(r)


if __name__=='__main__':main()
