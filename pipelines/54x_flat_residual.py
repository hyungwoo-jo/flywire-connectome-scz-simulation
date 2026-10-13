#!/usr/bin/env python3
"""Step 54 post hoc: after flattening KC input strength (P3 failed, H stays 3-5), which node property carries the remaining H?
Candidates (all per KC, mapped PN inputs): s_spont, noise SD of KC input over noise trials, input count, effective input count
(sum w)^2 / sum w^2, noise margin (s_spont - theta) / noise SD. Share of (H - 1) explained by deciles of each, as in step 54.
Conditions: flat (real composition) and rewired + flat (null 1). Not preregistered."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p54',Path(__file__).with_name('54_hub_graph_decomposition.py'))
p54=_u.module_from_spec(spec);sys.modules['p54']=p54;spec.loader.exec_module(p54)
p34,p24,p25,cal,p29=p54.p34,p54.p24,p54.p25,p54.cal,p54.p29


def _run(job):
    name,kind=job;D_=p54._D;d=D_['sets'][name];receptors,delta,sfr,calib=D_['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    if kind=='rewired_flat':
        mask=g['mask'];t,_=cal.robust.strength_preserving_null(c['pre'][mask],c['post'][mask],c['w'][mask],np.random.default_rng(p54.SEED0+p54.NAMES.index(name)*100+1))
        m,_=cal.partial_matrix(c,t)
    if kind!='original':m=p29.compensate(m,1.)
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    kt,_=p24.kc(m,g['x_full'],c,theta);kn,_=p24.kc(m,g['x_noise'],c,theta);mn=kn.mean(1);D=1-np.exp(-p25.ETA*kt)
    E=np.asarray(m@g['x_noise']);mu=E.mean(1);sd=E.std(1);mm=m.tocsr()
    cnt=np.diff(mm.indptr).astype(float);sq=np.asarray(mm.multiply(mm).sum(1)).ravel();s=np.asarray(mm.sum(1)).ravel()
    props=dict(s_spont=mu,noise_sd=sd,n_inputs=cnt,n_eff=np.where(sq>0,s**2/np.where(sq>0,sq,1),0),
               noise_margin=np.where(sd>0,(mu-theta)/np.where(sd>0,sd,1),-np.inf))
    w=g['w'];sel=w>0;out=dict(dataset=name,condition=kind)
    for k,v in props.items():
        H,_,sh=p54.shares(w[sel],mn[sel],D[sel],v[sel]);out[k]=float(np.nanmedian(sh))
    out["H_median"]=float(np.nanmedian(H));out["H_mean"]=float(np.nanmean(H));out["n_nan"]=int(np.isnan(H).sum());out['cv_s_spont']=float(mu[sel].std()/mu[sel].mean());out['cv_noise_sd']=float(sd[sel].std()/sd[sel].mean())
    return out


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/hub_graph_decomposition';p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p54.p37.build_all(root);p54._D.update(sets=sets,inputs=inputs)
    with Pool(10) as pool:rows=pool.map(_run,[(n,k) for n in p54.NAMES for k in ('original','flat','rewired_flat')])
    f=pd.DataFrame(rows);f.to_csv(out/'posthoc_flat_residual.csv',index=False)
    pd.set_option('display.width',220);print(f.round(3).to_string())


if __name__=='__main__':main()
