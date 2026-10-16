#!/usr/bin/env python3
"""Step 65: hub signature in Chan et al. 2026 KC imaging (GCaMP6f-nls, 20 flies, Zenodo 21821328). Per fly, Spearman of
odor-response breadth (6 Hallem odors) with solvent-trial activity and with pre-stimulus spontaneous events; compared with
the model (seven hemispheres, four conditions, KC subtype matched). Verdicts in docs/PREREGISTRATION_65.md."""
import hashlib
import io
import json
import pickle
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib import format as npfmt
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p64',Path(__file__).with_name('64_kc_spontaneous_events.py'))
p64=_u.module_from_spec(spec);sys.modules['p64']=p64;spec.loader.exec_module(p64)
p63=p64.p63;p55,p58,p34,cal=p63.p55,p63.p58,p63.p34,p63.cal
spec=_u.spec_from_file_location('dan_lib',Path(__file__).with_name('60_dan_gating_lib.py'))
lib=_u.module_from_spec(spec);spec.loader.exec_module(lib)

TRACES_MD5='0207facff58cd4721b15b95e85200c5c'
HALLEM={'EA':'ethyl acetate','PA':'pentyl acetate','EH':'ethyl hexanoate','Ace':'acetone','Hex':'1-hexanol','BB':'butyric acid'}
OIL='oil';NUM_STDS=4;MIN_TRIALS=2;BAD_MAX=15;NP_FACTOR=.7
NAMES,CONDS=p63.NAMES,p63.CONDS;N_TRIAL=3;N_BLANK=280;N_REAL=200
SEED0=650000;THRESH_SENS=(2.5,3.5)
_ALLOWED={('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),
          ('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar'),('builtins','dict'),('builtins','list'),('builtins','tuple')}
_D={}


class _Restricted(pickle.Unpickler):
    def find_class(self,mod,name):
        if (mod,name) not in _ALLOWED:raise pickle.UnpicklingError(f'blocked {mod}.{name}')
        return super().find_class(mod,name)


def load_npz(path):
    """npz reader that unpickles object arrays with numpy/builtin types only."""
    out={}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            b=z.read(n);f=io.BytesIO(b);v=npfmt.read_magic(f)
            _,_,dt=npfmt.read_array_header_1_0(f) if v==(1,0) else npfmt.read_array_header_2_0(f)
            a=_Restricted(f).load() if dt.hasobject else np.load(io.BytesIO(b),allow_pickle=False)
            out[n[:-4]]=a.item() if isinstance(a,np.ndarray) and a.dtype==object and a.shape==() else a
    return out


def normalize(tr,before,npl=None):
    """Authors' dF/F: F0 from frames [0, before]; optional neuropil subtraction (0.7) and clipping at 0."""
    f0=np.nanmean(tr[:,:before+1],1)[:,None];x=(tr-f0)/np.where(np.abs(f0)!=0,np.abs(f0),1)
    if npl is not None:
        n0=np.nanmean(npl[:,:before+1],1)[:,None];nn=np.clip((npl-n0)/np.where(np.abs(n0)!=0,np.abs(n0),1),0,None)
        x=np.clip(x-NP_FACTOR*nn,0,None)
    return x


def responds(x,before,after):
    """Authors' rule: any post-onset frame >= pre mean + 4 SD."""
    pre=x[:,:before+1];thr=np.nanmean(pre,1)+NUM_STDS*np.nanstd(pre,1)
    with np.errstate(invalid='ignore'):return (x[:,before:before+after]>=thr[:,None]).any(1)


def fly_measures(d,z=p64.Z):
    td,tn,before,after=d['tracedict'],d['tracedict_neuropil'],int(d['before_onset']),int(d['after_onset'])
    n=next(iter(next(iter(td.values())).values())).shape[0];resp={};pre=[]
    for k in td:
        for o,tr in td[k].items():
            if np.isnan(tr[0,:before+1]).sum()>BAD_MAX:continue
            resp.setdefault(o,[]).append(responds(normalize(tr,before,tn[k][o]),before,after))
            pre.append(normalize(tr,before)[:,:before])
    rel={o:np.sum(v,0)>=MIN_TRIALS for o,v in resp.items()}
    B=sum(rel[o].astype(int) for o in HALLEM if o in rel);B12=sum(rel[o].astype(int) for o in rel if o!=OIL)
    S=np.mean(resp[OIL],0) if OIL in resp else np.full(n,np.nan)
    base=np.stack(pre);ev={}
    for zz in (z,)+THRESH_SENS:ev[zz]=event_rates_nan(base,zz)
    frac={o:float(np.mean(rel[o])) for o in HALLEM if o in rel}
    return dict(n=n,B=B,B12=B12,S=S,Rp=ev[z][0],Rn=ev[z][1],sens={zz:ev[zz] for zz in THRESH_SENS},frac=frac,n_pre=base.shape[0])


def event_rates_nan(base,z):
    """64's event rule with NaN frames ignored for mu/sigma and breaking runs."""
    x=base.transpose(1,0,2).reshape(base.shape[1],-1);mu=np.nanmedian(x,1);sd=1.4826*np.nanmedian(np.abs(x-mu[:,None]),1)
    with np.errstate(invalid='ignore'):up=base>(mu+z*sd)[None,:,None];dn=base<(mu-z*sd)[None,:,None]
    return p64.runs(up,p64.MIN_RUN).sum(0)/base.shape[0],p64.runs(dn,p64.MIN_RUN).sum(0)/base.shape[0]


def rho(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<5 or np.ptp(a[ok])==0 or np.ptp(b[ok])==0:return np.nan
    return float(stats.spearmanr(a[ok],b[ok]).statistic)


def _run(job):
    name,kind=job;D=p63._D;d=D['sets'][name];receptors,delta,sfr,calib=D['inputs'];idx=D['odor_idx'];cls=_D['cls'][name]
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    if kind=='flat':m=D['equalize'](m)
    D['m']=m;E=m@cal.pn_drive(c,receptors,delta[:,calib]);theta0=cal.calibrate(E,c['inh'],c['drive'],.10)
    th=p63.thresholds(kind,E,np.asarray(m@g['x_cal']),c,theta0)[:,None]
    if len(cls)!=len(c['inh']):raise ValueError(f'KC class length mismatch in {name}')
    d7=np.concatenate([delta[:,idx],np.zeros((delta.shape[0],1))],1);no=len(idx)
    lab=np.concatenate([np.repeat(np.arange(no),N_TRIAL),np.full(N_TRIAL+N_BLANK,no)])
    pools={t:np.flatnonzero(cls==t) for t in ('ab','apbp','g')};pools['all']=np.arange(len(cls))
    rng=np.random.default_rng(SEED0+1000*NAMES.index(name)+100*CONDS.index(kind));out=[]
    for _ in range(N_REAL):
        on=p55.respond(m@cal.pn_drive(c,receptors,cal.sample_orn(d7,sfr,lab,.5,rng)),c,th)>1e-8
        B=sum((on[:,i*N_TRIAL:(i+1)*N_TRIAL].sum(1)>=MIN_TRIALS).astype(int) for i in range(no))
        S=on[:,no*N_TRIAL:(no+1)*N_TRIAL].mean(1);R=on[:,(no+1)*N_TRIAL:].mean(1);ro=[];rs=[]
        for t,n in _D['flies']:
            sel=rng.choice(pools[t],n,replace=n>len(pools[t]));ro.append(rho(B[sel],S[sel]));rs.append(rho(B[sel],R[sel]))
        out.append((float(np.nanmean(ro)),float(np.nanmean(rs)),float(on[:,:no*N_TRIAL].mean())))
    a=np.array(out);return dict(dataset=name,condition=kind,oil=a[:,0].tolist(),spont=a[:,1].tolist(),odor_active=float(a[:,2].mean()))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/kc_imaging_chan2026';out.mkdir(parents=True,exist_ok=True)
    dd=root/'data/chan2026'
    if hashlib.md5((dd/'traces.tar.xz').read_bytes()).hexdigest()!=TRACES_MD5:raise ValueError('Traces archive changed')
    flies=sorted(p.parent.name for p in (dd/'raw/traces').glob('*/traces.npz'));rows=[];cells=[]
    for fly in flies:
        r=fly_measures(load_npz(dd/f'raw/traces/{fly}/traces.npz'));t=fly.rstrip('0123456789')
        cells.append(pd.DataFrame(dict(fly=fly,type=t,B=r['B'],B12=r['B12'],S=r['S'],Rp=r['Rp'],Rn=r['Rn'])))
        rows.append(dict(fly=fly,type=t,n=r['n'],n_pre=r['n_pre'],rho_oil=rho(r['B'],r['S']),rho_sp=rho(r['B'],r['Rp']),rho_neg=rho(r['B'],r['Rn']),
                         rho_oil_B12=rho(r['B12'],r['S']),rho_sp_B12=rho(r['B12'],r['Rp']),
                         **{f'rho_sp_z{z}':rho(r['B'],r['sens'][z][0]) for z in THRESH_SENS},
                         mean_B=float(r['B'].mean()),mean_S=float(np.nanmean(r['S'])),mean_Rp=float(r['Rp'].mean()),mean_Rn=float(r['Rn'].mean()),
                         odor_frac=float(np.mean(list(r['frac'].values())))))
    f=pd.DataFrame(rows);f.to_csv(out/'per_fly.csv',index=False);pd.concat(cells).to_csv(out/'real_cells.csv',index=False)
    w=lambda x,alt:float(stats.wilcoxon(x.dropna(),alternative=alt).pvalue)
    real=dict(oil=float(f.rho_oil.mean()),p_oil=w(f.rho_oil,'greater'),spont=float(f.rho_sp.mean()),p_spont=w(f.rho_sp,'greater'),
              neg=float(f.rho_neg.mean()),p_neg=w(f.rho_neg,'two-sided'),n_flies_oil=int(f.rho_oil.notna().sum()),n_flies_sp=int(f.rho_sp.notna().sum()))
    p58.check_inputs(root);p55.p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);ms,_=p58.build(root);sets.update(ms)
    a,e=p34.load_flywire(root);gates={**lib.hemibrain(root),**lib.flywire(root,a,e),**lib.banc(root),**lib.malecns(root)};del a,e
    odors,_,names=cal.load_hallem(root);nm=names.reindex(odors.index).tolist()
    spec=_u.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'));p26=_u.module_from_spec(spec);spec.loader.exec_module(p26)
    p63._D.update(sets=sets,inputs=inputs,odor_idx=[nm.index(v) for v in HALLEM.values()],equalize=p26.equalize_rows)
    _D.update(cls={n:gates[n]['cls'] for n in NAMES},flies=list(zip(f.type,f.n)))
    with Pool(14) as pool:res=pool.map(_run,[(n,kd) for n in NAMES for kd in CONDS],chunksize=1)
    iv={};pct={};mrows=[]
    for meas,key in (('oil','oil'),('spont','spont')):
        iv[meas]={};pct[meas]={}
        for kd in CONDS:
            allr=np.concatenate([r[key] for r in res if r['condition']==kd]);lo,hi=np.percentile(allr,[2.5,97.5])
            iv[meas][kd]=(float(lo),float(hi));pct[meas][kd]=float(np.mean(allr<real[meas]))
    for r in res:
        mrows.append(dict(dataset=r['dataset'],condition=r['condition'],oil_med=float(np.median(r['oil'])),spont_med=float(np.median(r['spont'])),odor_active=r['odor_active']))
    mf=pd.DataFrame(mrows);mf.to_csv(out/'model_summary.csv',index=False)
    v=dict(X1=bool(real['oil']>0 and real['p_oil']<.05),X2=bool(real['spont']>0 and real['p_spont']<.05),X2_neg_guard_ok=bool(real['p_neg']>=.05),
           X3={meas:{kd:bool(iv[meas][kd][0]<=real[meas]<=iv[meas][kd][1]) for kd in CONDS} for meas in ('oil','spont')})
    by_type=f.groupby('type')[['rho_oil','rho_sp','rho_neg','mean_B','mean_S','mean_Rp','odor_frac']].mean()
    report=dict(verdicts=v,real=real,by_type=by_type.round(4).reset_index().to_dict(orient='records'),
                sens=dict(B12_oil=float(f.rho_oil_B12.mean()),B12_sp=float(f.rho_sp_B12.mean()),**{f'sp_z{z}':float(f[f'rho_sp_z{z}'].mean()) for z in THRESH_SENS}),
                model_interval=iv,obs_percentile=pct,model=mf.round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',220);print(f.round(3).to_string());print(by_type.round(3).to_string());print(mf.round(3).to_string())
    print(json.dumps(dict(real=real,sens=report['sens'],iv=iv,pct=pct,v=v),indent=1))


if __name__=='__main__':main()
