#!/usr/bin/env python3
"""Step 63: does real KC imaging (Campbell et al. 2013, Zenodo 8021893) show the hub mechanism? Correlation between a KC's
odor-response breadth (6 odors) and its activity on solvent (paraffin oil) trials, compared with the model under four
conditions (original, flat, theta_homeo, total_homeo) in seven hemispheres. Verdicts in docs/PREREGISTRATION_63.md."""
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p57',Path(__file__).with_name('57_mixed_homeostasis.py'))
p57=_u.module_from_spec(spec);sys.modules['p57']=p57;spec.loader.exec_module(p57)
spec=_u.spec_from_file_location('p58',Path(__file__).with_name('58_malecns_prediction.py'))
p58=_u.module_from_spec(spec);sys.modules['p58']=p58;spec.loader.exec_module(p58)
p55=p57.p55;p34,cal=p55.p34,p55.cal

ZIP_MD5='ba709dc3129dea4a22e33bef5c9616ce'
ODORS={'ethyl-lactate':'ethyl lactate','ethyloctanoate':'ethyl octanoate','methyloctanoate':'methyl octanoate',
       '2-heptanone':'2-heptanone','pentyl-acetate':'pentyl acetate','diethyl-succinate':'diethyl succinate'}
BLANK='paraffin-oil';N_TRIAL=6;MIN_REL=3;ALPHA=.01;BASE_WIN=(0.,7.);RESP_WIN=(7.5,9.0)
NAMES=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right','MaleCNS_right','MaleCNS_left']
CONDS=('original','flat','theta_homeo','total_homeo');N_REAL=200;N_SUB=5;N_CELL=124;N_PERM=10000
SEED0=630000;PERM_SEED=639000
_D={}


def load_imaging(root):
    d=root/'data/campbell2013'
    if hashlib.md5((d/'data_series_time_frames.zip').read_bytes()).hexdigest()!=ZIP_MD5:raise ValueError('Imaging archive changed')
    s=d/'raw/data_series';head=pd.read_csv(s/'tseries_head.csv')
    traces=np.stack([pd.read_csv(s/f'tseries_data{i+1}.csv',header=None).to_numpy(float) for i in range(len(head))])  # trial x cell x frame
    sig=pd.read_csv(s/'tseries_sigpvals.csv').to_numpy(float)  # trial x cell
    return head,traces,sig


def real_measures(head,traces,sig):
    """Per-cell breadth B (odors with >= MIN_REL of 6 significant trials), solvent activity S, S_r, baseline SD."""
    t=np.arange(traces.shape[2])*float(head.fp.iloc[0])
    bw=(t>=BASE_WIN[0])&(t<BASE_WIN[1]);rw=(t>=RESP_WIN[0])&(t<=RESP_WIN[1])
    r=traces[:,:,rw].mean(2)-traces[:,:,bw].mean(2);on=sig<ALPHA
    B=sum(((on[(head.odor==o).to_numpy()]).sum(0)>=MIN_REL).astype(int) for o in ODORS)
    blank=(head.odor==BLANK).to_numpy()
    odor_frac={o:float(on[(head.odor==o).to_numpy()].mean()) for o in ODORS}
    return dict(B=B,S=on[blank].mean(0),S_r=r[blank].mean(0),base_sd=np.median(traces[:,:,bw].std(2),axis=0),odor_frac=odor_frac,
                reliable_frac={o:float(((on[(head.odor==o).to_numpy()]).sum(0)>=MIN_REL).mean()) for o in ODORS})


def perm_test(B,S,rng,n=N_PERM):
    obs=stats.spearmanr(B,S).statistic;null=np.array([stats.spearmanr(B,rng.permutation(S)).statistic for _ in range(n)])
    return float(obs),float((1+np.sum(null>=obs))/(n+1))


def model_breadth(k,n_odor):
    """k: KC x (odor-major trials, then blank trials). Returns B, S per KC."""
    on=k>1e-8;B=sum((on[:,i*N_TRIAL:(i+1)*N_TRIAL].sum(1)>=MIN_REL).astype(int) for i in range(n_odor))
    return B,on[:,n_odor*N_TRIAL:].mean(1)


def thresholds(kind,E,En,c,theta0):
    if kind in ('original','flat'):return np.full(E.shape[0],theta0)
    s=np.asarray(_D['m'].sum(1)).ravel();tune=s>0
    if kind=='theta_homeo':
        a0=float(p55.respond(E,c,theta0)[tune].mean());th,_,_=p55.homeostasis(E,c,theta0,tune,a0,'theta')
    else:th,_=p57.mixed_homeostasis(E,En,c,theta0,tune,.01)
    return np.maximum(th+p55.offset_to_target(E,c,th),0.)


def _run(job):
    name,kind=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];idx=_D['odor_idx']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    if kind=='flat':m=_D['equalize'](m)
    _D['m']=m;E=m@cal.pn_drive(c,receptors,delta[:,calib]);theta0=cal.calibrate(E,c['inh'],c['drive'],.10)
    th=thresholds(kind,E,np.asarray(m@g['x_cal']),c,theta0)[:,None]
    d7=np.concatenate([delta[:,idx],np.zeros((delta.shape[0],1))],1);lab=np.repeat(np.arange(len(idx)+1),N_TRIAL)
    rng=np.random.default_rng(SEED0+1000*NAMES.index(name)+100*CONDS.index(kind));rhos=[];per_odor=[];blank_on=[]
    for _ in range(N_REAL):
        k=p55.respond(m@cal.pn_drive(c,receptors,cal.sample_orn(d7,sfr,lab,.5,rng)),c,th);B,S=model_breadth(k,len(idx))
        per_odor.append(float((k[:,:len(idx)*N_TRIAL]>1e-8).mean()));blank_on.append(float(S.mean()))
        for _ in range(N_SUB):
            sel=rng.choice(len(B),N_CELL,replace=False);r=stats.spearmanr(B[sel],S[sel]).statistic
            rhos.append(float(r) if np.isfinite(r) else 0.)
    return dict(dataset=name,condition=kind,rhos=rhos,odor_active=float(np.mean(per_odor)),blank_active=float(np.mean(blank_on)))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/kc_imaging_hubs';out.mkdir(parents=True,exist_ok=True)
    head,traces,sig=load_imaging(root);rm=real_measures(head,traces,sig)
    obs,p=perm_test(rm['B'],rm['S'],np.random.default_rng(PERM_SEED));obs_r,p_r=perm_test(rm['B'],rm['S_r'],np.random.default_rng(PERM_SEED+1))
    pd.DataFrame(dict(B=rm['B'],S=rm['S'],S_r=rm['S_r'],base_sd=rm['base_sd'])).to_csv(out/'real_cells.csv',index=False)
    p58.check_inputs(root);p55.p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);ms,_=p58.build(root);sets.update(ms)
    odors,_,names=cal.load_hallem(root);nm=names.reindex(odors.index).tolist()
    spec=_u.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'));p26=_u.module_from_spec(spec);spec.loader.exec_module(p26)
    _D.update(sets=sets,inputs=inputs,odor_idx=[nm.index(v) for v in ODORS.values()],equalize=p26.equalize_rows)
    with Pool(14) as pool:res=pool.map(_run,[(n,kd) for n in NAMES for kd in CONDS],chunksize=1)
    rows=[];iv={};pct={}
    for kd in CONDS:
        allr=np.concatenate([r['rhos'] for r in res if r['condition']==kd]);lo,hi=np.percentile(allr,[2.5,97.5]);iv[kd]=(float(lo),float(hi))
        pct[kd]=float(np.mean(allr<obs))  # post hoc: share of model correlations below the observed one
        for r in res:
            if r['condition']==kd:
                q=np.percentile(r['rhos'],[2.5,50,97.5]);rows.append(dict(dataset=r['dataset'],condition=kd,rho_lo=q[0],rho_med=q[1],rho_hi=q[2],
                                                                          odor_active=r['odor_active'],blank_active=r['blank_active']))
    f=pd.DataFrame(rows);f.to_csv(out/'model_rho.csv',index=False)
    v=dict(V1=bool(obs>0 and p<.05),V2={kd:bool(iv[kd][0]<=obs<=iv[kd][1]) for kd in CONDS})
    report=dict(verdicts=v,real=dict(rho_B_S=obs,p=p,rho_B_Sr=obs_r,p_Sr=p_r,n_cells=int(len(rm['B'])),B_counts=np.bincount(rm['B'],minlength=7).tolist(),
                S_mean=float(rm['S'].mean()),odor_sig_trial_frac=rm['odor_frac'],odor_reliable_frac=rm['reliable_frac'],
                rho_B_basesd=float(stats.spearmanr(rm['B'],rm['base_sd']).statistic)),
                model_pooled_interval=iv,posthoc_obs_percentile=pct,model=f.round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',220);print(f.round(3).to_string());print(json.dumps(report['real'],indent=1));print(iv,v)


if __name__=='__main__':main()
