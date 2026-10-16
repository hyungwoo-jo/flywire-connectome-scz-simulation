#!/usr/bin/env python3
"""Step 64: spontaneous calcium events in the pre-stimulus window of Campbell et al. 2013 KC imaging, related to odor-response
breadth; model spontaneous activity from 600 odorless trials under four conditions in seven hemispheres.
Verdicts in docs/PREREGISTRATION_64.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p63',Path(__file__).with_name('63_kc_imaging_hubs.py'))
p63=_u.module_from_spec(spec);sys.modules['p63']=p63;spec.loader.exec_module(p63)
p55,p58,p34,cal=p63.p55,p63.p58,p63.p34,p63.cal
NAMES,CONDS=p63.NAMES,p63.CONDS;N_BLANK=600;Z=3.;MIN_RUN=2;SENS=((2.5,2),(3.5,2),(3.,1))
SEED0=640000;PERM_SEED=649000


def runs(mask,min_run):
    """Count runs of >= min_run consecutive True values along the last axis; returns an array over the other axes."""
    pad=np.zeros(mask.shape[:-1]+(1,),bool);d=np.diff(np.concatenate([pad,mask,pad],-1).astype(int),axis=-1)
    out=np.zeros(mask.shape[:-1],int)
    for idx in np.ndindex(mask.shape[:-1]):
        out[idx]=int(np.sum(np.flatnonzero(d[idx]==-1)-np.flatnonzero(d[idx]==1)>=min_run))
    return out


def event_rates(base,z=Z,min_run=MIN_RUN):
    """base: trial x cell x frame (pre-stimulus). Per-cell positive and negative event counts per trial."""
    x=base.transpose(1,0,2).reshape(base.shape[1],-1);mu=np.median(x,1);sd=1.4826*np.median(np.abs(x-mu[:,None]),1)
    up=base>(mu+z*sd)[None,:,None];dn=base<(mu-z*sd)[None,:,None]
    return runs(up,min_run).sum(0)/base.shape[0],runs(dn,min_run).sum(0)/base.shape[0]


def baseline(head,traces):
    t=np.arange(traces.shape[2])*float(head.fp.iloc[0]);return traces[:,:,(t>=p63.BASE_WIN[0])&(t<p63.BASE_WIN[1])]


def perm_two_sided(B,R,rng,n=p63.N_PERM):
    obs=stats.spearmanr(B,R).statistic;null=np.array([stats.spearmanr(B,rng.permutation(R)).statistic for _ in range(n)])
    return float(obs),float((1+np.sum(np.abs(null)>=abs(obs)))/(n+1))


def _run(job):
    name,kind=job;D=p63._D;d=D['sets'][name];receptors,delta,sfr,calib=D['inputs'];idx=D['odor_idx']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    if kind=='flat':m=D['equalize'](m)
    D['m']=m;E=m@cal.pn_drive(c,receptors,delta[:,calib]);theta0=cal.calibrate(E,c['inh'],c['drive'],.10)
    th=p63.thresholds(kind,E,np.asarray(m@g['x_cal']),c,theta0)[:,None]
    d7=np.concatenate([delta[:,idx],np.zeros((delta.shape[0],1))],1)
    lab=np.concatenate([np.repeat(np.arange(len(idx)),p63.N_TRIAL),np.full(N_BLANK,len(idx))])
    rng=np.random.default_rng(SEED0+1000*NAMES.index(name)+100*CONDS.index(kind));rhos=[];blank_on=[]
    for _ in range(p63.N_REAL):
        k=p55.respond(m@cal.pn_drive(c,receptors,cal.sample_orn(d7,sfr,lab,.5,rng)),c,th);B,S=p63.model_breadth(k,len(idx))
        blank_on.append(float(S.mean()))
        for _ in range(p63.N_SUB):
            sel=rng.choice(len(B),p63.N_CELL,replace=False);r=stats.spearmanr(B[sel],S[sel]).statistic
            rhos.append(float(r) if np.isfinite(r) else 0.)
    return dict(dataset=name,condition=kind,rhos=rhos,blank_active=float(np.mean(blank_on)))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/kc_spontaneous_events';out.mkdir(parents=True,exist_ok=True)
    head,traces,sig=p63.load_imaging(root);rm=p63.real_measures(head,traces,sig);base=baseline(head,traces)
    B=rm['B'];pos,neg=event_rates(base)
    obs,p=p63.perm_test(B,pos,np.random.default_rng(PERM_SEED));obs_n,p_n=perm_two_sided(B,neg,np.random.default_rng(PERM_SEED+1))
    keep=B<4;obs_k,p_k=p63.perm_test(B[keep],pos[keep],np.random.default_rng(PERM_SEED+2))
    sens={}
    for i,(z,mr) in enumerate(SENS):
        a,b=event_rates(base,z,mr);sens[f'z{z}_run{mr}']=dict(rho_pos=float(stats.spearmanr(B,a).statistic),rho_neg=float(stats.spearmanr(B,b).statistic),
                                                             mean_pos=float(a.mean()),mean_neg=float(b.mean()))
    pd.DataFrame(dict(B=B,S=rm['S'],R_pos=pos,R_neg=neg)).to_csv(out/'real_cells.csv',index=False)
    p58.check_inputs(root);p55.p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);ms,_=p58.build(root);sets.update(ms)
    odors,_,names=cal.load_hallem(root);nm=names.reindex(odors.index).tolist()
    spec=_u.spec_from_file_location('p26',Path(__file__).with_name('26_hub_normalization.py'));p26=_u.module_from_spec(spec);spec.loader.exec_module(p26)
    p63._D.update(sets=sets,inputs=inputs,odor_idx=[nm.index(v) for v in p63.ODORS.values()],equalize=p26.equalize_rows)
    with Pool(14) as pool:res=pool.map(_run,[(n,kd) for n in NAMES for kd in CONDS],chunksize=1)
    rows=[];iv={};pct={}
    for kd in CONDS:
        allr=np.concatenate([r['rhos'] for r in res if r['condition']==kd]);lo,hi=np.percentile(allr,[2.5,97.5]);iv[kd]=(float(lo),float(hi))
        pct[kd]=float(np.mean(allr<obs))
        for r in res:
            if r['condition']==kd:
                q=np.percentile(r['rhos'],[2.5,50,97.5]);rows.append(dict(dataset=r['dataset'],condition=kd,rho_lo=q[0],rho_med=q[1],rho_hi=q[2],blank_active=r['blank_active']))
    f=pd.DataFrame(rows);f.to_csv(out/'model_rho.csv',index=False)
    v=dict(W1=bool(obs>0 and p<.05),W2_pass=bool(p_n>=.05),W3={kd:bool(iv[kd][0]<=obs<=iv[kd][1]) for kd in CONDS},W4=bool(obs_k>0 and p_k<.05))
    report=dict(verdicts=v,real=dict(rho_B_Rpos=obs,p=p,rho_B_Rneg=obs_n,p_neg_two_sided=p_n,rho_B_Rpos_Blt4=obs_k,p_Blt4=p_k,n_Blt4=int(keep.sum()),
                mean_Rpos=float(pos.mean()),mean_Rneg=float(neg.mean()),rho_Rpos_S=float(stats.spearmanr(pos,rm['S']).statistic),sensitivity=sens),
                model_pooled_interval=iv,obs_percentile=pct,model=f.round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',220);print(f.round(3).to_string());print(json.dumps(report['real'],indent=1));print(iv,pct,v)


if __name__=='__main__':main()
