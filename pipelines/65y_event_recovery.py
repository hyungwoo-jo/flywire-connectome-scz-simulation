#!/usr/bin/env python3
"""Exploratory known-effect injection into real pre-stimulus calcium traces."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('obs',Path(__file__).with_name('65x_observation_model.py'))
obs=importlib.util.module_from_spec(spec);sys.modules['obs']=obs;spec.loader.exec_module(obs)
p=obs.p65


def count_two_runs(mask):
    if mask.shape[-1]<2:return np.zeros(mask.shape[:-1],dtype=int)
    previous=np.concatenate([np.zeros(mask.shape[:-1]+(1,),bool),mask[...,:-2]],axis=-1)
    return (mask[...,:-1]&mask[...,1:]&~previous).sum(axis=-1)


def detect(base):
    x=base.transpose(1,0,2).reshape(base.shape[1],-1);mu=np.nanmedian(x,axis=1);sd=1.4826*np.nanmedian(np.abs(x-mu[:,None]),axis=1)
    return base>(mu+3*sd)[None,:,None],base<(mu-3*sd)[None,:,None],sd


def inject(base,B,probability,snr,rng):
    _,_,sd=detect(base)
    spread=np.std(B);z=(B-np.mean(B))/spread if spread>0 else np.zeros_like(B,dtype=float)
    weights=np.exp(z);prob=np.clip(probability*weights/weights.mean(),0,1)
    event=rng.random(base.shape[:2])<prob[None,:]
    start=rng.integers(2,base.shape[-1]-8+1,size=base.shape[:2])
    valid=np.isfinite(base).all(axis=-1)&(sd[None,:]>0);event&=valid
    modified=base.copy();kernel=np.exp(-np.arange(8)/2)
    for trial,cell in zip(*np.where(event)):
        modified[trial,cell,start[trial,cell]:start[trial,cell]+8]+=snr*sd[cell]*kernel
    return modified,event,start,prob


def plot_summary(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for probability,g in summary.groupby('probability'):
        positive=g[g.snr>0]
        axes[0].plot(positive.snr,100*positive.recovery_fraction,'-o',label=f'Nominal events/trial={probability}')
        axes[1].plot(g.snr,g.rho_observed,'-o',label=f'Nominal events/trial={probability}')
    axes[0].set_ylabel('New event window recovery (%)');axes[0].set_ylim(0,102)
    axes[1].axhline(summary.iloc[0].rho_observed,color='gray',linestyle='--');axes[1].set_ylabel('Breadth vs detected positive events (rho)')
    for ax in axes:ax.set_xlabel('Injected pulse peak / baseline robust SD');ax.legend(fontsize=8)
    fig.suptitle('Known-effect recovery in 12 real baseline recordings; exploratory')
    fig.tight_layout();fig.savefig(out/'event_recovery.png',dpi=180);plt.close(fig)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/event_recovery';out.mkdir(parents=True,exist_ok=True)
    if hashlib.md5((root/'data/chan2026/traces.tar.xz').read_bytes()).hexdigest()!=p.TRACES_MD5:raise ValueError('Raw archive changed')
    old=pd.read_csv(root/'qc_reports/kc_imaging_chan2026/real_cells.csv');split=pd.read_csv(root/'qc_reports/observation_audit/real_per_fly.csv');rows=[]
    for fi,fly in enumerate(split[split.split=='evaluation'].fly):
        d=p.load_npz(root/f'data/chan2026/raw/traces/{fly}/traces.npz');pre=int(d['before_onset']);chunks=[]
        for trial in d['tracedict']:
            for tr in d['tracedict'][trial].values():
                if np.isnan(tr[0,:pre+1]).sum()>p.BAD_MAX:continue
                chunks.append(p.normalize(tr,pre)[:,:pre])
        base=np.stack(chunks);B=old[old.fly==fly].B.to_numpy();up0,dn0,_=detect(base);baseline=count_two_runs(up0).mean(axis=0)
        np.testing.assert_allclose(baseline,old[old.fly==fly].Rp,atol=1e-14)
        for probability in (.02,.1):
            for repeat in range(10):
                seed=651900+fi*1000+repeat
                for snr in (0.,2.,4.,8.):
                    changed,event,start,propensity=inject(base,B,probability,snr,np.random.default_rng(seed))
                    up,dn,_=detect(changed);rate=count_two_runs(up).mean(axis=0)
                    recovered=already=0
                    for trial,cell in zip(*np.where(event)):
                        s=start[trial,cell];was=count_two_runs(up0[trial,cell,s:s+8])>0
                        now=count_two_runs(up[trial,cell,s:s+8])>0
                        already+=int(was);recovered+=int(now and not was)
                    denominator=int(event.sum())-already
                    rows.append(dict(fly=fly,type=fly.rstrip('0123456789'),probability=probability,repeat=repeat,snr=snr,seed=seed,
                        candidate_event_windows=int(event.sum()),injected_events=int(event.sum()) if snr>0 else 0,baseline_event_windows=already,eligible_injected=denominator if snr>0 else 0,recovered=recovered,
                        recovery_fraction=recovered/denominator if denominator and snr>0 else np.nan,
                        rho_propensity=p.rho(B,propensity),rho_injected=p.rho(B,event.sum(axis=0)),rho_observed=p.rho(B,rate),
                        rho_added=p.rho(B,rate-baseline),mean_positive_rate=float(rate.mean()),mean_negative_rate=float(count_two_runs(dn).mean())))
        print('recovery fly',fly,'complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'per_fly_repeat.csv',index=False)
    metrics=['recovery_fraction','rho_propensity','rho_injected','rho_observed','rho_added','mean_positive_rate','mean_negative_rate']
    per_fly=f.groupby(['fly','type','probability','snr'])[metrics].mean().reset_index();per_fly.to_csv(out/'per_fly.csv',index=False)
    summary=per_fly.groupby(['probability','snr'])[metrics].mean().reset_index();summary.to_csv(out/'summary.csv',index=False)
    report=dict(date='2026-10-18',plan='docs/EVENT_RECOVERY_PLAN.md',n_flies=12,seed_repeats=10,rows=len(f),
        source='data/chan2026; same raw traces MD5 and evaluation split as observation audit',
        caveat='Exploratory injected effects in previously analyzed baselines, not validation of original connectome or inferred biological event sizes.',
        recovery='New positive event anywhere in the 8-frame injection window when no pre-existing event was present; window attribution, not spike truth.',
        summary=summary.astype(object).where(pd.notna(summary),None).to_dict('records'))
    plot_summary(summary,out)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(summary.round(4).to_string(index=False))

if __name__=='__main__':main()
