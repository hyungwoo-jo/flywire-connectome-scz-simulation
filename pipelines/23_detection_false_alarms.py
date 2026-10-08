#!/usr/bin/env python3
"""Odor presence detection: false alarms, hits and sensitivity (AUC) under APL weakening, both hemispheres."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


rep=load('rep22','22_left_replication.py')
cal=rep.cal;base=cal.base;robust=cal.robust;disc=rep.disc

RIGHT_APL='720575941482622627'
SIDES=dict(right=dict(apl=RIGHT_APL,null_seed0=180000),left=dict(apl=rep.LEFT_APL,null_seed0=222000))
STRENGTHS=(1.,.5,0.)
CONCS=(.05,.1,.2)
N_NOISE=1000
TRIALS=20
SEEDS=dict(cal=230001,test=230002,signal=230003,boot=230100)
N_BOOT=1000
_G={}


def auc(signal,noise):
    r=rankdata(np.r_[signal,noise]);n1=len(signal);n0=len(noise)
    return float((r[:n1].sum()-n1*(n1+1)/2)/(n1*n0))


def sdt(hit,fa,n_hit,n_fa):
    h=np.clip(hit,1/(2*n_hit),1-1/(2*n_hit));f=np.clip(fa,1/(2*n_fa),1-1/(2*n_fa))
    return float(norm.ppf(h)-norm.ppf(f)),float(-(norm.ppf(h)+norm.ppf(f))/2)


def detector(m,x,c,theta,s):
    k,res=base.response(m@x,c['inh'],c['drive'],theta,s,gain=cal.APL_GAIN)
    return k.sum(0),res


def evaluate_graph(m,g,keep_values=False):
    c=g['c'];rows=[];values={};res_max=0.
    crit=None
    for s in STRENGTHS:
        dcal,r1=detector(m,g['x_cal'],c,g['theta'],s);dn,r2=detector(m,g['x_noise'],c,g['theta'],s);res_max=max(res_max,r1,r2)
        if s==1.:crit=float(np.percentile(dcal,95))
        fa=float(np.mean(dn>crit))
        for conc in CONCS:
            ds,r3=detector(m,g['x_signal'][conc],c,g['theta'],s);res_max=max(res_max,r3)
            hit=float(np.mean(ds>crit));dp,cs=sdt(hit,fa,len(ds),len(dn))
            rows.append(dict(apl=s,conc=conc,FA=fa,H=hit,AUC=auc(ds,dn),dprime=dp,c_sdt=cs,criterion=crit,noise_mean=float(dn.mean()),signal_mean=float(ds.mean())))
            if keep_values:values[(s,conc)]=(ds,dn)
    return rows,values,res_max


def _run(job):
    side,graph=job;g=_G[side];c=g['c']
    if graph:
        pm,qm,wm=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
        target,_=robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(SIDES[side]['null_seed0']+graph))
        m,_=cal.partial_matrix(c,target)
    else:m=g['m']
    rows,_,res=evaluate_graph(m,g)
    return [dict(side=side,graph=graph,residual=res,**r) for r in rows]


def bootstrap_delta(values,n_odors,rng):
    """Paired bootstrap over odors (all trials of a resampled odor) and noise trials."""
    (s1,n1),(s5,n5)=values[(1.,.1)],values[(.5,.1)]
    out=np.empty(N_BOOT)
    for b in range(N_BOOT):
        od=rng.integers(n_odors,size=n_odors);si=(od[:,None]*TRIALS+np.arange(TRIALS)[None,:]).ravel();ni=rng.integers(len(n1),size=len(n1))
        out[b]=auc(s5[si],n5[ni])-auc(s1[si],n1[ni])
    return out


def setup(root,side,args):
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=rep.load_side_circuit(root,side,SIDES[side]['apl'],receptors);m,mask=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    if side=='right':theta=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())['targets']['0.10']['theta']
    else:theta=json.loads((root/'qc_reports/left_replication/report.json').read_text())['calibration']['0.10']['theta']
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));d=delta[:,keep];n=len(keep)
    zero=np.zeros((24,1));sf=sfr.to_numpy()
    def drive(dd,labels,seed,window=.5):
        return cal.pn_drive(c,receptors,cal.sample_orn(dd,sf,labels,window,np.random.default_rng(seed)))
    lab=np.repeat(np.arange(n),TRIALS)
    g=dict(c=c,m=m,mask=mask,theta=theta,n_odors=n,
           x_cal=drive(zero,np.zeros(N_NOISE,int),SEEDS['cal']),x_noise=drive(zero,np.zeros(N_NOISE,int),SEEDS['test']),
           x_signal={conc:drive(conc*d,lab,SEEDS['signal']+int(conc*1000)) for conc in CONCS})
    # Secondary: 0.25 s window, real graph only.
    g25=dict(g,x_cal=drive(zero,np.zeros(N_NOISE,int),SEEDS['cal'],.25),x_noise=drive(zero,np.zeros(N_NOISE,int),SEEDS['test'],.25),
             x_signal={conc:drive(conc*d,lab,SEEDS['signal']+int(conc*1000),.25) for conc in CONCS})
    return g,g25


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/detection_false_alarms';out.mkdir(parents=True,exist_ok=True)
    real_rows=[];boot={};secondary=[]
    for side in SIDES:
        g,g25=setup(root,side,args);_G[side]=g
        rows,values,res=evaluate_graph(g['m'],g,keep_values=True)
        real_rows+= [dict(side=side,graph=0,residual=res,**r) for r in rows]
        boot[side]=bootstrap_delta(values,g['n_odors'],np.random.default_rng(SEEDS['boot']))
        rows25,_,_=evaluate_graph(g25['m'],g25);secondary+=[dict(side=side,window=.25,**r) for r in rows25]
    with Pool(args.workers) as pool:null_rows=[r for rs in pool.map(_run,[(s,i) for s in SIDES for i in range(1,args.nulls+1)]) for r in rs]
    f=pd.DataFrame(real_rows+null_rows);f.to_csv(out/'metrics.csv',index=False);pd.DataFrame(secondary).to_csv(out/'secondary_window025.csv',index=False)
    res={}
    for side in SIDES:
        r=f[(f.side==side)&(f.graph==0)&(f.conc==.1)].set_index('apl');nl=f[(f.side==side)&(f.graph>0)&(f.conc==.1)]
        d_real=float(r.loc[.5,'AUC']-r.loc[1.,'AUC'])
        dn=nl.pivot_table(index='graph',columns='apl',values='AUC');d_null=(dn[.5]-dn[1.]).to_numpy()
        ci=np.percentile(boot[side],[2.5,97.5])
        res[side]=dict(FA_s1=float(r.loc[1.,'FA']),FA_s05=float(r.loc[.5,'FA']),FA_s0=float(r.loc[0.,'FA']),
                       H_s1=float(r.loc[1.,'H']),H_s05=float(r.loc[.5,'H']),AUC_s1=float(r.loc[1.,'AUC']),AUC_s05=float(r.loc[.5,'AUC']),AUC_s0=float(r.loc[0.,'AUC']),
                       dprime_s1=float(r.loc[1.,'dprime']),dprime_s05=float(r.loc[.5,'dprime']),c_sdt_s1=float(r.loc[1.,'c_sdt']),c_sdt_s05=float(r.loc[.5,'c_sdt']),
                       delta_AUC=d_real,delta_AUC_ci=[float(ci[0]),float(ci[1])],
                       null_delta_AUC_mean=float(d_null.mean()),z_wiring=float((d_real-d_null.mean())/d_null.std(ddof=1)))
    R,L=res['right'],res['left']
    h_a='FA_increases' if (R['FA_s05']>R['FA_s1'] and L['FA_s05']>L['FA_s1']) else 'not_both'
    def cls(x):
        if x['delta_AUC_ci'][1]<0 and x['delta_AUC']<-.02:return 'down'
        if x['delta_AUC_ci'][0]>0 and x['delta_AUC']>.02:return 'up'
        if abs(x['delta_AUC'])<.02:return 'flat'
        return 'other'
    cr,cl=cls(R),cls(L)
    h_b={'down':'sensitivity_decreases','up':'sensitivity_increases','flat':'bias_shift_sensitivity_unchanged'}.get(cr,'undetermined') if cr==cl else 'hemispheres_disagree_or_undetermined'
    h_c='wiring_specific' if (abs(R['z_wiring'])>=1.96 and abs(L['z_wiring'])>=1.96 and np.sign(R['z_wiring'])==np.sign(L['z_wiring'])) else 'no_wiring_specificity'
    report=dict(nulls=args.nulls,seeds=SEEDS,H23a=h_a,H23b=dict(verdict=h_b,right=cr,left=cl),H23c=h_c,results=res,max_residual=float(f.residual.max()))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for side,col in (('right','#D55E00'),('left','#0072B2')):
        r=f[(f.side==side)&(f.graph==0)]
        for conc,ls in zip(CONCS,(':','-','--')):
            q=r[r.conc==conc].sort_values('apl');axes[1].plot(q.apl,q.AUC,ls,color=col,marker='o',label=f'{side} c={conc}')
        q=r[r.conc==.1].sort_values('apl');axes[0].plot(q.apl,q.FA,'-o',color=col,label=f'{side} FA');axes[0].plot(q.apl,q.H,'--o',color=col,label=f'{side} hit (c=0.1)')
        nl=f[(f.side==side)&(f.graph>0)&(f.conc==.1)].pivot_table(index='graph',columns='apl',values='AUC')
        axes[2].hist(nl[.5]-nl[1.],bins=20,alpha=.4,color=col,label=f'{side} nulls');axes[2].axvline(res[side]['delta_AUC'],color=col,lw=2,label=f'{side} BANC')
    axes[0].set_xlabel('APL strength');axes[0].set_ylabel('rate');axes[0].legend(fontsize=7);axes[0].set_title('False alarms and hits (fixed criterion)')
    axes[1].set_xlabel('APL strength');axes[1].set_ylabel('AUC');axes[1].legend(fontsize=6);axes[1].set_title('Sensitivity')
    axes[2].set_xlabel('AUC(s=0.5) - AUC(s=1)');axes[2].legend(fontsize=7);axes[2].set_title('Wiring specificity')
    fig.suptitle('23: odor presence detection under APL weakening');fig.tight_layout();fig.savefig(out/'detection.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
