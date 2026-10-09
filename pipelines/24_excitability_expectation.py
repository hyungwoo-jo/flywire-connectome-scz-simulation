#!/usr/bin/env python3
"""Global KC excitability and odor-specific expectation (KC threshold lowering) as sources of false alarms."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p23',Path(__file__).with_name('23_detection_false_alarms.py'))
p23=importlib.util.module_from_spec(spec);spec.loader.exec_module(p23)
cal=p23.cal;base=cal.base;robust=cal.robust;disc=p23.disc;rep=p23.rep

FACTORS=(1.,.75,.5)
BETAS=(.25,.5)
N_NOISE=500
TRIALS=20
SEEDS=dict(cal=230001,noise=240002,signal=240003,boot=240100,control=240200,boot_b=240300)
N_BOOT=1000
_G={}


def kc(m,x,c,theta):
    k,res=base.response(m@x,c['inh'],c['drive'],theta,1.,gain=cal.APL_GAIN)
    return k,res


def identify(k,templates):
    """Nearest template by cosine; templates are column-normalized."""
    norm=np.linalg.norm(k,axis=0);norm[norm==0]=1
    return np.argmax(templates.T@(k/norm),axis=0)


def expectation_stats(m,g,theta,a,active,beta,rng=None,capture=False):
    c=g['c'];th=np.full((len(c['inh']),1),theta)
    if rng is None:sel=active
    else:sel=np.zeros_like(active);sel[rng.choice(len(active),int(active.sum()),replace=False)]=True
    th[sel,0]*=(1-beta)
    kn,_=kc(m,g['x_noise'],c,th);dn=kn.sum(0);rep_n=dn>g['crit']
    ida=identify(kn,g['templates'])
    ks,_=kc(m,g['x_weak'][:,g['lab']==a],c,th);ds=ks.sum(0)
    out=dict(FA=float(rep_n.mean()),A_FA=float(np.mean(rep_n&(ida==a))),A_hit=float(np.mean(ds>g['crit'])),A_AUC=p23.auc(ds,dn))
    if capture:
        kw,_=kc(m,g['x_weak'],c,th);rw=(kw.sum(0)>g['crit'])&(g['lab']!=a)
        out['capture']=float(np.mean(rw&(identify(kw,g['templates'])==a))/np.mean(g['lab']!=a))
    return out


def evaluate(m,g,full=False):
    c=g['c'];theta=g['theta']
    kt,_=kc(m,g['x_full'],c,theta);act=kt>1e-8
    nrm=np.linalg.norm(kt,axis=0);nrm[nrm==0]=1;templates=kt/nrm
    dcal=kc(m,g['x_cal'],c,theta)[0].sum(0);crit=float(np.percentile(dcal,95))
    gg=dict(g,templates=templates,crit=crit)
    rows=[];vals={}
    for f in FACTORS:
        dn=kc(m,g['x_noise'],c,theta*f)[0].sum(0);ds=kc(m,g['x_weak'],c,theta*f)[0].sum(0)
        rows.append(dict(part='a',factor=f,FA=float(np.mean(dn>crit)),H=float(np.mean(ds>crit)),AUC=p23.auc(ds,dn)))
        if full:vals[f]=(ds,dn)
    base_n=kc(m,g['x_noise'],c,theta)[0];base_id=identify(base_n,templates);base_rep=base_n.sum(0)>crit
    for beta in (BETAS if full else BETAS[:1]):
        for a in range(g['n']):
            e=expectation_stats(m,gg,theta,a,act[:,a],beta,capture=full and beta==BETAS[0])
            ctl=expectation_stats(m,gg,theta,a,act[:,a],beta,rng=np.random.default_rng(SEEDS['control']+a))
            rows.append(dict(part='b',beta=beta,odor=a,n_active=int(act[:,a].sum()),A_FA_baseline=float(np.mean(base_rep&(base_id==a))),
                             **{f'exp_{k}':v for k,v in e.items()},**{f'ctl_{k}':v for k,v in ctl.items()}))
    return rows,vals


def _run(job):
    side,graph=job;g=_G[side];c=g['c']
    pm,qm,wm=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
    target,_=robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(p23.SIDES[side]['null_seed0']+graph))
    m,_=cal.partial_matrix(c,target)
    rows,_=evaluate(m,g)
    return [dict(side=side,graph=graph,**r) for r in rows]


def setup(root,side):
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=rep.load_side_circuit(root,side,p23.SIDES[side]['apl'],receptors);m,mask=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    theta=(json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())['targets']['0.10']['theta'] if side=='right'
           else json.loads((root/'qc_reports/left_replication/report.json').read_text())['calibration']['0.10']['theta'])
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));d=delta[:,keep];n=len(keep);sf=sfr.to_numpy();zero=np.zeros((24,1))
    drv=lambda dd,lab,seed:cal.pn_drive(c,receptors,cal.sample_orn(dd,sf,lab,.5,np.random.default_rng(seed)))
    lab=np.repeat(np.arange(n),TRIALS)
    return dict(c=c,m=m,mask=mask,theta=theta,n=n,lab=lab,x_full=cal.pn_drive(c,receptors,d),
                x_cal=drv(zero,np.zeros(1000,int),SEEDS['cal']),x_noise=drv(zero,np.zeros(N_NOISE,int),SEEDS['noise']),
                x_weak=drv(.1*d,lab,SEEDS['signal']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/excitability_expectation';out.mkdir(parents=True,exist_ok=True)
    rows=[];res={}
    for side in p23.SIDES:
        g=setup(root,side);_G[side]=g
        r,vals=evaluate(g['m'],g,full=True);rows+=[dict(side=side,graph=0,**x) for x in r]
        rng=np.random.default_rng(SEEDS['boot']);(s1,n1),(s75,n75)=vals[1.],vals[.75];bd=[]
        for _ in range(N_BOOT):
            od=rng.integers(g['n'],size=g['n']);si=(od[:,None]*TRIALS+np.arange(TRIALS)[None,:]).ravel();ni=rng.integers(len(n1),size=len(n1))
            bd.append(p23.auc(s75[si],n75[ni])-p23.auc(s1[si],n1[ni]))
        res[side]=dict(a_boot_ci=[float(x) for x in np.percentile(bd,[2.5,97.5])])
    with Pool(args.workers) as pool:rows+=[x for rs in pool.map(_run,[(s,i) for s in p23.SIDES for i in range(1,args.nulls+1)]) for x in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    for side in p23.SIDES:
        a=f[(f.side==side)&(f.graph==0)&(f.part=='a')].set_index('factor')
        res[side].update(FA={str(k):float(v) for k,v in a.FA.items()},AUC={str(k):float(v) for k,v in a.AUC.items()},H={str(k):float(v) for k,v in a.H.items()},
                         a_delta_AUC=float(a.loc[.75,'AUC']-a.loc[1.,'AUC']))
        for beta in BETAS:
            b=f[(f.side==side)&(f.graph==0)&(f.part=='b')&(f.beta==beta)]
            delta=(b.exp_A_FA-b.ctl_A_FA).to_numpy();rng=np.random.default_rng(SEEDS['boot_b'])
            boot=[delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(N_BOOT)]
            res[side][f'b_beta{beta}']=dict(mean_delta_A_FA=float(delta.mean()),ci=[float(x) for x in np.percentile(boot,[2.5,97.5])],
                A_FA_baseline=float(b.A_FA_baseline.mean()),A_FA_expect=float(b.exp_A_FA.mean()),A_FA_control=float(b.ctl_A_FA.mean()),
                FA_expect=float(b.exp_FA.mean()),FA_control=float(b.ctl_FA.mean()),A_hit_expect=float(b.exp_A_hit.mean()),A_hit_control=float(b.ctl_A_hit.mean()),
                A_AUC_expect=float(b.exp_A_AUC.mean()),A_AUC_control=float(b.ctl_A_AUC.mean()),
                capture_expect=float(b.exp_capture.mean()) if 'exp_capture' in b and b.exp_capture.notna().any() else None,
                mean_active_kc=float(b.n_active.mean()))
        nb=f[(f.side==side)&(f.graph>0)&(f.part=='b')&(f.beta==BETAS[0])].assign(d=lambda x:x.exp_A_FA-x.ctl_A_FA).groupby('graph').d.mean()
        res[side]['b_wiring_z']=float((res[side][f'b_beta{BETAS[0]}']['mean_delta_A_FA']-nb.mean())/nb.std(ddof=1))
        na=f[(f.side==side)&(f.graph>0)&(f.part=='a')].pivot_table(index='graph',columns='factor',values='AUC')
        res[side]['a_wiring_z']=float((res[side]['a_delta_AUC']-(na[.75]-na[1.]).mean())/(na[.75]-na[1.]).std(ddof=1))
    R,L=res['right'],res['left']
    def acls(x):
        if x['a_boot_ci'][1]<0 and x['a_delta_AUC']<-.02:return 'sensitivity_decreases'
        if abs(x['a_delta_AUC'])<.02:return 'bias_shift'
        return 'other'
    va=acls(R) if acls(R)==acls(L) else 'hemispheres_disagree'
    vb='odor_specific_beyond_nonspecific' if all(x[f'b_beta{BETAS[0]}']['ci'][0]>0 for x in (R,L)) else 'not_established'
    vw='wiring_specific' if (abs(R['b_wiring_z'])>=1.96 and abs(L['b_wiring_z'])>=1.96 and np.sign(R['b_wiring_z'])==np.sign(L['b_wiring_z'])) else 'no_wiring_specificity'
    report=dict(verdict_a=va,verdict_b=vb,verdict_wiring=vw,results=res,nulls=args.nulls,seeds=SEEDS)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for side,col in (('right','#D55E00'),('left','#0072B2')):
        a=f[(f.side==side)&(f.graph==0)&(f.part=='a')].sort_values('factor')
        axes[0].plot(a.factor,a.FA,'-o',color=col,label=f'{side} FA');axes[0].plot(a.factor,a.AUC-.85,'--s',color=col,label=f'{side} AUC-0.85')
        for beta,mk in zip(BETAS,('o','^')):
            r=res[side][f'b_beta{beta}']
            axes[1].bar([f'{side[0]} b{beta} ctl',f'{side[0]} b{beta} exp'],[r['A_FA_control']*100,r['A_FA_expect']*100],color=[ '#bbbbbb',col])
        nb=f[(f.side==side)&(f.graph>0)&(f.part=='b')&(f.beta==BETAS[0])].assign(d=lambda x:x.exp_A_FA-x.ctl_A_FA).groupby('graph').d.mean()
        axes[2].hist(nb*100,bins=20,alpha=.4,color=col,label=f'{side} nulls');axes[2].axvline(res[side][f'b_beta{BETAS[0]}']['mean_delta_A_FA']*100,color=col,lw=2,label=f'{side} BANC')
    axes[0].set_xlabel('theta factor');axes[0].legend(fontsize=7);axes[0].set_title('(a) global excitability')
    axes[1].set_ylabel('A-specific false alarms (%)');axes[1].tick_params(axis='x',rotation=60,labelsize=7);axes[1].set_title('(b) expectation vs size-matched control')
    axes[2].set_xlabel('expectation - control (A-specific FA, %)');axes[2].legend(fontsize=7);axes[2].set_title('wiring specificity (beta 0.25)')
    fig.suptitle('24: excitability and expectation');fig.tight_layout();fig.savefig(out/'excitability_expectation.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
