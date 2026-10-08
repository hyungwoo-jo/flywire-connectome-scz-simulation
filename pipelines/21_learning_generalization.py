#!/usr/bin/env python3
"""Does learning generalize along chemical classes more in the real PN->KC wiring than in nulls?"""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


disc=load('disc18','18_community_discrimination.py')
learning=load('learn08','08_associative_readout.py')
cal=disc.cal;base=cal.base;robust=cal.robust

MIN_CLASS=4
ETAS=(5.,20.)
_G={}


def generalization(k,w,eta):
    """k: KC x odors. Returns GI[A,B] = drop_B/drop_A after learning A, diagonal NaN."""
    before=w@k
    after=(w[:,None]*np.exp(-eta*k)).T@k
    drop=1-after/before[None,:]
    gi=drop/np.diag(drop)[:,None]
    np.fill_diagonal(gi,np.nan)
    return gi


def class_index(gi,cls,members=None):
    same=cls[:,None]==cls[None,:];off=~np.eye(len(cls),dtype=bool)
    if members is None:
        return float(np.nanmean(gi[same&off])-np.nanmean(gi[~same]))
    a=cls==members
    within=a[:,None]&a[None,:]&off;between=(a[:,None]&~a[None,:])|(~a[:,None]&a[None,:])
    return float(np.nanmean(gi[within])-np.nanmean(gi[between]))


def metrics(gi,cls,sim):
    off=~np.eye(len(cls),dtype=bool)
    return dict(CI=class_index(gi,cls),CI_ester=class_index(gi,cls,'ester'),
                fidelity=float(stats.spearmanr(gi[off],sim[off]).statistic),mean_GI=float(np.nanmean(gi[off])))


def _run(graph):
    g=_G;c=g['c']
    if graph:
        pm,qm,wm=c['pre'][g['mask']],c['post'][g['mask']],c['w'][g['mask']]
        target,_=robust.strength_preserving_null(pm,qm,wm,np.random.default_rng(disc.NULL_SEED0+graph))
        m,_=cal.partial_matrix(c,target)
    else:m=g['m']
    k,res=base.response(m@g['x'],c['inh'],c['drive'],g['theta'],1.,gain=cal.APL_GAIN)
    rows=[]
    for readout,w in g['readouts'].items():
        for eta in ETAS:
            rows.append(dict(graph=graph,readout=readout,eta=eta,residual=res,active_fraction=float(np.mean(k>1e-8)),
                             **metrics(generalization(k,w,eta),g['cls'],g['sim'])))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nulls',type=int,default=100);ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/learning_generalization';out.mkdir(parents=True,exist_ok=True)
    m16=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns)
    c=cal.load_partial_circuit(root,receptors);m,mask=cal.partial_matrix(c)
    delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    classes=pd.read_csv(root/'data/door/odor.csv',sep=';',index_col=0).drop_duplicates('InChIKey').set_index('InChIKey')['Class'].reindex(odors.index).to_numpy(str)
    ev=np.flatnonzero(~calib);counts=pd.Series(classes[ev]).value_counts()
    keep=np.array([i for i in ev if counts[classes[i]]>=MIN_CLASS])
    x=cal.pn_drive(c,receptors,delta[:,keep]);pn=cal.olsen(delta[:,keep]);sim=np.corrcoef(pn.T)
    # Readouts: uniform over all KCs (primary) and empirical MBON11 KC input weights.
    pre,post,count,totals,main_unit,info=learning.load_output(root)
    sel=post==main_unit;mbon=np.bincount(pre[sel],weights=count[sel],minlength=len(c['inh']))/totals[main_unit]
    readouts=dict(uniform=np.ones(len(c['inh']))/len(c['inh']),mbon11=mbon)
    _G.update(c=c,m=m,mask=mask,x=x,theta=m16['targets']['0.10']['theta'],readouts=readouts,cls=classes[keep],sim=sim)
    with Pool(args.workers) as pool:rows=[r for rs in pool.map(_run,range(args.nulls+1)) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    summary=[]
    for (readout,eta),g in f.groupby(['readout','eta']):
        real=g[g.graph==0].iloc[0];nl=g[g.graph>0];r=dict(readout=readout,eta=eta,primary=bool(readout=='uniform' and eta==5.))
        for metric in ('CI','CI_ester','fidelity','mean_GI'):
            r[f'{metric}_real']=float(real[metric]);r[f'{metric}_null_mean']=float(nl[metric].mean())
            r[f'{metric}_z']=float((real[metric]-nl[metric].mean())/nl[metric].std(ddof=1));r[f'{metric}_rank']=float((nl[metric]>=real[metric]).mean())
        summary.append(r)
    s=pd.DataFrame(summary);s.to_csv(out/'summary.csv',index=False)
    p=s[s.primary].iloc[0];z=p['CI_z']
    verdict='real_more_class_consistent' if z>=1.96 else ('real_less_class_consistent' if z<=-1.96 else 'no_difference')
    report=dict(nulls=args.nulls,odors=len(keep),classes=pd.Series(classes[keep]).value_counts().to_dict(),mbon11=info['main_type'],
                verdict=verdict,primary=p.to_dict(),max_residual=float(f.residual.max()),summary=s.to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pr=f[(f.readout=='uniform')&(f.eta==5.)]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,('CI','fidelity','mean_GI')):
        ax.hist(pr[pr.graph>0][metric],bins=20,color='#999999',label='100 nulls');ax.axvline(pr[pr.graph==0][metric].iloc[0],color='#D55E00',lw=2,label='BANC wiring')
        ax.set_title(metric);ax.legend(fontsize=7)
    fig.suptitle('21: learning generalization (uniform readout, eta 5)');fig.tight_layout();fig.savefig(out/'learning_generalization.png',dpi=180);plt.close(fig)
    print(verdict);print(s.T.to_string())


if __name__=='__main__':main()
