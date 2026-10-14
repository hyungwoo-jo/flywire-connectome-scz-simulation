#!/usr/bin/env python3
"""Step 62: why the weak-odor value signal (leak criterion) is smaller in MaleCNS left. Circuit counterfactuals (APL off,
uniform APL, binary PN->KC, glomerulus totals matched between sides), noise-free weak criterion, and the overlap x depth
decomposition L_weak = P * D. Five seeds. Verdicts in docs/PREREGISTRATION_62.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p61',Path(__file__).with_name('61_leak_left_right.py'))
p61=_u.module_from_spec(spec);sys.modules['p61']=p61;spec.loader.exec_module(p61)
p59,p58,p55=p61.p59,p61.p58,p61.p55;p34,p25,p24,cal=p61.p34,p61.p25,p61.p24,p61.cal
SIDES=('MaleCNS_right','MaleCNS_left');CONDS=('base','no_apl','uniform_apl','binary_pn','glom_matched');N_SEED=5
_D={}


def glom_totals(c):
    mp=c['mapped'];tot={}
    for gl,pi in mp.groupby('glomerulus').pn_index:tot[gl]=float(c['w'][np.isin(c['pre'],pi.to_numpy())].sum())
    return tot


def variant(c,kind,target=None):
    """Counterfactual copy of circuit c."""
    c=dict(c)
    if kind=='no_apl':c['inh']=np.zeros_like(c['inh'])
    elif kind=='uniform_apl':n=len(c['inh']);c['inh']=np.ones(n);c['drive']=np.full(n,1./n)
    elif kind=='binary_pn':c['w']=np.ones_like(c['w'])/np.median(np.bincount(c['post'],minlength=len(c['inh'])))
    elif kind=='glom_matched':
        w=c['w'].copy();own=glom_totals(c)
        for gl,pi in c['mapped'].groupby('glomerulus').pn_index:
            e=np.isin(c['pre'],pi.to_numpy());w[e]*=target[gl]/own[gl]
        c['w']=w
    return c


def overlap_terms(w,learn,kw):
    """Per weak trial: P = share of w-weighted activity on learned KCs, D = mean learning depth there."""
    d=1-np.exp(-p25.ETA*learn);on=learn>1e-8;wk=w[:,None]*kw
    tot=wk.sum(0);on_tot=wk[on].sum(0);ok=tot>0
    P=np.where(ok,on_tot/np.where(ok,tot,1),0.);D=np.where(on_tot>0,(d[on,None]*wk[on]).sum(0)/np.where(on_tot>0,on_tot,1),0.)
    return float(np.median(P)),float(np.median(D[on_tot>0])) if (on_tot>0).any() else 0.


def _run(job):
    name,kind,k=job;receptors,delta,sfr,calib=_D['inputs'];p25.SEEDS.update(p59.seeds(k))
    c=variant(_D['sets'][name]['c'],kind,_D['target']);c['kk']=None;w=_D['sets'][name]['w']
    g=p34.make_g(c,w,receptors,delta,sfr,calib);m=g['m']
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    kt,_=p24.kc(m,g['x_full'],c,theta);kn,_=p24.kc(m,g['x_noise'],c,theta);kw,_=p24.kc(m,g['x_weak'],c,theta)
    k0,_=p24.kc(m,cal.pn_drive(c,receptors,.1*delta[:,g['keep']]),c,theta)
    crit=float(np.percentile(p24.kc(m,g['x_cal'],c,theta)[0].sum(0),95))
    t=p61.odor_terms(g,kt,kn,kw,crit,g['w'])
    learned=p61.leak_from(t,t['crit']);control=float(np.mean([(lc>c_).sum()/t['n_trials'] for lc,c_ in zip(t['lc'],t['crit'])]))
    crit0=[.5*float(p25.value_signal(g['w'],kt[:,a],k0[:,[a]])[0]) for a in range(g['n'])]
    pd_=[overlap_terms(g['w'],kt[:,a],kw[:,g['lab']==a]) for a in range(g['n'])]
    return dict(dataset=name,condition=kind,seed=k,theta=float(theta),sparsity_eval=float(np.mean(kt>1e-8)),
                weak_active=float(np.mean(kw>1e-8)),rep_rate=float(t['rep'].mean()),crit_median=float(np.median(t['crit'])),
                crit0_median=float(np.median(crit0)),P_median=float(np.median([x[0] for x in pd_])),D_median=float(np.median([x[1] for x in pd_])),
                noise_value_mean=float(np.mean([le.mean() for le in t['le'] if len(le)])),learned=learned,control=control,specific=learned-control)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/weak_signal_sources';out.mkdir(parents=True,exist_ok=True)
    p58.check_inputs(root);p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p58.build(root);tl,tr=glom_totals(sets['MaleCNS_left']['c']),glom_totals(sets['MaleCNS_right']['c'])
    target={gl:(tl[gl]+tr[gl])/2 for gl in tl}
    _D.update(sets=sets,inputs=inputs,target=target)
    with Pool(14) as pool:rows=pool.map(_run,[(n,kd,k) for n in SIDES for kd in CONDS for k in range(N_SEED)],chunksize=1)
    f=pd.DataFrame(rows);f.to_csv(out/'per_seed.csv',index=False)
    cols=['theta','sparsity_eval','weak_active','rep_rate','crit_median','crit0_median','P_median','D_median','noise_value_mean','learned','control','specific']
    agg=f.groupby(['condition','dataset'])[cols].mean()
    R,L=agg.xs('MaleCNS_right',level=1),agg.xs('MaleCNS_left',level=1)
    gap=(R.crit_median-L.crit_median).loc[list(CONDS)];G0=float(gap['base'])
    gap0=float(R.crit0_median['base']-L.crit0_median['base'])
    relP=float((R.P_median['base']-L.P_median['base'])/R.P_median['base']);relD=float((R.D_median['base']-L.D_median['base'])/R.D_median['base'])
    leak_diff=((L.specific-R.specific)*100).loc[list(CONDS)]
    v=dict(R1={kd:bool(abs(gap[kd])<.5*abs(G0)) for kd in CONDS[1:]},R2=bool(abs(gap0)<.5*abs(G0)),R3='overlap' if abs(relP)>abs(relD) else 'depth')
    report=dict(verdicts=v,crit_gap=gap.round(5).to_dict(),crit0_gap_base=round(gap0,5),relP=round(relP,4),relD=round(relD,4),
                lr_specific_diff_pct=leak_diff.round(4).to_dict(),glom_totals=dict(left=tl,right=tr),
                summary=agg.reset_index().round(5).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',260);print(agg.round(4).to_string());print(gap.round(4).to_string());print(leak_diff.round(3).to_string())
    print(dict(G0=G0,gap0=gap0,relP=relP,relD=relD),v)


if __name__=='__main__':main()
