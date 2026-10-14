#!/usr/bin/env python3
"""Step 59: (A) seed-to-seed sampling noise of the specific leak (10 noise/control seeds, seven hemispheres, original circuits);
(B) six pre-registered candidate predictors of the 10-seed mean leak, chosen by leave-one-individual-out CV on the five female
hemispheres and tested on MaleCNS. Verdicts in docs/PREREGISTRATION_59.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p58',Path(__file__).with_name('58_malecns_prediction.py'))
p58=_u.module_from_spec(spec);sys.modules['p58']=p58;spec.loader.exec_module(p58)
p55=p58.p55;p43,p34,p25,p24,cal=p55.p43,p55.p34,p55.p25,p55.p24,p55.cal
FEMALE=p55.NAMES;MALE=p58.NAMES;NAMES=FEMALE+MALE
INDIVIDUAL={'BANC_right':'BANC','BANC_left':'BANC','FlyWire_right':'FlyWire','FlyWire_left':'FlyWire','hemibrain_right':'hemibrain'}
N_SEED=10;CANDS=('cv_spont','cv_spont_read','cv_spont_w','H','top10','rho_w')
BASE=dict(noise=p25.SEEDS['noise'],control=p25.SEEDS['control'])
_D={}


def seeds(k):
    return BASE if k==0 else dict(noise=590000+k,control=591000+100*k)


def candidates(w,s,mn,H):
    """Six predictors from mean spontaneous KC input s, MBON11 weights w, mean noise-trial KC activity mn."""
    has=s>0;rd=w>0;mw=(w*s).sum()/w.sum();top=s>=np.quantile(s[has],.9)
    return dict(cv_spont=float(s[has].std()/s[has].mean()),cv_spont_read=float(s[rd].std()/s[rd].mean()),
                cv_spont_w=float(np.sqrt((w*(s-mw)**2).sum()/w.sum())/mw),H=float(H),
                top10=float(mn[top].sum()/mn.sum()) if mn.sum()>0 else np.nan,rho_w=float(stats.spearmanr(s[rd],w[rd]).statistic))


def _run(job):
    name,k=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];p25.SEEDS.update(seeds(k))
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    rows=p43.evaluate(name,c,g['w'],g)
    out=dict(dataset=name,seed=k,specific=float(np.mean([r['specific'] for r in rows])),leak=float(np.mean([r['leak_learned'] for r in rows])),
             H=float(np.nanmean([r['H'] for r in rows])))
    if k==0:
        theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
        kn,_=p24.kc(m,g['x_noise'],c,theta);s=np.asarray(m@g['x_noise'].mean(1)).ravel()
        out.update(candidates(g['w'],s,kn.mean(1),out['H']))
    return out


def fit(x,y):
    b,a=np.polyfit(x,y,1);return a,b


def pred_interval(x,y,x0):
    a,b=fit(x,y);r=y-(a+b*x);s=r.std(ddof=2);xm=x.mean();sxx=((x-xm)**2).sum()
    se=s*np.sqrt(1+1/len(x)+(x0-xm)**2/sxx);p=a+b*x0;q=stats.t.ppf(.975,len(x)-2);return p,p-q*se,p+q*se


def loio_mae(x,y,groups):
    """Leave-one-individual-out mean absolute error of a straight-line fit."""
    err=[]
    for gname in sorted(set(groups)):
        te=np.array([g==gname for g in groups]);a,b=fit(x[~te],y[~te]);err+=list(np.abs(y[te]-(a+b*x[te])))
    return float(np.mean(err))


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/leak_predictors';out.mkdir(parents=True,exist_ok=True)
    p58.check_inputs(root);p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);ms,_=p58.build(root);sets.update(ms)
    _D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    with Pool(14) as pool:f=pd.DataFrame(pool.map(_run,[(n,k) for n in NAMES for k in range(N_SEED)],chunksize=1))
    f.to_csv(out/'per_seed.csv',index=False)
    agg=f.groupby('dataset').agg(specific_mean=('specific','mean'),specific_sd=('specific','std'),specific_seed0=('specific','first'))
    cand=f[f.seed==0].set_index('dataset')[list(CANDS)];tab=agg.join(cand).loc[NAMES]
    for col in ('specific_mean','specific_sd','specific_seed0'):tab[col]*=100
    tab.to_csv(out/'per_hemisphere.csv')
    fe=tab.loc[FEMALE];y=fe.specific_mean.to_numpy();groups=[INDIVIDUAL[n] for n in FEMALE]
    mae={c:loio_mae(fe[c].to_numpy(),y,groups) for c in CANDS};best=min(mae,key=mae.get)
    preds={}
    for c in {best,'cv_spont'}:
        for n in MALE:
            p,lo,hi=pred_interval(fe[c].to_numpy(),y,float(tab.loc[n,c]))
            preds[f'{c}|{n}']=dict(candidate=c,dataset=n,x=float(tab.loc[n,c]),pred=float(p),lo=float(lo),hi=float(hi),obs=float(tab.loc[n,'specific_mean']))
    pr=pd.DataFrame(preds.values());pr['abs_err']=(pr.pred-pr.obs).abs();pr['inside']=(pr.lo<=pr.obs)&(pr.obs<=pr.hi)
    bm=pr[pr.candidate==best];cm=pr[pr.candidate=='cv_spont']
    v=dict(P1=bool(tab.specific_sd.median()>=.25),P2=bool(mae[best]<mae['cv_spont']),
           P3=bool(bm.inside.all() and bm.abs_err.mean()<cm.abs_err.mean()),
           P4=bool(abs(tab.loc['MaleCNS_right','specific_mean']-tab.loc['MaleCNS_left','specific_mean'])<.5))
    report=dict(verdicts=v,best=best,loio_mae=mae,sd_median=float(tab.specific_sd.median()),predictions=pr.round(4).to_dict(orient='records'),
                per_hemisphere=tab.round(4).reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',240);print(tab.round(3).to_string());print(json.dumps(mae,indent=1),best);print(pr.round(3).to_string());print(v)


if __name__=='__main__':main()
