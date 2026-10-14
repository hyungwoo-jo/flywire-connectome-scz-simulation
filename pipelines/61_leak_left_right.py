#!/usr/bin/env python3
"""Step 61: why MaleCNS left leaks more than right. (1) criterion-swap decomposition of the learned-leak difference into a
criterion part (weak-odor value) and a noise-value part; (2) MBON11 weight counterfactuals (shuffled among read KCs, uniform);
(3) overlap of MBON11 weights with odor-response breadth, noise activity and spontaneous input. Seven hemispheres, five seeds.
Verdicts in docs/PREREGISTRATION_61.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p59',Path(__file__).with_name('59_leak_predictors.py'))
p59=_u.module_from_spec(spec);sys.modules['p59']=p59;spec.loader.exec_module(p59)
p58,p55=p59.p58,p59.p55;p43,p34,p25,p24,cal=p55.p43,p55.p34,p55.p25,p55.p24,p55.cal
NAMES=p59.NAMES;N_SEED=5;VARIANTS=('original','shuffle','uniform');PERM_SEED0=610000
_D={}


def weights(w,kind,rng):
    if kind=='original':return w
    rd=np.flatnonzero(w>0);out=np.zeros_like(w)
    out[rd]=w[rng.permutation(rd)] if kind=='shuffle' else 1.
    return out/out.sum()


def odor_terms(g,kt,kn,kw,crit,w):
    """Per odor: weak-odor criterion, learned and control value signals on reported noise trials."""
    rep=kn.sum(0)>crit;crits=[];le_rep=[];lc_rep=[]
    for a in range(g['n']):
        learn=kt[:,a];act=np.flatnonzero(learn>1e-8)
        crits.append(.5*np.median(p25.value_signal(w,learn,kw[:,g['lab']==a])))
        rng=np.random.default_rng(p25.SEEDS['control']+a);ctl=np.zeros_like(learn)
        ctl[rng.choice(len(learn),len(act),replace=False)]=rng.permutation(learn[act])
        le_rep.append(p25.value_signal(w,learn,kn[:,rep]));lc_rep.append(p25.value_signal(w,ctl,kn[:,rep]))
    return dict(rep=rep,n_trials=kn.shape[1],crit=np.array(crits),le=le_rep,lc=lc_rep)


def leak_from(t,crit):
    """Learned leak: mean over odors of P(reported and value > criterion)."""
    return float(np.mean([(le>c).sum()/t['n_trials'] for le,c in zip(t['le'],crit)]))


def _run(job):
    name,kind,k=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];p25.SEEDS.update(p59.seeds(k))
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    theta=cal.calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    kt,_=p24.kc(m,g['x_full'],c,theta);kn,_=p24.kc(m,g['x_noise'],c,theta);kw,_=p24.kc(m,g['x_weak'],c,theta)
    crit=float(np.percentile(p24.kc(m,g['x_cal'],c,theta)[0].sum(0),95))
    w=weights(g['w'],kind,np.random.default_rng(PERM_SEED0+100*NAMES.index(name)+k))
    t=odor_terms(g,kt,kn,kw,crit,w)
    learned=leak_from(t,t['crit']);control=float(np.mean([(lc>c_).sum()/t['n_trials'] for lc,c_ in zip(t['lc'],t['crit'])]))
    out=dict(dataset=name,variant=kind,seed=k,rep_rate=float(t['rep'].mean()),crit_median=float(np.median(t['crit'])),
             noise_value_mean=float(np.mean([le.mean() for le in t['le'] if len(le)])),learned=learned,control=control,specific=learned-control)
    if kind=='original':
        rd=g['w']>0;wr=g['w'][rd];breadth=(kt>1e-8).sum(1)[rd];pn=(kn>1e-8).mean(1)[rd];s=np.asarray(m@g['x_noise'].mean(1)).ravel()[rd]
        out.update(rho_w_breadth=float(stats.spearmanr(wr,breadth).statistic),rho_w_noise=float(stats.spearmanr(wr,pn).statistic),
                   rho_w_spont=float(stats.spearmanr(wr,s).statistic),w_gini=float(gini(wr)),
                   _crit=t['crit'].tolist(),_le=[le.tolist() for le in t['le']],_n=t['n_trials'])
    return out


def gini(x):
    x=np.sort(x);n=len(x);return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))


def swap_parts(tl,tr):
    """Criterion and noise-value parts of LL - RR (order-averaged)."""
    LL,RR=leak_from(tl,tl['crit']),leak_from(tr,tr['crit']);LR,RL=leak_from(tl,tr['crit']),leak_from(tr,tl['crit'])
    return dict(LL=LL,RR=RR,LR=LR,RL=RL,crit_part=((LL-LR)+(RL-RR))/2,noise_part=((LL-RL)+(LR-RR))/2)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/leak_left_right';out.mkdir(parents=True,exist_ok=True)
    p58.check_inputs(root);p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);ms,_=p58.build(root);sets.update(ms)
    _D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    with Pool(14) as pool:rows=pool.map(_run,[(n,kd,k) for n in NAMES for kd in VARIANTS for k in range(N_SEED)],chunksize=1)
    f=pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]);f.to_csv(out/'per_seed.csv',index=False)
    raw={(r['dataset'],r['seed']):dict(crit=np.array(r['_crit']),le=[np.array(x) for x in r['_le']],n_trials=r['_n']) for r in rows if r['variant']=='original'}
    sw=pd.DataFrame([dict(seed=k,**swap_parts(raw[('MaleCNS_left',k)],raw[('MaleCNS_right',k)])) for k in range(N_SEED)])
    sw.to_csv(out/'criterion_swap.csv',index=False);s=sw.mean(numeric_only=True)
    agg=f.groupby(['dataset','variant'])[['rep_rate','crit_median','noise_value_mean','learned','control','specific']].mean()
    ov=f[f.variant=='original'].groupby('dataset')[['rho_w_breadth','rho_w_noise','rho_w_spont','w_gini']].mean().loc[NAMES]
    spec=agg.specific.unstack()*100;diff={v:float(spec.loc['MaleCNS_left',v]-spec.loc['MaleCNS_right',v]) for v in VARIANTS}
    tot=s.LL-s.RR
    v=dict(Q1='criterion' if s.crit_part>=.5*tot else 'noise_value',Q2=bool(abs(diff['shuffle'])<.5*abs(diff['original'])),
           Q3=bool(abs(diff['uniform'])<.5*abs(diff['original'])))
    report=dict(verdicts=v,swap_mean_pct=(s*100).round(4).to_dict(),lr_specific_diff_pct=diff,
                summary=agg.reset_index().round(5).to_dict(orient='records'),overlap=ov.round(4).reset_index().to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',240);print((agg*[1,1,1,100,100,100]).round(3).to_string());print(ov.round(3).to_string())
    print((s*100).round(3).to_string());print(diff,v)


if __name__=='__main__':main()
