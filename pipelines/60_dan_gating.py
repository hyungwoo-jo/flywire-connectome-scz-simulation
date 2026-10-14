#!/usr/bin/env python3
"""Step 60: learning gated by the gamma1pedc DAN (PPL101) -> KC synapses. d_i = 1 - exp(-5 k_i gamma_i), gamma = g / E_w[g].
Conditions: uniform (original), dan, dan shuffled within KC subtype; five noise seeds each; seven hemispheres (BANC reported,
not in verdicts). Verdicts in docs/PREREGISTRATION_60.md."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402


def _load(name,file):
    spec=_u.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=_u.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p59=_load('p59','59_leak_predictors.py');lib=_load('dan_lib','60_dan_gating_lib.py')
p58,p55=p59.p58,p59.p55;p43,p34,p25,p24,cal=p55.p43,p55.p34,p55.p25,p55.p24,p55.cal
NAMES=p59.NAMES;PRIMARY=[n for n in NAMES if not n.startswith('BANC')];N_SEED=5;SHUF_SEED0=600000
CONDITIONS=('uniform','dan','dan_shuffle')
_ORIG=dict(vs=p25.value_signal,it=p43.index_terms);_D={}


def gamma(g,w):
    """Gate normalized so that its MBON11-weighted mean is 1."""
    return g/((w*g).sum()/w.sum())


def shuffle_within(g,cls,rng):
    out=g.copy()
    for c in np.unique(cls):
        i=np.flatnonzero(cls==c);out[i]=g[rng.permutation(i)]
    return out


def set_gate(gam):
    if gam is None:p25.value_signal,p43.index_terms=_ORIG['vs'],_ORIG['it'];return
    p25.value_signal=lambda w,learn,k:_ORIG['vs'](w,learn*gam,k)
    p43.index_terms=lambda w,learn,m:_ORIG['it'](w,learn*gam,m)


def _run(job):
    name,kind,k=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];gt=_D['gates'][name];p25.SEEDS.update(p59.seeds(k))
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);w=g['w']
    if len(gt['gate'])!=len(w):raise ValueError('KC order mismatch')
    gate=gt['gate'] if kind=='dan' else shuffle_within(gt['gate'],gt['cls'],np.random.default_rng(SHUF_SEED0+100*NAMES.index(name)+k))
    set_gate(None if kind=='uniform' else gamma(gate,w))
    try:rows=p43.evaluate(name,c,w,g)
    finally:set_gate(None)
    return dict(dataset=name,condition=kind,seed=k,specific=float(np.mean([r['specific'] for r in rows])),
                leak=float(np.mean([r['leak_learned'] for r in rows])),leak_control=float(np.mean([r['leak_control'] for r in rows])),
                H=float(np.nanmean([r['H'] for r in rows])))


def gate_stats(name):
    d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs'];gt=_D['gates'][name]
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib)
    s=np.asarray(g['m']@g['x_noise'].mean(1)).ravel();rd=g['w']>0;gate=gt['gate'];cls=gt['cls']
    within=[float(stats.spearmanr(s[rd&(cls==x)],gate[rd&(cls==x)]).statistic) for x in ('g','ab') if (rd&(cls==x)&(gate>0)).sum()>10]
    return dict(dataset=name,dan=gt['dan'],gated_kc=int((gate>0).sum()),gated_read_share=float((gate[rd]>0).mean()),
                rho_s_gate=float(stats.spearmanr(s[rd],gate[rd]).statistic),rho_s_gate_within_mean=float(np.mean(within)) if within else np.nan)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/dan_gating';out.mkdir(parents=True,exist_ok=True)
    p58.check_inputs(root);p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p55.p37.build_all(root);ms,_=p58.build(root);sets.update(ms)
    a,e=p34.load_flywire(root);gates={**lib.hemibrain(root),**lib.flywire(root,a,e),**lib.banc(root),**lib.malecns(root)};del a,e
    _D.update(sets=sets,inputs=inputs,gates=gates);p43._D.update(sets=sets,inputs=inputs)
    with Pool(14) as pool:
        f=pd.DataFrame(pool.map(_run,[(n,kd,k) for n in NAMES for kd in CONDITIONS for k in range(N_SEED)],chunksize=1))
        gs=pd.DataFrame(pool.map(gate_stats,NAMES))
    f.to_csv(out/'per_seed.csv',index=False);gs.to_csv(out/'gate_stats.csv',index=False)
    m=f.groupby(['dataset','condition']).specific.mean().unstack()*100;mh=f.groupby(['dataset','condition']).H.mean().unstack()
    sd=f.groupby(['dataset','condition']).specific.std().unstack()*100
    pm=m.loc[PRIMARY]
    v=dict(P1=bool((pm.dan>=.5).all()),P2=bool(((pm.dan-pm.uniform).abs()<1.).sum()>=4),P3=bool(((pm.dan-pm.dan_shuffle).abs()<.5).sum()>=4))
    report=dict(verdicts=v,specific_mean_pct=m.loc[NAMES].round(3).reset_index().to_dict(orient='records'),
                specific_sd_pct=sd.loc[NAMES].round(3).reset_index().to_dict(orient='records'),
                H_mean=mh.loc[NAMES].round(3).reset_index().to_dict(orient='records'),gate_stats=gs.round(4).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.set_option('display.width',240);print(m.loc[NAMES].round(3).to_string());print(sd.loc[NAMES].round(3).to_string());print(mh.loc[NAMES].round(2).to_string())
    print(gs.round(3).to_string());print(v)


if __name__=='__main__':main()
