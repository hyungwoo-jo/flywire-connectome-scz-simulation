#!/usr/bin/env python3
"""Model counterpart of Ahmed et al. 2023: add or remove KC claws with the threshold fixed; compare with the published effects."""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p37=load('p37','37_kc_recurrence.py');p34=p37.p34;p24=p37.p24;p25=p37.p25;cal=p37.cal;disc=p37.disc

CLAW_SYN=5
NAMES=['BANC_right','BANC_left','FlyWire_right','FlyWire_left','hemibrain_right']
_DATA={}


def counts_of(c):
    unit=c['w'][c['w']>0].min()  # every dataset has single-synapse edges
    return np.rint(c['w']/unit).astype(int),unit


def add_claws(c,rng,frac=.5):
    cnt,unit=counts_of(c);pre,post=c['pre'],c['post'];n_kc=len(c['inh']);n_pn=c['info']['pn']
    claw=cnt>=CLAW_SYN;claws_per_kc=np.bincount(post[claw],minlength=n_kc)
    deg=np.bincount(pre,minlength=n_pn).astype(float);pool=cnt[claw]
    existing=set(zip(pre.tolist(),post.tolist()));np_,nq,nw=[],[],[]
    for k in range(n_kc):
        need=int(round(frac*claws_per_kc[k]))
        if need==0:continue
        p=deg.copy();p[[a for a in range(n_pn) if (a,k) in existing]]=0
        if p.sum()==0:continue
        chosen=rng.choice(n_pn,size=min(need,int((p>0).sum())),replace=False,p=p/p.sum())
        np_+=chosen.tolist();nq+=[k]*len(chosen);nw+=rng.choice(pool,size=len(chosen)).tolist()
    out=dict(c);out['pre']=np.r_[pre,np.array(np_,int)];out['post']=np.r_[post,np.array(nq,int)];out['w']=np.r_[c['w'],np.array(nw,float)*unit]
    return out,len(np_)


def remove_claws(c,rng):
    cnt,_=counts_of(c);pre,post=c['pre'],c['post'];keep=np.zeros(len(pre),bool)
    for k in np.unique(post):
        ix=np.flatnonzero(post==k);cl=ix[cnt[ix]>=CLAW_SYN]
        keep[rng.choice(cl if len(cl) else ix)]=True
    out=dict(c);out['pre']=pre[keep];out['post']=post[keep];out['w']=c['w'][keep]
    return out,int((~keep).sum())


def measure(c,g,theta,receptors,delta,sfr,calib):
    m,_=cal.partial_matrix(c);gg=dict(g,c=c,m=m)
    kt,_=p37.respond(m@g['x_full'],c,theta);act=kt>1e-8
    R=float(act.mean(0).mean());resp=act.any(1);B=float((act[resp].mean(1)>=.5).mean()) if resp.any() else 0.
    cc=np.corrcoef(kt.T);iu=np.triu_indices(cc.shape[0],1);C=float(np.nanmean(cc[iu]))
    kn,_=p37.respond(m@g['x_noise'],c,theta);S_any=float((kn>1e-8).any(0).mean());S_frac=float((kn>1e-8).mean())
    D=float(p37.discrimination(m,gg,theta,delta,sfr,calib,receptors))
    cnt,_=counts_of(c);claws=float(np.bincount(c['post'][cnt>=CLAW_SYN],minlength=len(c['inh'])).mean())
    return dict(R=R,B=B,C=C,D=D,S_any=S_any,S_frac=S_frac,claws_mean=claws)


def _run(job):
    i,name=job;d=_DATA['sets'][name];receptors,delta,sfr,calib=_DATA['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib)
    theta=cal.calibrate(g['m']@cal.pn_drive(c,receptors,delta[:,calib]),c['inh'],c['drive'],.10)
    rows=[dict(dataset=name,condition='control',edges_changed=0,**measure(c,g,theta,receptors,delta,sfr,calib))]
    up,na=add_claws(c,np.random.default_rng(390001+i));rows.append(dict(dataset=name,condition='more_claws',edges_changed=na,**measure(up,g,theta,receptors,delta,sfr,calib)))
    dn,nr=remove_claws(c,np.random.default_rng(391001+i));rows.append(dict(dataset=name,condition='one_claw',edges_changed=nr,**measure(dn,g,theta,receptors,delta,sfr,calib)))
    return rows


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/claw_manipulation';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=p37.build_all(root);_DATA.update(sets=sets,inputs=inputs)
    with Pool(5) as pool:rows=[r for rs in pool.map(_run,list(enumerate(NAMES))) for r in rs]
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    res={}
    for name,q in f.groupby('dataset'):
        q=q.set_index('condition');c0,up,dn=q.loc['control'],q.loc['more_claws'],q.loc['one_claw']
        r=dict(R_ratio_up=float(up.R/c0.R),R_ratio_down=float(dn.R/c0.R),B=[float(c0.B),float(up.B)],C=[float(c0.C),float(up.C)],
               D=[float(c0.D),float(up.D)],S_frac=[float(c0.S_frac),float(up.S_frac)],S_any=[float(c0.S_any),float(up.S_any)],
               claws=[float(c0.claws_mean),float(up.claws_mean),float(dn.claws_mean)])
        r['P1']=r['R_ratio_up']>=1.5;r['P2']=r['B'][1]>r['B'][0];r['P3']=(r['C'][1]>r['C'][0]) and (r['D'][1]<r['D'][0])
        r['P4']=r['S_frac'][1]>r['S_frac'][0];r['P5']=r['R_ratio_down']<=.5;res[name]=r
    preds=['P1','P2','P3','P4','P5']
    if all(all(res[n][p] for p in preds) for n in res):v='qualitative_reproduction'
    elif sum(sum(res[n][p] for n in res)>=4 for p in preds)>=4:v='mostly_reproduced'
    else:v='not_reproduced'
    report=dict(verdict=v,results=res,experiment=dict(source='Ahmed et al. 2023 bioRxiv 10.1101/2023.01.25.525425',
        more_claws=dict(claw_change='~+50% (10-14 claws)',response_fraction_ratio='~2 (e.g. 30% -> 60%)',all_stimulus_responders='<20% -> 40%'),
        fewer_claws=dict(claw_change='median 7 -> 1',response_fraction_ratio='~1/3 (~30% -> ~10%)')))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=bool)+'\n')
    pd.set_option('display.width',200);print(json.dumps(dict(verdict=v),indent=2));print(f.round(4).to_string())
    print(pd.DataFrame(res).T[preds+['R_ratio_up','R_ratio_down']].to_string())


if __name__=='__main__':main()
