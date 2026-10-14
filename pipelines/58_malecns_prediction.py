#!/usr/bin/env python3
"""Step 58: out-of-sample test on MaleCNS v1.0 (male, both hemispheres). Rules fitted on the five female hemispheres
(steps 54-57) are fixed in docs/PREREGISTRATION_58.md before this script computes any leak or memory outcome.
Circuit as in step 36 (hemibrain naming): PN->KC (HB_PN rule), APL<->KC, KC->MBON11 of the same side.
Conditions: original, flat (step 29 alpha 1), odor threshold homeostasis (step 55, sparsity matched), spontaneous thresholds
(step 56, lambda 1), total-activity homeostasis (step 57, rho .1 and .01), linear mix beta .2 (step 57)."""
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import util as _u  # noqa: E402

spec=_u.spec_from_file_location('p57',Path(__file__).with_name('57_mixed_homeostasis.py'))
p57=_u.module_from_spec(spec);sys.modules['p57']=p57;spec.loader.exec_module(p57)
p56,p55=p57.p56,p57.p55;p43,p34,p25,p24,cal,p29=p55.p43,p55.p34,p55.p25,p55.p24,p55.cal,p55.p29
spec=_u.spec_from_file_location('p36',Path(__file__).with_name('36_hemibrain_replication.py'))
p36=_u.module_from_spec(spec);sys.modules['p36']=p36;spec.loader.exec_module(p36)

NAMES=['MaleCNS_right','MaleCNS_left']
FILES={'body-annotations-male-cns-v1.0-minconf-0.5.feather':'2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2',
       'connectome-weights-male-cns-v1.0-minconf-0.5.feather':'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'}
CONDITIONS=[('original',np.nan),('flat',np.nan),('odor_homeo',np.nan),('spont',1.),('total_homeo',.1),('total_homeo',.01),('mix',.2)]
# Rules fitted on the five female hemispheres (PREREGISTRATION_58): y = a + b * cv_spont, 95% prediction interval
RULES=dict(specific=(-5.138,9.754),H=(-17.104,31.765),odor_homeo=(-3.488,4.802))
CV_REF=np.array([.8967,.8918,.7194,.7033,.7910])
REF=dict(specific=np.array([3.425,3.701,1.616,1.877,2.729]),H=np.array([10.917,12.292,6.370,5.398,6.633]),
         odor_homeo=np.array([.997,.764,.077,-.020,-.040]))
_D={}


def check_inputs(root):
    for f,h in FILES.items():
        if hashlib.sha256((root/'data/malecns'/f).read_bytes()).hexdigest()!=h:raise ValueError(f'Input changed: {f}')


def malecns_tables(root):
    d=root/'data/malecns';a=pd.read_feather(d/'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    a['id']=a.bodyId.astype(str);a['type']=a['type'].fillna('');a['instance']=a.instance.fillna('')
    keep=a.type.str.startswith('KC')|a.type.str.contains(p36.HB_PN,regex=True)|a.type.isin(['APL','MBON11'])
    ids=set(a[keep].bodyId)
    e=pd.read_feather(d/'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
    e=e[e.body_pre.isin(ids)&e.body_post.isin(ids)]
    e=pd.DataFrame(dict(pre=e.body_pre.astype(str),post=e.body_post.astype(str),count=e.weight.to_numpy()))
    return a[keep].reset_index(drop=True),e


def malecns_circuit(n,e,receptors,root,side):
    """Step-36 circuit for one hemisphere: KCs and APL by soma side, MBON11 by instance suffix; PNs from either side."""
    s=side[0].upper();kc=sorted(n[n.type.str.startswith('KC')&(n.somaSide==s)].id)
    apl=n[(n.type=='APL')&(n.somaSide==s)].id.tolist();mb=n[(n.type=='MBON11')&n.instance.str.endswith(f'_{s}')].id.tolist()
    if len(apl)!=1 or len(mb)!=1:raise ValueError('APL/MBON11 lookup failed')
    apl,mb=apl[0],mb[0];pn_all=n[n.type.str.contains(p36.HB_PN,regex=True)]
    edges=e[e.pre.isin(set(pn_all.id))&e.post.isin(set(kc))]
    pn=sorted(edges.pre.unique());pi={x:i for i,x in enumerate(pn)};ki={x:i for i,x in enumerate(kc)}
    pre=edges.pre.map(pi).to_numpy(int);post=edges.post.map(ki).to_numpy(int);counts=edges['count'].to_numpy(float)
    inh=e[(e.pre==apl)&e.post.isin(ki)].groupby('post')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    drive=e[e.pre.isin(ki)&(e.post==apl)].groupby('pre')['count'].sum().reindex(kc,fill_value=0).to_numpy(float)
    scale=np.median(np.bincount(post,weights=counts,minlength=len(kc)));inh/=inh.mean();drive/=drive.sum()
    glom=cal.receptor_glomeruli(root,receptors);ann=pn_all.set_index('id');rows=[]
    for i,p in enumerate(pn):
        t=ann.loc[p,'type'];prefix=t.split('_')[0];multi=t.startswith('M_') or '+' in prefix
        rows.append(dict(pn_index=i,root_id=p,cell_type=t,glomerulus=prefix,status='multiple_glomeruli' if multi else 'ok',
                         receptor=glom[prefix] if (prefix in glom and not multi) else ''))
    pns=pd.DataFrame(rows)
    q=e[(e.post==mb)&e.pre.isin(ki)];w=np.zeros(len(kc));np.add.at(w,q.pre.map(ki).to_numpy(int),q['count'].to_numpy(float))
    info=dict(side=side,apl=apl,mbon11=mb,pn=len(pn),kc=len(kc),pn_kc_edges=len(edges),apl_kc_edges=int(np.count_nonzero(inh)),
              kc_apl_edges=int(np.count_nonzero(drive)),mbon11_kc_inputs=int(np.count_nonzero(w)),
              pn_kc_synapses=float(counts.sum()),kc_with_pn=int((np.bincount(post,minlength=len(kc))>0).sum()))
    return dict(pre=pre,post=post,w=counts/scale,inh=inh,drive=drive,info=info,pns=pns,mapped=pns[pns.receptor!=''].reset_index(drop=True)),w/w.sum()


def build(root):
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    n,e=malecns_tables(root);out={}
    for side in ('right','left'):
        c,w=malecns_circuit(n,e,receptors,root,side);out[f'MaleCNS_{side}']=dict(c=c,w=w,kk={})
    return out,(receptors,delta,sfr.to_numpy(),calib)


def prediction(key,cv):
    a,b=RULES[key];y=REF[key];r=y-(a+b*CV_REF);s=r.std(ddof=2);xm=CV_REF.mean();sxx=((CV_REF-xm)**2).sum()
    se=s*np.sqrt(1+1/len(y)+(cv-xm)**2/sxx);p=a+b*cv;return float(p),float(p-3.182*se),float(p+3.182*se)


def _run(job):
    name,kind,par=job;d=_D['sets'][name];receptors,delta,sfr,calib=_D['inputs']
    c=dict(d['c']);c['kk']=None;g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    xcal=cal.pn_drive(c,receptors,delta[:,calib]);E=m@xcal;theta0=cal.calibrate(E,c['inh'],c['drive'],.10)
    s=np.asarray(m.sum(1)).ravel();tune=s>0;En=np.asarray(m@g['x_cal']);extra={}
    if kind=='original':
        th=np.full(len(s),theta0);s_sp=np.asarray(m@g['x_noise'].mean(1)).ravel();has=s_sp>0
        extra=dict(cv_spont=float(s_sp[has].std()/s_sp[has].mean()),cv_raw=float(s[s>0].std()/s[s>0].mean()))
    elif kind=='flat':
        m=p29.compensate(m,1.);th=np.full(len(s),cal.calibrate(m@xcal,c['inh'],c['drive'],.10))
    elif kind=='odor_homeo':
        a0=float(p55.respond(E,c,theta0)[tune].mean());th,_,_=p55.homeostasis(E,c,theta0,tune,a0,'theta')
        th=np.maximum(th+p55.offset_to_target(E,c,th),0.)
    elif kind=='spont':
        z=p56.z_for_target(E,c,En.mean(1),En.std(1));th=np.maximum(En.mean(1)+z*En.std(1),0.)
    elif kind=='total_homeo':
        th,_=p57.mixed_homeostasis(E,En,c,theta0,tune,par);th=np.maximum(th+p55.offset_to_target(E,c,th),0.)
    else:
        a0=float(p55.respond(E,c,theta0)[tune].mean());to,_,_=p55.homeostasis(E,c,theta0,tune,a0,'theta')
        to=np.maximum(to+p55.offset_to_target(E,c,to),0.)
        z=p56.z_for_target(E,c,En.mean(1),En.std(1));ts=np.maximum(En.mean(1)+z*En.std(1),0.)
        th=(1-par)*to+par*ts;th=np.maximum(th+p55.offset_to_target(E,c,th),0.)
    theta=th[:,None];rows=p43.evaluate(name,c,g['w'],g,m_override=m,theta=theta,label=f'{kind}_{par}')
    x_test=cal.pn_drive(c,receptors,cal.sample_orn(delta[:,g['keep']],sfr,g['lab'],.5,np.random.default_rng(p55.SEED_TEST)))
    kt=p55.respond(m@g['x_full'],c,theta);kte=p55.respond(m@x_test,c,theta);kn=p55.respond(m@g['x_noise'],c,theta)
    aa=p55.memory(kt,kte,g['lab'],g['n'],g['n']);a20=p55.memory(kt,kte,g['lab'],g['n'],p55.N_SMALL)
    return dict(dataset=name,condition=kind,par=par,acc_all=float(np.mean(aa)),acc_20=float(np.mean(a20)),acc_all_draws=json.dumps(aa),acc_20_draws=json.dumps(a20),
                H=float(np.nanmean([r['H'] for r in rows])),H_median=float(np.nanmedian([r['H'] for r in rows])),
                leak=float(np.mean([r['leak_learned'] for r in rows])),leak_control=float(np.mean([r['leak_control'] for r in rows])),
                specific=float(np.mean([r['specific'] for r in rows])),active_odor=float((kt>1e-8).mean()),active_noise=float((kn>1e-8).mean()),**extra)


def verdicts(f,dfd):
    """Pre-registered checks (PREREGISTRATION_58); each must hold in both hemispheres."""
    v={};pred={}
    for n in NAMES:
        r=f[f.dataset==n].set_index(['condition','par'],drop=False);o=f[(f.dataset==n)&(f.condition=='original')].iloc[0]
        dd=dfd[dfd.dataset==n];get=lambda k,p,t:dd[(dd.condition==k)&((dd.par==p)|(dd.par.isna()&np.isnan(p)))&(dd.task==t)].iloc[0]
        row=lambda k,p:f[(f.dataset==n)&(f.condition==k)&((f.par==p)|(f.par.isna()&np.isnan(p)))].iloc[0]
        cv=float(o.cv_spont);ps=prediction('specific',cv);ph=prediction('H',cv);po=prediction('odor_homeo',cv)
        oh=row('odor_homeo',np.nan);sp=row('spont',1.);t01=row('total_homeo',.01);fl=row('flat',np.nan)
        g01=get('total_homeo',.01,'acc_20');gfl=get('flat',np.nan,'acc_all')
        pred[n]=dict(cv_spont=cv,specific_pred=ps,H_pred=ph,odor_homeo_pred=po,specific_obs=float(o.specific)*100,H_obs=float(o.H),
                     odor_homeo_obs=float(oh.specific)*100)
        chk=dict(Q1=bool(o.specific*100>=.5 and o.H>2),Q2=bool(ps[1]<=o.specific*100<=ps[2]),Q3=bool(ph[1]<=o.H<=ph[2]),
                 Q4=bool(po[1]<=oh.specific*100<=po[2]),Q5=bool(abs(sp.specific)*100<=.3),
                 Q6=bool(abs(t01.specific)*100<=.3 and g01['diff']>-g01['se']),Q7=bool(gfl['diff']<0 and fl.specific<o.specific/2))
        for k,x in chk.items():v.setdefault(k,[]).append(x)
    return {k:bool(all(x)) for k,x in v.items()},{k:x for k,x in v.items()},pred


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/malecns_prediction';out.mkdir(parents=True,exist_ok=True)
    check_inputs(root);p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=build(root);_D.update(sets=sets,inputs=inputs);p43._D.update(sets=sets,inputs=inputs)
    info={n:dict(**sets[n]['c']['info'],mapped_pns=len(sets[n]['c']['mapped']),mapped_glomeruli=int(sets[n]['c']['mapped'].glomerulus.nunique())) for n in NAMES}
    print(json.dumps(info,indent=1,default=str),flush=True)
    if '--circuit-only' in sys.argv:return
    jobs=[(n,k,p) for n in NAMES for k,p in CONDITIONS]
    with Pool(14) as pool:f=pd.DataFrame(pool.map(_run,jobs,chunksize=1))
    diffs=[]
    for n in NAMES:
        b=f[(f.dataset==n)&(f.condition=='original')].iloc[0]
        for _,r in f[(f.dataset==n)&(f.condition!='original')].iterrows():
            for k in ('acc_all_draws','acc_20_draws'):
                dd=np.array(json.loads(r[k]))-np.array(json.loads(b[k]))
                diffs.append(dict(dataset=n,condition=r.condition,par=r.par,task=k[:-6],diff=float(dd.mean()),se=float(dd.std(ddof=1)/np.sqrt(len(dd)))))
    dfd=pd.DataFrame(diffs);f.to_csv(out/'per_condition.csv',index=False);dfd.to_csv(out/'accuracy_diff_vs_original.csv',index=False)
    v,per,pred=verdicts(f,dfd)
    report=dict(verdicts=v,per_hemisphere=per,predictions=pred,circuit=info,rows=f.drop(columns=['acc_all_draws','acc_20_draws']).round(5).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.set_option('display.width',240);print(f.drop(columns=['acc_all_draws','acc_20_draws']).round(4).to_string())
    dfd['diff']*=100;dfd['se']*=100;print(dfd.round(2).to_string());print(json.dumps(dict(verdicts=v,per=per,pred=pred),indent=1))


if __name__=='__main__':main()
