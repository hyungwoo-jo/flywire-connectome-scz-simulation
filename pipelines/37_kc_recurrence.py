#!/usr/bin/env python3
"""Add KC->KC connections (calyx or total) with gain g to the calibrated model; measure value leakage and discrimination."""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


p36=load('p36','36_hemibrain_replication.py');p34=p36.p34;p26=p34.p26;p25=p34.p25;p24=p25.p24;p23=p24.p23
cal=p24.cal;base=cal.base;disc=p24.disc;rep=p23.rep

DAMP=.5
MAX_IT=200
TOL=1e-6
PRIMARY_G=(0.,.3,-.3)
HB_EXTRA_G=(.1,-.1,1.,-1.)
_DATA={}


def respond(E,c,theta,strength=1.):
    """Fixed point of k = clip(E + g W_kk k - theta - APL, 0, 1) with damped iteration around the APL solver."""
    k,res=base.response(E,c['inh'],c['drive'],theta,strength,gain=cal.APL_GAIN)
    kk=c.get('kk')
    if kk is None or kk[1]==0:return k,res
    W,g=kk;c['_converged']=False
    for _ in range(MAX_IT):
        knew,res=base.response(E+g*(W@k),c['inh'],c['drive'],theta,strength,gain=cal.APL_GAIN)
        diff=float(np.max(np.abs(knew-k)));k=DAMP*k+(1-DAMP)*knew
        if diff<TOL:c['_converged']=True;break
    return k,res


def kc_patched(m,x,c,theta):
    return respond(m@x,c,theta)


p24.kc=kc_patched


def calibrate(e_cal,c,target=.10):
    lo,hi=0.,float(e_cal.max())*2
    for _ in range(40):
        th=(lo+hi)/2;k,_=respond(e_cal,c,th)
        if np.mean(k>1e-8)>target:lo=th
        else:hi=th
    return (lo+hi)/2


def discrimination(m,g,theta,delta,sfr,calib,receptors):
    keep=np.flatnonzero((~calib)&(cal.olsen(delta).max(0)>=disc.MIN_PN_HZ));n=len(keep)
    labels=np.repeat(np.arange(n),disc.TRAIN+disc.TEST);train=np.tile(np.arange(disc.TRAIN+disc.TEST)<disc.TRAIN,n)
    x=cal.pn_drive(g['c'],receptors,cal.sample_orn(delta[:,keep],sfr,labels,.5,np.random.default_rng(disc.NOISE_SEED)))
    k,_=respond(m@x,g['c'],theta)
    return disc.evaluate(k,labels,train,{'ALL':np.arange(n)})['ALL_dprime']


def kk_matrix(edges,kc_ids,scale):
    idx={k:i for i,k in enumerate(kc_ids)};q=edges[edges.pre.isin(idx)&edges.post.isin(idx)&(edges.pre!=edges.post)]
    return sparse.csr_matrix((q['count'].to_numpy(float)/scale,(q.post.map(idx).to_numpy(int),q.pre.map(idx).to_numpy(int))),shape=(len(kc_ids),len(kc_ids)))


def pn_scale(c):
    raw=np.bincount(c['post'],weights=c['w'],minlength=len(c['inh']))  # already divided by scale -> median 1
    return 1.0


def build_all(root):
    odors,sfr,_=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T;calib=cal.split_odors(list(odors.index))
    out={}
    # hemibrain
    n,tot,roi=p36.hemibrain_tables(root);c,w=p36.hemibrain_circuit(n,tot,receptors,root);kc=sorted(n[n.type.str.startswith('KC')].id)
    sc=_scale(tot,n[n.type.str.contains(p36.HB_PN,regex=True)].id,kc)
    out['hemibrain_right']=dict(c=c,w=w,kk=dict(calyx=kk_matrix(roi[roi.roi=='CA(R)'],kc,sc),total=kk_matrix(tot,kc,sc)))
    # FlyWire
    a,e=p34.load_flywire(root)
    raw=pd.read_feather(root/'data/flywire/proofread_connections_783.feather',columns=['pre_pt_root_id','post_pt_root_id','neuropil','syn_count'])
    raw['pre']=raw.pre_pt_root_id.astype(str);raw['post']=raw.post_pt_root_id.astype(str)
    pnf=a[(a.cell_class=='ALPN')&(a.top_nt=='acetylcholine')].id
    for side,npl in (('right','MB_CA_R'),('left','MB_CA_L')):
        c,w,_=p34.side_circuit(a,e,side,receptors,root);kc=sorted(a[(a.cell_class=='Kenyon_Cell')&(a.side==side)].id);sc=_scale(e,pnf,kc)
        cal_e=raw[raw.neuropil==npl].groupby(['pre','post'],as_index=False).syn_count.sum().rename(columns={'syn_count':'count'})
        out[f'FlyWire_{side}']=dict(c=c,w=w,kk=dict(calyx=kk_matrix(cal_e,kc,sc),total=kk_matrix(e,kc,sc)))
    # BANC (total only)
    meta=pd.read_feather(root/'data/banc_888_meta.feather');be=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')[['pre','post','count']]
    pnb=meta[(meta.cell_class=='antennal_lobe_projection_neuron')&(meta.neurotransmitter_verified=='acetylcholine')].root_888.astype(str)
    for side in ('right','left'):
        c=rep.load_side_circuit(root,side,p23.SIDES[side]['apl'],receptors);w=p25.mbon_weights(root,side,len(c['inh']))
        kc=sorted(meta[(meta.cell_class=='kenyon_cell')&(meta.side==side)].root_888.astype(str));sc=_scale(be,pnb,kc)
        out[f'BANC_{side}']=dict(c=c,w=w,kk=dict(total=kk_matrix(be,kc,sc)))
    return out,(receptors,delta,sfr.to_numpy(),calib)


def _scale(edges,pn_ids,kc_ids):
    q=edges[edges.pre.isin(set(pn_ids))&edges.post.isin(set(kc_ids))];s=q.groupby('post')['count'].sum().reindex(kc_ids,fill_value=0)
    return float(np.median(s.to_numpy()))


def _run(job):
    name,kind,gval=job;d=_DATA['sets'][name];receptors,delta,sfr,calib=_DATA['inputs']
    c=dict(d['c']);c['kk']=None if gval==0 else (d['kk'][kind],gval)
    g=p34.make_g(c,d['w'],receptors,delta,sfr,calib);m=g['m']
    theta=calibrate(m@cal.pn_drive(c,receptors,delta[:,calib]),c)
    conv=c.get('_converged',True)
    lr,pp=p25.evaluate(m,dict(g,theta=theta));conv=conv and c.get('_converged',True)
    kt,_=respond(m@g['x_full'],c,theta);part=(kt>1e-8).sum(1)
    dp=discrimination(m,g,theta,delta,sfr,calib,receptors);conv=conv and c.get('_converged',True)
    d_spec=np.array([r['leak_learned']-r['leak_control'] for r in lr])
    return dict(dataset=name,kk=kind,g=gval,converged=bool(conv),theta=float(theta),specific=float(d_spec.mean()),
                leak_learned=float(np.mean([r['leak_learned'] for r in lr])),leak_control=float(np.mean([r['leak_control'] for r in lr])),
                noise_FA=pp['noise_FA'],dprime=float(dp),max_participation=int(part.max()),participation_gini=p26.gini(part[part>0]),
                kk_synapses_per_kc=float(d['kk'][kind].sum()/d['kk'][kind].shape[0]) if kind else 0.)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/kc_recurrence';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    sets,inputs=build_all(root);_DATA.update(sets=sets,inputs=inputs)
    jobs=[]
    for name in ('hemibrain_right','FlyWire_right','FlyWire_left'):
        jobs+=[(name,'calyx',gv) for gv in PRIMARY_G]+[(name,'total',gv) for gv in (.3,-.3)]
    jobs+=[('hemibrain_right','calyx',gv) for gv in HB_EXTRA_G]
    jobs+=[(name,'total',gv) for name in ('BANC_right','BANC_left') for gv in PRIMARY_G]
    with Pool(10) as pool:rows=pool.map(_run,jobs)
    f=pd.DataFrame(rows);f.to_csv(out/'metrics.csv',index=False)
    base0=f[f.g==0].set_index('dataset').specific
    f['ratio_to_g0']=f.apply(lambda r:r.specific/base0[r.dataset] if base0.get(r.dataset,0)>0 else np.nan,axis=1)
    prim=f[(f.kk=='calyx')&f.dataset.isin(['hemibrain_right','FlyWire_right','FlyWire_left'])]
    exc=prim[prim.g==.3].ratio_to_g0.to_numpy();inh=prim[prim.g==-.3].ratio_to_g0.to_numpy()
    va='excitatory_increases_leakage' if (len(exc)==3 and (exc>=1.2).all()) else ('little_effect' if (len(exc)==3 and ((exc>=.8)&(exc<=1.2)).all()) else 'partial')
    vb='inhibitory_reduces_leakage' if (len(inh)==3 and (inh<=.8).all()) else ('little_effect' if (len(inh)==3 and ((inh>=.8)&(inh<=1.2)).all()) else 'partial')
    unconverged=f[~f.converged][['dataset','kk','g']].to_dict(orient='records')
    report=dict(H37a=va,H37b=vb,unconverged=unconverged,results=f.round(5).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.set_option('display.width',220)
    print(json.dumps(dict(H37a=va,H37b=vb,unconverged=unconverged),indent=2))
    print(f[['dataset','kk','g','converged','specific','ratio_to_g0','noise_FA','dprime','max_participation','participation_gini','kk_synapses_per_kc']].round(4).to_string())


if __name__=='__main__':main()
