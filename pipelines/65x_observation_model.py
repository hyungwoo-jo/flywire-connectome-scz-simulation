#!/usr/bin/env python3
"""Exploratory calcium observation sensitivity audit; not biological model selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p65',Path(__file__).with_name('65_kc_imaging_chan2026.py'))
p65=importlib.util.module_from_spec(spec);sys.modules['p65']=p65;spec.loader.exec_module(p65)
p63=p65.p63;cal=p65.cal;p55=p65.p55
DT=1/5.58;REPEATS=3;TAUS=(.3,.7);GAIN_FACTORS=(.5,1.,2.);SEED=651800


def lowpass(x,tau,dt=DT):
    if tau<=0 or dt<=0:raise ValueError('Positive tau and dt required')
    a=np.exp(-dt/tau);result=np.empty_like(x,dtype=float);state=np.zeros(x.shape[:-1])
    for t in range(x.shape[-1]):
        state=a*state+(1-a)*x[...,t];result[...,t]=state
    return result


def ar_noise(shape,sd,rho,rng):
    if not 0<=rho<1:raise ValueError('rho outside [0,1)')
    out=np.empty(shape);out[...,0]=rng.normal(size=shape[:-1])*sd
    for t in range(1,shape[-1]):out[...,t]=rho*out[...,t-1]+np.sqrt(1-rho*rho)*sd*rng.normal(size=shape[:-1])
    return out


def metrics(B,S,Rp,Rn):
    return dict(rho_oil=p65.rho(B,S),rho_sp=p65.rho(B,Rp),rho_neg=p65.rho(B,Rn),
        mean_B=float(np.mean(B)),mean_S=float(np.mean(S)),mean_Rp=float(np.mean(Rp)),mean_Rn=float(np.mean(Rn)))


def extract_real(root):
    dd=root/'data/chan2026'
    if hashlib.md5((dd/'traces.tar.xz').read_bytes()).hexdigest()!=p65.TRACES_MD5:raise ValueError('Raw archive changed')
    previous=pd.read_csv(root/'qc_reports/kc_imaging_chan2026/real_cells.csv');params={};real=[];templates={}
    for path in sorted((dd/'raw/traces').glob('*/traces.npz')):
        fly=path.parent.name;kind=fly.rstrip('0123456789');d=p65.load_npz(path);r=p65.fly_measures(d)
        old=previous[previous.fly==fly]
        for old_col,new_col in [('B','B'),('S','S'),('Rp','Rp'),('Rn','Rn')]:
            np.testing.assert_allclose(old[old_col],r[new_col],rtol=0,atol=1e-14)
        split='calibration' if int(fly[len(kind):])<2 else 'evaluation'
        real.append(dict(fly=fly,type=kind,split=split,n=r['n'],**metrics(r['B'],r['S'],r['Rp'],r['Rn'])))
        templates[fly]=dict(type=kind,n=r['n'])
        if split!='calibration':continue
        before=int(d['before_onset']);chunks=[];peaks=[]
        for trial in d['tracedict']:
            for o,tr in d['tracedict'][trial].items():
                if np.isnan(tr[0,:before+1]).sum()>p65.BAD_MAX:continue
                raw=p65.normalize(tr,before);chunks.append(raw[:,:before])
                if o in p65.HALLEM:
                    norm=p65.normalize(tr,before,d['tracedict_neuropil'][trial][o]);ok=p65.responds(norm,before,int(d['after_onset']))
                    finite=np.isfinite(norm[:,before:]).any(axis=1)
                    peak=np.full(norm.shape[0],np.nan);peak[finite]=np.nanmax(norm[finite,before:],axis=1)
                    peaks.extend(peak[ok&np.isfinite(peak)].tolist())
        joined=np.stack(chunks)
        sd=np.nanstd(joined,axis=2).ravel();sd=sd[np.isfinite(sd)&(sd>0)]
        x=joined-np.nanmean(joined,axis=2,keepdims=True)
        lag=float(np.nanmean(x[:,:,:-1]*x[:,:,1:])/np.nanmean(x*x))
        params.setdefault(kind,[]).append(dict(sd=float(np.median(sd)),rho=float(np.clip(lag,0,.95)),peak=float(np.median(peaks))))
    fitted={kind:{key:float(np.median([p[key] for p in vals])) for key in ('sd','rho','peak')} for kind,vals in params.items()}
    return pd.DataFrame(real),templates,fitted


def build_banks(root):
    manifest=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    for filename,digest in manifest['source']['receptor_csv_sha256'].items():
        if hashlib.sha256((root/'data/door/receptors'/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Receptor data changed: '+filename)
    hashes=json.loads((root/'qc_reports/input_discrimination/manifest.json').read_text())['data_sha256']
    for filename,digest in hashes.items():
        if hashlib.sha256((root/'data'/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Connectome data changed: '+filename)
    odors,sfr,names=cal.load_hallem(root);receptors=list(odors.columns);delta=odors.to_numpy().T
    c=cal.load_partial_circuit(root,receptors);c['kk']=None
    # Existing step-34 helper supplies the same calibration noise as steps 63-65.
    p55.p24.SEEDS.update(noise=p55.p25.SEEDS['noise'],signal=p55.p25.SEEDS['signal'])
    g=p65.p34.make_g(c,np.ones(len(c['inh']))/len(c['inh']),receptors,delta,sfr.to_numpy(),cal.split_odors(list(odors.index)))
    m=g['m'];calib=cal.split_odors(list(odors.index));xcal=cal.pn_drive(c,receptors,delta[:,calib])
    idx=[names.tolist().index(n) for n in p65.HALLEM.values()]
    signal=np.concatenate([delta[:,idx],np.zeros((len(receptors),1))],axis=1)
    labels=np.r_[np.repeat(np.arange(6),3),np.full(1024,6)]
    x=cal.pn_drive(c,receptors,cal.sample_orn(signal,sfr.to_numpy(),labels,.5,np.random.default_rng(SEED)))
    meta=pd.read_feather(root/'data/banc_888_meta.feather');kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side=='right')].copy()
    kc['id']=kc.root_888.astype(str);kc=kc.set_index('id').sort_index();raw=kc.cell_sub_class.fillna(kc.cell_type).fillna('').astype(str)
    types=np.array(['apbp' if "a'b'" in t or t=='KCapbp' else 'ab' if t.startswith('KCab') else 'g' if t.startswith('KCg') else 'other' for t in raw])
    assert len(types)==len(c['inh'])
    banks={}
    for kind in p65.CONDS:
        mat=m if kind!='flat' else p55.p29.compensate(m,1.)
        E=mat@xcal;theta=cal.calibrate(E,c['inh'],c['drive'],.10)
        p63._D['m']=mat;th=p63.thresholds(kind,E,np.asarray(mat@g['x_cal']),c,theta)[:,None]
        k=p55.respond(mat@x,c,th)
        B=np.sum(np.stack([(k[:,i*3:(i+1)*3]>1e-8).sum(1)>=2 for i in range(6)]),axis=0)
        banks[kind]=dict(k=k,B=B,sp=(k[:,18:]>1e-8).mean(1),theta=th)
        print('latent bank',kind,'complete',flush=True)
    return banks,types


def observe(bank,indices,params,tau,gain,rng):
    # Each synthetic trial has a warm-up, 20 pre-onset and 9 response frames.
    n=len(indices);trials=21;frames=69;warm=40;pre=20;after=9
    b=bank['k'][indices,18:];choose=rng.integers(b.shape[1],size=(trials,(frames+2)//3))
    latent=b[:,choose].transpose(1,0,2).repeat(3,axis=2)[:,:,:frames]
    for o in range(6):
        for r in range(3):latent[o*3+r,:,warm+pre:]=bank['k'][indices,o*3+r][:,None]
    signal=lowpass(latent,tau);noise=ar_noise(latent.shape,params['sd'],params['rho'],rng)
    # Constant neuropil makes its subtraction zero, while retaining author clipping.
    raw=1+gain*signal+noise;raw=raw[:,:,warm:];td={0:{}};tn={0:{}}
    for i,o in enumerate(list(p65.HALLEM)+['oil']):
        for rep in range(3):
            td.setdefault(rep,{})[o]=raw[i*3+rep];tn.setdefault(rep,{})[o]=np.ones((n,pre+after))
    responses={};pre_chunks=[]
    for rep in td:
        for o,tr in td[rep].items():
            responses.setdefault(o,[]).append(p65.responds(p65.normalize(tr,pre,tn[rep][o]),pre,after))
            pre_chunks.append(p65.normalize(tr,pre)[:,:pre])
    B=sum((np.sum(responses[o],axis=0)>=2).astype(int) for o in p65.HALLEM)
    S=np.mean(responses['oil'],axis=0);Rp,Rn=p65.event_rates_nan(np.stack(pre_chunks),3.)
    return metrics(B,S,Rp,Rn)


def bootstrap_summary(frame,seed=SEED):
    rng=np.random.default_rng(seed);result={}
    # Preserve the four subtype/all group counts in each animal-level resample.
    for metric in ('rho_oil','rho_sp','rho_neg','mean_B','mean_S','mean_Rp','mean_Rn'):
        grouped=[g[metric].to_numpy() for _,g in frame.groupby('type')]
        draws=np.concatenate([x[rng.integers(len(x),size=(10000,len(x)))] for x in grouped],axis=1)
        result[metric]=dict(mean=float(frame[metric].mean()),lo=float(np.nanquantile(np.nanmean(draws,axis=1),.025)),hi=float(np.nanquantile(np.nanmean(draws,axis=1),.975)))
    return result


def audit_solvent_units(root,out):
    manifest=json.loads((root/'qc_reports/hallem_calibration/manifest.json').read_text())
    odor_table,_,_=cal.load_hallem(root);rows=[]
    for receptor in odor_table.columns:
        path=root/'data/door/receptors'/f'{receptor}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['source']['receptor_csv_sha256'][path.name]:raise ValueError('Receptor data changed')
        values=pd.read_csv(path,sep=';',index_col=0).set_index('InChIKey')['Hallem.2004.EN']
        solvent=float(values.loc['solvent']);sfr=float(values.get('SFR',np.nan))
        rows.append(dict(receptor=receptor,solvent_2004=solvent,sfr_2004=sfr,within_2004_difference=solvent-sfr,
            minimum_2004=float(values.min()),negative_values=int((values<0).sum())))
    result=pd.DataFrame(rows);result.to_csv(out/'solvent_unit_audit.csv',index=False)
    return dict(channels=len(result),available_same_study_sfr=int(result.sfr_2004.notna().sum()),
        metadata_sha256=hashlib.sha256((root/'data/door/door_dataset_info.csv').read_bytes()).hexdigest(),
        warning='Hallem 2004 EN baseline subtraction is not uniformly guaranteed. Do not use all raw solvent values as delta input in step66.')


def plot_summary(summary,observed,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    metrics_order=('rho_sp','mean_B','mean_Rp')
    conditions=list(p65.CONDS)
    for ax,metric in zip(axes,metrics_order):
        lo,hi=observed[metric]['lo'],observed[metric]['hi']
        ax.axhspan(lo,hi,color='gray',alpha=.2,label='12 observed flies, bootstrap band')
        ax.axhline(observed[metric]['mean'],color='black',linewidth=1)
        for i,kind in enumerate(conditions):
            g=summary[summary.condition==kind]
            for j,tau in enumerate(TAUS):
                sel=g[g.tau==tau].sort_values('gain_factor')
                ax.scatter([i+(v-1)*.15+(j-.5)*.07 for v in sel.gain_factor],sel[metric],s=30,marker='o' if j==0 else '^')
        ax.set_xticks(range(4),conditions,rotation=25);ax.set_title(metric)
        if metric=='rho_sp':ax.set_ylim(-.15,.65)
    axes[0].legend(fontsize=7);fig.suptitle('Exploratory observation audit: all 6 settings per circuit, BANC right only')
    fig.tight_layout();fig.savefig(out/'observation_audit.png',dpi=180);plt.close(fig)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/observation_audit';out.mkdir(parents=True,exist_ok=True)
    solvent_audit=audit_solvent_units(root,out)
    real,templates,fitted=extract_real(root);real.to_csv(out/'real_per_fly.csv',index=False)
    (out/'calibration.json').write_text(json.dumps(fitted,indent=2)+'\n')
    banks,types=build_banks(root);positive=banks['original']['k'][:,:18];median_k=float(np.median(positive[positive>1e-8]))
    rows=[];latent=[]
    for kind,bank in banks.items():
        for repeat in range(REPEATS):
            for fly in real[real.split=='evaluation'].fly:
                t=templates[fly];pool=np.arange(len(types)) if t['type']=='all' else np.flatnonzero(types==t['type'])
                seed=SEED+1000*repeat+int(real.index[real.fly==fly][0])
                indices=np.random.default_rng(seed).choice(pool,t['n'],replace=False)
                latent.append(dict(condition=kind,repeat=repeat,fly=fly,type=t['type'],rho_sp=p65.rho(bank['B'][indices],bank['sp'][indices])))
                gain0=fitted[t['type']]['peak']/median_k
                for tau in TAUS:
                    for factor in GAIN_FACTORS:
                        value=observe(bank,indices,fitted[t['type']],tau,gain0*factor,np.random.default_rng(seed+50000))
                        rows.append(dict(condition=kind,repeat=repeat,fly=fly,type=t['type'],tau=tau,gain_factor=factor,gain=gain0*factor,**value))
        print('observations',kind,'complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'model_per_fly.csv',index=False);pd.DataFrame(latent).to_csv(out/'latent_per_fly.csv',index=False)
    metrics_cols=['rho_oil','rho_sp','rho_neg','mean_B','mean_S','mean_Rp','mean_Rn']
    summaries=f.groupby(['condition','tau','gain_factor'])[metrics_cols].mean().reset_index()
    valid=f.groupby(['condition','tau','gain_factor']).rho_sp.count().reset_index(name='n_valid_sp');summaries=summaries.merge(valid)
    summaries.to_csv(out/'model_summary.csv',index=False)
    observed=bootstrap_summary(real[real.split=='evaluation'])
    for metric in metrics_cols:
        summaries[f'{metric}_in_real_interval']=summaries[metric].between(observed[metric]['lo'],observed[metric]['hi'])
    summaries['joint_marginal_coverage']=summaries[[f'{m}_in_real_interval' for m in ('rho_sp','mean_B','mean_Rp','mean_Rn')]].all(axis=1)
    summaries.to_csv(out/'comparison.csv',index=False)
    plot_summary(summaries,observed,out)
    report=dict(solvent_audit=solvent_audit,date='2026-10-18',plan='docs/OBSERVATION_AUDIT_PLAN.md',scope='BANC_right exploratory observation sensitivity; previously inspected animals, no independent validation.',
        calibration_flies=real[real.split=='calibration'].fly.tolist(),evaluation_flies=real[real.split=='evaluation'].fly.tolist(),real_evaluation=observed,
        latent_median_positive_k=median_k,calibration=fitted,tau_grid=TAUS,gain_factors=GAIN_FACTORS,repeats=REPEATS,model_fly_rows=len(f),
        latent_summary=pd.DataFrame(latent).groupby('condition').rho_sp.mean().to_dict(),n_joint_summary_matches=int(summaries.joint_marginal_coverage.sum()),
        data_md5=p65.TRACES_MD5,source='https://zenodo.org/records/21821328',
        limitations=['Relative rate lowpass, not identified spike-to-GCaMP6f model.',
            '0.5s ORN samples held 3 frames; unknown biological temporal covariance.',
            'AR noise fitted on real baseline includes biological activity; possible double counting.',
            'Finite common latent bank reused; repeats/hemispheres are not independent animals.',
            'No neuropil dynamics, motion artifacts, indicator saturation or missing frames simulated.',
            'Marginal summary intervals are descriptive, not a joint test or model-selection posterior.',
            'Oil input remains zero; step66 solvent input question is not answered here.',
            'KC subtype mappings coarse, one connectome specimen.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(real=observed,latent=report['latent_summary'],joint_matches=report['n_joint_summary_matches']),indent=2))

if __name__=='__main__':main()
