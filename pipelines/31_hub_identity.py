#!/usr/bin/env python3
"""Who are the hub KCs: subtype composition and input sources of high-participation KCs."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p25',Path(__file__).with_name('25_learned_value_leakage.py'))
p25=importlib.util.module_from_spec(spec);spec.loader.exec_module(p25)
p24=p25.p24;p23=p24.p23;cal=p24.cal

C_B=('DM2','DM4','VM2','VM3')
TOP=90


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/hub_identity';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    meta=pd.read_feather(root/'data/banc_888_meta.feather');res={};frames=[]
    for side in p23.SIDES:
        g=p24.setup(root,side);c=g['c'];th=g['theta']
        kt,_=p24.kc(g['m'],g['x_full'],c,th);kn,_=p24.kc(g['m'],g['x_noise'],c,th)
        part=(kt>1e-8).sum(1);noise=(kn>1e-8).mean(1)
        kc=meta[(meta.cell_class=='kenyon_cell')&(meta.side==side)].copy();kc['id']=kc.root_888.astype(str);kc=kc.set_index('id').sort_index()
        sub=kc.cell_sub_class.fillna(kc.cell_type).fillna('unknown').to_numpy(str)
        m=g['m'].tocsr();total=np.asarray(m.sum(1)).ravel();claws=np.diff(m.indptr)
        cb_cols=np.flatnonzero(np.isin(c['mapped'].glomerulus.to_numpy(),C_B));cb=np.asarray(m[:,cb_cols].sum(1)).ravel()
        resp=part>0;cut=np.percentile(part[resp],TOP);hub=resp&(part>=cut)
        f=pd.DataFrame(dict(side=side,kc=kc.index,subtype=sub,participation=part,noise_active=noise,input_total=total,claws_mapped=claws,
                            cb_share=np.where(total>0,cb/np.where(total>0,total,1),0.),hub=hub,responsive=resp))
        frames.append(f)
        ab_h=int(((sub=='KCab')&hub).sum());n_h=int(hub.sum());ab_r=int(((sub=='KCab')&resp).sum());n_r=int(resp.sum())
        odds,pa=stats.fisher_exact([[ab_h,n_h-ab_h],[ab_r-ab_h,(n_r-n_h)-(ab_r-ab_h)]],alternative='greater')
        pb=stats.mannwhitneyu(f[hub].cb_share,f[resp&~hub].cb_share,alternative='greater').pvalue
        bysub=f[resp].groupby('subtype').agg(n=('kc','size'),hub_fraction=('hub','mean'),participation_median=('participation','median'),
                                             noise_active_mean=('noise_active','mean'),input_total_median=('input_total','median')).round(4)
        res[side]=dict(hub_cut=float(cut),hubs=n_h,responsive=n_r,ab_share_hubs=ab_h/n_h,ab_share_responsive=ab_r/n_r,fisher_odds=float(odds),fisher_p=float(pa),
                       cb_share_hub_median=float(f[hub].cb_share.median()),cb_share_other_median=float(f[resp&~hub].cb_share.median()),mwu_p=float(pb),
                       hub_input_total_median=float(f[hub].input_total.median()),other_input_total_median=float(f[resp&~hub].input_total.median()),
                       hub_claws_median=float(f[hub].claws_mapped.median()),other_claws_median=float(f[resp&~hub].claws_mapped.median()),
                       hub_noise_active=float(f[hub].noise_active.mean()),other_noise_active=float(f[resp&~hub].noise_active.mean()),
                       by_subtype=bysub.reset_index().to_dict(orient='records'))
    pd.concat(frames).to_csv(out/'per_kc.csv',index=False)
    R,L=res['right'],res['left']
    v=dict(H31a='hubs_enriched_in_alphabeta' if (R['fisher_p']<.05 and L['fisher_p']<.05) else 'not_supported',
           H31b='hub_input_from_overconvergent_glomeruli' if (R['mwu_p']<.05 and L['mwu_p']<.05) else 'not_supported')
    report=dict(verdicts=v,results=res,top_percentile=TOP,C_B=C_B)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fa=pd.concat(frames);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,side in zip(axes,('right','left')):
        q=fa[(fa.side==side)&fa.responsive]
        for t,col in (('KCab','#D55E00'),('KCg','#0072B2'),("KCa'b'",'#009E73')):
            ax.hist(q[q.subtype==t].participation,bins=range(0,75,3),alpha=.5,color=col,label=t)
        ax.axvline(res[side]['hub_cut'],color='k',ls=':',label='hub cut (top 10%)');ax.set_xlabel('odors responded to (of 73)');ax.set_title(side);ax.legend(fontsize=7)
    fig.suptitle('31: participation by KC subtype');fig.tight_layout();fig.savefig(out/'hub_identity.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
