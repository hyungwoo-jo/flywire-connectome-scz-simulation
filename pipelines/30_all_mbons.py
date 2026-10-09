#!/usr/bin/env python3
"""Learned value leakage for every MBON with >=100 distinct KC inputs, both hemispheres."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('p25',Path(__file__).with_name('25_learned_value_leakage.py'))
p25=importlib.util.module_from_spec(spec);spec.loader.exec_module(p25)
p24=p25.p24;p23=p24.p23;cal=p24.cal

MIN_KC=100
BOOT_SEED0=300300
N_BOOT=1000


def mbon_table(root,side):
    m=pd.read_feather(root/'data/banc_888_meta.feather');e=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    kcm=m[(m.cell_class=='kenyon_cell')&(m.side==side)].copy();kcm['id']=kcm.root_888.astype(str);kcm=kcm.set_index('id').sort_index()
    kc=list(kcm.index);idx={k:i for i,k in enumerate(kc)};sub=kcm.cell_sub_class.fillna(kcm.cell_type).fillna('unknown').to_numpy(str)
    mb=m[(m.side==side)&m.cell_type.fillna('').str.startswith('MBON')].copy();mb['id']=mb.root_888.astype(str)
    ed=e[e.pre.isin(kc)&e.post.isin(mb.id)];n=ed.groupby('post').pre.nunique();keep=sorted(n[n>=MIN_KC].index)
    out=[]
    for i,mid in enumerate(keep):
        q=ed[ed.post==mid];w=np.zeros(len(kc));np.add.at(w,q.pre.map(idx).to_numpy(int),q['count'].to_numpy(float))
        shares={t:float(w[sub==t].sum()/w.sum()) for t in ('KCg','KCab',"KCa'b'")}
        out.append(dict(order=i,mbon_id=mid,cell_type=mb.set_index('id').loc[mid,'cell_type'],kc_inputs=int(n[mid]),w=w/w.sum(),**{f'share_{k}':v for k,v in shares.items()}))
    return out


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/all_mbons';out.mkdir(parents=True,exist_ok=True)
    p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])
    rows=[];res={}
    for side in p23.SIDES:
        g=p24.setup(root,side);c=g['c'];th=g['theta']
        kt,_=p24.kc(g['m'],g['x_full'],c,th);kn,_=p24.kc(g['m'],g['x_noise'],c,th);kw,_=p24.kc(g['m'],g['x_weak'],c,th)
        crit=float(np.percentile(p24.kc(g['m'],g['x_cal'],c,th)[0].sum(0),95))
        for mb in mbon_table(root,side):
            lr=p25.leakage(g,kt,kn,kw,crit,mb['w']);d=np.array([r['leak_learned']-r['leak_control'] for r in lr])
            rng=np.random.default_rng(BOOT_SEED0+mb['order']+(0 if side=='right' else 1000))
            boot=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(N_BOOT)];ci=np.percentile(boot,[2.5,97.5])
            rows.append(dict(side=side,**{k:v for k,v in mb.items() if k!='w'},specific=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                             leak_learned=float(np.mean([r['leak_learned'] for r in lr])),leak_control=float(np.mean([r['leak_control'] for r in lr]))))
        q=pd.DataFrame([r for r in rows if r['side']==side]);res[side]=dict(mbons=len(q),positive=int((q.ci_low>0).sum()),P=float((q.ci_low>0).mean()),
            median_specific=float(q.specific.median()),mbon11_specific=float(q[q.cell_type=='MBON11'].specific.iloc[0]) if (q.cell_type=='MBON11').any() else None)
    f=pd.DataFrame(rows);f.to_csv(out/'per_mbon.csv',index=False)
    R,L=res['right'],res['left']
    v='general' if (R['P']>=.75 and L['P']>=.75) else ('restricted' if (R['P']<=.25 and L['P']<=.25) else 'partial')
    from scipy import stats
    for side in p23.SIDES:
        q=f[f.side==side];res[side]['rho_kc_inputs']=float(stats.spearmanr(q.kc_inputs,q.specific).statistic)
        for t in ('KCg','KCab',"KCa'b'"):res[side][f'rho_share_{t}']=float(stats.spearmanr(q[f'share_{t}'],q.specific).statistic)
    report=dict(verdict=v,results=res,min_kc=MIN_KC)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,side in zip(axes,('right','left')):
        q=f[f.side==side].sort_values('specific')
        ax.errorbar(range(len(q)),q.specific*100,yerr=[(q.specific-q.ci_low)*100,(q.ci_high-q.specific)*100],fmt='o',color='#0072B2')
        ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(len(q)));ax.set_xticklabels(q.cell_type,rotation=90,fontsize=6);ax.set_title(f'{side}: {res[side]["positive"]}/{res[side]["mbons"]} MBONs CI>0')
    axes[0].set_ylabel('odor-specific value leakage (%p)');fig.suptitle('30: leakage across MBONs');fig.tight_layout();fig.savefig(out/'all_mbons.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2));print(f[['side','cell_type','kc_inputs','specific','ci_low','share_KCg','share_KCab',"share_KCa'b'"]].round(4).to_string())


if __name__=='__main__':main()
