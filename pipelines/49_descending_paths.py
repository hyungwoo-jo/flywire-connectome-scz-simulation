#!/usr/bin/env python3
"""Track 3: which descending neurons (DNs) receive the learned value signal, and does the noise-trial leak reach them?

Propagation (linear, signed by predicted transmitter, each stage normalized by the receiving neuron's total input):
dDN = W2 @ (W1 @ dMBON) + W0 @ dMBON, where W1: MBON -> intermediate central neurons, W2: intermediates -> DNs,
W0: MBON -> DN direct. Odor A learned aversively (PPL1 compartments).

Measures (fixed before running):
- For each odor, reference dDN = mean change for weak-A trials. Per DN, consistency = fraction of odors in which the DN is
  among the top 5% |dDN| (normalized per odor), and mean signed direction.
- Leak at DN level: fraction of noise trials (reported present) with cos(dDN, ref) > 0.5 and |dDN| > 0.5 |ref|,
  for learned, random and input-matched learning."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parent))
import downstream_common as dc  # noqa: E402


def norm_signed(e,a,pre_ids,post_ids):
    pi={k:i for i,k in enumerate(pre_ids)};qi={k:i for i,k in enumerate(post_ids)}
    q=e[e.pre.isin(pi)&e.post.isin(qi)]
    sgn=a.set_index('id').reindex(pre_ids).top_nt.map(dc.SIGN).fillna(0.).to_numpy()
    tot=e[e.post.isin(qi)].groupby('post')['count'].sum().reindex(post_ids).fillna(1.).to_numpy()
    r=q.post.map(qi).to_numpy(int);cidx=q.pre.map(pi).to_numpy(int)
    v=q['count'].to_numpy(float)*sgn[cidx]/tot[r]
    return sparse.csr_matrix((v,(r,cidx)),shape=(len(post_ids),len(pre_ids)))


def main():
    out=dc.ROOT/'qc_reports/descending_paths';out.mkdir(parents=True,exist_ok=True)
    S=dc.setup();a,e=S['a'],S['e'];keep=S['keep'];W=S['W']
    dns=a[a.super_class=='descending'].id.tolist()
    inter=sorted(set(e[e.pre.isin(keep)].post)&set(a[a.super_class=='central'].id)-set(keep))
    W1=norm_signed(e,a,keep,inter);W2=norm_signed(e,a,inter,dns);W0=norm_signed(e,a,keep,dns)
    T=(W2@W1+W0)                                   # DN x MBON transfer
    T=T.toarray() if sparse.issparse(T) else np.asarray(T)
    kn,rep=S['kn'],S['rep'];ann=a.set_index('id')
    top_hits=np.zeros(len(dns));signsum=np.zeros(len(dns));rows=[]
    for j,o in enumerate(S['ev']):
        kA=S['kfull'][:,j];rnd,mat=dc.control_patterns(kA,S,int(o));kw=S['kweak'][int(o)]
        ref=(T@(dc.learned_dW(W,S['comp'],kA)@kw)).mean(1);nr=np.linalg.norm(ref)
        if nr==0:continue
        z=np.abs(ref)/nr;top_hits+=z>=np.percentile(z,95);signsum+=np.sign(ref)
        row=dict(odor=int(o))
        for lab,pat in (('learned',kA),('random',rnd),('matched',mat)):
            D=T@(dc.learned_dW(W,S['comp'],pat)@kn);nd=np.linalg.norm(D,axis=0);cos=(ref@D)/(nr*np.maximum(nd,1e-12))
            row[lab]=float(np.mean(rep&(cos>.5)&(nd>.5*nr)))
        rows.append(row)
    n_od=len(rows);f=pd.DataFrame(rows);f.to_csv(out/'leak_per_odor.csv',index=False)
    dn=pd.DataFrame(dict(id=dns,cell_type=ann.reindex(dns).cell_type.to_numpy(),side=ann.reindex(dns).side.to_numpy(),
                         consistency=top_hits/n_od,direction=signsum/n_od,mbon_reach=np.abs(T).sum(1)))
    dn=dn.sort_values('consistency',ascending=False);dn.to_csv(out/'dn_consistency.csv',index=False)
    by_type=dn[dn.consistency>=.5].groupby(dn.cell_type.fillna('unknown')).agg(n=('id','size'),consistency=('consistency','mean'),direction=('direction','mean')).sort_values('n',ascending=False)
    report=dict(dns=len(dns),intermediates=len(inter),odors=n_od,leak=dict(learned=float(f.learned.mean()),random=float(f.random.mean()),matched=float(f.matched.mean())),
                consistent_dns=int((dn.consistency>=.5).sum()),consistent_types=by_type.round(3).reset_index().to_dict(orient='records'),
                top_dns=dn.head(20).round(3).to_dict(orient='records'))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:report[k] for k in ('dns','intermediates','odors','leak','consistent_dns')},indent=2))
    pd.set_option('display.width',200);print(by_type.round(3).to_string());print(dn.head(15).round(3).to_string())


if __name__=='__main__':main()
