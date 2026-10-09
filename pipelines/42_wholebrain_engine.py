#!/usr/bin/env python3
"""Whole-brain LIF engine built on Shiu et al. 2024 (external/Drosophila_brain_model) with odor-driven ORN Poisson input,
a KC-specific threshold, optional synapse scaling and APL silencing."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SHIU=ROOT/'external/Drosophila_brain_model'
sys.path.insert(0,str(SHIU))
import model as shiu  # noqa: E402
from brian2 import NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network, mV, ms, Hz, prefs, defaultclock  # noqa: E402

prefs.codegen.target='cython'
P=shiu.default_params
SFR_DEFAULT=None


def annotations():
    a=pd.read_csv(ROOT/'data/flywire_v783_neuron_annotations.tsv',sep='\t',low_memory=False)
    comp=pd.read_csv(SHIU/'Completeness_783.csv',index_col=0)
    idx={fid:i for i,fid in enumerate(comp.index)}
    a=a[a.root_id.isin(idx)].copy();a['idx']=a.root_id.map(idx)
    return a,len(comp)


def build(n_neurons,kc_idx,v_th_kc,syn_scale=None,silence_idx=()):
    """Return (Network, neuron group, synapses, spike monitor, poisson group, poisson synapses).
    v_th_kc: KC threshold (mV); syn_scale: dict {(pre_mask_idx, post_mask_idx): factor} applied after construction."""
    con=pd.read_parquet(SHIU/'Connectivity_783.parquet')
    eqs=P['eqs']+'v_th : volt\n'
    neu=NeuronGroup(n_neurons,model=eqs,method='linear',threshold='v > v_th',reset=P['eq_rst'],refractory='rfc',name='default_neurons',
                    namespace={k:v for k,v in P.items() if k!='v_th'})
    neu.v=P['v_0'];neu.g=0*mV;neu.rfc=P['t_rfc'];neu.v_th=P['v_th']
    neu.v_th[np.asarray(kc_idx)]=v_th_kc*mV
    syn=Synapses(neu,neu,'w : volt',on_pre='g += w',delay=P['t_dly'],name='default_synapses')
    i=con['Presynaptic_Index'].to_numpy();j=con['Postsynaptic_Index'].to_numpy();w=con['Excitatory x Connectivity'].to_numpy().astype(float)
    if syn_scale:
        for (pre_set,post_set),f in syn_scale.items():
            m=np.isin(i,list(pre_set))&np.isin(j,list(post_set));w[m]*=f
    if len(silence_idx):w[np.isin(i,list(silence_idx))]=0.
    syn.connect(i=i,j=j);syn.w=w*P['w_syn']
    mon=SpikeMonitor(neu)
    return neu,syn,mon


class Sim:
    """Build once, then run many trials: each trial restores the initial state and sets ORN Poisson rates."""
    def __init__(self,n_neurons,kc_idx,v_th_kc,orn_idx,syn_scale=None,silence_idx=()):
        from brian2 import Network
        self.neu,self.syn,self.mon=build(n_neurons,kc_idx,v_th_kc,syn_scale,silence_idx)
        self.pg=PoissonGroup(len(orn_idx),rates=0*Hz)
        self.ps=Synapses(self.pg,self.neu,on_pre='v += w_in',namespace=dict(w_in=P['w_syn']*P['f_poi']))
        self.ps.connect(i=np.arange(len(orn_idx)),j=np.asarray(orn_idx))
        self.net=Network(self.neu,self.syn,self.mon,self.pg,self.ps);self.net.store('init');self.n=n_neurons

    def trial(self,rates_hz,t_ms,seed):
        from brian2 import seed as bseed
        self.net.restore('init');bseed(seed);self.pg.rates=np.asarray(rates_hz)*Hz
        self.net.run(t_ms*ms)
        return np.bincount(np.asarray(self.mon.i),minlength=self.n)

    def set_weights(self,mask,factor):
        """Scale existing synapse weights (boolean mask over synapses) and store as the new initial state."""
        self.net.restore('init');w=np.asarray(self.syn.w[:]/mV);w[mask]*=factor;self.syn.w[:]=w*mV;self.net.store('init')
