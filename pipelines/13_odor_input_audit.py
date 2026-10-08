#!/usr/bin/env python3
"""Audit OSN consensus response to PN annotation compatibility; no simulation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('06_input_discrimination.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'qc_reports/odor_input_audit';out.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/'data/door/SOURCE.json').read_text())
    for f in source['files']:
        if hashlib.sha256((root/'data/door'/f['file']).read_bytes()).hexdigest()!=f['sha256']:raise ValueError('DoOR data changed')
    mappings=pd.read_csv(root/'data/door/door_mappings.csv',sep=';',index_col=0)
    responses=pd.read_csv(root/'data/door/door_response_matrix.csv',sep=';',index_col=0)
    annotations=pd.read_feather(root/'data/banc_888_meta.feather');edges=pd.read_feather(root/'data/banc_888_edgelist_simple_v2.feather')
    kc=annotations[(annotations.cell_class=='kenyon_cell')&(annotations.side=='right')].root_888.astype(str)
    pn=annotations[(annotations.cell_class=='antennal_lobe_projection_neuron')&(annotations.neurotransmitter_verified=='acetylcholine')].copy()
    pn=pn[pn.root_888.astype(str).isin(edges[edges.post.isin(kc)].pre)]
    rows=[]
    for a in pn.itertuples():
        typename=str(a.cell_type);prefix=typename.split('_')[0];fafb=str(a.fafb_cell_type).split('_')[0]
        candidates=sorted(set(mappings[mappings.glomerulus.eq(prefix)].receptor.astype(str))&set(responses.columns))
        status='mapped'
        if '+' in prefix:status='multiple_glomeruli'
        elif fafb!='nan' and fafb!=prefix:status='annotation_prefix_conflict'
        elif len(candidates)==0:status='no_response_channel'
        elif len(candidates)>1:status='ambiguous_response_channel'
        rows.append(dict(root_888=str(a.root_888),cell_type=typename,fafb_cell_type=str(a.fafb_cell_type),glomerulus=prefix,
                         status=status,response_channel=candidates[0] if status=='mapped' else '',candidates='|'.join(candidates)))
    result=pd.DataFrame(rows);result.to_csv(out/'pn_mapping_audit.csv',index=False)
    channels=sorted(result[result.status=='mapped'].response_channel.unique())
    odors=responses.drop(index='SFR',errors='ignore');coverage=odors[channels].notna().mean(axis=1)
    pd.DataFrame(dict(odor_key=odors.index,known_fraction=coverage.to_numpy(),known_channels=odors[channels].notna().sum(axis=1).to_numpy())).to_csv(out/'odor_coverage.csv',index=False)
    channel_coverage=pd.DataFrame(dict(channel=channels,known_odors=odors[channels].notna().sum().to_numpy(),
        pn_count=result[result.status=='mapped'].response_channel.value_counts().reindex(channels,fill_value=0).to_numpy()))
    channel_coverage.to_csv(out/'channel_coverage.csv',index=False)
    # A descriptive completeness audit, without choosing odor pairs or outcomes.
    kept=channels.copy();subsets=[]
    while kept:
        complete=int(odors[kept].notna().all(axis=1).sum())
        subsets.append(dict(channel_count=len(kept),complete_odor_rows=complete,channels=kept.copy()))
        least=odors[kept].notna().sum().sort_values(kind='stable').index[0];kept.remove(least)
    (out/'completeness_subsets.json').write_text(json.dumps(subsets,indent=2)+'\n')
    assert result.root_888.is_unique
    assert result[result.status!='mapped'].response_channel.eq('').all()
    report=dict(door_source=source,pn_total=len(result),mapping_status=result.status.value_counts().to_dict(),channels=channels,
                odor_rows=len(odors),coverage_counts={str(t):int((coverage>=t).sum()) for t in (.5,.8,.9,1.)},
                response_min=float(odors[channels].min().min()),response_max=float(odors[channels].max().max()),
                sfr_missing_channels=responses.loc['SFR',channels].index[responses.loc['SFR',channels].isna()].tolist(),
                interpretation='Normalized merged OSN/receptor responses, NOT measured PN activity. Potential glomerulus mapping only.',
                assumptions_needed=['Specify OSN-to-PN transfer; direct substitution is a model assumption.',
                                    'Handle spontaneous activity and inhibitory deviations explicitly.',
                                    'Do not equate missing values with zero response.',
                                    'Resolve conflicting/ambiguous annotations or exclude them.',
                                    'Recalibrate input/activity scaling independently of desired discrimination results.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
