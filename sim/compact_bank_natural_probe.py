"""Native probes for separately authored short-travel engagement candidates.

Unchanged constrained_trial physics and confirmed joint bounds. These runs do
not inherit a defence certificate from the old ready posture or admit a bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .motion_spline import C2JointSpline


def write(path,value):
    Path(path).write_text(json.dumps(_finite_json(value),indent=2)+'\n')


def run(source,output,*,region=None,attacker='a1',time_scale=1.):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve native trials')
    if attacker not in ('a1','a2'):raise ValueError('Explicit physical attacker slot')
    if not np.isfinite(time_scale) or time_scale<1.:raise ValueError('No implicit acceleration')
    raw=source.read_bytes();report=json.loads(raw)
    if not report.get('candidate_ready_sampled_pass'):
        raise ValueError('Ready geometry must pass first')
    ready=np.asarray(report['center_qpos_rad'])
    defender='a2' if attacker=='a1' else 'a1'
    metadata={'candidate_only':True,'attacker_slot':attacker,'region':region,
        'reference_source_sha256':hashlib.sha256(raw).hexdigest(),
        'reference_source':str(source),'new_ready_requires_new_defence_validation':True,
        'mirror_method':'exchange physical slots, keep identical local joint coordinates',
        'time_scale':float(time_scale)}
    if region is None:
        duration=5.
        programs={slot:hold_reference(ready,.75,1.) for slot in ('a1','a2')}
        metadata['trial_kind']='both_slots_hold_new_ready'
    else:
        candidates=[r for r in report['path_trials'] if r['region']==region and r['sampled_reference_pass']]
        if not candidates:raise ValueError('No sampled candidate for this target')
        selected=min(candidates,key=lambda r:(r['j0_roundtrip_deg'],r['all_joint_travel_deg']))
        curve=C2JointSpline(np.asarray(selected['knot_time_s'])*time_scale,selected['knot_qpos'])
        attack=ScheduledReference(curve,.75)
        programs={attacker:attack,defender:hold_reference(ready,.75,curve.duration_s)}
        duration=float(np.ceil(max(6.,attack.end_s+2.5)*1000)/1000)
        metadata.update(trial_kind='attack_vs_same_new_ready',action_name=selected['name'])
    own_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    native=run_constrained_trial(programs,output/'native',duration_s=duration,
                                 metadata=metadata,progress=print)
    audit=audit_recorded_triangle(output/'native',report['scoring'],region or 'left',attacker)
    source_unchanged=(hashlib.sha256(source.read_bytes()).hexdigest()==metadata['reference_source_sha256']
                      and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own_hash)
    result={'candidate_only':True,'admitted_to_training':False,'metadata':metadata,
        'source_unchanged':source_unchanged,'native_execution_pass':native['native_execution_pass'],
        'failure':native['failure'],'triangle_audit':audit,
        'hold_pass':bool(region is None and source_unchanged and native['native_execution_pass']
                         and not audit['all_entries'] and not audit['initial_overlap']
                         and not audit['sweep_failures']),
        'attack_probe_pass':bool(region is not None and source_unchanged and audit['reference_candidate_pass']),
        'new_defence_validated':False}
    write(output/'result.json',result)
    print(json.dumps({k:result[k] for k in ('hold_pass','attack_probe_pass','failure','native_execution_pass')},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'outputs/compact_bank_v1_natural/engagement01/report.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--region',choices=['left','right','upper'])
    p.add_argument('--attacker',choices=['a1','a2'],default='a1')
    p.add_argument('--time-scale',type=float,default=1.)
    a=p.parse_args();run(a.source,a.output,region=a.region,attacker=a.attacker,time_scale=a.time_scale)
