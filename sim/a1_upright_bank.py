"""Assemble exact upright-parry/low-guard transfer references without training.

The graph is an idle-opponent chaining candidate, not proof that a side switch
can defend an incoming attack.  Source references are immutable inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .a1_recovery_geometry import exact_scene, screen_path, static_screen, write
from .assembly import ROOT
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_native import load_action
from .triangle_task import TriangleGeometry


def build(output, *, right_source='outputs/a1_upright_right/direct03/report.json',
          right_action='parry_right_upright_direct'):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Keep earlier candidate banks and failures')
    specs=[('outputs/a1_upright_left/search03/report.json',['parry_left_upright_00']),
           (right_source,[right_action]),
           ('outputs/a1_guard_defence_screen/low_transfer06/report.json',
            ['switch_positive_low_to_negative_low','switch_negative_low_to_positive_low'])]
    sources={str(ROOT/path):(ROOT/path).read_bytes()for path,_ in specs}
    decoded={path:json.loads(data)for path,data in sources.items()}
    first=decoded[str(ROOT/specs[0][0])]
    guards,actions,provenance={}, {}, {}
    for path,names in specs:
        full=str(ROOT/path);source=decoded[full]
        if source['scoring']!=first['scoring']:
            raise ValueError('Scoring contracts must match exactly; no automatic rewrite')
        for name,node in source['guards'].items():
            if name in guards and guards[name]['qpos_rad']!=node['qpos_rad']:
                raise ValueError('Conflicting exact same-name guard '+name)
            guards.setdefault(name,copy.deepcopy(node))
        for name in names:
            row,curve=load_action(source,name)
            if name in actions:raise ValueError('Duplicate action '+name)
            actions[name]=copy.deepcopy(row)
            actions[name]['native_dynamics_tested']=False
            actions[name]['admitted_to_training']=False
            provenance[name]={'source_report':full,'source_report_sha256':hashlib.sha256(sources[full]).hexdigest(),
                'exact_reference_preserved':True,'source_candidate_pass':row['candidate_pass']}
    sequence=['parry_left_upright_00','switch_positive_low_to_negative_low',
              right_action,'switch_negative_low_to_positive_low']
    for a,b in zip(sequence,sequence[1:]+sequence[:1]):
        if actions[a]['end_guard']!=actions[b]['start_guard']:
            raise ValueError('Missing explicit route: '+a+' -> '+b)
    scene_path=ROOT/'outputs/a1_recovery_geometry/withdraw01/scene.xml'
    scene=exact_scene(scene_path)
    geometry=TriangleGeometry.from_mapping({k:first['scoring'][k]for k in ('base_distance','regions')})
    center=np.asarray(guards['center']['qpos_rad'])
    bounds=model_joint_ranges_rad(load_joint_constraints())
    model=build_limited_floor().model
    output.mkdir(parents=True)
    results={}
    for name,row in actions.items():
        curve=C2JointSpline(row['knot_time_s'],row['knot_qpos'])
        bounds_pass=bool(np.all(curve.bezier_controls>=bounds[:,0])and np.all(curve.bezier_controls<=bounds[:,1]))
        endpoint_pass=bool(np.array_equal(curve.knot_qpos[0],guards[row['start_guard']]['qpos_rad'])and
                           np.array_equal(curve.knot_qpos[-1],guards[row['end_guard']]['qpos_rad']))
        samples=max(501,int(np.ceil(curve.duration_s/.01))+1)
        collision=screen_path(scene,geometry,curve,center,samples=samples,both_slots=True)
        passed=bool(endpoint_pass and bounds_pass and collision['all_clear'])
        row['source_provenance']=provenance[name]
        row['center_reference_screen']=collision
        row['candidate_pass']=row['sampled_reference_pass']=passed
        results[name]={'bounds_pass':bounds_pass,'exact_endpoint_pass':endpoint_pass,
            'center_collision_pass':collision['all_clear'],'candidate_pass':passed,
            'static_screen':static_screen(model,curve,101)}
        ts=np.linspace(0,curve.duration_s,samples)
        np.savez_compressed(output/(name+'.npz'),time_s=ts,qpos=curve.sample(ts),
                            knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        print(json.dumps({'action':name,**{k:v for k,v in results[name].items()if k!='static_screen'}}),flush=True)
    outgoing={}
    for name,row in actions.items():outgoing.setdefault(row['start_guard'],[]).append(name)
    report={'schema_version':1,'status':'upright_parry_connected_idle_opponent_candidate',
        'scoring':copy.deepcopy(first['scoring']),'guards':guards,'actions':actions,
        'candidate_only':True,'native_dynamics_tested':False,'admitted_to_training':False,
        'graph_candidate_pass':all(x['candidate_pass']for x in results.values()),
        'source_reports_sha256':{p:hashlib.sha256(v).hexdigest()for p,v in sources.items()},
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scene_source':str(scene_path),'scene_source_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        'source_actions_preserved':True,'same_name_guards_exact_qpos':True,
        'idle_opponent_guard':'center','explicit_cycle':sequence,'outgoing_actions':outgoing,
        'initial_guard':'left_guard_positive','checks':results,
        'nominal_cycle_duration_s':sum(float(row['knot_time_s'][-1])for row in actions.values()),
        'scope':'Reference graph and held-center collision check only; next stage requires actual settled-state chained native execution.',
        'limitations':['This subset is not the complete six-attack/three-defence A1 bank.',
            'A held-center chaining pass cannot prove side-switching while under attack.',
            'Neither source native parry evidence nor direct-entry pose guarantees transfer to another opponent state.',
            'No implicit bridge, reset, wrapping, clipping or shortened action is introduced.',
            'A1/A2 training or arbitrary reactive calling is not admitted.']}
    report['source_unchanged']=all(Path(p).read_bytes()==v for p,v in sources.items())
    write(output/'report.json',report)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'scene.xml').write_bytes(scene_path.read_bytes())
    print(json.dumps({'graph_candidate_pass':report['graph_candidate_pass'],
                      'source_unchanged':report['source_unchanged'],
                      'cycle_duration_s':report['nominal_cycle_duration_s']}),flush=True)
    return report


def compare_existing(source,output):
    """Three finite public held-guard controls, not a pose search."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve the earlier center failure')
    raw=source.read_bytes();original=json.loads(raw)
    report=copy.deepcopy(original)
    # Restore only the exact original motion row, never the failed center gate.
    for name,row in report['actions'].items():
        path=Path(row['source_provenance']['source_report'])
        src=json.loads(path.read_bytes())
        original_row=src['actions'][name]
        if row['knot_qpos']!=original_row['knot_qpos'] or row['knot_time_s']!=original_row['knot_time_s']:
            raise ValueError('Unexpected motion change in first graph')
        row['candidate_pass']=row['sampled_reference_pass']=False
    output.mkdir(parents=True)
    geometry=TriangleGeometry.from_mapping({k:report['scoring'][k]for k in ('base_distance','regions')})
    scene=exact_scene(report['scene_source'])
    comparisons={}
    for guard in ('left_high_negative_robust','left_guard_positive','right_low_negative_robust'):
        comparisons[guard]={}
        q=np.asarray(report['guards'][guard]['qpos_rad'])
        for name,row in report['actions'].items():
            curve=C2JointSpline(row['knot_time_s'],row['knot_qpos'])
            screen=screen_path(scene,geometry,curve,q,
                samples=max(501,int(np.ceil(curve.duration_s/.01))+1),both_slots=True)
            comparisons[guard][name]=screen
            print(json.dumps({'guard':guard,'action':name,'all_clear':screen['all_clear']}),flush=True)
        write(output/'fixed_guard_comparisons.json',comparisons)
    passing=[name for name,actions in comparisons.items()if all(s['all_clear']for s in actions.values())]
    selected=passing[0]if passing else None
    report['status']='upright_cycle_explicit_public_held_guard_candidate'if selected else 'upright_cycle_fixed_guard_comparison_failed'
    report['graph_candidate_pass']=bool(selected)
    report['native_dynamics_tested']=report['admitted_to_training']=False
    report['previous_center_graph_source']=str(source)
    report['previous_center_graph_sha256']=hashlib.sha256(raw).hexdigest()
    report['center_failure_retained']=True
    report['fixed_guard_comparison']=comparisons
    report['passing_held_guard_conditions']=passing
    report['idle_opponent_guard']=selected
    report['held_opponent_condition']={'guard':selected,
        'publicly_declared_before_trial':True,'opponent_reactive':False,
        'geometry_matches_source_guard_without_relabeling_center':True,
        'arbitrary_opponent_or_universal_idle_claim':False}
    report['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if selected:
        for name,row in report['actions'].items():
            row['candidate_pass']=row['sampled_reference_pass']=True
            row['selected_held_guard_screen']=comparisons[selected][name]
            row['candidate_pass_scope']='Only full reference against declared fixed '+selected+'; not center or arbitrary opponent.'
            row['native_dynamics_tested']=row['admitted_to_training']=False
            curve=C2JointSpline(row['knot_time_s'],row['knot_qpos']);ts=np.linspace(0,curve.duration_s,501)
            np.savez_compressed(output/(name+'.npz'),time_s=ts,qpos=curve.sample(ts),
                                knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    report['scope']='Explicit held-guard conditional cycle only. Preserve original center identity and select the opponent guard in the runner; no reactive switching evidence.'
    report['limitations'].append('The original held-center collision failure remains valid; another public held guard is not a repaired universal idle benchmark.')
    report['source_unchanged']=source.read_bytes()==raw
    write(output/'report.json',report)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'scene.xml').write_bytes(Path(report['scene_source']).read_bytes())
    print(json.dumps({'passing_held_guard_conditions':passing,'selected':selected,'graph_candidate_pass':bool(selected)}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--compare-source')
    p.add_argument('--right-source',default='outputs/a1_upright_right/direct03/report.json')
    p.add_argument('--right-action',default='parry_right_upright_direct')
    args=p.parse_args()
    if args.compare_source:compare_existing(args.compare_source,args.output)
    else:build(args.output,right_source=args.right_source,right_action=args.right_action)
