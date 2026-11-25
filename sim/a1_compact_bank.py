"""Build and inspect a seven-ID candidate bank, separate from the old trainer.

The source of an action can be native-tested without the assembled bank being
qualified. Bridging, rescheduling, mirror coverage and interactive choices all
need new evidence. No function here grants training admission or teleports q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .candidate_graph import load_candidate_graph
from .joint_constraints import JOINT_NAMES, load_joint_constraints, model_joint_ranges_rad
from .motion_spline import C2JointSpline


ACTION_NAMES = ('wait', 'attack_left', 'attack_right', 'attack_upper',
                'defence_left', 'defence_right', 'defence_upper')
REQUIRED_GATES = (
    'same_ready_and_full_curve_constraints',
    'all_three_attacks_native_hit_in_both_slots',
    'all_three_full_window_defences_in_both_slots',
    'local_left_right_directional_coverage',
    'current_observation_reaction_windows',
    'early_late_wrong_direction_counterexamples',
    'repeated_attack_and_attack_defence_transitions',
    'no_dominant_wait_attack_or_guard_exploit',
    'shared_online_triangle_referee_and_control',
    'new_seven_id_rollout_actor_and_snapshot_identity',
    'time_based_discount_and_busy_decision_accounting',
)


def close_at_ready(curve, ready, *, reference_speed=1., rest_s=.15):
    """Explicit C2 rest bridges, never a qpos reset or a modulo repair.

    Duplicate endpoint holds force zero boundary velocity, preserving every
    original internal polynomial. The bridge itself still needs native tests.
    """
    ready=np.asarray(ready, float)
    if ready.shape!=(7,) or not np.isfinite(ready).all():
        raise ValueError('Finite seven-joint ready node required')
    if (isinstance(reference_speed,bool) or isinstance(rest_s,bool)
            or not np.isfinite([reference_speed,rest_s]).all()
            or reference_speed<=0 or rest_s<=0):
        raise ValueError('Positive finite bridge speed and rest time required')
    if curve.knot_time_s[0]!=0:raise ValueError('Source starts at zero')
    prefix=not np.array_equal(ready,curve.knot_qpos[0])
    suffix=not np.array_equal(ready,curve.knot_qpos[-1])
    if not prefix and not suffix:
        return curve, {'added_prefix':False,'added_suffix':False,'source_interval_s':[0.,curve.duration_s]}
    times=[];poses=[];offset=0.
    if prefix:
        bridge=max(.35,1.875*float(np.max(np.abs(curve.knot_qpos[0]-ready)))/reference_speed)
        times=[0.,bridge];poses=[ready,curve.knot_qpos[0]];offset=bridge+rest_s
    times.extend((curve.knot_time_s+offset).tolist());poses.extend(curve.knot_qpos)
    if suffix:
        bridge=max(.35,1.875*float(np.max(np.abs(curve.knot_qpos[-1]-ready)))/reference_speed)
        end=times[-1];times.extend([end+rest_s,end+rest_s+bridge])
        poses.extend([curve.knot_qpos[-1],ready])
    out=C2JointSpline(times,poses)
    first=2 if prefix else 0
    if not np.allclose(out.coefficients[first:first+len(curve.coefficients)],curve.coefficients,rtol=0,atol=1e-10):
        raise ValueError('Bridge changed the interior source polynomial')
    return out, {'added_prefix':prefix,'added_suffix':suffix,
                 'source_interval_s':[offset,offset+curve.duration_s],
                 'new_native_validation_required':True}


def _source(spec):
    path=ROOT/spec['source'];provenance={}
    def bind(p):provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    if spec['source_type']=='screen':
        bind(path);row=json.loads(path.read_text())['actions'][spec['source_action']]
        if not row.get('sampled_reference_pass'):raise ValueError('Source reference did not pass its own screen')
        scale=spec['time_scale']
        if isinstance(scale,bool) or not np.isfinite(scale) or scale<=0:raise ValueError('Positive source time scale')
        curve=C2JointSpline(np.asarray(row['knot_time_s'])*scale,row['knot_qpos'])
        return curve,provenance,'reference_only_requires_native_test'
    summary=json.loads((path/'summary.json').read_text())
    if not summary.get('native_execution_pass') or not summary.get('source_unchanged'):
        raise ValueError('Source native run must be complete and unchanged')
    filename='references.npz' if spec['source_type']=='scheduled' else 'graph_references.npz'
    for name in ('summary.json',filename,'trace.npz','scene.xml'):bind(path/name)
    prefix=spec['slot']+'_' if spec['source_type']=='scheduled' else spec['slot']+'__'+spec['source_action']+'__'
    with np.load(path/filename,allow_pickle=False) as data:
        curve=C2JointSpline(data[prefix+'knot_time_s'],data[prefix+'knot_qpos'])
        if not np.array_equal(curve.coefficients,data[prefix+'coefficients']):
            raise ValueError('Source curve coefficients do not match knots')
    if spec['source_type']=='scheduled':
        # A plausible curve in references.npz alone is not evidence that this
        # was the curve executed. Reuse the all-samples native trace binding.
        from .a1_reactive_library import _references
        checked=_references(path,spec['slot'],summary)
        if (not np.array_equal(curve.knot_time_s,checked['time_s'])
                or not np.array_equal(curve.knot_qpos,checked['qpos'])):
            raise ValueError('Curve differs from its complete native command trace')
    if (path/'contacts.json').exists():bind(path/'contacts.json')
    return curve,provenance,'source_native_complete_not_assembled_bank_admission'


def build(recipe_path, output):
    recipe_path=Path(recipe_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior candidate builds')
    recipe=json.loads(recipe_path.read_text())
    expected=[(i,n,'attack' if i<4 else 'defence',('left','right','upper')[(i-1)%3])
              for i,n in enumerate(ACTION_NAMES[1:],1)]
    if any(type(s.get('id')) is not int for s in recipe['skills']) or [(s['id'],s['name'],s['kind'],s['region']) for s in recipe['skills']]!=expected:
        raise ValueError('Exact stable 1..6 action map required; zero is wait/continue')
    ready_path=ROOT/recipe['ready_source']
    ready_report=json.loads(ready_path.read_text())
    ready=np.asarray(ready_report[recipe['ready_key']] if 'ready_key' in recipe else
                     ready_report['guards'][recipe['ready_guard']]['qpos_rad'])
    nodes={'ready':ready.tolist()};actions={};analyses={}
    provenance={str(recipe_path):hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
                str(ready_path.relative_to(ROOT)):hashlib.sha256(ready_path.read_bytes()).hexdigest()}
    for relative in (recipe['scoring'],'config/joint_constraints_observed.json',
                     'config/a1_runtime_rk4_candidate.json','config/assembly.json',
                     'sim/limited_model.py','sim/limited_duel.py','sim/feedback_policy.py'):
        provenance[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    for spec in recipe['skills']:
        source,hashes,evidence=_source(spec);provenance.update(hashes)
        review={}
        if 'review_evidence' in spec:
            review_path=ROOT/spec['review_evidence']
            review=json.loads(review_path.read_text())
            provenance[str(review_path.relative_to(ROOT))]=hashlib.sha256(review_path.read_bytes()).hexdigest()
            # Keep negative evidence with the candidate. This is deliberately
            # not an admission flag and cannot override any runtime gate.
            review={k:review[k] for k in ('status','attack_probe_pass','successful_physical_defence') if k in review}
        curve,bridge=close_at_ready(source,ready,reference_speed=recipe['bridge_reference_speed_rad_s'],rest_s=recipe['bridge_rest_s'])
        name=spec['name'];route=[]
        for i,q in enumerate(curve.knot_qpos):
            key='ready' if np.array_equal(q,ready) else f'{name}_knot_{i}'
            nodes[key]=q.tolist();route.append(key)
        if route[0]!='ready' or route[-1]!='ready':raise ValueError('Every candidate must be a closed ready loop')
        actions[name]={'id':spec['id'],'kind':spec['kind'],'intended_region':spec['region'],
            'start_node':'ready','end_node':'ready','waypoint_nodes':route,
            'knot_time_s':curve.knot_time_s.tolist(),'source':spec,
            'source_evidence_scope':evidence,'bridge':bridge,
            'review_evidence_summary':review,
            'assembled_native_validated':False,'training_admitted':False}
        delta=np.abs(np.diff(curve.knot_qpos,axis=0))
        report=curve.report()
        analyses[name]={'duration_s':curve.duration_s,'explicit_hold_s':report['hold_duration_s'],
            'total_joint_travel_deg':np.rad2deg(delta.sum(axis=0)).tolist(),
            'j0_travel_deg':float(np.rad2deg(delta[:,0].sum())),
            'reference_peak_speed_rad_s':report['extrema']['peak_speed_rad_s'],
            'minimum_joint_margin_rad':float(np.min(np.minimum(
                curve.bezier_controls-model_joint_ranges_rad(load_joint_constraints())[:,0],
                model_joint_ranges_rad(load_joint_constraints())[:,1]-curve.bezier_controls))),
            'continuity':report['continuity'],'bridge':bridge}
    horizon=float(np.ceil((max(a['duration_s'] for a in analyses.values())+.75+2.)*1000)/1000)
    cfg={'schema_version':1,'bank_version':recipe.get('bank_version','compact_v1_candidate_01'),'policy_family':'A1',
        'status':'candidate_built_training_blocked_until_gates_pass','admitted_to_training':False,
        'legacy_actor_compatible':False,'joint_names':list(JOINT_NAMES),'angle_unit':'radian',
        'joint_constraints':'config/joint_constraints_observed.json',
        'reference_generator':'sim.motion_spline.C2JointSpline','initial_node':'ready',
        'action_ids':dict(enumerate(ACTION_NAMES)),'nodes':nodes,'actions':actions,
        'wait_semantics':'Keep current legal reference, or continue busy skill. Never reset qpos or interrupt a skill.',
        'scoring_source':recipe['scoring'],'source_sha256':provenance,
        'minimum_single_trial_horizon_s':horizon,
        'admission_gates':{g:'pending' for g in REQUIRED_GATES},
        'notes':recipe['scope_notes']}
    cfg['admission_gates']['same_ready_and_full_curve_constraints']='reference_checked_native_pending'
    output.mkdir(parents=True)
    path=output/'bank.json';path.write_text(json.dumps(_finite_json(cfg),ensure_ascii=False,indent=2)+'\n')
    load_candidate_graph(path)
    (output/'construction_report.json').write_text(json.dumps(_finite_json({'skills':analyses,
        'bank_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'admitted_to_training':False}),indent=2)+'\n')
    return path


class CompactBankController:
    """Seven stable IDs for diagnostic use, not a retrofitted five-ID actor."""
    def __init__(self,path):
        self.graph=load_candidate_graph(path)
        cfg=json.loads(self.graph.source_bytes)
        if cfg.get('action_ids')!={str(i):n for i,n in enumerate(ACTION_NAMES)}:
            raise ValueError('Unknown action map')
        if set(cfg['actions'])!=set(ACTION_NAMES[1:]):raise ValueError('Incomplete compact bank')
        if cfg.get('initial_node')!='ready':raise ValueError('Compact bank must initialize at shared ready')
        for i,name in enumerate(ACTION_NAMES[1:],1):
            spec=cfg['actions'][name]
            if type(spec.get('id')) is not int or spec['id']!=i:
                raise ValueError('Compact action IDs must match their stable names')
            if (spec.get('start_node')!='ready' or spec.get('end_node')!='ready'
                    or spec['waypoint_nodes'][0]!='ready' or spec['waypoint_nodes'][-1]!='ready'):
                raise ValueError('Every compact action must close at the common ready node')
        self.executor=self.graph.new_executor()
        self.source_identity=cfg.get('source_sha256',{})

    def assert_identity(self):
        if self.graph.source_path.read_bytes()!=self.graph.source_bytes:
            raise ValueError('Bank changed; discard this controller and revalidate')
        for name,digest in self.source_identity.items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Bank source identity changed: '+name)

    def mask(self):
        allowed=self.executor.available()
        return np.asarray([True]+[name in allowed for name in ACTION_NAMES[1:]],dtype=bool)

    @property
    def actor_decision_allowed(self):
        # Busy continue ticks are NOT additional PPO choices. The always-legal
        # zero ID is an execution fallback, not a license to sample stale masks.
        return bool(self.executor.ready and self.executor.active is None)

    def own_status(self,now):
        active=self.executor.active
        fraction=0.
        if active is not None:
            duration=self.executor.actions[active].curve.duration_s
            fraction=float(np.clip((now-self.executor.started_s)/duration,0.,1.))
        return {'actor_decision_allowed':self.actor_decision_allowed,
                'available_mask':self.mask().tolist(),
                'own_active_action_id':ACTION_NAMES.index(active) if active else 0,
                'own_reference_progress':fraction,
                'own_node':self.executor.node}

    def request_id(self,action_id,now):
        if type(action_id) is not int or not 0<=action_id<len(ACTION_NAMES):
            raise ValueError('Integer action ID 0..6 required')
        if not self.mask()[action_id]:raise ValueError('Skill unavailable in actual own state')
        if action_id:self.executor.request(ACTION_NAMES[action_id],now)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',default=str(ROOT/'config/a1_compact_v1_recipe.json'))
    p.add_argument('--output',required=True);a=p.parse_args()
    print(build(a.recipe,a.output))
