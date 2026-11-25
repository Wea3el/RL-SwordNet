"""Bounded native reference diagnostics plus independent triangle entry audit.

The existing force-limited runner is reused unchanged. It still has its old
single-target diagnostic; that result is NEVER reused as triangle scoring.
Every saved physical state is independently checked against all three regions.
Neither this module nor a passing diagnostic admits a policy/skill for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .fencing_rules import FirstTouchScorer, blade_sections_from_scene
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


def checked_kinematic_updater(scene, qpos, checks=24):
    """Check the geometry-only path against full forward, on this saved model.

    Scoring reads cylinder endpoints, not accelerations or newly recomputed
    contact forces. Every recorded frame is still visited by the caller.
    Fall back to full forward if endpoint equivalence does not hold.
    """
    indices=np.unique(np.linspace(0,len(qpos)-1,min(checks,len(qpos)),dtype=int))
    difference=0.
    for i in indices:
        scene.data.qpos[:]=qpos[i]
        mujoco.mj_kinematics(scene.model,scene.data)
        kin=blade_sections_from_scene(scene)
        mujoco.mj_forward(scene.model,scene.data)
        full=blade_sections_from_scene(scene)
        for slot in ('a1','a2'):
            if len(kin[slot])!=len(full[slot]):
                raise ValueError('Blade topology changed during geometry check')
            for a,b in zip(kin[slot],full[slot]):
                for endpoint in ('start','end'):
                    difference=max(difference,float(np.max(np.abs(getattr(a,endpoint)-getattr(b,endpoint)))))
    use_kinematics=difference<=1e-12
    return (mujoco.mj_kinematics if use_kinematics else mujoco.mj_forward), {
        'geometry_update':'mj_kinematics' if use_kinematics else 'mj_forward_fallback',
        'kinematics_vs_full_forward_max_endpoint_difference_m':difference,
        'endpoint_equivalence_states_tested':len(indices),
        'all_native_frames_still_scored':True}


def scheduled_programs(action, ready, *, time_scale=1., settle_s=2., start_s=.75):
    """Preserve a full reference and add an explicit bounded settle tail."""
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in (time_scale, settle_s)):
        raise ValueError('Positive finite time scale and settle tail required')
    times = np.asarray(action['knot_time_s'], float)
    if times[0] != 0:
        raise ValueError('Reference must begin at zero')
    curve = C2JointSpline(times*time_scale, action['knot_qpos'])
    attacker = ScheduledReference(curve, start_s)
    duration = float(np.ceil((attacker.end_s+settle_s)*1000.)/1000.)
    if duration > 60:
        raise ValueError('Full reference and settle tail exceed the 60 second diagnostic bound')
    defender = hold_reference(np.asarray(ready, float), start_s, curve.duration_s)
    return {'a1': attacker, 'a2': defender}, duration


def event_velocity(native_dir, event, intended_direction):
    """Measured material-point velocity at the entry step's recorded endpoint.

    This is not the exact substep impact velocity. Position and velocity both
    come from the saved native state; no command derivative is substituted.
    """
    native_dir=Path(native_dir)
    direction=np.asarray(intended_direction,dtype=float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-9:
        raise ValueError('Finite nonzero direction required')
    model=mujoco.MjModel.from_xml_path(str(native_dir/'scene.xml'))
    data=mujoco.MjData(model)
    with np.load(native_dir/'trace.npz',allow_pickle=False) as trace:
        i=int(event['recorded_step'])
        data.qpos[:]=trace['qpos'][i]
        data.qvel[:]=trace['qvel'][i]
    mujoco.mj_forward(model,data)
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,event['blade_section'])
    if gid<0:
        raise ValueError('Missing recorded blade geometry')
    point=np.asarray(event['blade_closest_point_world_m'])
    jac=np.zeros((3,model.nv))
    mujoco.mj_jac(model,data,jac,None,point,int(model.geom_bodyid[gid]))
    velocity=jac@data.qvel
    if event['agent']=='a2':
        velocity*=np.array([-1.,-1.,1.])
    speed=float(np.linalg.norm(velocity))
    cosine=float(velocity@direction/(speed*np.linalg.norm(direction))) if speed>1e-9 else None
    return {'velocity_attacker_frame_m_s':velocity.tolist(), 'direction_cosine':cosine,
            'reference_derivative_used':False,
            'sampling':'actual recorded state at end of first-entry physics step, not exact impact time'}


def audit_recorded_triangle(native_dir, scoring, target_region, attacker='a1'):
    """Read physical trace; never integrate, command, or repair saved states."""
    native_dir = Path(native_dir)
    if attacker not in ('a1','a2'):
        raise ValueError('Attacker must be a physical slot')
    summary = json.loads((native_dir/'summary.json').read_text())
    geometry = TriangleGeometry.from_mapping({k: scoring[k] for k in ('base_distance','regions')})
    if not np.isclose(summary['runtime_config']['base_distance_m'], geometry.base_distance):
        raise ValueError('Native base spacing differs from target geometry')
    with np.load(native_dir/'trace.npz', allow_pickle=False) as trace:
        times, qpos = trace['time_s'].copy(), trace['qpos'].copy()
    if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(qpos).all():
        raise ValueError('Finite complete ordered physical trace required')
    # Re-evaluate the saved scene, not a later revision of the constructor.
    model=mujoco.MjModel.from_xml_path(str(native_dir/'scene.xml'))
    scene=SimpleNamespace(model=model,data=mujoco.MjData(model),
                          agents={s:{'prefix':s+'_'} for s in ('a1','a2')})
    if qpos.shape != (len(times), scene.model.nq):
        raise ValueError('Physical trace and model joint order disagree')
    updater,geometry_check=checked_kinematic_updater(scene,qpos)
    scorers = {r.name: FirstTouchScorer(
        {'a1': geometry.world_center(r.name,'a2'), 'a2': geometry.world_center(r.name,'a1')}, r.radius)
        for r in geometry.regions}
    entries = []
    previous = None
    for i, q in enumerate(qpos):
        scene.data.qpos[:] = q
        updater(scene.model, scene.data)
        current = blade_sections_from_scene(scene)
        if previous is None:
            previous = current
        for region, scorer in scorers.items():
            for hit in scorer.update(previous,current,float(times[max(0,i-1)]),float(times[i])):
                entries.append(dict(hit, region=region, recorded_step=i, recorded_step_time_s=float(times[i])))
        previous = current
    first_step = min((e['recorded_step'] for e in entries), default=None)
    earliest = [e for e in entries if e['recorded_step'] == first_step]
    first_attacker_regions = sorted({e['region'] for e in earliest if e['agent'] == attacker})
    initial_overlap = {r:s.initial_overlap_events for r,s in scorers.items() if s.initial_overlap_events}
    failures = {r:s.failure for r,s in scorers.items() if s.failure}
    intended_first = bool(earliest and first_attacker_regions == [target_region] and
                          all(e['agent']==attacker for e in earliest))
    return {'scope': 'independent swept region geometry from all recorded native states; not official settlement',
        'native_trace_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (native_dir/'summary.json',native_dir/'trace.npz',native_dir/'scene.xml')},
        'scoring_snapshot': scoring, 'target_region': target_region, 'attacker_slot':attacker,
        'geometry_check':geometry_check,
        'audit_implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'first_regions_at_recorded_native_step': first_attacker_regions,
        'intended_region_first': intended_first, 'initial_overlap': initial_overlap,
        'sweep_failures': failures, 'all_entries': entries,
        'minimum_clearance_m': {r:s.minimum_clearance_m for r,s in scorers.items()},
        'native_runner_full_window_pass': summary['native_execution_pass'],
        'native_runner_failure': summary['failure'],
        'full_recorded_duration_s': float(times[-1]),
        'reference_candidate_pass': bool(summary['native_execution_pass'] and intended_first
                                         and not initial_overlap and not failures),
        'foul_settlement_integrated_with_triangle': False,
        'attack_or_defence_certified': False, 'admitted_to_training': False}


def run(source, action_name, output, *, time_scale=1., settle_s=2.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve old diagnostics; choose a new output')
    report_path = source/'report.json'
    report = json.loads(report_path.read_text())
    action = report['actions'][action_name]
    if not action.get('sampled_reference_pass') or not action.get('target_region'):
        raise ValueError('A screened attack reference is required')
    programs, duration = scheduled_programs(action, report['guards']['center']['qpos_rad'],
                                          time_scale=time_scale, settle_s=settle_s)
    output.mkdir(parents=True)
    metadata = {'reference_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'action_name': action_name, 'time_scale': time_scale, 'settle_s': settle_s,
        'triangle_referee_is_not_the_existing_native_single_target_referee': True}
    result = run_constrained_trial(programs, output/'native', duration_s=duration,
                                   metadata=metadata, progress=print)
    audit = audit_recorded_triangle(output/'native',report['scoring'],action['target_region'])
    first=next((e for e in audit['all_entries'] if e['agent']=='a1' and e['region']==action['target_region']),None)
    if first is not None and action.get('direction_intent_attacker_frame') is not None:
        audit['first_entry_measured_direction']=event_velocity(output/'native',first,action['direction_intent_attacker_frame'])
        cosine=audit['first_entry_measured_direction']['direction_cosine']
        audit['native_direction_screen_pass']=cosine is not None and cosine >= .35
        audit['reference_candidate_pass'] &= audit['native_direction_screen_pass']
    audit['native_maximum_joint_reference_error_rad'] = result['maximum_joint_reference_error_rad']
    audit['time_scale'] = time_scale
    (output/'triangle_audit.json').write_text(json.dumps(_finite_json(audit),ensure_ascii=False,indent=2)+'\n')
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--action',required=True)
    p.add_argument('--output',required=True);p.add_argument('--time-scale',type=float,default=1.)
    p.add_argument('--settle-s',type=float,default=2.)
    a=p.parse_args()
    result=run(a.source,a.action,a.output,time_scale=a.time_scale,settle_s=a.settle_s)
    print(json.dumps({k:result[k] for k in ('reference_candidate_pass','native_runner_failure',
                                          'intended_region_first','full_recorded_duration_s')},indent=2))


if __name__ == '__main__':
    main()
