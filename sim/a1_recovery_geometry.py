"""Isolated withdrawal-first repair of the positive-side recovery.

No dynamics are integrated.  Exact legacy endpoints are retained, while an
explicit retracted waypoint avoids the previously blocked opponent corridor.
Every attempted fit is saved; geometry and ideal torque are only candidates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from plan1_skills.matching import tool_fk
from .a1_dynamics_screen import screen_curve
from .a1_guard_defence_screen import contact_state
from .assembly import ROOT, _finite_json
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import GripAxisSolver, unit
from .triangle_task import TriangleGeometry


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), ensure_ascii=False, indent=2) + '\n')


def exact_scene(path):
    model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    agents = {slot: {'prefix': slot+'_', 'qpos_ids': np.array([
        model.jnt_qposadr[model.joint(slot+'_J'+str(i)).id] for i in range(7)])}
        for slot in ('a1', 'a2')}
    return SimpleNamespace(model=model, data=mujoco.MjData(model), agents=agents)


def timed_curve(nodes):
    nodes = np.asarray(nodes)
    delta = abs(np.diff(nodes, axis=0))
    durations = np.maximum(.9, np.maximum(1.9*delta[:, 0]/.45,
                                         1.9*np.max(delta[:, 1:], axis=1)/1.0))
    return C2JointSpline(np.r_[0., np.cumsum(durations)], nodes)


def screen_path(scene, geometry, curve, opponent, samples=151, both_slots=False):
    """All actual collision categories, not merely illegal contacts, are saved."""
    slots = {}
    for slot in (('a1', 'a2') if both_slots else ('a1',)):
        contacts, target_entry = {}, False
        minimum_gap = .2
        minimum_target_gap = float('inf')
        for t in np.linspace(0, curve.duration_s, samples):
            state = contact_state(scene, geometry, curve.sample(t), opponent, slot)
            minimum_gap = min(minimum_gap, state['minimum_blade_gap_m'])
            target_gap = min(x for v in state['target_gaps_m'].values() for x in v.values())
            target_entry |= target_gap < 0
            minimum_target_gap = min(minimum_target_gap, target_gap)
            for c in state['contacts']:
                key = '|'.join(c['geoms'])
                if key not in contacts:
                    contacts[key] = dict(c, first_time_s=float(t), last_time_s=float(t))
                row = contacts[key]
                row['last_time_s'] = float(t)
                row['depth_m'] = max(row['depth_m'], c['depth_m'])
        slots[slot] = {'all_contacts': list(contacts.values()),
                       'minimum_blade_gap_m': minimum_gap,
                       'minimum_target_gap_m': minimum_target_gap,
                       'any_target_entry': bool(target_entry),
                       'clear_pass': not contacts and not target_entry}
    return {'slots': slots, 'sample_count_per_slot': samples,
            'all_clear': all(row['clear_pass'] for row in slots.values()),
            'continuous_collision_certificate': False,
            'scope': 'Sampled exact frozen native16 collision geometry, no integration.'}


def static_screen(model, curve, samples=101):
    data = mujoco.MjData(model)
    caps = np.max(abs(model.actuator_forcerange), axis=1)
    forces = []
    for q in curve.sample(np.linspace(0, curve.duration_s, samples)):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        forces.append(data.qfrc_bias.copy())
    forces = np.asarray(forces)
    return {'peak_static_cap_ratio': float(np.max(abs(forces)/caps)),
            'peak_static_abs_torque_nm': np.max(abs(forces), axis=0).tolist(),
            'physical_caps_nm': caps.tolist(), 'sample_count': samples}


def build(output, candidate_index=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output; preserve failed candidates')
    output.mkdir(parents=True)
    source = ROOT/'outputs/a1_chain_geometry/chain12/report.json'
    pair_source = ROOT/'outputs/a1_short_defence/native16_switch_right_to_left/source_report_snapshot.json'
    scene_path = pair_source.parent/'baseline/scene.xml'
    source_bytes, pair_bytes = source.read_bytes(), pair_source.read_bytes()
    base, pair = json.loads(source_bytes), json.loads(pair_bytes)
    geometry = TriangleGeometry.from_mapping({k: base['scoring'][k] for k in ('base_distance', 'regions')})
    scene = exact_scene(scene_path)
    model = build_limited_floor().model
    bounds = model_joint_ranges_rad(load_joint_constraints())
    # Interior authoring bounds apply only to newly solved waypoints.  Legacy
    # endpoints are copied exactly and audited against the physical bounds.
    interior = bounds.copy()
    interior[:, 0] += np.deg2rad(3)
    interior[:, 1] -= np.deg2rad(3)
    interior[2, 1] = np.deg2rad(350)
    interior[3, 1] = np.deg2rad(115)
    interior[5, 0] = np.deg2rad(23)
    interior[6, 1] = np.deg2rad(82)
    solver = GripAxisSolver(model, interior)
    old = base['actions']['recover_positive_high_to_low']
    qstart, qend = np.asarray(old['knot_qpos'])[0], np.asarray(old['knot_qpos'])[-1]
    start_p, start_axis, start_tip = tool_fk(model, qpos=qstart)
    end_p, end_axis, end_tip = tool_fk(model, qpos=qend)
    held = np.asarray(pair['guards']['right_parry_hold']['qpos_rad'])
    attempts = []
    # Bounded, task-directed set.  First move mostly toward the actor's base;
    # only then follow the finite joint branch toward the lateral low guard.
    for dx in (.20, .26, .32):
        for dz in (0., -.06):
            for axis_blend in (0., .35):
                goal = start_p + [-dx, 0., dz]
                axis = unit((1-axis_blend)*start_axis + axis_blend*end_axis)
                qmid, fit = solver.solve(goal, axis, qstart, regularization=.01, iterations=180)
                curve = timed_curve([qstart, qmid, qend])
                physical_bound_pass = bool(np.all(curve.bezier_controls >= bounds[:, 0]) and
                                           np.all(curve.bezier_controls <= bounds[:, 1]))
                coarse = screen_path(scene, geometry, curve, held)
                stat = static_screen(model, curve, samples=61)
                row = {'index':len(attempts), 'requested_grip_m':goal.tolist(),
                    'requested_axis':axis.tolist(), 'fit':fit,
                    'knot_qpos':curve.knot_qpos.tolist(), 'knot_time_s':curve.knot_time_s.tolist(),
                    'physical_bounds_pass':physical_bound_pass,
                    'right_hold_coarse':coarse, 'static':stat,
                    'candidate_pass':bool(fit['fit_pass'] and physical_bound_pass and
                        coarse['all_clear'] and stat['peak_static_cap_ratio'] < .98)}
                if row['candidate_pass']:
                    dyn = screen_curve(model, curve, period_s=.025)
                    row['ideal_free_space_dynamics'] = dyn
                    row['candidate_pass'] &= dyn['ideal_demand_within_caps_at_samples']
                attempts.append(row)
                write(output/'attempts.json', attempts)
                print(json.dumps({'attempt':row['index'], 'fit':fit['fit_pass'],
                    'clear':coarse['all_clear'], 'static':stat['peak_static_cap_ratio'],
                    'duration':curve.duration_s, 'candidate':row['candidate_pass']}), flush=True)
    candidates = [r for r in attempts if r['candidate_pass']]
    candidates.sort(key=lambda r: (r['knot_time_s'][-1],
        -r['right_hold_coarse']['slots']['a1']['minimum_blade_gap_m']))
    report = copy.deepcopy(base)
    report.update(status='withdrawal_first_recovery_geometry_candidate',
                  admitted_to_training=False, native_dynamics_tested=False)
    report['recovery_repair'] = {
        'source_report':str(source), 'source_report_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'failed_pair_source':str(pair_source), 'failed_pair_source_sha256':hashlib.sha256(pair_bytes).hexdigest(),
        'collision_scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exact_start_guard':old['start_guard'], 'exact_end_guard':old['end_guard'],
        'purpose':'Replace recovery only; preserve attack paths, guards, limits, caps and scores.',
        'stage':'geometry_and_ideal_torque_only', 'attempt_count':len(attempts),
        'candidate_count':len(candidates), 'selected_index':None,
        'native_test_required':'First hold right_parry_hold baseline through recovery+switch+attack; then paired RTL bridge, both slots.',
        'scope_limit':'No evidence of moving-opponent defence, reactive decisions, or training admission.'}
    report['guards'].update({k:copy.deepcopy(v) for k,v in pair['guards'].items()
                             if k in ('left_parry_hold','right_parry_hold')})
    if candidates:
        chosen = (candidates[0] if candidate_index is None else
                  next(row for row in candidates if row['index']==candidate_index))
        curve = C2JointSpline(chosen['knot_time_s'], chosen['knot_qpos'])
        screens = {}
        for name in ('right_parry_hold','left_parry_hold','center',
                     'left_high_negative_robust','right_low_negative_robust','left_guard_positive'):
            screens[name] = screen_path(scene, geometry, curve,
                np.asarray(report['guards'][name]['qpos_rad']), samples=701, both_slots=True)
        broad_pass = all(s['all_clear'] for s in screens.values())
        final_static = static_screen(model, curve, samples=301)
        final_dyn = screen_curve(model, curve, period_s=.01)
        passed = bool(broad_pass and final_static['peak_static_cap_ratio']<.98
                      and final_dyn['ideal_demand_within_caps_at_samples'])
        replacement = dict(kind='transition', family='explicit_withdrawal_recovery',
            start_guard=old['start_guard'], end_guard=old['end_guard'],
            knot_time_s=curve.knot_time_s.tolist(), knot_qpos=curve.knot_qpos.tolist(),
            candidate_pass=passed, sampled_reference_pass=passed,
            native_dynamics_tested=False, admitted_to_training=False,
            screen_by_held_guard=screens, static=final_static,
            ideal_free_space_dynamics=final_dyn,
            authoring_intent='First retract grip toward own base with near-preserved shaft direction, then turn laterally into the exact old low guard.',
            waypoint_semantics=['legacy positive high','explicit withdrawal','legacy positive low'],
            scope='Candidate only. No existing recovery/native/chain/defence certificate transfers.')
        report['actions']['recover_positive_high_to_low'] = replacement
        report['recovery_repair'].update(selected_index=chosen['index'],
            sampled_all_baselines_pass=passed, duration_s=curve.duration_s,
            unchanged_other_actions=[n for n in base['actions'] if n!='recover_positive_high_to_low'])
        ts = np.linspace(0,curve.duration_s,701)
        np.savez_compressed(output/'recover_positive_high_to_low.npz',time_s=ts,
            qpos=curve.sample(ts),knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    else:
        # Invalidate the old failed route in this diagnostic artifact rather
        # than leaving an inherited true flag where a new candidate is absent.
        report['actions']['recover_positive_high_to_low']['candidate_pass'] = False
        report['actions']['recover_positive_high_to_low']['sampled_reference_pass'] = False
    write(output/'report.json', report)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    shutil.copyfile(scene_path, output/'scene.xml')
    print(json.dumps(report['recovery_repair'],indent=2),flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--candidate-index', type=int)
    args = p.parse_args()
    build(args.output, candidate_index=args.candidate_index)
