"""Explicit side-guard candidates and paired diagnostics, not training skills.

This independent experiment runner accepts a named unchanged attack from a
source bank and an explicit seven-joint defensive path. It uses the common
physics, actuator limits, scoring and full-window comparison without edits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .a1_dynamics_screen import screen_curve
from .a1_triangle_candidates import illegal_contacts, region_gaps
from .a1_triangle_parry import compare
from .assembly import ROOT, _finite_json
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial, validate_references
from .duel_scene import classify_contact
from .joint_constraints import DEFAULT_PATH, load_joint_constraints
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .manual_coordinated_trial import _real
from .manual_guard_reference import DEFAULT_REFERENCE
from .manual_j0_trial import SOURCE
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Keep earlier candidate evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite_json(value), indent=2)+'\n')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_programs(recipe, variant='defended', *, attack_name='cut_down_center',
                  source=SOURCE, attacker='a1'):
    if variant not in ('baseline', 'defended') or attacker not in ('a1', 'a2'):
        raise ValueError('Explicit baseline/defended variant and physical slot required')
    required = {'schema_version', 'candidate_id', 'status', 'ready_qpos_cad_rad',
                'start_s', 'time_s', 'qpos_cad_rad'}
    if (not isinstance(recipe, dict) or set(recipe) != required
            or type(recipe['schema_version']) is not int or recipe['schema_version'] != 1
            or recipe['status'] != 'offline_candidate_not_for_training'
            or not isinstance(recipe['candidate_id'], str) or not recipe['candidate_id'].strip()):
        raise ValueError('Explicit version-one offline defence recipe required')
    start = _real(recipe['start_s'], 'start_s')
    ready = _real(recipe['ready_qpos_cad_rad'], 'ready_qpos_cad_rad')
    t = _real(recipe['time_s'], 'time_s')
    q = _real(recipe['qpos_cad_rad'], 'qpos_cad_rad')
    if (start.shape != () or start < 0. or ready.shape != (7,) or t.ndim != 1
            or len(t) < 2 or t[0] != 0. or q.shape != (len(t), 7)
            or not np.array_equal(q[0], ready) or not np.array_equal(q[-1], ready)):
        raise ValueError('Finite seven-joint closed defence from explicit ready with zero-based clock required')
    source = Path(source)
    bank = json.loads(source.read_text())
    action = bank['actions'].get(attack_name)
    if (not isinstance(action, dict) or action.get('kind') != 'attack'
            or action.get('target_region') not in ('left', 'right', 'upper')):
        raise ValueError('Named source attack and declared target region required')
    attack_curve = C2JointSpline(action['knot_time_s'], action['knot_qpos'])
    defence_curve = C2JointSpline(t, q)
    for curve in (attack_curve, defence_curve):
        if max(curve.analytic_extrema()['peak_speed_rad_s']) > 2.+1e-9:
            raise ValueError('The unchanged 2 rad/s reference ceiling applies to both motions')
    defender = 'a2' if attacker == 'a1' else 'a1'
    attack = ScheduledReference(attack_curve, .75)
    defence = ScheduledReference(defence_curve, float(start))
    horizon = float(np.ceil(1000.*max(16., attack.end_s+2., defence.end_s+2.))/1000.)
    constraints = load_joint_constraints()
    validate_references({attacker: attack, defender: defence}, constraints, horizon)
    programs = {attacker: attack, defender: defence if variant == 'defended'
                else hold_reference(ready, float(start), defence_curve.duration_s)}
    validate_references(programs, constraints, horizon)
    meta = {'experiment': 'explicit_pose_set_side_defence', 'variant': variant,
        'candidate_id': recipe['candidate_id'], 'recipe': copy.deepcopy(recipe),
        'attacker_slot': attacker, 'defender_slot': defender, 'attacker_action': attack_name,
        'target_region': action['target_region'], 'initial_guard_cad_rad': ready.tolist(),
        'source_attack_sha256': _sha(source), 'source_bank': str(source.resolve()),
        'manual_reference_sha256': _sha(DEFAULT_REFERENCE),
        'joint_constraints_sha256': _sha(DEFAULT_PATH), 'implementation_sha256': _sha(__file__),
        'coordinate_system': 'original_CAD_radians_no_additional_zero_shift',
        'attack_is_unchanged_source_trajectory': True,
        'shared_controller_geometry_limits_and_scoring_changed': False,
        'uses_known_clock_for_offline_probe_only': True, 'policy_reaction_demonstrated': False,
        'defence_certified': False, 'admitted_to_training': False,
        'required_recovery_tail_s': 2., 'reference_speed_ceiling_rad_s': 2.}
    return programs, horizon, meta, bank['scoring']


def screen_pair(programs, horizon, metadata, scoring, period_s=.025, torque_period_s=.01):
    """Forward-kinematic full-path contact/target checks, not integration."""
    if isinstance(period_s, bool) or not np.isfinite(period_s) or period_s <= 0.:
        raise ValueError('Finite positive geometry sample period required')
    initial = {s: programs[s].target(0.) for s in ('a1', 'a2')}
    scene = build_limited_duel(scoring['base_distance'], initial_qpos=initial)
    geometry = TriangleGeometry.from_mapping({k: scoring[k] for k in ('base_distance', 'regions')})
    times = np.unique(np.r_[np.arange(0., horizon, period_s), horizon,
        *[p.start_s+p.curve.knot_time_s for p in programs.values()]])
    rows = {'first_illegal': None, 'first_legal_weapon_contact': None,
            'first_target_entry': {s: None for s in programs},
            'minimum_region_gap_m': {s: {r.name: float('inf') for r in geometry.regions}
                                     for s in programs},
            'minimum_defender_axis_z': 1., 'illegal_sample_count': 0}
    for t in times:
        for slot, program in programs.items():
            scene.data.qpos[scene.agents[slot]['qpos_ids']] = program.target(float(t))
        scene.data.qvel[:] = 0.
        mujoco.mj_forward(scene.model, scene.data)
        illegal = illegal_contacts(scene)
        if illegal:
            rows['illegal_sample_count'] += 1
            if rows['first_illegal'] is None:
                rows['first_illegal'] = {'global_s': float(t), 'contacts': illegal}
        if rows['first_legal_weapon_contact'] is None:
            for contact in scene.data.contact:
                if contact.dist < 0. and classify_contact(scene.model, contact,
                        rules_version='whole_blade_first_touch_v2') == 'cross_blade_blade':
                    rows['first_legal_weapon_contact'] = float(t)
                    break
        for slot in programs:
            gaps = region_gaps(scene, geometry, slot)
            for region, gap in gaps.items():
                rows['minimum_region_gap_m'][slot][region] = min(
                    rows['minimum_region_gap_m'][slot][region], gap)
                if gap < 0. and rows['first_target_entry'][slot] is None:
                    rows['first_target_entry'][slot] = {'global_s': float(t), 'region': region}
        sites = scene.agents[metadata['defender_slot']]['sites']
        axis = scene.data.site_xpos[sites['shinai_tip']]-scene.data.site_xpos[sites['shinai_grip']]
        rows['minimum_defender_axis_z'] = min(rows['minimum_defender_axis_z'], float(axis[2]/np.linalg.norm(axis)))
    torque = screen_curve(build_limited_floor().model,
                          programs[metadata['defender_slot']].curve, torque_period_s)
    return {'metadata': metadata, 'requested_duration_s': horizon, 'scoring': scoring,
        'geometry': rows, 'torque': torque,
        'sampled_geometry_without_illegal_contact': rows['first_illegal'] is None,
        'native_dynamics_tested': False, 'continuous_collision_certificate': False,
        'successful_physical_defence': False, 'admitted_to_training': False}


def run(output, recipe, variant, *, attack_name='cut_down_center', source=SOURCE, attacker='a1'):
    output = Path(output).resolve()
    path = output/('screen.json' if variant == 'screen' else variant+'/plan.json')
    if path.exists() or variant != 'screen' and path.parent.exists():
        raise FileExistsError('Keep all preceding candidate results')
    programs, horizon, metadata, scoring = make_programs(recipe,
        'defended' if variant == 'screen' else variant,
        attack_name=attack_name, source=source, attacker=attacker)
    if variant == 'screen':
        result = screen_pair(programs, horizon, metadata, scoring)
        write(path, result)
        return result
    write(path, dict(metadata, requested_duration_s=horizon, scoring=scoring))
    return run_constrained_trial(programs, path.parent/'native', duration_s=horizon,
                                 metadata=metadata, progress=print)


def review(output, baseline_root=None):
    """Compare to a local or independently recorded matching waiting trial."""
    output = Path(output).resolve()
    destination = output/'comparison.json'
    if destination.exists():
        raise FileExistsError('Keep earlier paired comparisons')
    root = output if baseline_root is None else Path(baseline_root).resolve()
    if (root/'baseline/native/summary.json').is_file():
        baseline_native, baseline_plan = root/'baseline/native', root/'baseline/plan.json'
    elif (root/'native/summary.json').is_file():
        baseline_native, baseline_plan = root/'native', root/'plan.json'
    elif (root/'summary.json').is_file():
        baseline_native, baseline_plan = root, root/'plan.json'
    else:
        raise ValueError('A completed explicit waiting baseline is required')
    native = output/'defended/native'
    base = json.loads(baseline_plan.read_text())
    defence = json.loads((output/'defended/plan.json').read_text())
    keys = ('source_attack_sha256', 'initial_guard_cad_rad', 'attacker_slot',
            'defender_slot', 'attacker_action', 'target_region', 'requested_duration_s', 'scoring')
    if any(base.get(k) != defence[k] for k in keys):
        raise ValueError('Pair must use exactly the same source attack, ready, roles, rules and full horizon')
    attacker, defender = defence['attacker_slot'], defence['defender_slot']
    with np.load(baseline_native/'references.npz', allow_pickle=False) as refs:
        q = refs[defender+'_knot_qpos']
        if not np.array_equal(q, np.tile(defence['initial_guard_cad_rad'], (len(q), 1))):
            raise ValueError('Baseline defender must hold exactly the same ready pose')
        attack_end = .75+float(refs[attacker+'_knot_time_s'][-1])
    summaries = [json.loads((folder/'summary.json').read_text()) for folder in (baseline_native, native)]
    extras = {name: all(all(s.get(name, {}).get(slot) is True for slot in ('a1', 'a2'))
                       for s in summaries) for name in ('entry_ready', 'final_settled')}
    extras['no_mid_trial_resets'] = all(s.get('qpos_resets_after_initialization') == 0 for s in summaries)
    extras['two_second_attack_tail'] = defence['requested_duration_s'] >= attack_end+2.-1e-9
    result = compare(baseline_native, native, defence['scoring'],
                     target=defence['target_region'], attacker=attacker)
    hit = result.get('baseline_first_triangle_hit_s')
    extras['baseline_hit_during_attack'] = hit is not None and .75 <= hit <= attack_end
    if not all(extras.values()):
        result.update(status='invalid', successful_physical_defence=False)
    result.update(explicit_plan_identity_checks={k: True for k in keys},
                  entry_and_exit_checks=extras, baseline_root=str(root),
                  source_recipe=defence['recipe'], admitted_to_training=False,
                  policy_reaction_demonstrated=False,
                  saved_native_sha256={str(folder): {name: _sha(folder/name) for name in
                      ('summary.json', 'trace.npz', 'scene.xml', 'references.npz', 'contacts.json')}
                      for folder in (baseline_native, native)})
    write(destination, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--variant', choices=('screen', 'baseline', 'defended', 'review'), required=True)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--attack', default='cut_down_center')
    p.add_argument('--attacker-slot', choices=('a1', 'a2'), default='a1')
    p.add_argument('--baseline-root', type=Path)
    a = p.parse_args()
    if a.variant == 'review':
        result = review(a.output, a.baseline_root)
    else:
        if a.recipe is None:
            p.error('--recipe required for a new explicit candidate')
        result = run(a.output, json.loads(a.recipe.read_text()), a.variant,
                     source=a.source, attack_name=a.attack, attacker=a.attacker_slot)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'native_execution_pass',
        'successful_physical_defence', 'sampled_geometry_without_illegal_contact')}, indent=2))


if __name__ == '__main__':
    main()
