"""Whole-arm, re-timed candidates from a declared or original manual ready pose.

The seven-joint CAD-coordinate path is explicit, rather than a small overlay
on an old motion. Only the defender may be reauthored: the original centre-cut
attack, shared controller, physical model, targets and referee stay unchanged.
An optional explicit ready pose changes the matched reset, not a state midway
through a trial; any changed reset requires its own waiting baseline.
These are offline scheduled probes, not a policy or admitted training skills.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial, validate_references
from .joint_constraints import DEFAULT_PATH, JOINT_NAMES, enforce_joint_trajectory, load_joint_constraints
from .manual_coordinated_trial import _real, review
from .manual_guard_reference import DEFAULT_REFERENCE, choose_pose, load_reference
from .manual_j0_trial import SOURCE
from .motion_spline import C2JointSpline


ATTACK = 'cut_down_center'
MIN_DURATION_S = 16.
REFERENCE_SPEED_LIMIT_RAD_S = 2.


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def make_programs(recipe, variant, *, attacker='a1', source=SOURCE, reference=DEFAULT_REFERENCE):
    if variant not in ('baseline', 'defended') or attacker not in ('a1', 'a2'):
        raise ValueError('Explicit baseline/defended variant and physical attacker slot required')
    keys = {'schema_version', 'candidate_id', 'status', 'start_s', 'time_s', 'qpos_cad_rad'}
    if (not isinstance(recipe, dict) or not keys <= set(recipe)
            or set(recipe)-keys-{'ready_qpos_cad_rad'}
            or type(recipe['schema_version']) is not int or recipe['schema_version'] != 1
            or recipe['status'] != 'offline_candidate_not_for_training'
            or not isinstance(recipe['candidate_id'], str) or not recipe['candidate_id'].strip()):
        raise ValueError('Version-one explicit offline adaptive recipe required')
    start = _real(recipe['start_s'], 'start_s')
    if start.shape != () or float(start) < 0.:
        raise ValueError('Defender start_s must be a finite nonnegative scalar')
    times = _real(recipe['time_s'], 'time_s')
    qpos = _real(recipe['qpos_cad_rad'], 'qpos_cad_rad')
    if times.ndim != 1 or len(times) < 2 or times[0] != 0. or qpos.shape != (len(times), 7):
        raise ValueError('Increasing local times from zero and matching Nx7 CAD qpos required')
    constraints = load_joint_constraints()
    manual, display, _ = load_reference(reference, constraints)
    user_guard = choose_pose(display, constraints, manual['facing_variant']['base_yaw_delta_rad'])
    explicit_ready = 'ready_qpos_cad_rad' in recipe
    guard = (_real(recipe['ready_qpos_cad_rad'], 'ready_qpos_cad_rad').copy()
             if explicit_ready else user_guard.copy())
    if guard.shape != (7,):
        raise ValueError('Explicit ready pose must contain seven original-CAD joint coordinates')
    enforce_joint_trajectory(guard, constraints)
    if not np.array_equal(qpos[0], guard) or not np.array_equal(qpos[-1], guard):
        raise ValueError('Adaptive reference must begin and return to the exact selected ready pose')
    defense_curve = C2JointSpline(times, qpos)
    source, reference = Path(source), Path(reference)
    source_bytes = source.read_bytes()
    bank = json.loads(source_bytes)
    action = bank.get('actions', {}).get(ATTACK)
    if (not isinstance(action, dict) or action.get('kind') != 'attack'
            or action.get('target_region') not in ('left', 'right', 'upper')):
        raise ValueError('The unchanged cut_down_center attack with a declared region is required')
    attack_curve = C2JointSpline(action['knot_time_s'], action['knot_qpos'])
    attack_program = ScheduledReference(attack_curve, .75)
    defense_program = ScheduledReference(defense_curve, float(start))
    for name, curve in ((ATTACK, attack_curve), ('adaptive_defence', defense_curve)):
        if max(curve.analytic_extrema()['peak_speed_rad_s']) > REFERENCE_SPEED_LIMIT_RAD_S + 1e-9:
            raise ValueError(name + ' exceeds the unchanged 2 rad/s reference speed ceiling')
    horizon = float(np.ceil(max(MIN_DURATION_S,
        attack_program.end_s + 2., defense_program.end_s + 2.) * 1000.) / 1000.)
    defender = 'a2' if attacker == 'a1' else 'a1'
    # Validate the moving counterpart even while producing its waiting baseline.
    validate_references({attacker: attack_program, defender: defense_program}, constraints, horizon)
    programs = {attacker: attack_program, defender: defense_program if variant == 'defended'
                else hold_reference(guard, float(start), defense_curve.duration_s)}
    validate_references(programs, constraints, horizon)
    attack_identity = {'start_s': attack_program.start_s,
                       'time_s': attack_curve.knot_time_s.tolist(),
                       'qpos_cad_rad': attack_curve.knot_qpos.tolist(),
                       'coefficients': attack_curve.coefficients.tolist()}
    metadata = {
        'experiment': 'manual_guard_whole_arm_adaptive_defence_probe',
        'candidate_id': recipe['candidate_id'], 'variant': variant,
        'attacker_slot': attacker, 'defender_slot': defender,
        'attacker_action': ATTACK, 'target_region': action['target_region'],
        'recipe': copy.deepcopy(recipe), 'recipe_sha256': hashlib.sha256(_canonical(recipe)).hexdigest(),
        'source_attack_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'attack_reference_sha256': hashlib.sha256(_canonical(attack_identity)).hexdigest(),
        'manual_reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
        'joint_constraints_sha256': hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_guard_cad_rad': guard.tolist(),
        'user_reference_guard_cad_rad': user_guard.tolist(),
        'ready_pose_source': 'explicit' if explicit_ready else 'original',
        'requires_new_matched_baseline': not np.array_equal(guard, user_guard),
        'joint_names': list(JOINT_NAMES),
        'coordinate_system': 'original_CAD_radians_no_additional_zero_shift',
        'defender_start_s': float(start), 'attacker_start_s': .75,
        'minimum_full_duration_s': MIN_DURATION_S, 'required_recovery_tail_s': 2.,
        'reference_speed_ceiling_rad_s': REFERENCE_SPEED_LIMIT_RAD_S,
        'all_defender_joints_may_be_reauthored': True,
        'fixed_previous_time_grid_or_polynomials_required': False,
        'heuristic_small_offset_limits_applied': False,
        'planned_total_joint_travel_rad': np.abs(np.diff(qpos, axis=0)).sum(axis=0).tolist(),
        'manual_reference_is_rounded_screenshot': True,
        'manual_style_is_qualitative_reference_not_a_physical_certificate': True,
        'shared_controller_geometry_limits_and_scoring_changed': False,
        'uses_known_clock_for_offline_probe_only': True,
        'contact_relocation_or_downward_force_inferred_from_reference': False,
        'policy_reaction_demonstrated': False, 'defence_certified': False, 'admitted_to_training': False,
    }
    return programs, horizon, metadata, bank['scoring']


def run(output, recipe, variant, *, attacker='a1'):
    folder = Path(output).resolve() / variant
    if folder.exists():
        raise FileExistsError('Preserve all preceding adaptive experiments')
    programs, horizon, metadata, scoring = make_programs(recipe, variant, attacker=attacker)
    folder.mkdir(parents=True)
    (folder / 'plan.json').write_text(json.dumps(_finite_json(dict(
        metadata, requested_duration_s=horizon, scoring=scoring)), indent=2) + '\n')
    return run_constrained_trial(programs, folder / 'native', duration_s=horizon,
                                 metadata=metadata, progress=print)


def screen(output, recipe, *, attacker='a1'):
    from .a1_dynamics_screen import screen_curve
    from .limited_model import build_limited_floor
    path = Path(output).resolve() / 'screen.json'
    if path.exists():
        raise FileExistsError('Preserve previous adaptive screens')
    programs, _, metadata, _ = make_programs(recipe, 'defended', attacker=attacker)
    result = screen_curve(build_limited_floor().model, programs[metadata['defender_slot']].curve, .002)
    result.update(metadata=metadata, native_dynamics_tested=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite_json(result), indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('screen', 'baseline', 'defended', 'review'), required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--attacker-slot', choices=('a1', 'a2'), default='a1')
    args = parser.parse_args()
    if args.variant == 'review':
        result = review(args.output, args.baseline_root)
    else:
        if args.recipe is None:
            parser.error('Adaptive CAD-coordinate recipe required')
        recipe = json.loads(args.recipe.read_text())
        result = screen(args.output, recipe, attacker=args.attacker_slot) if args.variant == 'screen' else run(
            args.output, recipe, args.variant, attacker=args.attacker_slot)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'successful_physical_defence',
        'native_execution_pass', 'completed_time_s', 'ideal_demand_within_caps_at_samples',
        'worst_utilization')}, indent=2))


if __name__ == '__main__':
    main()
