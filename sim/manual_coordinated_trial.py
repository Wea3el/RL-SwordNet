"""Small upright-arm retraction probes, isolated from the historical J0 trials.

These explicit, scheduled references are offline feasibility experiments. They
do not add a training action, a future-script observation or a new game rule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_parry import compare
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial, validate_references
from .joint_constraints import JOINT_NAMES, load_joint_constraints
from .manual_guard_reference import DEFAULT_REFERENCE, choose_pose, load_reference
from .manual_j0_trial import SOURCE
from .motion_spline import C2JointSpline


ALLOWED_AUXILIARY_JOINTS = (1, 3, 5)
MAX_AUXILIARY_OFFSET_RAD = .25
MAX_J0_OFFSET_RAD = 1.


def _real(value, name):
    raw = np.asarray(value, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, float, np.integer, np.floating)) for v in raw.flat):
        raise ValueError(name + ' must contain real numbers, not booleans')
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def make_programs(recipe, variant, *, attacker='a1', source=SOURCE, reference=DEFAULT_REFERENCE):
    if variant not in ('baseline', 'defended') or attacker not in ('a1', 'a2'):
        raise ValueError('Explicit variant and physical attacker slot required')
    keys = {'schema_version', 'candidate_id', 'status', 'attack', 'start_s', 'time_s', 'offsets_rad'}
    if (not isinstance(recipe, dict) or set(recipe) != keys
            or type(recipe['schema_version']) is not int or recipe['schema_version'] != 1
            or recipe['status'] != 'offline_candidate_not_for_training'
            or not isinstance(recipe['candidate_id'], str) or not recipe['candidate_id']):
        raise ValueError('Explicit version-one offline candidate recipe required')
    start = _real(recipe['start_s'], 'start_s')
    if start.shape != () or float(start) < .75:
        raise ValueError('Retain at least the existing .75 second initial settle window')
    times = _real(recipe['time_s'], 'time_s')
    offsets = _real(recipe['offsets_rad'], 'offsets_rad')
    if (times.ndim != 1 or len(times) < 3 or offsets.shape != (len(times), 7)
            or times[0] != 0. or np.any(offsets[0]) or np.any(offsets[-1])):
        raise ValueError('Matching Nx7 offsets must begin and finish at the original guard')
    allowed = (0,) + ALLOWED_AUXILIARY_JOINTS
    if np.any(offsets[:, [i for i in range(7) if i not in allowed]]):
        raise ValueError('Only J0 and the explicitly bounded J1/J3/J5 auxiliary joints may change')
    if (np.max(np.abs(offsets[:, 0])) > MAX_J0_OFFSET_RAD
            or np.max(np.abs(offsets[:, 1:])) > MAX_AUXILIARY_OFFSET_RAD
            or not np.any(offsets[:, 0]) or not np.any(offsets[:, 1:])):
        raise ValueError('Require J0 plus small auxiliary motion within declared bounds')
    constraints = load_joint_constraints()
    manual, display, _ = load_reference(reference, constraints)
    guard = choose_pose(display, constraints, manual['facing_variant']['base_yaw_delta_rad'])
    defense_curve = C2JointSpline(times, guard + offsets)
    if max(defense_curve.analytic_extrema()['peak_speed_rad_s']) > 2. + 1e-9:
        raise ValueError('Candidate exceeds the unchanged 2 rad/s reference ceiling')
    source, reference = Path(source), Path(reference)
    bank = json.loads(source.read_text())
    name = recipe['attack']
    if not isinstance(name, str) or name not in bank['actions'] or bank['actions'][name].get('kind') != 'attack':
        raise ValueError('An existing unchanged attack reference is required')
    attack = bank['actions'][name]
    attack_curve = C2JointSpline(attack['knot_time_s'], attack['knot_qpos'])
    # Defence timing is variable, but the attack clock remains the historical
    # .75 seconds; shifting a candidate must not move the threat with it.
    ap = ScheduledReference(attack_curve, .75)
    dp = ScheduledReference(defense_curve, float(start))
    horizon = float(np.ceil((max(ap.end_s, dp.end_s) + 2.) * 1000.) / 1000.)
    defender = 'a2' if attacker == 'a1' else 'a1'
    validate_references({attacker: ap, defender: dp}, constraints, horizon)
    programs = {attacker: ap, defender: dp if variant == 'defended' else
                hold_reference(guard, float(start), defense_curve.duration_s)}
    validate_references(programs, constraints, horizon)
    canonical = json.dumps(recipe, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    metadata = {
        'experiment': 'rounded_upright_guard_small_coordinated_retraction',
        'candidate_id': recipe['candidate_id'], 'variant': variant,
        'attacker_slot': attacker, 'defender_slot': defender, 'attacker_action': name,
        'target_region': attack['target_region'], 'recipe': copy.deepcopy(recipe),
        'recipe_sha256': hashlib.sha256(canonical).hexdigest(),
        'source_attack_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'manual_reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_guard_cad_rad': guard.tolist(), 'offset_coordinate_system': 'original_CAD_radians',
        'joint_names': list(JOINT_NAMES), 'allowed_auxiliary_joints': ['J1', 'J3', 'J5'],
        'max_auxiliary_offset_rad': MAX_AUXILIARY_OFFSET_RAD,
        'planned_peak_absolute_offsets_rad': np.max(np.abs(offsets), axis=0).tolist(),
        'planned_total_joint_travel_rad': np.abs(np.diff(defense_curve.knot_qpos, axis=0)).sum(axis=0).tolist(),
        'manual_reference_is_rounded_screenshot': True,
        'uses_known_clock_for_offline_probe_only': True,
        'policy_reaction_demonstrated': False, 'admitted_to_training': False,
    }
    return programs, horizon, metadata, bank['scoring']


def run(root, variant, recipe, *, attacker='a1'):
    programs, horizon, metadata, scoring = make_programs(recipe, variant, attacker=attacker)
    folder = Path(root).resolve() / variant
    if folder.exists():
        raise FileExistsError('Keep previous experiments; choose a fresh output root')
    folder.mkdir(parents=True)
    plan = dict(metadata, requested_duration_s=horizon, scoring=scoring)
    (folder / 'plan.json').write_text(json.dumps(_finite_json(plan), indent=2) + '\n')
    return run_constrained_trial(programs, folder / 'native', duration_s=horizon,
                                 metadata=metadata, progress=print)


def screen(root, recipe, *, attacker='a1'):
    """Save a sampled combined-motion torque diagnostic, never admission."""
    from .a1_dynamics_screen import screen_curve
    from .limited_model import build_limited_floor
    programs, _, metadata, _ = make_programs(recipe, 'defended', attacker=attacker)
    output = Path(root).resolve() / 'screen.json'
    if output.exists():
        raise FileExistsError('Preserve earlier dynamics screens')
    result = screen_curve(build_limited_floor().model, programs[metadata['defender_slot']].curve, .002)
    result['metadata'] = metadata
    result['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['native_dynamics_tested'] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(result), indent=2) + '\n')
    return result


def review(root, baseline_root=None):
    root = Path(root).resolve()
    baseline_root = root if baseline_root is None else Path(baseline_root).resolve()
    output = root / 'comparison.json'
    if output.exists():
        raise FileExistsError('Keep earlier comparisons')
    plans = [json.loads((p / kind / 'plan.json').read_text()) for p, kind in
             ((baseline_root, 'baseline'), (root, 'defended'))]
    base, defended = plans
    paths = [baseline_root / 'baseline/native', root / 'defended/native']
    attacker, defender = defended['attacker_slot'], defended['defender_slot']
    identity_keys = ('source_attack_sha256', 'manual_reference_sha256', 'initial_guard_cad_rad',
                     'attacker_slot', 'defender_slot', 'attacker_action', 'target_region',
                     'requested_duration_s', 'scoring')
    if any(base[k] != defended[k] for k in identity_keys):
        raise ValueError('The baseline threat, reset, physical roles, score and full duration must match')
    result = compare(*paths, defended['scoring'], target=defended['target_region'], attacker=attacker)
    with np.load(paths[0] / 'references.npz', allow_pickle=False) as refs:
        guard = np.asarray(defended['initial_guard_cad_rad'])
        constant_baseline = bool(np.array_equal(refs[defender + '_knot_qpos'],
                                    np.tile(guard, (len(refs[defender + '_knot_time_s']), 1))))
        attack_end = .75 + float(refs[attacker + '_knot_time_s'][-1])
    summaries = [json.loads((p / 'summary.json').read_text()) for p in paths]
    hit = result.get('baseline_first_triangle_hit_s')
    extra_checks = {
        'baseline_defender_entire_reference_is_original_guard': constant_baseline,
        'baseline_hit_within_attack_window': bool(hit is not None and .75 <= hit <= attack_end),
        'both_initial_entries_settled': all(all(s.get('entry_ready', {}).get(a) is True
                                               for a in ('a1', 'a2')) for s in summaries),
        'both_final_states_settled': all(all(s.get('final_settled', {}).get(a) is True
                                           for a in ('a1', 'a2')) for s in summaries),
        'no_post_initialization_state_resets': all(s.get('qpos_resets_after_initialization') == 0 for s in summaries),
    }
    result['matching_checks'].update(extra_checks)
    if not all(extra_checks.values()):
        result.update(status='invalid', successful_physical_defence=False)
    result.update(baseline_root=str(baseline_root), recipe=defended['recipe'],
                  policy_reaction_demonstrated=False, admitted_to_training=False,
                  scope='One scheduled small-coordination probe; no general defence or training admission')
    output.write_text(json.dumps(_finite_json(result), indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=('baseline', 'defended', 'review', 'screen'))
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--attacker-slot', choices=('a1', 'a2'), default='a1')
    args = parser.parse_args()
    if args.variant == 'review':
        result = review(args.output, args.baseline_root)
    else:
        if args.recipe is None:
            parser.error('A recipe is required for a new native trial')
        recipe = json.loads(args.recipe.read_text())
        result = screen(args.output, recipe, attacker=args.attacker_slot) if args.variant == 'screen' else run(
            args.output, args.variant, recipe, attacker=args.attacker_slot)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'successful_physical_defence',
                                              'native_execution_pass', 'completed_time_s',
                                              'ideal_demand_within_caps_at_samples', 'worst_utilization')}, indent=2))


if __name__ == '__main__':
    main()
