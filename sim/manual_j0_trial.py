"""One bounded, matched J0-only parry experiment, not a training skill.

Use the same complete attack and exact reset for the waiting baseline and the
moving defender. The script's known attack clock is an offline feasibility
probe, never an observation or policy-reaction claim. Preserve every failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_parry import compare, parry_program
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial, validate_references
from .joint_constraints import load_joint_constraints
from .manual_guard_reference import DEFAULT_REFERENCE, choose_pose, load_reference
from .motion_spline import C2JointSpline


SOURCE = ROOT / 'outputs/a1_profile_v2/bank07_current_candidates/report.json'
ATTACK = 'cut_down_right_to_left'


def make_programs(variant, *, delta_rad=-.125, approach_s=1., attacker='a1',
                  attack_name=ATTACK, route=None):
    if variant not in ('baseline', 'defended') or attacker not in ('a1', 'a2'):
        raise ValueError('Explicit baseline/defended variant and physical slot required')
    if (isinstance(delta_rad, bool) or not np.isfinite(delta_rad) or delta_rad == 0
            or isinstance(approach_s, bool) or not np.isfinite(approach_s) or approach_s <= 0):
        raise ValueError('A nonzero finite displacement and positive duration are required')
    constraints = load_joint_constraints()
    reference, display, _ = load_reference(DEFAULT_REFERENCE, constraints)
    guard = choose_pose(display, constraints, reference['facing_variant']['base_yaw_delta_rad'])
    moved = guard.copy()
    moved[0] += delta_rad
    source = json.loads(SOURCE.read_text())
    if attack_name not in source['actions'] or source['actions'][attack_name].get('kind') != 'attack':
        raise ValueError('Select an existing attack reference, not a renamed defence')
    action = source['actions'][attack_name]
    attack_curve = C2JointSpline(action['knot_time_s'], action['knot_qpos'])
    start = .75
    if route is None:
        spec = {'start_s': start, 'approach_s': float(approach_s),
                'release_s': start + attack_curve.duration_s + .2, 'return_s': float(approach_s)}
        horizon = float(np.ceil((spec['release_s'] + spec['return_s'] + 2.) * 1000.) / 1000.)
        defensive_motion = parry_program(guard, moved, spec, horizon)
    else:
        if not isinstance(route, dict) or set(route) != {'time_s', 'offset_rad'}:
            raise ValueError('Route requires exactly time_s and offset_rad arrays')
        values = []
        for key in ('time_s', 'offset_rad'):
            raw = np.asarray(route[key], dtype=object)
            if raw.ndim != 1 or len(raw) < 2 or any(isinstance(v, (bool, np.bool_)) or
                    not isinstance(v, (int, float, np.integer, np.floating)) for v in raw):
                raise ValueError('Route arrays must contain finite real numbers')
            value = raw.astype(float)
            if not np.isfinite(value).all():
                raise ValueError('Route arrays must contain finite real numbers')
            values.append(value)
        times, offsets = values
        if (times.shape != offsets.shape or times[0] != 0. or offsets[0] != 0.
                or offsets[-1] != 0. or not np.any(offsets)):
            raise ValueError('J0 route must begin at t=0 and return to the original guard')
        knots = np.tile(guard, (len(times), 1))
        knots[:, 0] += offsets
        curve = C2JointSpline(times, knots)
        if max(curve.analytic_extrema()['peak_speed_rad_s']) > 2. + 1e-9:
            raise ValueError('J0 route exceeds the existing reference speed ceiling')
        defensive_motion = ScheduledReference(curve, start)
        horizon = float(np.ceil((max(start + attack_curve.duration_s, defensive_motion.end_s) + 2.) * 1000.) / 1000.)
        spec = {'start_s': start, 'route': route, 'tail_s': 2.}
    defender = 'a2' if attacker == 'a1' else 'a1'
    programs = {attacker: ScheduledReference(attack_curve, start), defender:
                (defensive_motion if variant == 'defended' else
                 hold_reference(guard, start, defensive_motion.curve.duration_s))}
    # A baseline must not hide an invalid planned moving counterpart.
    validate_references({attacker: programs[attacker], defender: defensive_motion}, constraints, horizon)
    validate_references(programs, constraints, horizon)
    unchanged = np.tile(guard[1:], (len(defensive_motion.curve.knot_qpos), 1))
    if not np.array_equal(defensive_motion.curve.knot_qpos[:, 1:], unchanged):
        raise ValueError('The defender reference must change J0 only')
    metadata = {
        'experiment': ('rounded_user_guard_J0_only_downward_RL_probe' if attack_name == ATTACK
                       else f'rounded_user_guard_J0_only_{attack_name}_probe'),
        'attacker_slot': attacker, 'attacker_action': attack_name,
        'target_region': action['target_region'], 'defender_slot': defender,
        'variant': variant, 'source_attack_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'manual_reference_sha256': hashlib.sha256(DEFAULT_REFERENCE.read_bytes()).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'planned_defender_J0_delta_rad': float(delta_rad) if route is None else None,
        'planned_defender_mode': 'explicit_offset_route' if route is not None else 'step_hold_return',
        'planned_defender_timing': spec,
        'only_defender_command_J0_changes': True,
        'manual_reference_is_rounded_screenshot': True,
        'uses_known_clock_for_offline_probe_only': True, 'policy_reaction_demonstrated': False,
        'admitted_to_training': False,
    }
    return programs, horizon, metadata, source['scoring']


def run(root, variant, **kwargs):
    root = Path(root).resolve()
    folder = root / variant
    if folder.exists():
        raise FileExistsError('Keep all previous trials; choose a fresh root')
    programs, horizon, metadata, scoring = make_programs(variant, **kwargs)
    folder.mkdir(parents=True)
    plan = dict(metadata, requested_duration_s=horizon, scoring=scoring)
    (folder / 'plan.json').write_text(json.dumps(_finite_json(plan), indent=2) + '\n')
    return run_constrained_trial(programs, folder / 'native', duration_s=horizon,
                                 metadata=metadata, progress=print)


def review(root):
    root = Path(root).resolve()
    path = root / 'comparison.json'
    if path.exists():
        raise FileExistsError('Existing comparison is immutable')
    base = json.loads((root / 'baseline/plan.json').read_text())
    defended = json.loads((root / 'defended/plan.json').read_text())
    keys = set(base) - {'variant'}
    if keys != set(defended) - {'variant'} or any(base[k] != defended[k] for k in keys):
        raise ValueError('Baseline and defended plans do not match')
    native_paths = [root / name / 'native' for name in ('baseline', 'defended')]
    summaries = [json.loads((p / 'summary.json').read_text()) for p in native_paths]
    if any(s['physics_steps'] == 0 for s in summaries):
        result = {'status': 'invalid', 'successful_physical_defence': False,
                  'reason': 'At least one trial could not advance from its initial state',
                  'native_failures': [s['failure'] for s in summaries]}
    else:
        result = compare(*native_paths, base['scoring'], target=base['target_region'],
                         attacker=base['attacker_slot'])
        with np.load(native_paths[0] / 'references.npz', allow_pickle=False) as references:
            attacker = base['attacker_slot']
            start = summaries[0]['program_start_s'][attacker]
            end = start + float(references[attacker + '_knot_time_s'][-1])
        hit = result.get('baseline_first_triangle_hit_s')
        in_attack = bool(hit is not None and start <= hit <= end)
        result['baseline_first_hit_in_attack_window'] = in_attack
        result['baseline_attack_window_s'] = [start, end]
        if not in_attack and result['status'] not in ('invalid', 'unmatched', 'no_threat'):
            result.update(status='no_threat_in_attack_window', successful_physical_defence=False)
    result.update(experiment=base['experiment'], policy_reaction_demonstrated=False,
                  review_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='One scheduled J0-only probe; not multi-direction coverage or a trained skill')
    path.write_text(json.dumps(_finite_json(result), indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=('baseline', 'defended', 'review'), required=True)
    parser.add_argument('--delta', type=float, default=-.125)
    parser.add_argument('--approach', type=float, default=1.)
    parser.add_argument('--attacker-slot', choices=('a1', 'a2'), default='a1')
    parser.add_argument('--attack', default=ATTACK)
    parser.add_argument('--route-file', type=Path, help='Explicit local-time/offset candidate; no hidden bridge')
    args = parser.parse_args()
    if args.variant == 'review':
        result = review(args.output)
    else:
        result = run(args.output, args.variant, delta_rad=args.delta, approach_s=args.approach,
                     attacker=args.attacker_slot, attack_name=args.attack,
                     route=json.loads(args.route_file.read_text()) if args.route_file else None)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'successful_physical_defence',
        'native_execution_pass', 'completed_time_s')}, indent=2))


if __name__ == '__main__':
    main()
