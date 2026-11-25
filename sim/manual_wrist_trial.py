"""Bounded wrist-rotation experiments on the unchanged coordinated reference.

The wrist command is not evidence of downward force: recorded contact forces,
all target entries and complete recovery must be checked independently.
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
from .joint_constraints import load_joint_constraints
from .manual_coordinated_trial import _real, make_programs as coordinated_programs, review
from .manual_guard_reference import DEFAULT_REFERENCE
from .manual_j0_trial import SOURCE
from .motion_spline import C2JointSpline


MAX_WRIST_INCREMENT_RAD = .35


def make_programs(recipe, variant, *, attacker='a1', base_recipe=None,
                  source=SOURCE, reference=DEFAULT_REFERENCE):
    keys = {'schema_version', 'candidate_id', 'status', 'base_recipe', 'wrist_offsets_rad'}
    if (not isinstance(recipe, dict) or set(recipe) != keys
            or type(recipe['schema_version']) is not int or recipe['schema_version'] != 1
            or recipe['status'] != 'offline_candidate_not_for_training'
            or not isinstance(recipe['candidate_id'], str) or not recipe['candidate_id']
            or not isinstance(recipe['base_recipe'], str) or not recipe['base_recipe']):
        raise ValueError('Version-one explicit offline wrist recipe required')
    if base_recipe is None:
        base_recipe = json.loads((ROOT / recipe['base_recipe']).read_text())
    base, horizon, metadata, scoring = coordinated_programs(base_recipe, 'defended',
        attacker=attacker, source=source, reference=reference)
    if variant not in ('baseline', 'defended'):
        raise ValueError('Explicit baseline or defended variant required')
    defender = metadata['defender_slot']
    base_program = base[defender]
    pulse = _real(recipe['wrist_offsets_rad'], 'wrist_offsets_rad')
    if (pulse.shape != (len(base_program.curve.knot_time_s), 3)
            or np.any(pulse[0]) or np.any(pulse[-1]) or not np.any(pulse)
            or np.max(np.abs(pulse)) > MAX_WRIST_INCREMENT_RAD):
        raise ValueError('Nx3 wrist increments must start/end at zero and remain within .35 rad')
    knots = base_program.curve.knot_qpos.copy()
    knots[:, 4:7] += pulse
    curve = C2JointSpline(base_program.curve.knot_time_s, knots)
    if not np.array_equal(curve.coefficients[:, :, :4], base_program.curve.coefficients[:, :, :4]):
        raise ValueError('The base J0-J3 polynomial must remain exactly unchanged')
    delta_controls = curve.bezier_controls[:, :, 4:7] - base_program.curve.bezier_controls[:, :, 4:7]
    if np.max(np.abs(delta_controls)) > MAX_WRIST_INCREMENT_RAD + 1e-12:
        raise ValueError('Whole-curve wrist increment exceeds its declared bound')
    if max(curve.analytic_extrema()['peak_speed_rad_s']) > 2. + 1e-9:
        raise ValueError('Wrist candidate exceeds the unchanged reference speed ceiling')
    moving = ScheduledReference(curve, base_program.start_s)
    constraints = load_joint_constraints()
    validate_references({attacker: base[attacker], defender: moving}, constraints, horizon)
    base[defender] = moving if variant == 'defended' else hold_reference(
        curve.sample(0.), base_program.start_s, curve.duration_s)
    validate_references(base, constraints, horizon)
    canonical = json.dumps(recipe, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    metadata.update(experiment='upright_guard_small_wrist_press_probe',
        variant=variant, candidate_id=recipe['candidate_id'],
        base_recipe=copy.deepcopy(base_recipe),
        recipe={'wrist': copy.deepcopy(recipe), 'base': copy.deepcopy(base_recipe)},
        recipe_sha256=hashlib.sha256(canonical).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_J0_J3_reference_exactly_unchanged=True,
        allowed_wrist_increment_joints=['J4', 'J5', 'J6'],
        max_wrist_increment_rad=MAX_WRIST_INCREMENT_RAD,
        planned_peak_wrist_increment_rad=np.max(np.abs(pulse), axis=0).tolist(),
        planned_total_joint_travel_rad=np.abs(np.diff(knots, axis=0)).sum(axis=0).tolist(),
        downward_force_or_success_not_inferred_from_command=True)
    return base, horizon, metadata, scoring


def run(output, recipe, variant, *, attacker='a1'):
    programs, horizon, metadata, scoring = make_programs(recipe, variant, attacker=attacker)
    folder = Path(output).resolve() / variant
    if folder.exists():
        raise FileExistsError('Preserve all preceding wrist trials')
    folder.mkdir(parents=True)
    (folder / 'plan.json').write_text(json.dumps(_finite_json(dict(
        metadata, requested_duration_s=horizon, scoring=scoring)), indent=2) + '\n')
    return run_constrained_trial(programs, folder / 'native', duration_s=horizon,
                                 metadata=metadata, progress=print)


def screen(output, recipe, *, attacker='a1'):
    from .a1_dynamics_screen import screen_curve
    from .limited_model import build_limited_floor
    programs, _, metadata, _ = make_programs(recipe, 'defended', attacker=attacker)
    path = Path(output).resolve() / 'screen.json'
    if path.exists():
        raise FileExistsError('Preserve previous wrist screens')
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
            parser.error('Wrist recipe required')
        recipe = json.loads(args.recipe.read_text())
        result = screen(args.output, recipe, attacker=args.attacker_slot) if args.variant == 'screen' else run(
            args.output, recipe, args.variant, attacker=args.attacker_slot)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'successful_physical_defence',
        'native_execution_pass', 'completed_time_s', 'ideal_demand_within_caps_at_samples',
        'worst_utilization')}, indent=2))


if __name__ == '__main__':
    main()
