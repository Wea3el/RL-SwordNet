"""Small reference overlays intended to move contact toward our own blade tip.

This preserves the base attack, reset, clock, and physical model. A joint-space
offset does not prove contact relocation, downward force, or a valid defence;
those require separate kinematic and full-window native evidence.
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
from .manual_wrist_trial import make_programs as wrist_programs
from .motion_spline import C2JointSpline


ALLOWED_COMPENSATION_JOINTS = (0, 1, 3, 5)
MAX_EXTRA_OFFSET_RAD = .25


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _repo_path(value):
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError('base_recipe must be a repository-relative path')
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('base_recipe must stay inside the repository')
    return result


def make_programs(recipe, variant, *, attacker='a1', base_recipe=None,
                  source=SOURCE, reference=DEFAULT_REFERENCE):
    if variant not in ('baseline', 'defended') or attacker not in ('a1', 'a2'):
        raise ValueError('Explicit baseline/defended variant and physical attacker slot required')
    keys = {'schema_version', 'candidate_id', 'status', 'base_kind', 'base_recipe', 'offsets_rad'}
    if (not isinstance(recipe, dict) or set(recipe) != keys
            or type(recipe['schema_version']) is not int or recipe['schema_version'] != 1
            or recipe['status'] != 'offline_candidate_not_for_training'
            or not isinstance(recipe['candidate_id'], str) or not recipe['candidate_id'].strip()
            or recipe['base_kind'] not in ('coordinated', 'wrist')):
        raise ValueError('Version-one explicit offline contact-shift recipe required')
    path = _repo_path(recipe['base_recipe'])
    base_recipe = json.loads(path.read_text()) if base_recipe is None else copy.deepcopy(base_recipe)
    builder = coordinated_programs if recipe['base_kind'] == 'coordinated' else wrist_programs
    programs, horizon, base_metadata, scoring = builder(
        base_recipe, 'defended', attacker=attacker, source=source, reference=reference)
    # Never mutate a caller's or the base builder's program/metadata objects.
    programs, metadata = dict(programs), copy.deepcopy(base_metadata)
    defender = metadata['defender_slot']
    original = programs[defender]
    curve = original.curve
    offsets = _real(recipe['offsets_rad'], 'offsets_rad')
    if (offsets.shape != curve.knot_qpos.shape or np.any(offsets[0])
            or np.any(offsets[-1]) or not np.any(offsets)
            or np.max(np.abs(offsets)) > MAX_EXTRA_OFFSET_RAD):
        raise ValueError('Nx7 extra offsets must begin/end at zero and stay within .25 rad')
    forbidden = [i for i in range(7) if i not in ALLOWED_COMPENSATION_JOINTS]
    if np.any(offsets[:, forbidden]):
        raise ValueError('Contact compensation may change only J0/J1/J3/J5')
    moved_curve = C2JointSpline(curve.knot_time_s, curve.knot_qpos + offsets)
    unchanged = np.flatnonzero(np.all(offsets == 0., axis=0))
    if not np.array_equal(moved_curve.coefficients[:, :, unchanged], curve.coefficients[:, :, unchanged]):
        raise ValueError('Every unchanged joint must preserve its exact base polynomial')
    delta = moved_curve.bezier_controls - curve.bezier_controls
    if np.max(np.abs(delta)) > MAX_EXTRA_OFFSET_RAD + 1e-12:
        raise ValueError('Whole-curve extra offset exceeds the .25 rad bound')
    if max(moved_curve.analytic_extrema()['peak_speed_rad_s']) > 2. + 1e-9:
        raise ValueError('Contact compensation exceeds the unchanged 2 rad/s reference speed ceiling')
    guard = curve.sample(0.)
    if (not np.array_equal(guard, np.asarray(metadata['initial_guard_cad_rad']))
            or not np.array_equal(curve.sample(curve.duration_s), guard)):
        raise ValueError('The base reference must start and return to the original guard')
    moving = ScheduledReference(moved_curve, original.start_s)
    if any(horizon < p.end_s + 2. - 1e-9 for p in (programs[attacker], moving)):
        raise ValueError('Preserve the complete base clocks and two-second recovery tail')
    constraints = load_joint_constraints()
    validate_references({attacker: programs[attacker], defender: moving}, constraints, horizon)
    programs[defender] = moving if variant == 'defended' else hold_reference(
        guard, original.start_s, curve.duration_s)
    validate_references(programs, constraints, horizon)
    expanded_base = copy.deepcopy(base_metadata['recipe'])
    full_recipe = {'contact_shift': copy.deepcopy(recipe), 'base_kind': recipe['base_kind'],
                   'base_expanded': expanded_base}
    base_reference = {'start_s': original.start_s, 'time_s': curve.knot_time_s.tolist(),
                      'qpos': curve.knot_qpos.tolist(), 'coefficients': curve.coefficients.tolist()}
    # Prior-layer claims (for example a wrist layer preserving J0-J3) describe
    # the base, not this new overlay. Retain them only inside its provenance.
    common_fields = ('attacker_slot', 'defender_slot', 'attacker_action', 'target_region',
                     'source_attack_sha256', 'manual_reference_sha256', 'initial_guard_cad_rad',
                     'joint_names', 'manual_reference_is_rounded_screenshot',
                     'uses_known_clock_for_offline_probe_only')
    metadata = {k: copy.deepcopy(base_metadata[k]) for k in common_fields if k in base_metadata}
    metadata.update(
        experiment='own_blade_distal_contact_shift_candidate', candidate_id=recipe['candidate_id'],
        variant=variant, base_recipe=copy.deepcopy(base_recipe), base_expanded_recipe=expanded_base,
        base_metadata=copy.deepcopy(base_metadata),
        recipe=full_recipe, recipe_sha256=_digest(full_recipe),
        base_recipe_sha256=_digest(base_recipe), expanded_base_recipe_sha256=_digest(expanded_base),
        base_reference_sha256=_digest(base_reference),
        base_builder_implementation_sha256=base_metadata.get('implementation_sha256'),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        allowed_compensation_joints=[f'J{i}' for i in ALLOWED_COMPENSATION_JOINTS],
        unchanged_joint_polynomials=[f'J{i}' for i in unchanged],
        extra_offset_coordinate_system='original_CAD_radians_relative_to_base_reference',
        maximum_extra_offset_rad=MAX_EXTRA_OFFSET_RAD,
        planned_peak_extra_offsets_rad=np.max(np.abs(offsets), axis=0).tolist(),
        whole_curve_extra_offset_envelope_rad=np.max(np.abs(delta), axis=(0, 1)).tolist(),
        planned_total_joint_travel_rad=np.abs(np.diff(moved_curve.knot_qpos, axis=0)).sum(axis=0).tolist(),
        base_attack_reset_start_and_horizon_unchanged=True,
        contact_point_target={'owner': defender, 'material_axis': 'own_shinai_grip_to_tip',
                              'direction': 'toward_own_blade_tip', 'achieved': False},
        own_blade_contact_shift_demonstrated=False,
        geometry_force_rules_or_targets_changed=False,
        downward_force_or_success_not_inferred_from_command=True,
        policy_reaction_demonstrated=False, defence_certified=False, admitted_to_training=False)
    return programs, horizon, metadata, scoring


def run(output, recipe, variant, *, attacker='a1'):
    programs, horizon, metadata, scoring = make_programs(recipe, variant, attacker=attacker)
    folder = Path(output).resolve() / variant
    if folder.exists():
        raise FileExistsError('Preserve all preceding contact-shift experiments')
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
        raise FileExistsError('Preserve previous contact-shift screens')
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
            parser.error('Contact-shift recipe required')
        recipe = json.loads(args.recipe.read_text())
        result = screen(args.output, recipe, attacker=args.attacker_slot) if args.variant == 'screen' else run(
            args.output, recipe, args.variant, attacker=args.attacker_slot)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'successful_physical_defence',
        'native_execution_pass', 'completed_time_s', 'ideal_demand_within_caps_at_samples',
        'worst_utilization')}, indent=2))


if __name__ == '__main__':
    main()
