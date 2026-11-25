"""Replay an exact packaged A1 case as a fixed scheduled diagnostic.

This is not an actor, an observed-response controller, or a training entry point.
It preserves the packaged clock/curves, or replaces only the defender by a
same-initial-state hold. The native runner's old single-target result is not a
three-region verdict; a fresh whole-trace triangle audit is always attempted.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .a1_release import DEFAULT_PACKAGE, REFERENCE_SPEED_CEILING_RAD_S, load_release
from .a1_runtime import load_runtime_profile
from .a1_triangle_native import audit_recorded_triangle
from .assembly import ROOT, _finite_json
from .constrained_trial import ScheduledReference, run_constrained_trial, validate_references
from .joint_constraints import DEFAULT_PATH, load_joint_constraints
from .motion_spline import C2JointSpline
from .triangle_task import REGIONS, TriangleGeometry


SLOTS = ('a1', 'a2')
VARIANTS = ('baseline', 'defended')
# Schema-1 historical fixtures omitted this descriptive field. This mapping
# affects audit reporting only: EVERY region for BOTH robots is still audited.
NOMINAL_REGIONS = {
    'parry_high_center': 'upper', 'parry_right_thrust': 'right',
    'parry_right_descending': 'right', 'parry_right_rising': 'right',
    'parry_left_descending': 'left', 'parry_left_rising': 'left',
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _number(value, label):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(f'{label} must be a finite real number')
    return float(value)


def _numbers(value, shape, label):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(type(x) not in (int, float) for x in raw.flat):
        raise ValueError(f'{label} must contain real numbers with shape {shape}')
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f'{label} must be finite')
    return result


def _physical_binding(release, runtime):
    """Compare packaged physical sources without compiling or changing a model."""
    manifest, contents = release.manifest, dict(release.file_bytes)
    if 'physical_contract_file' not in manifest:
        return dict(physical_contract_bound=False,
                    scope='generic fixture without packaged model or torque identity')
    name = manifest['physical_contract_file']
    if not isinstance(name, str) or name not in contents:
        raise ValueError('physical_contract_file must be a hash-bound package file')
    contract = json.loads(contents[name])
    if not isinstance(contract, dict) or contract.get('runtime_config') != runtime:
        raise ValueError('Packaged physical contract runtime differs from the current runtime')
    sources = contract.get('model_source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Packaged physical contract must bind model source files')
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not relative or '\\' in relative:
            raise ValueError('Physical source paths must be repository-relative POSIX names')
        parts = PurePosixPath(relative)
        path = (ROOT / relative).resolve()
        if parts.is_absolute() or '..' in parts.parts or parts.as_posix() != relative \
                or not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Physical source path escapes the repository')
        if not isinstance(expected, str) or len(expected) != 64 \
                or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Physical source hash must be a SHA256 digest')
        if not path.is_file() or _sha(path.read_bytes()) != expected:
            raise ValueError(f'Current model source differs from the packaged physical contract: {relative}')
    caps = _numbers(contract.get('torque_caps_nm'), (7,), 'packaged torque caps')
    if np.any(caps <= 0):
        raise ValueError('Packaged torque caps must be positive')
    assembly_path = ROOT / 'config/assembly.json'
    assembly_raw = assembly_path.read_bytes()
    assembly = json.loads(assembly_raw)
    actual_caps = _numbers(assembly.get('runtime', {}).get('torque_caps_nm'),
                           (7,), 'current assembly torque caps')
    if not np.array_equal(actual_caps, caps):
        raise ValueError('Current assembly torque caps differ from the packaged physical contract')
    return dict(physical_contract_bound=True, physical_contract_file=name,
                physical_contract_sha256=_sha(contents[name]),
                model_source_sha256=dict(sources), torque_caps_nm=caps.tolist(),
                current_assembly_sha256=_sha(assembly_raw))


def prepare_references(case, variant):
    """Validate reference data only; no scene, IK, physics, or file writes.

    Reconstruct and compare all stored coefficients exactly, including the
    original defender in baseline mode. Do not refit, wrap, retime or trim.
    """
    if variant not in VARIANTS or not isinstance(variant, str):
        raise ValueError('variant must be baseline or defended')
    if not isinstance(case, dict) or case.get('admitted_to_training') is not False:
        raise ValueError('An explicitly unadmitted case object is required')
    duration = _number(case.get('duration_s'), 'duration_s')
    if not 0. < duration <= 60.:
        raise ValueError('Case duration must be positive and at most 60 seconds')
    attacker, defender = case.get('attacker_slot'), case.get('defender_slot')
    if attacker not in SLOTS or defender not in SLOTS or attacker == defender:
        raise ValueError('Distinct explicit attacker and defender slots required')
    specs, initial = case.get('programs'), case.get('initial_qpos')
    if not isinstance(specs, dict) or set(specs) != set(SLOTS) \
            or not isinstance(initial, dict) or set(initial) != set(SLOTS):
        raise ValueError('Both programs and both initial states must be explicit')
    programs = {}
    for slot in SLOTS:
        spec = specs[slot]
        if not isinstance(spec, dict) or set(spec) != {'start_s', 'curve'}:
            raise ValueError(f'{slot}: explicit start_s and curve required')
        row = spec['curve']
        if not isinstance(row, dict) or set(row) != {'knot_time_s', 'knot_qpos', 'coefficients'}:
            raise ValueError(f'{slot}: exact knots and coefficients required')
        times = row['knot_time_s']
        if not isinstance(times, list) or len(times) < 2:
            raise ValueError(f'{slot}: at least two knot times required')
        t = _numbers(times, (len(times),), slot + ' times')
        q = _numbers(row['knot_qpos'], (len(times), 7), slot + ' qpos')
        curve = C2JointSpline(t, q)
        coefficients = _numbers(row['coefficients'], curve.coefficients.shape, slot + ' coefficients')
        if not np.array_equal(coefficients, curve.coefficients):
            raise ValueError(f'{slot}: coefficients differ from the exact packaged C2 curve')
        q0 = _numbers(initial[slot], (7,), slot + ' initial_qpos')
        if not np.array_equal(q0, curve.knot_qpos[0]):
            raise ValueError(f'{slot}: initial state differs from the curve start')
        program = ScheduledReference(curve, _number(spec['start_s'], slot + ' start_s'))
        if duration < program.end_s + .1 - 1e-9:
            raise ValueError(f'{slot}: preserve the whole curve and at least .1 s endpoint dwell')
        programs[slot] = program
    if variant == 'baseline':
        source = programs[defender]
        q0 = source.curve.knot_qpos[0]
        programs[defender] = ScheduledReference(
            C2JointSpline([0., source.curve.duration_s], [q0, q0]), source.start_s)
    return programs, duration


@dataclass(frozen=True)
class PreparedCase:
    case_name: str
    variant: str
    programs: dict
    duration_s: float
    scoring: dict
    target_region: str
    attacker_slot: str
    defender_slot: str
    source_case: dict
    metadata: dict


def prepare_case(case_name, variant, package=DEFAULT_PACKAGE):
    """Read the hash-bound package and validate references, without integration."""
    release = load_release(package)
    manifest, contents = release.manifest, dict(release.file_bytes)
    cases = json.loads(contents[manifest['cases_file']])
    if cases.get('admitted_to_training') is not False \
            or type(cases.get('native_steps')) is not int or cases['native_steps'] != 0:
        raise ValueError('Packaged cases must be unadmitted source fixtures, not native results')
    table = cases.get('cases')
    if not isinstance(case_name, str) or not isinstance(table, dict) or case_name not in table:
        raise ValueError(f'Unknown packaged case: {case_name!r}')
    case = copy.deepcopy(table[case_name])
    # The existing native runner loads these unchanged shared files itself.
    # Refuse a stale package rather than silently replacing its physics.
    if _sha(DEFAULT_PATH.read_bytes()) != manifest['files_sha256'][manifest['joint_constraints_file']]:
        raise ValueError('Current native joint constraints differ from the packaged constraints')
    runtime = load_runtime_profile(ROOT / 'config/a1_runtime_rk4_candidate.json').runtime_config()
    if cases.get('runtime_config') != runtime:
        raise ValueError('Current native runtime differs from the packaged runtime')
    physical_binding = _physical_binding(release, runtime)
    scoring = copy.deepcopy(cases.get('scoring'))
    if not isinstance(scoring, dict):
        raise ValueError('Explicit packaged three-region scoring required')
    geometry = TriangleGeometry.from_mapping({k: scoring[k] for k in ('base_distance', 'regions')})
    if {region.name for region in geometry.regions} != set(REGIONS) \
            or geometry.base_distance != runtime['base_distance_m']:
        raise ValueError('All three regions and the exact native base spacing are required')
    region = case.get('target_region', NOMINAL_REGIONS.get(case_name))
    if not isinstance(region, str) or region not in REGIONS:
        raise ValueError('Unknown case needs an explicit nominal target_region for audit reporting')
    original, duration = prepare_references(case, 'defended')
    programs, _ = prepare_references(case, variant)
    constraints = load_joint_constraints(release.package_dir / manifest['joint_constraints_file'])
    for references in (original, programs):
        validate_references(references, constraints, duration)
        for program in references.values():
            if max(program.curve.analytic_extrema()['peak_speed_rad_s']) > REFERENCE_SPEED_CEILING_RAD_S + 1e-9:
                raise ValueError('Packaged reference exceeds the unchanged 2 rad/s ceiling')
    dt = runtime['physics_timestep_s']
    if not np.isclose(duration / dt, round(duration / dt), atol=1e-8, rtol=0.):
        raise ValueError('Duration must contain an integer number of native steps')
    metadata = dict(
        scope='fixed scheduled diagnostic; not observed reaction, learned policy or training',
        case_name=case_name, variant=variant, package_dir=str(release.package_dir),
        release_manifest_sha256=_sha(release.manifest_bytes),
        release_files_sha256=dict(manifest['files_sha256']),
        source_case_sha256=_sha(json.dumps(case, sort_keys=True, allow_nan=False).encode()),
        source_proof_sha256=case.get('proof_sha256'),
        source_native_references_sha256=case.get('native_references_sha256'),
        source_proof_is_historical_not_new_run_evidence=True,
        baseline_change='defender same-initial-q hold only' if variant == 'baseline' else 'none',
        attacker_slot=case['attacker_slot'], defender_slot=case['defender_slot'],
        nominal_target_region=region, requested_duration_s=duration,
        runtime_config=runtime, physical_binding=physical_binding,
        physical_binding_verified=physical_binding['physical_contract_bound'],
        autonomous_reaction=False, trained_policy=False,
        admitted_to_training=False, pair_comparison_performed=False,
        triangle_referee_is_not_the_existing_native_single_target_referee=True)
    return PreparedCase(case_name, variant, programs, duration, scoring, region,
                        case['attacker_slot'], case['defender_slot'], case, metadata)


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(_finite_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _reference_rows(programs):
    return {slot: dict(start_s=program.start_s, curve=dict(
        knot_time_s=program.curve.knot_time_s.tolist(),
        knot_qpos=program.curve.knot_qpos.tolist(),
        coefficients=program.curve.coefficients.tolist())) for slot, program in programs.items()}


def verify_native_references(native_dir, prepared):
    """Read-only exact input binding; not a physical-success judgment."""
    native_dir = Path(native_dir)
    summary = json.loads((native_dir / 'summary.json').read_text())
    if summary['runtime_config'] != prepared.metadata['runtime_config'] \
            or summary['program_start_s'] != {s: p.start_s for s, p in prepared.programs.items()} \
            or summary['requested_duration_s'] != prepared.duration_s:
        raise ValueError('Saved native timing or runtime differs from the prepared case')
    physical = prepared.metadata['physical_binding']
    if physical['physical_contract_bound'] and summary.get('model_source_sha256') != physical['model_source_sha256']:
        raise ValueError('Saved native model sources differ from the packaged physical contract')
    with np.load(native_dir / 'references.npz', allow_pickle=False) as refs:
        for slot, program in prepared.programs.items():
            for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                if not np.array_equal(refs[slot + '_' + key], getattr(program.curve, key)):
                    raise ValueError('Saved native reference differs from the prepared case')
    return True


def run_case(case_name, variant, output, package=DEFAULT_PACKAGE, *, progress=print):
    """Explicit CLI operation: one new native run, followed by a fresh audit."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; preserve old and failed diagnostics')
    prepared = prepare_case(case_name, variant, package)
    wrapper_paths = (Path(__file__), ROOT / 'sim/a1_release.py', ROOT / 'sim/a1_triangle_native.py')
    wrapper_hashes = {str(p.relative_to(ROOT)): _sha(p.read_bytes()) for p in wrapper_paths}
    metadata = dict(prepared.metadata, wrapper_source_sha256=wrapper_hashes)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'request.json', dict(metadata=metadata, source_case=prepared.source_case,
        programs=_reference_rows(prepared.programs), scoring=prepared.scoring))
    native = run_constrained_trial(prepared.programs, output / 'native',
        duration_s=prepared.duration_s, metadata=metadata, progress=progress)
    errors, audit, inputs_match = [], None, False
    try:
        inputs_match = verify_native_references(output / 'native', prepared)
        current = load_release(package)
        if _sha(current.manifest_bytes) != metadata['release_manifest_sha256']:
            raise ValueError('Package changed during the diagnostic')
        if _physical_binding(current, prepared.metadata['runtime_config']) != prepared.metadata['physical_binding']:
            raise ValueError('Physical source binding changed during the diagnostic')
        if any(_sha((ROOT / p).read_bytes()) != h for p, h in wrapper_hashes.items()):
            raise ValueError('Release runner or audit implementation changed during the diagnostic')
    except (OSError, ValueError, KeyError) as exc:
        errors.append(dict(stage='input_binding', error=str(exc)))
    try:
        audit = audit_recorded_triangle(output / 'native', prepared.scoring,
                                        prepared.target_region, attacker=prepared.attacker_slot)
        _write_json(output / 'triangle_audit.json', audit)
    except (OSError, ValueError, KeyError) as exc:
        errors.append(dict(stage='triangle_audit', error=str(exc)))
    full_window = bool(native.get('native_execution_pass') and audit is not None
        and audit['full_recorded_duration_s'] >= prepared.duration_s - 1e-9
        and not audit['initial_overlap'] and not audit['sweep_failures'] and not errors)
    entries = audit['all_entries'] if audit is not None else []
    result = dict(status='completed_scheduled_diagnostic' if full_window else 'incomplete_or_invalid_diagnostic',
        case_name=case_name, variant=variant, requested_duration_s=prepared.duration_s,
        native_execution_pass=bool(native.get('native_execution_pass')),
        native_failure=native.get('failure'), exact_native_references_match=inputs_match,
        packaged_physical_binding_verified=bool(prepared.metadata['physical_binding_verified']
            and not any(e['stage'] == 'input_binding' for e in errors)),
        whole_window_three_region_audit_valid=full_window, all_target_entry_count=len(entries),
        attacker_entry_count=sum(e['agent'] == prepared.attacker_slot for e in entries),
        defender_entry_count=sum(e['agent'] == prepared.defender_slot for e in entries),
        zero_entries_observed=bool(full_window and not entries), errors=errors,
        single_target_native_result_used_for_triangle_success=False,
        scope='One fixed scheduled diagnostic only; a matched pair and interaction review are still required.',
        autonomous_reaction=False, trained_policy=False, pair_comparison_performed=False,
        attack_or_defence_certified=False, admitted_to_training=False)
    if errors:
        _write_json(output / 'audit_errors.json', errors)
    _write_json(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args(argv)
    try:
        result = run_case(args.case, args.variant, args.output, args.package)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['whole_window_three_region_audit_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
