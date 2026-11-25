"""Discrete start-time probes of the unchanged late defensive motion.

Only the final constant hold is resized to retain the same global finish and
16-second observation window. Earlier motion polynomials, attack, physical
model and scoring remain unchanged. Offline schedules are not policy reaction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .manual_adaptive_trial import make_programs as adaptive_programs
from .manual_coordinated_trial import _real, review as coordinated_review
from .manual_guard_reference import DEFAULT_REFERENCE
from .manual_j0_trial import SOURCE
from .motion_spline import C2JointSpline
from .constrained_trial import run_constrained_trial


DEFAULT_RECIPE = ROOT/'config/manual_recovery_side_ready_late_v1.json'
BASELINES = {'a1': ROOT/'outputs/manual_recovery_v1/early01',
             'a2': ROOT/'outputs/manual_recovery_v1/late_swapped01'}
REQUESTED_DELTAS_S = (-.25, -.1, .1, .25)
BOUND_FILES = ('baseline/plan.json', 'baseline/native/summary.json',
               'baseline/native/scene.xml', 'baseline/native/references.npz',
               'baseline/native/trace.npz', 'baseline/native/contacts.json', 'comparison.json')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def shift_recipe(recipe, delta_s):
    """Shift the start; resize only the provably flat final segment."""
    delta = _real(delta_s, 'delta_s')
    if delta.shape != ():
        raise ValueError('Timing offset must be one finite scalar')
    if not isinstance(recipe, dict):
        raise ValueError('An explicit adaptive source recipe is required')
    shifted = copy.deepcopy(recipe)
    start = _real(recipe.get('start_s'), 'start_s')
    if start.shape != () or float(start) != .75 or float(start+delta) < 0.:
        raise ValueError('Source starts at .75 seconds and shifted start cannot be negative')
    times = _real(recipe.get('time_s'), 'time_s')
    qpos = _real(recipe.get('qpos_cad_rad'), 'qpos_cad_rad')
    original = C2JointSpline(times, qpos)
    if (len(times) < 3 or times[0] != 0.
            or not np.array_equal(qpos[-2], qpos[-1])
            or not np.array_equal(original.bezier_controls[-1], np.tile(qpos[-1], (6, 1)))):
        raise ValueError('Only a final exactly constant hold may be resized')
    if not np.isclose(float(start)+original.duration_s, 14., rtol=0., atol=1e-12):
        raise ValueError('The source must finish at global 14 seconds for its full 16-second window')
    changed_times = times.copy()
    changed_times[-1] -= float(delta)
    shifted['start_s'] = float(start+delta)
    shifted['time_s'] = changed_times.tolist()
    changed = C2JointSpline(changed_times, qpos)
    if (not np.array_equal(changed.coefficients[:-1], original.coefficients[:-1])
            or not np.array_equal(changed.bezier_controls[:-1], original.bezier_controls[:-1])
            or not np.array_equal(changed.knot_time_s[:-1], original.knot_time_s[:-1])):
        raise ValueError('Timing probe changed a nonfinal motion polynomial')
    if not np.isclose(shifted['start_s']+changed.duration_s, 14., rtol=0., atol=1e-12):
        raise ValueError('Complete recovery and the two-second final tail must be preserved')
    return shifted


def _dependencies_current(hashes):
    if not isinstance(hashes, dict) or not hashes:
        return False
    for relative, expected in hashes.items():
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or _sha(path) != expected:
            return False
    return True


def bind_baseline(root, programs, horizon, metadata, scoring):
    """Read and bind an immutable, matching native waiting baseline."""
    root = Path(root).resolve()
    hashes = {name: _sha(root/name) for name in BOUND_FILES}
    plan = json.loads((root/'baseline/plan.json').read_text())
    summary = json.loads((root/'baseline/native/summary.json').read_text())
    previous = json.loads((root/'comparison.json').read_text())
    attacker, defender = metadata['attacker_slot'], metadata['defender_slot']
    keys = ('source_attack_sha256', 'manual_reference_sha256', 'initial_guard_cad_rad',
            'attacker_slot', 'defender_slot', 'attacker_action', 'target_region')
    if (any(plan.get(k) != metadata[k] for k in keys) or plan.get('scoring') != scoring
            or plan.get('requested_duration_s') != horizon or horizon != 16.
            or summary.get('requested_duration_s') != horizon):
        raise ValueError('Baseline must preserve the exact attack, ready pose, roles, score and 16-second horizon')
    if (not summary.get('native_execution_pass') or not summary.get('source_unchanged')
            or summary.get('failure') or summary.get('qpos_resets_after_initialization') != 0
            or any(summary.get(k, {}).get(s) is not True
                   for k in ('entry_ready', 'final_settled') for s in ('a1', 'a2'))
            or not _dependencies_current(summary.get('source_sha256'))
            or not _dependencies_current(summary.get('model_source_sha256'))):
        raise ValueError('Waiting baseline is failed, incomplete, stale or not settled')
    attack = programs[attacker]
    if summary.get('program_start_s', {}).get(attacker) != .75 or attack.start_s != .75:
        raise ValueError('Baseline attack clock must remain .75 seconds')
    with np.load(root/'baseline/native/references.npz', allow_pickle=False) as refs:
        if any(not np.array_equal(refs[attacker+'_'+key], getattr(attack.curve, key))
               for key in ('knot_time_s', 'knot_qpos', 'coefficients')):
            raise ValueError('Baseline attack reference has changed')
        guard = np.asarray(metadata['initial_guard_cad_rad'])
        t, q = refs[defender+'_knot_time_s'], refs[defender+'_knot_qpos']
        if (not np.array_equal(q, np.tile(guard, (len(t), 1)))
                or not np.array_equal(refs[defender+'_coefficients'], C2JointSpline(t, q).coefficients)):
            raise ValueError('Baseline defender must hold the exact ready pose for the whole reference')
    with np.load(root/'baseline/native/trace.npz', allow_pickle=False) as trace:
        t = trace['time_s']
        expected = np.r_[programs['a1'].target(0.), programs['a2'].target(0.)]
        dt = summary['runtime_config']['physics_timestep_s']
        if (not np.array_equal(trace['qpos'][0], expected) or np.any(trace['qvel'][0])
                or len(t) < 2 or abs(t[0]) > 1e-9 or abs(t[-1]-horizon) > 1e-8
                or not np.allclose(np.diff(t), dt, rtol=0., atol=1e-9)):
            raise ValueError('Baseline physical initial state or full recorded window does not match')
    audit = previous.get('baseline_triangle_audit', {})
    if (not previous.get('baseline_valid_threat') or previous.get('attacker_slot') != attacker
            or any(audit.get('native_trace_source_sha256', {}).get(name) != hashes['baseline/native/'+name]
                   for name in ('summary.json', 'trace.npz', 'scene.xml'))):
        raise ValueError('Baseline threat evidence is missing or not bound to these native files')
    hit = previous.get('baseline_first_triangle_hit_s')
    if not isinstance(hit, (int, float)) or isinstance(hit, bool) or not .75 <= hit <= attack.end_s:
        raise ValueError('Baseline must have a valid hit within the unchanged attack clock')
    return {'root': str(root), 'sha256': hashes, 'qualified_waiting_threat_bound': True,
            'baseline_read_only': True}


def make_programs(delta_s, *, attacker='a1', recipe=None, source=SOURCE,
                  reference=DEFAULT_REFERENCE, baseline_root=None):
    if attacker not in BASELINES:
        raise ValueError('Choose physical attacker a1 or a2')
    original = json.loads(DEFAULT_RECIPE.read_text()) if recipe is None else copy.deepcopy(recipe)
    shifted = shift_recipe(original, delta_s)
    programs, horizon, metadata, scoring = adaptive_programs(
        shifted, 'defended', attacker=attacker, source=source, reference=reference)
    if horizon != 16.:
        raise ValueError('Timing variants must retain the original 16-second window')
    metadata.update(experiment='late_defence_discrete_timing_probe',
        timing_source_recipe=original, timing_source_recipe_sha256=_canonical_sha(original),
        delta_s=float(delta_s), original_defender_start_s=.75,
        shifted_defender_start_s=shifted['start_s'],
        unchanged_global_defender_end_s=14.,
        final_constant_hold_only_resized=True,
        earlier_local_motion_polynomials_unchanged=True,
        implementation_sha256=_sha(Path(__file__)),
        uses_known_clock_for_offline_probe_only=True,
        policy_reaction_demonstrated=False, admitted_to_training=False)
    metadata['baseline_binding'] = bind_baseline(
        BASELINES[attacker] if baseline_root is None else baseline_root,
        programs, horizon, metadata, scoring)
    return programs, horizon, metadata, scoring


def _check_binding(binding):
    if not isinstance(binding, dict) or set(binding.get('sha256', {})) != set(BOUND_FILES):
        raise ValueError('Complete frozen baseline binding required')
    root = Path(binding['root'])
    if any(_sha(root/name) != expected for name, expected in binding['sha256'].items()):
        raise ValueError('A frozen baseline artifact changed')
    return root


def run(output, delta_s, *, attacker='a1'):
    folder = Path(output).resolve()/'defended'
    if folder.exists():
        raise FileExistsError('Preserve all earlier timing experiments')
    programs, horizon, metadata, scoring = make_programs(delta_s, attacker=attacker)
    folder.mkdir(parents=True)
    (folder/'plan.json').write_text(json.dumps(_finite_json(dict(
        metadata, requested_duration_s=horizon, scoring=scoring)), indent=2)+'\n')
    return run_constrained_trial(programs, folder/'native', duration_s=horizon,
                                 metadata=metadata, progress=print)


def screen(output, delta_s, *, attacker='a1'):
    from .a1_dynamics_screen import screen_curve
    from .limited_model import build_limited_floor
    path = Path(output).resolve()/'screen.json'
    if path.exists():
        raise FileExistsError('Preserve prior timing screens')
    programs, _, metadata, _ = make_programs(delta_s, attacker=attacker)
    result = screen_curve(build_limited_floor().model, programs[metadata['defender_slot']].curve, .002)
    result.update(metadata=metadata, native_dynamics_tested=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite_json(result), indent=2)+'\n')
    return result


def review(output):
    root = Path(output).resolve()
    plan = json.loads((root/'defended/plan.json').read_text())
    baseline = _check_binding(plan['baseline_binding'])
    return coordinated_review(root, baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('screen', 'defended', 'review'), required=True)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--attacker-slot', choices=('a1', 'a2'), default='a1')
    args = parser.parse_args()
    if args.variant == 'review':
        result = review(args.output)
    else:
        if args.delta is None:
            parser.error('--delta is required for a new timing probe')
        result = (screen if args.variant == 'screen' else run)(
            args.output, args.delta, attacker=args.attacker_slot)
    print(json.dumps({k: result.get(k) for k in ('status', 'failure', 'successful_physical_defence',
        'native_execution_pass', 'completed_time_s', 'ideal_demand_within_caps_at_samples')}, indent=2))


if __name__ == '__main__':
    main()
