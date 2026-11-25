"""Uniformly retime four legacy references without changing their geometry.

This is an ideal free-space torque *screen*, not a native or reaction test.
Every requested scale is retained, including failures.  The old source and all
other motion records remain untouched; a new candidate bank requires new
full-window collision, tracking and paired-effectiveness tests.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .a1_dynamics_screen import screen_curve
from .a1_profile_bank import blade_identity
from .assembly import _finite_json
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


ACTIONS = ('cut_down_right_to_left', 'cut_down_center', 'thrust_center', 'parry_high')
SCALES = (1., 1.25, 1.6, 2., 2.5, 3., 4.)


def scaled_curve(action, scale):
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('A finite positive uniform time scale is required')
    times = np.asarray(action['knot_time_s'], dtype=float)
    if times[0] != 0.:
        raise ValueError('The source clock must explicitly start at zero')
    return C2JointSpline(times * scale, action['knot_qpos'])


def full_window_s(duration_s, start_s=.75, tail_s=2., minimum_s=6.):
    """Same millisecond ceiling as the single-reference native helper.

    Pair tests must additionally cover the opposing reference's complete end.
    This is not a time budget to truncate either action.
    """
    if not np.isfinite([duration_s, start_s, tail_s, minimum_s]).all():
        raise ValueError('Finite timing is required')
    if duration_s <= 0 or start_s < .1 or tail_s < 2. or minimum_s <= 0:
        raise ValueError('Keep a complete action and at least a two-second tail')
    return float(np.ceil(max(minimum_s, start_s + duration_s + tail_s)*1000.)/1000.)


def scan_scales(action, evaluator, *, scales=SCALES, base_period_s=.02,
                maximum_window_s=60.):
    """Evaluate every scale on the same reference-phase sampling density."""
    values = tuple(float(s) for s in scales)
    if not values or any(s <= 0 or not np.isfinite(s) for s in values):
        raise ValueError('Finite positive scales are required')
    if tuple(sorted(set(values))) != values:
        raise ValueError('Scales must be strictly increasing')
    if not np.isfinite(base_period_s) or base_period_s <= 0:
        raise ValueError('Positive sample period required')
    if not np.isfinite(maximum_window_s) or maximum_window_s <= 0:
        raise ValueError('Positive full-window bound required')
    trials = []
    for scale in values:
        curve = scaled_curve(action, scale)
        window = full_window_s(curve.duration_s)
        trial = {'scale': scale, 'duration_s': curve.duration_s,
                 'minimum_complete_native_window_s': window,
                 'within_60s_window': bool(window <= maximum_window_s),
                 'sample_period_s': base_period_s*scale,
                 'admitted_to_training': False, 'native_tested': False}
        try:
            diagnostic = evaluator(curve, base_period_s*scale)
            ratio = float(diagnostic['worst_utilization'])
            torque_pass = bool(np.isfinite(ratio) and ratio <= 1.
                               and diagnostic['ideal_demand_within_caps_at_samples'] is True)
            trial.update(diagnostic=diagnostic, torque_screen_pass=torque_pass,
                         candidate_pass=bool(torque_pass and window <= maximum_window_s))
        except (ValueError, FloatingPointError) as exc:
            trial.update(error=str(exc), torque_screen_pass=False, candidate_pass=False)
        trials.append(trial)
    selected = next((t['scale'] for t in trials if t['candidate_pass']), None)
    return {'selected_scale': selected, 'trials': trials,
            'all_grid_scales_evaluated': len(trials) == len(values),
            'sample_density': 'period = original period * scale; same source-phase density plus exact knots',
            'admitted_to_training': False, 'native_tested': False}


def _scale_optional_timing(row, scale):
    # These are explicit reference-clock fields, never physics thresholds.
    for key in ('windup_end_s', 'active_start_s', 'active_end_s'):
        if key in row:
            row[key] = float(row[key])*scale
    for key in ('active_interval_s', 'active_intervals_s'):
        if key in row:
            row[key] = (np.asarray(row[key], dtype=float)*scale).tolist()


def retimed_bank(source, scans):
    """Copy the bank, changing only the specified four action records."""
    if set(scans) != set(ACTIONS):
        raise ValueError('Exactly the four requested legacy actions must be screened')
    result = copy.deepcopy(source)
    for name in ACTIONS:
        row = result['actions'][name]
        scan = scans[name]
        scale = scan['selected_scale']
        original_times = copy.deepcopy(row['knot_time_s'])
        original_audit = row.pop('audit', None)
        row['source_audit_before_retiming'] = original_audit
        if scale is not None:
            row['knot_time_s'] = scaled_curve(row, scale).knot_time_s.tolist()
            _scale_optional_timing(row, scale)
        selected = next((t for t in scan['trials'] if t['scale'] == scale), None)
        geometric = (source['actions'][name].get('candidate_pass') is True
                     and source['actions'][name].get('sampled_reference_pass') is True)
        eligible = bool(geometric and selected is not None and selected['candidate_pass'])
        row.update(candidate_pass=eligible, sampled_reference_pass=eligible,
                   native_dynamics_tested=False, admitted_to_training=False,
                   prior_certificate_inherited=False,
                   validation_stage='retimed_current_geometry_ideal_torque_candidate_only')
        if 'successful_defence' in row:
            row['successful_defence'] = False
        row['retiming'] = {'selected_scale': scale, 'original_knot_time_s': original_times,
            'qpos_knots_exactly_preserved': row['knot_qpos'] == source['actions'][name]['knot_qpos'],
            'source_geometry_candidate_pass': geometric,
            'grid': [t['scale'] for t in scan['trials']],
            'trials_report': f'retiming/{name}/trials.json',
            'minimum_complete_native_window_s': (selected['minimum_complete_native_window_s']
                                                  if selected is not None else None),
            'scope': 'Uniform clock only. No native, opponent, reaction or connectivity certificate inherited.'}
        row['audit'] = {'sampled_ideal_torque_pass': bool(selected and selected['torque_screen_pass']),
            'geometry_path_preserved': True, 'whole_path_native_tested': False,
            'reaction_window_tested': False, 'source_old_audit_not_current_evidence': True}
        if 'source_motion' in row:
            row['source_motion']['exact_knots_preserved'] = bool(scale == 1.)
            row['source_motion']['exact_qpos_knots_preserved'] = True
            row['source_motion']['joint_knots_preserved'] = True
            row['source_motion']['time_scale'] = scale
            row['source_motion']['original_time_s'] = original_times
    untouched = set(source['actions'])-set(ACTIONS)
    if not all(result['actions'][n] == source['actions'][n] for n in untouched):
        raise AssertionError('Unrequested action changed')
    if not all(result['actions'][n]['knot_qpos'] == source['actions'][n]['knot_qpos']
               for n in source['actions']):
        raise AssertionError('Joint geometry changed')
    result['status'] = 'current_geometry_uniformly_retimed_ideal_torque_candidates_only'
    result['admitted_to_training'] = False
    result['collision_geometry_sha256'] = result['blade_geometry']['sha256']
    result['outgoing_actions'] = {}
    for name, row in result['actions'].items():
        if row.get('candidate_pass') is True:
            result['outgoing_actions'].setdefault(row['start_guard'], []).append(name)
    result['limitations'] = list(result.get('limitations', [])) + [
        'Four legacy clocks alone were retimed. Qpos, joint bounds, physical caps, scoring and all other actions are unchanged.',
        'Ideal free-space torque feasibility is sampled, not a native tracking or contact certificate.',
        'Slower attacks may weaken threats and slower parries may be too late; re-test matching full-window baselines and reactions.',
        'Legacy guard graph remains disconnected. No new transition or A2 coverage is created.',
        'No high-parry 25-degree requirement is added; the user-confirmed left/right active orientation policy is unchanged.']
    return result


def build(source, output, *, base_period_s=.02):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory; retain prior trials')
    raw = source.read_bytes()
    original = json.loads(raw)
    assembly = build_limited_floor()
    identity = blade_identity(assembly.model)
    if identity['sha256'] != original['blade_geometry']['sha256']:
        raise ValueError('Current blade geometry differs from the frozen source')
    output.mkdir(parents=True)
    scans = {}
    for name in ACTIONS:
        scan = scan_scales(original['actions'][name],
                           lambda curve, period: screen_curve(assembly.model, curve, period),
                           base_period_s=base_period_s)
        scans[name] = scan
        folder = output/'retiming'/name
        folder.mkdir(parents=True)
        (folder/'trials.json').write_text(json.dumps(_finite_json(scan), indent=2)+'\n')
        print(json.dumps({'action': name, 'selected_scale': scan['selected_scale'],
            'trials': [{'scale': t['scale'], 'worst_utilization': t.get('diagnostic', {}).get('worst_utilization'),
                        'candidate_pass': t['candidate_pass']} for t in scan['trials']]}), flush=True)
    result = retimed_bank(original, scans)
    result['retiming_provenance'] = {
        'source_report': str(source), 'source_report_sha256': hashlib.sha256(raw).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_source_sha256': assembly.source_hashes, 'scales': list(SCALES),
        'original_phase_sample_period_s': base_period_s, 'complete_native_tail_s': 2.,
        'maximum_single_reference_native_window_s': 60.,
        'all_qpos_exactly_preserved': True, 'all_other_action_records_exactly_preserved': True,
        'scoring_exactly_preserved': result['scoring'] == original['scoring'],
        'guards_exactly_preserved': result['guards'] == original['guards'],
        'source_unchanged': source.read_bytes() == raw,
        'selected_scales': {n: scans[n]['selected_scale'] for n in ACTIONS},
        'native_tests_required': True, 'admitted_to_training': False}
    if not result['retiming_provenance']['source_unchanged']:
        raise RuntimeError('Frozen source changed while scanning; do not use this output')
    for name, row in result['actions'].items():
        curve = scaled_curve(row, 1.)
        times = np.unique(np.r_[np.linspace(0., curve.duration_s, max(2, int(curve.duration_s*60)+1)),
                                curve.knot_time_s])
        np.savez_compressed(output/f'{name}.npz', time_s=times, qpos=curve.sample(times),
                            knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'source_report_snapshot.json').write_bytes(raw)
    # Reuse the frozen scene: this tool changes time, never physical geometry.
    scene = source.parent/'scene.xml'
    (output/'scene.xml').write_bytes(scene.read_bytes())
    (output/'report.json').write_text(json.dumps(_finite_json(result), ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--period', type=float, default=.02)
    args = parser.parse_args()
    build(args.source, args.output, base_period_s=args.period)
