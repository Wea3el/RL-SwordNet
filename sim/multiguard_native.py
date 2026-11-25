"""Unchanged-physics native checks for explicit multi-guard candidate paths.

Different entry/exit guards are preserved. No implicit bridge, state reset,
training admission or defence certificate is created by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle, event_velocity
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .motion_spline import C2JointSpline


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2, ensure_ascii=False) + '\n')


def load_action(report, name):
    action = report['actions'][name]
    if action.get('candidate_pass', action.get('sampled_reference_pass', False)) is not True:
        raise ValueError('Only a screened candidate may enter a native diagnostic')
    curve = C2JointSpline(action['knot_time_s'], action['knot_qpos'])
    if curve.knot_time_s[0] != 0:
        raise ValueError('Local action clock must start at zero')
    for endpoint, index in (('start_guard', 0), ('end_guard', -1)):
        expected = np.asarray(report['guards'][action[endpoint]]['qpos_rad'])
        if not np.array_equal(curve.knot_qpos[index], expected):
            raise ValueError('Do not replace an explicit guard endpoint with an IK alias')
    ranges = model_joint_ranges_rad(load_joint_constraints())
    if np.any(curve.bezier_controls < ranges[:, 0]) or np.any(curve.bezier_controls > ranges[:, 1]):
        raise ValueError('Whole reference violates the confirmed joint range')
    return action, curve


def join_exact(curves, pause_s=.12):
    """Join exact source curves at matching rest nodes; preserve their interiors.

    This scripted sequence tests transitions, not a policy choosing while in
    motion. The explicit pause cannot substitute for measured native settling.
    """
    if not curves or isinstance(pause_s, bool) or not np.isfinite(pause_s) or pause_s <= 0:
        raise ValueError('Nonempty curves and a positive explicit rest pause required')
    times, poses, intervals = [], [], []
    offset = 0.
    previous = None
    for curve in curves:
        if curve.knot_time_s[0] != 0:
            raise ValueError('Each source starts at zero')
        if previous is not None:
            if not np.array_equal(previous, curve.knot_qpos[0]):
                raise ValueError('Sequence endpoints disagree; no implicit transition is allowed')
            offset += pause_s
        intervals.append((offset, offset + curve.duration_s))
        times.extend((curve.knot_time_s + offset).tolist())
        poses.extend(curve.knot_qpos)
        offset += curve.duration_s
        previous = curve.knot_qpos[-1]
    joined = C2JointSpline(times, poses)
    for curve, (begin, _) in zip(curves, intervals):
        ts = np.linspace(0., curve.duration_s, 31)
        if not np.allclose(joined.sample(ts + begin), curve.sample(ts), atol=2e-12, rtol=0):
            raise ValueError('Joining changed a source curve')
    return joined, intervals


def boundary_ready(time_s, qpos, qvel, node, begin_s, end_s, *, required_hold_s=.1):
    """Measured rest window, never a scheduled pause interpreted as settling."""
    times, positions, speeds = np.asarray(time_s), np.asarray(qpos), np.asarray(qvel)
    selected = (times >= begin_s-1e-9) & (times <= end_s+1e-9)
    stable = (np.max(np.abs(positions-np.asarray(node)), axis=1) <= .05) & (np.max(np.abs(speeds), axis=1) <= .10)
    since, longest = None, 0.
    for t, active, okay in zip(times, selected, stable):
        if not active or not okay:
            since = None
        else:
            if since is None:
                since = float(t)
            longest = max(longest, float(t)-since)
    return {'measured_stable_duration_s': longest, 'required_hold_s': required_hold_s,
            'ready': bool(longest >= required_hold_s-1e-9)}


def run(source, output, *, action_names=(), hold_guard=None, attacker='a1',
        start_s=.75, pause_s=.12, minimum_duration=6., time_scale=1.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior native tests')
    if attacker not in ('a1', 'a2'):
        raise ValueError('Explicit physical slot required')
    if bool(action_names) == bool(hold_guard):
        raise ValueError('Select action(s) or one guard hold')
    if not np.isfinite([start_s, minimum_duration, time_scale]).all() or start_s < .1 or minimum_duration <= 0 or time_scale <= 0:
        raise ValueError('Finite positive clocks and a settling prefix required')
    source_bytes = source.read_bytes()
    report = json.loads(source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    scoring_path = ROOT / report['scoring'] if isinstance(report['scoring'], str) else None
    scoring_bytes = scoring_path.read_bytes() if scoring_path else json.dumps(report['scoring'], sort_keys=True).encode()
    scoring = json.loads(scoring_bytes)
    scoring_hash = hashlib.sha256(scoring_bytes).hexdigest()
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    other = 'a2' if attacker == 'a1' else 'a1'
    center = np.asarray(report['guards']['center']['qpos_rad'])
    actions, intervals = [], []
    if hold_guard:
        q = np.asarray(report['guards'][hold_guard]['qpos_rad'])
        curve = C2JointSpline([0., 2.], [q, q])
    else:
        pairs = [load_action(report, n) for n in action_names]
        actions, curves = zip(*pairs)
        curve, intervals = join_exact(curves, pause_s)
        curve = C2JointSpline(curve.knot_time_s * time_scale, curve.knot_qpos)
    duration = float(np.ceil(max(minimum_duration, start_s + curve.duration_s + 2.) * 1000) / 1000)
    output.mkdir(parents=True)
    (output / 'scoring_snapshot.json').write_bytes(scoring_bytes)
    programs = {attacker: ScheduledReference(curve, start_s),
                other: hold_reference(center, start_s, curve.duration_s)}
    summary = run_constrained_trial(programs, output / 'native', duration_s=duration,
        metadata={'attacker_slot': attacker, 'source_report_sha256': source_hash,
                  'probe_implementation_sha256': own_hash,
                  'scoring_sha256': scoring_hash,
                  'action_names': list(action_names), 'hold_guard': hold_guard,
                  'different_guard_endpoints_preserved': True,
                  'sequence_intervals_unscaled_s': intervals, 'time_scale': time_scale,
                  'scripted_sequence_not_policy_selection': len(action_names) > 1,
                  'not_training_admitted': True}, progress=print)
    target = actions[0].get('target_region', 'upper') if actions else 'upper'
    if target not in ('left', 'right', 'upper'):
        target = 'upper'
    audit = audit_recorded_triangle(output / 'native', scoring, target, attacker)
    write(output / 'triangle_audit.json', audit)
    incoming = [e for e in audit['all_entries'] if e['agent'] == attacker]
    first = min(incoming, key=lambda e: e['time_s']) if incoming else None
    direction = actions[0].get('direction_intent_attacker_frame', actions[0].get('intended_direction')) if actions else None
    measured = event_velocity(output / 'native', first, direction) if first and direction is not None else None
    clean = bool(summary['native_execution_pass'] and not audit['initial_overlap'] and not audit['sweep_failures'])
    unchanged = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    implementation_unchanged = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
    scoring_unchanged = scoring_path is None or hashlib.sha256(scoring_path.read_bytes()).hexdigest() == scoring_hash
    boundaries = []
    if len(action_names) > 1:
        with np.load(output / 'native' / 'trace.npz', allow_pickle=False) as trace:
            ids = slice(0, 7) if attacker == 'a1' else slice(7, 14)
            for i in range(1, len(intervals)):
                begin = start_s + intervals[i-1][1]*time_scale
                end = start_s + intervals[i][0]*time_scale
                node = actions[i]['knot_qpos'][0]
                boundaries.append(dict(boundary_ready(trace['time_s'], trace['qpos'][:, ids],
                    trace['qvel'][:, ids], node, begin, end), before_action=action_names[i]))
    aligned = measured is not None and measured['direction_cosine'] is not None and measured['direction_cosine'] >= .35
    first_in_component = first is not None and start_s <= first['time_s'] <= start_s + (intervals[0][1] if intervals else curve.duration_s)*time_scale
    result = {'status': 'native_diagnostic_complete',
        'source_report_sha256': source_hash,
        'probe_implementation_sha256': own_hash,
        'source_unchanged': unchanged,
        'implementation_unchanged': implementation_unchanged,
        'scoring_source_sha256': scoring_hash,
        'scoring_source_unchanged': scoring_unchanged,
        'action_names': list(action_names), 'hold_guard': hold_guard, 'attacker_slot': attacker,
        'native_execution_clean': clean, 'first_actual_entry': first,
        'actual_regions_entered': sorted({e['region'] for e in incoming}),
        'first_actual_entry_direction': measured,
        'material_velocity_alignment_diagnostic': aligned,
        'first_entry_in_first_component': first_in_component,
        'single_attack_candidate_pass': bool(len(action_names) == 1 and actions[0]['kind'] == 'attack'
            and clean and unchanged and implementation_unchanged and scoring_unchanged and audit['intended_region_first']
            and first_in_component and aligned),
        'measured_sequence_boundaries': boundaries,
        'all_sequence_boundaries_ready': all(b['ready'] for b in boundaries) if boundaries else None,
        'full_recorded_duration_s': summary['completed_time_s'],
        'native_failure': summary['failure'],
        'scope': 'Path/hold and physical scoring diagnostic; not defence, learned policy, transition feedback, or bank admission.',
        'admitted_to_training': False}
    write(output / 'result.json', result)
    print(json.dumps(_finite_json(result), indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--action', action='append', default=[])
    p.add_argument('--hold-guard')
    p.add_argument('--attacker', choices=['a1', 'a2'], default='a1')
    p.add_argument('--start-s', type=float, default=.75)
    p.add_argument('--pause-s', type=float, default=.12)
    p.add_argument('--time-scale', type=float, default=1.)
    a = p.parse_args()
    run(a.source, a.output, action_names=a.action, hold_guard=a.hold_guard,
        attacker=a.attacker, start_s=a.start_s, pause_s=a.pause_s, time_scale=a.time_scale)
