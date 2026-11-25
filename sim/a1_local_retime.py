"""Bounded local timing repair; never modify poses, forces or score rules.

Changing adjacent knot durations can change an interval's interpolated path,
even though all joint knots and bounds remain fixed. Therefore no old geometry
or native certificate is inherited. Every iteration, including failures, is
saved. This is a planning diagnostic, not a reaction-time guarantee.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .a1_dynamics_screen import free_space_demand, joint_torque_bounds, screen_curve
from .assembly import _finite_json
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


def repair(source, output, action_name, max_iterations=30):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior timing attempts')
    raw = source.read_bytes()
    report = json.loads(raw)
    action = report['actions'][action_name]
    original = C2JointSpline(action['knot_time_s'], action['knot_qpos'])
    durations = np.maximum(np.diff(original.knot_time_s), .16)
    model = build_limited_floor().model
    data = mujoco.MjData(model)
    bounds = joint_torque_bounds(model)
    output.mkdir(parents=True)
    trials = []
    chosen = None
    for iteration in range(max_iterations):
        curve = C2JointSpline(np.r_[0., np.cumsum(durations)], original.knot_qpos)
        interval_peak = []
        for a, b in zip(curve.knot_time_s[:-1], curve.knot_time_s[1:]):
            ts = np.linspace(a, b, 13)
            q = curve.sample(ts)
            vs, accs = curve.derivatives(ts)
            ratios = []
            for qi, vi, ai in zip(q, vs, accs):
                demand, _ = free_space_demand(model, data, qi, vi, ai)
                ratios.append(np.where(demand >= 0, demand/bounds[:, 1], demand/bounds[:, 0]))
            interval_peak.append(float(np.max(ratios)))
        peaks = np.asarray(interval_peak)
        item = {'iteration': iteration, 'duration_s': curve.duration_s,
                'knot_time_s': curve.knot_time_s.tolist(),
                'maximum_sampled_torque_utilization': float(peaks.max()),
                'interval_utilization': peaks.tolist()}
        if peaks.max() <= 1.:
            dense = screen_curve(model, curve, period_s=.01)
            item['dense_check'] = dense
            if dense['ideal_demand_within_caps_at_samples']:
                chosen = curve
        trials.append(item)
        (output/'attempts.json').write_text(json.dumps(_finite_json(trials), indent=2)+'\n')
        print(json.dumps({k: item[k] for k in ('iteration','duration_s','maximum_sampled_torque_utilization')}), flush=True)
        if chosen is not None or curve.duration_s > 30.:
            break
        bad = peaks > 1.
        if not bad.any():
            # The denser grid, not the coarse success, decides the next step.
            durations *= 1.08
            continue
        factors = np.ones(len(durations))
        factors[bad] = np.maximum(1.08, np.minimum(1.6, np.sqrt(peaks[bad])*1.03))
        for i in np.flatnonzero(bad):
            if i: factors[i-1] = max(factors[i-1],1.035)
            if i+1 < len(factors): factors[i+1] = max(factors[i+1],1.035)
        durations *= factors
    result = {'source_bank': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'action': action_name, 'status': 'timing_candidate' if chosen is not None else 'no_passing_timing_found',
              'joint_knots_preserved': True, 'geometric_path_preserved': False,
              'native_tested': False, 'admitted_to_training': False,
              'trials': trials, 'source_unchanged': source.read_bytes() == raw}
    if chosen is not None:
        contact_pairs = {}
        for t in np.unique(np.r_[np.arange(0., chosen.duration_s, .01), chosen.knot_time_s]):
            data.qpos[:] = chosen.sample(t)
            data.qvel[:] = 0.
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if contact.dist < -1e-8:
                    names = [model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
                    contact_pairs.setdefault('|'.join(names), {'time_s': float(t), 'geoms': names,
                                                               'depth_m': float(-contact.dist)})
        row = copy.deepcopy(action)
        row.update(knot_time_s=chosen.knot_time_s.tolist(), candidate_pass=not contact_pairs,
                   sampled_reference_pass=not contact_pairs, native_dynamics_tested=False,
                   admitted_to_training=False, prior_certificate_inherited=False)
        row['local_retiming'] = {'source_knot_time_s': original.knot_time_s.tolist(),
                                'source_bank_sha256': result['source_sha256'],
                                'sampled_self_collision_pairs': list(contact_pairs.values()),
                                'requires_fresh_swept_pair_geometry_and_native_tests': True}
        result['candidate_action'] = row
    (output/'report.json').write_text(json.dumps(_finite_json(result), indent=2)+'\n')
    return result


def publish(source, candidate_report, output):
    """Publish a fresh diagnostic snapshot, retaining rejected predecessors."""
    source, candidate_report, output = map(lambda p: Path(p).resolve(), (source, candidate_report, output))
    if output.exists():
        raise FileExistsError('Do not overwrite a frozen bank')
    raw = source.read_bytes()
    candidate = json.loads(candidate_report.read_bytes())
    if candidate['source_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Timing candidate belongs to a different bank')
    row = copy.deepcopy(candidate.get('candidate_action'))
    if not row or not row.get('candidate_pass'):
        raise ValueError('A re-screened timing candidate is required')
    bank = json.loads(raw)
    name = candidate['action']
    previous = bank['actions'][name]
    if row['knot_qpos'] != previous['knot_qpos']:
        raise ValueError('This publisher permits timing-only candidate repair')
    row['prior_uniform_time_screen'] = row.pop('retiming', None)
    row['validation_stage'] = 'locally_retimed_self_collision_screened_not_native_verified'
    bank['actions'][name] = row
    bank['status'] = 'diagnostic_bank_with_local_time_repair_not_training_ready'
    bank['admitted_to_training'] = False
    bank['local_time_repair'] = {'action': name, 'source_bank_sha256': candidate['source_sha256'],
        'candidate_report': str(candidate_report),
        'candidate_report_sha256': hashlib.sha256(candidate_report.read_bytes()).hexdigest(),
        'all_other_actions_unchanged': all(bank['actions'][n] == a for n,a in json.loads(raw)['actions'].items() if n != name)}
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(_finite_json(bank), ensure_ascii=False, indent=2)+'\n')
    return bank


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--action', required=True)
    args = p.parse_args()
    repair(args.source, args.output, args.action)
