"""Bounded geometry search and native defence checks from engagement ready.

Geometry search uses a completed baseline only to propose poses. Native tests
remain offline feasibility trials, never a controller given future motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .assembly import ROOT, _finite_json
from .a1_triangle_candidates import illegal_contacts
from .a1_triangle_parry import compare
from .constrained_trial import ScheduledReference, run_constrained_trial
from .fencing_rules import blade_sections_from_scene
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from plan1_skills.matching import tool_fk

SOURCE = ROOT / 'outputs/compact_bank_v1_natural/engagement01/report.json'


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2) + '\n')


def search(region, baseline, output, limit=5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'geometry_candidates.json'
    if destination.exists():
        raise FileExistsError('Preserve earlier candidates')
    report = json.loads(SOURCE.read_text())
    ready = np.asarray(report['center_qpos_rad'])
    baseline = Path(baseline)
    summary = json.loads((baseline / 'summary.json').read_text())
    if not summary.get('native_execution_pass'):
        raise ValueError('A completed physically valid baseline is required')
    with np.load(baseline / 'trace.npz') as trace:
        times, qpos = trace['time_s'].copy(), trace['qpos'].copy()
    attacker = summary['metadata']['attacker_slot']
    if attacker != 'a1':
        raise ValueError('Geometry search uses a1 baseline; run mode supports swapping slots')
    scene = build_limited_duel(initial_qpos=ready)
    model, data = build_limited_floor().model, None
    data = mujoco.MjData(model)
    bounds = model_joint_ranges_rad(load_joint_constraints()) + np.array([.02, -.02])
    audit_file = baseline.parent / 'result.json'
    audit = json.loads(audit_file.read_text())['triangle_audit']
    hits = [e for e in audit['all_entries'] if e['agent'] == attacker and e['region'] == region]
    if not hits:
        raise ValueError('Baseline must actually hit requested target')
    hit = min(e['time_s'] for e in hits)
    rows = []
    def aq(t):
        return qpos[np.argmin(np.abs(times - t)), :7]
    reflection = np.array([-1., -1., 1.])
    for t in np.linspace(max(.85, hit - .35), hit - .04, 4):
        scene.data.qpos[:7], scene.data.qpos[7:] = aq(t), ready
        mujoco.mj_forward(scene.model, scene.data)
        sections = blade_sections_from_scene(scene)['a1']
        begin, end = sections[0].start, sections[-1].end
        incoming_axis = (end - begin) / np.linalg.norm(end - begin)
        for fraction in (.2, .5, .85):
            crossing = begin + fraction * (end - begin)
            target = (crossing - np.array([1.75, 0., 0.])) * reflection
            for distance in (.2, .4, .65):
                for tilt in (0.,):
                    def residual(state):
                        q, along = state[:7], state[7]
                        grip, axis, _ = tool_fk(model, data, q)
                        return np.r_[10 * (grip + along * axis - target),
                                     .05 * (q - ready), .006 * (along - distance),
                                     .3 * max(0., abs((axis * reflection) @ incoming_axis) - .8)]
                    fit = least_squares(residual, np.r_[ready, distance],
                        bounds=(np.r_[bounds[:, 0], .06], np.r_[bounds[:, 1], .77]), max_nfev=150)
                    q, distance_fit = fit.x[:7], float(fit.x[7])
                    grip, axis, _ = tool_fk(model, data, q)
                    error = float(np.linalg.norm(grip + distance_fit * axis - target))
                    approach = max(.3, float(np.max(np.abs(q - ready))) * 1.875 / 2.)
                    start = float(t - approach)
                    scene.data.qpos[:7], scene.data.qpos[7:] = aq(t), q
                    mujoco.mj_forward(scene.model, scene.data)
                    illegal = illegal_contacts(scene)
                    row = dict(qpos_rad=q.tolist(), crossing_s=float(t), start_s=start,
                               approach_s=approach, distance_m=distance_fit, fraction=fraction,
                               position_error_m=error, axis=axis.tolist(), tilt=tilt,
                               joint_travel=float(np.sum(np.abs(q - ready))),
                               illegal_contacts=illegal)
                    if error > .012 or start < .72 or illegal:
                        continue
                    # A second nearby position permits a tangential sweep, with
                    # unchanged tool orientation rather than a hard straight stop.
                    shift = np.array([0., .12 if region == 'left' else -.12, .03])
                    desired_g = grip + shift
                    def sweep_residual(x):
                        g, a, _ = tool_fk(model, data, x)
                        return np.r_[10 * (g - desired_g), .8 * (a - axis), .015 * (x - q)]
                    moved = least_squares(sweep_residual, q, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=100)
                    sweep_s = max(.5, float(np.max(np.abs(moved.x - q))) * 1.875 / 2.)
                    release_relative = max(approach + sweep_s + .25, 2.35 - start)
                    recovery = max(.5, float(np.max(np.abs(moved.x - ready))) * 1.875 / 2.)
                    knot_t = [0., approach, approach + sweep_s, release_relative, release_relative + recovery]
                    knot_q = [ready, q, moved.x, moved.x, ready]
                    curve = C2JointSpline(knot_t, knot_q)
                    if max(curve.analytic_extrema()['peak_speed_rad_s']) > 2. + 1e-8:
                        continue
                    bad_path = []
                    for now in np.arange(start, t + .001, .04):
                        scene.data.qpos[:7], scene.data.qpos[7:] = aq(now), curve.sample(now - start)
                        mujoco.mj_forward(scene.model, scene.data)
                        bad_path += illegal_contacts(scene)
                    if bad_path:
                        continue
                    row.update(knot_time_s=knot_t, knot_qpos=np.asarray(knot_q).tolist(),
                               sweep_grip_shift_m=shift.tolist(), max_reference_speed_rad_s=max(curve.analytic_extrema()['peak_speed_rad_s']))
                    rows.append(row)
        print(region, 'searched', round(float(t), 3), 'eligible', len(rows), flush=True)
    selected = []
    for row in sorted(rows, key=lambda r: (r['joint_travel'], r['position_error_m'])):
        if all(np.linalg.norm(np.asarray(row['qpos_rad']) - old['qpos_rad']) > .08 for old in selected):
            row['id'] = len(selected)
            selected.append(row)
        if len(selected) >= limit:
            break
    payload = {'region': region, 'source': str(SOURCE), 'baseline': str(baseline), 'baseline_hit_s': hit,
               'ready_qpos_rad': ready.tolist(), 'selected': selected, 'all_candidates': rows,
               'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'not_training_admitted': True, 'geometry_only_not_defence_certificate': True}
    write(destination, payload)
    print(json.dumps(selected, indent=2), flush=True)
    return payload


def run(region, baseline, candidates, index, output, offset=0., time_scale=1., dwell=0., recovery_scale=1.):
    baseline, output = Path(baseline), Path(output)
    if output.exists():
        raise FileExistsError('Preserve failed probes')
    search_result = json.loads(Path(candidates).read_text())
    if search_result['region'] != region:
        raise ValueError('Candidate region mismatch')
    selected = search_result['selected'][index]
    summary = json.loads((baseline / 'summary.json').read_text())
    attacker = summary['metadata']['attacker_slot']
    defender = 'a2' if attacker == 'a1' else 'a1'
    with np.load(baseline / 'references.npz') as refs:
        attack = ScheduledReference(C2JointSpline(refs[attacker + '_knot_time_s'], refs[attacker + '_knot_qpos']),
                                    summary['program_start_s'][attacker])
    knot_t = np.asarray(selected['knot_time_s']) * time_scale
    knot_q = np.asarray(selected['knot_qpos'])
    if dwell < 0 or recovery_scale < 1.:
        raise ValueError('No negative dwell or accelerated recovery')
    if dwell:
        knot_t = np.r_[knot_t[:2], knot_t[1] + dwell, knot_t[2:] + dwell]
        knot_q = np.concatenate([knot_q[:2], knot_q[1:2], knot_q[2:]], axis=0)
    knot_t[-1] = knot_t[-2] + (knot_t[-1] - knot_t[-2]) * recovery_scale
    curve = C2JointSpline(knot_t, knot_q)
    start = selected['start_s'] + offset
    duration = summary['requested_duration_s']
    if start + curve.duration_s + 1.5 > duration:
        raise ValueError('Insufficient full recovery observation window')
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    run_constrained_trial({attacker: attack, defender: ScheduledReference(curve, start)}, output / 'native',
        duration_s=duration, metadata={'attacker_slot': attacker, 'region': region,
            'candidate': selected, 'candidate_start_s': start, 'candidate_time_scale': time_scale,
            'interception_dwell_s': dwell, 'recovery_scale': recovery_scale,
            'candidate_sha256': hashlib.sha256(Path(candidates).read_bytes()).hexdigest(),
            'not_policy_reaction': True, 'candidate_only': True}, progress=print)
    scoring = json.loads(SOURCE.read_text())['scoring']
    gate = compare(baseline, output / 'native', scoring, region, attacker)
    gate['probe_source_unchanged'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
    write(output / 'comparison.json', gate)
    print(json.dumps({'status': gate['status'], 'region': region, 'index': index,
                      'contact': gate['first_pre_hit_weapon_contact_s'],
                      'incoming_entries': gate['attacker_entries_entire_window']}, indent=2), flush=True)
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('search', 'run'))
    p.add_argument('--region', choices=('left', 'right', 'upper'), required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--candidates')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--limit', type=int, default=5)
    p.add_argument('--offset', type=float, default=0.)
    p.add_argument('--time-scale', type=float, default=1.)
    p.add_argument('--dwell', type=float, default=0.)
    p.add_argument('--recovery-scale', type=float, default=1.)
    a = p.parse_args()
    if a.stage == 'search':
        search(a.region, a.baseline, a.output, a.limit)
    else:
        if not a.candidates:
            p.error('run requires --candidates')
        run(a.region, a.baseline, a.candidates, a.index, a.output, a.offset, a.time_scale,
            a.dwell, a.recovery_scale)


if __name__ == '__main__':
    main()
