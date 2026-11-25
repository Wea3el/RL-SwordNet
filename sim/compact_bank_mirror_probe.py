"""Read-only-source baseline diagnostics for a compact left/right skill bank.

This does not mirror joint signs, alter physics or certify defence. Distinguish
local target-side coverage from swapping the two physical robot slots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import mujoco

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle, event_velocity
from .a1_triangle_parry import compare
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .motion_spline import C2JointSpline
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from plan1_skills.matching import tool_fk


SOURCE = ROOT / 'outputs/a1_triangle_screen_v2/report.json'


def run(action_name: str, attacker: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Preserve earlier diagnostics; choose a fresh output')
    report = json.loads(SOURCE.read_text())
    action = report['actions'][action_name]
    if not action['sampled_reference_pass']:
        raise ValueError('Unscreened source action')
    ready = np.asarray(report['guards']['center']['qpos_rad'])
    curve = C2JointSpline(np.asarray(action['knot_time_s']) * 2., action['knot_qpos'])
    if not np.array_equal(curve.knot_qpos[0], ready) or not np.array_equal(curve.knot_qpos[-1], ready):
        raise ValueError('This probe requires exact common-center entry and exit')
    defender = 'a2' if attacker == 'a1' else 'a1'
    attack = ScheduledReference(curve, .75)
    programs = {attacker: attack, defender: hold_reference(ready, .75, curve.duration_s)}
    horizon = max(12., np.ceil((attack.end_s + 2.) * 1000.) / 1000.)
    output.mkdir(parents=True)
    summary = run_constrained_trial(programs, output / 'native', duration_s=horizon,
        metadata={'action': action_name, 'attacker_slot': attacker,
                  'source_report_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'time_scale': 2., 'exact_center_entry_exit': True,
                  'local_target_side_not_joint_mirror': True,
                  'candidate_only': True}, progress=lambda x, **kw: print(action_name, attacker, x, **kw))
    audit = audit_recorded_triangle(output / 'native', report['scoring'], action['target_region'], attacker)
    event = next((e for e in audit['all_entries'] if e['agent'] == attacker and e['region'] == action['target_region']), None)
    if event is not None:
        audit['first_entry_measured_direction'] = event_velocity(output / 'native', event, action['direction_intent_attacker_frame'])
    with np.load(output / 'native/trace.npz') as trace:
        sl = slice(0, 7) if attacker == 'a1' else slice(7, 14)
        audit['attacker_max_abs_actual_joint_speed_rad_s'] = np.max(np.abs(trace['qvel'][:, sl]), axis=0).tolist()
    audit.update(action=action_name, exact_center_entry_exit=True, local_geometric_mirror_certified=False,
                 defence_certified=False, admitted_to_training=False,
                 final_settled=summary['final_settled'])
    (output / 'triangle_audit.json').write_text(json.dumps(_finite_json(audit), indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('action', 'attacker_slot', 'reference_candidate_pass', 'native_runner_failure',
                                          'intended_region_first', 'final_settled')}, indent=2), flush=True)
    return audit


def run_defence(bank_path: Path, baseline: Path, start_s: float, output: Path, mirror_left: bool = False):
    """Test the bank's right parry against an identical saved right thrust."""
    if output.exists():
        raise FileExistsError('Preserve earlier defence candidates')
    bank = json.loads(bank_path.read_text())
    action = bank['actions']['defence_right']
    summary = json.loads((baseline / 'summary.json').read_text())
    attacker = summary['metadata']['attacker_slot']
    defender = 'a2' if attacker == 'a1' else 'a1'
    with np.load(baseline / 'references.npz') as arrays:
        attack = ScheduledReference(C2JointSpline(arrays[attacker + '_knot_time_s'], arrays[attacker + '_knot_qpos']),
                                    summary['program_start_s'][attacker])
    mirror_evidence = None
    if mirror_left:
        model = build_limited_floor().model
        data = mujoco.MjData(model)
        bounds = model_joint_ranges_rad(load_joint_constraints()) + np.array([.02, -.02])
        ready = np.asarray(bank['nodes']['ready'])
        poses, fits = [], []
        for name in ('defence_left_knot_1', 'defence_left_knot_2'):
            grip, axis, _ = tool_fk(model, data, bank['nodes'][name])
            target_g, target_a = grip * [1, -1, 1], axis * [1, -1, 1]
            def residual(q):
                g, a, _ = tool_fk(model, data, q)
                return np.r_[10 * (g - target_g), 2 * (a - target_a), .003 * (q - ready)]
            fit = least_squares(residual, ready, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=250)
            g, a, _ = tool_fk(model, data, fit.x)
            position_error, axis_error = float(np.linalg.norm(g - target_g)), float(np.linalg.norm(a - target_a))
            if position_error > .001 or axis_error > .001:
                raise ValueError('Task-space mirror IK did not reach the requested tool pose')
            poses.append(fit.x)
            fits.append(dict(source_node=name, qpos_rad=fit.x.tolist(),
                grip_target_m=target_g.tolist(), axis_target=target_a.tolist(),
                grip_error_m=position_error, axis_vector_error=axis_error))
        curve = C2JointSpline([0., .84, 1.84, 4.0, 5.0], [ready, poses[0], poses[1], poses[1], ready])
        mirror_evidence = {'source': 'left defence tool grip and axis reflected across local x-z plane',
                           'fits': fits, 'endpoint': 'shared original ready, not a reflected whole-arm ready'}
    else:
        curve = C2JointSpline(action['knot_time_s'], [bank['nodes'][n] for n in action['waypoint_nodes']])
    if max(curve.analytic_extrema()['peak_speed_rad_s']) > 2. + 1e-9:
        raise ValueError('Candidate reference speed exceeds existing reference design ceiling')
    output.mkdir(parents=True)
    result = run_constrained_trial({attacker: attack, defender: ScheduledReference(curve, start_s)}, output / 'native',
        duration_s=summary['requested_duration_s'],
        metadata={'attacker_slot': attacker, 'bank_sha256': hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                  'bank_path': str(bank_path), 'defence_action': 'defence_right', 'defence_start_s': start_s,
                  'taskspace_mirror_evidence': mirror_evidence,
                  'not_policy_reaction': True, 'candidate_only': True}, progress=print)
    score = json.loads((ROOT / bank['scoring_source']).read_text())
    gate = compare(baseline, output / 'native', score, target='right', attacker=attacker)
    (output / 'comparison.json').write_text(json.dumps(_finite_json(gate), indent=2) + '\n')
    print(json.dumps({'output': str(output), 'status': gate['status'], 'native_pass': result['native_execution_pass'],
                      'first_contact_s': gate['first_pre_hit_weapon_contact_s'],
                      'incoming_entries': gate['attacker_entries_entire_window']}, indent=2), flush=True)
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action')
    p.add_argument('--attacker', choices=('a1', 'a2'))
    p.add_argument('--output', required=True)
    p.add_argument('--bank')
    p.add_argument('--baseline')
    p.add_argument('--defence-start', type=float)
    p.add_argument('--mirror-left', action='store_true')
    a = p.parse_args()
    if a.bank:
        if a.baseline is None or a.defence_start is None:
            p.error('--bank requires --baseline and --defence-start')
        run_defence(Path(a.bank), Path(a.baseline), a.defence_start, Path(a.output), a.mirror_left)
    else:
        if a.action is None or a.attacker is None:
            p.error('baseline mode requires --action and --attacker')
        run(a.action, a.attacker, Path(a.output))


if __name__ == '__main__':
    main()
