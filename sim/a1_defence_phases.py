"""Split an explicit zero-velocity hold into selectable candidate graph phases.

No arbitrary mid-motion interruption is allowed. The reconstructed reference
must equal the old reference; the new measured-state executor and any different
hold duration or next action still require new physical validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import _finite_json
from .motion_spline import C2JointSpline
from .multiguard_native import load_action


def split_hold(curve, hold_index):
    """Split q[k]==q[k+1] with a nonempty approach and return."""
    if type(hold_index) is not int or not 1 <= hold_index < len(curve.knot_time_s)-2:
        raise ValueError('An internal hold with explicit approach and return is required')
    k = hold_index
    if not np.array_equal(curve.knot_qpos[k], curve.knot_qpos[k+1]):
        raise ValueError('Only an explicit constant-pose hold may be split')
    start, end = curve.knot_time_s[k:k+2]
    enter = C2JointSpline(curve.knot_time_s[:k+1], curve.knot_qpos[:k+1])
    leave = C2JointSpline(curve.knot_time_s[k+1:]-end, curve.knot_qpos[k+1:])
    checks = {}
    for name, new, offset in (('enter', enter, 0.), ('exit', leave, end)):
        time = np.linspace(0., new.duration_s, 101)
        old_v, old_a = curve.derivatives(time+offset)
        new_v, new_a = new.derivatives(time)
        checks[name+'_position_preserved'] = bool(np.allclose(new.sample(time), curve.sample(time+offset), atol=1e-12, rtol=0))
        checks[name+'_velocity_preserved'] = bool(np.allclose(new_v, old_v, atol=1e-12, rtol=0))
        checks[name+'_acceleration_preserved'] = bool(np.allclose(new_a, old_a, atol=1e-11, rtol=0))
    checks['stationary_cut'] = bool(np.max(np.abs(curve.derivatives(start)[0])) == 0.
                                    and np.max(np.abs(curve.derivatives(end)[0])) == 0.)
    if not all(checks.values()):
        raise ValueError('Splitting changed the reference; do not inherit geometry')
    return enter, leave, float(end-start), checks


def build(source, action_name, output, hold_index):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Keep earlier candidate graphs and trials')
    source_bytes = source.read_bytes(); original = json.loads(source_bytes)
    action, curve = load_action(original, action_name)
    if action['kind'] != 'defence':
        raise ValueError('Choose an explicit candidate defence, not an attack')
    enter, leave, hold_duration, checks = split_hold(curve, hold_index)
    report = copy.deepcopy(original)
    node = action_name+'_hold_node'
    enter_name, exit_name = action_name+'_enter', action_name+'_exit'
    if any(name in report['actions'] for name in (enter_name, exit_name)) or node in report['guards']:
        raise ValueError('Do not overwrite existing graph identities')
    report['guards'][node] = {'qpos_rad': enter.knot_qpos[-1].tolist(),
        'origin': 'Explicit constant-pose hold of source defence',
        'independent_static_or_contact_hold_certified': False}
    for name, ref, start, finish, kind in (
        (enter_name, enter, action['start_guard'], node, 'defence'),
        (exit_name, leave, node, action['end_guard'], 'transition')):
        report['actions'][name] = {'kind': kind, 'start_guard': start, 'end_guard': finish,
            'target_region': action.get('target_region'), 'candidate_pass': True,
            'sampled_reference_pass': True, 'knot_time_s': ref.knot_time_s.tolist(),
            'knot_qpos': ref.knot_qpos.tolist(), 'admitted_to_training': False,
            'native_dynamics_tested': False, 'source_reference_slice_only': True,
            'prior_defence_certificate_inherited': False}
    report['phase_split'] = {'source': str(source), 'source_action': action_name,
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'enter': enter_name, 'hold_node': node, 'exit': exit_name,
        'original_hold_s': hold_duration, 'reference_equivalence_checks': checks,
        'automatic_exit_after_hold': False, 'arbitrary_mid_motion_interrupt': False,
        'requires_new_measured_settling_and_contact_tests': True,
        'policy_available_actions_from_hold': [exit_name],
        'opposite_side_switch_or_counterattack_not_yet_added': True}
    report['status'] = 'candidate_selectable_hold_phases_not_physically_verified'
    report['admitted_to_training'] = False
    output.mkdir(parents=True)
    (output/'source_report_snapshot.json').write_bytes(source_bytes)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'report.json').write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2)+'\n')
    return report


def merge_graphs(left_source, right_source, bridge_source, output):
    """Join only exactly identical hold states; never invent a connector."""
    paths = [Path(p).resolve() for p in (left_source, right_source, bridge_source)]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve preceding graph candidates')
    reports = [json.loads(p.read_text()) for p in paths]
    left, right, bridge = reports
    if any(report['scoring'] != left['scoring'] for report in (right, bridge)):
        raise ValueError('All graph components must use the same scoring snapshot')
    names = {'left_parry_hold': left['phase_split']['hold_node'],
             'right_parry_hold': right['phase_split']['hold_node']}
    for alias, actual in names.items():
        source = left if alias == 'left_parry_hold' else right
        if not np.array_equal(bridge['guards'][alias]['qpos_rad'], source['guards'][actual]['qpos_rad']):
            raise ValueError('The bridge endpoint does not equal the selected defence hold')
    merged = copy.deepcopy(left)
    for source, remap in ((right, {}), (bridge, names)):
        for name, guard in source['guards'].items():
            key = remap.get(name, name)
            if key in merged['guards']:
                if not np.array_equal(merged['guards'][key]['qpos_rad'], guard['qpos_rad']):
                    raise ValueError('Conflicting guard coordinates; no implicit adapter')
            else:
                merged['guards'][key] = copy.deepcopy(guard)
        for name, original in source['actions'].items():
            action = copy.deepcopy(original)
            for key in ('start_guard', 'end_guard'):
                action[key] = remap.get(action[key], action[key])
            if name in merged['actions']:
                raise ValueError('Duplicate action names require explicit review')
            merged['actions'][name] = action
    merged['status'] = 'candidate_bilateral_hold_graph_not_physically_verified'
    merged['admitted_to_training'] = False
    merged['phase_splits'] = {'left': copy.deepcopy(left['phase_split']),
                              'right': copy.deepcopy(right['phase_split'])}
    merged['bilateral_phase_graph'] = {
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'hold_aliases_checked_exactly': names,
        'cycle': [left['phase_split']['enter'], 'switch_left_hold_to_right_hold',
                  'switch_right_hold_to_left_hold', left['phase_split']['exit']],
        'no_automatic_bridge': True,
        'individual_defence_certificates_do_not_certify_this_cycle': True,
    }
    output.mkdir(parents=True)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'report.json').write_text(json.dumps(_finite_json(merged), ensure_ascii=False, indent=2)+'\n')
    return merged


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source'); p.add_argument('--action')
    p.add_argument('--output', required=True); p.add_argument('--hold-index', type=int)
    p.add_argument('--merge', nargs=3, metavar=('LEFT_PHASE', 'RIGHT_PHASE', 'BRIDGE'))
    args = p.parse_args()
    if args.merge:
        result = merge_graphs(*args.merge, args.output)
        print(json.dumps(result['bilateral_phase_graph'], indent=2))
    elif args.source and args.action and args.hold_index is not None:
        result = build(args.source, args.action, args.output, args.hold_index)
        print(json.dumps(result['phase_split'], indent=2))
    else:
        p.error('Specify source/action/hold-index or exactly three --merge sources')
