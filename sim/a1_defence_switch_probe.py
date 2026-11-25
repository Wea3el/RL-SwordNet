"""Author exact guarded side-switch tests with visible opponent preparation.

The opponent uses existing explicit recovery/preparation paths before attacking.
The defender uses a declared hold-to-hold bridge. Scheduling proves physical
feasibility only, not observation-driven direction choice or training readiness.
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


def join_at_rest(first, second, pause_s):
    if not np.isfinite(pause_s) or pause_s <= 0:
        raise ValueError('An explicit positive pause at the common rest node is required')
    if not np.array_equal(first.knot_qpos[-1], second.knot_qpos[0]):
        raise ValueError('No implicit pose connector is allowed')
    shift = first.duration_s + pause_s
    result = C2JointSpline(np.r_[first.knot_time_s, second.knot_time_s + shift],
                          np.vstack([first.knot_qpos, second.knot_qpos]))
    for curve, offset in ((first, 0.), (second, shift)):
        times = np.linspace(0., curve.duration_s, 101)
        if not np.allclose(result.sample(times+offset), curve.sample(times), atol=1e-12, rtol=0):
            raise ValueError('A source path changed during concatenation')
        for a, b in zip(result.derivatives(times+offset), curve.derivatives(times)):
            if not np.allclose(a, b, atol=1e-10, rtol=0):
                raise ValueError('Source derivatives changed during concatenation')
    return result


def build(source, bridges, output, direction):
    source, bridges, output = map(lambda p: Path(p).resolve(), (source, bridges, output))
    if output.exists():
        raise FileExistsError('Preserve earlier side-switch tests')
    if direction not in ('left_to_right', 'right_to_left'):
        raise ValueError('Explicit physical defensive-side direction required')
    base, bridge = json.loads(source.read_text()), json.loads(bridges.read_text())
    if base['scoring'] != bridge['scoring']:
        raise ValueError('Different scoring snapshots cannot be joined')
    if direction == 'left_to_right':
        preparation = ['switch_negative_to_positive']
        attack_name = 'cut_up_left_to_right'
        forward_name, backward_name = 'switch_left_hold_to_right_hold', 'switch_right_hold_to_left_hold'
        target = 'right'
    else:
        preparation = ['recover_positive_high_to_low', 'switch_positive_to_negative']
        attack_name = 'cut_down_left_to_right'
        forward_name, backward_name = 'switch_right_hold_to_left_hold', 'switch_left_hold_to_right_hold'
        target = 'left'
    names = preparation + [attack_name]
    loaded = [load_action(base, name) for name in names]
    attack = loaded[0][1]
    for _, curve in loaded[1:]:
        attack = join_at_rest(attack, curve, .1)
    forward, fcurve = load_action(bridge, forward_name)
    backward, bcurve = load_action(bridge, backward_name)
    attack_start, defence_delay = .75, .1
    # Keep the destination hold through the complete prepared attack before
    # returning on its separately authored reverse path. No early-hit cutoff.
    hold = attack.duration_s + .25 - defence_delay - fcurve.duration_s
    if hold <= 0:
        raise ValueError('Bridge cannot reach and hold through the prepared attack')
    defence = join_at_rest(fcurve, bcurve, hold)
    report = copy.deepcopy(base)
    for name, node in bridge['guards'].items():
        if name in report['guards'] and not np.array_equal(node['qpos_rad'], report['guards'][name]['qpos_rad']):
            raise ValueError('Conflicting shared guard')
        report['guards'][name] = copy.deepcopy(node)
    for name, curve, kind, start, end in (
        ('prepared_attack', attack, 'attack', loaded[0][0]['start_guard'], loaded[-1][0]['end_guard']),
        ('switching_defence', defence, 'defence', forward['start_guard'], backward['end_guard'])):
        report['actions'][name] = {'kind':kind, 'start_guard':start, 'end_guard':end,
            'target_region':target, 'candidate_pass':True, 'sampled_reference_pass':True,
            'knot_time_s':curve.knot_time_s.tolist(), 'knot_qpos':curve.knot_qpos.tolist(),
            'native_dynamics_tested':False, 'admitted_to_training':False,
            'component_equivalence_only_not_whole_interaction_certified':True}
    report.update(status='candidate_prepared_attack_side_switch_pair', admitted_to_training=False)
    report['switch_probe'] = {'direction':direction, 'preparation_actions':preparation,
        'attack_action':attack_name, 'defender_forward_bridge':forward_name,
        'defender_return_bridge':backward_name, 'defence_offset_s':defence_delay,
        'attack_start_s':attack_start, 'destination_hold_s':hold,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,bridges)},
        'scope':'Scripted physical side-switch feasibility; no autonomous opponent prediction or general defence claim.'}
    output.mkdir(parents=True)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'report.json').write_text(json.dumps(_finite_json(report),ensure_ascii=False,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--bridges',required=True)
    p.add_argument('--output',required=True);p.add_argument('--direction',required=True,
        choices=('left_to_right','right_to_left'))
    a=p.parse_args();print(json.dumps(build(a.source,a.bridges,a.output,a.direction)['switch_probe'],indent=2))
