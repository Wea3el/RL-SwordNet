"""Candidate side transfer into a dynamic parry, not into its contact hold.

This authors a scheduled physical experiment only.  Timing taken from the
opponent script is NOT available to a deployed policy.  Successful execution
still needs a matched full-window native test and later observation control.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .a1_defence_switch_probe import join_at_rest
from .assembly import _finite_json
from .multiguard_native import load_action


def compose_reentry(transfer, parry, final_attack_offset_s, defence_offset_s=.1):
    """Preserve both authored paths; wait only at their common rest node."""
    if not np.isfinite(final_attack_offset_s) or not np.isfinite(defence_offset_s):
        raise ValueError('Finite physical clocks required')
    wait = final_attack_offset_s - defence_offset_s - transfer.duration_s
    if wait < .05:
        raise ValueError('Insufficient preparation window; do not silently delay the attack')
    combined = join_at_rest(transfer, parry, wait)
    return combined, float(wait)


def build(prepared_source, transfer_source, transfer_name, parry_source,
          parry_name, output, *, defence_offset_s=.1):
    paths = [Path(p).resolve() for p in (prepared_source, transfer_source, parry_source)]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve preceding experiments')
    prepared, transit, parries = [json.loads(p.read_text()) for p in paths]
    if any(r['scoring'] != prepared['scoring'] for r in (transit, parries)):
        raise ValueError('Identical scoring snapshots required')
    prep_meta = prepared['switch_probe']
    final_attack_name = prep_meta['attack_action']
    attack, attack_curve = load_action(prepared, 'prepared_attack')
    _, final_curve = load_action(prepared, final_attack_name)
    final_offset = attack_curve.duration_s - final_curve.duration_s
    times = np.linspace(0., final_curve.duration_s, 101)
    if not np.allclose(attack_curve.sample(final_offset+times), final_curve.sample(times),
                       atol=1e-12, rtol=0.):
        raise ValueError('Prepared attack does not preserve the claimed final strike')
    transfer, transfer_curve = load_action(transit, transfer_name)
    parry, parry_curve = load_action(parries, parry_name)
    if parry['kind'] != 'defence' or transfer['end_guard'] != parry['start_guard']:
        raise ValueError('Transfer must end at the declared dynamic-parry entry')
    curve, wait = compose_reentry(transfer_curve, parry_curve, final_offset, defence_offset_s)
    report = copy.deepcopy(prepared)
    for source in (transit, parries):
        for name, node in source['guards'].items():
            if name in report['guards'] and not np.array_equal(
                    node['qpos_rad'], report['guards'][name]['qpos_rad']):
                raise ValueError('Conflicting guard identity; no implicit connector')
            report['guards'][name] = copy.deepcopy(node)
    report['actions']['switch_then_dynamic_parry'] = {
        'kind': 'defence', 'start_guard': transfer['start_guard'],
        'end_guard': parry['end_guard'], 'target_region': attack['target_region'],
        'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
        'candidate_pass': True, 'sampled_reference_pass': False,
        'component_reference_preservation_pass': True,
        'candidate_admission_scope': 'Exact offline diagnostic only, not screened composite geometry or training.',
        'whole_composite_geometry_status': 'not_verified_by_this_builder',
        'native_dynamics_tested': False, 'admitted_to_training': False,
        'scope': 'Component-preserving composition only; not a whole-interaction certificate.'}
    report.update(status='candidate_side_transfer_then_dynamic_parry', admitted_to_training=False)
    report['reentry_probe'] = {
        'transfer_action': transfer_name, 'dynamic_parry_action': parry_name,
        'final_attack_action': final_attack_name, 'final_attack_offset_s': final_offset,
        'defence_offset_s': defence_offset_s, 'entry_wait_s': wait,
        'dynamic_parry_offset_in_defence_s': transfer_curve.duration_s+wait,
        'parry_and_final_attack_start_simultaneously': True,
        'future_opponent_schedule_for_offline_test_only': True,
        'autonomous_direction_choice': False, 'admitted_to_training': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    output.mkdir(parents=True)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'report.json').write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-source', required=True)
    p.add_argument('--transfer-source', required=True)
    p.add_argument('--transfer', required=True)
    p.add_argument('--parry-source', required=True)
    p.add_argument('--parry', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--defence-offset-s', type=float, default=.1)
    a = p.parse_args()
    r = build(a.prepared_source, a.transfer_source, a.transfer, a.parry_source,
              a.parry, a.output, defence_offset_s=a.defence_offset_s)
    print(json.dumps(r['reentry_probe'], indent=2))
