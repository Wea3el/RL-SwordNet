"""Read-only attack effectiveness review of a completed continuous chain.

Graph continuity alone does not imply that every repeated attack still hits.
This review assigns actual swept entries to actual execution intervals and
checks material-point velocity, without changing the common game referee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .assembly import _finite_json
from .a1_triangle_native import event_velocity


def assign_entries(actions, events, entries, actor):
    starts = [e for e in events if e['event'] == 'started']
    ends = [e for e in events if e['event'] == 'completed']
    if len(starts) != len(ends) or any(a['action'] != b['action'] for a, b in zip(starts, ends)):
        raise ValueError('Complete, matched execution intervals are required')
    own = sorted((e for e in entries if e['agent'] == actor), key=lambda e: e['time_s'])
    rows, assigned = [], set()
    for index, (start, end) in enumerate(zip(starts, ends)):
        if (end['time_s'] < start['time_s'] or index and start['time_s'] < ends[index-1]['time_s']):
            raise ValueError('Execution intervals must be ordered and non-overlapping')
        action = actions[start['action']]
        # At a shared timestamp the physical step ended BEFORE the next
        # decision. Attribute it once, to the preceding execution interval.
        selected = [(i, e) for i, e in enumerate(own)
                    if i not in assigned and start['time_s'] <= e['time_s'] <= end['time_s']]
        assigned.update(i for i, _ in selected)
        hits = [e for _, e in selected]
        row = {'sequence_index': index, 'name': start['action'], 'kind': action['kind'],
               'start_s': start['time_s'], 'completed_s': end['time_s'], 'entries': hits}
        if action['kind'] == 'attack':
            row.update(expected_region=action['target_region'],
                       intended_region_first=bool(hits and hits[0]['region'] == action['target_region']),
                       direction=action.get('direction_intent_attacker_frame', action.get('intended_direction')))
        rows.append(row)
    return rows, [e for i, e in enumerate(own) if i not in assigned]


def audit(source, chain_directory, output):
    source, chain_directory, output = map(lambda p: Path(p).resolve(), (source, chain_directory, output))
    if output.exists():
        raise FileExistsError('Preserve prior reviews')
    report = json.loads(source.read_text())
    result = json.loads((chain_directory/'result.json').read_text())
    summary = json.loads((chain_directory/'native/summary.json').read_text())
    triangle_path = chain_directory/'triangle_audit.json'
    triangle = json.loads(triangle_path.read_text())
    actor = result['actor_slot']
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_checks = {
        'source_matches_chain': result['source_report_sha256'] == expected_hash,
        'source_matches_native': summary['metadata']['source_report_sha256'] == expected_hash,
        'continuity_check_passed': result['chain_feasibility_pass'] is True,
        'native_execution_passed': summary['native_execution_pass'] is True,
        'no_triangle_overlap_or_evidence_error': not triangle['initial_overlap'] and not triangle['sweep_failures'],
    }
    for name in ('scene.xml', 'summary.json', 'trace.npz'):
        source_checks['triangle_binds_'+name] = (
            hashlib.sha256((chain_directory/'native'/name).read_bytes()).hexdigest()
            == triangle['native_trace_source_sha256'].get(name))
    rows, unassigned = assign_entries(report['actions'], summary['executor_events'][actor],
                                      triangle['all_entries'], actor)
    attacks = [row for row in rows if row['kind'] == 'attack']
    for row in attacks:
        row['actual_material_velocity'] = (event_velocity(chain_directory/'native', row['entries'][0], row['direction'])
                                           if row['entries'] and row['direction'] is not None else None)
        velocity = row['actual_material_velocity']
        row['direction_pass'] = bool(velocity is not None and velocity['direction_cosine'] is not None
                                     and velocity['direction_cosine'] >= .35)
    transition_entries = [dict(action=row['name'], event=e) for row in rows if row['kind'] != 'attack'
                          for e in row['entries']]
    passed = bool(all(source_checks.values()) and attacks and all(
        row['intended_region_first'] and row['direction_pass'] for row in attacks)
        and not transition_entries and not unassigned)
    value = {'schema_version': 1, 'attack_chain_effectiveness_pass': passed,
             'source_checks': source_checks, 'executed_segments': rows,
             'attack_count': len(attacks), 'attack_counts_by_region': {
                 region: sum(row['expected_region'] == region for row in attacks)
                 for region in ('left', 'right', 'upper')},
             'unexpected_transition_entries': transition_entries, 'unassigned_entries': unassigned,
             'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (source, chain_directory/'result.json', triangle_path, Path(__file__))},
             'scope': 'Scripted repeated attacks against a held opponent; not reactive defence, learned policy, or training admission.',
             'transition_entry_rule_is_diagnostic_only': True,
             'admitted_to_training': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(value), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in ('attack_chain_effectiveness_pass', 'attack_count', 'attack_counts_by_region')}, indent=2))
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--chain', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    audit(args.source, args.chain, args.output)
