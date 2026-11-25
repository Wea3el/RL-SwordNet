"""Bounded scripted chain through an explicit named-edge candidate catalog.

This closes the catalog-bypass gap, not the training gap. No learned actor is
loaded, no opponent label selects an action, and a held opponent is not a
reactive defence test. The existing physical producer and referee are intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_chain_trial import chain_verdict, declared_opponent_qpos, event_chain_checks
from .a1_triangle_native import audit_recorded_triangle
from .a1_variant_bank import NamedEdgeBank, load_variant_bank
from .feedback_policy import SkillGraphExecutor, WaitPolicy
from .feedback_trial import run_feedback_trial


IMPLEMENTATIONS = ('sim/a1_named_chain_trial.py', 'sim/a1_variant_bank.py',
    'sim/a1_chain_trial.py', 'sim/feedback_trial.py', 'sim/feedback_policy.py',
    'sim/motion_spline.py', 'sim/joint_constraints.py', 'sim/a1_triangle_native.py')
NATIVE_ARTIFACTS = ('summary.json', 'scene.xml', 'trace.npz', 'contacts.json',
                    'observations.json', 'graph_references.npz')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_named_chain(bank, sequence, *, repeats=1):
    if not isinstance(bank, NamedEdgeBank):
        raise ValueError('An explicitly loaded named-edge bank is required')
    if (not isinstance(sequence, (list, tuple)) or not sequence or
        any(not isinstance(name, str) or name not in bank.policy_order for name in sequence)):
        raise ValueError('Every scripted edge must occur in policy_order; wait is not a timed edge')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Positive integer repeat count required')
    executor = bank.new_executor()
    names, previous = list(sequence)*repeats, bank.initial_guard
    report = json.loads(bank.source_bytes)
    rows, curves = [], {}
    for name in names:
        edge = executor.actions[name]
        if edge.start_node != previous:
            raise ValueError(f'No selected explicit route: {previous} -> {edge.start_node}')
        rows.append(report['actions'][name])
        curves[name] = edge.curve
        previous = edge.end_node
    duration = float(np.ceil((sum(curves[name].duration_s for name in names)
                              + max(4., len(names)*1.) + 1.)*1000.)/1000.)
    if duration > 60.:
        raise ValueError('Complete named chain plus settle/tail exceeds bounded 60 seconds')
    return {'names': names, 'actions': rows, 'curves': curves,
            'initial_guard': bank.initial_guard, 'final_guard': previous,
            'duration_s': duration,
            'action_ids': [bank.action_names.index(name) for name in names]}


class CatalogSequence:
    """Scripted choices gated by the real executor's current availability."""
    def __init__(self, bank, plan):
        bank.assert_identity()
        # The unchanged native producer snapshots policy.__dict__ as JSON.
        # Keep only plain metadata here, never a bank/Path/executor object.
        report = json.loads(bank.source_bytes)
        self.policy_order = list(bank.policy_order)
        self.action_names = list(bank.action_names)
        self.catalog_signature = bank.actor_signature()
        self.entry_qpos = {name: report['guards'][report['actions'][name]['start_guard']]['qpos_rad']
                           for name in bank.policy_order}
        self.names = list(plan['names'])
        self.next = 0
        self.events = []

    def choose(self, observation, available):
        if set(available)-set(self.policy_order):
            raise ValueError('Actual executor exposed an edge outside the selected catalog')
        if self.next == len(self.names) or self.names[self.next] not in available:
            return None
        name = self.names[self.next]
        q = np.asarray(observation['own_q_rad'], dtype=float)
        velocity = np.asarray(observation['own_qvel_rad_s'], dtype=float)
        if (q.shape != (7,) or velocity.shape != (7,) or not np.isfinite([q, velocity]).all() or
            np.max(np.abs(q-np.asarray(self.entry_qpos[name]))) > .05 or
            np.max(np.abs(velocity)) > .10):
            raise ValueError('Available edge does not match measured settled entry state')
        self.events.append({'event': 'catalog_edge_available', 'sequence_index': self.next,
            'action': name, 'action_id': self.action_names.index(name),
            'action_mask': [True]+[edge in available for edge in self.policy_order],
            'own_q_rad': q.tolist(), 'own_qvel_rad_s': velocity.tolist()})
        self.next += 1
        return name


def recorded_catalog_checks(folder, bank, actor):
    """Verify actual initialized executor and frozen references match catalog."""
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_bytes())
    executor = bank.new_executor()
    graph = summary.get('graph', {}).get(actor, {})
    checks = {
        'actor_executor_has_exact_catalog_edges': set(graph.get('actions', {})) == set(bank.policy_order),
        'actor_executor_has_exact_catalog_nodes': graph.get('nodes') == {
            name: q.tolist() for name, q in executor.nodes.items()},
        'actor_executor_has_catalog_initial_node': graph.get('initial_node') == bank.initial_guard,
        'native_metadata_binds_catalog_signature': summary.get('metadata', {}).get('catalog_signature') == bank.actor_signature(),
    }
    expected = {f'{actor}__{name}__{key}': getattr(edge.curve, key)
                for name, edge in executor.actions.items()
                for key in ('knot_time_s', 'knot_qpos', 'coefficients')}
    with np.load(folder/'graph_references.npz', allow_pickle=False) as arrays:
        keys = {key for key in arrays.files if key.startswith(actor+'__')}
        checks['actual_frozen_reference_keys_match_catalog'] = keys == set(expected)
        checks['actual_frozen_references_match_catalog'] = bool(keys == set(expected) and all(
            np.array_equal(arrays[key], value) for key, value in expected.items()))
    checks['actor_executor_edge_endpoints_match_catalog'] = all(
        graph.get('actions', {}).get(name) == {'start_node': edge.start_node, 'end_node': edge.end_node}
        for name, edge in executor.actions.items())
    return checks


def recorded_choice_checks(folder, bank, actor, policy, executor_events):
    observations = json.loads((Path(folder)/'observations.json').read_bytes())
    requests = [row for row in observations if row.get('requests', {}).get(actor) is not None]
    starts = [row for row in executor_events if row['event'] == 'started']
    checks = {'native_requests_match_scripted_catalog_choices':
        [row['requests'][actor] for row in requests] == [e['action'] for e in policy.events],
        'native_requests_match_real_executor_start_times': len(requests) == len(starts) and all(
            row['requests'][actor] == event['action'] and abs(row['time_s']-event['time_s']) <= 1e-9
            for row, event in zip(requests, starts)),
        'choice_ids_match_frozen_catalog_order': all(
            e['action_id'] == bank.action_names.index(e['action']) for e in policy.events)}
    checks['choices_bind_current_public_state_and_availability'] = bool(
        len(requests) == len(policy.events) and all(
            np.array_equal(request.get('physical', {}).get(actor, {}).get('own_q_rad'), event['own_q_rad']) and
            np.array_equal(request.get('physical', {}).get(actor, {}).get('own_qvel_rad_s'), event['own_qvel_rad_s']) and
            event['action_mask'] == [True]+[name in request.get('available', {}).get(actor, ())
                                             for name in bank.policy_order]
            for request, event in zip(requests, policy.events)))
    log = [dict(event, time_s=request['time_s']) for event, request in zip(policy.events, requests)]
    return checks, log


def run(source, output, *, policy_order, initial_guard, sequence, repeats=1,
        actor='a1', opponent_guard):
    if actor not in ('a1', 'a2'):
        raise ValueError('Explicit physical actor slot required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior named-chain evidence; choose a fresh output')
    bank = load_variant_bank(source, policy_order=policy_order, initial_guard=initial_guard)
    plan = plan_named_chain(bank, sequence, repeats=repeats)
    report = json.loads(bank.source_bytes)
    other = 'a2' if actor == 'a1' else 'a1'
    actor_executor = bank.new_executor()  # All and only catalog edges, not all report edges.
    held = SkillGraphExecutor({'held': declared_opponent_qpos(report, opponent_guard)}, {}, 'held')
    policy = CatalogSequence(bank, plan)
    signature = bank.actor_signature()
    hashes = {name: _hash(ROOT/name) for name in IMPLEMENTATIONS}
    scoring_path = ROOT/report['scoring'] if isinstance(report['scoring'], str) else None
    scoring_bytes = (scoring_path.read_bytes() if scoring_path else
                     json.dumps(report['scoring'], sort_keys=True).encode())
    output.mkdir(parents=True)
    (output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'source_report_snapshot.json').write_bytes(bank.source_bytes)
    (output/'scoring_snapshot.json').write_bytes(scoring_bytes)
    (output/'catalog_snapshot.json').write_text(json.dumps(bank.summary(), indent=2)+'\n')
    catalog_hash = _hash(output/'catalog_snapshot.json')
    metadata = {'actor_slot': actor, 'source_report_sha256': bank.identity()['bank_sha256'],
        'catalog_signature': signature, 'scripted_sequence': plan['names'],
        'scripted_action_ids': plan['action_ids'], 'opponent_guard': opponent_guard,
        'not_a_learned_actor_test': True, 'not_reactive_defence': True, 'admitted_to_training': False}
    summary = run_feedback_trial({actor: actor_executor, other: held},
        {actor: policy, other: WaitPolicy()}, output/'native', duration_s=plan['duration_s'],
        metadata=metadata, progress=print)
    events = summary['executor_events'][actor]
    checks = event_chain_checks(plan, events, policy)
    checks.update(recorded_catalog_checks(output/'native', bank, actor))
    choice_checks, choice_log = recorded_choice_checks(output/'native', bank, actor, policy, events)
    checks.update(choice_checks)
    checks.update(native_execution_pass=summary['native_execution_pass'] is True,
        evidence_complete=summary['evidence_complete'] is True,
        no_post_initialization_resets=summary['qpos_resets_after_initialization'] == 0,
        source_unchanged=Path(source).resolve().read_bytes() == bank.source_bytes,
        implementation_unchanged=all(_hash(ROOT/name) == value for name, value in hashes.items()),
        scoring_unchanged=scoring_path is None or scoring_path.read_bytes() == scoring_bytes,
        catalog_snapshot_unchanged=_hash(output/'catalog_snapshot.json') == catalog_hash,
        both_finally_settled=all(summary['final_settled'].get(slot) is True for slot in ('a1', 'a2')),
        final_node_matches_plan=summary['final_nodes'][actor] == plan['final_guard'])
    native_hashes = {name: _hash(output/'native'/name) for name in NATIVE_ARTIFACTS}
    target = next((row.get('target_region') for row in plan['actions']
        if row.get('kind') == 'attack' and row.get('target_region') in ('left', 'right', 'upper')), 'upper')
    audit = audit_recorded_triangle(output/'native', json.loads(scoring_bytes), target, actor)
    checks['no_initial_triangle_overlap'] = not audit['initial_overlap']
    checks['triangle_evidence_valid'] = not audit['sweep_failures']
    checks['native_artifacts_unchanged_during_audit'] = all(
        _hash(output/'native'/name) == value for name, value in native_hashes.items())
    checks['triangle_audit_bound_to_same_native_trace'] = audit.get('native_trace_source_sha256') == {
        name: native_hashes[name] for name in ('summary.json', 'trace.npz', 'scene.xml')}
    (output/'triangle_audit.json').write_text(json.dumps(_finite_json(audit), indent=2)+'\n')
    result = {'schema_version': 1, 'status': 'named_catalog_chain_diagnostic_complete',
        'verdict': chain_verdict(checks, summary, policy),
        'named_catalog_chain_pass': all(checks.values()), 'checks': checks,
        'catalog_signature': signature, 'source_report_sha256': bank.identity()['bank_sha256'],
        'implementation_sha256': hashes,
        'native_artifact_sha256': native_hashes,
        'scripted_sequence': plan['names'], 'scripted_action_ids': plan['action_ids'],
        'choice_id_name_log': choice_log, 'executor_events': events,
        'initial_guard': plan['initial_guard'], 'final_guard': plan['final_guard'],
        'actor_slot': actor, 'opponent_guard': opponent_guard,
        'full_observation_window_s': plan['duration_s'], 'native_failure': summary['failure'],
        'native_completion': summary['completion'],
        'not_a_learned_actor_test': True, 'not_selfplay': True,
        'not_a_reactive_opponent_test': True, 'not_a_defence_certificate': True,
        'admitted_to_training': False,
        'scope': 'Selected catalog edges physically sequenced against a held opponent; not combat policy performance.'}
    (output/'result.json').write_text(json.dumps(_finite_json(result), indent=2)+'\n')
    print(json.dumps({'named_catalog_chain_pass': result['named_catalog_chain_pass'],
        'verdict': result['verdict'], 'admitted_to_training': False}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--policy-action', action='append', required=True,
                        help='Catalog edge in frozen ID order after wait=0')
    parser.add_argument('--initial-guard', required=True)
    parser.add_argument('--action', action='append', required=True, help='Ordered scripted edge from that catalog')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--actor', choices=('a1', 'a2'), default='a1')
    parser.add_argument('--opponent-guard', required=True, help='Explicit held public test condition')
    args = parser.parse_args()
    run(args.source, args.output, policy_order=tuple(args.policy_action), initial_guard=args.initial_guard,
        sequence=args.action, repeats=args.repeats, actor=args.actor, opponent_guard=args.opponent_guard)


if __name__ == '__main__':
    main()
