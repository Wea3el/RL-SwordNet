"""Measured-state multi-action chaining diagnostic, never an RL admission gate.

The sequence is intentionally scripted. Its next edge is requested only when
the existing executor exposes it after actual endpoint settling. No scheduled
splice, implicit connector, interrupt, or post-initialization reset is added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle
from .feedback_policy import GraphAction, SkillGraphExecutor, WaitPolicy
from .feedback_trial import run_feedback_trial
from .multiguard_native import load_action


def hashes_unchanged(hashes):
    return all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in hashes.items())


def chain_verdict(checks, summary, decisions):
    if all(checks.values()):
        return 'passed_candidate_chain_only'
    if summary.get('completion', {}).get('is_truncation'):
        return 'incomplete_not_a_physical_failure'
    if not summary.get('failure') and decisions.next < len(decisions.names):
        return 'incomplete_pending_sequence_not_a_physical_failure'
    return 'not_passed'


def plan_chain(report, sequence, *, repeats=1):
    if (not isinstance(sequence, (list, tuple)) or not sequence
            or any(not isinstance(name, str) or not name for name in sequence)):
        raise ValueError('Explicit nonempty action-name sequence required')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Positive integer repeat count required')
    names = list(sequence) * repeats
    actions, curves = [], {}
    previous = None
    for name in names:
        action, curve = load_action(report, name)
        if previous is not None and previous != action['start_guard']:
            raise ValueError(f'Missing explicit route: {previous} -> {action["start_guard"]}')
        actions.append(action)
        curves[name] = curve
        previous = action['end_guard']
    # Reference time plus real measured-settle slack for EVERY edge and tail.
    duration = float(np.ceil((sum(curves[name].duration_s for name in names)
                              + max(4., 1.0 * len(names)) + 1.) * 1000.) / 1000.)
    if duration > 60.:
        raise ValueError('Split this diagnostic: complete chain plus settling exceeds 60 s')
    return {'names': names, 'actions': actions, 'curves': curves,
            'initial_guard': actions[0]['start_guard'], 'final_guard': actions[-1]['end_guard'],
            'duration_s': duration}


class AvailableSequence:
    """Scripted edge order, but start times come only from observed availability."""
    def __init__(self, names):
        self.names = tuple(names)
        self.next = 0
        self.events = []

    def choose(self, observation, available):
        if self.next == len(self.names) or self.names[self.next] not in available:
            return None
        action = self.names[self.next]
        self.events.append({'event': 'next_edge_available', 'sequence_index': self.next,
                            'action': action, 'own_q_rad': list(observation['own_q_rad']),
                            'own_qvel_rad_s': list(observation['own_qvel_rad_s'])})
        self.next += 1
        return action


def make_executor(report, plan):
    nodes, actions = {}, {}
    for name, row in zip(plan['names'], plan['actions']):
        for key in ('start_guard', 'end_guard'):
            node = row[key]
            nodes[node] = np.asarray(report['guards'][node]['qpos_rad'])
        actions[name] = GraphAction(name, row['start_guard'], row['end_guard'], plan['curves'][name])
    return SkillGraphExecutor(nodes, actions, plan['initial_guard'])


def declared_opponent_qpos(report, guard_name='center'):
    """Choose a named public test condition, never rename/replace the center."""
    if not isinstance(guard_name, str) or guard_name not in report['guards']:
        raise ValueError('Explicit existing held opponent guard required')
    q = np.asarray(report['guards'][guard_name]['qpos_rad'], dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('Held opponent guard must contain seven finite joint angles')
    return q.copy()


def event_chain_checks(plan, events, decisions):
    starts = [e for e in events if e['event'] == 'started']
    ends = [e for e in events if e['event'] == 'completed']
    names = plan['names']
    checks = {
        'all_actions_started_in_order': [e['action'] for e in starts] == names,
        'all_actions_completed_in_order': [e['action'] for e in ends] == names,
        'all_script_requests_consumed': decisions.next == len(names),
        'decision_events_match_starts': len(decisions.events) == len(starts),
    }
    checks['all_starts_at_declared_nodes'] = len(starts) == len(names) and all(
        e['node'] == a['start_guard'] for e, a in zip(starts, plan['actions']))
    checks['all_completions_at_declared_nodes'] = len(ends) == len(names) and all(
        e['node'] == a['end_guard'] for e, a in zip(ends, plan['actions']))
    checks['next_action_never_precedes_actual_completion'] = len(starts) == len(ends) == len(names) and all(
        starts[i]['time_s'] >= ends[i-1]['time_s'] for i in range(1, len(names)))
    checks['each_edge_runs_its_complete_clock'] = len(starts) == len(ends) == len(names) and all(
        end['time_s'] - start['time_s'] >= plan['curves'][name].duration_s - 1e-9
        for name, start, end in zip(names, starts, ends))
    checks['requests_use_observed_entry_state'] = len(decisions.events) == len(names) and all(
        np.max(np.abs(np.asarray(e['own_q_rad']) - np.asarray(plan['curves'][name].knot_qpos[0]))) <= .05
        and np.max(np.abs(e['own_qvel_rad_s'])) <= .10
        for name, e in zip(names, decisions.events))
    return checks


def run(source, output, sequence, *, repeats=1, actor='a1', opponent_guard='center'):
    if actor not in ('a1', 'a2'):
        raise ValueError('Explicit physical actor slot required')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous trials; use a fresh output')
    source_bytes = source.read_bytes()
    report = json.loads(source_bytes)
    plan = plan_chain(report, sequence, repeats=repeats)
    other = 'a2' if actor == 'a1' else 'a1'
    # This is a hold baseline, not a reactive opponent or learned policy.
    held = SkillGraphExecutor({'held': declared_opponent_qpos(report, opponent_guard)}, {}, 'held')
    policy = AvailableSequence(plan['names'])
    executors = {actor: make_executor(report, plan), other: held}
    policies = {actor: policy, other: WaitPolicy()}
    dependencies = (Path(__file__), ROOT/'sim/feedback_trial.py', ROOT/'sim/feedback_policy.py',
                    ROOT/'sim/multiguard_native.py', ROOT/'sim/a1_triangle_native.py')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    scoring_path = ROOT/report['scoring'] if isinstance(report['scoring'], str) else None
    scoring_bytes = (scoring_path.read_bytes() if scoring_path else
                     json.dumps(report['scoring'], sort_keys=True).encode())
    scoring = json.loads(scoring_bytes)
    output.mkdir(parents=True)
    (output/'runner_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'source_report_snapshot.json').write_bytes(source_bytes)
    (output/'scoring_snapshot.json').write_bytes(scoring_bytes)
    metadata = {'actor_slot': actor, 'scripted_chain': plan['names'],
                'source_report_sha256': source_hash, 'scripted_selection_not_rl': True,
                'runner_source_sha256': hashes[str(Path(__file__))],
                'opponent_condition': f'held explicit source {opponent_guard}; not reactive',
                'opponent_guard': opponent_guard,
                'no_implicit_connectors': True, 'admitted_to_training': False}
    summary = run_feedback_trial(executors, policies, output/'native',
                                 duration_s=plan['duration_s'], metadata=metadata, progress=print)
    checks = event_chain_checks(plan, summary['executor_events'][actor], policy)
    checks.update(
        native_execution_pass=summary['native_execution_pass'] is True,
        evidence_complete=summary['evidence_complete'] is True,
        no_post_initialization_resets=summary['qpos_resets_after_initialization'] == 0,
        source_unchanged=source.read_bytes() == source_bytes,
        implementation_unchanged=hashes_unchanged(hashes),
        scoring_unchanged=scoring_path is None or scoring_path.read_bytes() == scoring_bytes,
        both_finally_settled=all(summary['final_settled'].get(s) is True for s in ('a1', 'a2')),
        final_node_matches_plan=summary['final_nodes'][actor] == plan['final_guard'])
    target = next((a.get('target_region') for a in plan['actions']
                   if a.get('kind') == 'attack' and a.get('target_region') in ('left', 'right', 'upper')), 'upper')
    audit = audit_recorded_triangle(output/'native', scoring, target, actor)
    checks['no_initial_triangle_overlap'] = not audit['initial_overlap']
    checks['triangle_evidence_valid'] = not audit['sweep_failures']
    (output/'triangle_audit.json').write_text(json.dumps(_finite_json(audit), indent=2)+'\n')
    result = {'schema_version': 1, 'status': 'chain_diagnostic_complete',
              'verdict': chain_verdict(checks, summary, policy),
              'chain_feasibility_pass': all(checks.values()), 'checks': checks,
              'source_report_sha256': source_hash, 'implementation_sha256': hashes,
              'scripted_sequence': plan['names'], 'initial_guard': plan['initial_guard'],
              'final_guard': plan['final_guard'], 'actor_slot': actor,
              'opponent_guard': opponent_guard,
              'full_observation_window_s': plan['duration_s'],
              'native_failure': summary['failure'], 'native_completion': summary['completion'],
              'executor_events': summary['executor_events'][actor],
              'attack_entries_diagnostic': [e for e in audit['all_entries'] if e['agent'] == actor],
              'timing': {name: plan['curves'][name].report() for name in dict.fromkeys(plan['names'])},
              'not_a_reactive_opponent_test': True, 'not_a_naturalness_certificate': True,
              'not_a_defence_certificate': True, 'admitted_to_training': False}
    (output/'result.json').write_text(json.dumps(_finite_json(result), indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('chain_feasibility_pass', 'checks', 'native_failure')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--action', action='append', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--actor', choices=('a1', 'a2'), default='a1')
    parser.add_argument('--opponent-guard', default='center',
                        help='Explicit held test posture; does not replace the source center')
    args = parser.parse_args()
    run(args.source, args.output, args.action, repeats=args.repeats, actor=args.actor,
        opponent_guard=args.opponent_guard)
