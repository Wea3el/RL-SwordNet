"""Observation-cued diagnostic for an already qualified short defence.

Reuses the shared physical observer, executor and judge. A movement-onset cue
is deliberately simple and noise-free; it is not fencing intent recognition,
a learned policy, or proof of two-sided skill selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import _finite_json
from .a1_triangle_reactive import compare_feedback, PUBLIC_FIELDS
from .feedback_policy import GraphAction, SkillGraphExecutor, TimedRequests, ThreatParry, WaitPolicy
from .feedback_trial import run_feedback_trial
from .multiguard_native import load_action


def make_executor(report, action_name, runtime_name):
    action, curve = load_action(report, action_name)
    nodes = {key: np.asarray(report['guards'][key]['qpos_rad'])
             for key in (action['start_guard'], action['end_guard'])}
    graph_action = GraphAction(runtime_name, action['start_guard'], action['end_guard'], curve)
    return SkillGraphExecutor(nodes, {runtime_name: graph_action}, action['start_guard'])


def cue_checks(summary, observations, attacker, *, stationary=False):
    defender = 'a2' if attacker == 'a1' else 'a1'
    starts = [e for e in summary['executor_events'][defender] if e['event'] == 'started']
    requests = [row for row in observations if row['requests'][defender] == 'parry']
    cues = [e for e in summary['policy_events'][defender] if e['event'] == 'observed_threat']
    fields_ok = all(set(row['physical'][slot]) == set(PUBLIC_FIELDS)
                    for row in observations for slot in ('a1', 'a2'))
    checks = {'shared_physical_fields_only': fields_ok,
              'requests_only_when_available': all(row['requests'][defender] is None
                  or row['requests'][defender] in row['available'][defender] for row in observations)}
    if stationary:
        checks['no_stationary_false_trigger'] = not starts and not requests and not cues
        return checks
    checks['one_observation_cued_defence'] = len(starts) == len(requests) == len(cues) == 1
    checks['cue_matches_actual_shared_snapshot'] = bool(len(requests) == len(cues) == 1 and
        np.array_equal(requests[0]['physical'][defender][PUBLIC_FIELDS[2]], cues[0]['observed_tip_m']) and
        np.array_equal(requests[0]['physical'][defender][PUBLIC_FIELDS[4]], cues[0]['observed_tip_velocity_m_s']))
    attack_starts = [e for e in summary['executor_events'][attacker] if e['event'] == 'started']
    checks['no_defence_before_actual_attack'] = bool(len(starts) == len(attack_starts) == 1
        and starts[0]['time_s'] >= attack_starts[0]['time_s'])
    checks['defence_completed'] = len([e for e in summary['executor_events'][defender]
                                      if e['event'] == 'completed']) == 1
    return checks


def run(source, qualified_pair, output, *, attacker='a1', start_s=.76, stationary=False):
    source, qualified_pair, output = map(lambda p: Path(p).resolve(), (source, qualified_pair, output))
    if output.exists():
        raise FileExistsError('Preserve previous observed-cue experiments')
    if attacker not in ('a1', 'a2') or not .2 <= start_s <= 3.:
        raise ValueError('Explicit slot and bounded attack start required')
    source_bytes = source.read_bytes(); source_hash = hashlib.sha256(source_bytes).hexdigest()
    gate_bytes = qualified_pair.read_bytes(); gate = json.loads(gate_bytes)
    if (gate.get('status') != 'qualified' or gate.get('successful_physical_defence') is not True
            or gate.get('source_report_sha256') != source_hash
            or not gate.get('matching_checks') or not all(gate['matching_checks'].values())):
        raise ValueError('An exactly matched qualified source pair is required')
    report = json.loads(source_bytes)
    attack_name, defence_name = gate['attack_name'], gate['defence_name']
    attack, ac = load_action(report, attack_name)
    defence, dc = load_action(report, defence_name)
    if defence['start_guard'] != defence['end_guard']:
        raise ValueError('This diagnostic requires a closed defence')
    defender = 'a2' if attacker == 'a1' else 'a1'
    duration = 3. if stationary else float(np.ceil((start_s + max(ac.duration_s, dc.duration_s) + 4.)*1000.)/1000.)
    scoring = report['scoring']
    if not isinstance(scoring, dict):
        raise ValueError('Use the exact embedded source scoring snapshot')
    output.mkdir(parents=True)
    (output/'source_report_snapshot.json').write_bytes(source_bytes)
    (output/'qualified_pair_snapshot.json').write_bytes(gate_bytes)
    (output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    metadata = {'attacker_slot': attacker, 'source_report_sha256': source_hash,
        'scope': 'Shared current-observation movement cue; no future action/phase/time available to defender.',
        'stationary_negative_control': stationary, 'repeat_count': 1,
        'cue': {'displacement_m': .001, 'speed_m_s': .01, 'consecutive_samples': 2,
                'require_approach': False, 'limitation': 'Detects movement, not intent or general fencing threats.'},
        'admitted_to_training': False}
    labels = ('stationary',) if stationary else ('baseline', 'defended')
    summaries = {}
    for label in labels:
        executors = {attacker: make_executor(report, attack_name, 'attack'),
                     defender: make_executor(report, defence_name, 'parry')}
        policies = {attacker: WaitPolicy() if stationary else TimedRequests([(round(start_s/.02), 'attack')]),
                    defender: ThreatParry('parry', displacement_m=.001, approaching_speed_m_s=.01,
                                           consecutive_samples=2, require_approach=False)
                    if label != 'baseline' else WaitPolicy()}
        summaries[label] = run_feedback_trial(executors, policies, output/label,
            duration_s=duration, metadata=dict(metadata, condition=label), progress=print)
    last_label = labels[-1]
    observations = json.loads((output/last_label/'observations.json').read_text())
    checks = cue_checks(summaries[last_label], observations, attacker, stationary=stationary)
    checks.update(source_unchanged=source.read_bytes() == source_bytes,
                  qualified_pair_unchanged=qualified_pair.read_bytes() == gate_bytes,
                  native_execution_pass=bool(summaries[last_label]['native_execution_pass']))
    if stationary:
        result = {'status': 'passed_stationary_negative_control' if all(checks.values()) else 'failed',
                  'successful_physical_defence': False}
    else:
        result = compare_feedback(output/'baseline', output/'defended', scoring, attack['target_region'], attacker)
        result['observation_cued_defence_pass'] = bool(result['status'] == 'qualified' and all(checks.values()))
    result.update(cue_checks=checks, attacker_slot=attacker, source_report_sha256=source_hash,
                  source_gate_sha256=hashlib.sha256(gate_bytes).hexdigest(), admitted_to_training=False,
                  scope='This one observed movement cue and one defence only; not bilateral coverage, intent recognition or self-play.')
    (output/'comparison.json').write_text(json.dumps(_finite_json(result), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'cue_checks')}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--qualified-pair', required=True)
    p.add_argument('--output', required=True); p.add_argument('--attacker', choices=('a1', 'a2'), default='a1')
    p.add_argument('--start', type=float, default=.76); p.add_argument('--stationary', action='store_true')
    a = p.parse_args()
    run(a.source, a.qualified_pair, a.output, attacker=a.attacker, start_s=a.start, stationary=a.stationary)
