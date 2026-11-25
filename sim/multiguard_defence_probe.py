"""Matched, scripted full-window multi-guard defence feasibility probes.

The baseline holds the exact defence entry pose, not an unrelated center guard.
Both trials have identical attack, initial state, rules and duration.  Contact
before a baseline-valid hit plus absence of all later incoming entries is
required.  A passing pair is only evidence for these exact motions and timing;
it is not causal observation feedback, general defence, or training admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_parry import compare, triangle_defence_gate
from .constrained_trial import (ScheduledReference, hold_reference,
                                run_constrained_trial, validate_references)
from .joint_constraints import load_joint_constraints
from .multiguard_native import load_action


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_pair(report, attack_name, defence_name, *, defence_offset_s=0., attacker='a1'):
    """Construct exact matched references without running or resetting physics."""
    if attacker not in ('a1', 'a2'):
        raise ValueError('Explicit attacker slot a1 or a2 required')
    if type(defence_offset_s) not in (int, float) or not np.isfinite(defence_offset_s):
        raise ValueError('Finite numeric defence offset required')
    attack, attack_curve = load_action(report, attack_name)
    defence, defence_curve = load_action(report, defence_name)
    if attack.get('kind') != 'attack' or defence.get('kind') != 'defence':
        raise ValueError('Explicit attack and defence roles required')
    target = attack.get('target_region')
    if target not in ('left', 'right', 'upper'):
        raise ValueError('Attack needs one explicit intended target region')
    attack_start = .75
    defence_start = attack_start + float(defence_offset_s)
    if defence_start < .1-1e-12:
        raise ValueError('Defence start must leave at least .1 seconds initial observation')
    defence_start = max(.1, defence_start)  # only eliminate roundoff at the .1 boundary
    defender = 'a2' if attacker == 'a1' else 'a1'
    ap = ScheduledReference(attack_curve, attack_start)
    dp = ScheduledReference(defence_curve, defence_start)
    duration = float(np.ceil(max(6., ap.end_s+2., dp.end_s+2.)*1000.)/1000.)
    baseline = {attacker: ap, defender: hold_reference(defence_curve.knot_qpos[0],
                                                     defence_start, defence_curve.duration_s)}
    defended = {attacker: ap, defender: dp}
    constraints = load_joint_constraints()
    validate_references(baseline, constraints, duration)
    validate_references(defended, constraints, duration)
    return {'baseline': baseline, 'defended': defended, 'duration_s': duration,
            'attacker': attacker, 'defender': defender, 'target_region': target,
            'attack_name': attack_name, 'defence_name': defence_name,
            'attack_start_s': attack_start, 'defence_start_s': defence_start,
            'defence_offset_s': float(defence_offset_s)}


def _same_curve(a, b):
    return all(np.array_equal(getattr(a, name), getattr(b, name))
               for name in ('knot_time_s', 'knot_qpos', 'coefficients'))


def validate_matched_plan(plan):
    """Reject an altered baseline, rather than treating it as a held comparison."""
    a, d = plan['attacker'], plan['defender']
    base, trial = plan['baseline'], plan['defended']
    attack_equal = (base[a].start_s == trial[a].start_s and _same_curve(base[a].curve, trial[a].curve))
    hold = base[d].curve
    hold_equal = (base[d].start_s == trial[d].start_s
                  and np.all(hold.knot_qpos == trial[d].curve.knot_qpos[0])
                  and hold.duration_s == trial[d].curve.duration_s)
    checks = {'prescribed_attack_identical': bool(attack_equal),
              'baseline_holds_exact_defence_entry': bool(hold_equal),
              'identical_initial_reference': all(np.array_equal(base[s].curve.sample(0.),
                  trial[s].curve.sample(0.)) for s in ('a1', 'a2')),
              'complete_attack_plus_two_second_tail': plan['duration_s'] >= trial[a].end_s+2.-1e-9,
              'complete_defence_plus_two_second_tail': plan['duration_s'] >= trial[d].end_s+2.-1e-9}
    if not all(checks.values()):
        raise ValueError('Unmatched or truncated pair: ' + ', '.join(k for k, v in checks.items() if not v))
    return checks


def recorded_plan_checks(folder, programs):
    """Bind saved references/start clocks to the exact predeclared trial."""
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text())
    checks = {}
    with np.load(folder/'references.npz', allow_pickle=False) as references:
        for slot, program in programs.items():
            checks[slot+'_recorded_start_matches_plan'] = summary['program_start_s'][slot] == program.start_s
            checks[slot+'_recorded_curve_matches_plan'] = all(np.array_equal(
                references[slot+'_'+key], getattr(program.curve, key))
                for key in ('knot_time_s', 'knot_qpos', 'coefficients'))
    return checks


def run(source, output, *, attack_name, defence_name, defence_offset_s=0., attacker='a1'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier probes; select a fresh output directory')
    source_bytes = source.read_bytes()
    report = json.loads(source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    plan = plan_pair(report, attack_name, defence_name, defence_offset_s=defence_offset_s, attacker=attacker)
    plan_checks = validate_matched_plan(plan)
    scoring_path = ROOT/report['scoring'] if isinstance(report['scoring'], str) else None
    scoring_bytes = scoring_path.read_bytes() if scoring_path else json.dumps(report['scoring'], sort_keys=True).encode()
    scoring = json.loads(scoring_bytes)
    scoring_hash = hashlib.sha256(scoring_bytes).hexdigest()
    dependencies = (Path(__file__), ROOT/'sim/multiguard_native.py', ROOT/'sim/a1_triangle_parry.py',
                    ROOT/'sim/a1_triangle_native.py', ROOT/'sim/constrained_trial.py', ROOT/'sim/motion_spline.py')
    hashes = {str(p): _hash(p) for p in dependencies}
    output.mkdir(parents=True)
    (output/'source_report_snapshot.json').write_bytes(source_bytes)
    (output/'scoring_snapshot.json').write_bytes(scoring_bytes)
    metadata = {'attacker_slot': attacker, 'attack_name': attack_name, 'defence_name': defence_name,
        'attack_start_s': plan['attack_start_s'], 'defence_start_s': plan['defence_start_s'],
        'defence_offset_s': plan['defence_offset_s'], 'source_report_sha256': source_hash,
        'scoring_sha256': scoring_hash, 'probe_source_sha256': hashes,
        'scripted_timing_not_policy_reaction': True, 'admitted_to_training': False}
    for label in ('baseline', 'defended'):
        run_constrained_trial(plan[label], output/label, duration_s=plan['duration_s'],
                              metadata=dict(metadata, condition=label), progress=print)
    comparison = compare(output/'baseline', output/'defended', scoring, plan['target_region'], attacker)
    checks = dict(comparison['matching_checks'], **plan_checks)
    checks['source_report_unchanged'] = _hash(source) == source_hash
    checks['scoring_source_unchanged'] = scoring_path is None or _hash(scoring_path) == scoring_hash
    checks['probe_dependencies_unchanged'] = all(_hash(p) == digest for p, digest in hashes.items())
    for label in ('baseline', 'defended'):
        checks.update({label+'_'+k: v for k, v in recorded_plan_checks(output/label, plan[label]).items()})
    summaries = [json.loads((output/label/'summary.json').read_text()) for label in ('baseline', 'defended')]
    for label, summary in zip(('baseline', 'defended'), summaries):
        checks[label+'_both_slots_finally_settled'] = all(
            summary.get('final_settled', {}).get(slot) is True for slot in ('a1', 'a2'))
    baseline_hits = [event for event in comparison['baseline_triangle_audit']['all_entries']
                     if event['agent'] == attacker]
    first_hit = min((event['time_s'] for event in baseline_hits), default=None)
    checks['baseline_hit_within_attack_curve'] = bool(first_hit is not None
        and plan['attack_start_s'] <= first_hit <= plan['defended'][attacker].end_s)
    contacts = json.loads((output/'defended/contacts.json').read_text())
    gate = triangle_defence_gate(summaries[0], comparison['baseline_triangle_audit'],
        summaries[1], comparison['defended_triangle_audit'], contacts,
        matching_checks=checks, attacker=attacker)
    result = dict(comparison, **gate, attack_name=attack_name, defence_name=defence_name,
        source_report_sha256=source_hash, scoring_source_sha256=scoring_hash,
        probe_source_sha256=hashes, duration_s=plan['duration_s'],
        attack_start_s=plan['attack_start_s'], defence_start_s=plan['defence_start_s'],
        defence_offset_s=plan['defence_offset_s'], scripted_timing_not_policy_reaction=True,
        validation_scope='Exact paired full-window feasibility only; no general defence, policy reaction, or training admission.')
    (output/'comparison.json').write_text(json.dumps(_finite_json(result), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(_finite_json(result), ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attack', required=True)
    parser.add_argument('--defence', required=True)
    parser.add_argument('--defence-offset-s', type=float, default=0.)
    parser.add_argument('--attacker', choices=('a1', 'a2'), default='a1')
    args = parser.parse_args()
    run(args.source, args.output, attack_name=args.attack, defence_name=args.defence,
        defence_offset_s=args.defence_offset_s, attacker=args.attacker)
