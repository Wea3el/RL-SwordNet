"""Short left parry and physical-slot mirror: immutable native diagnostics.

This does not admit a skill to training.  It preserves source attack geometry,
physics and the full all-region scoring gate.  Slot exchange is a rigid base
rotation in the model, never joint-angle sign negation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_parry import compare
from .candidate_graph import load_candidate_graph
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .motion_spline import C2JointSpline

SOURCE = ROOT / 'outputs/a1_left_defence_completion/v2/swipe_negative'
BASELINE = ROOT / 'outputs/a1_left_defence_completion/v2/baseline'
DEFAULT_OUTPUT = ROOT / 'outputs/compact_bank_v1_defence'


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2) + '\n')


def programs(release, start):
    source_summary = json.loads((SOURCE / 'summary.json').read_text())
    with np.load(SOURCE / 'references.npz', allow_pickle=False) as refs:
        attack = ScheduledReference(C2JointSpline(refs['a1_knot_time_s'],
                                                  refs['a1_knot_qpos']),
                                    source_summary['program_start_s']['a1'])
        original_q = refs['a2_knot_qpos'].copy()
    if not release > start + 1.3:
        raise ValueError('Keep the complete approach and side swipe before recovery')
    defence = ScheduledReference(C2JointSpline(
        [0., .3, 1.3, release - start, release - start + 1.],
        original_q), start)
    return attack, defence, original_q[0]


def run(output, release=4.2, start=2.34, attacker='a1'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Earlier native evidence is immutable')
    output.mkdir(parents=True)
    attack, defence, ready = programs(release, start)
    defender = 'a2' if attacker == 'a1' else 'a1'
    metadata = {'attacker_slot': attacker, 'candidate_only': True,
                'not_policy_reaction': True,
                'short_hold_probe': {'start_s': start, 'release_s': release,
                                     'recovery_end_s': defence.end_s},
                'source_native': str(SOURCE.relative_to(ROOT)),
                'source_reference_sha256': hashlib.sha256((SOURCE / 'references.npz').read_bytes()).hexdigest(),
                'mirror_method': 'exchange physical slots without changing local joint values'}
    baseline = BASELINE
    if attacker != 'a1':
        baseline = output / 'baseline'
        run_constrained_trial({attacker: attack, defender: hold_reference(ready, attack.start_s, attack.curve.duration_s)},
                              baseline, duration_s=12., metadata=metadata, progress=print)
    run_constrained_trial({attacker: attack, defender: defence}, output / 'defended',
                          duration_s=12., metadata=metadata, progress=print)
    scoring = json.loads((ROOT / 'config/scoring_triangle_v1_candidate.json').read_text())
    result = compare(baseline, output / 'defended', scoring, target='left', attacker=attacker)
    result.update(baseline_native=str(baseline.relative_to(ROOT)),
                  defended_native=str((output / 'defended').relative_to(ROOT)),
                  short_hold_probe=metadata['short_hold_probe'])
    write(output / 'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'baseline_first_triangle_hit_s',
          'first_pre_hit_weapon_contact_s', 'attacker_entries_entire_window', 'short_hold_probe')}, indent=2), flush=True)
    return result


def upper_closed_probe(output, start):
    """Change only upper-defence timing; reuse the exact complete attack baseline."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Earlier native evidence is immutable')
    bank = ROOT / 'outputs/compact_bank_v1/build_03/bank.json'
    baseline = ROOT / 'outputs/compact_bank_v1/upper_closed_pair/baseline'
    cfg = json.loads(bank.read_text())
    graph = load_candidate_graph(bank).new_executor()
    base = json.loads((baseline / 'summary.json').read_text())
    ac = graph.actions['attack_upper'].curve
    dc = graph.actions['defence_upper'].curve
    if not base['native_execution_pass'] or not base['source_unchanged']:
        raise ValueError('The reusable baseline must be complete and clean')
    if base['requested_duration_s'] != 12. or base['program_start_s']['a1'] != .75:
        raise ValueError('Unexpected baseline schedule')
    for relative, expected in base['source_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Baseline implementation changed: ' + relative)
    with np.load(baseline / 'references.npz', allow_pickle=False) as refs:
        for suffix, value in [('knot_time_s', ac.knot_time_s),
                              ('knot_qpos', ac.knot_qpos),
                              ('coefficients', ac.coefficients)]:
            if not np.array_equal(refs['a1_' + suffix], value):
                raise ValueError('Reused attack differs from current bank')
        if not np.array_equal(refs['a2_knot_qpos'][0], graph.initial_q):
            raise ValueError('Reused baseline has a different ready state')
    output.mkdir(parents=True)
    metadata = {'attacker_slot': 'a1', 'not_policy_reaction': True,
                'bank_path': str(bank.relative_to(ROOT)),
                'bank_sha256': hashlib.sha256(bank.read_bytes()).hexdigest(),
                'attack_action': 'attack_upper', 'defence_action': 'defence_upper',
                'defence_start_s': start, 'reused_exact_baseline': str(baseline.relative_to(ROOT)),
                'candidate_only': True}
    run_constrained_trial({'a1': ScheduledReference(ac, .75),
                           'a2': ScheduledReference(dc, start)}, output / 'defended',
                          duration_s=12., metadata=metadata, progress=print)
    scoring = json.loads((ROOT / cfg['scoring_source']).read_text())
    result = compare(baseline, output / 'defended', scoring, target='upper')
    result.update(baseline_native=str(baseline.relative_to(ROOT)),
                  defended_native=str((output / 'defended').relative_to(ROOT)),
                  defence_start_s=start)
    write(output / 'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'baseline_first_triangle_hit_s',
          'first_pre_hit_weapon_contact_s', 'attacker_entries_entire_window')}, indent=2), flush=True)
    return result


def render_new_ready(source, output):
    """Preview a completed new-ready attack from actual recorded states only."""
    from .constrained_replay import render_trial
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous rendering evidence')
    result = json.loads((source / 'result.json').read_text())
    if not result.get('attack_probe_pass') or not result.get('native_execution_pass'):
        raise ValueError('Only completed passing attack probes use this preview')
    region = result['metadata']['region']
    output.mkdir(parents=True)
    audit = output / 'triangle_audit.json'
    write(audit, result['triangle_audit'])
    return render_trial(source / 'native', output / 'replay', fps=30,
                        label=f'New ready / {region} attack probe / not defence or training',
                        triangle_audit=audit, progress=print)


def render_matched_defence(source, output):
    """Render a qualified exact native parry, never imply policy reaction."""
    from .constrained_replay import render_trial
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous rendering evidence')
    comparison = json.loads((source / 'comparison.json').read_text())
    if comparison.get('status') != 'qualified':
        raise ValueError('This rendering label requires a qualified matched test')
    audit_data = comparison['defended_triangle_audit']
    if not comparison.get('all_regions_checked') or not comparison.get('numerical_and_full_execution_clean'):
        raise ValueError('Complete clean full-window evidence required')
    output.mkdir(parents=True)
    audit = output / 'triangle_audit.json'
    write(audit, audit_data)
    region = audit_data['target_region'].capitalize()
    return render_trial(source / 'native', output / 'replay', fps=30,
                        label=f'{region} matched defence / full-window native test / not policy or training',
                        triangle_audit=audit, progress=print)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT / 'short_left')
    parser.add_argument('--release', type=float, default=4.2)
    parser.add_argument('--start', type=float, default=2.34)
    parser.add_argument('--attacker', choices=['a1', 'a2'], default='a1')
    parser.add_argument('--case', choices=['left', 'upper', 'preview', 'defence-preview'], default='left')
    parser.add_argument('--preview-source', type=Path)
    args = parser.parse_args()
    if args.case in ('preview', 'defence-preview'):
        if args.preview_source is None:
            parser.error('--preview-source required for preview')
        (render_new_ready if args.case == 'preview' else render_matched_defence)(args.preview_source, args.output)
    elif args.case == 'upper':
        upper_closed_probe(args.output, args.start)
    else:
        run(args.output, args.release, args.start, args.attacker)


if __name__ == '__main__':
    main()
