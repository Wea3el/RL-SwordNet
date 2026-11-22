"""Post-hoc per-attack evidence for continuous A1 defence diagnostics.

No policy, scoring or runtime changes. Request IDs/times define evaluation
windows ONLY; they must never be fed back to a defender. Every specified
attack needs its own hitting Guard control and pre-hit physical interception.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .a1_validation import execution_gate
from .skill_validation import V2_CONFIG, compare_defence, save_json


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, str)) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def _actions(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'Explicit nonempty {name} required before evaluating a trial')
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f'{name} must contain nonzero integer skill IDs')
    return list(values)


def sequence_defence_gate(baseline, defended, attacker, expected_attack_actions,
                          expected_defence_actions, cfg=None):
    """Require matched whole-window validity plus a real block of EACH attack.

    Caller supplies fixed expected actions; actual observed counts never set
    their own acceptance target. Existing execution gates are recomputed and
    cached booleans are ignored. No input trial or historical score is edited.
    """
    if attacker not in {'a1', 'a2'}:
        raise ValueError('Either physical slot may be the attacker')
    attack_actions = _actions(expected_attack_actions, 'attack actions')
    defence_actions = _actions(expected_defence_actions, 'defence actions')
    defender = 'a2' if attacker == 'a1' else 'a1'
    required = {'physics_timestep_s', 'rules_version', 'minimum_blade_force_n',
                'minimum_blade_impulse_ns'}
    if any(not required.issubset(r['runtime_config']) for r in (baseline, defended)):
        raise ValueError('Native timestep and acceptance thresholds must be recorded, not inferred')
    runtime = dict(V2_CONFIG, **baseline['runtime_config'])
    if cfg is not None and any(k not in runtime or v != runtime[k] for k, v in cfg.items()):
        raise ValueError('Post-hoc callers cannot override the frozen native settings')
    _number(runtime['minimum_blade_force_n'], 'recorded contact force threshold', positive=True)
    _number(runtime['minimum_blade_impulse_ns'], 'recorded impulse threshold', positive=True)
    dt = _number(runtime['physics_timestep_s'], 'recorded physics timestep', positive=True)
    controllers = [r.get('controller_implementation_sha256') for r in (baseline, defended)]
    same_controllers = bool(all(isinstance(c, dict) and c for c in controllers)
                            and controllers[0] == controllers[1])
    if any(r['runtime_config'].get('rules_version') != 'whole_blade_first_touch_v2'
           for r in (baseline, defended)):
        raise ValueError('Explicit ungated whole-blade v2 evidence required')
    base_execution = execution_gate(baseline, {attacker: attack_actions, defender: []})
    defended_execution = execution_gate(defended,
        {attacker: attack_actions, defender: defence_actions})
    physical = compare_defence(baseline, defended, attacker, cfg=runtime)
    requests = baseline['action_events'][attacker]
    defended_requests = defended['action_events'][attacker]
    starts = [_number(e['time_s'], 'attack request time') for e in requests]
    observed_starts = [_number(e['time_s'], 'defended attack request time')
                       for e in defended_requests]
    duration = _number(baseline['requested_duration_s'], 'full window', positive=True)
    ordered = bool(starts and all(a < b for a, b in zip(starts, starts[1:]))
                   and starts[-1] < duration)
    matching_requests = bool(ordered and len(starts) == len(observed_starts)
        and all(abs(a-b) <= 1e-9 for a, b in zip(starts, observed_starts)))
    samples = defended.get('blade_contact_impulse_samples')
    evidence_available = samples is not None
    parsed = []
    for sample in samples or []:
        time = _number(sample['time_s'], 'contact time')
        interval = _number(sample['interval_s'], 'contact interval', positive=True)
        force = _number(sample['normal_force_n'], 'contact force')
        tolerance = max(1e-9, dt*1e-6)
        if interval > dt+tolerance:
            raise ValueError('A contact sample cannot span more than its frozen native timestep')
        if parsed and (time <= parsed[-1][0]
                       or time < parsed[-1][0]+parsed[-1][1]-tolerance):
            raise ValueError('Contact intervals must be strictly ordered and nonoverlapping')
        parsed.append((time, interval, force))
    hits = [h for h in baseline['hits'] if h['agent'] == attacker]
    windows = []
    if ordered:
        for index, start in enumerate(starts):
            end = starts[index+1] if index+1 < len(starts) else duration
            entries = [_number(h['time_s'], 'baseline hit time') for h in hits
                       if start <= h['time_s'] < end]
            cutoff = min(entries) if entries else None
            impulse, contacts = 0., []
            if cutoff is not None:
                for time, interval, force in parsed:
                    overlap = max(0., min(time+interval, cutoff)-max(time, start))
                    impulse += force*overlap
                    if (start <= time < cutoff
                            and force > runtime['minimum_blade_force_n']):
                        contacts.append(time)
            passed = bool(cutoff is not None and evidence_available and contacts
                and impulse >= runtime['minimum_blade_impulse_ns'])
            windows.append(dict(index=index, evaluation_only_action_id=requests[index]['action_id'],
                request_s=start, next_request_or_window_end_s=end,
                baseline_first_entry_s=cutoff,
                first_above_threshold_prehit_contact_s=min(contacts) if contacts else None,
                clipped_prehit_blade_impulse_ns=impulse,
                own_prehit_interception_evidence_pass=passed))
    execution = bool(base_execution['native_execution_pass']
                     and defended_execution['native_execution_pass'])
    each_attack = bool(len(windows) == len(attack_actions)
                       and all(w['own_prehit_interception_evidence_pass'] for w in windows))
    success = bool(physical['successful_physical_defence'] and matching_requests and same_controllers
                   and execution and each_attack)
    return dict(successful_continuous_physical_defence=success,
        matched_whole_window_physical_comparison=physical,
        baseline_fixed_expected_execution=base_execution,
        defended_fixed_expected_execution=defended_execution,
        matched_ordered_attack_request_times=matching_requests,
        matched_actual_controller_implementations=same_controllers,
        contact_interval_evidence_available=evidence_available,
        every_expected_attack_has_own_prehit_evidence=each_attack,
        attack_windows=windows,
        scope='Post-hoc fixed-sequence physical evidence only; no policy inputs, scoring changes or selfplay',
        admitted_to_training=False, may_start_RL=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--defended', type=Path, required=True)
    parser.add_argument('--attacker', choices=('a1', 'a2'), required=True)
    parser.add_argument('--attack-actions', type=int, nargs='+', required=True)
    parser.add_argument('--defence-actions', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = sequence_defence_gate(json.loads(args.baseline.read_text()),
        json.loads(args.defended.read_text()), args.attacker,
        args.attack_actions, args.defence_actions)
    result['posthoc_source_sha256'] = {str(path.resolve()):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.baseline, args.defended, Path(__file__),
            Path(__file__).with_name('a1_validation.py'),
            Path(__file__).with_name('skill_validation.py'))}
    if args.output:
        if args.output.exists():
            raise FileExistsError('Do not overwrite historical evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
