"""Matched whole-window physical defence gates for constrained trials."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def defence_evidence_gate(baseline, defended, contact_intervals, *, attacker, matched):
    """A label, first touch or a missed attack alone cannot certify a block."""
    baseline_hits = [h for h in baseline['all_target_entries'] if h['agent'] == attacker]
    baseline_valid = bool(baseline.get('native_execution_pass') and baseline.get('source_unchanged')
                          and baseline_hits)
    hit_time = min((h['time_s'] for h in baseline_hits), default=None)
    late_hits = [h for h in defended['all_target_entries'] if h['agent'] == attacker]
    impulse, peak, contact_times = 0., 0., []
    if hit_time is not None:
        for interval in contact_intervals:
            t0, t1 = interval['interval_start_s'], interval['interval_end_s']
            overlap = max(0., min(t1, hit_time)-t0)
            if overlap <= 0: continue
            for c in interval['contacts']:
                if c['category'] in ('cross_blade_blade', 'cross_blade_guard'):
                    force = max(0., float(c['normal_force_n']))
                    impulse += force*overlap
                    peak = max(peak, force)
                    if force > 0: contact_times.append(t1)
    cfg = baseline['runtime_config']
    meaningful_contact = (peak >= cfg['minimum_blade_force_n']
                          and impulse >= cfg['minimum_blade_impulse_ns'])
    complete = bool(defended.get('native_execution_pass') and defended.get('source_unchanged'))
    passed = bool(matched and baseline_valid and complete and not late_hits and meaningful_contact)
    return {'matched_experiment': bool(matched), 'baseline_valid_attack': baseline_valid,
        'baseline_first_hit_time_s': hit_time, 'defended_full_execution_pass': complete,
        'attacker_entries_entire_window': late_hits,
        'legal_weapon_impulse_before_baseline_hit_ns': impulse,
        'peak_legal_weapon_force_before_baseline_hit_n': peak,
        'first_legal_force_time_s': min(contact_times, default=None),
        'meaningful_pre_hit_weapon_contact': meaningful_contact,
        'successful_physical_defence': passed, 'admitted_to_training': False,
        'scope': 'This matched attack, entry state, start time and full window only; no direction-general or policy claim'}


def compare_trial_directories(baseline_folder, defended_folder, *, attacker='a1'):
    baseline_folder, defended_folder = Path(baseline_folder), Path(defended_folder)
    baseline = json.loads((baseline_folder/'summary.json').read_text())
    defended = json.loads((defended_folder/'summary.json').read_text())
    checks = {name: baseline.get(name) == defended.get(name) for name in
        ('runtime_config', 'joint_constraints', 'source_sha256', 'requested_duration_s', 'model_source_sha256')}
    checks['attacker_start'] = baseline['program_start_s'][attacker] == defended['program_start_s'][attacker]
    with np.load(baseline_folder/'references.npz', allow_pickle=False) as first, \
         np.load(defended_folder/'references.npz', allow_pickle=False) as second:
        checks['attacker_reference'] = all(np.array_equal(first[attacker+'_'+key], second[attacker+'_'+key])
            for key in ('knot_time_s', 'knot_qpos', 'coefficients'))
    with np.load(baseline_folder/'trace.npz', allow_pickle=False) as first, \
         np.load(defended_folder/'trace.npz', allow_pickle=False) as second:
        checks['initial_qpos'] = np.array_equal(first['qpos'][0], second['qpos'][0])
        checks['initial_qvel'] = np.array_equal(first['qvel'][0], second['qvel'][0])
    contacts = json.loads((defended_folder/'contacts.json').read_text())
    result = defence_evidence_gate(baseline, defended, contacts, attacker=attacker, matched=all(checks.values()))
    result['matching_checks'] = checks
    return result
