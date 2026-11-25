"""Explicit second timing version for both short side-attack references.

Only source reference clocks are multiplied by 1.5. Model, controls, contact
thresholds and score geometry are unchanged. This does not constrain A2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .motion_spline import C2JointSpline


def baseline(region, source, output):
    source, output = Path(source), Path(output)
    if region not in ('left', 'right') or output.exists():
        raise ValueError('Explicit side and new output required')
    old = json.loads((source / 'summary.json').read_text())
    if not old['native_execution_pass']:
        raise ValueError('Source fast-timing baseline must have completed')
    with np.load(source / 'references.npz') as arrays:
        attack = C2JointSpline(arrays['a1_knot_time_s'] * 1.5, arrays['a1_knot_qpos'])
        ready = arrays['a2_knot_qpos'][0]
    output.mkdir(parents=True)
    summary = run_constrained_trial({'a1': ScheduledReference(attack, .75),
                                    'a2': hold_reference(ready, .75, attack.duration_s)},
        output / 'native', duration_s=8., progress=print,
        metadata={'attacker_slot': 'a1', 'region': region,
            'timing_version': 'side_attack_timing_v2', 'both_side_scale': 1.5,
            'source_native': str(source),
            'source_references_sha256': hashlib.sha256((source / 'references.npz').read_bytes()).hexdigest(),
            'a2_action_space_or_speed_restricted': False, 'model_or_score_changed': False,
            'candidate_only': True})
    scoring = json.loads((ROOT / 'config/scoring_triangle_v1_candidate.json').read_text())
    audit = audit_recorded_triangle(output / 'native', scoring, region)
    result = {'version': 'side_attack_timing_v2', 'scale': 1.5,
        'attack_probe_pass': audit['reference_candidate_pass'], 'triangle_audit': audit,
        'native_execution_pass': summary['native_execution_pass'], 'admitted_to_training': False}
    (output / 'result.json').write_text(json.dumps(_finite_json(result), indent=2) + '\n')
    print(json.dumps({'region': region, 'pass': result['attack_probe_pass'],
        'entries': [(e['agent'], e['region'], e['time_s']) for e in audit['all_entries']]}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region', choices=('left', 'right'), required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    baseline(a.region, a.source, a.output)
