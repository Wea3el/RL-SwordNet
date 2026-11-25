"""Author a reverse cut on the existing, explicitly connected joint branch.

Reversing a continuous rising-cut reference supplies a descending-cut
candidate, not an inherited hit/defence certificate. Actual entry direction,
force-limited execution, endpoint settling and paired coverage must be retested.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .a1_dynamics_screen import screen_curve
from .assembly import _finite_json
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


def reverse_curve(curve):
    # Preserve the original clock even for a reference whose first knot is
    # nonzero.  Only its traversal direction changes; no joint wrapping or
    # per-knot clipping is permitted.
    return C2JointSpline(curve.knot_time_s[0]+curve.knot_time_s[-1]-curve.knot_time_s[::-1],
                        curve.knot_qpos[::-1])


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve prior banks')
    raw = source.read_bytes()
    bank = json.loads(raw)
    original = bank['actions']['cut_up_left_to_right']
    curve = reverse_curve(C2JointSpline(original['knot_time_s'], original['knot_qpos']))
    model = build_limited_floor().model
    trials = []
    chosen = None
    for scale in (1.,1.1,1.25,1.5):
        c = C2JointSpline(curve.knot_time_s*scale, curve.knot_qpos)
        check = screen_curve(model,c,.01)
        trials.append({'scale':scale,'ideal_demand':check})
        if (check['ideal_demand_within_caps_at_samples'] and
                check['worst_utilization'] <= .98 and chosen is None):
            chosen=c
    if chosen is None:
        raise ValueError('Reverse reference needs further dynamic repair')
    row = copy.deepcopy(original)
    row.update(family='cut_down_right_to_left', start_guard=original['end_guard'],
               end_guard=original['start_guard'], knot_time_s=chosen.knot_time_s.tolist(),
               knot_qpos=chosen.knot_qpos.tolist(), native_dynamics_tested=False,
               admitted_to_training=False, prior_certificate_inherited=False)
    row['direction_intent_attacker_frame']=(-np.array(original['direction_intent_attacker_frame'])).tolist()
    row['reverse_reference_authoring']={'source_action':'cut_up_left_to_right',
        'source_bank_sha256':hashlib.sha256(raw).hexdigest(),'candidate_only':True,
        'reason':'Use the current connected high-to-low branch instead of disconnected legacy endpoints.'}
    row.pop('audit',None)
    bank['actions']['cut_down_right_to_left']=row
    bank['reverse_attack_revision']={'source_sha256':hashlib.sha256(raw).hexdigest(),
        'replaced_action':'cut_down_right_to_left','ideal_trials':trials,
        'authoring_utilization_margin':.98,
        'old_candidate_retained_in_source':True,'native_tested':False}
    bank['admitted_to_training']=False
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(_finite_json(bank),ensure_ascii=False,indent=2)+'\n')
    return bank


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.source,a.output)
