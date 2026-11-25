"""Full-window native probes of an exact compact-bank build, no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .assembly import ROOT, _finite_json
from .candidate_graph import load_candidate_graph
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .a1_triangle_native import audit_recorded_triangle
from .a1_triangle_parry import compare


def run(bank, output, attack, *, defender=None, attack_start=.75, defence_start=2.9,
        attacker_slot='a1', minimum_duration=12.):
    bank=Path(bank).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve old native probes')
    cfg=json.loads(bank.read_text());graph=load_candidate_graph(bank).new_executor()
    if attack not in graph.actions or cfg['actions'][attack]['kind']!='attack':
        raise ValueError('Named attack from this bank required')
    if defender is not None and (defender not in graph.actions or cfg['actions'][defender]['kind']!='defence'):
        raise ValueError('Named defence from this bank required')
    if attacker_slot not in ('a1','a2'):raise ValueError('Explicit physical slot required')
    other='a2' if attacker_slot=='a1' else 'a1'
    ac=graph.actions[attack].curve
    duration=max(minimum_duration,attack_start+ac.duration_s+2.)
    if defender:duration=max(duration,defence_start+graph.actions[defender].curve.duration_s+2.)
    duration=math.ceil(duration*1000)/1000
    scoring=json.loads((ROOT/cfg['scoring_source']).read_text())
    metadata={'bank_sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),
              'bank_path':str(bank),'attack_action':attack,'defence_action':defender,
              'attacker_slot':attacker_slot,'not_policy_reaction':True,
              'admitted_to_training':False}
    output.mkdir(parents=True)
    summaries={}
    for name in (('baseline','defended') if defender else ('baseline',)):
        programs={attacker_slot:ScheduledReference(ac,attack_start),
                  other:hold_reference(graph.initial_q,attack_start,ac.duration_s)}
        if name=='defended':programs[other]=ScheduledReference(graph.actions[defender].curve,defence_start)
        summaries[name]=run_constrained_trial(programs,output/name,duration_s=duration,
                                               metadata=metadata,progress=print)
    audit=audit_recorded_triangle(output/'baseline',scoring,cfg['actions'][attack]['intended_region'],attacker_slot)
    (output/'baseline_audit.json').write_text(json.dumps(_finite_json(audit),indent=2)+'\n')
    if defender:
        result=compare(output/'baseline',output/'defended',scoring,
                       cfg['actions'][attack]['intended_region'],attacker_slot)
        (output/'comparison.json').write_text(json.dumps(_finite_json(result),indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('status','baseline_valid_threat','first_pre_hit_weapon_contact_s','attacker_entries_entire_window')}))
    else:
        print(json.dumps({'native':summaries['baseline']['native_execution_pass'],
                          'intended_region_first':audit['intended_region_first']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    p.add_argument('--attack',required=True);p.add_argument('--defence')
    p.add_argument('--attack-start',type=float,default=.75);p.add_argument('--defence-start',type=float,default=2.9)
    p.add_argument('--attacker-slot',choices=('a1','a2'),default='a1')
    a=p.parse_args();run(a.bank,a.output,a.attack,defender=a.defence,
        attack_start=a.attack_start,defence_start=a.defence_start,attacker_slot=a.attacker_slot)
