"""Observation-cued parry, measured-settle gate, then a continuous riposte.

Unlike a time-spliced composite, the riposte cannot start while the real arm
has not settled at the parry node. No clock or opponent action ID is supplied
to the policy. This remains a scripted diagnostic baseline, not learned RL.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_reactive import CorridorPolicy, PUBLIC_FIELDS, _executor, compare_feedback
from .feedback_policy import GraphAction, SkillGraphExecutor, TimedRequests, WaitPolicy
from .feedback_trial import run_feedback_trial
from .motion_spline import C2JointSpline


class SettleThenRiposte(CorridorPolicy):
    def __init__(self,corridor):
        super().__init__([dict(corridor,action='enter_parry')])
        self.entered=False;self.countered=False

    def choose(self,obs,available):
        if self.entered and not self.countered and 'riposte' in available:
            self.countered=True
            self.events.append({'event':'measured_settle_then_riposte',
                'own_q_rad':list(obs['own_q_rad']),'own_qvel_rad_s':list(obs['own_qvel_rad_s'])})
            return 'riposte'
        if not self.entered:
            action=super().choose(obs,available)
            if action:
                self.entered=True
            return action
        return None


def counter_executor(skill,ready):
    curve=skill['counter'];t=np.asarray(curve['time_s']);q=np.asarray(curve['qpos'])
    departure=skill['counter_departure_relative_s']
    index=np.flatnonzero(np.isclose(t,departure,atol=1e-10,rtol=0))
    if len(index)!=1 or index[0]<1:raise ValueError('An explicit held parry departure knot is required')
    index=int(index[0]);parry=q[index]
    if not np.array_equal(parry,q[1]):raise ValueError('Counter must depart the held parry pose')
    entry=C2JointSpline([0.,float(t[1])],[ready,parry])
    riposte=C2JointSpline(t[index:]-t[index],q[index:])
    actions={'enter_parry':GraphAction('enter_parry','ready','parry',entry),
             'riposte':GraphAction('riposte','parry','ready',riposte)}
    return SkillGraphExecutor({'ready':ready,'parry':parry},actions,'ready')


def run(library,output,*,attacker='a1',start_s=.76,time_scale=1.,duration_s=20.,baseline_from=None):
    library=Path(library).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Keep old experiments')
    cfg=json.loads(library.read_text());entry=cfg['cases']['upper']
    skill=next(s for s in cfg['skills'] if s['region']=='upper')
    if 'counter' not in skill:raise ValueError('A separately validated counter source is required')
    if attacker not in ('a1','a2'):raise ValueError('Physical slot required')
    defender='a2' if attacker=='a1' else 'a1'
    output.mkdir(parents=True)
    source=Path(__file__).read_bytes();(output/'controller_source_snapshot.py').write_bytes(source)
    metadata={'attacker_slot':attacker,'repeat_count':1,'scope':'observed parry and actual settled-state continuation',
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'controller_source_sha256':hashlib.sha256(source).hexdigest(),
        'opponent_future_information':False,'public_policy_fields':PUBLIC_FIELDS,
        'attack_start_requested_s':start_s,'attack_time_scale':time_scale}
    for label,active in [('baseline',False),('defended',True)]:
        if label=='baseline' and baseline_from:
            origin=Path(baseline_from).resolve()
            previous=json.loads((origin/'summary.json').read_text())
            if (not previous.get('evidence_complete') or previous['requested_duration_s']!=duration_s
                    or previous['metadata']['attacker_slot']!=attacker
                    or previous['metadata']['attack_start_requested_s']!=start_s
                    or previous['metadata']['attack_time_scale']!=time_scale):
                raise ValueError('Only an identical completed baseline may be reused')
            shutil.copytree(origin,output/label)
            continue
        executors={attacker:_executor(entry,time_scale=time_scale),
                   defender:counter_executor(skill,np.asarray(entry['ready_qpos']))}
        policies={attacker:TimedRequests([(round(start_s/.02),'attack')]),
                  defender:SettleThenRiposte(skill['corridor']) if active else WaitPolicy()}
        run_feedback_trial(executors,policies,output/label,duration_s=duration_s,metadata=metadata,progress=print)
    scoring=json.loads((ROOT/cfg['scoring']).read_text())
    result=compare_feedback(output/'baseline',output/'defended',scoring,'upper',attacker)
    summary=json.loads((output/'defended/summary.json').read_text())
    events=summary['executor_events'][defender]
    entry_done=[e for e in events if e['event']=='completed' and e['action']=='enter_parry']
    counter_start=[e for e in events if e['event']=='started' and e['action']=='riposte']
    counter_done=[e for e in events if e['event']=='completed' and e['action']=='riposte']
    own_events=[e for e in summary['policy_events'][defender] if e['event']=='measured_settle_then_riposte']
    after_settle=bool(len(entry_done)==len(counter_start)==len(counter_done)==len(own_events)==1
                     and entry_done[0]['time_s']<=counter_start[0]['time_s'])
    real_counters=[h for h in result['defender_entries_entire_window'] if counter_start and h['time_s']>=counter_start[0]['time_s']]
    result.update(counter_started_after_measured_settling=after_settle,
        counter_start_events=counter_start,counter_completion_events=counter_done,
        counter_entries_after_phase=real_counters,
        controller_source_unchanged=Path(__file__).read_bytes()==source,
        counterattack_feasibility_pass=bool(result['status']=='defence_with_counter_candidate'
            and after_settle and real_counters and Path(__file__).read_bytes()==source),
        scope='Observed known upper corridor, measured entry gate, finite counter; not trained policy or arbitrary opponent coverage')
    (output/'comparison.json').write_text(json.dumps(_finite_json(result),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counterattack_feasibility_pass','counter_start_events','counter_entries_after_phase')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',required=True);p.add_argument('--output',required=True)
    p.add_argument('--attacker-slot',choices=('a1','a2'),default='a1');p.add_argument('--start-s',type=float,default=.76)
    p.add_argument('--time-scale',type=float,default=1.);p.add_argument('--duration-s',type=float,default=20.)
    p.add_argument('--baseline-from')
    a=p.parse_args();run(a.library,a.output,attacker=a.attacker_slot,start_s=a.start_s,time_scale=a.time_scale,duration_s=a.duration_s,baseline_from=a.baseline_from)
