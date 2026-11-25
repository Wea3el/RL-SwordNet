"""Causal, geometry-cued candidate skills under the shared native controller.

This is a diagnostic baseline, not a learned policy. A corridor is calibrated
from current physical sword observations at a validated interception entry;
it does not supply the opponent's action name, phase, future or game clock.
Passing known corridors does not certify unseen A2 movements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle
from .a1_triangle_parry import triangle_defence_gate
from .feedback_policy import GraphAction, SkillGraphExecutor, TimedRequests, WaitPolicy
from .feedback_trial import physical_observation, run_feedback_trial
from .motion_spline import C2JointSpline


PUBLIC_FIELDS = ('own_q_rad', 'own_qvel_rad_s',
    'opponent_tip_minus_own_target_local_m',
    'opponent_grip_minus_own_target_local_m', 'opponent_tip_velocity_local_m_s')


def corridor_observation(native_folder, at_s, observer='a2'):
    """Calibration only: read a saved actual state, never during policy choice."""
    folder=Path(native_folder)
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    data=mujoco.MjData(model)
    agents={}
    for slot in ('a1','a2'):
        ids=[model.joint(slot+'_'+n).id for n in ('J0','J1','J2','J3','J4','J5','J6')]
        agents[slot]={'qpos_ids':model.jnt_qposadr[ids], 'dof_ids':model.jnt_dofadr[ids],
            'sites':{n:model.site(slot+'_'+n).id for n in ('shinai_tip','shinai_grip')}}
    with np.load(folder/'trace.npz',allow_pickle=False) as trace:
        index=int(np.argmin(abs(trace['time_s']-at_s)))
        data.qpos[:]=trace['qpos'][index];data.qvel[:]=trace['qvel'][index]
        actual_s=float(trace['time_s'][index])
    mujoco.mj_forward(model,data)
    # The common observation contract is relative to the fixed reference at
    # z=1.05, not the active scoring region. No scoring geometry is changed.
    summary=json.loads((folder/'summary.json').read_text())
    spacing=summary['runtime_config']['base_distance_m']
    scene=SimpleNamespace(model=model,data=data,agents=agents,config={'virtual_targets':{
        'a1':{'center_world_m':[.32,0.,1.05]},
        'a2':{'center_world_m':[spacing-.32,0.,1.05]}}})
    obs=physical_observation(scene,observer)
    return {'sampled_native_time_s':actual_s,
        'physical':{k:v.tolist() for k,v in obs.items()},
        'source_sha256':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest()
                         for n in ('scene.xml','trace.npz','summary.json')}}


class CorridorPolicy:
    """Finite observation corridors with direction checks and physical rearm.

    Matching uses ONLY the public fields below; arbitrary extra fields, wall
    time and opponent skill labels cannot select an action. Re-arm requires
    observed opponent return AND own executor availability. A corridor is a
    finite-library baseline with explicit out-of-distribution limitations.
    """
    def __init__(self, corridors, *, confirmation_samples=2, reaction_delay_steps=0,
                 tip_tolerance_m=.065, grip_tolerance_m=.04,
                 minimum_speed_m_s=.025, minimum_direction_cosine=.2):
        if type(confirmation_samples) is not int or confirmation_samples<1:
            raise ValueError('Positive integer confirmation samples required')
        if type(reaction_delay_steps) is not int or not 0<=reaction_delay_steps<=20:
            raise ValueError('Bounded nonnegative decision delay required')
        if not np.isfinite([tip_tolerance_m,grip_tolerance_m,minimum_speed_m_s,minimum_direction_cosine]).all() \
                or min(tip_tolerance_m,grip_tolerance_m,minimum_speed_m_s)<=0 \
                or not -1<=minimum_direction_cosine<=1:
            raise ValueError('Finite geometric cue tolerances required')
        self.corridors=copy.deepcopy(corridors)
        if not corridors or len({c['action'] for c in corridors})!=len(corridors):
            raise ValueError('Nonempty uniquely named corridor actions required')
        for c in self.corridors:
            for key in ('tip','grip','velocity'):
                value=np.asarray(c[key],float)
                if value.shape!=(3,) or not np.isfinite(value).all():
                    raise ValueError('Finite three-vector corridor required')
                c[key]=value.tolist()
        self.confirmation_samples=confirmation_samples
        self.reaction_delay_steps=reaction_delay_steps
        self.tip_tolerance_m=tip_tolerance_m;self.grip_tolerance_m=grip_tolerance_m
        self.minimum_speed_m_s=minimum_speed_m_s;self.minimum_direction_cosine=minimum_direction_cosine
        self.count=0;self.candidate=None;self.fired=False;self.recovery_count=0
        self.initial_tip=None;self.initial_grip=None;self.events=[];self.samples=0

    def choose(self, observation, available):
        values={k:np.asarray(observation[k],float) for k in PUBLIC_FIELDS}
        if any(v.shape!=((7,) if k.startswith('own_') else (3,)) or not np.isfinite(v).all()
               for k,v in values.items()):
            raise ValueError('Finite public physical observations required')
        tip=values[PUBLIC_FIELDS[2]];grip=values[PUBLIC_FIELDS[3]];velocity=values[PUBLIC_FIELDS[4]]
        self.samples+=1
        if self.initial_tip is None:
            self.initial_tip=tip.tolist();self.initial_grip=grip.tolist()
        speed=float(np.linalg.norm(velocity))
        if self.fired:
            returned=(np.linalg.norm(tip-self.initial_tip)<.025
                and np.linalg.norm(grip-self.initial_grip)<.025 and speed<.015 and bool(available))
            self.recovery_count=self.recovery_count+1 if returned else 0
            if self.recovery_count>=5:
                self.fired=False;self.count=0;self.candidate=None
                self.events.append({'event':'observed_rearm','sample':self.samples})
            return None
        matches=[]
        for corridor in self.corridors:
            te=float(np.linalg.norm(tip-corridor['tip']))
            ge=float(np.linalg.norm(grip-corridor['grip']))
            pv=np.asarray(corridor['velocity'])
            cosine=float(velocity@pv/(speed*np.linalg.norm(pv))) if speed>0 and np.linalg.norm(pv)>0 else -1.
            if (corridor['action'] in available and te<=self.tip_tolerance_m
                    and ge<=self.grip_tolerance_m and speed>=self.minimum_speed_m_s
                    and cosine>=self.minimum_direction_cosine):
                matches.append((te/self.tip_tolerance_m+ge/self.grip_tolerance_m,
                    corridor['action'],te,ge,cosine))
        winner=min(matches) if matches else None
        name=winner[1] if winner else None
        self.count=self.count+1 if name and name==self.candidate else (1 if name else 0)
        self.candidate=name
        if name and self.count>=self.confirmation_samples+self.reaction_delay_steps:
            self.fired=True;self.recovery_count=0
            self.events.append({'event':'current_geometry_trigger','action':name,
                'sample':self.samples,'tip_error_m':winner[2], 'grip_error_m':winner[3],
                'velocity_cosine':winner[4], 'public_physical':{k:v.tolist() for k,v in values.items()}})
            return name
        return None


def _executor(entry, *, defender=False, mode='defence', time_scale=1.):
    if defender:
        initial=np.asarray(entry['ready_qpos'],float)
        nodes={'ready':initial}
        actions={}
        for spec in entry['bank']:
            key='counter' if mode=='counter' else 'defence'
            if key not in spec:
                continue
            curve=C2JointSpline(spec[key]['time_s'],spec[key]['qpos'])
            if not np.array_equal(curve.knot_qpos[0],initial) or not np.array_equal(curve.knot_qpos[-1],initial):
                raise ValueError('Each reactive skill must return to the same measured ready node')
            actions[spec['name']]=GraphAction(spec['name'],'ready','ready',curve)
        return SkillGraphExecutor(nodes,actions,'ready')
    curve=C2JointSpline(np.asarray(entry['attack']['time_s'])*time_scale,entry['attack']['qpos'])
    end_node='start' if np.array_equal(curve.knot_qpos[0],curve.knot_qpos[-1]) else 'end'
    nodes={'start':curve.knot_qpos[0],end_node:curve.knot_qpos[-1]}
    return SkillGraphExecutor(nodes,{'attack':GraphAction('attack','start',end_node,curve)},'start')


def compare_feedback(baseline_dir,defended_dir,scoring,region,attacker):
    baseline_dir,defended_dir=Path(baseline_dir),Path(defended_dir)
    b=json.loads((baseline_dir/'summary.json').read_text());d=json.loads((defended_dir/'summary.json').read_text())
    checks={key:b[key]==d[key] for key in
        ('runtime_config','joint_constraints','source_sha256','requested_duration_s','model_source_sha256')}
    for key in ('initial_qpos_rad','numerical_profile_source_sha256'):
        checks[key]=b[key]==d[key]
    with np.load(baseline_dir/'graph_references.npz') as br,np.load(defended_dir/'graph_references.npz') as dr:
        prefix=attacker+'__attack__'
        checks['same_attack_curve']=all(np.array_equal(br[prefix+k],dr[prefix+k])
            for k in ('knot_time_s','knot_qpos','coefficients'))
    for label,summary in [('baseline',b),('defended',d)]:
        starts=[e for e in summary['executor_events'][attacker] if e['event']=='started']
        checks[label+'_expected_attack_count']=len(starts)==summary['metadata'].get('repeat_count',1)
        if label=='baseline': baseline_starts=starts
        else: checks['same_attack_start']=starts==baseline_starts
    for label,path in [('baseline',baseline_dir),('defended',defended_dir)]:
        with np.load(path/'trace.npz') as tr:
            checks[label+'_full_native_trace']=bool(abs(tr['time_s'][0])<1e-8 and
                abs(tr['time_s'][-1]-b['requested_duration_s'])<1e-8 and
                np.allclose(np.diff(tr['time_s']),b['runtime_config']['physics_timestep_s'],atol=1e-9,rtol=0))
            if label=='baseline': initial_q=tr['qpos'][0].copy();initial_v=tr['qvel'][0].copy()
            else: checks['same_actual_initial_state']=bool(np.array_equal(initial_q,tr['qpos'][0]) and np.array_equal(initial_v,tr['qvel'][0]))
    ba=audit_recorded_triangle(baseline_dir,scoring,region,attacker)
    da=audit_recorded_triangle(defended_dir,scoring,region,attacker)
    contacts=json.loads((defended_dir/'contacts.json').read_text())
    result=triangle_defence_gate(b,ba,d,da,contacts,matching_checks=checks,attacker=attacker)
    defender='a2' if attacker=='a1' else 'a1'
    result.update(baseline_triangle_audit=ba,defended_triangle_audit=da,
        physical_trigger_events=d['policy_events'][defender],executor_events=d['executor_events'],
        native_contact_source_sha256=hashlib.sha256((defended_dir/'contacts.json').read_bytes()).hexdigest(),
        scope='Current-geometry corridor baseline only; not RL, unseen-motion coverage, or real sensing')
    return result


def run(library, case, output, *, attacker='a1', start_s=.76, time_scale=1.,delay_steps=0,mode='defence',repeat_count=1,baseline_from=None):
    library,output=Path(library).resolve(),Path(output).resolve()
    cfg=json.loads(library.read_text());entry=dict(cfg['cases'][case],bank=cfg['skills'])
    if attacker not in ('a1','a2') or mode not in ('defence','counter'):
        raise ValueError('Explicit slot and mode required')
    if not .2<=start_s<=5 or not .8<=time_scale<=1.2:
        raise ValueError('Use bounded held-out timing and time-scale perturbations')
    if type(repeat_count) is not int or not 1<=repeat_count<=2:
        raise ValueError('Use one or two explicitly scheduled test attacks')
    if output.exists(): raise FileExistsError('Preserve previous tests')
    output.mkdir(parents=True)
    defender='a2' if attacker=='a1' else 'a1'
    scoring=json.loads((ROOT/cfg['scoring']).read_text())
    source_bytes=Path(__file__).read_bytes()
    (output/'reactive_source_snapshot.py').write_bytes(source_bytes)
    attack_duration=float(entry['attack']['time_s'][-1])*time_scale
    longest_skill=max(float(s['counter' if mode=='counter' else 'defence']['time_s'][-1])
                      for s in cfg['skills'] if ('counter' if mode=='counter' else 'defence') in s)
    repeat_gap=max(10.,attack_duration+3.,longest_skill+3.)
    duration=float(cfg.get('duration_s',14.))
    if repeat_count>1:
        if not np.array_equal(entry['attack']['qpos'][0],entry['attack']['qpos'][-1]):
            raise ValueError('Repeats require an explicit physically closed attack path')
        duration=float(np.ceil((start_s+repeat_gap+attack_duration+longest_skill+2.)*1000)/1000)
    metadata={'attacker_slot':attacker,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'reactive_source_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'scope':'causal geometry diagnostic; not learned policy','case':case,'mode':mode,
        'attack_start_requested_s':start_s,'attack_time_scale':time_scale,'reaction_delay_steps':delay_steps,
        'repeat_count':repeat_count,'repeat_gap_s':repeat_gap if repeat_count>1 else None,
        'public_policy_fields':PUBLIC_FIELDS,'opponent_future_information':False}
    for name,active in [('baseline',False),('defended',True)]:
        if name=='baseline' and baseline_from is not None:
            original=Path(baseline_from).resolve()
            summary=json.loads((original/'summary.json').read_text())
            if not summary.get('native_execution_pass') or summary.get('status')!='completed':
                raise ValueError('Only complete successful native baseline evidence can be reused')
            shutil.copytree(original,output/name)
            continue
        ex={attacker:_executor(entry,time_scale=time_scale),defender:_executor(entry,defender=True,mode=mode)}
        corridors=[dict(s['corridor'],action=s['name']) for s in cfg['skills']
                   if s['name'] in ex[defender].actions]
        pol={attacker:TimedRequests([(round((start_s+i*repeat_gap)/.02),'attack') for i in range(repeat_count)]),
             defender:CorridorPolicy(corridors,reaction_delay_steps=delay_steps) if active else WaitPolicy()}
        run_feedback_trial(ex,pol,output/name,duration_s=duration,metadata=metadata,progress=print)
    result=compare_feedback(output/'baseline',output/'defended',scoring,entry['region'],attacker)
    result['reactive_source_unchanged_during_experiment']=Path(__file__).read_bytes()==source_bytes
    if not result['reactive_source_unchanged_during_experiment']:
        result.update(status='unmatched',successful_physical_defence=False)
    (output/'comparison.json').write_text(json.dumps(_finite_json(result),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','physical_trigger_events','attacker_entries_entire_window')},indent=2))
    return result


def build_upper_library(output):
    """Freeze the existing upper candidate and a current-state corridor."""
    output=Path(output)
    if output.exists(): raise FileExistsError('Do not overwrite a frozen library')
    source=ROOT/'outputs/a1_triangle_refine_v3/report.json'
    report=json.loads(source.read_text())
    spec=json.loads((ROOT/'config/a1_triangle_parry_v1.json').read_text())
    q=report['guards']['center']['qpos_rad']
    name=spec['attack'];attack=report['actions'][name]
    obs=corridor_observation(ROOT/'outputs/a1_triangle_parry_v1/baseline',2.2)['physical']
    curve={'time_s':[0.,.3,5.8,6.8],
           'qpos':[q,spec['geometry_derived_poses']['cross_early'],
                   spec['geometry_derived_poses']['cross_early'],q]}
    library={'schema_version':1,'candidate_only':True,'admitted_to_training':False,
        'scoring':'config/scoring_triangle_v1_candidate.json','duration_s':14.,
        'cases':{'upper':{'region':'upper','ready_qpos':q,'attack':{
            'source_action':name,'time_s':(np.asarray(attack['knot_time_s'])*2.).tolist(),
            'qpos':attack['knot_qpos']}}},
        'skills':[{'name':'parry_upper','defence':curve,
            'corridor':{'tip':obs[PUBLIC_FIELDS[2]],'grip':obs[PUBLIC_FIELDS[3]],'velocity':obs[PUBLIC_FIELDS[4]]}}],
        'calibration_note':'Current sword state at a previously validated entry, no clock or action label at runtime',
        'source_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(_finite_json(library),indent=2)+'\n')
    return library


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library');p.add_argument('--case');p.add_argument('--output',required=True)
    p.add_argument('--build-upper-library',action='store_true')
    p.add_argument('--attacker-slot',choices=('a1','a2'),default='a1');p.add_argument('--start-s',type=float,default=.76)
    p.add_argument('--time-scale',type=float,default=1.);p.add_argument('--delay-steps',type=int,default=0)
    p.add_argument('--mode',choices=('defence','counter'),default='defence')
    p.add_argument('--repeat-count',type=int,default=1)
    p.add_argument('--baseline-from',help='Reuse a complete native baseline; exact attack/initial state is still checked')
    a=p.parse_args()
    if a.build_upper_library: build_upper_library(a.output)
    elif not a.library or not a.case: p.error('--library and --case required')
    else: run(a.library,a.case,a.output,attacker=a.attacker_slot,start_s=a.start_s,
              time_scale=a.time_scale,delay_steps=a.delay_steps,mode=a.mode,repeat_count=a.repeat_count,baseline_from=a.baseline_from)


if __name__=='__main__': main()
