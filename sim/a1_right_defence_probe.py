"""Bounded right-target native attack/parry development, never training admission.

Keeps source/physics/scoring fixed and saves failed candidates.  Geometry search
uses a frozen undeflected attack only to propose poses; the actual gate is the
coupled native full-window experiment, not a pose or collision label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plan1_skills.matching import tool_fk
from .assembly import ROOT, _finite_json
from .a1_triangle_candidates import illegal_contacts, region_gaps
from .a1_triangle_native import audit_recorded_triangle
from .a1_triangle_parry import parry_program, triangle_defence_gate
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .fencing_rules import FirstTouchScorer, blade_sections_from_scene, point_cylinder_distance
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


OUT=ROOT/'outputs/a1_right_defence_completion/run_v2'
SOURCE=ROOT/'outputs/a1_triangle_screen_v2'
ACTION='cut_left_to_right__from_left__to_right'
SCALE=2.
START=.75
DURATION=12.


def inputs():
    report=json.loads((SOURCE/'report.json').read_text())
    attack=report['actions'][ACTION]
    program=ScheduledReference(C2JointSpline(np.asarray(attack['knot_time_s'])*SCALE,
                                             attack['knot_qpos']),START)
    ready=np.asarray(report['guards']['center']['qpos_rad'])
    metadata={'source':str(SOURCE),'source_report_sha256':hashlib.sha256((SOURCE/'report.json').read_bytes()).hexdigest(),
              'action_name':ACTION,'time_scale':SCALE,'attacker_slot':'a1',
              'not_policy_reaction':True,'candidate_only':True}
    return report,program,ready,metadata


def save(path,value):
    path.write_text(json.dumps(_finite_json(value),ensure_ascii=False,indent=2)+'\n')


def recorded_audit(folder, scoring):
    """Same all-step swept cylinders; positions need kinematics, not dynamics.

    Check kinematic blade endpoints against full mj_forward at sampled native
    states before using them. This avoids recomputing constrained accelerations
    at every saved state, without subsampling the actual geometric audit.
    """
    folder=Path(folder)
    geometry=TriangleGeometry.from_mapping({k:scoring[k] for k in ('base_distance','regions')})
    summary=json.loads((folder/'summary.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    scene=SimpleNamespace(model=model,data=data,agents={s:{'prefix':s+'_'} for s in ('a1','a2')})
    with np.load(folder/'trace.npz',allow_pickle=False) as trace:
        times,qpos=trace['time_s'].copy(),trace['qpos'].copy()
    difference=0.
    for i in np.linspace(0,len(times)-1,24,dtype=int):
        data.qpos[:]=qpos[i];mujoco.mj_kinematics(model,data)
        kin=blade_sections_from_scene(scene)
        mujoco.mj_forward(model,data);full=blade_sections_from_scene(scene)
        difference=max(difference,max(float(np.max(np.abs(getattr(a,key)-getattr(b,key))))
            for s in ('a1','a2') for a,b in zip(kin[s],full[s]) for key in ('start','end')))
    if difference>1e-12:raise ValueError('Forward/kinematics geometric mismatch')
    scorers={r.name:FirstTouchScorer({'a1':geometry.world_center(r.name,'a2'),
          'a2':geometry.world_center(r.name,'a1')},r.radius) for r in geometry.regions}
    entries=[];previous=None
    for i,q in enumerate(qpos):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data);current=blade_sections_from_scene(scene)
        if previous is None:previous=current
        for region,scorer in scorers.items():
            for event in scorer.update(previous,current,float(times[max(0,i-1)]),float(times[i])):
                entries.append(dict(event,region=region,recorded_step=i,recorded_step_time_s=float(times[i])))
        previous=current
    first_step=min((x['recorded_step'] for x in entries),default=None)
    first=[x for x in entries if x['recorded_step']==first_step]
    first_regions=sorted({x['region'] for x in first if x['agent']=='a1'})
    return {'scope':'same all-step swept native cylinders; accelerations not recomputed',
       'native_trace_source_sha256':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('summary.json','trace.npz','scene.xml')},
       'scoring_snapshot':scoring,'target_region':'right','attacker_slot':'a1',
       'first_regions_at_recorded_native_step':first_regions,
       'intended_region_first':bool(first and first_regions==['right'] and all(x['agent']=='a1' for x in first)),
       'initial_overlap':{r:s.initial_overlap_events for r,s in scorers.items() if s.initial_overlap_events},
       'sweep_failures':{r:s.failure for r,s in scorers.items() if s.failure},'all_entries':entries,
       'minimum_clearance_m':{r:s.minimum_clearance_m for r,s in scorers.items()},
       'native_runner_full_window_pass':summary['native_execution_pass'],'native_runner_failure':summary['failure'],
       'full_recorded_duration_s':float(times[-1]),'kinematics_vs_full_forward_max_endpoint_difference_m':difference,
       'endpoint_equivalence_states_tested':24,'admitted_to_training':False}


def comparison(folder, scoring, baseline_folder=None):
    folder=Path(folder);base=Path(baseline_folder) if baseline_folder is not None else OUT/'baseline'
    b=json.loads((base/'summary.json').read_text());d=json.loads((folder/'summary.json').read_text())
    checks={n:b[n]==d[n] for n in ('runtime_config','joint_constraints','source_sha256','requested_duration_s','model_source_sha256')}
    checks['attack_start_s']=b['program_start_s']['a1']==d['program_start_s']['a1']
    with np.load(base/'references.npz',allow_pickle=False) as br,np.load(folder/'references.npz',allow_pickle=False) as dr:
        checks['attack_reference']=all(np.array_equal(br['a1_'+k],dr['a1_'+k]) for k in ('knot_time_s','knot_qpos','coefficients'))
    for label,p in [('baseline',base),('defended',folder)]:
        with np.load(p/'trace.npz',allow_pickle=False) as trace:
            t=trace['time_s'];checks[label+'_complete_native_trace']=bool(abs(t[0])<1e-9 and abs(t[-1]-b['requested_duration_s'])<1e-8 and np.allclose(np.diff(t),b['runtime_config']['physics_timestep_s'],rtol=0,atol=1e-9))
            if label=='baseline':q0=trace['qpos'][0].copy();dq0=trace['qvel'][0].copy()
            else:
                checks['same_initial_qpos']=np.array_equal(q0,trace['qpos'][0])
                checks['same_initial_qvel']=np.array_equal(dq0,trace['qvel'][0])
    ba=json.loads((base.parent/'baseline_audit.json').read_text())
    checks['baseline_saved_audit_source_unchanged']=all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in ba['native_trace_source_sha256'].items())
    da=recorded_audit(folder,scoring)
    contacts=json.loads((folder/'contacts.json').read_text())
    gate=triangle_defence_gate(b,ba,d,da,contacts,matching_checks=checks)
    return dict(gate,baseline_triangle_audit=ba,defended_triangle_audit=da,attacker_slot='a1',
          native_contact_source_sha256=hashlib.sha256((folder/'contacts.json').read_bytes()).hexdigest())


def baseline():
    report,attack,ready,metadata=inputs()
    OUT.mkdir(parents=True,exist_ok=True)
    result=run_constrained_trial({'a1':attack,'a2':hold_reference(ready,START,attack.curve.duration_s)},
                                OUT/'baseline',duration_s=DURATION,metadata=metadata,progress=print)
    audit=audit_recorded_triangle(OUT/'baseline',report['scoring'],'right')
    save(OUT/'baseline_audit.json',audit)
    print('baseline',result['native_execution_pass'],audit['intended_region_first'],flush=True)


def search():
    report,_,ready,_=inputs()
    constraints=load_joint_constraints()
    bounds=model_joint_ranges_rad(constraints)+np.array([.01,-.01])
    single=build_limited_floor(constraints).model
    single_data=mujoco.MjData(single)
    scene=build_limited_duel(1.75,initial_qpos=ready,constraints=constraints)
    geom=TriangleGeometry.from_mapping({k:report['scoring'][k] for k in ('base_distance','regions')})
    with np.load(OUT/'baseline/trace.npz',allow_pickle=False) as trace:
        times,qpos=trace['time_s'].copy(),trace['qpos'].copy()
    audit_path=OUT/'baseline_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        hits=[x for x in audit['all_entries'] if x['agent']=='a1']
        if not audit['intended_region_first'] or not hits:
            raise ValueError('A native right-region threat is required')
        hit=min(h['time_s'] for h in hits)
        timing_source='full native swept audit'
    else:
        # A coarse observed entry can seed search while the independent full
        # audit runs; it can never pass the final defence gate.
        hit=None
        target=geom.world_center('right','a2')
        radius=next(r.radius for r in geom.regions if r.name=='right')
        for i in range(0,len(times),20):
            scene.data.qpos[:]=qpos[i];mujoco.mj_forward(scene.model,scene.data)
            if min(point_cylinder_distance(target,s)[0] for s in blade_sections_from_scene(scene)['a1']) < radius:
                hit=float(times[i]);break
        if hit is None:raise ValueError('No coarse native entry to seed search')
        timing_source='20 ms native samples, proposal only; final full swept audit required'
    seeds=[(n,np.asarray(v['qpos_rad'])) for n,v in report['guards'].items()]
    rows=[]
    for time_s in np.linspace(max(.95,hit-1.25),hit-.12,7):
        frame=int(np.argmin(np.abs(times-time_s)))
        scene.data.qpos[:]=qpos[frame];mujoco.mj_forward(scene.model,scene.data)
        blade=blade_sections_from_scene(scene)['a1']
        start,end=blade[0].start,blade[-1].end
        for fraction in (.42,.60,.78):
            world=start+fraction*(end-start)
            target=np.array([1.75-world[0],-world[1],world[2]])
            for distance in (.30,.43,.56):
                for seed_name,seed in seeds:
                    def residual(q):
                        grip,axis,_=tool_fk(single,single_data,q)
                        return np.r_[grip+distance*axis-target,.002*(q-ready)]
                    fit=least_squares(residual,np.clip(seed,bounds[:,0]+1e-8,bounds[:,1]-1e-8),
                                      bounds=(bounds[:,0],bounds[:,1]),max_nfev=80)
                    q=fit.x;grip,axis,_=tool_fk(single,single_data,q)
                    error=float(np.linalg.norm(grip+distance*axis-target))
                    cross=float(abs(axis@((end-start)/np.linalg.norm(end-start)*[-1,-1,1])))
                    scene.data.qpos[:]=qpos[frame]
                    scene.data.qpos[scene.agents['a2']['qpos_ids']]=q
                    mujoco.mj_forward(scene.model,scene.data)
                    illegal=illegal_contacts(scene)
                    gaps=region_gaps(scene,geom,'a2')
                    contacts=[{'distance_m':float(c.dist),'geom1':mujoco.mj_id2name(scene.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),
                               'geom2':mujoco.mj_id2name(scene.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)}
                              for c in scene.data.contact if c.dist<0]
                    travel=float(np.linalg.norm(q-ready))
                    row={'qpos_rad':q.tolist(),'time_s':float(times[frame]),'fraction':fraction,'distance_m':distance,
                         'error_m':error,'axis_cosine':cross,'travel_rad':travel,'seed':seed_name,
                         'illegal_contacts':illegal,'target_gaps_m':gaps,'contacts':contacts,
                         'candidate':bool(error<.005 and cross<.9 and not illegal and min(gaps.values())>.05)}
                    rows.append(row)
    valid=sorted([r for r in rows if r['candidate']],key=lambda r:r['travel_rad'])
    selected=[]
    for row in valid:
        if any(np.linalg.norm(np.array(row['qpos_rad'])-r['qpos_rad'])<.12 for r in selected):continue
        duration=max(.35,1.9*max(abs(np.asarray(row['qpos_rad'])-ready)))
        row=dict(row,approach_s=duration,start_s=max(.05,row['time_s']-duration),release_s=8.,return_s=max(1.,duration))
        selected.append(row)
        if len(selected)>=12:break
    save(OUT/'search.json',{'scope':'frozen-trajectory crossing proposals, not a physical block certificate',
         'baseline_first_hit_s':hit,'timing_source':timing_source,'rows':rows,'selected':selected})
    print('selected',len(selected),flush=True)


def native(indices):
    report,attack,ready,metadata=inputs()
    proposals=json.loads((OUT/'search.json').read_text())['selected']
    for index in indices:
        row=proposals[index]
        spec={k:row[k] for k in ('start_s','approach_s','release_s','return_s')}
        program=parry_program(ready,row['qpos_rad'],spec,DURATION)
        name=f'candidate_{index:02d}'
        run_constrained_trial({'a1':attack,'a2':program},OUT/name,duration_s=DURATION,
            metadata=dict(metadata,candidate=row),progress=print)
        result=comparison(OUT/name,report['scoring'])
        save(OUT/(name+'_comparison.json'),result)
        print(name,result['status'],flush=True)


def guards(names):
    report,attack,ready,metadata=inputs()
    for name in names:
        q=np.asarray(report['guards'][name]['qpos_rad'])
        spec={'start_s':1.,'approach_s':.7,'release_s':8.,'return_s':1.}
        program=parry_program(ready,q,spec,DURATION)
        folder=OUT/('guard_'+name)
        run_constrained_trial({'a1':attack,'a2':program},folder,duration_s=DURATION,
                              metadata=dict(metadata,guard=name),progress=print)
        result=comparison(folder,report['scoring'])
        save(OUT/(folder.name+'_comparison.json'),result)
        print(folder.name,result['status'],flush=True)


def bridge(short=False, close_loop=False):
    """Explicit continuous center -> left -> cut, no state teleportation."""
    report,attack,ready,metadata=inputs()
    folder=OUT/('center_loop' if close_loop else 'center_bridge');folder.mkdir(parents=True,exist_ok=True)
    offset=1.75
    times=np.r_[0.,1.,offset,attack.curve.knot_time_s[1:]+offset]
    poses=np.vstack([ready,attack.curve.knot_qpos[0],attack.curve.knot_qpos])
    if close_loop:
        # The duplicate endpoint is a 0.5 s rest, preserving the original
        # strike's endpoint derivative before the distinct return edge.
        end=times[-1]
        times=np.r_[times,end+.5,end+1.5]
        poses=np.vstack([poses,poses[-1],ready])
    curve=C2JointSpline(times,poses)
    original_slice=curve.coefficients[2:2+len(attack.curve.coefficients)]
    if not np.allclose(original_slice,attack.curve.coefficients,rtol=0,atol=1e-11):
        raise ValueError('Preparation/recovery changed the original attack polynomial')
    if max(curve.analytic_extrema()['peak_speed_rad_s'])>2.+1e-9:
        raise ValueError('Bridge reference too fast')
    program=ScheduledReference(curve,START)
    duration=14.
    base=folder/'baseline'
    if not base.exists():
        run_constrained_trial({'a1':program,'a2':hold_reference(ready,START,curve.duration_s)},
            base,duration_s=duration,metadata=dict(metadata,center_bridge_s=offset,closed_center_loop=close_loop),progress=print)
    summary=json.loads((base/'summary.json').read_text())
    if not summary.get('native_execution_pass'):raise ValueError('Bridge baseline did not complete')
    if not (folder/'baseline_audit.json').exists():
        audit=recorded_audit(base,report['scoring']);save(folder/'baseline_audit.json',audit)
    row=json.loads((OUT/'search.json').read_text())['selected'][0]
    spec={k:row[k] for k in ('start_s','approach_s','release_s','return_s')}
    spec['start_s']+=offset;spec['release_s']=(3. if short else spec['release_s'])+offset
    defence=parry_program(ready,row['qpos_rad'],spec,duration)
    target=folder/('defended_short' if short else 'defended')
    run_constrained_trial({'a1':program,'a2':defence},target,duration_s=duration,
        metadata=dict(metadata,center_bridge_s=offset,closed_center_loop=close_loop,defence_probe=spec),progress=print)
    result=comparison(target,report['scoring'],base);save(folder/(target.name+'_comparison.json'),result)
    print('center_bridge',target.name,result['status'],flush=True)


def short_hold():
    report,attack,ready,metadata=inputs()
    row=json.loads((OUT/'search.json').read_text())['selected'][0]
    for release in (3.,3.5):
        spec={k:row[k] for k in ('start_s','approach_s','release_s','return_s')}
        spec['release_s']=release
        name='short_hold_'+str(release).replace('.','p')
        defence=parry_program(ready,row['qpos_rad'],spec,DURATION)
        run_constrained_trial({'a1':attack,'a2':defence},OUT/name,duration_s=DURATION,
            metadata=dict(metadata,defence_probe=spec),progress=print)
        result=comparison(OUT/name,report['scoring']);save(OUT/(name+'_comparison.json'),result)
        print(name,result['status'],flush=True)


def faster_response():
    """Bounded response-window probes; preserve previous safe timing."""
    report,attack,ready,metadata=inputs()
    row=json.loads((OUT/'search.json').read_text())['selected'][0]
    for start in (1.25,1.4):
        spec={'start_s':start,'approach_s':.7,'release_s':3.,'return_s':.9}
        name='fast_start_'+str(start).replace('.','p')
        defence=parry_program(ready,row['qpos_rad'],spec,DURATION)
        run_constrained_trial({'a1':attack,'a2':defence},OUT/name,duration_s=DURATION,
            metadata=dict(metadata,defence_probe=spec),progress=print)
        result=comparison(OUT/name,report['scoring']);save(OUT/(name+'_comparison.json'),result)
        print(name,result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('baseline','search','native','guards','review','bridge','bridge-short','loop','short','fast'))
    p.add_argument('--indices',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--names',nargs='+',default=['right','left','high'])
    a=p.parse_args()
    if a.mode=='baseline':baseline()
    elif a.mode=='search':search()
    elif a.mode=='native':native(a.indices)
    elif a.mode=='guards':guards(a.names)
    elif a.mode=='bridge':bridge()
    elif a.mode=='bridge-short':bridge(short=True)
    elif a.mode=='loop':bridge(short=True,close_loop=True)
    elif a.mode=='short':short_hold()
    elif a.mode=='fast':faster_response()
    else:
        report,_,_,_=inputs()
        for i in a.indices:
            name=f'candidate_{i:02d}'
            result=comparison(OUT/name,report['scoring']);save(OUT/(name+'_comparison.json'),result)
            print(name,result['status'],flush=True)
