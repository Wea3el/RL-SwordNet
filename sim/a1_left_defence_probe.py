"""Bounded left-region native attack/parry diagnostics; never training admission."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .assembly import ROOT, _finite_json
from .a1_triangle_candidates import illegal_contacts, retimed_curve, travel_cost, region_gaps
from .a1_triangle_parry import parry_program, compare
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .fencing_rules import blade_sections_from_scene
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry
from plan1_skills.matching import tool_fk

OUT=ROOT/'outputs/a1_left_defence_completion/v2'
SOURCE=ROOT/'outputs/a1_triangle_screen_v2/report.json'
ACTION='thrust_left__from_center__to_center'
HORIZON=12.

def write(path, value):
    path.write_text(json.dumps(_finite_json(value),indent=2)+'\n')

def inputs():
    report=json.loads(SOURCE.read_text())
    action=report['actions'][ACTION]
    attack=ScheduledReference(C2JointSpline(np.asarray(action['knot_time_s'])*2.,action['knot_qpos']),.75)
    ready=np.asarray(report['guards']['center']['qpos_rad'])
    return report,attack,ready

def baseline():
    OUT.mkdir(parents=True,exist_ok=True)
    report,attack,ready=inputs()
    run_constrained_trial({'a1':attack,'a2':hold_reference(ready,.75,attack.curve.duration_s)},OUT/'baseline',duration_s=HORIZON,
      metadata={'source_report':str(SOURCE.relative_to(ROOT)),'action':ACTION,'time_scale':2.,'candidate_only':True},progress=print)

def search():
    report,attack,ready=inputs()
    native=OUT/'baseline'
    search_trace=native/'trace.npz'
    if not search_trace.exists():
        search_trace=Path('/tmp/dynamic_readiness_triangle_20261123/left_slow2/trace.npz')
    trace=np.load(search_trace); tt=trace['time_s'];qq=trace['qpos']
    model=build_limited_floor().model;data=mujoco.MjData(model)
    scene=build_limited_duel(initial_qpos=ready)
    bounds=model_joint_ranges_rad(load_joint_constraints())+np.array([.012,-.012])
    reflect=np.array([-1.,-1.,1.]);trials=[];candidates=[]
    def at(t):return qq[np.argmin(abs(tt-t)),:7]
    for t in [2.25,2.5,2.75,2.95]:
      aq=at(t);scene.data.qpos[:7]=aq;scene.data.qpos[7:]=ready;mujoco.mj_forward(scene.model,scene.data)
      blades=blade_sections_from_scene(scene)['a1'];base=blades[0].start;tip=blades[-1].end;axis=(tip-base)/np.linalg.norm(tip-base)
      for frac in [.7,.9]:
       crossing=base+(tip-base)*frac;target=(crossing-np.array([1.75,0,0]))*reflect
       for distance in [.32,.46,.6]:
        for seedname in ['center','left','high']:
         seed=np.asarray(report['guards'][seedname]['qpos_rad'])
         def residual(q):
            grip,daxis,_=tool_fk(model,data,q)
            return np.r_[8*(grip+distance*daxis-target),.22*(daxis-np.array([0.,0.,1.])),.4*max(0.,abs(daxis*reflect@axis)-.45),.014*(q-ready)]
         fit=least_squares(residual,np.clip(seed,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=85)
         q=fit.x;grip,daxis,_=tool_fk(model,data,q);error=np.linalg.norm(grip+distance*daxis-target)
         scene.data.qpos[:7]=aq;scene.data.qpos[7:]=q;mujoco.mj_forward(scene.model,scene.data)
         illegal=illegal_contacts(scene)
         row={'qpos_rad':q.tolist(),'at_s':t,'fraction':frac,'distance_m':distance,'error_m':float(error),'illegal_contacts':illegal,'cost':travel_cost([ready,q]),'seed':seedname}
         trials.append(row)
         if error>.01 or illegal:continue
         curve=retimed_curve([ready,q],speed_limit=2.)
         start=t-curve.duration_s
         if start<.8:continue
         path_bad=[]
         for when in np.arange(start,3.16,.03):
          scene.data.qpos[:7]=at(when);scene.data.qpos[7:]=curve.sample(min(max(when-start,0),curve.duration_s));mujoco.mj_forward(scene.model,scene.data)
          path_bad+=illegal_contacts(scene)
         row.update(start_s=float(start),approach_s=float(curve.duration_s),path_illegal=path_bad)
         if not path_bad:candidates.append(row)
      print('searched',t,len(candidates),flush=True)
    selected=[]
    for r in sorted(candidates,key=lambda r:r['cost']):
      if all(np.linalg.norm(np.asarray(r['qpos_rad'])-s['qpos_rad'])>.12 for s in selected):selected.append(r)
      if len(selected)>=6:break
    write(OUT/'geometry_candidates.json',{'selected':selected,'trials':trials,'search_trace':str(search_trace),'scope':'IK geometry only, never defence certification'})
    print(json.dumps(selected,indent=2))

def run_cases(indices, starts=None, fast_screen=False):
    report,attack,ready=inputs()
    candidates=json.loads((OUT/'geometry_candidates.json').read_text())['selected']
    score=json.loads((ROOT/'config/scoring_triangle_v1_candidate.json').read_text())
    for index in indices:
      row=candidates[index]
      for start in (starts if starts is not None else [max(.8,row['start_s']-.10)]):
       name=f'candidate_{index}_start_{start:.3f}'.replace('.','p')
       spec={'start_s':start,'approach_s':max(.3,row['approach_s']),'release_s':8.,'return_s':1.}
       program=parry_program(ready,row['qpos_rad'],spec,HORIZON)
       run_constrained_trial({'a1':attack,'a2':program},OUT/name,duration_s=HORIZON,metadata={'candidate':row,'defence_probe':spec,'not_policy_reaction':True},progress=print)
       if fast_screen:
        summary=json.loads((OUT/name/'summary.json').read_text())
        scene=build_limited_duel(initial_qpos=ready)
        geometry=TriangleGeometry.from_mapping({k:score[k] for k in ('base_distance','regions')})
        entries=[]
        with np.load(OUT/name/'trace.npz') as trace:
         for t,q in zip(trace['time_s'],trace['qpos']):
          scene.data.qpos[:]=q;mujoco.mj_forward(scene.model,scene.data)
          gaps=region_gaps(scene,geometry,'a1')
          if min(gaps.values())<-.00001:
           entries.append({'t':float(t),'gaps':gaps});break
        write(OUT/(name+'_fast_screen.json'),{'native_pass':summary['native_execution_pass'],'entries':entries,'not_a_certificate':True})
        if not summary['native_execution_pass'] or entries:
         print(name,'rejected_fast_screen',summary['native_execution_pass'],entries,flush=True)
         continue
       result=compare(OUT/'baseline',OUT/name,score,target='left')
       write(OUT/(name+'_comparison.json'),result)
       print(name,result['status'],flush=True)

def swipe(sign,start=2.34,release=None):
    report,attack,ready=inputs(); candidates=json.loads((OUT/'geometry_candidates.json').read_text())['selected']
    q=np.asarray(candidates[1]['qpos_rad']); model=build_limited_floor().model;data=mujoco.MjData(model)
    grip,axis,_=tool_fk(model,data,q);target=grip+np.array([0.,sign*.14,0.])
    bounds=model_joint_ranges_rad(load_joint_constraints())+np.array([.025,-.025])
    def residual(x):
      g,a,_=tool_fk(model,data,x)
      return np.r_[8*(g-target),.25*(a-axis),.01*(x-q)]
    fit=least_squares(residual,q,bounds=(bounds[:,0],bounds[:,1]),max_nfev=150)
    release=start+5.66 if release is None else release
    curve=C2JointSpline([0.,.3,1.3,release-start,release-start+1.],[ready,q,fit.x,fit.x,ready])
    spec={'start_s':start,'first_contact_pose_s':start+.3,'swipe_end_s':start+1.3,'release_s':release,'return_s':1.,'sign':sign,'target_grip':target.tolist(),'q0':q.tolist(),'q1':fit.x.tolist(),'peak_speed':curve.analytic_extrema()['peak_speed_rad_s']}
    if max(spec['peak_speed'])>2.:raise ValueError('Too fast')
    name='swipe_'+('positive' if sign>0 else 'negative')
    if start!=2.34:name+=f'_start_{start:.3f}'.replace('.','p')
    if release!=start+5.66:name+=f'_release_{release:.3f}'.replace('.','p')
    run_constrained_trial({'a1':attack,'a2':ScheduledReference(curve,start)},OUT/name,duration_s=12.,metadata={'defence_probe':spec,'not_policy_reaction':True},progress=print)
    score=json.loads((ROOT/'config/scoring_triangle_v1_candidate.json').read_text())
    result=compare(OUT/'baseline',OUT/name,score,target='left');write(OUT/(name+'_comparison.json'),result)
    print(name,result['status'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['baseline','search','cases','swipe']);p.add_argument('--indices',type=int,nargs='+',default=[0]);p.add_argument('--starts',type=float,nargs='+');p.add_argument('--fast-screen',action='store_true');p.add_argument('--sign',type=int,default=1);p.add_argument('--release',type=float);a=p.parse_args()
    if a.stage=='baseline':baseline()
    elif a.stage=='search':search()
    elif a.stage=='cases':run_cases(a.indices,a.starts,a.fast_screen)
    else:
      for start in (a.starts or [2.34]):swipe(a.sign,start,a.release)

if __name__=='__main__':main()
