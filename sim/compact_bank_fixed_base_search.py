"""Offline constrained-base interception search; no native or learning claim.

J0 is constrained only inside the IK search box. Runtime remains the unmodified
robot with the full confirmed joint range, controller, force cap and scoring.
Outputs are compatible with compact_bank_new_defence.run for explicit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .a1_triangle_candidates import illegal_contacts
from .fencing_rules import blade_sections_from_scene
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from plan1_skills.matching import tool_fk


SOURCE=ROOT/'outputs/compact_bank_v1_natural/engagement01/report.json'


def write(path,value):
    Path(path).write_text(json.dumps(_finite_json(value),indent=2)+'\n')


def search(output,regions=('left','right','upper'),*,max_nfev=100,limit=5,
           fractions=(.15,.40,.65,.90)):
    if not fractions or any(not np.isfinite(f) or not 0<=f<=1 for f in fractions):
        raise ValueError('Fractions must lie within the physical blade segment')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Keep previous search evidence')
    output.mkdir(parents=True)
    source=json.loads(SOURCE.read_text());ready=np.asarray(source['center_qpos_rad'])
    constraints=load_joint_constraints();physical=model_joint_ranges_rad(constraints)
    box=physical+np.array([.02,-.02])
    box[0]=[max(box[0,0],ready[0]-np.deg2rad(3)),min(box[0,1],ready[0]+np.deg2rad(3))]
    scene=build_limited_duel(initial_qpos=ready)
    model=build_limited_floor(constraints).model;data=mujoco.MjData(model)
    a1q=np.array([scene.model.jnt_qposadr[scene.model.joint('a1_J'+str(i)).id] for i in range(7)])
    a2q=np.array([scene.model.jnt_qposadr[scene.model.joint('a2_J'+str(i)).id] for i in range(7)])
    reflection=np.array([-1.,-1.,1.]);base_shift=np.array([scene.config['base_distance_m'],0.,0.])
    blade_ids={slot:blade_collision_geom_ids(scene.model,slot+'_') for slot in ('a1','a2')}
    overview={'status':'running','candidate_only':True,'admitted_to_training':False,
        'runtime_j0_clamped':False,'search_j0_half_width_deg':3.,'regions':{},
        'attacker_blade_fractions':list(fractions),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),SOURCE,ROOT/'config/joint_constraints_observed.json']}}
    write(output/'report.json',overview)

    def actual_blade_gap(aq,dq):
        scene.data.qpos[a1q]=aq;scene.data.qpos[a2q]=dq
        mujoco.mj_forward(scene.model,scene.data)
        return min(mujoco.mj_geomDistance(scene.model,scene.data,i,j,5.,None)
                   for i in blade_ids['a1'] for j in blade_ids['a2'])

    for region in regions:
        folder=output/region;folder.mkdir()
        baseline=ROOT/f'outputs/compact_bank_v1_natural/native_{region}01/native'
        bs=json.loads((baseline/'summary.json').read_text())
        if bs.get('native_execution_pass') is not True or bs['metadata']['attacker_slot']!='a1':
            raise ValueError('Completed a1 baseline required')
        ba=json.loads((baseline.parent/'result.json').read_text())['triangle_audit']
        hit=min(e['time_s'] for e in ba['all_entries'] if e['agent']=='a1' and e['region']==region)
        with np.load(baseline/'trace.npz',allow_pickle=False) as z:
            times=z['time_s'].copy();states=z['qpos'].copy()
        def aq(t):return states[np.argmin(np.abs(times-t)),a1q]
        trials=[];eligible=[]
        for ti,t in enumerate(np.linspace(max(.80,hit-.35),hit-.04,5)):
            scene.data.qpos[a1q]=aq(t);scene.data.qpos[a2q]=ready
            mujoco.mj_forward(scene.model,scene.data)
            sections=blade_sections_from_scene(scene)['a1']
            begin,end=sections[0].start,sections[-1].end
            incoming=(end-begin)/np.linalg.norm(end-begin)
            previous=None
            for fraction in fractions:
                crossing=begin+fraction*(end-begin)
                target=(crossing-base_shift)*reflection
                for distance in (.18,.43,.68):
                    def residual(state):
                        q,along=state[:7],state[7]
                        g,axis,_=tool_fk(model,data,q)
                        return np.r_[10*(g+along*axis-target),
                            .035*np.array([1.,1.,1.,1.,1.,1.,.5])*(q-ready),
                            .004*(along-distance),
                            .2*max(0.,abs((axis*reflection)@incoming)-.85)]
                    seeds=[np.r_[ready,distance]]
                    if previous is not None:seeds.append(previous)
                    for si,seed in enumerate(seeds):
                        lo=np.r_[box[:,0],.06];hi=np.r_[box[:,1],.77]
                        fit=least_squares(residual,np.clip(seed,lo+1e-8,hi-1e-8),
                            bounds=(lo,hi),max_nfev=max_nfev)
                        q,along=fit.x[:7],float(fit.x[7]);g,axis,_=tool_fk(model,data,q)
                        error=float(np.linalg.norm(g+along*axis-target))
                        approach=max(.30,float(np.max(np.abs(q-ready)))*1.875/2.)
                        start=float(t-approach)
                        # Hold the intercept pose; do not reverse weak J0 at
                        # the nominal crossing before actual state can catch up.
                        release=max(float(t+.30),2.35)
                        recovery=max(.50,float(np.max(np.abs(q-ready)))*1.875/1.2)
                        kt=[0.,approach,release-start,release-start+recovery]
                        kq=np.array([ready,q,q,ready])
                        curve=C2JointSpline(kt,kq)
                        controls=curve.bezier_controls
                        bounds_ok=bool(np.all(controls>=physical[:,0]) and np.all(controls<=physical[:,1]))
                        exact_gap=actual_blade_gap(aq(t),q)
                        illegal=illegal_contacts(scene)
                        reasons=[]
                        if error>.012:reasons.append('material_point_fit')
                        if exact_gap>=-.001:reasons.append('no_actual_blade_proxy_intersection')
                        if illegal:reasons.append('illegal_intercept_contact')
                        if start<.75:reasons.append('would_start_before_attack')
                        if not bounds_ok:reasons.append('whole_curve_joint_bound')
                        peak=float(max(curve.analytic_extrema()['peak_speed_rad_s']))
                        if peak>2.+1e-8:reasons.append('reference_speed_over_2')
                        row={'qpos_rad':q.tolist(),'crossing_s':float(t),'start_s':start,
                            'approach_s':approach,'distance_m':along,'fraction':fraction,
                            'position_error_m':error,'axis':axis.tolist(),'tilt':0.,
                            'joint_travel':float(np.sum(np.abs(q-ready))),
                            'j0_delta_deg':float(np.rad2deg(q[0]-ready[0])),
                            'illegal_contacts':illegal,'exact_blade_proxy_gap_m':float(exact_gap),
                            'max_reference_speed_rad_s':peak,'whole_curve_bounds_pass':bounds_ok,
                            'knot_time_s':kt,'knot_qpos':kq.tolist(),'sweep_grip_shift_m':[0.,0.,0.],
                            'motion_kind':'hold_interception_then_recover','seed_index':si,
                            'reject_reasons':reasons,'solver_nfev':int(fit.nfev),
                            'actual_dynamic_tracking_tested':False}
                        if not reasons:
                            bad=[]
                            for now in np.arange(start,start+curve.duration_s+.001,.04):
                                scene.data.qpos[a1q]=aq(now)
                                scene.data.qpos[a2q]=curve.sample(now-start)
                                mujoco.mj_forward(scene.model,scene.data)
                                bad+=illegal_contacts(scene)
                            if bad:
                                row['reject_reasons'].append('illegal_full_reference_path')
                                row['illegal_path_contacts']=bad
                            else:eligible.append(row)
                        trials.append(row)
                        if error<.012:previous=fit.x.copy()
            print(region,'crossing',round(float(t),3),'eligible',len(eligible),flush=True)
        chosen=[]
        # Prefer smaller maximum displacement, then whole-arm travel. This
        # cannot be interpreted as proof of motor tracking or a timing window.
        for row in sorted(eligible,key=lambda r:(r['approach_s'],r['joint_travel'])):
            if all(np.linalg.norm(np.asarray(row['qpos_rad'])-old['qpos_rad'])>.05 for old in chosen):
                row=dict(row,id=len(chosen));chosen.append(row)
            if len(chosen)>=limit:break
        payload={'region':region,'source':str(SOURCE),'baseline':str(baseline),
            'baseline_hit_s':hit,'ready_qpos_rad':ready.tolist(),'selected':chosen,
            'all_candidates':trials,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'search_j0_half_width_deg':3.,'runtime_j0_clamped':False,
            'not_training_admitted':True,'geometry_only_not_defence_certificate':True,
            'selected_count':len(chosen)}
        write(folder/'geometry_candidates.json',payload)
        counts={reason:sum(reason in r['reject_reasons'] for r in trials)
                for reason in sorted({e for r in trials for e in r['reject_reasons']})}
        overview['regions'][region]={'trials':len(trials),'eligible':len(eligible),
            'selected':len(chosen),'rejection_counts':counts,
            'selected_geometry_file':str(folder/'geometry_candidates.json')}
        write(output/'report.json',overview)
    overview['status']='bounded_search_completed_candidate_only'
    write(output/'report.json',overview)
    return overview


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--regions',nargs='+',choices=['left','right','upper'],default=['left','right','upper'])
    p.add_argument('--max-nfev',type=int,default=100);p.add_argument('--limit',type=int,default=5)
    p.add_argument('--fractions',nargs='+',type=float,default=[.15,.40,.65,.90])
    a=p.parse_args();search(a.output,a.regions,max_nfev=a.max_nfev,limit=a.limit,fractions=a.fractions)
