"""Independent right-descending defensive subskill candidates.

The existing right parry remains unchanged. These bounded candidates constrain
the incoming distal blade instead of counting proximal weapon contact as a
successful block. All are offline authored, and need complete native pairs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from plan1_skills.matching import tool_fk
from .a1_dynamics_screen import screen_curve
from .a1_high_parry_repair import scene_checks
from .a1_high_parry_repair import radius_lookup
from .a1_guard_defence_screen import tool_point
from .a1_profile_bank import blade_identity
from .a1_recovery_geometry import static_screen
from .a1_upright_right import active_check, solve
from .assembly import _finite_json
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import GripAxisSolver
from .triangle_task import TriangleGeometry


def save(path, value):
    Path(path).write_text(json.dumps(_finite_json(value),indent=2)+'\n')


def build(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Retain prior candidates and failures')
    raw=source.read_bytes();base=json.loads(raw);output.mkdir(parents=True)
    model=build_limited_floor().model
    if blade_identity(model)['sha256']!=base['blade_geometry']['sha256']:
        raise ValueError('Frozen bank and current physical geometry differ')
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    entry_name='right_low_negative_robust';entry=np.asarray(base['guards'][entry_name]['qpos_rad'])
    scene=build_limited_duel(geometry.base_distance,initial_qpos=entry)
    attack_row=base['actions']['cut_down_right_to_left'];attack=C2JointSpline(attack_row['knot_time_s'],attack_row['knot_qpos'])
    prior_right=base['actions']['parry_right_upright_early']
    seed=np.asarray(prior_right['knot_qpos'][2])
    bounds=model_joint_ranges_rad(load_joint_constraints())+np.deg2rad([3.,-3.])
    bounds[0]=max(bounds[0,0],entry[0]-np.deg2rad(35)),min(bounds[0,1],entry[0]+np.deg2rad(35))
    bounds[2,1]=np.deg2rad(350);bounds[3,1]=np.deg2rad(115)
    bounds[5,0]=np.deg2rad(23);bounds[6]=np.deg2rad([-82.,82.])
    solver=GripAxisSolver(model,bounds)
    master={'status':'bounded_descending_right_parry_candidates','source_bank':str(source),
        'source_bank_sha256':hashlib.sha256(raw).hexdigest(),
        'collision_geometry_sha256':base['blade_geometry']['sha256'],
        'trials':[],'candidate_sources':[],'native_trials_performed':0,'admitted_to_training':False,
        'limitations':['A named subskill is not a defensive success.',
            'Distal point and force-normal fits are geometric hypotheses, not measured forces.',
            'Same entry and exit do not certify transitions under incoming attacks.',
            'Future attack phase is used offline only, not supplied to the policy.']}
    specs=(('distal_brace',.95,.75,1.),('distal_outward_sweep',1.08,.74,1.),
           ('early_distal_intercept',.78,.77,1.))
    for i,(kind,phase,length,sign)in enumerate(specs):
        block,fit=solve(model,bounds,entry,attack.sample(phase),length,seed,sign)
        row={'id':i,'design':kind,'attack_local_intercept_s':phase,'attacker_material_m':length,
             'block_qpos_rad':block.tolist(),'fit':fit,'candidate_pass':False}
        master['trials'].append(row)
        if fit['fit_pass']:
            p,axis,_=tool_fk(model,qpos=block);normal=np.asarray(fit['normal_force_on_attacker_defender_local'])
            if kind=='distal_outward_sweep':
                pre,pf=solver.solve(p-.035*normal,axis,block,iterations=110)
                post,sf=solver.solve(p+.075*normal+np.array([0.,0.,.025]),axis,block,iterations=110)
                times=[0.,phase-.27,phase,phase+.5,6.5,9.3]
                knots=[entry,pre,block,post,post,entry]
                row['pre_fit']=pf;row['post_fit']=sf
                fit_pass=pf['fit_pass']and sf['fit_pass']
            elif kind=='early_distal_intercept':
                post,sf=solver.solve(p+np.array([-.035,.060,.020]),axis,block,iterations=110)
                times=[0.,phase,phase+.65,6.5,9.3]
                knots=[entry,block,post,post,entry];row['post_fit']=sf;fit_pass=sf['fit_pass']
            else:
                times=[0.,phase,6.5,9.3];knots=[entry,block,block,entry];fit_pass=True
            curve=C2JointSpline(times,knots)
            active_start=.65*phase
            upright=active_check(model,curve,active_start,6.5)
            stat=static_screen(model,curve,181);dyn=screen_curve(model,curve,.015)
            idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True,samples=301)
            prefix=scene_checks(scene,geometry,attack,curve,through=phase+.02,samples=221)
            first=prefix['first_blade_contact']
            contact_up=False
            if first:
                gp,ga,_=tool_fk(model,qpos=curve.sample(first['local_s']))
                contact_up=bool(ga[2]>=np.sin(np.deg2rad(25.))and gp[2]>=.35)
                row['first_reference_contact_elevation_deg']=float(np.rad2deg(np.arcsin(ga[2])))
            idle_safe=not idle['illegal_contacts']and not idle['target_first_entries_local_s']
            passed=bool(fit_pass and upright['sampled_upright_pass'] and contact_up
                        and idle_safe and prefix['prefix_contact_candidate']
                        and stat['peak_static_cap_ratio']<1. and dyn['worst_utilization']<=1.)
            row.update(knot_time_s=times,knot_qpos=curve.knot_qpos.tolist(),upright=upright,
                       static_screen=stat,ideal_torque_screen=dyn,idle_screen=idle,prefix_screen=prefix,
                       candidate_pass=passed)
            if passed:
                folder=output/f'candidate{i:02d}';folder.mkdir();bank=copy.deepcopy(base)
                bank['actions']['parry_right_descending']={'kind':'defence','family':'parry_right',
                    'start_guard':entry_name,'end_guard':entry_name,'target_region':'right',
                    'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
                    'candidate_pass':True,'sampled_reference_pass':True,'successful_defence':False,
                    'native_dynamics_tested':False,'admitted_to_training':False,
                    'active_interval_s':[active_start,6.5],
                    'orientation_requirement':{'minimum_elevation_deg':25.,'minimum_grip_height_m':.35},
                    'phases':{'prepare':[0.,phase],'intercept':[phase,phase+.5],
                              'hold':[phase+.5,6.5],'exit':[6.5,9.3]},
                    'design':kind,'source_attack':'cut_down_right_to_left',
                    'runtime_action_mask_and_snapshot_update_required':True,
                    'source_report_sha256':master['source_bank_sha256']}
                bank.update(status='specialized_subskill_candidate_not_training_admitted',
                    parent_bank_sha256=master['source_bank_sha256'],admitted_to_training=False,
                    added_actions=['parry_right_descending'],existing_actions_unchanged=True)
                bank.setdefault('outgoing_actions',{}).setdefault(entry_name,[]).append('parry_right_descending')
                save(folder/'report.json',bank);(folder/'scene.xml').write_text(scene.spec.to_xml())
                ts=np.linspace(0.,curve.duration_s,601)
                np.savez_compressed(folder/'parry_right_descending.npz',time_s=ts,qpos=curve.sample(ts),
                                    knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
                row['candidate_source']=str(folder/'report.json');master['candidate_sources'].append(row['candidate_source'])
        print(json.dumps({'trial':i,'design':kind,'fit':fit['fit_pass'],
            'elevation':fit['active_elevation_deg'],'candidate':row['candidate_pass'],
            'ideal':row.get('ideal_torque_screen',{}).get('worst_utilization')}),flush=True)
        save(output/'report.json',master)
    master['source_unchanged']=source.read_bytes()==raw
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes());save(output/'report.json',master)
    return master


def retime_brace(source,previous,output):
    """One bounded clock correction of the first distal brace, no new pose."""
    source,previous,output=map(lambda p:Path(p).resolve(),(source,previous,output))
    if output.exists():raise FileExistsError('Keep failed clocks')
    raw=source.read_bytes();base=json.loads(raw);prior=json.loads(previous.read_text())
    trial=prior['trials'][0]
    if trial['design']!='distal_brace':raise ValueError('Only the inspected first brace is retimed')
    times=[0.,1.25,6.5,9.3];curve=C2JointSpline(times,trial['knot_qpos'])
    model=build_limited_floor().model
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    entry_name='right_low_negative_robust';scene=build_limited_duel(geometry.base_distance,
        initial_qpos=base['guards'][entry_name]['qpos_rad'])
    a=base['actions']['cut_down_right_to_left'];attack=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
    active_start=.65*1.25
    upright=active_check(model,curve,active_start,6.5)
    stat=static_screen(model,curve,181);dyn=screen_curve(model,curve,.01)
    idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True,samples=301)
    prefix=scene_checks(scene,geometry,attack,curve,through=1.27,samples=401)
    first=prefix['first_blade_contact'];contact_up=False;contact_angle=None
    if first:
        gp,ga,_=tool_fk(model,qpos=curve.sample(first['local_s']))
        contact_angle=float(np.rad2deg(np.arcsin(ga[2])))
        contact_up=bool(ga[2]>=np.sin(np.deg2rad(25.))and gp[2]>=.35)
    passed=bool(upright['sampled_upright_pass']and contact_up and
        not idle['illegal_contacts']and not idle['target_first_entries_local_s']and
        prefix['prefix_contact_candidate']and stat['peak_static_cap_ratio']<1. and dyn['worst_utilization']<=1.)
    output.mkdir(parents=True)
    result={'status':'one_local_clock_repair_of_distal_brace','candidate_pass':passed,
        'source_bank_sha256':hashlib.sha256(raw).hexdigest(),
        'failed_source':str(previous),'failed_source_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),
        'old_prepare_end_s':.95,'new_prepare_end_s':1.25,'same_qpos_knots':True,
        'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
        'upright':upright,'static_screen':stat,'ideal_torque_screen':dyn,
        'idle_screen':idle,'prefix_screen':prefix,'first_reference_contact_elevation_deg':contact_angle,
        'admitted_to_training':False,'native_tested':False}
    if passed:
        folder=output/'candidate00';folder.mkdir();bank=copy.deepcopy(base)
        bank['actions']['parry_right_descending']={'kind':'defence','family':'parry_right',
            'start_guard':entry_name,'end_guard':entry_name,'target_region':'right',
            'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
            'candidate_pass':True,'sampled_reference_pass':True,'successful_defence':False,
            'native_dynamics_tested':False,'admitted_to_training':False,
            'active_interval_s':[active_start,6.5],
            'orientation_requirement':{'minimum_elevation_deg':25.,'minimum_grip_height_m':.35},
            'phases':{'prepare':[0.,1.25],'intercept':[1.25,1.5],
                      'hold':[1.5,6.5],'exit':[6.5,9.3]},
            'design':'distal_brace_local_clock_repair','source_attack':'cut_down_right_to_left',
            'runtime_action_mask_and_snapshot_update_required':True,
            'source_report_sha256':result['source_bank_sha256']}
        bank.update(status='specialized_subskill_candidate_not_training_admitted',
            parent_bank_sha256=result['source_bank_sha256'],admitted_to_training=False,
            added_actions=['parry_right_descending'],existing_actions_unchanged=True)
        bank.setdefault('outgoing_actions',{}).setdefault(entry_name,[]).append('parry_right_descending')
        save(folder/'report.json',bank);(folder/'scene.xml').write_text(scene.spec.to_xml())
        ts=np.linspace(0,curve.duration_s,601)
        np.savez_compressed(folder/'parry_right_descending.npz',time_s=ts,qpos=curve.sample(ts),
                            knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        result['candidate_source']=str(folder/'report.json')
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    save(output/'report.json',result)
    print(json.dumps({'candidate':passed,'ideal':dyn['worst_utilization'],
        'first':first,'contact_elevation':contact_angle,'max_penetration':prefix['maximum_reference_penetration_m']}),flush=True)
    return result


def forward_branches(source,output):
    """Two explicitly forward-shaft branches, not more backward-point fits."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Retain previous branch failures')
    raw=source.read_bytes();base=json.loads(raw);output.mkdir(parents=True)
    model=build_limited_floor().model;data=mujoco.MjData(model);adata=mujoco.MjData(model)
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    entry_name='right_low_negative_robust';entry=np.array(base['guards'][entry_name]['qpos_rad'])
    scene=build_limited_duel(geometry.base_distance,initial_qpos=entry)
    a=base['actions']['cut_down_right_to_left'];attack=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
    bounds=model_joint_ranges_rad(load_joint_constraints())+np.deg2rad([3.,-3.])
    bounds[0]=max(bounds[0,0],entry[0]-np.deg2rad(35)),min(bounds[0,1],entry[0]+np.deg2rad(35))
    bounds[2,1]=np.deg2rad(350);bounds[3,1]=np.deg2rad(115)
    bounds[5,0]=np.deg2rad(23);bounds[6]=np.deg2rad([-82.,82.])
    solver=GripAxisSolver(model,bounds);rotation=np.diag([-1.,-1.,1.])
    caps=np.max(abs(model.actuator_forcerange),axis=1);radius=radius_lookup(model,data,entry)
    up=np.sin(np.deg2rad(25.))
    master={'status':'two_forward_axis_branch_prototypes','source_bank_sha256':hashlib.sha256(raw).hexdigest(),
        'trials':[],'candidate_sources':[],'admitted_to_training':False,'native_trials_performed':0,
        'branch_requirement':{'axis_x_min':.2,'axis_z_min':up,'grip_behind_contact':True},
        'scope':'Same original entry and attack; two bounded branch attempts, no universal feasibility claim.'}
    for idx,(axis_goal,phase,length)in enumerate(((np.array([.55,.25,.8]),1.0,.75),
                                                 (np.array([.8,-.15,.55]),.95,.77))):
        axis_goal=axis_goal/np.linalg.norm(axis_goal)
        aq=attack.sample(phase);ap,aa,aj=tool_point(model,adata,aq,length)
        target=np.array([geometry.base_distance,0.,0.])+rotation@ap;incoming=rotation@aa
        # Seed the forward-facing branch explicitly, rather than warm-starting
        # the earlier backward-shaft local solution.
        seed,seedfit=solver.solve(target-.32*axis_goal,axis_goal,entry,iterations=150,regularization=.003)
        def evaluate(x):
            p,axis,jac=tool_point(model,data,x[:7],x[7]);cross=np.cross(incoming,axis)
            normal=cross/max(np.linalg.norm(cross),1e-12)
            if normal[1]<0:normal*=-1
            sep=radius(length)+radius(x[7])-.0003
            return p,axis,normal,target-sep*normal,jac
        def residual(x):
            p,axis,normal,goal,jac=evaluate(x)
            data.qvel[:]=0.;mujoco.mj_forward(model,data)
            grip=p-x[7]*axis
            return np.r_[16.*(p-goal),.50*(axis-axis_goal),
                15.*max(.23-axis[0],0.),15.*max(up+.025-axis[2],0.),
                1.0*max(.35-grip[2],0.),.4*max(.40-np.linalg.norm(np.cross(incoming,axis)),0.),
                .25*np.maximum(abs(data.qfrc_bias)/caps-.96,0.),
                .016*np.array([3.,1.,1.,1.,2.,1.,1.])*(x[:7]-entry)]
        x0=np.r_[np.minimum(np.maximum(seed,bounds[:,0]+1e-8),bounds[:,1]-1e-8),.32]
        fitted=least_squares(residual,x0,bounds=(np.r_[bounds[:,0],.19],np.r_[bounds[:,1],.76]),
                             max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        block=fitted.x[:7];p,axis,normal,goal,jac=evaluate(fitted.x)
        grip=p-fitted.x[7]*axis;error=float(np.linalg.norm(p-goal))
        fit_pass=bool(error<.012 and axis[0]>=.2 and axis[2]>=up and grip[2]>=.35)
        row={'id':idx,'axis_goal':axis_goal.tolist(),'seed_fit':seedfit,'fit_pass':fit_pass,
             'point_error_m':error,'actual_axis':axis.tolist(),'actual_grip_m':grip.tolist(),
             'material_m':float(fitted.x[7]),'attack_material_m':length,'attack_local_s':phase,
             'block_qpos_rad':block.tolist(),'unwrapped_delta_deg':np.rad2deg(block-entry).tolist(),
             'force_on_attacker_defender_local':normal.tolist(),'candidate_pass':False}
        master['trials'].append(row)
        if fit_pass:
            times=[0.,1.20,6.5,9.3];curve=C2JointSpline(times,[entry,block,block,entry])
            active_start=.78;upright=active_check(model,curve,active_start,6.5)
            idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True,samples=301)
            prefix=scene_checks(scene,geometry,attack,curve,through=1.23,samples=401)
            stat=static_screen(model,curve,181);dyn=screen_curve(model,curve,.01)
            first=prefix['first_blade_contact'];contact_up=False
            if first:
                gp,ga,_=tool_fk(model,qpos=curve.sample(first['local_s']))
                row['first_contact_elevation_deg']=float(np.rad2deg(np.arcsin(ga[2])))
                contact_up=bool(ga[2]>=up and gp[2]>=.35)
            passed=bool(upright['sampled_upright_pass']and contact_up and
                not idle['illegal_contacts']and not idle['target_first_entries_local_s']and
                prefix['prefix_contact_candidate']and stat['peak_static_cap_ratio']<1. and dyn['worst_utilization']<=1.)
            row.update(knot_qpos=curve.knot_qpos.tolist(),knot_time_s=times,upright=upright,
                idle_screen=idle,prefix_screen=prefix,static_screen=stat,ideal_torque_screen=dyn,candidate_pass=passed)
            if passed:
                folder=output/f'candidate{idx:02d}';folder.mkdir();bank=copy.deepcopy(base)
                bank['actions']['parry_right_descending']={'kind':'defence','family':'parry_right',
                    'start_guard':entry_name,'end_guard':entry_name,'target_region':'right',
                    'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
                    'candidate_pass':True,'sampled_reference_pass':True,'successful_defence':False,
                    'native_dynamics_tested':False,'admitted_to_training':False,
                    'active_interval_s':[active_start,6.5],
                    'orientation_requirement':{'minimum_elevation_deg':25.,'minimum_grip_height_m':.35},
                    'phases':{'prepare':[0.,1.2],'intercept':[1.2,1.4],
                              'hold':[1.4,6.5],'exit':[6.5,9.3]},
                    'design':'forward_distal_brace','runtime_action_mask_and_snapshot_update_required':True}
                bank.update(parent_bank_sha256=master['source_bank_sha256'],added_actions=['parry_right_descending'],
                    existing_actions_unchanged=True,admitted_to_training=False,status='forward_descending_parry_candidate')
                bank.setdefault('outgoing_actions',{}).setdefault(entry_name,[]).append('parry_right_descending')
                save(folder/'report.json',bank);(folder/'scene.xml').write_text(scene.spec.to_xml())
                ts=np.linspace(0.,curve.duration_s,601)
                np.savez_compressed(folder/'parry_right_descending.npz',time_s=ts,qpos=curve.sample(ts),
                    knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
                row['candidate_source']=str(folder/'report.json');master['candidate_sources'].append(row['candidate_source'])
        print(json.dumps({'trial':idx,'fit':fit_pass,'axis':axis.tolist(),
            'delta_j4_deg':row['unwrapped_delta_deg'][4], 'ideal':row.get('ideal_torque_screen',{}).get('worst_utilization'),
            'candidate':row['candidate_pass']}),flush=True)
        save(output/'report.json',master)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes());save(output/'report.json',master)
    return master


def staged_forward(source,previous,output):
    """One near-body upright waypoint revision of forward branch 1.

    The existing interception pose and 1.20 s arrival deadline are immutable.
    One scalar time allocation is optimized, not a new pose/skill search grid.
    Failure remains a failed candidate, never an executable training action.
    """
    source,previous,output=map(lambda p:Path(p).resolve(),(source,previous,output))
    if output.exists():raise FileExistsError('Retain the previous route evidence')
    raw=source.read_bytes();base=json.loads(raw);oldraw=previous.read_bytes()
    prior=json.loads(oldraw);old=prior['trials'][1]
    model=build_limited_floor().model;data=mujoco.MjData(model)
    if blade_identity(model)['sha256']!=base['blade_geometry']['sha256']:
        raise ValueError('The frozen collision geometry changed')
    entry_name='right_low_negative_robust'
    entry=np.asarray(base['guards'][entry_name]['qpos_rad']);block=np.asarray(old['block_qpos_rad'])
    bounds=model_joint_ranges_rad(load_joint_constraints())+np.deg2rad([3.,-3.])
    # No long-way J0/J2/J4 detour: the midpoint stays between exact endpoints.
    for j in (0,2,4):bounds[j]=[min(entry[j],block[j]),max(entry[j],block[j])]
    bounds[2,1]=min(bounds[2,1],np.deg2rad(350.));bounds[3,1]=np.deg2rad(115.)
    bounds[5,0]=np.deg2rad(23.);bounds[6]=np.deg2rad([-82.,82.])
    seed=(entry+block)/2.;caps=np.max(abs(model.actuator_forcerange),axis=1)
    def residual(q):
        p,axis,_=tool_fk(model,data,q);data.qvel[:]=0.;mujoco.mj_forward(model,data)
        return np.r_[12.*max(.75-axis[2],0.),8.*max(.2-axis[0],0.),
            4.*max(.35-p[2],0.),3.*max(p[0]-.56,0.),2.*max(abs(p[1])-.22,0.),
            1.5*max(p[2]-.95,0.),.2*np.maximum(abs(data.qfrc_bias)/caps-.92,0.),
            .03*np.array([4.,1.,1.,1.,2.,1.,1.])*(q-seed)]
    fit=least_squares(residual,np.minimum(np.maximum(seed,bounds[:,0]+1e-8),bounds[:,1]-1e-8),
                      bounds=bounds.T,max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    mid=fit.x;p,axis,tip=tool_fk(model,data,mid)
    mid_pass=bool(axis[2]>=.749 and axis[0]>=.2 and .35<=p[2]<=.96 and p[0]<=.57 and abs(p[1])<=.23)
    evaluations=[]
    def make_curve(tmid,arrival=1.2):
        return C2JointSpline([0.,tmid,arrival,6.5,9.3],[entry,mid,block,block,entry])
    def objective(tmid):
        diagnostic=screen_curve(model,make_curve(float(tmid)),.02)
        evaluations.append({'midpoint_local_s':float(tmid),'worst_utilization':diagnostic['worst_utilization'],
            'worst_joint':diagnostic['worst_joint'],'worst_local_s':diagnostic['worst_time_s']})
        return diagnostic['worst_utilization']
    optimum=minimize_scalar(objective,bounds=(.35,.85),method='bounded',options={'xatol':.003,'maxiter':14})
    tmid=float(optimum.x);curve=make_curve(tmid);geometry=TriangleGeometry.from_mapping(
        {k:base['scoring'][k]for k in ('base_distance','regions')})
    a=base['actions']['cut_down_right_to_left'];attack=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
    scene=build_limited_duel(geometry.base_distance,initial_qpos=entry)
    output.mkdir(parents=True)
    result={'status':'one_staged_forward_route_revision','source_bank_sha256':hashlib.sha256(raw).hexdigest(),
        'source_bank':str(source),'prior_forward_source':str(previous),
        'prior_forward_sha256':hashlib.sha256(oldraw).hexdigest(),
        'collision_geometry_sha256':base['blade_geometry']['sha256'],
        'entry_guard':entry_name,'exit_guard':entry_name,'entry_and_exit_exact':True,
        'interception_pose_exactly_preserved':bool(np.array_equal(block,np.asarray(old['block_qpos_rad']))),
        'arrival_deadline_local_s':1.2,'baseline_hit_local_s':1.321422746552037,
        'candidate_pass':False,'admitted_to_training':False,'native_trials_performed':0,
        'midpoint':{'qpos_rad':mid.tolist(),'actual_grip_m':p.tolist(),'actual_axis':axis.tolist(),
            'actual_tip_m':tip.tolist(),'elevation_deg':float(np.rad2deg(np.arcsin(axis[2]))),
            'near_body_upright_fit_pass':mid_pass,'nfev':fit.nfev,
            'new_ready_node':False,'J0_J2_J4_monotone_between_endpoints':True},
        'time_allocation_evaluations':evaluations,'midpoint_local_s':tmid,
        'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
        'phases':{'near_body_raise':[0.,tmid],'forward_intercept':[tmid,1.2],
                  'hold':[1.2,6.5],'exit':[6.5,9.3]},
        'limitations':['This is one bounded route repair, not a broad search or an impossibility proof.',
            'A slower approach would change the reaction assumption; it is reported separately, not substituted.',
            'No dynamic, defensive, policy or runtime-graph certificate is inherited.']}
    save(output/'report.json',result)
    dyn=screen_curve(model,curve,.005);stat=static_screen(model,curve,201)
    upright=active_check(model,curve,tmid,6.5)
    idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True,samples=351)
    prefix=scene_checks(scene,geometry,attack,curve,through=1.23,samples=501)
    first=prefix['first_blade_contact'];contact_angle=None;contact_up=False
    if first:
        gp,ga,_=tool_fk(model,qpos=curve.sample(first['local_s']))
        contact_angle=float(np.rad2deg(np.arcsin(ga[2])))
        contact_up=bool(ga[2]>=np.sin(np.deg2rad(25.))and gp[2]>=.35)
    result.update(ideal_torque_screen=dyn,static_screen=stat,upright=upright,idle_screen=idle,
                  prefix_screen=prefix,first_contact_elevation_deg=contact_angle,
                  first_contact_upright=contact_up)
    passed=bool(mid_pass and upright['sampled_upright_pass']and contact_up and
        not idle['illegal_contacts']and not idle['target_first_entries_local_s']and
        prefix['prefix_contact_candidate']and stat['peak_static_cap_ratio']<1. and dyn['worst_utilization']<=1.)
    result['candidate_pass']=passed
    # Quantify just the ideal torque preparation requirement on this SAME path.
    # No new defence proposal is smuggled in under a later arrival.
    ratio=tmid/1.2;timing=[]
    def approach_ratio(duration):
        c=C2JointSpline([0.,ratio*duration,duration],[entry,mid,block])
        d=screen_curve(model,c,.01)
        timing.append({'approach_duration_s':float(duration),'worst_utilization':d['worst_utilization'],
                       'worst_joint':d['worst_joint'],'worst_local_s':d['worst_time_s']})
        return d['worst_utilization']
    lo,hi=1.2,3.6
    if approach_ratio(hi)<=1.:
        for _ in range(11):
            center=.5*(lo+hi)
            if approach_ratio(center)<=1.:hi=center
            else:lo=center
        result['torque_only_required_approach_s_upper_estimate']=hi
        result['torque_only_reaction_deadline_deficit_s']=hi-1.2
        result['torque_only_time_before_original_baseline_hit_s']=1.321422746552037-hi
    result['torque_only_preparation_diagnostic']=timing
    if passed:
        folder=output/'candidate01';folder.mkdir();bank=copy.deepcopy(base)
        bank['actions']['parry_right_descending']={'kind':'defence','family':'parry_right',
            'start_guard':entry_name,'end_guard':entry_name,'target_region':'right',
            'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
            'candidate_pass':True,'sampled_reference_pass':True,'successful_defence':False,
            'native_dynamics_tested':False,'admitted_to_training':False,
            'active_interval_s':[tmid,6.5],
            'orientation_requirement':{'minimum_elevation_deg':25.,'minimum_grip_height_m':.35},
            'phases':result['phases'],'design':'near_body_raise_then_forward_intercept',
            'runtime_action_mask_and_snapshot_update_required':True}
        bank.update(parent_bank_sha256=result['source_bank_sha256'],added_actions=['parry_right_descending'],
            existing_actions_unchanged=True,admitted_to_training=False,status='staged_descending_candidate')
        bank.setdefault('outgoing_actions',{}).setdefault(entry_name,[]).append('parry_right_descending')
        save(folder/'report.json',bank);(folder/'scene.xml').write_text(scene.spec.to_xml())
        result['candidate_source']=str(folder/'report.json')
    np.savez_compressed(output/'staged_path.npz',time_s=np.linspace(0.,curve.duration_s,601),
        qpos=curve.sample(np.linspace(0.,curve.duration_s,601)),knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    result['source_unchanged']=source.read_bytes()==raw and previous.read_bytes()==oldraw
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes());save(output/'report.json',result)
    print(json.dumps({'candidate_pass':passed,'midpoint_local_s':tmid,'ideal':dyn['worst_utilization'],
        'worst_joint':dyn['worst_joint'],'first_contact_elevation_deg':contact_angle,
        'first_contact_local_s':None if first is None else first['local_s'],
        'torque_only_approach_s':result.get('torque_only_required_approach_s_upper_estimate')}),flush=True)
    return result


def existing_high_entry(source,previous,output):
    """One changed-entry candidate, never presented as low-guard response."""
    source,previous,output=map(lambda p:Path(p).resolve(),(source,previous,output))
    if output.exists():raise FileExistsError('Keep changed-entry evidence separate')
    raw=source.read_bytes();base=json.loads(raw);oldraw=previous.read_bytes()
    old=json.loads(oldraw)['trials'][1];block=np.asarray(old['block_qpos_rad'])
    entry_name='right_high_positive_robust';entry=np.asarray(base['guards'][entry_name]['qpos_rad'])
    low=np.asarray(base['guards']['right_low_negative_robust']['qpos_rad'])
    model=build_limited_floor().model
    if blade_identity(model)['sha256']!=base['blade_geometry']['sha256']:
        raise ValueError('The frozen blade geometry changed')
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    scene=build_limited_duel(geometry.base_distance,initial_qpos=entry)
    a=base['actions']['cut_down_right_to_left'];attack=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
    times=[0.,1.2,6.5,9.3];curve=C2JointSpline(times,[entry,block,block,entry])
    held=C2JointSpline([0.,9.3],[entry,entry])
    result={'status':'existing_high_guard_changed_entry_candidate','source_bank':str(source),
        'source_bank_sha256':hashlib.sha256(raw).hexdigest(),
        'prior_forward_source':str(previous),'prior_forward_sha256':hashlib.sha256(oldraw).hexdigest(),
        'collision_geometry_sha256':base['blade_geometry']['sha256'],
        'entry_guard':entry_name,'exit_guard':entry_name,
        'entry_and_exit_exact':True,'new_ready_node':False,
        'old_low_guard_available':False,'candidate_pass':False,'admitted_to_training':False,
        'native_trials_performed':0,'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
        'joint_travel_comparison_deg':{'old_low_to_block':np.rad2deg(block-low).tolist(),
            'existing_high_to_block':np.rad2deg(block-entry).tolist()},
        'scope':'Only from the existing right_high_positive_robust node; prior arrival and state stability must be established. It is not a right-low immediate reaction.',
        'downstream_requirements':['Revalidate matched native baseline from this exact held high guard.',
            'Verify every active/loaded actual blade contact is at least 25 degrees upward.',
            'Complete all-target, full-window native defence and return with no fault.',
            'Validate real incoming-context transfer to this high guard before enabling its action-mask entry.',
            'Do not use the future attack identity to select preparation.']}
    output.mkdir(parents=True);save(output/'report.json',result)
    dyn=screen_curve(model,curve,.005);stat=static_screen(model,curve,201)
    baseline=scene_checks(scene,geometry,attack,held,samples=451)
    idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True,samples=351)
    prefix=scene_checks(scene,geometry,attack,curve,through=1.23,samples=501)
    first=prefix['first_blade_contact'];angle=None;contact_up=False
    if first:
        gp,ga,_=tool_fk(model,qpos=curve.sample(first['local_s']))
        angle=float(np.rad2deg(np.arcsin(ga[2])))
        contact_up=bool(ga[2]>=np.sin(np.deg2rad(25.))and gp[2]>=.35)
    active_start=.9;upright=active_check(model,curve,active_start,6.5)
    extrema=dyn['reference_extrema'];bounds=model_joint_ranges_rad(load_joint_constraints())
    bounds_pass=bool(np.all(np.asarray(extrema['position_min_rad'])>=bounds[:,0])and
                     np.all(np.asarray(extrema['position_max_rad'])<=bounds[:,1]))
    baseline_geometry_threat=bool('a1_right'in baseline['target_first_entries_local_s'])
    passed=bool(bounds_pass and contact_up and upright['sampled_upright_pass']and
        stat['peak_static_cap_ratio']<1. and dyn['worst_utilization']<=1. and
        not idle['illegal_contacts']and not idle['target_first_entries_local_s']and
        prefix['prefix_contact_candidate']and baseline_geometry_threat and not baseline['illegal_contacts'])
    result.update(candidate_pass=passed,ideal_torque_screen=dyn,static_screen=stat,
        baseline_geometry_screen=baseline,baseline_reference_threat=baseline_geometry_threat,
        baseline_actual_threat_unverified=True,idle_screen=idle,prefix_screen=prefix,
        upright=upright,first_contact_elevation_deg=angle,first_contact_upright=contact_up,
        analytic_physical_bounds_pass=bounds_pass)
    if passed:
        folder=output/'candidate01';folder.mkdir();bank=copy.deepcopy(base)
        bank['actions']['parry_right_descending']={'kind':'defence','family':'parry_right',
            'start_guard':entry_name,'end_guard':entry_name,'target_region':'right',
            'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
            'candidate_pass':True,'sampled_reference_pass':True,'successful_defence':False,
            'native_dynamics_tested':False,'admitted_to_training':False,
            'active_interval_s':[active_start,6.5],
            'orientation_requirement':{'minimum_elevation_deg':25.,'minimum_grip_height_m':.35},
            'phases':{'prepare':[0.,1.2],'intercept':[1.2,1.4],
                      'hold':[1.4,6.5],'exit':[6.5,9.3]},
            'design':'existing_high_guard_forward_distal_brace',
            'scope':result['scope'],'runtime_action_mask_and_snapshot_update_required':True}
        bank.update(parent_bank_sha256=result['source_bank_sha256'],added_actions=['parry_right_descending'],
            existing_actions_unchanged=True,admitted_to_training=False,status='high_entry_descending_candidate')
        bank.setdefault('outgoing_actions',{}).setdefault(entry_name,[]).append('parry_right_descending')
        save(folder/'report.json',bank);(folder/'scene.xml').write_text(scene.spec.to_xml())
        result['candidate_source']=str(folder/'report.json')
    np.savez_compressed(output/'high_entry_path.npz',time_s=np.linspace(0.,curve.duration_s,601),
        qpos=curve.sample(np.linspace(0.,curve.duration_s,601)),knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    result['source_unchanged']=source.read_bytes()==raw and previous.read_bytes()==oldraw
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes());save(output/'report.json',result)
    print(json.dumps({'candidate':passed,'ideal':dyn['worst_utilization'],
        'first_contact_local_s':None if first is None else first['local_s'],
        'contact_elevation':angle,'prefix_depth':prefix['maximum_reference_penetration_m'],
        'baseline_reference_threat':baseline_geometry_threat,
        'baseline_depth':baseline['maximum_reference_penetration_m']}),flush=True)
    return result


def research_ready_bank(source,assessment,output):
    """Freeze a disconnected ready-entry hypothesis for one native study."""
    source,assessment,output=map(lambda p:Path(p).resolve(),(source,assessment,output))
    if output.exists():raise FileExistsError('Research banks are immutable')
    raw=source.read_bytes();base=json.loads(raw);araw=assessment.read_bytes();r=json.loads(araw)
    boundary_path=assessment.parent/'contact_boundary.json';boundary=json.loads(boundary_path.read_bytes())
    if boundary['source_assessment_sha256']!=hashlib.sha256(araw).hexdigest():
        raise ValueError('Boundary check was not performed on this assessment')
    before=boundary['phases']['before_first_blade_contact'];first=boundary['phases']['first_contact_sample']
    if (before['all_contacts']!=0 or before['target_entries']or first['max_illegal_overlap_m']>0.
        or first['max_blade_overlap_m']>.005 or r['ideal']['worst_utilization']>1.
        or not r['upright']['sampled_upright_pass']):
        raise ValueError('Pre-contact or ideal-demand hypothesis gate failed')
    bank=copy.deepcopy(base);guard='descending_upright_research_ready'
    q=r['hypothetical_entry_qpos_rad'];curve=C2JointSpline(r['knot_time_s'],r['knot_qpos'])
    if not np.array_equal(curve.knot_qpos[0],q)or not np.array_equal(curve.knot_qpos[-1],q):
        raise ValueError('Research endpoints must be exact and explicit')
    bank['guards'][guard]={'qpos_rad':q,'static_reference_pass':True,
        'native_hold_tested':False,'reachable_from_existing_graph':False,
        'research_only':True,'source':'staged04 intermediate, not formally registered in A1'}
    bank['actions']['parry_right_descending']={'kind':'defence','family':'parry_right',
        'start_guard':guard,'end_guard':guard,'target_region':'right',
        'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
        'candidate_pass':True,'candidate_stage':'authorized_native_research_hypothesis',
        'sampled_reference_pass':False,'precontact_reference_pass':True,
        'postcontact_reference_crossing_retained':True,
        'successful_defence':False,'native_dynamics_tested':False,'admitted_to_training':False,
        'active_interval_s':[0.,6.5],
        'orientation_requirement':{'minimum_elevation_deg':25.,'minimum_grip_height_m':.35},
        'phases':{'prepare':[0.,1.2],'intercept':[1.2,1.4],'hold':[1.4,6.5],'exit':[6.5,9.3]},
        'scope':'One A1 native study from a hypothetical isolated upright ready; not executable from the existing A1 graph.',
        'runtime_action_mask_and_snapshot_update_required':True}
    bank.setdefault('outgoing_actions',{})[guard]=['parry_right_descending']
    bank.update(status='disconnected_ready_native_research_hypothesis',
        parent_bank_sha256=hashlib.sha256(raw).hexdigest(),
        collision_geometry_sha256=base['blade_geometry']['sha256'],
        added_actions=['parry_right_descending'],added_guards=[guard],
        existing_actions_unchanged=True,admitted_to_training=False,
        existing_A1_graph_coverage_improved=False,
        hypothesis_provenance={'assessment':str(assessment),'assessment_sha256':hashlib.sha256(araw).hexdigest(),
            'boundary_check':str(boundary_path),'boundary_sha256':hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
            'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'old_authoring_failure_kept':True,
            'postcontact_reference_overlap_is_not_measured_native_penetration':True,
            'physical_penetration_limit_m_unchanged':.005,
            'native_budget':{'A1_matched_pairs':1,'A2_matched_pairs':0}})
    model=build_limited_floor().model
    if blade_identity(model)['sha256']!=base['blade_geometry']['sha256']:
        raise ValueError('Frozen physical geometry changed')
    output.mkdir(parents=True);save(output/'report.json',bank)
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    scene=build_limited_duel(geometry.base_distance,initial_qpos=q)
    (output/'scene.xml').write_text(scene.spec.to_xml());(output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    ts=np.linspace(0.,curve.duration_s,601)
    np.savez_compressed(output/'parry_right_descending.npz',time_s=ts,qpos=curve.sample(ts),
                        knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    print(json.dumps({'source':str(output/'report.json'),
        'sha256':hashlib.sha256((output/'report.json').read_bytes()).hexdigest(),
        'native_research_only':True,'graph_connected':False}),flush=True)
    return bank


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--output',required=True);p.add_argument('--retime-source')
    p.add_argument('--forward-branches',action='store_true')
    p.add_argument('--staged-source')
    p.add_argument('--existing-high-source')
    p.add_argument('--research-assessment')
    a=p.parse_args()
    if a.research_assessment:research_ready_bank(a.source,a.research_assessment,a.output)
    elif a.existing_high_source:existing_high_entry(a.source,a.existing_high_source,a.output)
    elif a.staged_source:staged_forward(a.source,a.staged_source,a.output)
    elif a.forward_branches:forward_branches(a.source,a.output)
    elif a.retime_source:retime_brace(a.source,a.retime_source,a.output)
    else:build(a.source,a.output)
