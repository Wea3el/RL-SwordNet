"""Offline small-J0 defence geometry from existing non-central guard nodes.

This screens contact authoring, static loading, and explicit approach paths.
It never runs native dynamics or grants a defensive success certificate.
Incoming future paths are used only offline and are not policy observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plan1_skills.matching import tool_fk, _jacobian
from .assembly import ROOT, _finite_json
from .authoring_bounds import intersect_authoring_bounds
from .blade_geometry import blade_collision_geom_ids
from .a1_triangle_candidates import region_gaps
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import pair_screen, unit, audit_motion
from .a1_dynamics_screen import screen_curve
from .triangle_task import TriangleGeometry


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value),ensure_ascii=False,indent=2)+'\n')


def _authoring_bounds(physical, qguard, j0_deg):
    interior = np.asarray(physical) + np.deg2rad([5., -5.])
    requested = interior.copy()
    requested[2] = np.deg2rad([10., 350.])
    requested[3] = np.deg2rad([-15., 115.])
    requested[5] = np.deg2rad([23., 137.])
    requested[6] = np.deg2rad([-82., 82.])
    requested[0] = (max(interior[0, 0], qguard[0]-np.deg2rad(j0_deg)),
                    min(interior[0, 1], qguard[0]+np.deg2rad(j0_deg)))
    return intersect_authoring_bounds(interior, requested)


def tool_point(model,data,q,length):
    p,a,t=tool_fk(model,data,q)
    jp,ja=_jacobian(model,data,model.site('shinai_grip').id,model.site('shinai_tip').id,a,np.linalg.norm(t-p))
    return p+length*a,a,jp+length*ja


def static(model,data,q,caps):
    data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data)
    return {'bias_torque_nm':data.qfrc_bias.copy().tolist(),
            'max_static_cap_ratio':float(max(abs(data.qfrc_bias)/caps))}


def solve_point(model,qguard,qattack,attack_length,bounds,normal_sign):
    data=mujoco.MjData(model);adata=mujoco.MjData(model)
    pa,aa,ja=tool_point(model,adata,qattack,attack_length)
    # All contact math below is in defender local frame.
    rotation=np.diag([-1.,-1.,1.]);target=np.array([1.75,0,0])+rotation@pa
    attack_axis=rotation@aa
    caps=np.max(abs(model.actuator_forcerange),axis=1)
    # Actual flat-cylinder radii differ substantially along the tapered blade.
    # A fixed24mm separation could request >5mm penetration on proximal parts.
    gp,ga,_=tool_fk(model,data,qguard)
    sections=[]
    for g in blade_collision_geom_ids(model, ''):
        midpoint=float((data.geom_xpos[g]-gp)@ga);half=float(model.geom_size[g,1])
        sections.append((midpoint-half,midpoint+half,float(model.geom_size[g,0])))
    def radius(length):
        return max((r for lo,hi,r in sections if lo-1e-8<=length<=hi+1e-8),default=sections[-1][2])
    attacker_radius=radius(attack_length)
    def describe(v):
        q,ell=v[:7],v[7];point,axis,jd=tool_point(model,data,q,ell)
        cross=np.cross(attack_axis,axis);norm=np.linalg.norm(cross)
        normal=normal_sign*cross/max(norm,1e-9) # force on attacker, defender local
        attacker_force_world=rotation@normal
        ta=ja.T@attacker_force_world;td=jd.T@(-normal)
        return point,axis,jd,normal,ta,td,norm
    def residual(v):
        point,axis,jd,normal,ta,td,norm=describe(v)
        centerline_gap=attacker_radius+radius(v[7])-.0005
        goal=target-centerline_gap*normal
        data.qvel[:]=0;mujoco.mj_forward(model,data)
        excess=np.maximum(abs(data.qfrc_bias)/caps-.96,0)
        return np.r_[8*(point-goal),2*max(.65-norm,0),
            1.2*max(-ta[5],0),1.2*max(-td[5],0),.12*excess,
            .018*np.array([4,1,1,1,1,1,1])*(v[:7]-qguard),.01*(v[7]-.40)]
    initial=np.r_[np.minimum(np.maximum(qguard,bounds[:,0]+1e-8),bounds[:,1]-1e-8),.40]
    result=least_squares(residual,initial,bounds=(np.r_[bounds[:,0],.20],np.r_[bounds[:,1],.76]),
                         max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    point,axis,jd,n,ta,td,norm=describe(result.x)
    centerline_gap=attacker_radius+radius(result.x[7])-.0005
    error=np.linalg.norm(point-(target-centerline_gap*n))
    return result.x[:7],{'material_distance_m':float(result.x[7]),
        'attacker_material_distance_m':attack_length,'target_defender_local_m':target.tolist(),
        'defender_material_point_m':point.tolist(),'defender_axis':axis.tolist(),
        'attacker_axis_defender_local':attack_axis.tolist(),'point_fit_error_m':float(error),
        'crossing_sine':float(norm),'crossing_angle_acute_deg':float(np.rad2deg(np.arcsin(min(norm,1)))),
        'predicted_unit_force_on_attacker_defender_local':n.tolist(),
        'predicted_attacker_joint_torque_per_newton_nm':ta.tolist(),
        'predicted_defender_joint_torque_per_newton_nm':td.tolist(),
        'authoring_centerline_separation_m':centerline_gap,
        'actual_proxy_radii_m':[attacker_radius,radius(result.x[7])],
        'fit_pass':bool(error<.015 and norm>.60),
        'force_note':'Geometric signed cross-axis normal and Jacobian leverage only. Not measured force or validated deflection.'}


def contact_state(scene,geometry,qa,qd,attacker='a1'):
    defender='a2' if attacker=='a1' else 'a1'
    scene.data.qpos[scene.agents[attacker]['qpos_ids']]=qa
    scene.data.qpos[scene.agents[defender]['qpos_ids']]=qd
    scene.data.qvel[:]=0;mujoco.mj_forward(scene.model,scene.data)
    contacts=[]
    from .duel_scene import classify_contact
    for c in scene.data.contact:
        if c.dist>=0:continue
        contacts.append({'category':classify_contact(scene.model,c,rules_version='whole_blade_first_touch_v2'),
            'depth_m':float(-c.dist),'geoms':[scene.model.geom(int(g)).name for g in (c.geom1,c.geom2)]})
    gap=.2
    for ai in blade_collision_geom_ids(scene.model, attacker+'_'):
        for di in blade_collision_geom_ids(scene.model, defender+'_'):
            gap=min(gap,float(mujoco.mj_geomDistance(scene.model,scene.data,
                ai,di,.2,None)))
    return {'contacts':contacts,'minimum_blade_gap_m':gap,
        'target_gaps_m':{a:region_gaps(scene,geometry,a) for a in ('a1','a2')}}


def approach_check(scene,geometry,model,attack_curve,qguard,qblock,phase,caps,speed=1.1):
    delta=abs(qblock-qguard)
    duration=max(.65,1.9*delta[0]/.45,1.9*max(delta[1:])/speed)
    start=phase-duration
    if start<.15:
        return {'candidate_pass':False,'reason':'insufficient approach time after minimum reaction delay',
                'duration_s':float(duration),'planned_start_after_attack_s':float(start)},None
    curve=C2JointSpline([0.,duration],[qguard,qblock])
    first=None;contacts={};min_gap=.2;score=False;bias=[]
    for time in np.linspace(0,phase,int(np.ceil(phase/.025))+1):
        qd=curve.sample(max(0,time-start));qa=attack_curve.sample(time)
        state=contact_state(scene,geometry,qa,qd)
        min_gap=min(min_gap,state['minimum_blade_gap_m'])
        if state['minimum_blade_gap_m']<.002 and first is None:first=float(time)
        for c in state['contacts']:contacts.setdefault('|'.join(c['geoms']),dict(c,time_s=float(time)))
        score |= any(g<0 for d in state['target_gaps_m'].values() for g in d.values())
        data=mujoco.MjData(model);data.qpos[:]=qd;mujoco.mj_forward(model,data);bias.append(data.qfrc_bias.copy())
    illegal=[c for c in contacts.values() if c['category'].startswith('illegal_')]
    deep=[c for c in contacts.values() if c['depth_m']>.003]
    dynamics=screen_curve(model,curve)
    maximum_bias_ratio=float(np.max(abs(np.asarray(bias))/caps))
    passed=not illegal and not deep and not score and first is not None and maximum_bias_ratio<1.
    return {'candidate_pass':bool(passed),'duration_s':float(duration),'planned_start_after_attack_s':float(start),
        'first_nominal_close_contact_s':first,'contacts':list(contacts.values()),'minimum_blade_gap_m':min_gap,
        'illegal_contacts':illegal,'penetrations_over_3mm':deep,'any_pre_contact_target_entry':bool(score),
        'max_static_cap_ratio_during_approach':maximum_bias_ratio,
        'ideal_free_space_dynamics':dynamics,
        'scope':'Sampled nominal approach only through planned contact; no full-window deflection or native proof.'},curve


def build(output,focus_positive=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous screens')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_chain_geometry/chain12/report.json';r=json.loads(source.read_text())
    geometry=TriangleGeometry.from_mapping({k:r['scoring'][k] for k in ('base_distance','regions')})
    constraints=load_joint_constraints();physical=model_joint_ranges_rad(constraints)
    model=build_limited_floor(constraints).model;data=mujoco.MjData(model)
    caps=np.max(abs(model.actuator_forcerange),axis=1)
    scene=build_limited_duel(geometry.base_distance,initial_qpos=r['guards']['center']['qpos_rad'],constraints=constraints)
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'bounded_other_guard_contact_screen','scoring':r['scoring'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,snapshot)},
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'guards':r['guards'],'actions':{},'trials':[],
        'scope':'Other reachable guards and both incoming sides, small J0 motion, no native execution.',
        'limitations':['Sampled geometric contact is not successful defence.',
            'Cross-axis force signs are hypothetical; contact solver can produce other force directions.',
            'Full incoming/recovery and mirrored-slot native tests remain mandatory.',
            'Known future attack trajectory is offline authoring data only.']}
    attack_specs=([('cut_up_left_to_right',(2.15,2.30,2.40))] if focus_positive else
        [('cut_down_left_to_right',(2.4,2.9,3.2)),('cut_up_left_to_right',(1.9,2.15,2.35))])
    guards=(('left_high_negative_robust','right_low_negative_robust') if focus_positive else
            ('left_high_negative_robust','right_low_negative_robust','right_high_positive_robust'))
    report['baseline_assessments']=[]
    for name,_ in attack_specs:report['actions'][name]=r['actions'][name]
    write(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    trial_id=0
    for name,phases in attack_specs:
        a=r['actions'][name];curve=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
        for guard in guards:
            qguard=np.array(r['guards'][guard]['qpos_rad'])
            baseline,_=audit_motion(scene,geometry,curve,qguard,a['direction_intent_attacker_frame'],a['target_region'],period=.035)
            report['baseline_assessments'].append({'attack':name,'guard':guard,'audit':baseline,
                'meaning':'A sampled reference-clear baseline only; native matched baseline still required.'})
            if not baseline['sampled_reference_pass']:
                report['trials'].append({'id':trial_id,'attack':name,'guard':guard,'candidate_pass':False,
                    'reason':'held-guard full-window baseline has contact or invalid attack reference',
                    'held_guard_audit':baseline})
                trial_id+=1;continue
            initial=contact_state(scene,geometry,curve.knot_qpos[0],qguard)
            if initial['contacts'] or any(v<0 for g in initial['target_gaps_m'].values() for v in g.values()):
                report['trials'].append({'id':trial_id,'attack':name,'guard':guard,
                    'initial_state':initial,'candidate_pass':False,'reason':'initial pair overlap or target entry'})
                trial_id+=1;continue
            found=False
            for j0_deg in (15.,25.):
                for phase in phases:
                    qa=curve.sample(phase)
                    for attack_length in (.55,.73):
                        for sign in (-1.,1.):
                            bounds=_authoring_bounds(physical,qguard,j0_deg)
                            qb,fit=solve_point(model,qguard,qa,attack_length,bounds,sign)
                            stat=static(model,data,qb,caps);at_contact=contact_state(scene,geometry,qa,qb)
                            row={'id':trial_id,'attack':name,'guard':guard,'phase_after_attack_s':phase,
                                'J0_excursion_bound_deg':j0_deg,'normal_sign':sign,'block_qpos_rad':qb.tolist(),
                                'unwrapped_joint_delta_deg':np.rad2deg(qb-qguard).tolist(),
                                'fit':fit,'block_static':stat,'contact_state':at_contact,'candidate_pass':False}
                            trial_id+=1
                            if fit['fit_pass'] and stat['max_static_cap_ratio']<1. and not any(
                                c['category'].startswith('illegal_') or c['depth_m']>.003 for c in at_contact['contacts']):
                                approach,dc=approach_check(scene,geometry,model,curve,qguard,qb,phase,caps,
                                                         speed=1.3 if focus_positive else 1.1)
                                row['approach']=approach
                                row['candidate_pass']=approach['candidate_pass']
                                if row['candidate_pass']:
                                    key='candidate_'+str(row['id'])
                                    report['actions'][key]={'kind':'defence','family':'parry_left' if name=='cut_down_left_to_right' else 'parry_right',
                                        'start_guard':guard,'end_guard':key+'_block','knot_qpos':dc.knot_qpos.tolist(),
                                        'knot_time_s':dc.knot_time_s.tolist(),'candidate_pass':True,
                                        'sampled_reference_pass':True,'screened_attack':name,
                                        'planned_start_after_attack_s':approach['planned_start_after_attack_s'],
                                        'geometry_trial_id':row['id'],'validation_stage':'partial_approach_geometry_only',
                                        'native_dynamics_tested':False,'admitted_to_training':False}
                                    report['guards'][key+'_block']={'qpos_rad':qb.tolist(),'static_reference_pass':True}
                                    np.savez_compressed(output/(key+'.npz'),knot_time_s=dc.knot_time_s,knot_qpos=dc.knot_qpos,
                                        time_s=np.linspace(0,dc.duration_s,101),qpos=dc.sample(np.linspace(0,dc.duration_s,101)))
                                    found=True
                            report['trials'].append(row);write(output/'report.json',report)
                        if found:break
                    if found:break
                if found:break
            print(name,guard,'candidate',found,flush=True)
    report['status']='bounded_other_guard_geometry_screen_complete'
    write(output/'report.json',report)
    return report


def package_return(output,candidate_id=11):
    """Bounded explicit return search for a mirrored-side intercept candidate."""
    from .a1_chain_geometry import transition_audit
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier diagnostics')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_guard_defence_screen/positive_other03/report.json'
    original=json.loads(source.read_text())
    trial=next(t for t in original['trials'] if t.get('candidate_pass') and t['id']==candidate_id)
    cons=load_joint_constraints();bounds=model_joint_ranges_rad(cons)
    model=build_limited_floor(cons).model;data=mujoco.MjData(model)
    cap=np.max(abs(model.actuator_forcerange),axis=1)
    geometry=TriangleGeometry.from_mapping({k:original['scoring'][k] for k in ('base_distance','regions')})
    scene=build_limited_duel(geometry.base_distance,initial_qpos=original['guards']['center']['qpos_rad'],constraints=cons)
    attack=original['actions']['cut_up_left_to_right'];attackend=np.array(attack['knot_qpos'][-1])
    qguard=np.array(original['guards'][trial['guard']]['qpos_rad']);qblock=np.array(trial['block_qpos_rad'])
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'searching_explicit_positive_up_return','scoring':original['scoring'],
        'guards':original['guards'],'actions':{'cut_up_left_to_right':attack},'return_trials':[],
        'candidate_only':True,'native_dynamics_tested':False,'admitted_to_training':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,snapshot)},
        'baseline_assessments':original['baseline_assessments'],
        'partial_intercept_trial':trial,
        'limitations':['Future attack endpoints are used for offline clearance authoring only.',
            'Return geometry is tested against the unperturbed held attack end; actual deflection must be retested natively.',
            'No successful defence or slot-mirror certificate is claimed.']}
    def state_clear(q):
        stat=static(model,data,q,cap)
        if stat['max_static_cap_ratio']>=1:return False
        st=contact_state(scene,geometry,attackend,q)
        return not st['contacts'] and st['minimum_blade_gap_m']>.043 and all(
            g>.025 for a in st['target_gaps_m'].values() for g in a.values())
    def edge_clear(a,b):
        return all(state_clear((1-t)*a+t*b) for t in np.linspace(0,1,17))
    rng=np.random.default_rng(8804)
    candidates=[.5*(qguard+qblock)]
    # Only a single intermediate state, bounded J0 excursion. No wrapped q.
    for i in range(450):
        sigma=np.array([.035,.30,.50,.40,.65,.48,.40])*(1. if i<200 else 1.7)
        candidates.append(.5*(qguard+qblock)+rng.normal(size=7)*sigma)
    selected=None
    for index,mid in enumerate(candidates):
        interior=bounds+np.deg2rad([3.,-3.]);interior[0]=[qguard[0]-np.deg2rad(15),qguard[0]+np.deg2rad(15)]
        if np.any(mid<interior[:,0]) or np.any(mid>interior[:,1]):continue
        okay=state_clear(mid) and edge_clear(qblock,mid) and edge_clear(mid,qguard)
        report['return_trials'].append({'index':index,'middle_qpos_rad':mid.tolist(),'coarse_clear':okay})
        if not okay:continue
        nodes=np.array([qblock,mid,qguard])
        durations=np.maximum(.8,1.9*np.max(abs(np.diff(nodes,axis=0)),axis=1)/.8)
        c=C2JointSpline(np.r_[0,np.cumsum(durations)],nodes)
        check,arrays=transition_audit(scene,geometry,model,c,attackend,cap)
        report['return_trials'][-1]['full_geometry_audit']=check
        if check['sampled_reference_pass']:
            selected=c;break
    if selected:
        start=trial['approach']['planned_start_after_attack_s'];arrival=trial['approach']['duration_s']
        hold_until=float(attack['knot_time_s'][-1])+.6-start
        times=np.r_[0.,arrival,hold_until,hold_until+selected.knot_time_s[1:]]
        nodes=np.vstack([qguard,qblock,qblock,selected.knot_qpos[1:]])
        curve=C2JointSpline(times,nodes)
        key='parry_right_from_negative_low' if candidate_id==11 else 'parry_right_from_negative_high'
        report['actions'][key]={'kind':'defence','family':'parry_right','start_guard':trial['guard'],
            'end_guard':trial['guard'],'knot_qpos':nodes.tolist(),'knot_time_s':times.tolist(),
            'candidate_pass':True,'sampled_reference_pass':True,'native_dynamics_tested':False,
            'admitted_to_training':False,'screened_attack':'cut_up_left_to_right',
            'defence_offset_s_relative_to_attack_start':start,
            'planned_contact_s_relative_to_attack_start':trial['phase_after_attack_s'],
            'return_audit':report['return_trials'][-1]['full_geometry_audit'],
            'ideal_free_space_dynamics':screen_curve(model,curve)}
        ts=np.linspace(0,curve.duration_s,int(curve.duration_s/.02)+1)
        np.savez_compressed(output/(key+'.npz'),time_s=ts,qpos=curve.sample(ts),
                            knot_time_s=times,knot_qpos=nodes)
        report['status']='frozen_complete_positive_up_defence_candidate_not_native_verified'
    else:report['status']='no_clear_single_waypoint_return_found'
    ac=C2JointSpline(attack['knot_time_s'],attack['knot_qpos']);ats=np.linspace(0,ac.duration_s,301)
    np.savez_compressed(output/'cut_up_left_to_right.npz',time_s=ats,qpos=ac.sample(ats),
                        knot_time_s=ac.knot_time_s,knot_qpos=ac.knot_qpos)
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    print(report['status'],len(report['return_trials']),flush=True)
    return report


def quiet_low_transfer(output):
    """A direct non-attacking low-guard connection, not a defence certificate."""
    from .a1_chain_geometry import transition_audit
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing transfers')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_chain_geometry/chain12/report.json';old=json.loads(source.read_text())
    geometry=TriangleGeometry.from_mapping({k:old['scoring'][k] for k in ('base_distance','regions')})
    model=build_limited_floor().model;data=mujoco.MjData(model);caps=np.max(abs(model.actuator_forcerange),axis=1)
    scene=build_limited_duel(geometry.base_distance,initial_qpos=old['guards']['center']['qpos_rad'])
    neutral=np.array(old['guards']['center']['qpos_rad'])
    qa=np.array(old['guards']['left_guard_positive']['qpos_rad'])
    qb=np.array(old['guards']['right_low_negative_robust']['qpos_rad'])
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'quiet_low_guard_transfer_search','scoring':old['scoring'],
        'guards':old['guards'],'actions':{},'trials':[],
        'candidate_only':True,'native_dynamics_tested':False,'admitted_to_training':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,snapshot)},
        'opponent_condition':'Exact source center B held; no moving threat tested.',
        'limitations':['A non-attacking guard transfer is not blocking during the transition.',
            'Approximately58deg J0 displacement is inherent in these exact endpoint states.',
            'No RL family binding, reactive window, or native contact robustness is certified.']}
    def clear(q):
        if static(model,data,q,caps)['max_static_cap_ratio']>=1:return False
        p=pair_screen(scene,geometry,q,neutral)
        return p['zero_overlap_start_pass'] and min(p['region_gaps_m']['a1'].values())>.04
    def edge(a,b):return all(clear((1-t)*a+t*b) for t in np.linspace(0,1,19))
    rng=np.random.default_rng(8821);midpoints=[None]
    physical=model_joint_ranges_rad(load_joint_constraints())
    for i in range(450):
        mid=.5*(qa+qb)+rng.normal(size=7)*[.01,.4,.25,.5,.6,.6,.45]
        mid[0]=(1-(i%7+2)/10)*qa[0]+((i%7+2)/10)*qb[0]
        midpoints.append(mid)
    selected=None
    for index,mid in enumerate(midpoints):
        nodes=np.array([qa,qb] if mid is None else [qa,mid,qb])
        if np.any(nodes<physical[:,0]) or np.any(nodes>physical[:,1]):continue
        okay=all(edge(a,b) for a,b in zip(nodes[:-1],nodes[1:]))
        trial={'index':index,'middle_qpos_rad':None if mid is None else mid.tolist(),'coarse_clear':okay}
        report['trials'].append(trial)
        if not okay:continue
        delta=abs(np.diff(nodes,axis=0))
        durations=np.maximum(.6,np.maximum(1.9*delta[:,0]/.45,1.9*np.max(delta[:,1:],axis=1)/.9))
        curve=C2JointSpline(np.r_[0,np.cumsum(durations)],nodes)
        check,arrays=transition_audit(scene,geometry,model,curve,neutral,caps)
        trial['full_geometry_audit']=check
        if check['sampled_reference_pass']:selected=curve;break
    if selected:
        for name,start,end,nodes in [
            ('switch_positive_low_to_negative_low','left_guard_positive','right_low_negative_robust',selected.knot_qpos),
            ('switch_negative_low_to_positive_low','right_low_negative_robust','left_guard_positive',selected.knot_qpos[::-1])]:
            times=selected.knot_time_s if start=='left_guard_positive' else selected.duration_s-selected.knot_time_s[::-1]
            c=C2JointSpline(times,nodes);check,arrays=transition_audit(scene,geometry,model,c,neutral,caps)
            report['actions'][name]={'kind':'transition','family':'explicit_guard_transfer','start_guard':start,
                'end_guard':end,'knot_time_s':times.tolist(),'knot_qpos':nodes.tolist(),
                'candidate_pass':check['sampled_reference_pass'],'sampled_reference_pass':check['sampled_reference_pass'],
                'geometry_audit':check,'ideal_free_space_dynamics':screen_curve(model,c),
                'not_a_defence_skill':True,'native_dynamics_tested':False,'admitted_to_training':False}
            np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=times,knot_qpos=nodes)
        report['status']='frozen_quiet_bidirectional_low_transfer_geometry_candidate'
    else:report['status']='no_quiet_low_transfer_in_bounded_search'
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    print(report['status'],len(report['trials']),flush=True)
    return report


def hold_transfer(output,right_candidate=7):
    """Explicit bridge between real zero-velocity left/right parry holds."""
    from .a1_chain_geometry import transition_audit
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous hold bridges')
    output.mkdir(parents=True)
    left_path=ROOT/'outputs/a1_short_defence/small15_geometry02_slow/report.json'
    right_path=(ROOT/'outputs/a1_guard_defence_screen/negative_high_return05/report.json' if right_candidate==7
                else ROOT/'outputs/a1_guard_defence_screen/mirrored_return04/report.json')
    source=ROOT/'outputs/a1_chain_geometry/chain12/report.json'
    left=json.loads(left_path.read_text());right=json.loads(right_path.read_text());original=json.loads(source.read_text())
    right_action='parry_right_from_negative_high' if right_candidate==7 else 'parry_right_from_negative_low'
    la=left['actions']['parry_small_J0_00_0'];ra=right['actions'][right_action]
    lc=C2JointSpline(la['knot_time_s'],la['knot_qpos']);rc=C2JointSpline(ra['knot_time_s'],ra['knot_qpos'])
    ql=lc.sample(3.91);qr=rc.sample(ra['knot_time_s'][1])
    geometry=TriangleGeometry.from_mapping({k:original['scoring'][k] for k in ('base_distance','regions')})
    model=build_limited_floor().model;data=mujoco.MjData(model);caps=np.max(abs(model.actuator_forcerange),axis=1)
    scene=build_limited_duel(geometry.base_distance,initial_qpos=original['guards']['center']['qpos_rad'])
    physical=model_joint_ranges_rad(load_joint_constraints())
    states={'neutral_B':np.array(original['guards']['center']['qpos_rad'])}
    for side,an in [('negative','cut_down_left_to_right'),('positive','cut_up_left_to_right')]:
        for phase,index in [('start',0),('end',-1)]:states[side+'_'+phase]=np.array(original['actions'][an]['knot_qpos'][index])
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'parry_hold_bridge_search','scoring':original['scoring'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,left_path,right_path,snapshot)},
        'guards':{'center':original['guards']['center'],
            'left_parry_hold':{'qpos_rad':ql.tolist()},'right_parry_hold':{'qpos_rad':qr.tolist()}},
        'actions':{},'trials':[],'candidate_only':True,'native_dynamics_tested':False,'admitted_to_training':False,
        'hold_identity':{'left_source_action':'parry_small_J0_00_0','left_time_s':3.91,
            'left_reference_velocity_rad_s':lc.derivatives([3.91])[0][0].tolist(),
            'right_source_action':right_action,'right_time_s':ra['knot_time_s'][1],
            'right_reference_velocity_rad_s':rc.derivatives([ra['knot_time_s'][1]])[0][0].tolist()},
        'opponent_static_conditions':{k:v.tolist() for k,v in states.items()},
        'limitations':['Zero reference velocity is not measured physical settling or a policy switching certificate.',
            'Static opponent poses do not establish a valid moving-threat reaction window.',
            'The right defensive hold still needs its own full native defence certificate.']}
    def timed(nodes):
        q=np.asarray(nodes);delta=abs(np.diff(q,axis=0))
        dur=np.maximum(.4,np.maximum(1.9*delta[:,0]/.45,1.9*np.max(delta[:,1:],axis=1)/.9))
        return C2JointSpline(np.r_[0,np.cumsum(dur)],q)
    direct=timed([ql,qr]);direct_audits={name:transition_audit(scene,geometry,model,direct,q,caps)[0] for name,q in states.items()}
    report['direct_route']={'knot_qpos':direct.knot_qpos.tolist(),'knot_time_s':direct.knot_time_s.tolist(),
                            'condition_audits':direct_audits}
    def state_clear(q):
        if static(model,data,q,caps)['max_static_cap_ratio']>=1:return False
        for other in states.values():
            st=contact_state(scene,geometry,other,q)
            if st['contacts'] or any(g<.035 for a in st['target_gaps_m'].values() for g in a.values()):return False
        return True
    def edge_clear(a,b):return all(state_clear((1-t)*a+t*b) for t in np.linspace(0,1,15))
    rng=np.random.default_rng(9107);selected=(direct,direct_audits) if all(
        a['sampled_reference_pass'] for a in direct_audits.values()) else None
    for i in range(350):
        if selected:break
        mid=.5*(ql+qr)+rng.normal(size=7)*[.015,.24,.28,.35,.45,.38,.32]
        if np.any(mid<physical[:,0]+.035) or np.any(mid>physical[:,1]-.035):continue
        if not min(ql[0],qr[0])-.035<=mid[0]<=max(ql[0],qr[0])+.035:continue
        coarse=state_clear(mid) and edge_clear(ql,mid) and edge_clear(mid,qr)
        row={'index':i,'middle_qpos_rad':mid.tolist(),'coarse_all_conditions_clear':coarse}
        report['trials'].append(row)
        if not coarse:continue
        curve=timed([ql,mid,qr]);audits={name:transition_audit(scene,geometry,model,curve,q,caps)[0] for name,q in states.items()}
        row['condition_audits']=audits
        if all(a['sampled_reference_pass'] for a in audits.values()):selected=(curve,audits);break
    if selected:
        curve,audits=selected
        for node in ('left_parry_hold','right_parry_hold'):
            report['guards'][node]['static_reference_pass']=True
            report['guards'][node]['scope']='Static and sampled approach evidence only; native hold not certified here.'
        for name,start,end,nodes in [('switch_left_hold_to_right_hold','left_parry_hold','right_parry_hold',curve.knot_qpos),
            ('switch_right_hold_to_left_hold','right_parry_hold','left_parry_hold',curve.knot_qpos[::-1])]:
            times=curve.knot_time_s if start=='left_parry_hold' else curve.duration_s-curve.knot_time_s[::-1]
            c=C2JointSpline(times,nodes)
            all_audits={state:transition_audit(scene,geometry,model,c,q,caps)[0] for state,q in states.items()}
            report['actions'][name]={'kind':'transition','family':'explicit_parry_hold_transfer',
                'start_guard':start,'end_guard':end,'knot_qpos':nodes.tolist(),'knot_time_s':times.tolist(),
                'candidate_pass':all(a['sampled_reference_pass'] for a in all_audits.values()),
                'sampled_reference_pass':all(a['sampled_reference_pass'] for a in all_audits.values()),
                'opponent_condition_audits':all_audits,'ideal_free_space_dynamics':screen_curve(model,c),
                'native_dynamics_tested':False,'admitted_to_training':False}
            ts=np.linspace(0,c.duration_s,201);np.savez_compressed(output/(name+'.npz'),time_s=ts,qpos=c.sample(ts),
                knot_time_s=times,knot_qpos=nodes)
        report['status']='frozen_parry_hold_bridges_static_conditions_only'
    else:report['status']='no_universal_single_waypoint_bridge_direct_conditional_only'
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    print(report['status'],len(report['trials']),flush=True)
    return report


def sync_hold_bridge(output):
    """Read-only nominal synchronization audit, not contact dynamics."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier synchronized checks')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_chain_geometry/chain12/report.json'
    bridge_source=ROOT/'outputs/a1_guard_defence_screen/qualified_hold_bridge09/report.json'
    original=json.loads(source.read_text());bridges=json.loads(bridge_source.read_text())
    geometry=TriangleGeometry.from_mapping({k:original['scoring'][k] for k in ('base_distance','regions')})
    scene=build_limited_duel(geometry.base_distance,initial_qpos=original['guards']['center']['qpos_rad'])
    prep=original['actions']['switch_negative_to_positive'];attack=original['actions']['cut_up_left_to_right']
    pc=C2JointSpline(prep['knot_time_s'],prep['knot_qpos']);ac=C2JointSpline(attack['knot_time_s'],attack['knot_qpos'])
    bridge=bridges['actions']['switch_left_hold_to_right_hold'];bc=C2JointSpline(bridge['knot_time_s'],bridge['knot_qpos'])
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'nominal_schedule_geometry_complete_not_native',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,bridge_source,snapshot)},
        'scoring':original['scoring'],'candidate_only':True,'native_dynamics_tested':False,
        'admitted_to_training':False,'scenarios':[],
        'clock_definition':'All listed times are relative to opponent preparation start. Add0.75s only if a native runner separately uses that global start.',
        'defender_bridge_start_s':.1,'defender_reference_arrival_s':.1+bc.duration_s,
        'limitations':['Future reference schedules are offline authoring evidence, not policy input.',
            'Reference penetration after initial contact does not predict dynamically deflected motion.',
            'No reaction, settling, legal-contact force or full-window defence success is certified.']}
    for pause in (0.,.1):
        horizon=pc.duration_s+pause+ac.duration_s+1.
        ts=np.linspace(0,horizon,int(np.ceil(horizon/.01))+1)
        for owner in ('a1','a2'):
            qa=[];qd=[];contacts=[];first_weapon=None;first_illegal=None;first_hit=None;first_defender_hit=None
            defender='a2' if owner=='a1' else 'a1'
            for t in ts:
                ap=pc.sample(t) if t<=pc.duration_s else ac.sample(max(0,t-pc.duration_s-pause))
                dp=bc.sample(max(0,t-.1))
                qa.append(ap);qd.append(dp)
                state=contact_state(scene,geometry,ap,dp,attacker=owner)
                for c in state['contacts']:
                    row=dict(c,time_s=float(t))
                    if c['category']=='cross_blade_blade' and first_weapon is None:first_weapon=row
                    if c['category'].startswith('illegal_') and first_illegal is None:first_illegal=row
                    if not contacts or (contacts[-1]['geoms']!=row['geoms']):contacts.append(row)
                if first_hit is None and any(g<0 for g in state['target_gaps_m'][owner].values()):first_hit=float(t)
                if first_defender_hit is None and any(g<0 for g in state['target_gaps_m'][defender].values()):first_defender_hit=float(t)
            before_contact_illegal=first_illegal is not None and (first_weapon is None or first_illegal['time_s']<=first_weapon['time_s'])
            row={'attacker_slot':owner,'extra_pre_attack_pause_s':pause,'reference_window_s':horizon,
                 'first_reference_weapon_contact':first_weapon,'first_reference_illegal_contact':first_illegal,
                 'illegal_contact_before_any_weapon_contact':before_contact_illegal,
                 'first_reference_attack_target_entry_s':first_hit,
                 'first_reference_defender_target_entry_s':first_defender_hit,
                 'bridge_arrives_before_reference_hit':first_hit is not None and .1+bc.duration_s<first_hit,
                 'all_reference_contacts':contacts,
                 'scope':'Both references imposed; later overlaps/hits cannot be scored as a physical defence outcome.'}
            report['scenarios'].append(row)
            np.savez_compressed(output/f'sync_{owner}_pause_{pause:.1f}.npz',time_s=ts,
                                attacker_qpos=np.array(qa),defender_qpos=np.array(qd))
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--focus-positive',action='store_true')
    p.add_argument('--package-return',action='store_true')
    p.add_argument('--candidate-id',type=int,choices=[7,11],default=11)
    p.add_argument('--quiet-low-transfer',action='store_true')
    p.add_argument('--hold-transfer',action='store_true')
    p.add_argument('--sync-hold-bridge',action='store_true')
    args=p.parse_args()
    if args.sync_hold_bridge:sync_hold_bridge(args.output)
    elif args.hold_transfer:hold_transfer(args.output,right_candidate=args.candidate_id)
    elif args.quiet_low_transfer:quiet_low_transfer(args.output)
    elif args.package_return:package_return(args.output,args.candidate_id)
    else:build(args.output,focus_positive=args.focus_positive)
