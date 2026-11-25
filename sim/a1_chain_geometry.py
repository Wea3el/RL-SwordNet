"""Bounded explicit switching candidates; never an inherited native certificate.

Both ends are unwrapped joint states.  A short range-preserving spline is
accepted only after whole-blade, both-slot clearance and gravity screening.
This module does not alter existing motions, rules, controllers, or evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plan1_skills.matching import tool_fk
from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .a1_triangle_candidates import region_gaps
from .balanced_orientation_geometry import static_row
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import GripAxisSolver, audit_motion, pair_screen, unit
from .triangle_task import TriangleGeometry


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), ensure_ascii=False, indent=2)+'\n')


def source_setup():
    source=ROOT/'outputs/orientation_v2_geometry/trim07/report.json'
    report=json.loads(source.read_text())
    geometry=TriangleGeometry.from_mapping({k:report['scoring'][k] for k in ('base_distance','regions')})
    constraints=load_joint_constraints()
    bounds=model_joint_ranges_rad(constraints)
    model=build_limited_floor(constraints).model
    neutral=np.asarray(report['guards']['center']['qpos_rad'])
    scene=build_limited_duel(geometry.base_distance, initial_qpos=neutral, constraints=constraints)
    caps=np.max(abs(model.actuator_forcerange),axis=1)*abs(model.actuator_gear[:,0])
    return source,report,geometry,bounds,model,neutral,scene,caps


def retime(nodes, minimum_s=1.6, speed=1.0):
    q=np.asarray(nodes)
    duration=max(minimum_s,float(1.9*np.max(np.abs(q[-1]-q[0]))/speed))
    curve=C2JointSpline(np.linspace(0,duration,len(q)),q)
    peak=max(curve.analytic_extrema()['peak_speed_rad_s'])
    if peak>speed:
        curve=C2JointSpline(curve.knot_time_s*peak/speed*(1+1e-9),q)
    return curve


def transition_audit(scene, geometry, model, curve, neutral, caps, period=.025):
    ts=np.linspace(0,curve.duration_s,int(np.ceil(curve.duration_s/period))+1)
    qs=curve.sample(ts);vs=curve.derivatives(ts)[0]
    bounds=model_joint_ranges_rad(load_joint_constraints())
    bounds_pass=bool(np.all(curve.bezier_controls>=bounds[:,0]) and np.all(curve.bezier_controls<=bounds[:,1]))
    data=mujoco.MjData(model);axes=[];grips=[];tips=[];gravity=[]
    for q in qs:
        p,a,t=tool_fk(model,data,q);grips.append(p);axes.append(a);tips.append(t)
        data.qvel[:]=0;mujoco.mj_forward(model,data);gravity.append(data.qfrc_bias.copy())
    gravity=np.asarray(gravity);slots={}
    for actor,other in [('a1','a2'),('a2','a1')]:
        all_contacts={};min_gap={name:float('inf') for name in ('left','right','upper')}
        min_other={name:float('inf') for name in ('left','right','upper')}
        weapon_gap=.2;hardware_gap=.2
        other_ids=[i for i in range(scene.model.ngeom) if scene.model.geom_contype[i]
                   and (scene.model.geom(i).name or '').startswith(other+'_')]
        blade_ids=blade_collision_geom_ids(scene.model,actor+'_')
        for t,q,v in zip(ts,qs,vs):
            scene.data.qpos[scene.agents[actor]['qpos_ids']]=q
            scene.data.qpos[scene.agents[other]['qpos_ids']]=neutral
            scene.data.qvel[:]=0;scene.data.qvel[scene.agents[actor]['dof_ids']]=v
            mujoco.mj_forward(scene.model,scene.data)
            for c in scene.data.contact:
                if c.dist<-1e-8:
                    names=[scene.model.geom(int(g)).name for g in (c.geom1,c.geom2)]
                    all_contacts.setdefault('|'.join(names),{'time_s':float(t),'geoms':names,'depth_m':float(-c.dist)})
            for name,gap in region_gaps(scene,geometry,actor).items():min_gap[name]=min(min_gap[name],gap)
            for name,gap in region_gaps(scene,geometry,other).items():min_other[name]=min(min_other[name],gap)
            for gi in blade_ids:
                for gj in other_ids:
                    gap=float(mujoco.mj_geomDistance(scene.model,scene.data,gi,gj,.2,None))
                    if 'shinai' in scene.model.geom(gj).name:weapon_gap=min(weapon_gap,gap)
                    else:hardware_gap=min(hardware_gap,gap)
        slots[actor]={'all_reference_contacts':list(all_contacts.values()),
                      'minimum_own_scoring_gaps_m':min_gap,'minimum_opponent_scoring_gaps_m':min_other,
                      'minimum_weapon_gap_m':weapon_gap,'minimum_hardware_gap_m':hardware_gap,
                      'clear_pass':bool(not all_contacts and min(min_gap.values())>.025 and
                          min(min_other.values())>.025 and weapon_gap>.04 and hardware_gap>.04)}
    gravity_ratio=np.max(abs(gravity)/caps,axis=1)
    peak=curve.analytic_extrema()
    passed=bounds_pass and all(s['clear_pass'] for s in slots.values()) and max(gravity_ratio)<1.
    axes=np.asarray(axes);tips=np.asarray(tips);grips=np.asarray(grips)
    result={'sampled_reference_pass':bool(passed),'native_dynamics_tested':False,
            'analytic_bounds_pass':bounds_pass,'slots':slots,
            'peak_speed_rad_s':peak['peak_speed_rad_s'],
            'peak_acceleration_rad_s2':peak['peak_acceleration_rad_s2'],
            'maximum_static_gravity_ratio':float(max(gravity_ratio)),
            'maximum_static_gravity_by_joint_nm':np.max(abs(gravity),axis=0).tolist(),
            'unwrapped_joint_travel_deg':np.rad2deg(np.sum(abs(np.diff(qs,axis=0)),axis=0)).tolist(),
            'axis_y_range':[float(min(axes[:,1])),float(max(axes[:,1]))],
            'shaft_angular_travel_deg':float(np.rad2deg(np.sum(np.arccos(np.clip(np.sum(axes[:-1]*axes[1:],axis=1),-1,1))))),
            'tip_path_length_m':float(np.sum(np.linalg.norm(np.diff(tips,axis=0),axis=1))),
            'grip_path_length_m':float(np.sum(np.linalg.norm(np.diff(grips,axis=0),axis=1))),
            'meaning':'Finite both-slot whole-blade clearance + static torque screen, not dynamic or defence proof.'}
    arrays={'time_s':ts,'qpos':qs,'qvel':vs,'grip_m':grips,'axis':axes,'tip_m':tips,
            'static_gravity_bias_nm':gravity}
    return result,arrays


def positive_guard_candidates(model, caps, q_start, bounds, count=20000):
    """Bounded joint-branch proposals; full FK retains the entire shaft pose.

    Randomness generates endpoint seeds, not policy actions.  Every accepted
    path is explicitly constructed and separately screened.  Refined seed
    selection keeps J2 high and J6 positive, rather than wrapping old angles.
    """
    rng=np.random.default_rng(90);data=mujoco.MjData(model);rows=[];counts={}
    margin=bounds+np.array([.04,-.04])
    for k in range(count):
        q=q_start+rng.normal(size=7)*[.35,.30,.50,.45,.65,.50,.65]
        if np.any(q<margin[:,0]) or np.any(q>margin[:,1]):
            counts['out_of_margin']=counts.get('out_of_margin',0)+1;continue
        p,a,t=tool_fk(model,data,q)
        if a[1]<.25 or a[0]<.2 or not .25<p[0]<.8 or not .35<t[2]<1.6:
            counts['not_positive_guard']=counts.get('not_positive_guard',0)+1;continue
        data.qvel[:]=0;mujoco.mj_forward(model,data);bias=data.qfrc_bias.copy();ratio=float(max(abs(bias)/caps))
        if ratio>=.97:
            counts['static_torque_margin']=counts.get('static_torque_margin',0)+1;continue
        cost=float(abs(q-q_start)@np.array([2,1,1,1,1,1,1]))
        rows.append({'sample_index':k,'qpos_rad':q.tolist(),'actual_grip_m':p.tolist(),
                     'actual_axis':a.tolist(),'actual_tip_m':t.tolist(),
                     'static_gravity_bias_nm':bias.tolist(),'max_static_gravity_ratio':ratio,
                     'weighted_unwrapped_travel':cost})
    rows.sort(key=lambda r:r['weighted_unwrapped_travel'])
    return rows,counts


def build(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh directory; do not overwrite diagnostics')
    output.mkdir(parents=True)
    source,old,geometry,bounds,model,neutral,scene,caps=source_setup()
    negative=np.array(old['guards']['left_high_negative']['qpos_rad'])
    report={'schema_version':1,'status':'bounded_guard_switch_search','scoring':old['scoring'],
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'coordinate_frame':'actor-local +X toward opponent, +Y left',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,Path(__file__),ROOT/'config/joint_constraints_observed.json')},
        'model_source_sha256':scene.source_hashes,'guards':{},'actions':{},'transition_trials':[],
        'neutral_note':'Exact shared exploratory B from trim07 for BOTH actors. It is laterally offset +0.50 m and not a final canonical guard. No route from this neutral to these high-J2 guards is implied.',
        'limitations':['No training admission or inherited native certificate.',
            'Endpoint switching only; not mid-action reaction or defence.',
            'Static gravity and sampled clearance do not ensure dynamic execution.',
            'This bounded result does not complete six attacks, three defences, or a connected training bank.']}
    data=mujoco.MjData(model)
    for name,row in old['guards'].items():
        q=np.asarray(row['qpos_rad']);m=static_row(model,data,scene,geometry,q,neutral,'guard',caps)
        report['guards'][name]={'qpos_rad':q.tolist(),**m,
            'static_reference_pass':bool(m['static_hold_under_cap'] and m['static_pair_screen']['zero_overlap_start_pass'])}
    candidates,counts=positive_guard_candidates(model,caps,negative,bounds)
    report['endpoint_proposal_counts']=counts;report['endpoint_candidates']=candidates
    write(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    selected=None
    # All physically-screened endpoints and up to twelve shortest routes are
    # retained. This is bounded, not an unlogged broad optimization grid.
    for i,row in enumerate(candidates[:12]):
        q=np.array(row['qpos_rad'])
        end=static_row(model,data,scene,geometry,q,neutral,'guard',caps)
        same=pair_screen(scene,geometry,q,q)
        if not end['static_pair_screen']['zero_overlap_start_pass'] or not same['zero_overlap_start_pass']:
            report['transition_trials'].append({'candidate_index':i,'end_checks':end,'same_guard_checks':same,
                'candidate_pass':False,'reason':'endpoint initial overlap or target overlap'})
            continue
        curve=retime([negative,q],minimum_s=2.0,speed=.9)
        audit,arrays=transition_audit(scene,geometry,model,curve,neutral,caps)
        trial={'candidate_index':i,'end_checks':end,'same_guard_checks':same,
            'knot_qpos':curve.knot_qpos.tolist(),'knot_time_s':curve.knot_time_s.tolist(),
            'audit':audit,'candidate_pass':audit['sampled_reference_pass']}
        report['transition_trials'].append(trial)
        print('switch',i,'pass',audit['sampled_reference_pass'],'axis',np.round(row['actual_axis'],3),
              'gravity',round(audit['maximum_static_gravity_ratio'],3),'duration',round(curve.duration_s,3),flush=True)
        write(output/'report.json',report)
        if audit['sampled_reference_pass']:
            selected=(row,end,curve,audit,arrays);break
    if selected:
        row,end,curve,audit,arrays=selected
        name='left_guard_positive'
        report['guards'][name]={'qpos_rad':row['qpos_rad'],**end,'static_reference_pass':True,
            'semantics':'Positive-Y whole-shaft waiting/transition guard; no defensive effectiveness assumed.'}
        for action_name,start,end_name,nodes in [
            ('switch_negative_to_positive','left_high_negative',name,curve.knot_qpos),
            ('switch_positive_to_negative',name,'left_high_negative',curve.knot_qpos[::-1])]:
            c=C2JointSpline(curve.knot_time_s,nodes)
            check,ar=transition_audit(scene,geometry,model,c,neutral,caps)
            report['actions'][action_name]={'kind':'transition','family':'explicit_guard_switch',
                'start_guard':start,'end_guard':end_name,'knot_time_s':c.knot_time_s.tolist(),
                'knot_qpos':c.knot_qpos.tolist(),'candidate_pass':check['sampled_reference_pass'],
                'sampled_reference_pass':check['sampled_reference_pass'],'audit':check,
                'validation_stage':'geometry_candidate_only','native_dynamics_tested':False,
                'admitted_to_training':False,'spline':c.report()}
            np.savez_compressed(output/(action_name+'.npz'),**ar,knot_time_s=c.knot_time_s,knot_qpos=c.knot_qpos)
        for action_name,a in old['actions'].items():
            c=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
            check,ar=audit_motion(scene,geometry,c,neutral,a['direction_intent_attacker_frame'],a['target_region'])
            report['actions'][action_name]={**a,'audit':check,'candidate_pass':check['sampled_reference_pass'],
                'sampled_reference_pass':check['sampled_reference_pass'],'native_dynamics_tested':False,
                'source_exact_motion_sha256':hashlib.sha256(json.dumps({k:a[k] for k in ('knot_qpos','knot_time_s')},sort_keys=True).encode()).hexdigest(),
                'source_native_evidence_not_inherited':True}
            np.savez_compressed(output/(action_name+'.npz'),**ar,knot_time_s=c.knot_time_s,knot_qpos=c.knot_qpos)
        report['explicit_repeated_sequence']=['cut_down_left_to_right','cut_up_right_to_left',
            'switch_negative_to_positive','switch_positive_to_negative','cut_down_left_to_right','cut_up_right_to_left']
        report['orientation_requirements']={'guard_axes':{'left_high_negative':{'y_max':-.25},
            'left_guard_positive':{'y_min':.25},'center':{'x_min':.85,'abs_y_max':.2}}}
        report['status']='frozen_bidirectional_switch_geometry_candidate'
    else:report['status']='no_passing_direct_guard_switch_in_bounded_candidates'
    write(output/'report.json',report)
    return report


def positive_contacts(output):
    """A bounded full-grip/axis contact search on the bridge's high-J2 branch."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior attempts')
    output.mkdir(parents=True)
    source,old,geometry,bounds,model,neutral,scene,caps=source_setup()
    switch_path=ROOT/'outputs/a1_chain_geometry/switch01/report.json'
    switch=json.loads(switch_path.read_text())
    positive=np.asarray(switch['guards']['left_guard_positive']['qpos_rad'])
    # This search explicitly stays on the high-J2 interval. It does not wrap
    # the old positive attack's ~195 degree coordinates into a new trajectory.
    local=bounds+np.array([.035,-.035]);local[2,0]=4.7
    solver=GripAxisSolver(model,local)
    seeds=[positive,np.asarray(old['guards']['left_high_negative']['qpos_rad']),
           np.asarray(old['guards']['right_low_negative']['qpos_rad'])]
    forward=json.loads((ROOT/'outputs/multiguard_v1_geometry/guards04/report.json').read_text())
    seeds.append(np.asarray(forward['guards']['center']['qpos_rad']))
    report={'schema_version':1,'status':'bounded_positive_contact_search','scoring':old['scoring'],
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,switch_path,Path(__file__))},
        'joint2_search_interval_rad':local[2].tolist(),'trials':[],'contacts':[],
        'guards':switch['guards'],'actions':{},'neutral_note':switch['neutral_note']}
    data=mujoco.MjData(model)
    for region in ('right','left'):
        target=np.asarray(geometry.world_center(region,'a2'))+[-.055,0,0]
        for ay in (.30,.50,.70):
            for az in (.25,.50):
                axis=unit([.8,ay,az])
                for distance in (.73,.78):
                    p=target-distance*axis
                    for si,seed in enumerate(seeds):
                        q,fit=solver.solve(p,axis,seed,regularization=.003,iterations=130)
                        row={'region':region,'requested_grip_m':p.tolist(),'requested_axis':axis.tolist(),
                            'seed_index':si,'qpos_rad':q.tolist(),**fit,'candidate_pass':False}
                        if fit['fit_pass']:
                            row.update(static_row(model,data,scene,geometry,q,neutral,'contact',caps))
                            gap=row['static_pair_screen']['region_gaps_m']['a1'][region]
                            row['actual_requested_region_gap_m']=gap
                            row['candidate_pass']=bool(gap<-.005 and row['no_physical_overlap'] and
                                                       row['static_hold_under_cap'])
                            if row['candidate_pass']:
                                row['distance_from_positive_guard_rad']=float(np.linalg.norm(q-positive))
                                report['contacts'].append(row)
                        report['trials'].append(row)
                    write(output/'report.json',report)
            print('contacts',region,ay,'passing',len(report['contacts']),flush=True)
    report['status']='positive_contact_search_complete'
    write(output/'report.json',report)
    return report


def recondition_negative(output, extra_margin=False):
    """Re-solve the complete task path with interior J2 margin, never clip it."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior candidates')
    output.mkdir(parents=True)
    source,old,geometry,bounds,model,neutral,scene,caps=source_setup()
    data=mujoco.MjData(model)
    # Freeze the generator bytes alongside this newly re-solved candidate.
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'interior_J2_complete_path_refit','scoring':old['scoring'],
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,snapshot)},
        'model_source_sha256':scene.source_hashes,'guards':{},'actions':{},'refit_trials':[],
        'neutral_note':old.get('neutral_note','Exact shared exploratory B, not final canonical neutral.'),
        'limitations':['Newly fitted paths have no inherited dynamic/contact certificate.',
            'Interior J2 margin is necessary but not a guarantee of contact robustness.',
            'No defence or repeated-switching admission is implied.']}
    neutral_metrics=static_row(model,data,scene,geometry,neutral,neutral,'guard',caps)
    report['guards']['center']={'qpos_rad':neutral.tolist(),**neutral_metrics,
        'static_reference_pass':neutral_metrics['static_pair_screen']['zero_overlap_start_pass'] and neutral_metrics['static_hold_under_cap']}
    action=old['actions']['cut_down_left_to_right']
    oldq=np.array(action['knot_qpos']);times=np.array(action['knot_time_s'])
    desired=[tool_fk(model,qpos=q)[:2] for q in oldq]
    selected=None
    report['extra_margin_request']={'J5_min_deg':20.,'J6_max_deg':85.} if extra_margin else None
    for ceiling in ((350.,) if extra_margin else (350.,345.)):
        local=bounds+np.array([.03,-.03]);local[2,1]=np.deg2rad(ceiling)
        if extra_margin:
            local[5,0]=np.deg2rad(20.)
            local[6,1]=np.deg2rad(85.)
        solver=GripAxisSolver(model,local);nodes=[];fits=[];q=oldq[0]
        for i,(p,axis) in enumerate(desired):
            seed=q if i else oldq[0]
            q,fit=solver.solve(p,axis,seed,regularization=.0015,iterations=160)
            fits.append({'index':i,'requested_grip_m':p.tolist(),'requested_axis':axis.tolist(),
                         'qpos_rad':q.tolist(),**fit});nodes.append(q)
        curve=C2JointSpline(times,nodes)
        audit,arrays=audit_motion(scene,geometry,curve,neutral,[0,-1,-1],'left')
        endpoint_metrics=[static_row(model,data,scene,geometry,q,neutral,'guard',caps) for q in (nodes[0],nodes[-1])]
        allbias=[]
        for q in nodes:
            data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data);allbias.append(data.qfrc_bias.copy())
        maximum_ratio=float(np.max(abs(np.asarray(allbias))/caps))
        passed=all(f['fit_pass'] for f in fits) and audit['sampled_reference_pass'] and all(
            m['static_hold_under_cap'] and m['static_pair_screen']['zero_overlap_start_pass'] for m in endpoint_metrics)
        trial={'J2_ceiling_deg':ceiling,'fits':fits,'knot_qpos':np.asarray(nodes).tolist(),
               'knot_time_s':times.tolist(),'audit':audit,'endpoint_metrics':endpoint_metrics,
               'max_knot_static_gravity_ratio':maximum_ratio,'candidate_pass':bool(passed)}
        report['refit_trials'].append(trial)
        print('refit',ceiling,'fits',sum(f['fit_pass'] for f in fits),'/',len(fits),
              'geometry',audit['sampled_reference_pass'],'endhold',all(m['static_hold_under_cap'] for m in endpoint_metrics),
              'maxratio',round(maximum_ratio,3),flush=True)
        write(output/'report.json',report)
        if passed:selected=(ceiling,curve,endpoint_metrics)
    if selected:
        ceiling,curve,metrics=selected
        report['selected_J2_ceiling_deg']=ceiling
        for name,q,metric in zip(('left_high_negative_interior','right_low_negative_interior'),
                                 (curve.knot_qpos[0],curve.knot_qpos[-1]),metrics):
            report['guards'][name]={'qpos_rad':q.tolist(),**metric,'static_reference_pass':True}
        for name,start,end,q,direction in [
            ('cut_down_left_to_right','left_high_negative_interior','right_low_negative_interior',curve.knot_qpos,[0,-1,-1]),
            ('cut_up_right_to_left','right_low_negative_interior','left_high_negative_interior',curve.knot_qpos[::-1],[0,1,1])]:
            c=C2JointSpline(times,q);check,ar=audit_motion(scene,geometry,c,neutral,direction,'left')
            report['actions'][name]={'kind':'attack','family':name,'start_guard':start,'end_guard':end,
                'target_region':'left','direction_intent_attacker_frame':direction,
                'knot_qpos':q.tolist(),'knot_time_s':times.tolist(),'audit':check,
                'candidate_pass':check['sampled_reference_pass'],'sampled_reference_pass':check['sampled_reference_pass'],
                'native_dynamics_tested':False,'admitted_to_training':False,'spline':c.report()}
            np.savez_compressed(output/(name+'.npz'),**ar,knot_time_s=times,knot_qpos=q)
        report['status']='frozen_interior_J2_geometry_candidate'
    else:report['status']='interior_J2_refit_failed_geometry'
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


def rebridge(output):
    """Bind the interior-margin attacks to an explicit positive-shaft guard."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior results')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_chain_geometry/interior03/report.json'
    positive_source=ROOT/'outputs/a1_chain_geometry/switch01/report.json'
    report=json.loads(source.read_text())
    old_positive=json.loads(positive_source.read_text())['guards']['left_guard_positive']
    _,_,geometry,bounds,model,neutral,scene,caps=source_setup()
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report['parent_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (source,positive_source,snapshot)}
    qnegative=np.array(report['guards']['left_high_negative_interior']['qpos_rad'])
    qpositive=np.array(old_positive['qpos_rad'])
    positive_metrics=static_row(model,mujoco.MjData(model),scene,geometry,qpositive,neutral,'guard',caps)
    report['guards']['left_guard_positive']={'qpos_rad':qpositive.tolist(),**positive_metrics,
        'static_reference_pass':bool(positive_metrics['static_hold_under_cap'] and
                                     positive_metrics['static_pair_screen']['zero_overlap_start_pass']),
        'semantics':'Positive-Y waiting and switching guard, not an established defensive skill.'}
    report['bridge_trials']=[]
    for name,start,end,nodes in [
        ('switch_negative_to_positive','left_high_negative_interior','left_guard_positive',[qnegative,qpositive]),
        ('switch_positive_to_negative','left_guard_positive','left_high_negative_interior',[qpositive,qnegative])]:
        curve=retime(nodes,minimum_s=2.0,speed=.9)
        check,arrays=transition_audit(scene,geometry,model,curve,neutral,caps)
        passed=bool(check['sampled_reference_pass'] and report['guards']['left_guard_positive']['static_reference_pass'])
        report['bridge_trials'].append({'name':name,'audit':check,'candidate_pass':passed})
        if passed:
            report['actions'][name]={'kind':'transition','family':'explicit_guard_switch',
                'start_guard':start,'end_guard':end,'knot_time_s':curve.knot_time_s.tolist(),
                'knot_qpos':curve.knot_qpos.tolist(),'candidate_pass':passed,
                'sampled_reference_pass':passed,'audit':check,'validation_stage':'geometry_candidate_only',
                'native_dynamics_tested':False,'admitted_to_training':False,'spline':curve.report()}
            np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        print('rebridge',name,passed,'gravity',check['maximum_static_gravity_ratio'],flush=True)
    for name,action in report['actions'].items():
        if action['kind']!='attack':continue
        curve=C2JointSpline(action['knot_time_s'],action['knot_qpos'])
        check,arrays=audit_motion(scene,geometry,curve,neutral,action['direction_intent_attacker_frame'],action['target_region'])
        action['audit']=check;action['candidate_pass']=check['sampled_reference_pass'];action['sampled_reference_pass']=check['sampled_reference_pass']
        np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    report['explicit_repeated_sequence']=['switch_positive_to_negative','cut_down_left_to_right',
        'cut_up_right_to_left','switch_negative_to_positive']
    report['orientation_requirements']={'guard_axes':{
        'left_high_negative_interior':{'y_max':-.25},'left_guard_positive':{'y_min':.25},
        'center':{'x_min':.85,'abs_y_max':.2}}}
    report['status']='frozen_interior_J2_and_bidirectional_switch_candidates'
    report['limitations'] += ['Positive-Y guard has no validated positive-Y attack follow-on yet.',
        'The declared repeated sequence is a diagnostic request, not a passed native loop.']
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


def positive_blade_contacts(output):
    """Optimize actual shaft material-point contact, not an arbitrary grip pose.

    Shaft direction is a broad positive-Y half-space constraint.  All point
    fits, collisions, cap failures, and unsuccessful attempts remain evidence.
    A static contact candidate is not exported as an attack action.
    """
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior results')
    output.mkdir(parents=True)
    source,old,geometry,bounds,model,neutral,scene,caps=source_setup()
    bridge_path=ROOT/'outputs/a1_chain_geometry/bridge04/report.json'
    previous=json.loads(bridge_path.read_text());positive=np.array(previous['guards']['left_guard_positive']['qpos_rad'])
    historical=json.loads((ROOT/'outputs/multiguard_v1_geometry/bundle13/report.json').read_text())
    local=bounds+np.array([.035,-.035])
    local[2,1]=np.deg2rad(350.)
    base_seeds=[positive,np.array(previous['guards']['left_high_negative_interior']['qpos_rad'])]
    for action_name in ('cut_down_left_to_right','cut_down_right_to_left'):
        qs=historical['actions'][action_name]['knot_qpos'];base_seeds.append(np.array(qs[len(qs)//2]))
    rng=np.random.default_rng(413)
    seeds=base_seeds+[positive+rng.normal(size=7)*[.3,.25,.7,.3,.4,.3,.4] for _ in range(4)]
    data=mujoco.MjData(model)
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'actual_blade_point_positive_heading_search',
        'scoring':old['scoring'],'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (source,bridge_path,snapshot)},
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'guards':previous['guards'],'actions':{},'trials':[],'contacts':[],
        'meaning':'Static actual whole-shaft contacts only. No complete attack or inherited certificate.'}
    for region in ('right','upper','left'):
        center=np.array(geometry.world_center(region,'a2'))
        target=center+[-.04,0,0]
        for index,seed in enumerate(seeds):
            seed=np.minimum(np.maximum(seed,local[:,0]+1e-6),local[:,1]-1e-6)
            initial=np.r_[seed,.73]
            def residual(v):
                q=v[:7];length=v[7];p,axis,_=tool_fk(model,data,q)
                point=p+length*axis
                data.qvel[:]=0;mujoco.mj_forward(model,data)
                overload=np.maximum(abs(data.qfrc_bias)/caps-.95,0.)
                return np.r_[8.*(point-target),
                    2.*max(.22-axis[1],0),2.*max(.45-axis[0],0),
                    .08*overload,.004*np.array([2,1,1,1,1,1,1])*(q-seed)]
            result=least_squares(residual,initial,bounds=(np.r_[local[:,0],.40],np.r_[local[:,1],.785]),
                                 max_nfev=160,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            q=result.x[:7];length=result.x[7];metrics=static_row(model,data,scene,geometry,q,neutral,'contact',caps)
            p,axis,_=tool_fk(model,qpos=q);point=p+length*axis
            error=float(np.linalg.norm(point-target));gap=metrics['static_pair_screen']['region_gaps_m']['a1'][region]
            passed=bool(error<.012 and axis[1]>=.18 and axis[0]>=.40 and gap<-.005
                        and metrics['no_physical_overlap'] and metrics['static_hold_under_cap'])
            row={'region':region,'seed_index':index,'qpos_rad':q.tolist(),**metrics,
                'requested_material_point_m':target.tolist(),'actual_material_point_m':point.tolist(),
                'material_distance_m':float(length),'material_point_error_m':error,
                'actual_requested_region_gap_m':gap,'candidate_pass':passed,
                'unwrapped_distance_from_positive_guard':float(np.linalg.norm(q-positive))}
            report['trials'].append(row)
            if passed:report['contacts'].append(row)
            write(output/'report.json',report)
            print('material-contact',region,index,passed,'error',round(error,4),'axis',np.round(axis,3),
                  'ratio',round(metrics['max_static_gravity_ratio'],3),flush=True)
    report['status']='bounded_positive_blade_contact_search_complete'
    write(output/'report.json',report)
    return report


def positive_sweep(output):
    """Grow one genuine positive-shaft upward cut from actual contact evidence."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior attempts')
    output.mkdir(parents=True)
    _,_,geometry,bounds,model,neutral,scene,caps=source_setup()
    source=ROOT/'outputs/a1_chain_geometry/material06/report.json'
    bridge_source=ROOT/'outputs/a1_chain_geometry/bridge04/report.json'
    contacts=json.loads(source.read_text());report=json.loads(bridge_source.read_text())
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report['positive_sweep_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (source,bridge_source,snapshot)}
    report['positive_sweep_trials']=[]
    solver=GripAxisSolver(model,bounds+np.array([.03,-.03]));data=mujoco.MjData(model)
    qguard=np.array(report['guards']['left_guard_positive']['qpos_rad'])
    selected=None
    candidates=[c for c in contacts['contacts'] if c['region']=='right' and c['seed_index'] in (0,1)]
    for source_index,row in enumerate(candidates):
        qc=np.array(row['qpos_rad']);p0,a0,_=tool_fk(model,qpos=qc)
        for scale in (.6,.85,1.):
            fits={0.:{'qpos_rad':qc.tolist(),'fit_pass':True}}
            for sign in (-1.,1.):
                q=qc.copy()
                for phase in sign*np.linspace(.125,1.,8):
                    s=phase*scale
                    p=p0+[-.18*s*s,-.20*s,.12*s]
                    # Gentle pitch change leaves a continuous redundant IK
                    # branch. The earlier .18*s request drove J5 into its stop
                    # and caused a 29-degree J6 null-space jump after contact.
                    axis=unit(a0+[0,-.02*s,.06*s])
                    q,fit=solver.solve(p,axis,q,regularization=.003,iterations=140)
                    fits[float(phase)]={'qpos_rad':q.tolist(),'requested_grip_m':p.tolist(),
                                       'requested_axis':axis.tolist(),**fit}
            rows=[fits[k] for k in sorted(fits)];core=np.array([r['qpos_rad'] for r in rows])
            # The existing positive waiting guard is an explicit entry node.
            # Preparation and strike are part of this recorded full reference.
            nodes=np.vstack([qguard,core])
            ts=np.r_[0.,1.8+np.linspace(0,2.4,len(core))]
            curve=C2JointSpline(ts,nodes)
            peak=max(curve.analytic_extrema()['peak_speed_rad_s'])
            if peak>1.1:curve=C2JointSpline(ts*peak/1.1,nodes)
            audit,arrays=audit_motion(scene,geometry,curve,neutral,[0,-1,1],'right')
            end=static_row(model,data,scene,geometry,core[-1],neutral,'guard',caps)
            biases=[]
            for q in nodes:
                data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data);biases.append(data.qfrc_bias.copy())
            axes=np.array([tool_fk(model,qpos=q)[1] for q in arrays['qpos']])
            natural_candidate=bool(curve.duration_s<=6. and np.max(abs(np.diff(core,axis=0)))<.15)
            passed=bool(natural_candidate and all(r['fit_pass'] for r in rows) and audit['sampled_reference_pass'] and
                        end['static_hold_under_cap'] and end['static_pair_screen']['zero_overlap_start_pass'] and
                        min(axes[:,1])>.15)
            trial={'contact_source_index':source_index,'scale':scale,'fits':rows,'audit':audit,
                'knot_qpos':nodes.tolist(),'knot_time_s':curve.knot_time_s.tolist(),
                'end_checks':end,'axis_y_range':[float(min(axes[:,1])),float(max(axes[:,1]))],
                'maximum_knot_static_gravity_ratio':float(np.max(abs(np.asarray(biases))/caps)),
                'duration_s':curve.duration_s,'bounded_timing_and_local_joint_continuity_pass':natural_candidate,
                'maximum_adjacent_core_joint_delta_rad':float(np.max(abs(np.diff(core,axis=0)))),
                'candidate_pass':passed}
            report['positive_sweep_trials'].append(trial)
            print('positive-sweep',source_index,scale,'fits',sum(r['fit_pass'] for r in rows),
                  'geometry',audit['sampled_reference_pass'],'endhold',end['static_hold_under_cap'],
                  'Y',np.round(trial['axis_y_range'],3),'PASS',passed,flush=True)
            write(output/'report.json',report)
            if passed:
                selected=(curve,audit,arrays,end,trial);break
        if selected:break
    if selected:
        curve,audit,arrays,end,trial=selected
        report['guards']['right_high_positive']={'qpos_rad':curve.knot_qpos[-1].tolist(),**end,
                                                 'static_reference_pass':True}
        action='cut_up_left_to_right'
        report['actions'][action]={'kind':'attack','family':action,'start_guard':'left_guard_positive',
            'end_guard':'right_high_positive','target_region':'right',
            'direction_intent_attacker_frame':[0,-1,1],'knot_qpos':curve.knot_qpos.tolist(),
            'knot_time_s':curve.knot_time_s.tolist(),'candidate_pass':True,'sampled_reference_pass':True,
            'audit':audit,'native_dynamics_tested':False,'admitted_to_training':False,
            'reference_preparation_ends_s':float(curve.knot_time_s[1]),
            'reference_contact_knot_s':float(curve.knot_time_s[9]),
            'axis_y_range':trial['axis_y_range'],'spline':curve.report()}
        np.savez_compressed(output/(action+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        # Explicit exit path is separately audited, not a hidden reset.
        back=retime([curve.knot_qpos[-1],qguard],minimum_s=2.,speed=.9)
        backaudit,backarrays=transition_audit(scene,geometry,model,back,neutral,caps)
        report['positive_recovery_audit']=backaudit
        if backaudit['sampled_reference_pass']:
            report['actions']['recover_positive_high_to_low']={'kind':'transition','family':'explicit_guard_recovery',
                'start_guard':'right_high_positive','end_guard':'left_guard_positive',
                'knot_qpos':back.knot_qpos.tolist(),'knot_time_s':back.knot_time_s.tolist(),
                'candidate_pass':True,'sampled_reference_pass':True,'audit':backaudit,
                'native_dynamics_tested':False,'admitted_to_training':False,'spline':back.report()}
            np.savez_compressed(output/'recover_positive_high_to_low.npz',**backarrays,
                                knot_time_s=back.knot_time_s,knot_qpos=back.knot_qpos)
        report['status']='frozen_positive_shaft_upcut_geometry_candidate'
    else:report['status']='no_positive_sweep_pass_in_bounded_trials'
    # Copy exact inherited references through audited sampling, not native certs.
    for name,action in report['actions'].items():
        if (output/(name+'.npz')).exists():continue
        curve=C2JointSpline(action['knot_time_s'],action['knot_qpos'])
        if action['kind']=='attack':
            audit,arrays=audit_motion(scene,geometry,curve,neutral,action['direction_intent_attacker_frame'],action['target_region'])
        else:audit,arrays=transition_audit(scene,geometry,model,curve,neutral,caps)
        action['audit']=audit;action['candidate_pass']=audit['sampled_reference_pass'];action['sampled_reference_pass']=audit['sampled_reference_pass']
        np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


def robust_negative(output):
    """Refit material-point motion with contact-tolerant interior joint limits.

    Exact historical grip/shaft constraints are not mandatory.  The old shaft
    material point defines the cut shape; orientation is only a soft prior
    plus an independently checked negative-Y heading requirement.
    """
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior results')
    output.mkdir(parents=True)
    source,old,geometry,bounds,model,neutral,scene,caps=source_setup()
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    physical=model_joint_ranges_rad(load_joint_constraints())
    interior=physical+np.deg2rad(np.array([5.,-5.]))
    interior[2]=np.deg2rad([10.,350.]);interior[3]=np.deg2rad([-15.,115.])
    interior[5]=np.deg2rad([23.,137.]);interior[6]=np.deg2rad([-82.,82.])
    old_action=old['actions']['cut_down_left_to_right']
    oldq=np.array(old_action['knot_qpos']);times=np.array(old_action['knot_time_s'])
    old_fk=[tool_fk(model,qpos=q) for q in oldq]
    reference_length=.74
    points=[p+reference_length*a for p,a,_ in old_fk]
    report={'schema_version':1,'status':'robust_negative_material_point_search',
        'scoring':old['scoring'],'guards':{},'actions':{},'robust_refit_trials':[],
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,snapshot)},
        'model_source_sha256':scene.source_hashes,'interior_joint_bounds_deg':np.rad2deg(interior).tolist(),
        'reference_material_distance_m':reference_length,
        'scope':'Actual shaft-material-point path, broad shaft direction, unchanged contact physics and limits.',
        'limitations':['Geometry-only candidates; no inherited native or defence certificate.',
            'Endpoint guards and every bridge change and need independent validation.',
            'Interior reference margins do not guarantee bounded actual q after collision.']}
    data=mujoco.MjData(model)
    m=static_row(model,data,scene,geometry,neutral,neutral,'guard',caps)
    report['guards']['center']={'qpos_rad':neutral.tolist(),**m,
        'static_reference_pass':m['static_hold_under_cap'] and m['static_pair_screen']['zero_overlap_start_pass']}
    selected=None
    for axis_weight,regularizer in ((.30,.02),(.15,.02),(.30,.006)):
        qprevious=oldq[0];length=reference_length;fits=[];nodes=[]
        for i,(target,(_,old_axis,_)) in enumerate(zip(points,old_fk)):
            seed=qprevious.copy()
            initial=np.r_[np.minimum(np.maximum(seed,interior[:,0]+1e-8),interior[:,1]-1e-8),length]
            def residual(v):
                q=v[:7];ell=v[7];p,a,_=tool_fk(model,data,q);point=p+ell*a
                data.qvel[:]=0;mujoco.mj_forward(model,data)
                excess=np.maximum(abs(data.qfrc_bias)/caps-.94,0.)
                return np.r_[8.*(point-target),axis_weight*(a-old_axis),
                    2.*max(a[1]+.35,0.),2.*max(.5-a[0],0.),
                    .15*excess,regularizer*np.array([2,1,1,1,1,1,1])*(q-seed),.015*(ell-reference_length)]
            fit=least_squares(residual,initial,bounds=(np.r_[interior[:,0],.48],np.r_[interior[:,1],.785]),
                max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            qprevious=fit.x[:7];length=float(fit.x[7]);p,a,tip=tool_fk(model,qpos=qprevious)
            error=float(np.linalg.norm(p+length*a-target))
            fits.append({'index':i,'qpos_rad':qprevious.tolist(),'material_distance_m':length,
                'material_point_error_m':error,'desired_material_point_m':target.tolist(),
                'actual_material_point_m':(p+length*a).tolist(),'actual_grip_m':p.tolist(),
                'actual_axis':a.tolist(),'actual_tip_m':tip.tolist(),
                'fit_pass':bool(error<.012 and a[1]<-.35 and a[0]>.50)})
            nodes.append(qprevious.copy())
        curve=C2JointSpline(times,nodes)
        audit,arrays=audit_motion(scene,geometry,curve,neutral,[0,-1,-1],'left')
        endpoint_metrics=[static_row(model,data,scene,geometry,q,neutral,'guard',caps) for q in (nodes[0],nodes[-1])]
        axes=np.array([tool_fk(model,qpos=q)[1] for q in arrays['qpos']])
        margins=np.minimum(np.array(nodes)-physical[:,0],physical[:,1]-np.array(nodes))
        bounded=bool(np.all(curve.bezier_controls>=interior[:,0]) and np.all(curve.bezier_controls<=interior[:,1]))
        passed=bool(all(f['fit_pass'] for f in fits) and audit['sampled_reference_pass'] and bounded and
            max(axes[:,1])<-.25 and all(e['static_hold_under_cap'] and
                                       e['static_pair_screen']['zero_overlap_start_pass'] for e in endpoint_metrics))
        trial={'axis_prior_weight':axis_weight,'joint_regularizer':regularizer,'fits':fits,
            'knot_qpos':np.array(nodes).tolist(),'knot_time_s':times.tolist(),'audit':audit,
            'endpoint_metrics':endpoint_metrics,'analytic_interior_bounds_pass':bounded,
            'minimum_reference_joint_margin_deg':np.rad2deg(np.min(margins,axis=0)).tolist(),
            'axis_y_range':[float(min(axes[:,1])),float(max(axes[:,1]))],
            'candidate_pass':passed}
        report['robust_refit_trials'].append(trial)
        print('robust',axis_weight,regularizer,'fits',sum(f['fit_pass'] for f in fits),
            'geometry',audit['sampled_reference_pass'],'Y',np.round(trial['axis_y_range'],3),
            'endholds',[e['static_hold_under_cap'] for e in endpoint_metrics],'PASS',passed,flush=True)
        write(output/'report.json',report)
        if passed:selected=(curve,endpoint_metrics,trial);break
    if selected:
        curve,metrics,trial=selected
        for name,q,metric in zip(('left_high_negative_robust','right_low_negative_robust'),
                                 (curve.knot_qpos[0],curve.knot_qpos[-1]),metrics):
            report['guards'][name]={'qpos_rad':q.tolist(),**metric,'static_reference_pass':True}
        for name,start,end,qs,direction in [
            ('cut_down_left_to_right','left_high_negative_robust','right_low_negative_robust',curve.knot_qpos,[0,-1,-1]),
            ('cut_up_right_to_left','right_low_negative_robust','left_high_negative_robust',curve.knot_qpos[::-1],[0,1,1])]:
            c=C2JointSpline(times,qs);check,ar=audit_motion(scene,geometry,c,neutral,direction,'left')
            report['actions'][name]={'kind':'attack','family':name,'start_guard':start,'end_guard':end,
                'target_region':'left','direction_intent_attacker_frame':direction,
                'knot_qpos':qs.tolist(),'knot_time_s':times.tolist(),'candidate_pass':check['sampled_reference_pass'],
                'sampled_reference_pass':check['sampled_reference_pass'],'audit':check,
                'native_dynamics_tested':False,'admitted_to_training':False,'spline':c.report()}
            np.savez_compressed(output/(name+'.npz'),**ar,knot_time_s=times,knot_qpos=qs)
        report['status']='frozen_robust_negative_material_path_candidate'
    else:report['status']='no_robust_negative_candidate_in_bounded_search'
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


def robust_chain(output, positive_version='positive08'):
    """Bind only explicit rechecked routes to the new robust negative guards."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous chains')
    output.mkdir(parents=True)
    robust_path=ROOT/'outputs/a1_chain_geometry/robust09/report.json'
    positive_path=ROOT/'outputs/a1_chain_geometry'/positive_version/'report.json'
    report=json.loads(robust_path.read_text());positive=json.loads(positive_path.read_text())
    _,_,geometry,_,model,neutral,scene,caps=source_setup()
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report['chain_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (robust_path,positive_path,snapshot)}
    for name in ('cut_up_left_to_right','recover_positive_high_to_low'):
        report['actions'][name]=positive['actions'][name]
        for key in ('start_guard','end_guard'):
            node=positive['actions'][name][key]
            report['guards'][node]=positive['guards'][node]
    if 'entry_exception' in positive:report['positive_entry_exception']=positive['entry_exception']
    qnegative=np.array(report['guards']['left_high_negative_robust']['qpos_rad'])
    qpositive=np.array(report['guards']['left_guard_positive']['qpos_rad'])
    report['bridge_trials']=[]
    for name,start,end,nodes in [
        ('switch_negative_to_positive','left_high_negative_robust','left_guard_positive',[qnegative,qpositive]),
        ('switch_positive_to_negative','left_guard_positive','left_high_negative_robust',[qpositive,qnegative])]:
        curve=retime(nodes,minimum_s=2.,speed=.9)
        audit,arrays=transition_audit(scene,geometry,model,curve,neutral,caps)
        report['bridge_trials'].append({'name':name,'audit':audit,'candidate_pass':audit['sampled_reference_pass']})
        if audit['sampled_reference_pass']:
            report['actions'][name]={'kind':'transition','family':'explicit_guard_switch',
                'start_guard':start,'end_guard':end,'knot_qpos':curve.knot_qpos.tolist(),
                'knot_time_s':curve.knot_time_s.tolist(),'candidate_pass':True,'sampled_reference_pass':True,
                'audit':audit,'native_dynamics_tested':False,'admitted_to_training':False,'spline':curve.report()}
        print('robust-bridge',name,audit['sampled_reference_pass'],flush=True)
    for name,action in report['actions'].items():
        curve=C2JointSpline(action['knot_time_s'],action['knot_qpos'])
        if action['kind']=='attack':
            audit,arrays=audit_motion(scene,geometry,curve,neutral,action['direction_intent_attacker_frame'],action['target_region'])
        else:audit,arrays=transition_audit(scene,geometry,model,curve,neutral,caps)
        action['audit']=audit;action['candidate_pass']=audit['sampled_reference_pass'];action['sampled_reference_pass']=audit['sampled_reference_pass']
        action['native_dynamics_tested']=False;action['admitted_to_training']=False
        np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    report['explicit_alternating_sequence']=['cut_up_left_to_right','recover_positive_high_to_low',
        'switch_positive_to_negative','cut_down_left_to_right','cut_up_right_to_left','switch_negative_to_positive']
    report['status']='frozen_robust_negative_and_positive_sequence_geometry'
    report['limitations'] += ['Positive waiting/preparation entry and fixed neutral are not certified under all strike interior margins.',
        'Transition names are diagnostic edges, not additions to the approved nine policy family IDs.']
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


def robust_positive(output):
    """Retain the explicit entry, interiorize the entire positive strike phase."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier native evidence')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_chain_geometry/positive08/report.json'
    old=json.loads(source.read_text())
    _,_,geometry,physical,model,neutral,scene,caps=source_setup()
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    interior=physical+np.deg2rad(np.array([5.,-5.]))
    interior[2]=np.deg2rad([10.,350.]);interior[3]=np.deg2rad([-15.,115.])
    interior[5]=np.deg2rad([23.,137.]);interior[6]=np.deg2rad([-82.,82.])
    old_action=old['actions']['cut_up_left_to_right']
    oldq=np.array(old_action['knot_qpos']);times=np.array(old_action['knot_time_s'])
    qentry=oldq[0].copy();coreold=oldq[1:]
    old_fk=[tool_fk(model,qpos=q) for q in coreold]
    material_distance=.74;points=[p+material_distance*a for p,a,_ in old_fk]
    report={'schema_version':1,'status':'robust_positive_material_point_search','scoring':old['scoring'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,snapshot)},
        'model_source_sha256':scene.source_hashes,'candidate_only':True,'admitted_to_training':False,
        'native_dynamics_tested':False,'guards':{},'actions':{},'robust_refit_trials':[],
        'interior_strike_bounds_deg':np.rad2deg(interior).tolist(),
        'entry_exception':'Existing positive guard is preserved exactly (J5=19.3975 deg); only preparation may be below the new23-degree J5 margin. Strike and exit guard are interior.',
        'reference_preparation_ends_s':float(times[1]),
        'limitations':['Not all-phase contact robust: common entry remains below the new strike margin.',
            'Material-point refit is geometry evidence only; paired native contact test required.',
            'No old positive strike or return certificate is inherited.']}
    data=mujoco.MjData(model)
    for name,q in [('center',neutral),('left_guard_positive',qentry)]:
        check=static_row(model,data,scene,geometry,q,neutral,'guard',caps)
        report['guards'][name]={'qpos_rad':q.tolist(),**check,
            'static_reference_pass':check['static_hold_under_cap'] and check['static_pair_screen']['zero_overlap_start_pass']}
    selected=None
    for axis_weight,regularizer in ((.30,.02),(.15,.02),(.30,.006)):
        seed=coreold[0].copy();length=material_distance;fits=[];nodes=[]
        for i,(target,(_,old_axis,_)) in enumerate(zip(points,old_fk)):
            initial=np.r_[np.minimum(np.maximum(seed,interior[:,0]+1e-8),interior[:,1]-1e-8),length]
            reference_seed=seed.copy()
            def residual(v):
                q=v[:7];ell=v[7];p,a,_=tool_fk(model,data,q)
                data.qvel[:]=0;mujoco.mj_forward(model,data)
                excess=np.maximum(abs(data.qfrc_bias)/caps-.94,0.)
                return np.r_[8.*(p+ell*a-target),axis_weight*(a-old_axis),
                    2.*max(.19-a[1],0.),2.*max(.5-a[0],0.),.15*excess,
                    regularizer*np.array([2,1,1,1,1,1,1])*(q-reference_seed),.015*(ell-material_distance)]
            result=least_squares(residual,initial,bounds=(np.r_[interior[:,0],.48],np.r_[interior[:,1],.785]),
                max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            seed=result.x[:7];length=float(result.x[7]);p,a,tip=tool_fk(model,qpos=seed)
            error=float(np.linalg.norm(p+length*a-target))
            fits.append({'index':i,'qpos_rad':seed.tolist(),'material_distance_m':length,
                'material_point_error_m':error,'actual_material_point_m':(p+length*a).tolist(),
                'desired_material_point_m':target.tolist(),'actual_axis':a.tolist(),
                'fit_pass':bool(error<.012 and a[1]>.15 and a[0]>.5)})
            nodes.append(seed.copy())
        nodes=np.vstack([qentry,nodes]);curve=C2JointSpline(times,nodes)
        audit,arrays=audit_motion(scene,geometry,curve,neutral,[0,-1,1],'right')
        end=static_row(model,data,scene,geometry,nodes[-1],neutral,'guard',caps)
        axes=np.array([tool_fk(model,qpos=q)[1] for q in arrays['qpos']])
        sampled=arrays['qpos'][arrays['time_s']>=times[1]]
        margins=np.minimum(sampled-physical[:,0],physical[:,1]-sampled)
        core_controls=C2JointSpline(times[1:],nodes[1:]).bezier_controls
        bounded=bool(np.all(core_controls>=interior[:,0]) and np.all(core_controls<=interior[:,1]))
        smooth=bool(np.max(abs(np.diff(nodes[1:],axis=0)))<.15 and max(curve.analytic_extrema()['peak_speed_rad_s'])<1.2)
        passed=bool(all(f['fit_pass'] for f in fits) and audit['sampled_reference_pass'] and
                    max(axes[:,1])>.15 and min(axes[:,1])>.15 and bounded and smooth and
                    end['static_hold_under_cap'] and end['static_pair_screen']['zero_overlap_start_pass'])
        trial={'axis_prior_weight':axis_weight,'regularizer':regularizer,'fits':fits,
            'knot_qpos':nodes.tolist(),'knot_time_s':times.tolist(),'audit':audit,'end_metrics':end,
            'minimum_strike_reference_joint_margin_deg':np.rad2deg(np.min(margins,axis=0)).tolist(),
            'axis_y_range':[float(min(axes[:,1])),float(max(axes[:,1]))],
            'interior_strike_bounds_pass':bounded,'bounded_local_motion_pass':smooth,
            'candidate_pass':passed}
        report['robust_refit_trials'].append(trial)
        print('positive-robust',axis_weight,regularizer,'fits',sum(f['fit_pass'] for f in fits),
            'geometry',audit['sampled_reference_pass'],'smooth',smooth,'endhold',end['static_hold_under_cap'],'PASS',passed,flush=True)
        write(output/'report.json',report)
        if passed:selected=(curve,audit,arrays,end,trial);break
    if selected:
        curve,audit,arrays,end,trial=selected
        high='right_high_positive_robust'
        report['guards'][high]={'qpos_rad':curve.knot_qpos[-1].tolist(),**end,'static_reference_pass':True}
        report['actions']['cut_up_left_to_right']={'kind':'attack','family':'cut_up_left_to_right',
            'start_guard':'left_guard_positive','end_guard':high,'target_region':'right',
            'direction_intent_attacker_frame':[0,-1,1],'knot_qpos':curve.knot_qpos.tolist(),
            'knot_time_s':times.tolist(),'candidate_pass':True,'sampled_reference_pass':True,
            'reference_preparation_ends_s':float(times[1]),'audit':audit,'native_dynamics_tested':False,
            'admitted_to_training':False,'spline':curve.report()}
        np.savez_compressed(output/'cut_up_left_to_right.npz',**arrays,knot_time_s=times,knot_qpos=curve.knot_qpos)
        recovery=retime([curve.knot_qpos[-1],qentry],minimum_s=2.,speed=.9)
        check,ar=transition_audit(scene,geometry,model,recovery,neutral,caps)
        report['recovery_audit']=check
        if check['sampled_reference_pass']:
            report['actions']['recover_positive_high_to_low']={'kind':'transition','family':'explicit_guard_recovery',
                'start_guard':high,'end_guard':'left_guard_positive','knot_qpos':recovery.knot_qpos.tolist(),
                'knot_time_s':recovery.knot_time_s.tolist(),'candidate_pass':True,'sampled_reference_pass':True,
                'audit':check,'native_dynamics_tested':False,'admitted_to_training':False,'spline':recovery.report()}
            np.savez_compressed(output/'recover_positive_high_to_low.npz',**ar,
                                knot_time_s=recovery.knot_time_s,knot_qpos=recovery.knot_qpos)
        report['status']='frozen_positive_strike_interior_margin_candidate'
    else:report['status']='no_positive_strike_interior_candidate_in_bounded_search'
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['build','positive_contacts','recondition_negative','rebridge','safety_refit','positive_blade_contacts','positive_sweep','robust_negative','robust_chain','robust_positive','robust_chain_safe'],default='build')
    args=p.parse_args();{'build':build,'positive_contacts':positive_contacts,
        'recondition_negative':recondition_negative,'rebridge':rebridge,
        'safety_refit':lambda p:recondition_negative(p,extra_margin=True),
        'positive_blade_contacts':positive_blade_contacts,'positive_sweep':positive_sweep,
        'robust_negative':robust_negative,'robust_chain':robust_chain,'robust_positive':robust_positive,
        'robust_chain_safe':lambda p:robust_chain(p,'positive_margin11')}[args.mode](args.output)
