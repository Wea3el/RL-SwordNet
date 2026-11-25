"""Bounded orientation-v2 geometry search; no inherited dynamics certificates.

Actor-local +X faces the opponent, +Y is left.  Every IK request specifies
both the grip and full shaft direction.  Global seeds include several legal
branches; candidates are never obtained by clipping or wrapping a trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from plan1_skills.matching import tool_fk
from .assembly import ROOT, _finite_json
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import GripAxisSolver, pair_screen, unit, audit_motion
from .triangle_task import TriangleGeometry


def save(path, value):
    path.write_text(json.dumps(_finite_json(value), indent=2, ensure_ascii=False)+'\n')


def source_data():
    path=ROOT/'outputs/multiguard_v1_geometry/bundle13/report.json'
    old=json.loads(path.read_text())
    geometry=TriangleGeometry.from_mapping({k:old['scoring'][k] for k in ('base_distance','regions')})
    old_forward=json.loads((ROOT/'outputs/multiguard_v1_geometry/guards04/report.json').read_text())
    neutral=np.array(old_forward['guards']['center']['qpos_rad'])
    return path,old,old_forward,geometry,neutral


def global_seeds(bounds,old,forward,starts):
    seeds=[np.asarray(x['qpos_rad']) for x in forward['guards'].values()]
    seeds += [np.asarray(x['qpos_rad']) for x in old['guards'].values()]
    rng=np.random.default_rng(20261125)
    # Deterministic broad seeds span joint branches, not only the old +Y ones.
    seeds += list(rng.uniform(bounds[:,0]+.03,bounds[:,1]-.03,size=(starts,7)))
    return seeds


def goals(geometry):
    rows=[]
    # Guard targets have genuinely opposing shaft headings, not renamed tips.
    for side,sign in [('left',1.),('right',-1.)]:
        for level in ('high','low'):
            for x in (.40,.55):
                for z in ((.90,1.04) if level=='high' else (.65,.80)):
                    a=unit([.78,sign*.47,.42 if level=='high' else -.30])
                    p=np.array([x,sign*.20,z])
                    rows.append({'name':side+'_'+level,'kind':'guard','grip_m':p.tolist(),'axis':a.tolist()})
    rows.append({'name':'center','kind':'guard','grip_m':[.5,-.12,.9],'axis':[1.,0.,0.]})
    for region in ('left','right','upper'):
        target=np.asarray(geometry.world_center(region,'a2'))+[-.06,0.,-.025 if region=='upper' else 0.]
        for az in (.30,.50):
            for ay in (-.40,-.60):
                a=unit([.75,ay,az])
                for distance in (.72,.78):
                    p=target-distance*a
                    rows.append({'name':'contact_'+region+'_negative','kind':'contact','target_region':region,
                        'grip_m':p.tolist(),'axis':a.tolist(),'material_distance_m':distance,
                        'requested_contact_point_m':target.tolist()})
    return rows


def static_row(model,data,scene,geometry,q,neutral,kind,cap):
    p,a,t=tool_fk(model,data,q)
    data.qvel[:]=0;mujoco.mj_forward(model,data)
    bias=data.qfrc_bias.copy()
    check=pair_screen(scene,geometry,q,neutral)
    # Contact poses should overlap a scoring region but never robot hardware.
    illegal=[c for c in check['all_penetrating_contacts'] if c['category'].startswith('illegal_')]
    no_physical_overlap=not check['all_penetrating_contacts']
    return {'actual_grip_m':p.tolist(),'actual_axis':a.tolist(),'actual_tip_m':t.tolist(),
        'static_gravity_bias_nm':bias.tolist(),'static_gravity_cap_nm':cap.tolist(),
        'max_static_gravity_ratio':float(np.max(abs(bias)/cap)),
        'static_hold_under_cap':bool(np.all(abs(bias)<cap)),
        'static_pair_screen':check,'no_illegal_overlap':not illegal,
        'no_physical_overlap':no_physical_overlap}


def neutral_pass(metrics,q):
    q=np.asarray(q);bounds=model_joint_ranges_rad(load_joint_constraints());axis=np.asarray(metrics['actual_axis'])
    return bool(metrics['static_hold_under_cap'] and metrics['no_physical_overlap'] and
        np.all(q>=bounds[:,0]) and np.all(q<=bounds[:,1]) and axis[0]>=.85 and abs(axis[1])<=.2)


def search(output,starts=24):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use fresh output; preserve all failures')
    output.mkdir(parents=True)
    source,old,forward,geometry,neutral=source_data()
    cons=load_joint_constraints();physical=model_joint_ranges_rad(cons)
    bounds=physical+np.array([.03,-.03])
    single=build_limited_floor(cons);model=single.model;data=mujoco.MjData(model)
    scene=build_limited_duel(geometry.base_distance,initial_qpos=neutral,constraints=cons)
    solver=GripAxisSolver(model,bounds)
    cap=np.max(abs(model.actuator_forcerange),axis=1)*abs(model.actuator_gear[:,0])
    seeds=global_seeds(bounds,old,forward,starts)
    report={'schema_version':2,'status':'searching_orientation_balanced_geometry',
        'scope':'Static full-grip/full-shaft multi-branch IK; no native certificate transfer',
        'scoring':old['scoring'],'coordinate_frame':'actor local +X forward, +Y left, +Z up',
        'admitted_to_training':False,'native_dynamics_tested':False,'candidate_only':True,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,Path(__file__),ROOT/'config/joint_constraints_observed.json')},
        'joint_search_margin_rad':.03,'random_seed':20261125,'global_random_starts':starts,
        'guards':{},'contact_candidates':{},'trials':[],'actions':{},
        'orientation_requirements':{'guard_axes':{'left_high':{'y_min':.25},'left_low':{'y_min':.25},
            'right_high':{'y_max':-.25},'right_low':{'y_max':-.25},'center':{'x_min':.85,'abs_y_max':.20}}}}
    save(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    for goal_index,goal in enumerate(goals(geometry)):
        name=goal['name'];passing=[]
        for seed_index,seed in enumerate(seeds):
            q,fit=solver.solve(goal['grip_m'],goal['axis'],seed,regularization=.003,iterations=140)
            row={'goal_index':goal_index,'goal':goal,'seed_index':seed_index,
                'qpos_rad':q.tolist(),**fit}
            if fit['fit_pass']:
                row.update(static_row(model,data,scene,geometry,q,neutral,goal['kind'],cap))
                row['candidate_pass']=row['no_physical_overlap'] and (goal['kind']=='contact' or row['static_hold_under_cap'])
                if goal['kind']=='contact':
                    gap=row['static_pair_screen']['region_gaps_m']['a1'][goal['target_region']]
                    row['actual_requested_region_gap_m']=gap
                    row['candidate_pass'] &= gap<=-.005
                if goal['kind']=='guard' and name!='center':
                    tip=np.array(row['actual_tip_m']);a=np.array(row['actual_axis'])
                    row['candidate_pass'] &= bool((a[1]>=.25 if name.startswith('left') else a[1]<=-.25)
                        and (tip[2]>=1.17 if name.endswith('high') else tip[2]<=.80))
                if row['candidate_pass']:passing.append(row)
            else:row['candidate_pass']=False
            report['trials'].append(row)
            if len(passing)>=3:break
        if passing:
            passing.sort(key=lambda r:(r['max_static_gravity_ratio'],np.sum(abs(np.asarray(r['qpos_rad'])-neutral))))
            if goal['kind']=='guard':
                previous=report['guards'].get(name)
                if previous is None or passing[0]['max_static_gravity_ratio']<previous['max_static_gravity_ratio']:
                    report['guards'][name]=passing[0]
            else:
                report['contact_candidates'].setdefault(name,[]).extend(passing)
        report['status']='partial_static_results'
        save(output/'report.json',report)
        print(goal_index,name,'passes',len(passing),'negative_contacts',sum(len(x) for x in report['contact_candidates'].values()),flush=True)
    report['status']='static_orientation_search_complete_not_path_validated'
    save(output/'report.json',report)
    return report


def arc_task(p,a,side,phase,scale):
    s=phase*scale
    if side=='right':
        # Right high -> left low: negative shaft heading becomes positive.
        shift=np.array([-(.30 if s>0 else .12)*s*s,
            (-.216 if s>0 else .10)*s,-(.18 if s>0 else .25)*s])
        axis=unit(a+[0,.85*s,-.40*s])
    else:
        shift=np.array([-(.30 if s>0 else .12)*s*s,-.15*s,-(.18 if s>0 else .25)*s])
        axis=unit(a+[0,-1.10*s,-.40*s])
    return p+shift,axis


def grow_paths(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve candidates; use new directory')
    output.mkdir(parents=True)
    prior=json.loads(source.read_text());geometry=TriangleGeometry.from_mapping({k:prior['scoring'][k] for k in ('base_distance','regions')})
    center=np.array(prior['guards']['center']['qpos_rad'])
    cons=load_joint_constraints();bounds=model_joint_ranges_rad(cons)+np.array([.03,-.03])
    single=build_limited_floor(cons);m=single.model;solver=GripAxisSolver(m,bounds)
    cap=np.max(abs(m.actuator_forcerange),axis=1)*abs(m.actuator_gear[:,0])
    scene=build_limited_duel(geometry.base_distance,initial_qpos=center,constraints=cons)
    report={'schema_version':2,'status':'growing_signed_heading_arcs','scoring':prior['scoring'],
        'source_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'guards':{'center':dict(prior['guards']['center'],static_reference_pass=True)},
        'actions':{},'path_trials':[],
        'orientation_requirements':{'guard_axes':{'left_high':{'y_min':.25},'left_low':{'y_min':.25},
            'right_high':{'y_max':-.25},'right_low':{'y_max':-.25},'center':{'x_min':.85,'abs_y_max':.2}}},
        'limitations':['New joint branches and neutral invalidate every old native certificate.',
            'Static contacts and sampled trajectories are only preliminary geometry evidence.',
            'Six attacks, three defences, transitions and common-platform admission remain required.']}
    for side in ('right','left'):
        sources=sorted(prior['contact_candidates']['contact_'+side+'_negative'],
            key=lambda r:(not np.asarray(r['qpos_rad'])[2]<np.pi,r['max_static_gravity_ratio']))
        # Two distinct feasible branches per side, at most six finite trajectories.
        candidates=[]
        for srow in sources:
            if all(np.linalg.norm(np.asarray(srow['qpos_rad'])-np.asarray(o['qpos_rad']))>.15 for o in candidates):
                candidates.append(srow)
            if len(candidates)==2:break
        selected=None
        for ci,srow in enumerate(candidates):
            qc=np.array(srow['qpos_rad']);p,a,_=tool_fk(m,qpos=qc)
            for scale in (.85,1.,1.15):
                rows={0.:{'qpos_rad':qc.tolist(),'fit_pass':True,'phase':0.}}
                for sign in (-1.,1.):
                    q=qc.copy()
                    for phase in sign*np.linspace(.1,1.,10):
                        pg,ag=arc_task(p,a,side,phase,scale)
                        q,fit=solver.solve(pg,ag,q,regularization=.004,iterations=160)
                        rows[float(phase)]={'qpos_rad':q.tolist(),'phase':float(phase),
                            'requested_grip_m':pg.tolist(),'requested_axis':ag.tolist(),**fit}
                rows=[rows[k] for k in sorted(rows)];q=np.array([r['qpos_rad'] for r in rows])
                times=np.linspace(0,5.,len(q));curve=C2JointSpline(times,q)
                peak=max(curve.analytic_extrema()['peak_speed_rad_s'])
                if peak>1.2:curve=C2JointSpline(times*(peak/1.2),q)
                direction=[0,1,-1] if side=='right' else [0,-1,-1]
                audit,arrays=audit_motion(scene,geometry,curve,center,direction,side)
                start=static_row(m,solver.data,scene,geometry,q[0],center,'guard',cap)
                end=static_row(m,solver.data,scene,geometry,q[-1],center,'guard',cap)
                sign_goal=-1 if side=='right' else 1
                headings=start['actual_axis'][1]*sign_goal>=.25 and end['actual_axis'][1]*sign_goal<=-.25
                guard_heights=start['actual_tip_m'][2]>=1.17 and end['actual_tip_m'][2]<=.8
                fitpass=all(r['fit_pass'] for r in rows)
                passed=fitpass and headings and guard_heights and audit['sampled_reference_pass'] and start['static_hold_under_cap'] and end['static_hold_under_cap']
                trial={'side':side,'contact_index':ci,'scale':scale,'ik_rows':rows,'fit_pass':fitpass,
                    'heading_anchors_pass':bool(headings),'guard_heights_pass':bool(guard_heights),
                    'start_metrics':start,'end_metrics':end,'audit':audit,'candidate_pass':bool(passed),
                    'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':q.tolist()}
                name=f'{side}_signed_arc_{ci}_{scale:.2f}';report['path_trials'].append(trial)
                np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=q)
                print(name,'fit',fitpass,'heading',headings,'height',guard_heights,'screen',audit['sampled_reference_pass'],
                    'hold',start['static_hold_under_cap'],end['static_hold_under_cap'],flush=True)
                save(output/'report.json',report)
                if passed and selected is None:selected=trial
        if selected:
            high='right_high' if side=='right' else 'left_high';low='left_low' if side=='right' else 'right_low'
            q=np.array(selected['knot_qpos'])
            for n,pose,metrics in ((high,q[0],selected['start_metrics']),(low,q[-1],selected['end_metrics'])):
                report['guards'][n]=dict(metrics,qpos_rad=pose.tolist(),static_reference_pass=True)
            down='cut_down_right_to_left' if side=='right' else 'cut_down_left_to_right'
            up='cut_up_left_to_right' if side=='right' else 'cut_up_right_to_left'
            for n,start_name,end_name,poses,d in ((down,high,low,q,direction),(up,low,high,q[::-1],-np.asarray(direction))):
                curve=C2JointSpline(selected['knot_time_s'],poses)
                audit,arrays=audit_motion(scene,geometry,curve,center,d,side)
                row={'kind':'attack','start_guard':start_name,'end_guard':end_name,'target_region':side,
                    'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':poses.tolist(),
                    'candidate_pass':audit['sampled_reference_pass'],'sampled_reference_pass':audit['sampled_reference_pass'],
                    'audit':audit,'native_dynamics_tested':False,'admitted_to_training':False}
                report['actions'][n]=row
                np.savez_compressed(output/(n+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=poses)
    report['status']='signed_heading_arc_candidates_complete'
    (output/'scene.xml').write_text(scene.spec.to_xml());save(output/'report.json',report)
    return report


def negative_corridor(source,output,neutral_override=None):
    """First negative-heading diagonal, without forcing a wrist hemisphere flip."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Use fresh output')
    output.mkdir(parents=True);prior=json.loads(source.read_text())
    g=TriangleGeometry.from_mapping({k:prior['scoring'][k] for k in ('base_distance','regions')})
    center=np.array(prior['guards']['center']['qpos_rad'] if neutral_override is None else neutral_override);cons=load_joint_constraints()
    m=build_limited_floor(cons).model;solver=GripAxisSolver(m,model_joint_ranges_rad(cons)+[.03,-.03])
    cap=np.max(abs(m.actuator_forcerange),axis=1);scene=build_limited_duel(g.base_distance,initial_qpos=center)
    center_metrics=static_row(m,solver.data,scene,g,center,center,'guard',cap)
    report={'schema_version':2,'scoring':prior['scoring'],'status':'negative_corridor_search',
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'guards':{'center':dict(center_metrics,qpos_rad=center.tolist(),static_reference_pass=neutral_pass(center_metrics,center))},'actions':{},'path_trials':[],
        'orientation_requirements':{'guard_axes':{'center':{'x_min':.85,'abs_y_max':.2}}},
        'limitations':['First negative-heading motion only; positive counterpart and common guard graph remain unbuilt.'],
        'model_source_sha256':scene.source_hashes,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    poses=prior['contact_candidates']['contact_left_negative']
    sources=[]
    for upper in (True,False):
        eligible=[r for r in poses if (np.asarray(r['qpos_rad'])[2]>np.pi)==upper]
        if eligible:sources.append(min(eligible,key=lambda r:r['max_static_gravity_ratio']))
    selected=None
    for ci,row in enumerate(sources):
        qc=np.array(row['qpos_rad']);p,a,_=tool_fk(m,qpos=qc)
        for scale in (.85,1.,1.15):
            fitted={0.:{'qpos_rad':qc.tolist(),'fit_pass':True}}
            for sign in (-1.,1.):
                q=qc.copy()
                for u in sign*np.linspace(.1,1.,10):
                    s=u*scale
                    # High hand lies actor-left; low hand moves inward/right.
                    pp=p+[-.55*s if s>0 else -.10*s*s,
                        -(.10 if s>0 else .35)*s,
                        -.10*s-.10*np.sin(np.pi*s) if s>0 else -.20*s]
                    aa=unit(a+[0,-.12*s,-.30*s])
                    q,fit=solver.solve(pp,aa,q,regularization=.004,iterations=160)
                    fitted[float(u)]={'qpos_rad':q.tolist(),'phase':float(u),**fit}
            rows=[fitted[k] for k in sorted(fitted)];q=np.array([x['qpos_rad'] for x in rows])
            times=np.linspace(0,5.,len(q));curve=C2JointSpline(times,q)
            peak=max(curve.analytic_extrema()['peak_speed_rad_s'])
            if peak>1.2:curve=C2JointSpline(times*peak/1.2,q)
            audit,arrays=audit_motion(scene,g,curve,center,[0,-1,-1],'left')
            start=static_row(m,solver.data,scene,g,q[0],center,'guard',cap)
            end=static_row(m,solver.data,scene,g,q[-1],center,'guard',cap)
            fitpass=all(x['fit_pass'] for x in rows)
            heading=all(x['actual_axis'][1]<=-.25 for x in (start,end))
            heights=start['actual_tip_m'][2]>=1.17 and end['actual_tip_m'][2]<=.8
            passed=fitpass and heading and heights and audit['sampled_reference_pass'] and start['static_hold_under_cap'] and end['static_hold_under_cap']
            trial={'source_index':ci,'scale':scale,'fit_pass':fitpass,'heading_pass':heading,'height_pass':heights,
                'start_metrics':start,'end_metrics':end,'audit':audit,'candidate_pass':bool(passed),
                'rows':rows,'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':q.tolist()}
            report['path_trials'].append(trial)
            np.savez_compressed(output/f'negative_diagonal_{ci}_{scale}.npz',**arrays,
                knot_time_s=curve.knot_time_s,knot_qpos=q)
            print('negative',ci,scale,'fit',fitpass,'heading',heading,'height',heights,'geometry',audit['sampled_reference_pass'],
                'hold',start['static_hold_under_cap'],end['static_hold_under_cap'],flush=True)
            save(output/'report.json',report)
            if passed and selected is None:selected=trial
    if selected:
        q=np.array(selected['knot_qpos'])
        for name,pose,metrics in [('left_high_negative',q[0],selected['start_metrics']),('right_low_negative',q[-1],selected['end_metrics'])]:
            report['guards'][name]=dict(metrics,qpos_rad=pose.tolist(),static_reference_pass=True)
            report['orientation_requirements']['guard_axes'][name]={'y_max':-.25}
        for name,start,end,qs,direction in [('cut_down_left_to_right','left_high_negative','right_low_negative',q,[0,-1,-1]),
            ('cut_up_right_to_left','right_low_negative','left_high_negative',q[::-1],[0,1,1])]:
            curve=C2JointSpline(selected['knot_time_s'],qs);audit,arrays=audit_motion(scene,g,curve,center,direction,'left')
            report['actions'][name]={'kind':'attack','start_guard':start,'end_guard':end,'target_region':'left',
                'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':qs.tolist(),'candidate_pass':audit['sampled_reference_pass'],
                'sampled_reference_pass':audit['sampled_reference_pass'],'audit':audit,'admitted_to_training':False,'native_dynamics_tested':False}
            np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=qs)
    report['status']='negative_corridor_candidates_complete'
    (output/'scene.xml').write_text(scene.spec.to_xml());save(output/'report.json',report)
    return report


def trim_feasible(source,output):
    """Explicitly shorten unreachable recovery ends and re-audit NEW curves."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Use fresh output')
    output.mkdir(parents=True);old=json.loads(source.read_text())
    g=TriangleGeometry.from_mapping({k:old['scoring'][k] for k in ('base_distance','regions')})
    center=np.array(old['guards']['center']['qpos_rad']);cons=load_joint_constraints()
    m=build_limited_floor(cons).model;data=mujoco.MjData(m)
    cap=np.max(abs(m.actuator_forcerange),axis=1);scene=build_limited_duel(g.base_distance,initial_qpos=center)
    report={'schema_version':2,'scoring':old['scoring'],'guards':{'center':old['guards']['center']},
        'actions':{},'trim_trials':[],'candidate_only':True,'admitted_to_training':False,
        'native_dynamics_tested':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'orientation_requirements':{'guard_axes':{'center':{'x_min':.85,'abs_y_max':.2}}},
        'scope':'New shorter feasible negative-heading arcs; failed old end frames retained in source, no frame skipping.'}
    passed=[]
    for i,trial in enumerate(old['path_trials']):
        rows=trial['rows'];first_bad=next((j for j,x in enumerate(rows) if not x['fit_pass']),len(rows))
        if first_bad<=11:continue
        nodes=np.array(trial['knot_qpos'])[:first_bad]
        # Require the new terminal point to be a real low guard, not a contact pose.
        end=static_row(m,data,scene,g,nodes[-1],center,'guard',cap)
        if end['actual_tip_m'][2]>.8 or end['actual_axis'][1]>-.25:continue
        times=np.linspace(0,4.5,len(nodes));curve=C2JointSpline(times,nodes)
        speed=max(curve.analytic_extrema()['peak_speed_rad_s'])
        if speed>1.:curve=C2JointSpline(times*speed,nodes)
        audit,arrays=audit_motion(scene,g,curve,center,[0,-1,-1],'left')
        start=static_row(m,data,scene,g,nodes[0],center,'guard',cap)
        start_shape=start['actual_axis'][1]<=-.25 and start['actual_tip_m'][2]>=1.17
        good=audit['sampled_reference_pass'] and start_shape and start['static_hold_under_cap'] and end['static_hold_under_cap']
        row={'source_trial_index':i,'first_discarded_terminal_row_index':first_bad,
            'new_end_original_phase':rows[first_bad-1].get('phase'),'candidate_pass':bool(good),'start_heading_and_height_pass':bool(start_shape),
            'audit':audit,'start_metrics':start,'end_metrics':end,
            'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':nodes.tolist()}
        report['trim_trials'].append(row)
        np.savez_compressed(output/f'trim_{i}.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=nodes)
        print('trim',i,'endphase',row['new_end_original_phase'],'duration',curve.duration_s,'pass',good,flush=True)
        if good:passed.append(row)
    if passed:
        # First physical prototype minimizes motion duration.  A safe path from
        # the exploratory neutral to its starting branch is NOT implied.
        chosen=min(passed,key=lambda r:r['knot_time_s'][-1])
        nodes=np.asarray(chosen['knot_qpos'])
        for name,q,metrics in [('left_high_negative',nodes[0],chosen['start_metrics']),
                ('right_low_negative',nodes[-1],chosen['end_metrics'])]:
            report['guards'][name]=dict(metrics,qpos_rad=q.tolist(),static_reference_pass=True)
            report['orientation_requirements']['guard_axes'][name]={'y_max':-.25}
        for name,qs,sn,en,d in [('cut_down_left_to_right',nodes,'left_high_negative','right_low_negative',[0,-1,-1]),
                ('cut_up_right_to_left',nodes[::-1],'right_low_negative','left_high_negative',[0,1,1])]:
            curve=C2JointSpline(chosen['knot_time_s'],qs)
            audit,arrays=audit_motion(scene,g,curve,center,d,'left')
            report['actions'][name]={'kind':'attack','start_guard':sn,'end_guard':en,'target_region':'left',
                'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':qs.tolist(),
                'candidate_pass':audit['sampled_reference_pass'],'sampled_reference_pass':audit['sampled_reference_pass'],
                'direction_intent_attacker_frame':d,
                'audit':audit,'admitted_to_training':False,'native_dynamics_tested':False}
            np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=qs)
    report['status']='trimmed_negative_candidates_complete'
    report['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['exploratory_neutral_not_final_canonical_guard']=True
    report['neutral_to_action_entry_transition_tested']=False
    report['limitations']=['Only a standalone negative-heading physical prototype; no complete skill bank or training admission.',
        'The exploratory forward neutral is laterally offset +0.50 m. It was not chosen as a final common neutral.',
        'A reset at the action entry is declared; no untested neutral-to-entry bridge or angle wrapping is performed.',
        'Center/entry joint branches differ; a lawful, natural connecting route remains to be established.']
    save(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    return report


def upper_negative(source,output):
    """Bounded vertical cut around a negative-heading upper contact pose."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Use fresh output')
    output.mkdir(parents=True);old=json.loads(source.read_text())
    _,_,forward,g,center=source_data()
    cons=load_joint_constraints();m=build_limited_floor(cons).model
    solver=GripAxisSolver(m,model_joint_ranges_rad(cons)+[.03,-.03])
    cap=np.max(abs(m.actuator_forcerange),axis=1);scene=build_limited_duel(g.base_distance,initial_qpos=center)
    source_row=min(old['contact_candidates']['contact_upper_negative'],key=lambda r:r['max_static_gravity_ratio'])
    qc=np.asarray(source_row['qpos_rad']);p,a,_=tool_fk(m,qpos=qc)
    cm=static_row(m,solver.data,scene,g,center,center,'guard',cap)
    report={'schema_version':2,'scoring':old['scoring'],'guards':{'center':dict(cm,qpos_rad=center.tolist(),static_reference_pass=neutral_pass(cm,center))},
        'actions':{},'trials':[],'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'scope':'Negative-heading vertical upper cut with original forward-neutral opponent; no neutral repositioning.',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_source_sha256':scene.source_hashes,'orientation_requirements':{'guard_axes':{'center':{'x_min':.85,'abs_y_max':.2}}}}
    chosen=None
    for scale in (.65,.85,1.):
        rows={0.:{'qpos_rad':qc.tolist(),'fit_pass':True,'phase':0.}}
        for sign in (-1.,1.):
            q=qc.copy()
            for phase in sign*np.linspace(.1,1.,10):
                s=float(phase)*scale
                if s<0:
                    pp=p+[-.07*s*s,0.,-.14*s];aa=unit(a+[0,0,-.20*s])
                else:
                    pp=p+[-.16*s,0.,-.025*s];aa=unit(a+[0,0,-.13*s])
                q,fit=solver.solve(pp,aa,q,regularization=.003,iterations=180)
                rows[float(phase)]={'phase':float(phase),'qpos_rad':q.tolist(),**fit}
        rows=[rows[k] for k in sorted(rows)];q=np.array([x['qpos_rad'] for x in rows])
        curve=C2JointSpline(np.linspace(0,4.,len(q)),q)
        peak=max(curve.analytic_extrema()['peak_speed_rad_s'])
        if peak>1.:curve=C2JointSpline(curve.knot_time_s*peak,q)
        audit,arrays=audit_motion(scene,g,curve,center,[0,0,-1],'upper')
        start=static_row(m,solver.data,scene,g,q[0],center,'guard',cap);end=static_row(m,solver.data,scene,g,q[-1],center,'guard',cap)
        fitpass=all(x['fit_pass'] for x in rows)
        passed=fitpass and audit['sampled_reference_pass'] and start['static_hold_under_cap'] and end['static_hold_under_cap'] and start['actual_axis'][1]<-.25
        trial={'scale':scale,'rows':rows,'fit_pass':fitpass,'candidate_pass':bool(passed),'audit':audit,
            'start_metrics':start,'end_metrics':end,'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':q.tolist()}
        report['trials'].append(trial)
        np.savez_compressed(output/f'upper_{scale}.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=q)
        print('upper negative',scale,'fit',fitpass,'geometry',audit['sampled_reference_pass'],'hold',start['static_hold_under_cap'],end['static_hold_under_cap'],flush=True)
        if passed and chosen is None:chosen=trial
        save(output/'report.json',report)
    if chosen:
        q=np.array(chosen['knot_qpos'])
        for name,pose,metrics in [('upper_high_negative',q[0],chosen['start_metrics']),('upper_recovery_negative',q[-1],chosen['end_metrics'])]:
            report['guards'][name]=dict(metrics,qpos_rad=pose.tolist(),static_reference_pass=True)
            report['orientation_requirements']['guard_axes'][name]={'y_max':-.25}
        row={'kind':'attack','start_guard':'upper_high_negative','end_guard':'upper_recovery_negative','target_region':'upper',
            'knot_time_s':chosen['knot_time_s'],'knot_qpos':chosen['knot_qpos'],'candidate_pass':True,'sampled_reference_pass':True,
            'direction_intent_attacker_frame':[0,0,-1],'audit':chosen['audit'],'admitted_to_training':False,'native_dynamics_tested':False}
        report['actions']['cut_down_center']=row
        curve=C2JointSpline(chosen['knot_time_s'],q);ts=np.linspace(0,curve.duration_s,301)
        np.savez_compressed(output/'cut_down_center.npz',time_s=ts,qpos=curve.sample(ts),qvel=curve.derivatives(ts)[0],knot_time_s=curve.knot_time_s,knot_qpos=q)
    report['status']='upper_negative_candidate_complete'
    save(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    return report


def upper_retarget(source,output):
    """One final bounded full-pose retarget diagnostic, four goals/three seeds."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh output required')
    output.mkdir(parents=True);old=json.loads(source.read_text())
    _,_,_,g,center=source_data();m=build_limited_floor().model
    solver=GripAxisSolver(m,model_joint_ranges_rad(load_joint_constraints())+[.03,-.03])
    cap=np.max(abs(m.actuator_forcerange),axis=1);scene=build_limited_duel(g.base_distance,initial_qpos=center)
    report={'schema_version':2,'status':'bounded_upper_retarget_diagnostic','scoring':old['scoring'],
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,'actions':{},'trials':[],
        'goals_limit':4,'seeds_per_goal':3,'actual_region_overlap_required_m':-.005,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    metrics=static_row(m,solver.data,scene,g,center,center,'guard',cap)
    report['guards']={'center':dict(metrics,qpos_rad=center.tolist(),static_reference_pass=neutral_pass(metrics,center))}
    seeds=[old['contact_candidates'][n][0]['qpos_rad'] for n in ('contact_upper_negative','contact_left_negative','contact_right_negative')]
    goals4=[([.75,.42,.95],[.8,-.5,.33]),([.78,.40,.93],[.8,-.45,.4]),
        ([.76,.45,.95],[.75,-.5,.4]),([.80,.42,.90],[.8,-.5,.4])]
    for gi,(p,a) in enumerate(goals4):
        for si,seed in enumerate(seeds):
            q,fit=solver.solve(p,unit(a),seed,regularization=.002,iterations=180)
            metric=static_row(m,solver.data,scene,g,q,center,'contact',cap)
            gap=metric['static_pair_screen']['region_gaps_m']['a1']['upper']
            row={'goal_index':gi,'seed_index':si,'requested_grip_m':p,'requested_axis':unit(a).tolist(),
                'qpos_rad':q.tolist(),**fit,**metric,'actual_region_gap_m':gap,
                'candidate_pass':fit['fit_pass'] and metric['no_physical_overlap'] and gap<=-.005}
            report['trials'].append(row)
        print('upper fullpose goal',gi,'bestactualgap',min(r['actual_region_gap_m'] for r in report['trials'] if r['goal_index']==gi),flush=True)
        save(output/'report.json',report)
    report['actual_contact_candidate_count']=sum(r['candidate_pass'] for r in report['trials'])
    report['status']='bounded_upper_retarget_complete_not_impossibility_proof'
    save(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--starts',type=int,default=24)
    p.add_argument('--grow-source')
    p.add_argument('--negative-corridor-source')
    p.add_argument('--neutral-rad',type=float,nargs=7)
    p.add_argument('--trim-source')
    p.add_argument('--upper-negative-source')
    p.add_argument('--upper-retarget-source')
    a=p.parse_args()
    if a.upper_retarget_source:upper_retarget(a.upper_retarget_source,a.output)
    elif a.upper_negative_source:upper_negative(a.upper_negative_source,a.output)
    elif a.trim_source:trim_feasible(a.trim_source,a.output)
    elif a.negative_corridor_source:negative_corridor(a.negative_corridor_source,a.output,a.neutral_rad)
    elif a.grow_source:grow_paths(a.grow_source,a.output)
    else:search(a.output,a.starts)


if __name__=='__main__':main()
