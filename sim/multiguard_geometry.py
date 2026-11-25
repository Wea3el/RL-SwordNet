"""Isolated multi-guard sword-arc candidates, never a training certificate.

Cartesian grip AND shaft direction are specified throughout each candidate.
Warm-started bounded IK preserves an unwrapped joint branch.  All rejected
fits and sampled paths remain in the output; unchanged rules are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plan1_skills.matching import tool_fk, _jacobian
from .a1_triangle_candidates import illegal_contacts, region_gaps, retimed_curve, static_screen
from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .fencing_rules import blade_sections_from_scene, point_cylinder_distance
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .triangle_task import TriangleGeometry
from .motion_spline import C2JointSpline


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def write(path, obj):
    path.write_text(json.dumps(_finite_json(obj), ensure_ascii=False, indent=2) + '\n')


class GripAxisSolver:
    def __init__(self, model, bounds):
        self.model, self.bounds = model, np.asarray(bounds)
        self.data = mujoco.MjData(model)
        self.gid = model.site('shinai_grip').id
        self.tid = model.site('shinai_tip').id
        self.weights = np.array([3., 1., 1., 1., 1., 1., 1.])

    def solve(self, p, axis, seed, *, regularization=.012, iterations=100):
        p, axis, seed = np.asarray(p), unit(axis), np.asarray(seed)
        def residual(q):
            pos, a, _ = tool_fk(self.model, self.data, q)
            return np.r_[5.*(pos-p), 1.5*(a-axis), regularization*self.weights*(q-seed)]
        def jac(q):
            pos, a, tip = tool_fk(self.model, self.data, q)
            jp, ja = _jacobian(self.model, self.data, self.gid, self.tid, a, np.linalg.norm(tip-pos))
            return np.vstack([5.*jp, 1.5*ja, regularization*np.diag(self.weights)])
        # Projection applies to optimizer initialization only, not output paths.
        initial = np.clip(seed, self.bounds[:, 0]+1e-8, self.bounds[:, 1]-1e-8)
        fit = least_squares(residual, initial, jac=jac,
            bounds=(self.bounds[:, 0], self.bounds[:, 1]), max_nfev=iterations,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q = fit.x
        actual_p, actual_a, tip = tool_fk(self.model, self.data, q)
        pe = np.linalg.norm(actual_p-p)
        ae = np.rad2deg(np.arccos(np.clip(actual_a@axis, -1., 1.)))
        return q, {'position_error_m': float(pe), 'axis_error_deg': float(ae),
            'actual_grip_m': actual_p.tolist(), 'actual_axis': actual_a.tolist(),
            'actual_tip_m': tip.tolist(), 'nfev': fit.nfev,
            'fit_pass': bool(pe < .025 and ae < 5.)}


def requests(family='forward'):
    """Guard proposals are facing-frame geometry, not scoring exemptions."""
    result = {'center': []}
    if family == 'offset':
        for side in ('left','right'):
            for level in ('high','low'):
                name=side+'_'+level; result[name]=[]
                axes = {('left','high'): [[.2,.85,.48],[.4,.75,.52]],
                    ('right','high'): [[.8,0.,.6],[.8,.3,.52]],
                    ('left','low'): [[.5,.8,-.3],[.3,.9,-.3]],
                    ('right','low'): [[.95,0.,-.3],[.8,.3,-.52]]}[(side,level)]
                for x in (.65,.8):
                    for y in (-.3,-.45):
                        for z in ((.75,.95) if level=='high' else (.6,.75)):
                            for a in axes:
                                result[name].append({'grip_m':[x,y,z], 'axis':unit(a).tolist()})
        return result
    for x in (.5, .6, .7):
        for z in (.8, .9):
            result['center'].append({'grip_m': [x, -.12, z], 'axis': [1., 0., 0.]})
    for side, sign in [('left', 1.), ('right', -1.)]:
        result[side+'_high'], result[side+'_low'] = [], []
        for x in (.3, .4, .5):
            for y in (.20, .30):
                for z in (1.18,):
                    result[side+'_high'].append({'grip_m': [x, sign*y, z],
                        'axis': unit([.98, sign*.02, .18]).tolist()})
                for z in (.65, .8):
                    result[side+'_low'].append({'grip_m': [x, sign*y, z],
                        'axis': unit([.88, sign*.20, -.43]).tolist()})
    return result


def pair_screen(scene, geometry, a, b):
    scene.data.qpos[scene.agents['a1']['qpos_ids']] = a
    scene.data.qpos[scene.agents['a2']['qpos_ids']] = b
    scene.data.qvel[:] = 0.
    mujoco.mj_forward(scene.model, scene.data)
    contacts = []
    from .duel_scene import classify_contact
    for c in scene.data.contact:
        if c.dist >= -1e-8:
            continue
        contacts.append({'category': classify_contact(scene.model, c, rules_version='whole_blade_first_touch_v2'),
            'depth_m': float(-c.dist), 'geoms': [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (c.geom1,c.geom2)]})
    gaps = {a: region_gaps(scene, geometry, a) for a in ('a1','a2')}
    okay = not contacts and all(min(g.values()) > 0 for g in gaps.values())
    return {'all_penetrating_contacts': contacts, 'region_gaps_m': gaps, 'zero_overlap_start_pass': bool(okay)}


def guard_search(solver, scene, geometry, seed, *, family='forward', limit=3):
    found, trials = {}, {}
    old = json.loads((ROOT/'outputs/a1_named_flow_connected_v4/report.json').read_text())['poses']
    seeds = [np.asarray(old[n]['qpos_rad']) for n in ('long_point', 'kron', 'plow_left', 'plow_right')]
    seeds.append(seed)
    if family=='offset':
        seeds=[seed]
        found['center']={'qpos_rad':seed.tolist(),'guard_candidate_pass':True,
            'fit_pass':True,**dict(zip(('actual_grip_m','actual_axis','actual_tip_m'),
                [v.tolist() for v in tool_fk(solver.model,qpos=seed)]))}
    for name, goals in requests(family).items():
        if not goals: continue
        rows = []
        for i, goal in enumerate(goals):
            for si, initial in enumerate(seeds):
                q, fit = solver.solve(goal['grip_m'], goal['axis'], initial, iterations=100)
                checks = static_screen(scene, geometry, q, seed)
                valid = fit['fit_pass'] and all(not r['illegal_contacts'] and
                    min(r['region_gaps_m'].values()) > .02 for r in checks.values())
                row = dict(fit, request=goal, seed_index=si, qpos_rad=q.tolist(), static=checks,
                    guard_candidate_pass=bool(valid), seed_unwrapped_delta_deg=np.rad2deg(q-seeds[0]).tolist(),
                    total_weighted_travel=float(np.abs(q-seeds[0])@solver.weights))
                rows.append(row)
        rows.sort(key=lambda r: (not r['guard_candidate_pass'], r['total_weighted_travel']))
        trials[name] = rows
        good = [r for r in rows if r['guard_candidate_pass']]
        if name == 'center':
            for r in good:
                r['same_guard_pair_screen'] = pair_screen(scene, geometry, r['qpos_rad'], r['qpos_rad'])
            good = [r for r in good if r['same_guard_pair_screen']['zero_overlap_start_pass']]
        elif 'center' in found:
            for r in good:
                r['actual_center_pair_screen'] = pair_screen(scene, geometry, r['qpos_rad'], found['center']['qpos_rad'])
                r['same_guard_pair_screen'] = pair_screen(scene, geometry, r['qpos_rad'], r['qpos_rad'])
            good = [r for r in good if r['actual_center_pair_screen']['zero_overlap_start_pass']
                    and r['same_guard_pair_screen']['zero_overlap_start_pass']]
        if family=='offset':
            sign = 1. if name.startswith('left') else -1.
            good=[r for r in good if sign*r['actual_tip_m'][1]>.12 and
                (r['actual_tip_m'][2]>1.16 if name.endswith('high') else r['actual_tip_m'][2]<.78)]
            for r in good:
                r['total_weighted_travel']=float(np.abs(np.asarray(r['qpos_rad'])-seed)@solver.weights)
            good.sort(key=lambda r:r['total_weighted_travel'])
        if good:
            found[name] = good[0]
        print('guard', name, 'passing', len(good), flush=True)
    return found, trials


def source_geometry():
    source = ROOT/'outputs/compact_bank_v1_natural/engagement01/report.json'
    report = json.loads(source.read_text())
    seed = np.asarray(report['center_qpos_rad'])
    geometry = TriangleGeometry.from_mapping({k: report['scoring'][k] for k in ('base_distance', 'regions')})
    return source, report, seed, geometry


def arc_requests(p, a, side, scale, phase):
    """A smooth non-thrusting stroke through one measured contact pose.

    The grip retracts at both ends, and simultaneously moves laterally and
    vertically. Shaft direction rotates throughout, rather than holding the
    rod fixed and merely targeting its tip. s=-1 is high, +1 is low.
    """
    s=float(phase)*scale
    if s<0:
        s*=1.25 if side=='left' else .80
    if side=='left':
        dp=np.array([-(.35 if s>0 else .12)*s*s, -(.10 if s>0 else .18)*s, -.18*s])
        da=np.array([0.,-.30*s,-.35*s])
    else:
        # Withdraw promptly after the contact corridor, before the descending
        # blade reaches the opponent's low wrist/adapter envelope.
        dp=np.array([-.50*s if s>0 else -.14*s*s, .18*s, -.18*s])
        if s>0:
            dp[2]+=.12*np.sin(np.pi*s)
        da=np.array([0.,.45*s,-.45*s])
    return np.asarray(p)+dp,unit(np.asarray(a)+da)


def grow_contact_arc(solver, qcontact, side, scale):
    p,a,_=tool_fk(solver.model,qpos=qcontact)
    rows={0.: {'phase':0., 'qpos_rad':np.asarray(qcontact).tolist(),
        'fit_pass':True,'position_error_m':0.,'axis_error_deg':0.,
        'requested_grip_m':p.tolist(),'requested_axis':a.tolist()}}
    for sign in (-1.,1.):
        current=np.asarray(qcontact)
        for phase in sign*np.linspace(.1,1.,10):
            pp,aa=arc_requests(p,a,side,scale,phase)
            current,fit=solver.solve(pp,aa,current,regularization=.005,iterations=150)
            rows[float(phase)]={'phase':float(phase),'qpos_rad':current.tolist(),
                'requested_grip_m':pp.tolist(),'requested_axis':aa.tolist(),**fit}
    return [rows[k] for k in sorted(rows)]


def audit_motion(scene,geometry,curve,opponent,direction,target,period=.025):
    ts=np.linspace(0,curve.duration_s,int(np.ceil(curve.duration_s/period))+1)
    qs=curve.sample(ts); vs=curve.derivatives(ts)[0]
    slots={}
    for owner,other in [('a1','a2'),('a2','a1')]:
        initial_illegal=None; contacts={};entries={};witnesses={};all_contacts={};weapon_gap=1.;robot_gap=1.
        minima={r.name:float('inf') for r in geometry.regions}; other_entry=False
        mapping=scene.agents[owner]
        other_collision_ids=[gid for gid in range(scene.model.ngeom) if scene.model.geom_contype[gid]
            and (mujoco.mj_id2name(scene.model,mujoco.mjtObj.mjOBJ_GEOM,gid) or '').startswith(other+'_')]
        blade_ids=blade_collision_geom_ids(scene.model,owner+'_')
        other_weapon_ids=(scene.model.geom(other+'_shinai_guard_collision').id,
                          *blade_collision_geom_ids(scene.model,other+'_'),
                          *(scene.model.geom(other+'_shinai_handle_collision_'+str(j)).id for j in range(4)))
        for i,(t,q,v) in enumerate(zip(ts,qs,vs)):
            scene.data.qpos[mapping['qpos_ids']]=q
            scene.data.qpos[scene.agents[other]['qpos_ids']]=opponent
            scene.data.qvel[:]=0;scene.data.qvel[mapping['dof_ids']]=v
            mujoco.mj_forward(scene.model,scene.data)
            for c in illegal_contacts(scene):
                contacts.setdefault('|'.join(c['geoms']),dict(c,time_s=float(t)))
            for c in scene.data.contact:
                if c.dist>=-1e-8:continue
                names=[mujoco.mj_id2name(scene.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
                all_contacts.setdefault('|'.join(names),{'time_s':float(t),'geoms':names,'depth_m':float(-c.dist)})
            for gi in blade_ids:
                for gj in other_collision_ids:
                    robot_gap=min(robot_gap,float(mujoco.mj_geomDistance(scene.model,scene.data,gi,gj,.1,None)))
                for gj in other_weapon_ids:
                    weapon_gap=min(weapon_gap,float(mujoco.mj_geomDistance(scene.model,scene.data,gi,gj,.1,None)))
            sections=blade_sections_from_scene(scene)[owner]
            for region in geometry.regions:
                best=min(((*point_cylinder_distance(geometry.world_center(region.name,other),b),b.name)
                    for b in sections),key=lambda r:r[0])
                distance,point,gname=best;gap=float(distance-region.radius)
                minima[region.name]=min(minima[region.name],gap)
                if gap<0 and region.name not in entries:
                    entries[region.name]=float(t)
                    gid=scene.model.geom(gname).id;jac=np.zeros((3,scene.model.nv))
                    mujoco.mj_jac(scene.model,scene.data,jac,None,point,int(scene.model.geom_bodyid[gid]))
                    velocity=jac@scene.data.qvel
                    if owner=='a2':velocity*=np.array([-1.,-1.,1.])
                    cosine=float(velocity@unit(direction)/max(np.linalg.norm(velocity),1e-12))
                    witnesses[region.name]={'point_world_m':point.tolist(),
                        'velocity_actor_local_m_s':velocity.tolist(),'intended_direction_cosine':cosine}
            other_entry|=any(g<0 for g in region_gaps(scene,geometry,other).values())
        first=min(entries,key=entries.get) if entries else None
        slots[owner]={'illegal_contacts':list(contacts.values()),'first_entries_s':entries,
            'first_region':first,'entry_witnesses':witnesses,'minimum_gaps_m':minima,
            'opponent_entered_actor_targets':bool(other_entry),
            'target_entry_pass':first==target and entries.get(target,0)>0,
            'direction_pass':target in witnesses and witnesses[target]['intended_direction_cosine']>.35,
            'all_reference_contacts':list(all_contacts.values()),'minimum_weapon_gap_m':weapon_gap,
            'minimum_robot_gap_m':robot_gap,
            'baseline_weapon_clearance_pass':not all_contacts and weapon_gap>=.01 and robot_gap>=.02}
    bounds=model_joint_ranges_rad(load_joint_constraints())
    boundpass=bool(np.all(curve.bezier_controls>=bounds[:,0]) and np.all(curve.bezier_controls<=bounds[:,1]))
    okay=boundpass and all(v['target_entry_pass'] and v['direction_pass'] and not v['illegal_contacts'] and
        v['baseline_weapon_clearance_pass'] and
        not v['opponent_entered_actor_targets'] for v in slots.values())
    return {'sampled_reference_pass':bool(okay),'slots':slots,'analytic_bounds_pass':boundpass,
        'native_dynamics_tested':False,'continuous_collision_certificate':False}, {'time_s':ts,'qpos':qs,'qvel':vs}


def action_aliases(report):
    for a in report['actions'].values():
        a['candidate_pass']=bool(a['sampled_reference_pass'])
        a['validation_stage']='geometry_candidate_only'
    return report


def contact_build(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose fresh output directory')
    output.mkdir(parents=True)
    source,old,center,geometry=source_geometry()
    constraints=load_joint_constraints();bounds=model_joint_ranges_rad(constraints)+np.array([.025,-.025])
    single=build_limited_floor(constraints);solver=GripAxisSolver(single.model,bounds)
    scene=build_limited_duel(geometry.base_distance,initial_qpos=center,constraints=constraints)
    report={'schema_version':1,'status':'growing_contact_centered_arcs','scoring':old['scoring'],
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'coordinate_frame':'Actor-local left/right; target_region is defender-local name.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,Path(__file__))},
        'guards':{'center':{'qpos_rad':center.tolist(),'static_reference_pass':True}},
        'arc_trials':{},'actions':{},'model_source_sha256':scene.source_hashes,
        'construction':'Grow full grip+axis trajectory in both directions from native-success contact geometry. Never fit isolated arbitrary target tip and label it a cut.',
        'limitations':['All published paths remain sampled geometry candidates. No parry or training certificate.',
            'Frozen center is previous engagement guard, not a forced endpoint of every action.']}
    native_names={'left':'left_short_whole_blade_00','right':'right_short_whole_blade_01'}
    chosen={}
    for side in ('left','right'):
        action=next(r for r in old['path_trials'] if r['name']==native_names[side])
        qcontact=np.array(action['knot_qpos'])[2]
        report['arc_trials'][side]=[]
        for scale in (1.,.85,.7,.55):
            rows=grow_contact_arc(solver,qcontact,side,scale)
            qs=np.array([r['qpos_rad'] for r in rows])
            curve=retimed_curve(qs,1.2)
            direction=[0.,-1.,-1.] if side=='left' else [0.,1.,-1.]
            result,arrays=audit_motion(scene,geometry,curve,center,direction,side)
            high_p,high_a,high_t=tool_fk(single.model,qpos=qs[0]);low_p,low_a,low_t=tool_fk(single.model,qpos=qs[-1])
            endpoints={'high':{'grip_m':high_p.tolist(),'axis':high_a.tolist(),'tip_m':high_t.tolist()},
                'low':{'grip_m':low_p.tolist(),'axis':low_a.tolist(),'tip_m':low_t.tolist()}}
            fitpass=all(r['fit_pass'] for r in rows)
            trial={'scale':scale,'rows':rows,'fit_pass':fitpass,'audit':result,'endpoints':endpoints,
                'knot_qpos':qs.tolist(),'knot_time_s':curve.knot_time_s.tolist(),
                'sampled_reference_pass':fitpass and result['sampled_reference_pass']}
            gravity=[]
            cap=np.max(np.abs(single.model.actuator_forcerange),axis=1)*np.abs(single.model.actuator_gear[:,0])
            for q in qs:
                solver.data.qpos[:]=q;solver.data.qvel[:]=0;mujoco.mj_forward(single.model,solver.data)
                gravity.append(solver.data.qfrc_bias.copy())
            gravity=np.array(gravity)
            trial['static_gravity_screen']={'actuator_joint_torque_cap_nm':cap.tolist(),
                'bias_torque_nm_by_knot':gravity.tolist(),
                'maximum_abs_bias_torque_nm':np.max(np.abs(gravity),axis=0).tolist(),
                'start_ratio':(np.abs(gravity[0])/cap).tolist(),'end_ratio':(np.abs(gravity[-1])/cap).tolist(),
                'endpoints_below_cap':bool(np.all(np.abs(gravity[[0,-1]])<cap)),
                'meaning':'Necessary static hold screen, not dynamic torque or tracking certification.'}
            report['arc_trials'][side].append(trial)
            np.savez_compressed(output/f'{side}_arc_scale_{scale:.2f}.npz',**arrays,
                knot_time_s=curve.knot_time_s,knot_qpos=qs,
                requested_grip_m=np.array([r['requested_grip_m'] for r in rows]),
                requested_axis=np.array([r['requested_axis'] for r in rows]))
            write(output/'report.json',report)
            print('arc',side,scale,'fit',fitpass,'screen',result['sampled_reference_pass'],
                'high',np.round(high_t,3),'low',np.round(low_t,3),flush=True)
            if fitpass and result['sampled_reference_pass'] and trial['static_gravity_screen']['endpoints_below_cap'] and side not in chosen:
                chosen[side]=trial
    for side,trial in chosen.items():
        left_to_right=side=='left'
        high_name='left_high' if left_to_right else 'right_high'
        low_name='right_low' if left_to_right else 'left_low'
        qs=np.asarray(trial['knot_qpos'])
        for name,q in [(high_name,qs[0]),(low_name,qs[-1])]:
            p,a,t=tool_fk(single.model,qpos=q)
            report['guards'][name]={'qpos_rad':q.tolist(),'actual_grip_m':p.tolist(),
                'actual_axis':a.tolist(),'actual_tip_m':t.tolist(),
                'static_reference_pass':pair_screen(scene,geometry,q,center)['zero_overlap_start_pass']}
        specs=[('cut_down_left_to_right' if left_to_right else 'cut_down_right_to_left',high_name,low_name,qs,
            [0.,-1. if left_to_right else 1.,-1.]),
            ('cut_up_right_to_left' if left_to_right else 'cut_up_left_to_right',low_name,high_name,qs[::-1],
            [0.,1. if left_to_right else -1.,1.])]
        for name,start,end,nodes,direction in specs:
            curve=retimed_curve(nodes,1.2)
            audit,arrays=audit_motion(scene,geometry,curve,center,direction,side)
            report['actions'][name]={'kind':'attack','start_guard':start,'end_guard':end,
                'target_region':side,'direction_intent_attacker_frame':direction,
                'knot_qpos':nodes.tolist(),'knot_time_s':curve.knot_time_s.tolist(),
                'sampled_reference_pass':audit['sampled_reference_pass'],'audit':audit,
                'native_dynamics_tested':False,'admitted_to_training':False,'spline':curve.report()}
            np.savez_compressed(output/f'{name}.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=nodes)
    report['guard_pair_matrix']={a:{b:pair_screen(scene,geometry,ga['qpos_rad'],gb['qpos_rad'])
        for b,gb in report['guards'].items()} for a,ga in report['guards'].items()}
    report['status']='contact_centered_arcs_screened_not_native_verified'
    action_aliases(report)
    write(output/'report.json',report);(output/'scene.xml').write_text(scene.spec.to_xml())
    return report


def faster_curve(nodes, nominal_duration=3., speed=1.2):
    q=np.asarray(nodes)
    durations=np.maximum(.08,1.9*np.max(np.abs(np.diff(q,axis=0)),axis=1)/speed)
    ts=np.r_[0.,np.cumsum(durations)]
    curve=C2JointSpline(ts,q)
    peak=max(curve.analytic_extrema()['peak_speed_rad_s'])
    if peak>speed:curve=C2JointSpline(ts*(peak/speed)*(1+1e-9),q)
    return curve


def augment_center(source,output):
    """Add true vertical-cut candidate and one previously tested thrust family."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Choose fresh output directory')
    output.mkdir(parents=True)
    report=json.loads((source/'report.json').read_text())
    report['augmentation_source_sha256']=hashlib.sha256((source/'report.json').read_bytes()).hexdigest()
    report['augmentation_implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if 'cut_down_center' in report['actions']:
        report['superseded_center_cut']=report['actions'].pop('cut_down_center')
    for p in source.glob('*.npz'):shutil.copyfile(p,output/p.name)
    shutil.copyfile(source/'scene.xml',output/'scene.xml')
    _,old,_,geometry=source_geometry();center=np.array(report['guards']['center']['qpos_rad'])
    constraints=load_joint_constraints();single=build_limited_floor(constraints)
    solver=GripAxisSolver(single.model,model_joint_ranges_rad(constraints)+np.array([.025,-.025]))
    scene=build_limited_duel(geometry.base_distance,initial_qpos=center,constraints=constraints)
    thrust=next(a for a in old['path_trials'] if a['name']=='right_short_whole_blade_01')
    curve=C2JointSpline(thrust['knot_time_s'],thrust['knot_qpos'])
    audit,arrays=audit_motion(scene,geometry,curve,center,[1.,0.,0.],'right')
    report['actions']['thrust_center']={'kind':'attack','start_guard':'center','end_guard':'center',
        'target_region':'right','direction_intent_attacker_frame':[1.,0.,0.],
        'knot_qpos':curve.knot_qpos.tolist(),'knot_time_s':curve.knot_time_s.tolist(),
        'sampled_reference_pass':audit['sampled_reference_pass'],'audit':audit,
        'native_dynamics_tested':False,'admitted_to_training':False,
        'source_note':'Single retained diagonal forward thrust; not three duplicated scoring-target pokes.'}
    np.savez_compressed(output/'thrust_center.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    upper=next(a for a in old['path_trials'] if a['name']=='upper_short_whole_blade_01')
    upperq=np.asarray(upper['knot_qpos'])[2];p,a,_=tool_fk(single.model,qpos=upperq)
    rh=np.array(report['guards']['right_high']['qpos_rad'])
    ps,axs,_=tool_fk(single.model,qpos=rh);pe,axe,_=tool_fk(single.model,qpos=center)
    report['vertical_trials']=[]
    for scale in (.65,.85,1.):
        pre_p=p+np.array([-.14,0.,.20])*scale;pre_a=unit(a+np.array([0.,0.,.30])*scale)
        post_p=p+np.array([-.24,0.,-.16])*scale;post_a=unit(a+np.array([0.,0.,-.30])*scale)
        tasks=[(ps,axs),(pre_p,pre_a),(p,a),(post_p,post_a),(pe,axe)]
        q=rh.copy();rows=[];nodes=[q.copy()]
        for seg in range(len(tasks)-1):
            (p0,a0),(p1,a1)=tasks[seg:seg+2]
            for u in np.linspace(.2,1.,5):
                pp=(1-u)*p0+u*p1;aa=unit((1-u)*a0+u*a1)
                q,fit=solver.solve(pp,aa,q,regularization=.005,iterations=150)
                nodes.append(q.copy());rows.append(dict(fit,segment=seg,u=float(u),qpos_rad=q.tolist()))
        # Explicit checked joint interpolation to the canonical center, not a reset.
        nodes.append(center.copy())
        curve=faster_curve(nodes,3.5)
        audit,arrays=audit_motion(scene,geometry,curve,center,[0.,0.,-1.],'upper')
        first=audit['slots']['a1']['first_entries_s'].get('upper')
        windup_end=float(curve.knot_time_s[5])
        fitpass=all(r['fit_pass'] for r in rows)
        passed=fitpass and audit['sampled_reference_pass'] and first is not None and first>=windup_end
        trial={'scale':scale,'fit_pass':fitpass,'ik_rows':rows,'audit':audit,
            'sampled_reference_pass':bool(passed),'windup_end_s':windup_end,
            'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
            'end_joint_bridge_delta_rad':(np.asarray(nodes[-1])-np.asarray(nodes[-2])).tolist()}
        report['vertical_trials'].append(trial)
        np.savez_compressed(output/f'vertical_scale_{scale}.npz',**arrays,
            knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        print('vertical',scale,'fit',fitpass,'screen',audit['sampled_reference_pass'],'afterwindup',first is not None and first>=windup_end,flush=True)
        if passed and 'cut_down_center' not in report['actions']:
            report['actions']['cut_down_center']={'kind':'attack','start_guard':'right_high','end_guard':'center',
                'target_region':'upper','direction_intent_attacker_frame':[0.,0.,-1.],
                **{k:trial[k] for k in ('knot_time_s','knot_qpos','sampled_reference_pass','audit','windup_end_s')},
                'native_dynamics_tested':False,'admitted_to_training':False}
            np.savez_compressed(output/'cut_down_center.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        write(output/'report.json',report)
    report['status']='six_attack_design_candidates_not_native_verified'
    report['guard_pair_matrix']={aa:{bb:pair_screen(scene,geometry,ga['qpos_rad'],gb['qpos_rad'])
        for bb,gb in report['guards'].items()} for aa,ga in report['guards'].items()}
    action_aliases(report)
    write(output/'report.json',report)
    return report


def recovery_search(source,output):
    """Search post-cut clearance without treating legal weapon contact as safe."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Choose fresh output')
    output.mkdir(parents=True)
    report=json.loads((source/'report.json').read_text())
    for pth in source.glob('*.npz'):shutil.copyfile(pth,output/pth.name)
    shutil.copyfile(source/'scene.xml',output/'scene.xml')
    constraints=load_joint_constraints();m=build_limited_floor(constraints).model
    solver=GripAxisSolver(m,model_joint_ranges_rad(constraints)+np.array([.025,-.025]))
    g=TriangleGeometry.from_mapping({k:report['scoring'][k] for k in ('base_distance','regions')})
    center=np.asarray(report['guards']['center']['qpos_rad']);scene=build_limited_duel(g.base_distance,initial_qpos=center)
    original=report['actions']['cut_down_right_to_left']; oldq=np.array(original['knot_qpos'])
    qcontact=oldq[10];p,a,_=tool_fk(m,qpos=qcontact)
    prefix=oldq[:11];trials=[];passing=[]
    report['right_recovery_trials']=trials
    cap=np.max(np.abs(m.actuator_forcerange),axis=1)
    for arch in (-.24,-.16,-.08,0.,.08,.16,.24):
        for retract in (.4,.6,.8):
            q=qcontact.copy();rows=[];nodes=list(prefix.copy())
            for fraction in np.linspace(.1,1.,10):
                s=.85*fraction
                pp=p+[-retract*s,.18*s,-.18*s+arch*np.sin(np.pi*s)]
                aa=unit(a+[0.,.45*s,-.45*s])
                q,fit=solver.solve(pp,aa,q,regularization=.005,iterations=130)
                rows.append(dict(fit,qpos_rad=q.tolist()));nodes.append(q.copy())
            curve=faster_curve(nodes,3.5)
            fitpass=all(r['fit_pass'] for r in rows)
            if fitpass:
                audit,arrays=audit_motion(scene,g,curve,center,[0,1,-1],'right',.035)
            else:
                audit={'sampled_reference_pass':False,'reason':'IK_failure'};arrays={}
            solver.data.qpos[:]=nodes[-1];solver.data.qvel[:]=0;mujoco.mj_forward(m,solver.data)
            endbias=solver.data.qfrc_bias.copy()
            good=fitpass and audit['sampled_reference_pass'] and np.all(abs(endbias)<cap)
            idx=len(trials);trial={'arch_m':arch,'retraction_m_per_phase':retract,
                'fit_pass':fitpass,'audit':audit,'end_static_bias_nm':endbias.tolist(),
                'candidate_pass':bool(good),'rows':rows,'knot_time_s':curve.knot_time_s.tolist(),
                'knot_qpos':curve.knot_qpos.tolist()}
            trials.append(trial)
            if good:passing.append(idx)
            if arrays:np.savez_compressed(output/f'recovery_{idx:02d}.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
            write(output/'report.json',report)
            print('recovery',idx,arch,retract,'fit',fitpass,'pass',bool(good),flush=True)
    report['right_recovery_passing_indices']=passing
    if passing:
        # First prefer more clearance, then lower endpoint torque and less motion.
        index=min(passing,key=lambda i:(-min(v['minimum_weapon_gap_m'] for v in trials[i]['audit']['slots'].values()),
            max(abs(np.array(trials[i]['end_static_bias_nm']))/cap)))
        selected=trials[index];nodes=np.asarray(selected['knot_qpos'])
        report['right_recovery_selected_index']=index
        endp,enda,endt=tool_fk(m,qpos=nodes[-1])
        report['guards']['left_low']={'qpos_rad':nodes[-1].tolist(),'actual_grip_m':endp.tolist(),
            'actual_axis':enda.tolist(),'actual_tip_m':endt.tolist(),
            'static_reference_pass':pair_screen(scene,g,nodes[-1],center)['zero_overlap_start_pass']}
        for name,start,end,qs,direction in [('cut_down_right_to_left','right_high','left_low',nodes,[0,1,-1]),
            ('cut_up_left_to_right','left_low','right_high',nodes[::-1],[0,-1,1])]:
            curve=faster_curve(qs,3.5)
            audit,arrays=audit_motion(scene,g,curve,center,direction,'right',.02)
            report['actions'][name]={'kind':'attack','start_guard':start,'end_guard':end,'target_region':'right',
                'knot_qpos':qs.tolist(),'knot_time_s':curve.knot_time_s.tolist(),
                'direction_intent_attacker_frame':direction,'sampled_reference_pass':audit['sampled_reference_pass'],
                'audit':audit,'native_dynamics_tested':False,'admitted_to_training':False,
                'candidate_pass':audit['sampled_reference_pass'],'validation_stage':'geometry_candidate_only'}
            np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=qs)
    report['status']='right_weapon_clearance_search_finished_not_native_verified'
    action_aliases(report);write(output/'report.json',report)
    return report


def refine_up_search(source,output):
    """Keep accepted guard endpoints/down-cut; vary only ascending interior."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Choose fresh output')
    output.mkdir(parents=True)
    report=json.loads((source/'report.json').read_text())
    for pth in source.glob('*.npz'):shutil.copyfile(pth,output/pth.name)
    shutil.copyfile(source/'scene.xml',output/'scene.xml')
    constraints=load_joint_constraints();m=build_limited_floor(constraints).model
    solver=GripAxisSolver(m,model_joint_ranges_rad(constraints)+np.array([.025,-.025]))
    g=TriangleGeometry.from_mapping({k:report['scoring'][k] for k in ('base_distance','regions')})
    center=np.asarray(report['guards']['center']['qpos_rad']);scene=build_limited_duel(g.base_distance,initial_qpos=center)
    oldq=np.array(report['actions']['cut_down_right_to_left']['knot_qpos'])
    p,a,_=tool_fk(m,qpos=oldq[10]); trials=[];passing=[]
    report['up_interior_trials']=trials
    for dx in (-.04,-.08,-.12):
        for dy in (-.08,0.,.08):
            for dz in (-.08,0.,.08):
                q=oldq[10].copy();rows=[];nodes=list(oldq[:11].copy())
                for i,f in enumerate(np.linspace(.1,1.,10)):
                    s=.85*f;bump=np.sin(np.pi*f)
                    pp=p+[-.6*s,.18*s,-.18*s-.16*np.sin(np.pi*s)]+bump*np.array([dx,dy,dz])
                    aa=unit(a+[0,.45*s,-.45*s])
                    q,fit=solver.solve(pp,aa,q,regularization=.004,iterations=150)
                    if i==9:q=oldq[-1].copy()
                    nodes.append(q.copy());rows.append(dict(fit,qpos_rad=q.tolist()))
                up=np.array(nodes)[::-1];curve=faster_curve(up,4.)
                fitpass=all(r['fit_pass'] for r in rows)
                if fitpass:audit,arrays=audit_motion(scene,g,curve,center,[0,-1,1],'right',.035)
                else:audit={'sampled_reference_pass':False,'reason':'IK_failure'};arrays={}
                idx=len(trials);good=fitpass and audit['sampled_reference_pass']
                trial={'bump_m':[dx,dy,dz],'candidate_pass':bool(good),'rows':rows,'audit':audit,
                    'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':up.tolist()}
                trials.append(trial)
                if good:passing.append(idx)
                if arrays:np.savez_compressed(output/f'up_refine_{idx:02d}.npz',**arrays,
                    knot_time_s=curve.knot_time_s,knot_qpos=up)
                print('uprefine',idx,[dx,dy,dz],'fit',fitpass,'pass',bool(good),flush=True)
                write(output/'report.json',report)
    report['up_refinement_passing_indices']=passing
    if passing:
        idx=max(passing,key=lambda i:min(v['minimum_robot_gap_m'] for v in trials[i]['audit']['slots'].values()))
        selected=trials[idx];qs=np.array(selected['knot_qpos']);curve=C2JointSpline(selected['knot_time_s'],qs)
        audit,arrays=audit_motion(scene,g,curve,center,[0,-1,1],'right',.02)
        report['up_refinement_selected_index']=idx
        report['actions']['cut_up_left_to_right']={'kind':'attack','start_guard':'left_low','end_guard':'right_high',
            'target_region':'right','direction_intent_attacker_frame':[0,-1,1],
            'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':qs.tolist(),'audit':audit,
            'sampled_reference_pass':audit['sampled_reference_pass'],'native_dynamics_tested':False,'admitted_to_training':False}
        np.savez_compressed(output/'cut_up_left_to_right.npz',**arrays,knot_time_s=curve.knot_time_s,knot_qpos=qs)
    report['status']='up_interior_refinement_complete_not_native_verified'
    action_aliases(report);write(output/'report.json',report)
    return report


def defence_reference_audit(scene,geometry,curve,opponent,period=.03):
    """Idle-opponent geometry only; deliberate moving blade contact is untested."""
    ts=np.linspace(0,curve.duration_s,int(np.ceil(curve.duration_s/period))+1)
    qs=curve.sample(ts);records={};entries=[]
    for t,q in zip(ts,qs):
        scene.data.qpos[scene.agents['a1']['qpos_ids']]=q
        scene.data.qpos[scene.agents['a2']['qpos_ids']]=opponent
        scene.data.qvel[:]=0;mujoco.mj_forward(scene.model,scene.data)
        for c in illegal_contacts(scene):records.setdefault('|'.join(c['geoms']),dict(c,time_s=float(t)))
        if any(v<0 for v in region_gaps(scene,geometry,'a1').values()):entries.append(float(t))
    bounds=model_joint_ranges_rad(load_joint_constraints())
    okay=not records and not entries and np.all(curve.bezier_controls>=bounds[:,0]) and np.all(curve.bezier_controls<=bounds[:,1])
    return {'sampled_reference_pass':bool(okay),'illegal_contacts':list(records.values()),
        'unintended_attack_overlap_times_s':entries,'defence_effectiveness_tested':False,
        'reference_only_idle_opponent_screen':True}, {'time_s':ts,'qpos':qs,'qvel':curve.derivatives(ts)[0]}


def add_parry_candidates(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Choose fresh output directory')
    output.mkdir(parents=True);report=json.loads((source/'report.json').read_text())
    for pth in source.glob('*.npz'):shutil.copyfile(pth,output/pth.name)
    shutil.copyfile(source/'scene.xml',output/'scene.xml')
    cons=load_joint_constraints();bounds=model_joint_ranges_rad(cons)+np.array([.025,-.025])
    single=build_limited_floor(cons);model=single.model;data=mujoco.MjData(model)
    solver=GripAxisSolver(model,bounds)
    geometry=TriangleGeometry.from_mapping({k:report['scoring'][k] for k in ('base_distance','regions')})
    guards=report['guards'];center=np.array(guards['center']['qpos_rad'])
    scene=build_limited_duel(geometry.base_distance,initial_qpos=center)
    report['parry_design_trials']={};report['context_routes']={}
    def save_action(name,nodes,start,end,region,note,extras=None):
        curve=faster_curve(nodes,3.,1.2)
        audit,arrays=defence_reference_audit(scene,geometry,curve,center)
        action={'kind':'defence','start_guard':start,'end_guard':end,'target_region':region,
            'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
            'candidate_pass':audit['sampled_reference_pass'],'sampled_reference_pass':audit['sampled_reference_pass'],
            'audit':audit,'successful_defence':False,'native_dynamics_tested':False,
            'validation_stage':'idle_reference_geometry_only_not_defence','admitted_to_training':False,'note':note}
        if extras:action.update(extras)
        np.savez_compressed(output/(name+'.npz'),**arrays,knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
        return action
    for side,attack in [('left','cut_down_left_to_right_05'),('right','cut_down_right_to_left_07')]:
        native=ROOT/'outputs/multiguard_v1_native'/attack
        evidence=json.loads((native/'result.json').read_text())
        hit=float(evidence['first_actual_entry']['time_s'])
        with np.load(native/'native/trace.npz') as trace:
            atimes=trace['time_s'].copy();aq=trace['qpos'][:,:7].copy()
        found=[];trials=[]
        for dt in ((.18,.35) if side=='left' else (.18,.35,.08,.04)):
            when=hit-dt
            qatt=aq[np.argmin(abs(atimes-when))]
            scene.data.qpos[:7]=qatt;scene.data.qpos[7:]=center;mujoco.mj_forward(scene.model,scene.data)
            segments=blade_sections_from_scene(scene)['a1'];begin,end=segments[0].start,segments[-1].end
            incoming=unit(end-begin)
            for fraction in (.3,.6,.9):
                crossing=begin+fraction*(end-begin)
                target=(crossing-np.array([geometry.base_distance,0,0]))*[-1,-1,1]
                for distance in (.35,.65):
                    def residual(state):
                        q,along=state[:7],state[7];p,a,_=tool_fk(model,data,q)
                        return np.r_[10.*(p+along*a-target),.025*(q-center),
                            .3*max(0.,abs((a*np.array([-1,-1,1]))@incoming)-.8)]
                    fit=least_squares(residual,np.r_[center,distance],
                        bounds=(np.r_[bounds[:,0],.05],np.r_[bounds[:,1],.77]),max_nfev=130)
                    q=fit.x[:7];p,a,_=tool_fk(model,data,q)
                    err=float(np.linalg.norm(p+fit.x[7]*a-target))
                    qend=np.array(guards['left_high' if side=='left' else 'right_high']['qpos_rad'])
                    shift=np.array([-.06,.10 if side=='left' else -.10,.06])
                    moved,mfit=solver.solve(p+shift,a,q,iterations=100)
                    action=save_action('temporary_parry', [center,q,moved,qend], 'center',
                        'left_high' if side=='left' else 'right_high',side,'temporary')
                    valid=err<.015 and mfit['fit_pass'] and action['candidate_pass']
                    row={'native_baseline':str(native),'source_hit_time_s':hit,'crossing_time_s':when,
                        'incoming_fraction':fraction,'defending_material_distance_m':float(fit.x[7]),
                        'intercept_error_m':err,'q_intercept':q.tolist(),'q_sweep':moved.tolist(),
                        'audit':action['audit'],'candidate_pass':bool(valid),
                        'joint_travel_rad':float(np.sum(abs(q-center))+np.sum(abs(qend-moved))),
                        'intercept_relative_time_s':action['knot_time_s'][1],
                        'suggested_defence_offset_s':when-.75-action['knot_time_s'][1]}
                    trials.append(row)
                    if valid:found.append(row)
        report['parry_design_trials'][side]=trials
        if found:
            chosen=min(found,key=lambda r:(r['suggested_defence_offset_s']<.05,r['joint_travel_rad']))
            qend=np.array(guards['left_high' if side=='left' else 'right_high']['qpos_rad'])
            action=save_action('parry_'+side,[center,chosen['q_intercept'],chosen['q_sweep'],qend],
                'center','left_high' if side=='left' else 'right_high',side,
                'Bounded intercept proposal from completed native attack; only idle-opponent geometry passed.',
                {'intercept_design':chosen,'suggested_defence_offset_s':chosen['suggested_defence_offset_s'],
                 'scripted_timing_not_reactive_policy':True})
            report['actions']['parry_'+side]=action
            if side=='left':
                context=save_action('parry_left__from_left_high',
                    [guards['left_high']['qpos_rad'],chosen['q_intercept'],chosen['q_sweep'],guards['right_high']['qpos_rad']],
                    'left_high','right_high','left','Context route needs a new baseline with LH defender; no certificate transfer.',
                    {'policy_family_id':'parry_left','new_baseline_required':True})
                report['context_routes']['parry_left__from_left_high']=context
        print('parry',side,'eligible',len(found),'of',len(trials),flush=True)
        write(output/'report.json',report)
    oldbank=json.loads((ROOT/'outputs/compact_bank_v1/build_engagement_02/bank.json').read_text())
    old=oldbank['actions']['defence_upper']
    nodes=[oldbank['nodes'][n] for n in old['waypoint_nodes']]
    nodes[-1]=guards['left_high']['qpos_rad']
    report['actions']['parry_high']=save_action('parry_high',nodes,'center','left_high','upper',
        'Previous upper interception geometry, but new exit and new attacks; old defence certificate explicitly not inherited.')
    report['status']='nine_family_geometric_candidates_with_unverified_parries'
    report['defence_effectiveness_validated']=False
    report['training_admitted']=False
    action_aliases(report);write(output/'report.json',report)
    return report


def finalize_bundle(source,output):
    """Materialize explicit context routes without inventing a tenth policy ID."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Choose fresh output')
    output.mkdir(parents=True)
    report=json.loads((source/'report.json').read_text())
    for p in source.glob('*.npz'):shutil.copyfile(p,output/p.name)
    shutil.copyfile(source/'scene.xml',output/'scene.xml')
    report['bundle_input_sha256']=hashlib.sha256((source/'report.json').read_bytes()).hexdigest()
    report['bundle_builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,row in report['actions'].items():row['family']=name
    for name,row in report.get('context_routes',{}).items():
        report['actions'][name]=dict(row,family=row['policy_family_id'])
    report['canonical_policy_families']=['cut_down_left_to_right','cut_down_right_to_left',
        'cut_up_left_to_right','cut_up_right_to_left','cut_down_center','thrust_center',
        'parry_left','parry_right','parry_high']
    report['policy_action_count_excluding_wait']=9
    report['explicit_reference_route_count']=len(report['actions'])
    report['stage']='bounded_candidate_bundle_not_training_admitted'
    report['action_mask_rule']='Only exact physically settled start_guard routes may be selected; contextual routes share their canonical family ID.'
    report['unverified_items']=['Three parry families have no successful full-window native pair certificate.',
        'Right parry only has a proposed 64 ms post-onset initiation window; reaction and tracking must be measured.',
        'Low guards lack direct defensive-family exits; report this response gap before training.',
        'No contextual route certificate or all-slots/all-sequences certificate is implied.',
        'Source geometry flags are not substitutions for the root native evidence and common platform admission gates.']
    action_aliases(report);write(output/'report.json',report)
    return report


def build(output, *, guards_only=False, family='forward'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Retain trials: select a fresh output directory')
    output.mkdir(parents=True)
    source, old, seed, geometry = source_geometry()
    constraints = load_joint_constraints()
    bounds = model_joint_ranges_rad(constraints)+np.array([.04, -.04])
    if family=='offset':
        bounds[0]=np.deg2rad([210.,305.]);bounds[2]=np.deg2rad([145.,250.])
        bounds[4]=np.deg2rad([110.,225.]);bounds[6]=np.deg2rad([-20.,87.])
    single = build_limited_floor(constraints)
    scene = build_limited_duel(geometry.base_distance, initial_qpos=seed, constraints=constraints)
    solver = GripAxisSolver(single.model, bounds)
    report = {'schema_version': 1, 'status': 'searching_multi_guard_candidates',
        'candidate_only': True, 'admitted_to_training': False, 'native_dynamics_tested': False,
        'scoring': old['scoring'], 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, Path(__file__), ROOT/'config/joint_constraints_observed.json')},
        'model_source_sha256': scene.source_hashes, 'joint_solver_margin_rad': .04,
        'coordinate_frame': 'actor local, +x toward opponent, +y actor left, +z world up',
        'guards': {}, 'guard_trials': {}, 'actions': {},
        'limitations': ['Bounded IK and sampled geometry are not dynamic validation.',
            'No opponent intention or future trajectory is used at runtime.',
            'Multi-guard masks, reactive defence, shared referee, and independent A2 platform require separate tests.']}
    report['guard_family']=family
    report['guards'], report['guard_trials'] = guard_search(solver, scene, geometry, seed,family=family)
    report['guard_pair_matrix'] = {a: {b: pair_screen(scene, geometry, ga['qpos_rad'], gb['qpos_rad'])
        for b,gb in report['guards'].items()} for a,ga in report['guards'].items()}
    report['status'] = 'guard_search_complete_candidate_only'
    write(output/'report.json', report)
    (output/'scene.xml').write_text(scene.spec.to_xml())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--guards-only', action='store_true')
    p.add_argument('--family', choices=('forward','offset'),default='forward')
    p.add_argument('--contact-arcs',action='store_true')
    p.add_argument('--add-center-actions')
    p.add_argument('--recovery-search')
    p.add_argument('--refine-up')
    p.add_argument('--add-parries')
    p.add_argument('--finalize-bundle')
    a = p.parse_args()
    if a.finalize_bundle:finalize_bundle(a.finalize_bundle,a.output)
    elif a.add_parries:add_parry_candidates(a.add_parries,a.output)
    elif a.refine_up:refine_up_search(a.refine_up,a.output)
    elif a.recovery_search:recovery_search(a.recovery_search,a.output)
    elif a.add_center_actions:augment_center(a.add_center_actions,a.output)
    elif a.contact_arcs:contact_build(a.output)
    else:build(a.output, guards_only=a.guards_only,family=a.family)


if __name__ == '__main__':
    main()
