"""Bounded signed-tip-up left-parry authoring; no dynamics or policy claim.

Only new candidates are written.  Future attack samples are offline authoring
targets, never policy observations.  Passing this screen does not certify a
parry: the unchanged matched full-window native probe remains necessary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .a1_dynamics_screen import screen_curve
from .duel_scene import classify_contact
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import defence_reference_audit, unit
from .triangle_task import TriangleGeometry
from plan1_skills.matching import tool_fk


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), ensure_ascii=False, indent=2)+"\n")


def build(source, output, *, max_j0_degrees=35., max_solutions=3):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve preceding evidence; select a fresh output")
    raw = source.read_bytes(); original = json.loads(raw)
    guard_name = "left_guard_positive"
    entry = np.array(original["guards"][guard_name]["qpos_rad"])
    cons = load_joint_constraints(); physical = model_joint_ranges_rad(cons)
    bounds = physical + [.15, -.15]
    bounds[0] = [max(bounds[0, 0], entry[0]-np.deg2rad(max_j0_degrees)),
                 min(bounds[0, 1], entry[0]+np.deg2rad(max_j0_degrees))]
    model = build_limited_floor(cons).model; data = mujoco.MjData(model)
    caps = np.max(abs(model.actuator_forcerange), axis=1)*abs(model.actuator_gear[:, 0])
    geometry = TriangleGeometry.from_mapping({k: original['scoring'][k] for k in ('base_distance','regions')})
    scene = build_limited_duel(geometry.base_distance, initial_qpos=entry, constraints=cons)
    attack = original['actions']['cut_down_left_to_right']
    attack_curve = C2JointSpline(attack['knot_time_s'], attack['knot_qpos'])
    pairs = [(gid, model.geom(f'arm_proxy_{j}').id)
             for gid in blade_collision_geom_ids(model, '') for j in range(7)]
    minimum_up = float(np.sin(np.deg2rad(25.)))
    authoring_up = float(np.sin(np.deg2rad(35.)))

    def clearance():
        return float(min(mujoco.mj_geomDistance(model,data,a,b,.08,None) for a,b in pairs))

    def interior_seed(seed):
        # Initial guesses only, never clipping or repairing an output trajectory.
        seed = np.array(seed, copy=True)
        for j in range(7):
            if not bounds[j,0] < seed[j] < bounds[j,1]:
                seed[j] = .5*(bounds[j,0]+bounds[j,1])
        return seed

    def fit(point, incoming, seed, along_seed=.45):
        seed = interior_seed(seed)
        def residual(x):
            q, along = x[:7], x[7]
            grip, axis, _ = tool_fk(model, data, q)
            gap = clearance()
            # Signed upward direction is deliberate. abs(axis.z) is incorrect.
            return np.r_[18.*(grip+along*axis-point),
                10.*max(0., authoring_up-axis[2]),
                10.*max(0., .36-grip[2]),
                .8*max(0., abs(axis@incoming)-.60),
                12.*max(0., .012-gap),
                .012*np.array([3.,1.,1.,1.,1.,1.,1.])*(q-seed),
                .012*(along-along_seed)]
        result = least_squares(residual, np.r_[seed,along_seed],
            bounds=(np.r_[bounds[:,0],.18],np.r_[bounds[:,1],.76]),
            max_nfev=160, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q,along=result.x[:7],float(result.x[7]);grip,axis,_=tool_fk(model,data,q)
        gap=clearance();data.qvel[:]=0.;mujoco.mj_forward(model,data)
        gravity=float(max(abs(data.qfrc_bias)/caps))
        err=float(np.linalg.norm(grip+along*axis-point));cross=float(abs(axis@incoming))
        passed=err<=.015 and axis[2]>=minimum_up and grip[2]>=.35 and gap>=.005 and cross<=.72 and gravity<1.
        return q, {'point_error_m':err,'material_distance_m':along,'axis':axis.tolist(),
            'grip_m':grip.tolist(),'elevation_deg':float(np.rad2deg(np.arcsin(axis[2]))),
            'minimum_self_blade_arm_clearance_m':gap,'incoming_abs_axis_dot':cross,
            'static_torque_ratio':gravity,'nfev':int(result.nfev),'pose_screen_pass':bool(passed)}

    output.mkdir(parents=True)
    snapshot=output/'builder_snapshot.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report={'schema_version':1,'status':'bounded_signed_tip_up_geometry_search',
        'candidate_only':True,'admitted_to_training':False,'native_dynamics_tested':False,
        'guards':copy.deepcopy(original['guards']),'scoring':copy.deepcopy(original['scoring']),
        'actions':{'cut_down_left_to_right':copy.deepcopy(attack)},'pose_trials':[], 'path_trials':[],
        'entry_exit_guard':guard_name,'J0_bounds_deg':np.rad2deg(bounds[0]).tolist(),
        'joint_authoring_bounds_rad':bounds.tolist(),
        'orientation_requirement':{'active_signed_axis_z_min':minimum_up,'minimum_elevation_degrees':25.,
            'authoring_elevation_degrees':35.,
            'grip_z_min_m':.35,'phase':'pre/block/sweep/hold and sampled interpolation between these nodes',
            'entry_and_return_not_required_upright':True},
        'source_sha256':{str(source):hashlib.sha256(raw).hexdigest(),str(snapshot):hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        'limitations':['Offline future trajectory targets are authoring inputs, not observations.',
            'No native integration, actual force, full-window defence, observed reaction, or training admission.',
            'New timing and orientation invalidate inheritance of earlier defence certificates.',
            'This search does not establish bilateral coverage or defence-switch compatibility.']}
    # Deliberately permit either J2 branch; no J2 >= pi authoring restriction.
    rng=np.random.default_rng(251125)
    seeds=[interior_seed(entry)]
    for j5 in (60.,100.):
        for j6 in (-35.,35.):
            s=entry.copy();s[5:]=np.deg2rad([j5,j6]);seeds.append(interior_seed(s))
    for _ in range(3):
        s=bounds[:,0]+rng.uniform(.12,.88,7)*(bounds[:,1]-bounds[:,0]);
        s[0]=entry[0]+np.deg2rad(rng.uniform(-20,20));seeds.append(s)
    candidates=[]
    for target_i,(world_time,attack_material) in enumerate(((3.15,.55),(3.15,.72),(3.55,.55),(3.55,.72),(3.9,.55),(3.9,.72))):
        ap,aa,tip=tool_fk(model,qpos=attack_curve.sample(world_time-.75))
        if attack_material>np.linalg.norm(tip-ap):raise ValueError('Target outside actual sword')
        incoming=aa*np.array([-1.,-1.,1.])
        point=(ap+attack_material*aa-[geometry.base_distance,0.,0.])*[-1.,-1.,1.]
        for seed_i,seed in enumerate(seeds):
            q,info=fit(point,incoming,seed)
            row={'id':f'{target_i:02d}_{seed_i:02d}','world_target_time_s':world_time,
                'attack_material_m':attack_material,'target_point_m':point.tolist(),
                'incoming_axis':incoming.tolist(),'qpos_rad':q.tolist(),**info}
            report['pose_trials'].append(row);write(output/'report.json',report)
            print('pose',row['id'],'pass',info['pose_screen_pass'],'err',round(info['point_error_m'],4),
                'elevation',round(info['elevation_deg'],1),'gripz',round(info['grip_m'][2],3),
                'static',round(info['static_torque_ratio'],3),flush=True)
            if info['pose_screen_pass']:
                candidates.append((q,row))
        if len(candidates)>=max_solutions*2:break
    def moving_reference(curve, active):
        """Only sampled prescribed geometry, never an integrated-contact claim."""
        first=None;illegal=[];inverted=[];maximum_depth=0.
        for time in np.arange(.75, max(.75+curve.duration_s, .75+attack_curve.duration_s)+.001, .02):
            qa=attack_curve.sample(time-.75);qd=curve.sample(time-.75)
            scene.data.qpos[scene.agents['a1']['qpos_ids']]=qa
            scene.data.qpos[scene.agents['a2']['qpos_ids']]=qd
            scene.data.qvel[:]=0.;mujoco.mj_forward(scene.model,scene.data)
            for contact in scene.data.contact:
                if contact.dist>=0:continue
                category=classify_contact(scene.model,contact,rules_version='whole_blade_first_touch_v2')
                names=[scene.model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
                if category not in ('cross_blade_blade','cross_blade_guard'):
                    if len(illegal)<12:illegal.append({'time_s':float(time),'category':category,'geoms':names,'depth_m':float(-contact.dist)})
                    continue
                maximum_depth=max(maximum_depth,float(-contact.dist))
                grip,axis,_=tool_fk(model,data,qd)
                if axis[2]<minimum_up or grip[2]<.35:
                    if len(inverted)<12:inverted.append({'time_s':float(time),'axis_z':float(axis[2]),'grip_z':float(grip[2])})
                if first is None:
                    normal=contact.frame[:3].copy()
                    a_first=names[0].startswith('a1_')
                    if not a_first:normal*=-1
                    moments={}
                    for slot,geom,force in (('a1',contact.geom1 if a_first else contact.geom2,-normal),
                                            ('a2',contact.geom2 if a_first else contact.geom1,normal)):
                        jac=np.zeros((3,scene.model.nv))
                        mujoco.mj_jac(scene.model,scene.data,jac,None,contact.pos,int(scene.model.geom_bodyid[geom]))
                        moments[slot]=(jac[:,scene.agents[slot]['dof_ids']].T@force).tolist()
                    first={'time_s':float(time),'defence_phase_s':float(time-.75),'geoms':names,
                        'normal_attacker_to_defender_world':normal.tolist(),
                        'joint_normal_moment_per_newton':moments,'defender_axis':axis.tolist(),
                        'defender_grip_m':grip.tolist(),'in_declared_active_interval':bool(active[0]<=time-.75<=active[1])}
        return {'first_prescribed_weapon_contact':first,'first_illegal_samples':illegal,
            'weapon_contacts_below_orientation_requirement':inverted,'maximum_prescribed_weapon_penetration_m':maximum_depth,
            'meaning':'Prescribed kinematics can interpenetrate; this diagnoses timing/orientation, not physical deflection.',
            'sample_period_s':.02,'native_dynamics_tested':False}

    # Fit short local motion around each upright solution, then keep the whole
    # actual entry/exit interpolation exposed to the original idle screen.
    candidates.sort(key=lambda qr:qr[1]['point_error_m']+.01*qr[1]['static_torque_ratio']+
                    .002*np.linalg.norm(np.array(qr[1]['qpos_rad'])-entry))
    for index,(block,row) in enumerate(candidates[:max_solutions]):
        point=np.array(row['target_point_m']);incoming=np.array(row['incoming_axis'])
        axis=np.array(row['axis']);normal=unit(np.cross(incoming,axis))
        # Prefer the normal direction that moves the defender laterally outward.
        if normal[1]>0:normal*=-1
        pre,pf=fit(point+.055*normal,incoming,block,row['material_distance_m'])
        sweep,sf=fit(point-.055*normal,incoming,block,row['material_distance_m'])
        # Contact clock is targeted at row time; native tracking must establish
        # whether real contact precedes the unchanged baseline's genuine hit.
        arrival=row['world_target_time_s']-.75
        times=np.array([0.,max(.6,arrival-.55),arrival,arrival+.6,6.4,8.6])
        nodes=np.array([entry,pre,block,sweep,sweep,entry])
        curve=C2JointSpline(times,nodes)
        sample_t=np.unique(np.r_[np.linspace(0,curve.duration_s,321),times]);upvalues=[];zvalues=[];ratios=[]
        for time,q in zip(sample_t,curve.sample(sample_t)):
            p,a,_=tool_fk(model,data,q);data.qvel[:]=0;mujoco.mj_forward(model,data)
            ratios.append(float(max(abs(data.qfrc_bias)/caps)))
            if times[1]<=time<=times[4]:upvalues.append(float(a[2]));zvalues.append(float(p[2]))
        idle,_=defence_reference_audit(scene,geometry,curve,attack_curve.knot_qpos[0],period=.03)
        dynamics=screen_curve(model,curve,period_s=.03)
        peak=float(max(curve.analytic_extrema()['peak_speed_rad_s']))
        orientation=bool(min(upvalues)>=minimum_up and min(zvalues)>=.35)
        moving=moving_reference(curve,[float(times[1]),float(times[4])])
        passed=bool(pf['pose_screen_pass'] and sf['pose_screen_pass'] and orientation and max(ratios)<1. and idle['sampled_reference_pass'])
        name=f'parry_left_upright_{index:02d}'
        trial={'name':name,'pose_id':row['id'],'pre':pf,'block':row,'sweep':sf,'normal':normal.tolist(),
            'knot_time_s':times.tolist(),'knot_qpos':nodes.tolist(),
            'active_axis_z_min':min(upvalues),'active_grip_z_min_m':min(zvalues),
            'orientation_screen_pass':orientation,'whole_path_static_torque_ratio':max(ratios),
            'peak_reference_speed_rad_s':peak,'idle_geometry_audit':idle,
            'ideal_free_space_dynamics':dynamics,'moving_prescribed_geometry':moving,'candidate_screen_pass':passed,
            'suggested_defence_offset_s':0.}
        report['path_trials'].append(trial)
        if passed:
            report['actions'][name]={'kind':'defence','family':'parry_left','start_guard':guard_name,
                'end_guard':guard_name,'target_region':'left','candidate_pass':True,'sampled_reference_pass':True,
                'native_dynamics_tested':False,'admitted_to_training':False,
                'orientation_requirement':report['orientation_requirement'],
                'active_interval_s':[float(times[1]),float(times[4])],
                'min_elevation_deg':25.,'min_grip_height':.35,
                'knot_time_s':times.tolist(),'knot_qpos':nodes.tolist(),
                'suggested_defence_offset_s':0.}
        write(output/'report.json',report)
        print('path',name,'pass',passed,'orientation',orientation,'idle',idle['sampled_reference_pass'],
            'static',round(max(ratios),3),'speed',round(peak,3),'ideal',round(dynamics['worst_utilization'],3),flush=True)
    report['source_unchanged']=source.read_bytes()==raw
    report['status']='upright_geometry_candidates_not_native_verified' if len(report['actions'])>1 else 'no_admissible_upright_path_in_bounded_search'
    (output/'scene.xml').write_text(scene.spec.to_xml());write(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/a1_chain_geometry/chain12/report.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-j0-degrees',type=float,default=35.)
    p.add_argument('--max-solutions',type=int,default=3)
    a=p.parse_args();build(a.source,a.output,max_j0_degrees=a.max_j0_degrees,max_solutions=a.max_solutions)


if __name__=='__main__':main()
