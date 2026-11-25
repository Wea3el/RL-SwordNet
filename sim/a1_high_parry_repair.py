"""Bounded high-parry contact repair, with unchanged attack and physical model.

Authors whole-sword crossing points upstream of the upper scoring sphere, not
an immunity pose. Future attack samples are offline design inputs only. These
reference candidates require complete matched native tests before use.
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

from plan1_skills.matching import tool_fk
from .a1_dynamics_screen import screen_curve
from .a1_guard_defence_screen import tool_point
from .a1_profile_bank import blade_identity
from .a1_recovery_geometry import exact_scene, static_screen
from .assembly import _finite_json
from .authoring_bounds import intersect_authoring_bounds
from .blade_geometry import blade_collision_geom_ids
from .duel_scene import classify_contact
from .fencing_rules import _point_cylinder_distances
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


def unit(x):
    return np.asarray(x)/np.linalg.norm(x)


def _authoring_bounds(physical, entry):
    interior = np.asarray(physical) + np.deg2rad([3., -3.])
    requested = interior.copy()
    requested[0] = (max(interior[0, 0], entry[0]-np.deg2rad(38)),
                    min(interior[0, 1], entry[0]+np.deg2rad(38)))
    requested[2, 1] = np.deg2rad(350)
    requested[3, 1] = np.deg2rad(115)
    requested[5, 0] = np.deg2rad(23)
    requested[6] = np.deg2rad([-82, 82])
    return intersect_authoring_bounds(interior, requested)


def save(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2)+'\n')


def radius_lookup(model, data, q):
    p, a, _ = tool_fk(model, data, q)
    rows = []
    for gid in blade_collision_geom_ids(model, ''):
        mid = float((data.geom_xpos[gid]-p)@a)
        rows.append((mid-model.geom_size[gid,1], mid+model.geom_size[gid,1], model.geom_size[gid,0]))
    def radius(distance):
        found = [r for lo, hi, r in rows if lo-1e-8 <= distance <= hi+1e-8]
        if not found:
            raise ValueError('The requested material point is outside the finite blade')
        return float(max(found))
    return radius


def fit_point(model, bounds, entry, attack_q, attack_length, seed, *, displacement=None):
    data, adata = mujoco.MjData(model), mujoco.MjData(model)
    ap, aa, aj = tool_point(model, adata, attack_q, attack_length)
    rotation = np.diag([-1.,-1.,1.])
    target = np.array([1.75,0.,0.])+rotation@ap
    incoming = rotation@aa
    desired_normal = unit(np.array([1.,0.,.15])-incoming*float(incoming@np.array([1.,0.,.15])))
    desired_axis = unit(-np.cross(desired_normal, incoming))
    if desired_axis[2] < 0:
        desired_axis *= -1
    caps = np.max(np.abs(model.actuator_forcerange),axis=1)
    radius = radius_lookup(model, data, entry)
    attack_radius = radius(attack_length)
    shift = np.zeros(3) if displacement is None else np.asarray(displacement)
    def state(x):
        point, axis, jac = tool_point(model, data, x[:7], x[7])
        cross = np.cross(incoming, axis)
        normal = cross/max(np.linalg.norm(cross),1e-12)
        if normal@desired_normal < 0:
            normal *= -1
        moment_a = aj.T@(rotation@normal)
        moment_d = jac.T@(-normal)
        separation = attack_radius+radius(x[7])-.0003
        goal = target-separation*normal+shift
        return point, axis, normal, moment_a, moment_d, goal
    def residual(x):
        point, axis, normal, ma, md, goal = state(x)
        data.qvel[:]=0.;mujoco.mj_forward(model,data)
        grip = point-x[7]*axis
        return np.r_[15.*(point-goal), .40*(axis-desired_axis),
            .65*max(.06-ma[5],0.), .5*max(.72-np.linalg.norm(np.cross(incoming,axis)),0.),
            .45*np.maximum(np.abs(data.qfrc_bias)/caps-.96,0.),
            .8*max(.45-grip[2],0.), .018*np.array([3.,1.,1.,1.,1.,1.,1.])*(x[:7]-entry)]
    initial = np.r_[np.maximum(bounds[:,0]+1e-8,np.minimum(bounds[:,1]-1e-8,seed)),.58]
    result = least_squares(residual,initial,bounds=(np.r_[bounds[:,0],.22],np.r_[bounds[:,1],.76]),
                           max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    point, axis, normal, ma, md, goal = state(result.x)
    q = result.x[:7];grip=point-result.x[7]*axis
    data.qvel[:]=0.;mujoco.mj_forward(model,data)
    ratio=float(np.max(np.abs(data.qfrc_bias)/caps));error=float(np.linalg.norm(point-goal))
    return q, {'point_error_m':error,'material_distance_m':float(result.x[7]),
        'attack_material_distance_m':attack_length, 'defender_local_point_m':point.tolist(),
        'desired_point_m':goal.tolist(),'defender_grip_m':grip.tolist(),'defender_axis':axis.tolist(),
        'incoming_axis_defender_local':incoming.tolist(),'predicted_force_on_attacker_defender_local':normal.tolist(),
        'attacker_joint_moment_per_newton':ma.tolist(),'defender_joint_moment_per_newton':md.tolist(),
        'static_cap_ratio':ratio,'nfev':int(result.nfev),
        'fit_pass':bool(error<.012 and ratio<1. and np.linalg.norm(np.cross(incoming,axis))>.65),
        'force_prediction_is_not_measured_contact':True}


def scene_checks(scene, geometry, attack, defence, *, through=None, fixed_attack=False, samples=351):
    """Complete self/idle check, or moving-prefix contact authoring diagnostic."""
    model, data = scene.model, scene.data
    ids = {s:np.asarray(blade_collision_geom_ids(model,s+'_')) for s in ('a1','a2')}
    contacts={};entries={};first=None;max_depth=0.
    duration = defence.duration_s if through is None else through
    for t in np.unique(np.r_[np.linspace(0.,duration,samples),
                            defence.knot_time_s[defence.knot_time_s<=duration]]):
        data.qpos[scene.agents['a1']['qpos_ids']] = attack.knot_qpos[0] if fixed_attack else attack.sample(t)
        data.qpos[scene.agents['a2']['qpos_ids']] = defence.sample(t)
        data.qvel[:]=0.;mujoco.mj_forward(model,data)
        for c in data.contact:
            if c.dist >= -1e-8: continue
            category=classify_contact(model,c,rules_version='whole_blade_first_touch_v2')
            names=[model.geom(int(g)).name for g in (c.geom1,c.geom2)]
            key='|'.join(names)
            if key not in contacts:
                contacts[key]={'geoms':names,'category':category,'first_local_s':float(t),
                               'last_local_s':float(t),'maximum_depth_m':float(-c.dist)}
            row=contacts[key];row['last_local_s']=float(t);row['maximum_depth_m']=max(row['maximum_depth_m'],float(-c.dist))
            max_depth=max(max_depth,float(-c.dist))
            if category=='cross_blade_blade' and first is None:
                normal=c.frame[:3].copy()
                if not names[0].startswith('a1_'):normal*=-1
                first={'local_s':float(t),'world_point_m':c.pos.tolist(),
                       'normal_geom_a1_to_a2_world':normal.tolist(),'depth_m':float(-c.dist)}
        for owner, other in (('a1','a2'),('a2','a1')):
            gi=ids[owner];axes=data.geom_xmat[gi].reshape(-1,3,3)[:,:,2]
            half=model.geom_size[gi,1,None];centers=data.geom_xpos[gi]
            for region in geometry.regions:
                d,_=_point_cylinder_distances(np.array(geometry.world_center(region.name,other)),
                      centers-half*axes,centers+half*axes,model.geom_size[gi,0])
                if np.min(d)<region.radius:
                    entries.setdefault(owner+'_'+region.name,float(t))
    illegal=[c for c in contacts.values() if c['category'].startswith('illegal_')]
    return {'complete_reference_interval_s':duration,'samples':samples,
            'contacts':list(contacts.values()),'illegal_contacts':illegal,
            'first_blade_contact':first,'maximum_reference_penetration_m':max_depth,
            'target_first_entries_local_s':entries,
            'idle_clear':bool(not contacts and not entries),
            'prefix_contact_candidate':bool(not illegal and first is not None and not entries and max_depth<.005),
            'not_a_continuous_or_native_certificate':True}


def build(source, native_scene, output, *, withdraw_refinement=False):
    source,native_scene,output=map(lambda p:Path(p).resolve(),(source,native_scene,output))
    if output.exists():raise FileExistsError('Preserve previous bounded repair results')
    original=json.loads(source.read_text());source_raw=source.read_bytes()
    output.mkdir(parents=True)
    model=build_limited_floor().model
    if blade_identity(model)['sha256']!=original['blade_geometry']['sha256']:
        raise ValueError('Physical blade differs from source')
    scene=exact_scene(native_scene)
    geometry=TriangleGeometry.from_mapping({k:original['scoring'][k]for k in ('base_distance','regions')})
    attack_row=original['actions']['cut_down_center'];attack=C2JointSpline(attack_row['knot_time_s'],attack_row['knot_qpos'])
    entry=np.asarray(original['guards']['legacy_center']['qpos_rad'])
    old=original['actions']['parry_high'];oldq=np.asarray(old['knot_qpos'])
    bounds=_authoring_bounds(model_joint_ranges_rad(load_joint_constraints()),entry)
    report={'status':'bounded_high_parry_repair_candidates','source_bank':str(source),
            'source_bank_sha256':hashlib.sha256(source_raw).hexdigest(),
            'collision_geometry_sha256':original['blade_geometry']['sha256'],
            'native_scene_sha256':hashlib.sha256(native_scene.read_bytes()).hexdigest(),
            'trials':[],'candidate_sources':[],'admitted_to_training':False,
            'scope':'Unchanged retimed vertical attack, exact legacy_center entry and return; no policy reaction or generalized defence certificate.'}
    save(output/'report.json',report)
    successes=0
    # A bounded two-time/two-material set; the old final high pose and the
    # current center are independent seeds, not wrapped joint coordinates.
    specs = ((2.7,.70,0,.25),(3.0,.73,0,.25),(2.7,.70,0,.35),(3.0,.73,0,.35)) if withdraw_refinement else (
             (2.7,.70,0,.09),(3.0,.73,0,.09),(2.7,.70,1,.09),(3.0,.73,1,.09))
    report['withdraw_refinement']=withdraw_refinement
    for phase,length,seed_id,withdraw in specs:
        seed=oldq[-1] if seed_id==0 else entry
        block,fit=fit_point(model,bounds,entry,attack.sample(phase),length,seed)
        trial={'phase_local_s':phase,'attack_material_m':length,'seed':seed_id,
               'block_qpos_rad':block.tolist(),'fit':fit,'candidate_pass':False,'pre_withdraw_m':withdraw}
        report['trials'].append(trial)
        if fit['fit_pass']:
            normal=np.array(fit['predicted_force_on_attacker_defender_local'])
            pre,prefit=fit_point(model,bounds,entry,attack.sample(phase),length,block,
                                displacement=-withdraw*normal+np.array([0.,0.,.08 if withdraw_refinement else .04]))
            trial['pre_fit']=prefit
            if prefit['fit_pass']:
                # Deliberate high recovery: retain blocking placement beyond
                # the dangerous portion; don't retract as the tip enters.
                curve=(C2JointSpline([0.,1.7,phase-.50,phase,7.,10.3],
                                    [entry,pre,pre,block,block,entry]) if withdraw_refinement else
                       C2JointSpline([0.,phase-.75,phase,7.,10.3],[entry,pre,block,block,entry]))
                idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True)
                prefix=scene_checks(scene,geometry,attack,curve,through=phase+.025,samples=251)
                stat=static_screen(model,curve,161);dynamics=screen_curve(model,curve,.02)
                trial.update(knot_qpos=curve.knot_qpos.tolist(),knot_time_s=curve.knot_time_s.tolist(),
                             idle_screen=idle,prefix_screen=prefix,static_screen=stat,
                             ideal_torque_screen=dynamics)
                passed=bool(idle['idle_clear']and prefix['prefix_contact_candidate']
                            and stat['peak_static_cap_ratio']<1. and dynamics['worst_utilization']<=1.)
                trial['candidate_pass']=passed
                if passed and successes<3:
                    folder=output/f'candidate{successes:02d}';folder.mkdir()
                    bank=copy.deepcopy(original)
                    bank['actions']['parry_high']={'kind':'defence','family':'parry_high',
                        'start_guard':'legacy_center','end_guard':'legacy_center','target_region':'upper',
                        'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
                        'candidate_pass':True,'sampled_reference_pass':True,'native_dynamics_tested':False,
                        'admitted_to_training':False,'successful_defence':False,
                        'active_interception_interval_s':[phase-.75,7.],
                        'validation_stage':'bounded_prefix_and_idle_geometry_plus_ideal_torque_only',
                        'offline_contact_authoring_trial':len(report['trials'])-1,
                        'no_25_degree_high_parry_rule_invented':True}
                    bank.update(status='high_parry_repair_unverified_candidate',admitted_to_training=False,
                        parent_bank_sha256=report['source_bank_sha256'],
                        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                    bank['outgoing_actions']={}
                    for name,row in bank['actions'].items():
                        if row.get('candidate_pass')is True:bank['outgoing_actions'].setdefault(row['start_guard'],[]).append(name)
                    save(folder/'report.json',bank)
                    (folder/'scene.xml').write_bytes((source.parent/'scene.xml').read_bytes())
                    ts=np.linspace(0.,curve.duration_s,619)
                    np.savez_compressed(folder/'parry_high.npz',time_s=ts,qpos=curve.sample(ts),
                        knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
                    trial['source']=str(folder/'report.json');report['candidate_sources'].append(trial['source']);successes+=1
        print(json.dumps({'trial':len(report['trials'])-1,'fit':fit['fit_pass'],
                          'point_error':fit['point_error_m'],'static':fit['static_cap_ratio'],
                          'candidate':trial['candidate_pass']}),flush=True)
        save(output/'report.json',report)
        if successes>=3:break
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    report['source_bank_unchanged']=source.read_bytes()==source_raw
    save(output/'report.json',report)
    return report


def refine_clock(source,native_scene,previous,output, *, final_clock=False):
    """Redistribute time around the one safe withdrawn approach; no new IK."""
    source,native_scene,previous,output=map(lambda p:Path(p).resolve(),(source,native_scene,previous,output))
    if output.exists():raise FileExistsError('Preserve earlier clock candidates')
    original=json.loads(source.read_text());prior=json.loads(previous.read_text())
    # This explicit retained failed trial had correct first-contact geometry
    # but a half-second final approach exceeded ideal J0 torque.
    chosen=prior['trials'][3]
    if chosen['pre_withdraw_m']!=.35 or not chosen['prefix_screen']['prefix_contact_candidate']:
        raise ValueError('Expected retained withdrawn safe-prefix trial 3')
    entry,pre,_,block,_,end=np.asarray(chosen['knot_qpos'])
    model=build_limited_floor().model;scene=exact_scene(native_scene)
    if blade_identity(model)['sha256']!=original['blade_geometry']['sha256']:
        raise ValueError('Geometry changed')
    geometry=TriangleGeometry.from_mapping({k:original['scoring'][k]for k in ('base_distance','regions')})
    ar=original['actions']['cut_down_center'];attack=C2JointSpline(ar['knot_time_s'],ar['knot_qpos'])
    output.mkdir(parents=True)
    report={'status':'high_parry_withdrawn_approach_clock_candidates',
        'source_bank_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'retained_failed_trial_source':str(previous),'retained_failed_trial_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),
        'collision_geometry_sha256':original['blade_geometry']['sha256'],'trials':[],
        'candidate_sources':[],'admitted_to_training':False,'native_tested':False}
    clocks=((1.9,3.),(2.,3.),(2.05,3.)) if final_clock else ((1.7,3.),(1.5,3.),(2.,3.2))
    for idx,(pre_time,block_time) in enumerate(clocks):
        curve=C2JointSpline([0.,pre_time,block_time,7.,10.3],[entry,pre,block,block,end])
        idle=scene_checks(scene,geometry,attack,curve,fixed_attack=True)
        prefix=scene_checks(scene,geometry,attack,curve,through=block_time+.025,samples=351)
        stat=static_screen(model,curve,161);dyn=screen_curve(model,curve,.02)
        first=prefix['first_blade_contact']
        favorable=bool(first and first['normal_geom_a1_to_a2_world'][0]>.15)
        passing=bool(idle['idle_clear']and prefix['prefix_contact_candidate']and favorable
                     and stat['peak_static_cap_ratio']<1. and dyn['worst_utilization']<=1.)
        row={'index':idx,'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
             'idle_screen':idle,'prefix_screen':prefix,'static_screen':stat,'ideal_torque_screen':dyn,
             'first_normal_opposes_attacker_forward_motion':favorable,'candidate_pass':passing}
        report['trials'].append(row)
        if passing:
            folder=output/f'candidate{idx:02d}';folder.mkdir();bank=copy.deepcopy(original)
            bank['actions']['parry_high']={'kind':'defence','family':'parry_high','target_region':'upper',
                 'start_guard':'legacy_center','end_guard':'legacy_center',
                 'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
                 'candidate_pass':True,'sampled_reference_pass':True,'successful_defence':False,
                 'native_dynamics_tested':False,'admitted_to_training':False,
                 'offline_contact_authoring_trial':idx,'defence_offset_s':0.,
                 'active_interception_interval_s':[pre_time,7.],
                 'validation_stage':'withdrawn_approach_prefix_idle_and_ideal_torque_only',
                 'no_high_parry_25_degree_rule_added':True}
            bank.update(parent_bank_sha256=report['source_bank_sha256'],admitted_to_training=False,
                        status='unverified_high_parry_clock_repair',
                        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            bank['outgoing_actions']={}
            for name,a in bank['actions'].items():
                if a.get('candidate_pass')is True:bank['outgoing_actions'].setdefault(a['start_guard'],[]).append(name)
            save(folder/'report.json',bank);(folder/'scene.xml').write_bytes((source.parent/'scene.xml').read_bytes())
            ts=np.linspace(0,curve.duration_s,619)
            np.savez_compressed(folder/'parry_high.npz',time_s=ts,qpos=curve.sample(ts),
                                knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
            report['candidate_sources'].append(str(folder/'report.json'))
        print(json.dumps({'trial':idx,'ideal':dyn['worst_utilization'],'prefix':prefix['prefix_contact_candidate'],
                          'max_overlap':prefix['maximum_reference_penetration_m'],'first':first,
                          'candidate':passing}),flush=True)
        save(output/'report.json',report)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--native-scene',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--withdraw-refinement',action='store_true')
    p.add_argument('--clock-refinement')
    p.add_argument('--final-clock',action='store_true')
    a=p.parse_args()
    if a.clock_refinement:refine_clock(a.source,a.native_scene,a.clock_refinement,a.output,final_clock=a.final_clock)
    else:build(a.source,a.native_scene,a.output,withdraw_refinement=a.withdraw_refinement)
