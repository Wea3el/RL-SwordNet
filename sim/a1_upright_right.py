"""Independent upright right parry geometry candidates, never native proof.

Use finite blade material points, unchanged joint/collision/force constraints,
and a declared upward active interval.  All failed fits remain in the report.
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
from .a1_guard_defence_screen import contact_state, tool_point
from .a1_recovery_geometry import exact_scene, screen_path, static_screen, write
from .assembly import ROOT
from .blade_geometry import blade_collision_geom_ids
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import GripAxisSolver, unit
from .triangle_task import TriangleGeometry
from .duel_scene import classify_contact


UP = float(np.sin(np.deg2rad(25.)))


def solve(model, bounds, entry, attack_q, attack_length, seed, normal_sign):
    data, adata = mujoco.MjData(model), mujoco.MjData(model)
    p, axis, jac = tool_point(model,adata,attack_q,attack_length)
    rot = np.diag([-1.,-1.,1.])
    target = np.array([1.75,0.,0.])+rot@p
    incoming = rot@axis
    caps = np.max(abs(model.actuator_forcerange),axis=1)
    gp,ga,_=tool_fk(model,data,entry)
    sections=[]
    blade_ids=blade_collision_geom_ids(model, '')
    for gid in blade_ids:
        center=float((data.geom_xpos[gid]-gp)@ga)
        half=float(model.geom_size[gid,1])
        sections.append((center-half,center+half,float(model.geom_size[gid,0])))
    def radius_at(length):
        return max((radius for low,high,radius in sections if low-1e-8<=length<=high+1e-8),
                   default=sections[-1][2])
    self_pairs=[(gid,model.geom('arm_proxy_'+str(j)).id)
                for gid in blade_ids for j in range(7)]
    def evaluate(x):
        q, length=x[:7],x[7]
        point, shaft,jd=tool_point(model,data,q,length)
        cross=np.cross(incoming,shaft)
        normal=normal_sign*cross/max(np.linalg.norm(cross),1e-12)
        ta=jac.T@(rot@normal);td=jd.T@(-normal)
        grip=point-length*shaft
        self_gaps=np.array([mujoco.mj_geomDistance(model,data,a,b,.06,None)for a,b in self_pairs])
        return point,shaft,jd,normal,ta,td,grip,self_gaps
    def residual(x):
        point,shaft,jd,normal,ta,td,grip,gaps=evaluate(x)
        data.qvel[:]=0.;mujoco.mj_forward(model,data)
        separation=radius_at(attack_length)+radius_at(x[7])-.0004
        return np.r_[12*(point-(target-separation*normal)),
            8*max(UP+.035-shaft[2],0),8*max(.37-grip[2],0),
            1.5*max(.65-np.linalg.norm(np.cross(incoming,shaft)),0),
            .5*max(-ta[5],0),.5*max(-td[5],0),
            10*np.maximum(.01-gaps,0),.3*np.maximum(abs(data.qfrc_bias)/caps-.96,0),
            .012*np.array([3,1,1,1,1,1,1])*(x[:7]-entry)]
    initial=np.r_[np.clip(seed,bounds[:,0]+1e-8,bounds[:,1]-1e-8),.45]
    result=least_squares(residual,initial,bounds=(np.r_[bounds[:,0],.18],np.r_[bounds[:,1],.76]),
        max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    point,shaft,jd,n,ta,td,grip,gaps=evaluate(result.x)
    separation=radius_at(attack_length)+radius_at(result.x[7])-.0004
    error=float(np.linalg.norm(point-(target-separation*n)))
    return result.x[:7],{'point_error_m':error,'active_axis':shaft.tolist(),
        'active_grip_m':grip.tolist(),'active_elevation_deg':float(np.rad2deg(np.arcsin(shaft[2]))),
        'material_length_m':float(result.x[7]),'attack_material_length_m':attack_length,
        'contact_target_defender_local_m':target.tolist(),'normal_force_on_attacker_defender_local':n.tolist(),
        'attacker_joint_moment_per_n':ta.tolist(),'defender_joint_moment_per_n':td.tolist(),
        'minimum_own_blade_arm_gap_m':float(min(gaps)),
        'fit_pass':bool(error<.015 and shaft[2]>=UP and grip[2]>=.35 and min(gaps)>.005),
        'finite_material_points':True,'native_proxy_centerline_separation_m':separation,
        'native_proxy_radii_m':[radius_at(attack_length),radius_at(result.x[7])],
        'actual_collision_screen_required':True,'nfev':int(result.nfev)}


def active_check(model,curve,start,end):
    rows=[tool_fk(model,qpos=q)for q in curve.sample(np.linspace(start,end,251))]
    axes=np.array([v[1]for v in rows]);grips=np.array([v[0]for v in rows])
    return {'active_interval_s':[float(start),float(end)],'min_elevation_deg':25.,
        'minimum_sampled_elevation_deg':float(np.rad2deg(np.arcsin(axes[:,2].min()))),
        'minimum_sampled_grip_z_m':float(grips[:,2].min()),
        'sampled_upright_pass':bool(axes[:,2].min()>=UP and grips[:,2].min()>=.35)}


def paired_prefix(scene,geometry,attack,defence,offset,through):
    contacts=[];first=None;target_entry=False;deep=False;illegal=False
    for t in np.linspace(min(0.,offset),through,int(np.ceil((through-min(0.,offset))/.01))+1):
        state=contact_state(scene,geometry,attack.sample(t),defence.sample(max(0,t-offset)))
        target_entry|=any(v<0 for d in state['target_gaps_m'].values()for v in d.values())
        for c in state['contacts']:
            record=dict(c,attack_local_s=float(t),defence_local_s=float(max(0,t-offset)))
            contacts.append(record)
            if first is None:first=record
            illegal|=c['category'].startswith('illegal_')
            deep|=c['depth_m']>.003
    return {'first_reference_contact':first,'reference_contacts':contacts,
        'illegal_contact':illegal,'penetration_over_3mm':deep,'any_target_entry':target_entry,
        'prefix_pass':bool(first is not None and not illegal and not deep and not target_entry),
        'scope':'Nominal approach until authored interception only. Post-contact response needs native dynamics.'}


def build(output,refine_source=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier right-upright candidates')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_chain_geometry/chain12/report.json';base=json.loads(source.read_bytes())
    scene_path=ROOT/'outputs/a1_recovery_geometry/withdraw01/scene.xml'
    scene=exact_scene(scene_path);model=build_limited_floor().model
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    entry=np.asarray(base['guards']['right_low_negative_robust']['qpos_rad'])
    bounds=model_joint_ranges_rad(load_joint_constraints())+np.deg2rad([3.,-3.])
    bounds[0]=[max(bounds[0,0],entry[0]-np.deg2rad(35)),min(bounds[0,1],entry[0]+np.deg2rad(35))]
    bounds[2,1]=np.deg2rad(350);bounds[3,1]=np.deg2rad(115)
    bounds[5,0]=np.deg2rad(23);bounds[6]=np.deg2rad([-82,82])
    old=json.loads((ROOT/'outputs/a1_guard_defence_screen/mirrored_return04/report.json').read_bytes())
    oldhold=np.asarray(old['actions']['parry_right_from_negative_low']['knot_qpos'][1])
    seed_up=entry.copy();seed_up[1:]=np.deg2rad([55,290,80,205,65,5])
    attackrow=base['actions']['cut_up_left_to_right'];attack=C2JointSpline(attackrow['knot_time_s'],attackrow['knot_qpos'])
    report={'schema_version':1,'status':'upright_right_contact_search_incomplete','guards':copy.deepcopy(base['guards']),
        'scoring':base['scoring'],'actions':{'cut_up_left_to_right':copy.deepcopy(attackrow)},'trials':[],
        'source_report':str(source),'source_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_dynamics_tested':False,'admitted_to_training':False,
        'upright_requirement':{'min_elevation_deg':25.,'minimum_grip_height_m':.35,
           'scope':'Active interception/hold and every actual loaded weapon contact; not initial staging or post-disengagement recovery.'},
        'limitations':['No former downward-parry certificate transfers.','Finite material point fit is not a physical block.',
                       'Full-window recovery and both slots require fresh native validation.']}
    write(output/'report.json',report)
    solver=GripAxisSolver(model,bounds)
    next_id=0
    if refine_source:
        previous=json.loads(Path(refine_source).read_bytes())
        specs=[(r['attack_local_contact_s'],r['fit']['attack_material_length_m'],
                r['normal_sign'],r['id'],np.asarray(r['block_qpos_rad']))
               for r in previous['trials']if r['id']in(12,15,18)]
        report['refinement_source']=str(Path(refine_source).resolve())
        report['refinement_source_sha256']=hashlib.sha256(Path(refine_source).read_bytes()).hexdigest()
    else:
        specs=[(phase,length,sign,si,seed)
               for phase in (2.15,2.30,1.95)for length in (.72,.60)
               for sign in (1.,-1.)for si,seed in enumerate((seed_up,oldhold,entry))]
    for phase,length,sign,si,seed in specs:
            block,fit=solve(model,bounds,entry,attack.sample(phase),length,seed,sign)
            row={'id':next_id,'attack_local_contact_s':phase,'seed':si,'normal_sign':sign,
                 'block_qpos_rad':block.tolist(),'fit':fit,'candidate_pass':False}
            next_id+=1
            report['trials'].append(row)
            stat=static_screen(model,C2JointSpline([0,1],[block,block]),21);row['block_static']=stat
            if fit['fit_pass'] and stat['peak_static_cap_ratio']<1:
                p,axis,_=tool_fk(model,qpos=block)
                # Upright pre-position, translated away from incoming blade.
                n=np.array(fit['normal_force_on_attacker_defender_local'])
                pre,pf=solver.solve(p+.055*n,axis,block,regularization=.005,iterations=140)
                row['pre_fit']=pf
                approach=max(.8,1.9*abs(pre[0]-entry[0])/.60,
                             1.9*max(abs(pre[1:]-entry[1:]))/1.35)
                duration=approach+.32;offset=phase-duration
                row['defence_offset_s']=offset
                if pf['fit_pass']and offset>=(-.30 if refine_source else .05):
                    hold_end=duration+max(1.5,4.8-phase)
                    return_time=max(1.7,1.9*abs(block[0]-entry[0])/.45,
                                    1.9*max(abs(block[1:]-entry[1:]))/1.0)
                    curve=C2JointSpline([0,approach,duration,hold_end,hold_end+return_time],
                                        [entry,pre,block,block,entry])
                    active=active_check(model,curve,approach,hold_end);row['upright']=active
                    idle=screen_path(scene,geometry,curve,attack.knot_qpos[0],samples=251,both_slots=True)
                    row['idle_screen']=idle
                    prefix=paired_prefix(scene,geometry,attack,curve,offset,phase+.015);row['prefix']=prefix
                    static=static_screen(model,curve,151);row['path_static']=static
                    dyn=screen_curve(model,curve,.025);row['ideal_dynamics']=dyn
                    passed=bool(active['sampled_upright_pass']and idle['all_clear']and prefix['prefix_pass']
                                and static['peak_static_cap_ratio']<1 and dyn['worst_utilization']<=1.08)
                    row['candidate_pass']=passed
                    if passed:
                        name='parry_right_upright_'+str(row['id'])
                        report['actions'][name]={'kind':'defence','family':'parry_right',
                            'start_guard':'right_low_negative_robust','end_guard':'right_low_negative_robust',
                            'target_region':'right','knot_time_s':curve.knot_time_s.tolist(),
                            'knot_qpos':curve.knot_qpos.tolist(),'candidate_pass':True,'sampled_reference_pass':True,
                            'native_dynamics_tested':False,'admitted_to_training':False,
                            'active_interval_s':active['active_interval_s'],'min_elevation_deg':25.,
                            'minimum_grip_height_m':.35,'defence_offset_s':offset,
                            'preemptive_start_before_attack':bool(offset<0),
                            'reactive_observation_demonstrated':False,
                            'authoring_trial':row['id'],'scope':'Upright nominal geometry only; no physical parry claim.'}
                        ts=np.linspace(0,curve.duration_s,401)
                        np.savez_compressed(output/(name+'.npz'),time_s=ts,qpos=curve.sample(ts),
                                            knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
            write(output/'report.json',report)
            print(json.dumps({'id':row['id'],'phase':phase,'fit':fit['fit_pass'],
                'elevation':fit['active_elevation_deg'],'static':stat['peak_static_cap_ratio'],
                'candidate':row['candidate_pass'],'offset':row.get('defence_offset_s')}),flush=True)
            if len(report['actions'])>=3:break
    report['status']='upright_right_contact_search_complete'
    report['candidate_count']=len(report['actions'])-1
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'scene.xml').write_bytes(scene_path.read_bytes())
    write(output/'report.json',report)
    return report


def direct_candidate(source,output,early_upright=False):
    """Bounded direct-route repair of the retained upright fit, no new IK grid."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve preceding direct candidates')
    original=json.loads(source.read_bytes())
    output.mkdir(parents=True)
    # Retained fit #2 is the low-J0 branch from the exact existing low guard.
    trial=original['trials'][2]
    entry=np.asarray(original['guards']['right_low_negative_robust']['qpos_rad'])
    block=np.asarray(trial['block_qpos_rad'])
    attackrow=original['actions']['cut_up_left_to_right']
    attack=C2JointSpline(attackrow['knot_time_s'],attackrow['knot_qpos'])
    model=build_limited_floor().model
    prefix_description=None
    if early_upright:
        data=mujoco.MjData(model)
        grip0,_,_=tool_fk(model,qpos=entry)
        bounds=model_joint_ranges_rad(load_joint_constraints())+np.deg2rad([3.,-3.])
        bounds[0]=[entry[0]-np.deg2rad(35),min(entry[0]+np.deg2rad(35),bounds[0,1])]
        bounds[2,1]=np.deg2rad(350);bounds[3,1]=np.deg2rad(115)
        bounds[5,0]=np.deg2rad(23);bounds[6]=np.deg2rad([-82,82])
        caps=np.max(abs(model.actuator_forcerange),axis=1)
        def residual(q):
            grip,axis,tip=tool_fk(model,data,q)
            data.qvel[:]=0.;mujoco.mj_forward(model,data)
            return np.r_[8*max(np.sin(np.deg2rad(40))-axis[2],0),
                3*max(tip[0]-1.03,0),3*max(.4-grip[2],0),.7*(grip-grip0),
                .025*np.array([3,1,1,1,1,1,1])*(q-entry),
                .25*np.maximum(abs(data.qfrc_bias)/caps-.96,0)]
        fit=least_squares(residual,np.clip(entry,bounds[:,0]+1e-7,bounds[:,1]-1e-7),
                          bounds=(bounds[:,0],bounds[:,1]),max_nfev=220)
        pre=fit.x
        curve=C2JointSpline([0.,.85,2.3,4.8,7.3],[entry,pre,block,block,entry])
        active_start=.85
        pp,aa,tt=tool_fk(model,qpos=pre)
        prefix_description={'intent':'Lift and withdraw before the original early-contact corridor, then retain the same block and exit.',
            'qpos_rad':pre.tolist(),'grip_m':pp.tolist(),'axis':aa.tolist(),'tip_m':tt.tolist(),
            'first_waypoint_time_s':.85,'target_prefix_elevation_deg':40.,
            'actual_prefix_elevation_deg':float(np.rad2deg(np.arcsin(aa[2])))}
    else:
        curve=C2JointSpline([0.,2.3,4.8,7.3],[entry,block,block,entry])
        active_start=1.2
    scene_path=ROOT/'outputs/a1_recovery_geometry/withdraw01/scene.xml'
    scene=exact_scene(scene_path)
    geometry=TriangleGeometry.from_mapping({k:original['scoring'][k]for k in ('base_distance','regions')})
    active=active_check(model,curve,active_start,4.8)
    idle={name:screen_path(scene,geometry,curve,q,samples=401,both_slots=True)
          for name,q in [('attacker_start',attack.knot_qpos[0]),('attacker_end',attack.knot_qpos[-1])]}
    first=None; before=[]; after=[]
    for t in np.arange(0.,curve.duration_s+.0001,.002):
        state=contact_state(scene,geometry,attack.sample(t),curve.sample(t))
        for c in state['contacts']:
            if first is None:
                first=dict(c,attack_local_s=float(t),defence_local_s=float(t))
                # Physical contact frame: report force leverage, never assume
                # a nominal crossing produces the desired dynamic deflection.
                for contact in scene.data.contact:
                    if contact.dist>=0:continue
                    if classify_contact(scene.model,contact,rules_version='whole_blade_first_touch_v2')!=c['category']:continue
                    normal=contact.frame[:3].copy()
                    a_first=scene.model.geom(int(contact.geom1)).name.startswith('a1_')
                    if not a_first:normal*=-1.
                    first['normal_attacker_to_defender_world']=normal.tolist()
                    moments={}
                    for slot,gid,force in [('a1',contact.geom1 if a_first else contact.geom2,-normal),
                                           ('a2',contact.geom2 if a_first else contact.geom1,normal)]:
                        jac=np.zeros((3,scene.model.nv))
                        mujoco.mj_jac(scene.model,scene.data,jac,None,contact.pos,int(scene.model.geom_bodyid[gid]))
                        ids=scene.agents[slot]['qpos_ids']
                        moments[slot]=(jac[:,ids].T@force).tolist()
                    first['joint_normal_moment_per_n']=moments
                    break
            if first is not None and t>first['attack_local_s']+.0021:
                after.append(dict(c,time_s=float(t)))
            else:before.append(dict(c,time_s=float(t)))
    def summarize(items):
        result={}
        for c in items:
            key='|'.join(c['geoms'])
            row=result.setdefault(key,dict(c,first_time_s=c['time_s'],last_time_s=c['time_s']))
            row['last_time_s']=c['time_s'];row['depth_m']=max(row['depth_m'],c['depth_m'])
        return list(result.values())
    dyn=screen_curve(model,curve,.01);stat=static_screen(model,curve,301)
    prefix_ok=bool(first and first['category']=='cross_blade_blade'and first['depth_m']<.003
                   and not any(c['category'].startswith('illegal_')for c in before))
    passed=bool(prefix_ok and active['sampled_upright_pass']and all(v['all_clear']for v in idle.values())
                and dyn['ideal_demand_within_caps_at_samples']and stat['peak_static_cap_ratio']<1.)
    name='parry_right_upright_early'if early_upright else 'parry_right_upright_direct'
    action={'kind':'defence','family':'parry_right','start_guard':'right_low_negative_robust',
        'end_guard':'right_low_negative_robust','target_region':'right',
        'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
        'candidate_pass':passed,'sampled_reference_pass':passed,
        'native_dynamics_tested':False,'admitted_to_training':False,
        'defence_offset_s':0.,'preemptive_start_before_attack':False,
        'reactive_observation_demonstrated':False,
        'active_interval_s':[active_start,4.8],'min_elevation_deg':25.,'minimum_grip_height_m':.35,
        'upright_screen':active,'static_screen':stat,'ideal_free_space_dynamics':dyn,
        'idle_screen':idle,'first_reference_contact':first,
        'pre_contact_screen_pass':prefix_ok,'early_contact_summary':summarize(before),
        'post_contact_reference_summary':summarize(after),
        'post_contact_reference_is_not_physical_prediction':True,
        'scope':'Native-test candidate only. No successful block, recovery or loading certificate; unchanged incoming reference after first contact may geometrically overlap.'}
    if prefix_description:action['early_upright_prefix']=prefix_description
    report={'schema_version':1,'status':'upright_right_direct_native_test_candidate',
        'scoring':original['scoring'],'guards':original['guards'],
        'actions':{'cut_up_left_to_right':attackrow,name:action},
        'source_report':str(source),'source_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_dynamics_tested':False,'admitted_to_training':False,
        'active_interval_clock':'Seconds since defence begins; attack and defence begin together for offset zero.',
        'source_fit_id':trial['id'],'source_fit':trial['fit'],
        'limitations':['No inherited old downward-parry evidence.',
          'Deep post-contact unresponsive reference overlap is retained as a risk, never counted as blocking success.',
          'Actual loaded contacts, upright elevation, full-window no-score and recovery require new native tests.',
          'Offset zero is a scripted feasibility test, not demonstrated policy reaction.']}
    if early_upright:
        audit_path=ROOT/'outputs/a1_upright_right/native04_upright_audit/audit.json'
        report['retained_failure_evidence']={'path':str(audit_path),
            'sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            'reason':'Prior direct route physically blocked but loaded contact began at only 19.302 degrees.',
            'clock_check':'Native first loaded contact global1.698 minus start.75 = local.948; source reference first contact local1.686.',
            'actual_counterfactual':'At global1.698 the actual defender with reference attacker overlaps15.12mm; both references clear. Mainly defender tracking deviation, not a clock bug.'}
    write(output/'report.json',report)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'scene.xml').write_bytes(scene_path.read_bytes())
    ts=np.linspace(0,curve.duration_s,701)
    np.savez_compressed(output/(name+'.npz'),time_s=ts,qpos=curve.sample(ts),
                        knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    print(json.dumps({'action':name,'candidate_pass':passed,'first_contact':first,
                     'upright':active,'dynamic_peak':dyn['worst_utilization']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--refine-source')
    p.add_argument('--direct-source')
    p.add_argument('--early-upright-source')
    args=p.parse_args()
    if args.early_upright_source:direct_candidate(args.early_upright_source,args.output,True)
    elif args.direct_source:direct_candidate(args.direct_source,args.output)
    else:build(args.output,args.refine_source)
