"""Target-directed static follow-up to the preserved random joint audit.

This is a bounded geometric optimizer, NOT an A1 motion, controller, dynamic
simulation or training rollout. It starts from collision-free random samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from sim.duel_scene import classify_contact
from sim.fencing_rules import BladeSection, point_cylinder_distance

ROOT=Path(__file__).resolve().parents[1]
SLOTS=('a1','a2')


class ProbeBudget(Exception): pass
class WitnessFound(Exception): pass


def run(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve prior evidence; choose new output')
    output.mkdir(parents=True)
    started=time.monotonic()
    source_files=[source/'scene.xml',source/'report.json',source/'samples.npz',Path(__file__).resolve()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    report0=json.loads((source/'report.json').read_text())
    with np.load(source/'samples.npz',allow_pickle=False) as archive:
        samples=archive['q_rad'].copy()
        clean=archive['illegal_overlap_count'][0]==0
        gaps=archive['compact_torso_probe_zone_clearance_m'][0].copy()
    model=mujoco.MjModel.from_xml_path(str(source/'scene.xml'))
    data=mujoco.MjData(model)
    ready=np.asarray(report0['shared_ready_q_rad'])
    limits=np.asarray(report0['joint_ranges_rad'])
    margin=float(report0['minimum_requested_boundary_margin_rad'])
    bounds=list(zip(limits[:,0]+margin,limits[:,1]-margin))
    qids={slot:np.array([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{slot}_J{i}')] for i in range(7)]) for slot in SLOTS}
    dofs={slot:np.array([model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{slot}_J{i}')] for i in range(7)]) for slot in SLOTS}
    geoms={slot:[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'{slot}_shinai_blade_collision_{i}') for i in range(3)] for slot in SLOTS}
    calls=0; jac_calls=0; records=[]; trials=[]
    total_budget=400; per_zone_budget=120
    def evaluate(q,owner,center,radius,zone):
        nonlocal calls,jac_calls
        if calls>=total_budget: raise ProbeBudget('global_fk_budget')
        q=np.asarray(q,float)
        if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<limits[:,0]+margin-1e-12) or np.any(q>limits[:,1]-margin+1e-12):
            raise ValueError('Optimizer proposed an out-of-contract joint vector')
        for slot in SLOTS: data.qpos[qids[slot]]=q if slot==owner else ready
        data.qvel[:]=0.; data.time=0.
        mujoco.mj_forward(model,data);calls+=1
        choices=[]
        for gid in geoms[owner]:
            direction=data.geom_xmat[gid].reshape(3,3)[:,2]
            half=model.geom_size[gid,1];pos=data.geom_xpos[gid]
            section=BladeSection(pos-half*direction,pos+half*direction,model.geom_size[gid,0])
            distance,point=point_cylinder_distance(center,section)
            choices.append((distance,point,gid))
        distance,point,gid=min(choices,key=lambda row:row[0])
        jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,point,int(model.geom_bodyid[gid]));jac_calls+=1
        objective=distance**2
        gradient=2*(point-center)@jp[:,dofs[owner]]
        illegal=[];legal=[]
        for c in data.contact:
            if c.dist>=-1e-9: continue
            category=classify_contact(model,c,rules_version='whole_blade_first_touch_v2')
            item={'category':category,'depth_m':float(-c.dist),
                  'geoms':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]}
            if not category.startswith('illegal_'):
                legal.append(item);continue
            illegal.append(item)
            # Static search penalty only; does not modify physics or scoring.
            jp1,jp2=np.zeros_like(jp),np.zeros_like(jp)
            mujoco.mj_jac(model,data,jp1,None,c.pos,int(model.geom_bodyid[c.geom1]))
            mujoco.mj_jac(model,data,jp2,None,c.pos,int(model.geom_bodyid[c.geom2]));jac_calls+=2
            dgradient=c.frame[:3]@(jp2-jp1)[:,dofs[owner]]
            penalty=.002-float(c.dist)
            objective+=100.*penalty**2
            gradient-=200.*penalty*dgradient
        record={'evaluation':calls,'zone':zone,'owner':owner,'q_rad':q.tolist(),
            'target_center_m':center.tolist(),'target_radius_m':radius,
            'target_signed_gap_m':float(distance-radius),
            'minimum_joint_boundary_margin_rad':float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q))),
            'native_contact_count':int(data.ncon),'illegal_proxy_overlaps':illegal,'legal_weapon_overlaps':legal,
            'closest_blade_geom':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid),
            'closest_blade_surface_point_m':point.tolist(),
            'inside_without_illegal_overlap':bool(distance-radius<-.005 and not illegal)}
        records.append(record)
        return objective,np.asarray(gradient,float),record
    target_zones=report0['roles']['a1']['candidates']['compact_torso_probe']['zone_results']
    for zone,info in enumerate(target_zones):
        center=np.asarray(info['center_m']);radius=float(info['radius_m'])
        indices=np.flatnonzero(clean);start_index=int(indices[np.argmin(gaps[indices,zone])])
        initial=samples[start_index]
        phase_start=calls;cache={};found=None;status='not_started'
        def objective(q):
            nonlocal found
            if calls-phase_start>=per_zone_budget: raise ProbeBudget('zone_fk_budget')
            key=tuple(q)
            if key not in cache:
                cache[key]=evaluate(q,'a1',center,radius,zone)
            value,grad,record=cache[key]
            if record['inside_without_illegal_overlap']:
                found=record;raise WitnessFound()
            return value,grad
        try:
            result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                            options={'maxiter':100,'maxfun':per_zone_budget,'ftol':1e-14,'gtol':1e-10,'maxls':20})
            status=str(result.message)
        except WitnessFound: status='static_clean_witness_found'
        except ProbeBudget as exc: status=str(exc)
        mirror=None
        if found is not None:
            mirror_center=center.copy();mirror_center[0]=.32
            _,_,mirror=evaluate(found['q_rad'],'a2',mirror_center,radius,zone)
        trials.append({'zone':zone,'center_m':center.tolist(),'radius_m':radius,
            'start_random_sample_index':start_index,'initial_random_gap_m':float(gaps[start_index,zone]),
            'optimizer_status':status,'forward_calls_including_mirror':calls-phase_start,
            'a1_static_witness':found,'a2_mirrored_static_check':mirror})
        print(f'zone {zone}: {status}; total FK={calls}',flush=True)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report={'schema_version':1,'status':'completed','source_random_audit':str(source),
        'source_sha256':hashes,'source_unchanged':unchanged,
        'method':'bounded target-directed static L-BFGS-B, nearest finite-blade surface Jacobian and proxy overlap penalty',
        'total_forward_calls':calls,'maximum_forward_calls':400,'jacobian_calls':jac_calls,
        'preceding_random_forward_calls':report0['total_mj_forward_sample_evaluations'],
        'combined_forward_calls':calls+report0['total_mj_forward_sample_evaluations'],
        'trials':trials,'all_evaluations':records,'admitted_to_training':False,
        'limitations':['No A1 node other than the shared initial ready, no A1 trajectory or action label is used.',
            'Targets are arbitrary compact sensitivity regions, not confirmed anatomy or scoring rules.',
            'A witness proves only a joint configuration with >5mm geometric entry and no disallowed proxy overlap under current masks.',
            'No path from ready, velocities, controller, contact dynamics or stable hold is established.',
            'Mirroring checks the same joint state, not a second policy or independently trained A2.',
            'The static search penalty does not modify the production model, forces, scores or collision masks.',
            'Optimizer failure or missing witnesses never proves global impossibility.'],
        'elapsed_wall_s':time.monotonic()-started}
    if not unchanged: raise RuntimeError('Input changed during refinement')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'evaluations.npz',q_rad=np.array([r['q_rad'] for r in records]),
        target_signed_gap_m=np.array([r['target_signed_gap_m'] for r in records]),
        illegal_overlap_count=np.array([len(r['illegal_proxy_overlaps']) for r in records]))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/scoring_design_v1/reachability')
    p.add_argument('--output',default='outputs/scoring_design_v1/reachability_refinement')
    args=p.parse_args();run(args.source,args.output)
