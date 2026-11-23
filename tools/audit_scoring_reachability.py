"""Bounded static joint-space probes for candidate scoring geometry.

This is independent of A1 action curves, not a motion planner or a trained A2.
Joint samples are assigned only in an offline MjData and evaluated by
mj_forward. No integration, state teleport in an episode, score or reward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from sim.duel_scene import classify_contact
from sim.fencing_rules import blade_sections_from_scene, point_cylinder_distance
from sim.joint_constraints import load_joint_constraints, model_joint_ranges_rad
from sim.limited_duel import build_limited_duel
from tools.audit_target_candidates import candidate_geometry

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ('a1', 'a2')
SEED = 20261123
SAMPLES_PER_ROLE = 768
BOUNDARY_MARGIN_RAD = .05


def sample_joint_space(ranges, ready, *, count=768, seed=SEED, margin=.05):
    """Global uniform plus bounded local rejection; never clip or wrap."""
    ranges, ready = np.asarray(ranges, float), np.asarray(ready, float)
    if ranges.shape != (7, 2) or ready.shape != (7,) \
            or not np.isfinite(ranges).all() or not np.isfinite(ready).all() \
            or not 3 <= count <= 2000 or not np.isfinite(margin) or margin <= 0:
        raise ValueError('Bounded count and finite 7-joint sampling contract required')
    lo, hi = ranges[:, 0]+margin, ranges[:, 1]-margin
    if np.any(lo >= hi) or np.any(ready < lo) or np.any(ready > hi):
        raise ValueError('Margin must preserve ranges and include the explicit ready')
    rng = np.random.default_rng(seed)
    global_count = count//2
    global_samples = rng.uniform(lo, hi, size=(global_count, 7))
    remaining, local, proposals = count-global_count-1, [], 0
    while sum(len(x) for x in local) < remaining:
        proposal = ready+rng.normal(size=(256, 7))*(hi-lo)*.22
        proposals += len(proposal)
        if proposals > 100000:
            raise ValueError('Local rejection sampling exceeded its finite proposal budget')
        local.append(proposal[np.all((proposal >= lo) & (proposal <= hi), axis=1)])
    local_samples = np.concatenate(local, axis=0)[:remaining]
    samples = np.vstack((ready, global_samples, local_samples))
    labels = np.array(['shared_ready']+['global_uniform']*global_count+['local_rejection']*remaining)
    return samples, labels, proposals


def primitive_signed_distance(point, kind, size):
    """Exact local signed distance for current sphere/box/capsule/cylinder proxies."""
    point, size = np.asarray(point, float), np.asarray(size, float)
    if point.shape != (3,) or size.shape != (3,) or not np.isfinite([point,size]).all():
        raise ValueError('Finite 3D primitive point and size required')
    if kind == 'sphere': return float(np.linalg.norm(point)-size[0])
    if kind == 'box':
        q = np.abs(point)-size
        return float(np.linalg.norm(np.maximum(q,0.))+min(float(np.max(q)),0.))
    if kind == 'capsule':
        q = point.copy();q[2] -= np.clip(q[2],-size[1],size[1])
        return float(np.linalg.norm(q)-size[0])
    if kind == 'cylinder':
        q = np.array([np.linalg.norm(point[:2])-size[0],abs(point[2])-size[1]])
        return float(np.linalg.norm(np.maximum(q,0.))+min(float(np.max(q)),0.))
    raise ValueError('Unsupported exact primitive')


def segment_bound_intersects(start, end, center, radius):
    """Conservative bounding-sphere test, NOT proof of line occlusion."""
    start,end,center = (np.asarray(x,float) for x in (start,end,center))
    delta = end-start
    t = np.clip(float((center-start)@delta)/float(delta@delta),0.,1.) if delta@delta > 0 else 0.
    return bool(np.linalg.norm(center-(start+t*delta)) <= radius)


def _hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_target_proxy_audit(scene, geometry):
    kinds = {int(mujoco.mjtGeom.mjGEOM_SPHERE):'sphere', int(mujoco.mjtGeom.mjGEOM_BOX):'box',
             int(mujoco.mjtGeom.mjGEOM_CAPSULE):'capsule', int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder'}
    model,data = scene.model,scene.data
    result = {}
    for name,(centers,radii) in geometry.items():
        result[name] = {}
        for attacker_index,attacker in enumerate(SLOTS):
            defender = SLOTS[1-attacker_index]
            origin = data.site_xpos[scene.agents[attacker]['sites']['shinai_tip']].copy()
            regions = []
            for center,radius in zip(centers[attacker_index],radii):
                records,possible_blockers = [],[]
                for gid in range(model.ngeom):
                    geom_name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) or ''
                    if not geom_name.startswith(defender+'_') \
                            or not (model.geom_contype[gid] or model.geom_conaffinity[gid]): continue
                    rotation = data.geom_xmat[gid].reshape(3,3)
                    local = rotation.T@(center-data.geom_xpos[gid])
                    kind = kinds.get(int(model.geom_type[gid]))
                    exact = kind is not None
                    distance = primitive_signed_distance(local,kind,model.geom_size[gid]) if exact else \
                        float(np.linalg.norm(local)-model.geom_rbound[gid])
                    record = {'geom':geom_name,'kind':kind or 'bounding_sphere_only',
                        'sphere_to_proxy_signed_gap_m':float(distance-radius),
                        'target_center_inside_proxy':bool(distance < 0),
                        'intersection_method':'exact_primitive_signed_distance' if exact else 'conservative_bound'}
                    records.append(record)
                    if segment_bound_intersects(origin,center,data.geom_xpos[gid],model.geom_rbound[gid]):
                        possible_blockers.append(geom_name)
                ordered = sorted(records,key=lambda x:x['sphere_to_proxy_signed_gap_m'])
                regions.append({'center_m':center.tolist(),'radius_m':float(radius),
                    'intersecting_proxies':[x for x in ordered if x['sphere_to_proxy_signed_gap_m'] < 0],
                    'nearest_three_proxies':ordered[:3],
                    'ready_tip_to_target_center_possible_bound_blockers':possible_blockers,
                    'occlusion_test':'conservative bounding spheres on one ready-tip-to-center line; not exact occlusion, not all approach paths'})
            result[name][defender] = regions
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/scoring_design_v1/reachability')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Use a fresh output; preserve prior evidence')
    output.mkdir(parents=True)
    began=time.monotonic()
    files = ['tools/audit_scoring_reachability.py','tools/audit_target_candidates.py',
        'config/a1_direction_candidates_v1.json','config/joint_constraints_observed.json',
        'sim/limited_duel.py','sim/duel_scene.py','sim/fencing_rules.py']
    hashes={p:_hash(ROOT/p) for p in files}
    ready=np.asarray(json.loads((ROOT/'config/a1_direction_candidates_v1.json').read_text())['nodes']['ready'])
    ranges=model_joint_ranges_rad(load_joint_constraints())
    samples,labels,proposals=sample_joint_space(ranges,ready,count=SAMPLES_PER_ROLE,margin=BOUNDARY_MARGIN_RAD)
    scene=build_limited_duel(1.75,initial_qpos=ready)
    model,data=scene.model,scene.data
    (output/'scene.xml').write_text(scene.spec.to_xml())
    original_centers=[scene.config['virtual_targets'][s]['center_world_m'] for s in reversed(SLOTS)]
    geometry={k:v for k,v in candidate_geometry(original_centers).items() if k!='compact_plus_head_probe'}
    ready_audit=initial_target_proxy_audit(scene,geometry)
    all_clearance={name:np.empty((2,len(samples),len(radii))) for name,(_,radii) in geometry.items()}
    illegal=np.zeros((2,len(samples)),dtype=int)
    max_overlap=np.zeros_like(illegal,dtype=float)
    legal_overlap=np.zeros_like(illegal)
    first_illegal_examples={s:[] for s in SLOTS}
    for owner_index,owner in enumerate(SLOTS):
        other=SLOTS[1-owner_index]
        for sample_index,q in enumerate(samples):
            data.qpos[scene.agents[owner]['qpos_ids']]=q
            data.qpos[scene.agents[other]['qpos_ids']]=ready
            data.qvel[:]=0.;data.time=0.
            mujoco.mj_forward(model,data)
            contacts=[]
            for c in data.contact:
                if c.dist >= -1e-9: continue
                category=classify_contact(model,c,rules_version='whole_blade_first_touch_v2')
                record={'category':category,'depth_m':float(-c.dist),
                    'geoms':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]}
                if category.startswith('illegal_'): contacts.append(record)
                else: legal_overlap[owner_index,sample_index]+=1
            illegal[owner_index,sample_index]=len(contacts)
            max_overlap[owner_index,sample_index]=max([0.]+[c['depth_m'] for c in contacts])
            if contacts and len(first_illegal_examples[owner])<5:
                first_illegal_examples[owner].append({'sample_index':sample_index,'contacts':contacts})
            blade=blade_sections_from_scene(scene)[owner]
            for name,(centers,radii) in geometry.items():
                for zone,(center,radius) in enumerate(zip(centers[owner_index],radii)):
                    all_clearance[name][owner_index,sample_index,zone]=min(
                        point_cylinder_distance(center,section)[0] for section in blade)-radius
        print(f'{owner}: {len(samples)} static joint-space samples completed',flush=True)
    report={'schema_version':1,'status':'completed','seed':SEED,
        'samples_per_role':len(samples),'unique_joint_vectors':len(samples),'total_mj_forward_sample_evaluations':2*len(samples),
        'sampling_labels':{name:int(np.count_nonzero(labels==name)) for name in np.unique(labels)},
        'minimum_requested_boundary_margin_rad':BOUNDARY_MARGIN_RAD,
        'actual_minimum_sample_boundary_margin_rad':float(np.min(np.minimum(samples-ranges[:,0],ranges[:,1]-samples))),
        'local_rejection_proposals':proposals,'joint_ranges_rad':ranges.tolist(),'shared_ready_q_rad':ready.tolist(),
        'source_sha256':hashes,'model_source_sha256':scene.source_hashes,
        'scene_xml_sha256':_hash(output/'scene.xml'),
        'initial_target_vs_defender_proxies':ready_audit,'roles':{},
        'limitations':['Only static geometry under existing collision masks; excluded adjacent proxies are not checked.',
            'No mj_step, forces, speed, timing, path from ready, recovery or dynamic hold tested.',
            'No A1 curves used for the samples: only a shared initial q vector and confirmed joint ranges.',
            'Uniform/global and ready-local samples are neither exhaustive nor an estimate of trained A2 behavior.',
            'A clean static witness is not dynamically reachable certification; no witness is not global impossibility.',
            'Target candidates are arbitrary sensitivity probes, not anatomical recommendations or a finalized rule.',
            'Target/proxy intersection is tested exactly for current primitives; line-occlusion screening uses conservative bounds only.'],
        'admitted_to_training':False}
    for owner_index,owner in enumerate(SLOTS):
        clean=illegal[owner_index]==0
        role={'no_disallowed_overlap_samples':int(np.count_nonzero(clean)),
            'disallowed_overlap_samples':int(np.count_nonzero(~clean)),
            'max_disallowed_overlap_depth_m':float(np.max(max_overlap[owner_index])),
            'legal_weapon_overlap_samples':int(np.count_nonzero(legal_overlap[owner_index])),
            'out_of_joint_range_samples':int(np.count_nonzero(np.any((samples<ranges[:,0])|(samples>ranges[:,1]),axis=1))),
            'first_illegal_examples':first_illegal_examples[owner],'candidates':{}}
        for name,(centers,radii) in geometry.items():
            zones=[]
            for zone,(center,radius) in enumerate(zip(centers[owner_index],radii)):
                clear=all_clearance[name][owner_index,:,zone]
                inside=clear < -1e-10
                witness_indices=np.flatnonzero(inside&clean)
                witness=None
                if len(witness_indices):
                    index=int(witness_indices[np.argmin(clear[witness_indices])])
                    witness={'sample_index':index,'sampling_source':str(labels[index]),
                        'q_rad':samples[index].tolist(),'signed_gap_m':float(clear[index]),
                        'minimum_joint_boundary_margin_rad':float(np.min(np.minimum(samples[index]-ranges[:,0],ranges[:,1]-samples[index]))),
                        'disallowed_proxy_overlaps':int(illegal[owner_index,index])}
                zones.append({'center_m':center.tolist(),'radius_m':float(radius),
                    'inside_samples_including_collisions':int(np.count_nonzero(inside)),
                    'inside_samples_without_disallowed_overlap':int(np.count_nonzero(inside&clean)),
                    'minimum_signed_gap_all_samples_m':float(np.min(clear)),
                    'minimum_signed_gap_clean_samples_m':float(np.min(clear[clean])) if clean.any() else None,
                    'clean_static_witness':witness})
            union=np.any(all_clearance[name][owner_index] < -1e-10,axis=1)
            role['candidates'][name]={'zone_results':zones,
                'union_inside_samples':int(np.count_nonzero(union)),
                'union_inside_samples_without_disallowed_overlap':int(np.count_nonzero(union&clean))}
        report['roles'][owner]=role
    report['source_unchanged']=all(_hash(ROOT/p)==h for p,h in hashes.items())
    if not report['source_unchanged']: raise RuntimeError('Source changed during bounded static audit')
    report['elapsed_wall_s']=time.monotonic()-began
    np.savez_compressed(output/'samples.npz',q_rad=samples,sampling_labels=labels,
        illegal_overlap_count=illegal,maximum_illegal_overlap_depth_m=max_overlap,
        legal_weapon_overlap_count=legal_overlap,**{k+'_zone_clearance_m':v for k,v in all_clearance.items()})
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Completed: {output}/report.json',flush=True)


if __name__=='__main__': main()
