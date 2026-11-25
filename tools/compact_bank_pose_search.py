"""Bounded whole-blade short-travel candidates; NEVER native certification.

Rank existing references first. Search the physical finite blade rather than
forcing its tip to a target. Solver seeds may be projected into a search box;
published curves are never clipped or wrapped. No physics/scoring is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from sim.a1_triangle_candidates import retimed_curve, static_screen, travel_cost
from sim.a1_triangle_screen import audit
from sim.assembly import ROOT, _finite_json
from sim.fencing_rules import BladeSection, point_cylinder_distance
from sim.joint_constraints import load_joint_constraints, model_joint_ranges_rad
from sim.limited_duel import build_limited_duel
from sim.limited_model import build_limited_floor
from sim.triangle_task import TriangleGeometry


def _save(path, data):
    path.write_text(json.dumps(_finite_json(data), indent=2, ensure_ascii=False)+'\n')


def rank_sources(paths):
    rows=[]
    for path in paths:
        report=json.loads(path.read_text())
        for name,action in report['actions'].items():
            if not action.get('sampled_reference_pass') or action.get('kind')!='attack':
                continue
            q=np.asarray(action['knot_qpos'])
            per_joint=np.sum(np.abs(np.diff(q,axis=0)),axis=0)
            rows.append({'source':str(path),'name':name,'region':action['target_region'],
                'start_guard':action['start_guard'],'end_guard':action['end_guard'],
                'j0_roundtrip_deg':float(np.rad2deg(per_joint[0])),
                'all_joint_travel_deg':float(np.rad2deg(per_joint.sum())),
                'duration_s':action['duration_s'],
                'first_j0_delta_deg':float(np.rad2deg(q[1,0]-q[0,0])),
                'native_dynamics_tested':False})
    return sorted(rows,key=lambda r:(r['j0_roundtrip_deg'],r['all_joint_travel_deg'],r['duration_s']))


def search(output, *, seed=418, starts=12, max_nfev=180, caps_deg=(30.,60.), ready_deg=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a fresh evidence directory')
    output.mkdir(parents=True)
    sources=[ROOT/'outputs/a1_triangle_screen_v2/report.json',
             ROOT/'outputs/a1_triangle_refine_v3/report.json']
    original=json.loads(sources[0].read_text())
    scoring=original['scoring']
    geometry=TriangleGeometry.from_mapping({k:scoring[k] for k in ('base_distance','regions')})
    constraints=load_joint_constraints()
    physical=model_joint_ranges_rad(constraints)
    old_center=np.asarray(original['guards']['center']['qpos_rad'])
    center=old_center if ready_deg is None else np.deg2rad(ready_deg)
    if center.shape!=(7,) or not np.isfinite(center).all() or np.any(center<=physical[:,0]) or np.any(center>=physical[:,1]):
        raise ValueError('Explicit finite interior ready pose required')
    model=build_limited_floor(constraints).model
    data=mujoco.MjData(model)
    scene=build_limited_duel(geometry.base_distance,initial_qpos=center,constraints=constraints)
    blade_ids=[model.geom('shinai_blade_collision_'+str(i)).id for i in range(3)]
    rng=np.random.default_rng(seed)
    report={'status':'searching','candidate_only':True,'admitted_to_training':False,
        'native_dynamics_tested':False,'scoring':scoring,'center_qpos_rad':center.tolist(),
        'old_center_qpos_rad':old_center.tolist(),'changed_common_ready':ready_deg is not None,
        'seed':seed,'starts_per_target_per_j0_cap':starts,'max_nfev':max_nfev,
        'joint_search_margin_rad':.02,'j0_caps_deg':list(caps_deg),
        'scope':'Endpoint whole-blade reachability and sampled path geometry only; no defence, dynamics or learning guarantee.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sources,Path(__file__),ROOT/'config/joint_constraints_observed.json']},
        'existing_ranking':rank_sources(sources),'endpoint_trials':[],'path_trials':[]}
    _save(output/'report.json',report)
    initial=static_screen(scene,geometry,center,center)
    report['candidate_ready_static_screen']=initial
    report['candidate_ready_sampled_pass']=all(not row['illegal_contacts'] and
        min(row['region_gaps_m'].values())>0 for row in initial.values())
    if not report['candidate_ready_sampled_pass']:
        report['status']='invalid_ready_pose';_save(output/'report.json',report)
        return report

    def blade_gap(q,target):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        best=np.inf
        for gid in blade_ids:
            radius,half=model.geom_size[gid,:2]
            axis=data.geom_xmat[gid].reshape(3,3)[:,2]
            p=data.geom_xpos[gid]
            section=BladeSection(p-half*axis,p+half*axis,radius)
            best=min(best,point_cylinder_distance(target,section)[0]-.1)
        return float(best)

    for region in ('left','right','upper'):
        target=np.asarray(geometry.world_center(region,'a2'))
        source_seeds=[center]
        for name,a in original['actions'].items():
            if a.get('target_region')==region and a.get('sampled_reference_pass'):
                q=np.asarray(a['knot_qpos'])[2]
                source_seeds.extend((q,(q+center)/2))
        unique=[]
        for value in source_seeds:
            if not any(np.max(np.abs(value-old))<1e-3 for old in unique):unique.append(value)
        source_seeds=unique+[center+rng.normal(0,.45,7) for _ in range(starts)]
        seen=[]; candidates=[]
        for cap in caps_deg:
            bounds=physical+np.array([.02,-.02])
            bounds[0]=[max(bounds[0,0],center[0]-np.deg2rad(cap)),
                       min(bounds[0,1],center[0]+np.deg2rad(cap))]
            weights=np.array([3.5,1.,1.,1.,1.,1.,.7])
            for idx,raw in enumerate(source_seeds[:starts]):
                initial=np.clip(raw,bounds[:,0]+1e-8,bounds[:,1]-1e-8)
                def residual(q):
                    # Deep-enough finite-cylinder overlap plus unwrapped
                    # displacement. J0 is discouraged, never frozen by fiat.
                    gap=blade_gap(q,target)
                    return np.r_[12.*max(0.,gap+.025),.025*weights*(q-center)]
                fit=least_squares(residual,initial,bounds=(bounds[:,0],bounds[:,1]),
                                  max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-8)
                q=fit.x;gap=blade_gap(q,target)
                check=static_screen(scene,geometry,q,center)
                collision_free=all(not row['illegal_contacts'] for row in check.values())
                good=gap<=-.012 and collision_free
                row={'region':region,'j0_cap_deg':cap,'seed_index':idx,
                    'qpos_rad':q.tolist(),'whole_blade_target_gap_m':gap,
                    'joint_delta_deg':np.rad2deg(q-center).tolist(),
                    'weighted_travel_rad':travel_cost([center,q],3.5),
                    'static_screen':check,'endpoint_pass':bool(good),
                    'solver_success':bool(fit.success),'nfev':int(fit.nfev)}
                report['endpoint_trials'].append(row)
                if good and not any(np.max(np.abs(q-old))<.015 for old in seen):
                    seen.append(q.copy());candidates.append((row,q))
            print(region,'cap',cap,'endpoint passes',len(candidates),flush=True)
            _save(output/'report.json',report)
        candidates.sort(key=lambda pair:pair[0]['weighted_travel_rad'])
        for index,(endpoint,q) in enumerate(candidates[:8]):
            near=center+.40*(q-center)
            curve=retimed_curve([center,near,q,q,near,center],1.2)
            result,arrays=audit(scene,geometry,curve,center,region,.02,None)
            name=f'{region}_short_whole_blade_{index:02d}'
            per_joint=np.abs(np.diff(curve.knot_qpos,axis=0)).sum(axis=0)
            trial={'name':name,'region':region,'endpoint_seed':endpoint,
                'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
                'duration_s':curve.duration_s,'j0_roundtrip_deg':float(np.rad2deg(per_joint[0])),
                'all_joint_travel_deg':float(np.rad2deg(per_joint.sum())),
                'sampled_reference_pass':result['sampled_reference_pass'],'audit':result,
                'native_dynamics_tested':False,'defence_validated':False}
            report['path_trials'].append(trial)
            np.savez_compressed(output/(name+'.npz'),**arrays,
                knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
            print(name,'sampled',result['sampled_reference_pass'],
                  'j0',round(trial['j0_roundtrip_deg'],2),flush=True)
            _save(output/'report.json',report)
    report['status']='bounded_search_completed_candidate_only'
    report['sampled_pass_names']=[r['name'] for r in report['path_trials'] if r['sampled_reference_pass']]
    _save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--starts',type=int,default=12)
    parser.add_argument('--max-nfev',type=int,default=180)
    parser.add_argument('--ready-deg',type=float,nargs=7)
    args=parser.parse_args()
    search(args.output,starts=args.starts,max_nfev=args.max_nfev,ready_deg=args.ready_deg)
