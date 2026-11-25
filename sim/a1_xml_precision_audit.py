"""Measure saved-XML rounding without modifying a model, trace or scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .a1_profile_bank import blade_identity
from .assembly import _finite_json
from .blade_geometry import blade_collision_geom_ids
from .fencing_rules import _point_cylinder_distances
from .limited_duel import build_limited_duel
from .triangle_task import TriangleGeometry


def endpoints(model,data,ids):
    axis=data.geom_xmat[ids].reshape(-1,3,3)[:,:,2]
    half=model.geom_size[ids,1,None]
    return data.geom_xpos[ids]-half*axis,data.geom_xpos[ids]+half*axis,axis


def run(bank_path,native,output):
    bank_path,native,output=map(lambda p:Path(p).resolve(),(bank_path,native,output))
    if output.exists():raise FileExistsError('Preserve prior precision diagnostics')
    report=json.loads(bank_path.read_text())
    geometry=TriangleGeometry.from_mapping({k:report['scoring'][k]for k in ('base_distance','regions')})
    summary=json.loads((native/'summary.json').read_text())
    if summary.get('status')!='completed':raise ValueError('Completed trace required')
    exact=build_limited_duel(geometry.base_distance,initial_qpos=report['guards']['legacy_center']['qpos_rad'])
    rounded=mujoco.MjModel.from_xml_path(str(native/'scene.xml'));rd=mujoco.MjData(rounded)
    if blade_identity(exact.model,'a1_')['sha256']!=report['blade_geometry']['sha256']:
        raise ValueError('Current exact geometry differs from frozen bank')
    ids={s:np.asarray(blade_collision_geom_ids(exact.model,s+'_'))for s in ('a1','a2')}
    names={s:[exact.model.geom(int(i)).name for i in ids[s]]for s in ids}
    rids={s:np.array([rounded.geom(name).id for name in names[s]])for s in ids}
    qmap=[int(rounded.jnt_qposadr[rounded.joint(exact.model.joint(i).name).id])for i in range(exact.model.njnt)]
    if exact.model.nq!=exact.model.njnt or rounded.nq!=rounded.njnt:
        raise ValueError('This audit expects scalar joints and explicit name mapping')
    local={}
    for s in ids:
        q0,q1=exact.model.geom_quat[ids[s]],rounded.geom_quat[rids[s]]
        # Quaternion signs encode the same orientation; take the smaller norm.
        qdiff=np.minimum(np.linalg.norm(q0-q1,axis=1),np.linalg.norm(q0+q1,axis=1))
        local[s]={'maximum_radius_error_m':float(np.max(abs(exact.model.geom_size[ids[s],0]-rounded.geom_size[rids[s],0]))),
            'maximum_half_length_error_m':float(np.max(abs(exact.model.geom_size[ids[s],1]-rounded.geom_size[rids[s],1]))),
            'maximum_local_position_component_error_m':float(np.max(abs(exact.model.geom_pos[ids[s]]-rounded.geom_pos[rids[s]]))),
            'maximum_local_position_euclidean_error_m':float(np.max(np.linalg.norm(exact.model.geom_pos[ids[s]]-rounded.geom_pos[rids[s]],axis=1))),
            'maximum_quaternion_equivalent_l2_error':float(max(qdiff))}
    with np.load(native/'trace.npz',allow_pickle=False)as n:
        times,qpos=n['time_s'].copy(),n['qpos'].copy()
    max_endpoint={'error_m':0.};max_axis=0.;gap_delta={'error_m':0.}
    minima={m:{s:{r.name:float('inf')for r in geometry.regions}for s in ids}for m in ('exact','xml')}
    for t,q in zip(times,qpos):
        exact.data.qpos[:]=q[qmap];rd.qpos[:]=q
        mujoco.mj_kinematics(exact.model,exact.data);mujoco.mj_kinematics(rounded,rd)
        for s in ids:
            ea,eb,eu=endpoints(exact.model,exact.data,ids[s]);ra,rb,ru=endpoints(rounded,rd,rids[s])
            for endpoint,delta in (('start',ea-ra),('end',eb-rb)):
                norms=np.linalg.norm(delta,axis=1);j=int(np.argmax(norms))
                if norms[j]>max_endpoint['error_m']:
                    max_endpoint={'error_m':float(norms[j]),'time_s':float(t),
                                  'geom':names[s][j],'endpoint':endpoint,'component_error_m':delta[j].tolist()}
            max_axis=max(max_axis,float(np.max(np.linalg.norm(eu-ru,axis=1))))
            other='a2'if s=='a1'else'a1'
            for region in geometry.regions:
                c=np.array(geometry.world_center(region.name,other))
                ed,_=_point_cylinder_distances(c,ea,eb,exact.model.geom_size[ids[s],0])
                rdist,_=_point_cylinder_distances(c,ra,rb,rounded.geom_size[rids[s],0])
                eg,rg=float(min(ed)-region.radius),float(min(rdist)-region.radius)
                minima['exact'][s][region.name]=min(minima['exact'][s][region.name],eg)
                minima['xml'][s][region.name]=min(minima['xml'][s][region.name],rg)
                if abs(eg-rg)>gap_delta['error_m']:
                    gap_delta={'error_m':abs(eg-rg),'time_s':float(t),'owner':s,'region':region.name,
                               'exact_clearance_m':eg,'xml_clearance_m':rg}
    attacker=summary['metadata']['attacker_slot']
    minimum_attacker_gap=min(minima['exact'][attacker].values())
    result={'kind':'all_saved_native_states_XML_serialization_precision_audit',
            'source_bank':str(bank_path),'source_bank_sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            'native':str(native),'native_sha256':{n:hashlib.sha256((native/n).read_bytes()).hexdigest()
                for n in ('scene.xml','trace.npz','summary.json')},
            'exact_geometry_sha256':blade_identity(exact.model,'a1_')['sha256'],
            'xml_reload_geometry_sha256':blade_identity(rounded,'a1_')['sha256'],
            'native_recorded_runtime_geometry_sha256':summary['metadata']['collision_geometry_sha256'],
            'all_saved_frames_checked':True,'sample_count':len(times),'duration_s':float(times[-1]),
            'local_shape_errors':local,'maximum_world_endpoint_error':max_endpoint,
            'maximum_axis_vector_l2_error':max_axis,'maximum_target_clearance_difference':gap_delta,
            'minimum_target_clearances_m':minima,
            'minimum_exact_attacker_target_clearance_m':minimum_attacker_gap,
            'endpoint_error_to_minimum_gap_ratio':float(max_endpoint['error_m']/minimum_attacker_gap) if minimum_attacker_gap>0 else None,
            'scope':'Same recorded native joint states; exact constructor versus reloaded XML FK only. No re-integration or trajectory repair.',
            'scoring_tolerance_changed':False,'saved_data_changed':False,'admitted_to_training':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(_finite_json(result),indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ('local_shape_errors','maximum_world_endpoint_error',
          'maximum_target_clearance_difference','minimum_exact_attacker_target_clearance_m')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True)
    p.add_argument('--native',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.bank,a.native,a.output)
