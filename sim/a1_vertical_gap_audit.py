"""Read-only exact finite-cylinder gap audit of a saved vertical-cut baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import _finite_json
from .blade_geometry import blade_collision_geom_ids
from .fencing_rules import _point_cylinder_distances
from .triangle_task import TriangleGeometry


def audit(native, bank, output, attacker='a1'):
    native, bank, output = Path(native), Path(bank), Path(output)
    if output.exists():
        raise FileExistsError('Keep previous diagnostics')
    report = json.loads(bank.read_text())
    summary = json.loads((native/'summary.json').read_text())
    if summary['status'] == 'running':
        raise ValueError('Audit only a completed or failed saved native trace')
    geometry = TriangleGeometry.from_mapping({k: report['scoring'][k] for k in ('regions', 'base_distance')})
    other = 'a2' if attacker == 'a1' else 'a1'
    model = mujoco.MjModel.from_xml_path(str(native/'scene.xml'))
    data = mujoco.MjData(model)
    ids = np.asarray(blade_collision_geom_ids(model, attacker+'_'))
    radii, half = model.geom_size[ids, 0], model.geom_size[ids, 1, None]
    qids = [int(model.jnt_qposadr[model.joint(attacker+'_J'+str(j)).id]) for j in range(7)]
    dofs = [int(model.jnt_dofadr[model.joint(attacker+'_J'+str(j)).id]) for j in range(7)]
    with np.load(native/'trace.npz', allow_pickle=False) as saved:
        trace = {k: saved[k].copy() for k in saved.files}
    results = {}
    for kind, qpath in [('actual', trace['qpos']), ('desired_reference', trace['desired_reference_qpos'])]:
        best = {r.name: None for r in geometry.regions}
        for i, q in enumerate(qpath):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            axis = data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
            starts, ends = data.geom_xpos[ids]-half*axis, data.geom_xpos[ids]+half*axis
            for region in geometry.regions:
                center = np.asarray(geometry.world_center(region.name, other))
                distance, points = _point_cylinder_distances(center, starts, ends, radii)
                j = int(np.argmin(distance)); gap = float(distance[j]-region.radius)
                if best[region.name] is None or gap < best[region.name]['clearance_m']:
                    best[region.name] = {'clearance_m': gap, 'index': i,
                        'global_time_s': float(trace['time_s'][i]),
                        'local_time_s': float(trace['time_s'][i]-.75),
                        'blade_geom': model.geom(int(ids[j])).name,
                        'blade_closest_point_world_m': points[j].tolist(),
                        'target_world_m': center.tolist(), 'target_radius_m': region.radius,
                        'qpos_rad': q[qids].tolist()}
        results[kind] = best
    for kind in results:
        for row in results[kind].values():
            i = row['index']
            row['actual_qpos_rad'] = trace['qpos'][i, qids].tolist()
            row['reference_qpos_rad'] = trace['desired_reference_qpos'][i, qids].tolist()
            row['actual_minus_reference_rad'] = (trace['qpos'][i, qids]-trace['desired_reference_qpos'][i, qids]).tolist()
            row['actual_qvel_rad_s'] = trace['qvel'][i, dofs].tolist()
            row['actuator_force_nm'] = trace['actuator_force_nm'][i, dofs].tolist()
    result = {'kind': 'read_only_saved_trace_finite_blade_target_gap',
              'bank_sha256': hashlib.sha256(bank.read_bytes()).hexdigest(),
              'native_source_sha256': {n: hashlib.sha256((native/n).read_bytes()).hexdigest()
                 for n in ('summary.json', 'scene.xml', 'trace.npz')},
              'attacker': attacker, 'region_results': results,
              'every_saved_frame_checked': True, 'sample_count': len(trace['time_s']),
              'swept_between_frames_recomputed': False,
              'method': 'Same finite-cylinder closest-point distance as scorer, sampled at every saved 1 ms frame.',
              'admitted_to_training': False, 'source_modified': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(result), indent=2)+'\n')
    print(json.dumps({k: v['upper'] for k, v in results.items()}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', required=True); p.add_argument('--bank', required=True)
    p.add_argument('--output', required=True); p.add_argument('--attacker', default='a1')
    args = p.parse_args(); audit(args.native, args.bank, args.output, args.attacker)
