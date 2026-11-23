"""Named guard/continuous-cut candidates under the confirmed joint limits.

This is a new, separately versioned kinematic construction layer. It is not a
replacement training bank: the old five-action actor cannot load these names.
No failed pose is dropped from the report, no command is clipped or wrapped,
and no native dynamics/defensive effectiveness is claimed by this builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from plan1_skills.matching import match_sequence, screen_joint_path, solve_pose, tool_fk
from .assembly import ROOT, _finite_json
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad, enforce_joint_trajectory
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


DEFAULT_REQUIREMENTS = ROOT/'config/a1_pose_requirements_v1.json'


def load_requirements(path=DEFAULT_REQUIREMENTS):
    cfg = json.loads(Path(path).read_text())
    if cfg.get('schema_version') != 1 or len(cfg.get('poses', {})) != 10 or len(cfg.get('cuts', {})) != 5:
        raise ValueError('The explicit ten-guard, five-cut schema is required')
    for name, pose in cfg['poses'].items():
        p, a = np.asarray(pose['grip_m'], dtype=float), pose['axis']
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f'{name}: finite grip target required')
        if a is not None and (np.asarray(a).shape != (3,) or not np.isfinite(a).all()
                              or not np.isclose(np.linalg.norm(a), 1., atol=1e-9)):
            raise ValueError(f'{name}: unit blade axis or explicit null required')
    for route in cfg['cuts'].values():
        if len(route) < 2 or any(name not in cfg['poses'] for name in route):
            raise ValueError('Cut routes must refer to existing named guards')
    if not np.isfinite(cfg['preview_seconds_per_leg']) or cfg['preview_seconds_per_leg'] <= 0:
        raise ValueError('Positive preview timing required')
    return cfg


def _position_only(model, target, initial, bounds, opts):
    """Kron has no supplied blade orientation: do not invent an axis gate."""
    rng = np.random.default_rng(opts['seed'])
    seeds = [initial, (bounds[:, 0]+bounds[:, 1])/2]
    seeds += list(rng.uniform(bounds[:, 0], bounds[:, 1], size=(opts['first_frame_starts']-2, 7)))
    best, rank = None, None
    for seed in seeds:
        _, axis, _ = tool_fk(model, qpos=seed)
        fit = solve_pose(model, target, axis, bounds, initial_q=seed,
            max_iterations=opts['max_iterations'], position_tolerance_m=opts['position_tolerance_m'],
            angle_tolerance_rad=np.pi, orientation_weight_m=1e-6)
        found, _ = screen_joint_path(model, fit['qpos'], collision_threshold_m=.001)
        key = (not fit['pose_success'], bool(found), fit['position_error_m'])
        if rank is None or key < rank:
            best, rank = dict(fit, contacts=found), key
        if fit['pose_success'] and not found:
            break
    return {'qpos_rad': best['qpos'].tolist(), 'pose_fit': bool(best['pose_success']),
        'collision_free': not best['contacts'], 'position_error_m': best['position_error_m'],
        'axis_error_deg': None, 'axis_unconstrained_by_request': True, 'contacts': best['contacts']}


def fit_named_pose(model, request, initial, bounds, opts):
    if request['axis'] is None:
        return _position_only(model, request['grip_m'], initial, bounds, opts)
    fit = match_sequence(model, np.asarray([request['grip_m']]), np.asarray([request['axis']]), bounds,
        initial_q=initial, seed=opts['seed'], max_iterations=opts['max_iterations'],
        position_tolerance_m=opts['position_tolerance_m'],
        angle_tolerance_rad=np.deg2rad(opts['axis_tolerance_deg']),
        tip_tolerance_m=opts['tip_tolerance_m'], first_frame_starts=opts['first_frame_starts'])
    return {'qpos_rad': fit.qpos[0].tolist(), 'pose_fit': bool(fit.pose_success[0]),
        'collision_free': bool(fit.collision_free[0]), 'position_error_m': float(fit.position_error_m[0]),
        'axis_error_deg': float(np.rad2deg(fit.angle_error_rad[0])),
        'axis_unconstrained_by_request': False, 'contacts': fit.contacts[0]}


def canonical_graph_nodes(poses, spokes):
    """Choose one exact joint state per connected node, never mix IK branches.

    Independent static fits remain diagnostics in ``poses``.  A future hold or
    transition executor must use these graph nodes and the associated checked
    spoke, not teleport between two IK solutions with the same tool pose.
    This function does not certify a new edge between arbitrary node pairs.
    """
    hub_pose = poses['long_point']
    if not hub_pose['usable_candidate']:
        return {}
    hub = np.asarray(hub_pose['qpos_rad'], dtype=float)
    nodes = {'long_point': {'qpos_rad': hub.tolist(), 'source': 'shared_hub'}}
    for name, path in spokes.items():
        q = np.asarray(path, dtype=float)
        if (name == 'long_point' or name not in poses or not poses[name]['usable_candidate']
                or q.ndim != 2 or q.shape[1:] != (7,) or len(q) < 2
                or not np.isfinite(q).all() or not np.array_equal(q[0], hub)):
            raise ValueError('A graph node requires a continuous spoke from the exact shared hub')
        nodes[name] = {'qpos_rad': q[-1].tolist(), 'source': 'checked_spoke_endpoint',
            'independent_static_solution_max_delta_rad': float(np.max(np.abs(
                q[-1]-np.asarray(poses[name]['qpos_rad']))))}
    return nodes


def connect_from_long_point(model, poses, name, bounds, cfg):
    """Warm-start a complete task-space spoke; no independent IK branch jumps."""
    source, target = poses['long_point'], poses[name]
    p0, p1 = np.asarray(source['actual_grip_m']), np.asarray(target['actual_grip_m'])
    a0, a1 = np.asarray(source['actual_axis']), np.asarray(target['actual_axis'])
    phase = np.linspace(0., 1., cfg['connection_samples_per_leg'])
    angle = np.arccos(np.clip(a0@a1, -1., 1.))
    if angle > np.pi-1e-5:
        raise ValueError('Antipodal spoke needs an explicit direction waypoint')
    axis = ((1-phase[:, None])*a0+phase[:, None]*a1 if angle < 1e-6 else
        (np.sin((1-phase)*angle)[:, None]*a0+np.sin(phase*angle)[:, None]*a1)/np.sin(angle))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    grip = p0+phase[:, None]*(p1-p0)
    grip[:, 0] += cfg['connection_forward_clearance_m']*np.sin(np.pi*phase)
    opts = cfg['solver']
    fit = match_sequence(model, grip, axis, bounds, initial_q=np.asarray(source['qpos_rad']),
        seed=opts['seed'], max_iterations=opts['max_iterations'],
        position_tolerance_m=opts['position_tolerance_m'],
        angle_tolerance_rad=np.deg2rad(opts['axis_tolerance_deg']),
        tip_tolerance_m=opts['tip_tolerance_m'], first_frame_starts=1)
    return fit


def compile_connected_route(spokes, route, seconds_per_leg):
    if len(route) == 2 and route[1] == 'long_point':
        q = spokes[route[0]][::-1].copy()
        return C2JointSpline(np.linspace(0., seconds_per_leg, len(q)), q)
    if len(route) != 3 or route[1] != 'long_point':
        raise ValueError('This prototype connects one cut through its explicit Long Point hub')
    start, end = spokes[route[0]][::-1], spokes[route[2]]
    if not np.array_equal(start[-1], end[0]):
        raise ValueError('Connected spokes must share exactly the same joint-space Long Point')
    times = np.r_[np.linspace(0., seconds_per_leg, len(start)),
                   np.linspace(seconds_per_leg, 2*seconds_per_leg, len(end))[1:]]
    return C2JointSpline(times, np.vstack((start, end[1:])))


def audit_flow(model, spline, constraints, *, sample_period_s=.01):
    """Full polynomial angle bounds plus sampled geometry, not a dynamics test."""
    ranges = model_joint_ranges_rad(constraints)
    controls = spline.bezier_controls
    # Bezier convex hull certifies every interpolated command, not only samples.
    bounded = bool(np.all(controls >= ranges[:, 0]-1e-12) and np.all(controls <= ranges[:, 1]+1e-12))
    times = np.linspace(0., spline.duration_s, int(np.ceil(spline.duration_s/sample_period_s))+1)
    qs = spline.sample(times)
    enforce_joint_trajectory(qs, constraints)
    data = mujoco.MjData(model)
    contacts, max_depth, tips, tip_speeds = {}, 0., [], []
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'shinai_tip')
    velocities, _ = spline.derivatives(times)
    for t, q, velocity in zip(times, qs, velocities):
        data.qpos[:] = q
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
        tips.append(data.site_xpos[tip_id].copy())
        jac = np.zeros((3, model.nv)); mujoco.mj_jacSite(model, data, jac, None, tip_id)
        tip_speeds.append(float(np.linalg.norm(jac@velocity)))
        for contact in data.contact:
            if contact.dist < -.001:
                pair = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in (contact.geom1, contact.geom2))
                contacts.setdefault('|'.join(pair), {'time_s': float(t), 'distance_m': float(contact.dist)})
            max_depth = max(max_depth, -float(contact.dist))
    via = []
    for t in spline.knot_time_s[1:-1]:
        q, (v, _) = spline.sample(t), spline.derivatives(t)
        data.qpos[:], data.qvel[:] = q, v; mujoco.mj_forward(model, data)
        jac = np.zeros((3, model.nv)); mujoco.mj_jacSite(model, data, jac, None, tip_id)
        via.append({'time_s': float(t), 'joint_speed_norm_rad_s': float(np.linalg.norm(v)),
                    'tip_speed_m_s': float(np.linalg.norm(jac@v))})
    report = {'angle_bounds_entire_curve': bounded, 'sample_count': len(times),
        'sampled_collision_free': not contacts, 'sampled_illegal_geometry': contacts,
        'sampled_max_penetration_m': max_depth, 'tip_speed_peak_m_s': max(tip_speeds),
        'via_points': via, 'spline': spline.report(),
        'native_physics_tested': False, 'attack_or_defence_effectiveness_tested': False,
        'requested_velocity_gt_1_checked': False,
        'usable_kinematic_candidate': bounded and not contacts}
    return report, {'time_s': times, 'qpos': qs, 'qvel': velocities,
                    'tip_m': np.asarray(tips), 'tip_speed_m_s': np.asarray(tip_speeds)}


def build_flow(output, requirements=DEFAULT_REQUIREMENTS):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new directory; failed candidates must remain available')
    cfg = load_requirements(requirements)
    constraints = load_joint_constraints()
    assembly = build_limited_floor(constraints)
    bounds = model_joint_ranges_rad(constraints)+np.array([cfg['solver']['joint_margin_rad'], -cfg['solver']['joint_margin_rad']])
    output.mkdir(parents=True)
    report = {'schema_version': 2, 'status': 'building', 'scope': 'new constrained named-guard / flow prototype',
        'requirements': cfg, 'poses': {}, 'cuts': {}, 'admitted_to_training': False,
        'pose_solution_role': 'Independent static IK diagnostics, not transition endpoints; use graph_nodes with their checked spokes.',
        'old_v5_compatible': False, 'old_actor_compatible': False,
        'reason': 'Different guard graph and confirmed angle bounds; old five-action actor and validation cannot be reused.',
        'model_source_sha256': assembly.source_hashes,
        'implementation_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('sim/a1_flow.py', 'sim/motion_spline.py', 'sim/joint_constraints.py', 'sim/limited_model.py')},
        'joint_bounds_deg': [list(x) for x in constraints.bounds_deg]}
    def save(): (output/'report.json').write_text(json.dumps(_finite_json(report), indent=2)+'\n')
    save()
    initial = assembly.home.copy()
    for name, pose in cfg['poses'].items():
        result = fit_named_pose(assembly.model, pose, initial, bounds, cfg['solver'])
        p, a, _ = tool_fk(assembly.model, qpos=np.asarray(result['qpos_rad']))
        result.update(request=pose, actual_grip_m=p.tolist(), actual_axis=a.tolist(),
                      usable_candidate=result['pose_fit'] and result['collision_free'])
        enforce_joint_trajectory(result['qpos_rad'], constraints)
        report['poses'][name] = result
        if name == 'long_point' and result['usable_candidate']:
            initial = np.asarray(result['qpos_rad'])
        save(); print(name, result['usable_candidate'], flush=True)
    spokes = {}
    report['connections'] = {}
    if report['poses']['long_point']['usable_candidate']:
        hub = np.asarray(report['poses']['long_point']['qpos_rad'])
        for name, pose in report['poses'].items():
            if name == 'long_point' or not pose['usable_candidate']:
                continue
            fit = connect_from_long_point(assembly.model, report['poses'], name, bounds, cfg)
            # Do not repair a branch mismatch by rewriting its first row.
            if not np.array_equal(fit.qpos[0], hub):
                raise ValueError('Spoke solver changed the exact shared Long Point state')
            report['connections'][name] = fit.summary()
            np.savez_compressed(output/f'connection_{name}.npz', qpos=fit.qpos,
                pose_success=fit.pose_success, collision_free=fit.collision_free)
            if fit.success.all():
                spoke_curve = C2JointSpline(np.linspace(0., cfg['preview_seconds_per_leg'],
                                                       len(fit.qpos)), fit.qpos)
                validation, _ = audit_flow(assembly.model, spoke_curve, constraints)
                report['connections'][name]['continuous_reference_audit'] = validation
                if validation['usable_kinematic_candidate']:
                    spokes[name] = fit.qpos.copy()
            save()
    report['graph_nodes'] = canonical_graph_nodes(report['poses'], spokes)
    for name, route in cfg['cuts'].items():
        missing = [n for n in route if not report['poses'][n]['usable_candidate']]
        missing += [n for n in route if n != 'long_point' and n not in spokes]
        if missing:
            report['cuts'][name] = {'route': route, 'built': False, 'failed_required_poses': missing}
        else:
            spline = compile_connected_route(spokes, route, cfg['preview_seconds_per_leg'])
            validation, arrays = audit_flow(assembly.model, spline, constraints)
            report['cuts'][name] = dict(validation, route=route, built=True)
            np.savez_compressed(output/f'{name}.npz', **arrays,
                knot_time_s=spline.knot_time_s, knot_qpos=spline.knot_qpos,
                coefficients=spline.coefficients)
        save()
    report['missing_poses'] = [n for n,p in report['poses'].items() if not p['usable_candidate']]
    report['status'] = 'kinematic_candidates_complete' if not report['missing_poses'] and all(
        v.get('usable_kinematic_candidate', False) for v in report['cuts'].values()) else 'partial_candidates_not_training_ready'
    (output/'scene.xml').write_text(assembly.spec.to_xml())
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--requirements', default=str(DEFAULT_REQUIREMENTS))
    a = p.parse_args()
    result = build_flow(a.output, a.requirements)
    print(json.dumps({'status': result['status'], 'missing_poses': result['missing_poses']}))


if __name__ == '__main__': main()
