"""Build and screen closed Plan 2 reference motions without changing live rules.

Every path starts/ends at the same exact ready. Failed fits and full-path
collisions are retained. First target overlap is sampled geometry, not an
official score, dynamic execution or a successful defence certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from plan1_skills.matching import solve_pose, tool_fk
from .a1_plan2_keyframes import tip_cell
from .assembly import ROOT, _finite_json
from .duel_scene import classify_contact
from .fencing_rules import blade_sections_from_scene, point_cylinder_distance
from .joint_constraints import enforce_joint_trajectory, load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


DEFAULT = ROOT/'config/a1_plan2_continuous_v1.json'


def phase_at(now, strike_start, strike_end, withdraw_end):
    if now < strike_start:
        return 'approach'
    if now <= strike_end:
        return 'strike_or_hold'
    return 'withdraw' if now <= withdraw_end else 'return_ready'


def fit_offset(model, q_seed, displacement, bounds, opts):
    """One deterministic branch-local fit; no random branch jump on failure."""
    grip, axis, _ = tool_fk(model, qpos=q_seed)
    fit = solve_pose(model, grip+np.asarray(displacement), axis, bounds, initial_q=q_seed,
        max_iterations=opts['max_iterations'], position_tolerance_m=opts['position_tolerance_m'],
        angle_tolerance_rad=np.deg2rad(opts['axis_tolerance_deg']),
        tip_tolerance_m=opts['tip_tolerance_m'])
    return fit['qpos'], {'success': bool(fit['pose_success']),
        'position_error_m': float(fit['position_error_m']),
        'axis_error_deg': float(np.rad2deg(fit['angle_error_rad'])),
        'tip_error_m': float(fit['tip_error_m'])}


def closed_curve(times, nodes, ready):
    if not np.array_equal(nodes[0], ready) or not np.array_equal(nodes[-1], ready):
        raise ValueError('Closed references require the exact shared ready at both ends')
    return C2JointSpline(times, nodes)


def audit_curve(scene, curve, ready, scoring, sample_period, phase_bounds):
    times = np.linspace(0., curve.duration_s, int(np.ceil(curve.duration_s/sample_period))+1)
    qs = curve.sample(times)
    velocities, _ = curve.derivatives(times)
    ranges = model_joint_ranges_rad(load_joint_constraints())
    entire_bounds = bool(np.all(curve.bezier_controls >= ranges[:, 0]) and
                         np.all(curve.bezier_controls <= ranges[:, 1]))
    enforce_joint_trajectory(qs, load_joint_constraints())
    report = {'entire_reference_angle_bounds': entire_bounds,
        'sample_period_s_max': float(np.max(np.diff(times))), 'sample_count_per_slot': len(times),
        'native_dynamics_tested': False, 'sampled_slots': {},
        'continuous_collision_certificate': False, 'spline': curve.report()}
    for owner, defender in [('a1', 'a2'), ('a2', 'a1')]:
        model, data = scene.model, scene.data
        mapping = scene.agents[owner]
        qids, dofs = mapping['qpos_ids'], mapping['dof_ids']
        contacts, first, max_depth, min_gap = {}, None, 0., np.inf
        tip_positions = []
        target = scoring['targets'][defender]
        for i, (t, q, dq) in enumerate(zip(times, qs, velocities)):
            data.qpos[qids] = q
            data.qpos[scene.agents[defender]['qpos_ids']] = ready
            data.qvel[:] = 0.; data.qvel[dofs] = dq
            mujoco.mj_forward(model, data)
            tip_positions.append(data.site_xpos[mapping['sites']['shinai_tip']].copy())
            for c in data.contact:
                if c.dist >= -1e-9:
                    continue
                category = classify_contact(model, c, rules_version='whole_blade_first_touch_v2')
                max_depth = max(max_depth, -float(c.dist))
                if category.startswith('illegal_'):
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (c.geom1, c.geom2)]
                    contacts.setdefault('|'.join(names), {'first_time_s': float(t),
                        'depth_m_at_first': float(-c.dist), 'category': category, 'geoms': names})
            choices = []
            for section in blade_sections_from_scene(scene)[owner]:
                for zone, sphere in enumerate(target):
                    distance, point = point_cylinder_distance(np.asarray(sphere['center_world_m']), section)
                    choices.append((float(distance-sphere['radius_m']), point, section.name, zone))
            gap, point, geom_name, zone = min(choices, key=lambda x: x[0])
            min_gap = min(min_gap, gap)
            if gap < 0 and first is None:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                jac = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac, None, point, int(model.geom_bodyid[gid]))
                world_velocity = jac@data.qvel
                local_velocity = world_velocity * (np.array([1.,1.,1.]) if owner == 'a1' else np.array([-1.,-1.,1.]))
                first = {'first_inside_sample_s': float(t),
                    'entry_bracket_s': [float(times[max(0,i-1)]), float(t)],
                    'phase': phase_at(t, *phase_bounds), 'blade_geom': geom_name,
                    'closest_surface_point_world_m': point.tolist(), 'sphere_index': zone,
                    'point_cell_in_defender_frame': tip_cell(point, defender == 'a1'),
                    'reference_material_point_velocity_attacker_frame_m_s': local_velocity.tolist(),
                    'reference_material_point_speed_m_s': float(np.linalg.norm(world_velocity)),
                    'velocity_note': 'Instantaneous Jacobian velocity of this material point at first inside sample; not native measured velocity or a precontact-window statistic.',
                    'official_score': False}
        report['sampled_slots'][owner] = {'illegal_contacts': list(contacts.values()),
            'sampled_no_illegal_overlap': not contacts, 'maximum_sampled_penetration_m': max_depth,
            'minimum_sampled_target_gap_m': float(min_gap), 'first_sampled_target_overlap': first}
    report['reference_geometry_pass'] = entire_bounds and all(x['sampled_no_illegal_overlap'] for x in report['sampled_slots'].values())
    return report, {'time_s': times, 'qpos': qs, 'qvel': velocities}


def build(output, config=DEFAULT):
    output, config = Path(output).resolve(), Path(config).resolve()
    if output.exists():
        raise FileExistsError('Keep earlier candidates and failures; choose a new output')
    cfg = json.loads(config.read_text())
    sources = [config, ROOT/cfg['source_keyframes'], ROOT/cfg['scoring_proposal'], ROOT/cfg['source_seed_graph']]
    keyframes, scoring = (json.loads(p.read_text()) for p in sources[1:3])
    if cfg['admitted_to_training'] or scoring['runtime_activation_allowed']:
        raise ValueError('Candidate-only construction required')
    poses = keyframes['poses']; ready = np.asarray(poses['ready']['qpos_rad'])
    constraints = load_joint_constraints()
    ranges = model_joint_ranges_rad(constraints)
    bounds = ranges+np.array([cfg['solver']['joint_margin_rad'], -cfg['solver']['joint_margin_rad']])
    single = build_limited_floor(constraints)
    scene = build_limited_duel(scoring['base_distance_m'], initial_qpos=ready)
    output.mkdir(parents=True)
    report = {'status': 'building', 'scope': cfg['scope'], 'actions': {},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'implementation_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ('sim/a1_plan2_motion.py','sim/motion_spline.py','sim/limited_duel.py','sim/fencing_rules.py')},
        'rejected_high_static_poses_preserved': keyframes['failed_request_names'],
        'native_dynamics_tested': False, 'attack_or_defence_validated': False,
        'admitted_to_training': False, 'old_actor_compatible': False}
    def save():
        (output/'report.json').write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2)+'\n')
    candidates = []
    for name in cfg['thrust']['contact_names']:
        pose = poses[name]
        if not pose['static_candidate_pass']:
            raise ValueError('Cannot build an admitted source from a failed static pose')
        q = np.asarray(pose['qpos_rad'])
        pre, pfit = fit_offset(single.model, q, cfg['thrust']['pre_offset_m'], bounds, cfg['solver'])
        follow, ffit = fit_offset(single.model, q, cfg['thrust']['follow_offset_m'], bounds, cfg['solver'])
        candidates.append(('thrust_'+name.removeprefix('contact_'), 'attack', cfg['thrust']['times_s'],
                           [ready, pre, q, follow, pre, ready], [pfit,ffit], (3.,4.7,6.5), [1.,0.,0.]))
    for name, spec in cfg['sweeps'].items():
        ordered = [('right' if y > 0 else 'left') for y in spec['world_y_sequence_m']]
        base0 = poses[f"contact_{spec['height']}_{ordered[0]}"]
        base1 = poses[f"contact_{spec['height']}_{ordered[-1]}"]
        q0, q1 = np.asarray(base0['qpos_rad']), np.asarray(base1['qpos_rad'])
        pre, pfit = fit_offset(single.model, q0, [0., spec['world_y_sequence_m'][0]-base0['actual_tip_m'][1],0.], bounds, cfg['solver'])
        post, ffit = fit_offset(single.model, q1, [0., spec['world_y_sequence_m'][-1]-base1['actual_tip_m'][1],0.], bounds, cfg['solver'])
        withdraw, wfit = fit_offset(single.model, post, [-.22,0.,0.], bounds, cfg['solver'])
        candidates.append((name, 'attack', cfg['sweep_times_s'], [ready,pre,q0,q1,post,withdraw,ready],
                           [pfit,ffit,wfit], (3.,4.9,6.5), spec['intended_attacker_local_direction']))
    for name in cfg['interception_names']:
        if not poses[name]['static_candidate_pass']:
            raise ValueError('Failed interception static pose cannot be accepted')
        q = np.asarray(poses[name]['qpos_rad'])
        candidates.append((name, 'interception_reference_only', cfg['interception_times_s'],
                           [ready,q,q,ready], [], (3.,3.6,3.6), None))
    save()
    for name, kind, times, nodes, fits, phases, direction in candidates:
        curve = closed_curve(times, nodes, ready)
        audit, arrays = audit_curve(scene, curve, ready, scoring, cfg['sample_period_s'], phases)
        entry_expected = all(x['first_sampled_target_overlap'] is not None and
            x['first_sampled_target_overlap']['phase'] == 'strike_or_hold' for x in audit['sampled_slots'].values()) if kind == 'attack' else all(x['first_sampled_target_overlap'] is None for x in audit['sampled_slots'].values())
        passed = audit['reference_geometry_pass'] and all(f['success'] for f in fits) and entry_expected
        report['actions'][name] = {'kind': kind, 'source_fits': fits,
            'intended_attacker_local_direction': direction, 'expected_overlap_phase_pass': entry_expected,
            'continuous_reference_candidate_pass': passed, 'audit': audit,
            'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
            'native_dynamics_tested': False, 'successful_defence': False, 'admitted_to_training': False}
        np.savez_compressed(output/f'{name}.npz', **arrays, knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
        save(); print(name, 'continuous_reference_candidate_pass='+str(passed), flush=True)
    report['status'] = 'continuous_reference_candidates_complete_not_training_ready'
    (output/'scene.xml').write_text(scene.spec.to_xml())
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(DEFAULT))
    a = p.parse_args(); build(a.output, a.config)


if __name__ == '__main__':
    main()
