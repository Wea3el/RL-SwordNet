"""Task-space A1 redesign experiments. Not a live scorer or training library.

Joint branches and blade direction are optimized, not copied from an old
strike. Endpoints are named guards, not necessarily the same ready pose.
Every rejected fit/path is retained. FK sampling cannot certify a parry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plan1_skills.matching import solve_pose, tool_fk
from .assembly import ROOT, _finite_json
from .duel_scene import classify_contact
from .fencing_rules import blade_sections_from_scene, point_cylinder_distance
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


DEFAULT = ROOT / 'config/a1_triangle_motion_v1.json'


def travel_cost(nodes, j0_weight=1.6):
    """Unwrapped joint travel, not a modular shortest-angle shortcut."""
    q = np.asarray(nodes, dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('Finite Nx7 joint path required')
    weights = np.array([j0_weight, 1., 1., 1., 1., 1., 1.])
    return float(np.sum(np.abs(np.diff(q, axis=0)) * weights))


def attack_waypoints(center, shape, direction=None):
    """Targets are witnesses inside finite regions, not new scoring centers.

    The upper target also has a lower/near-side witness. Its original center
    request is retained separately by the builder, including failure.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Finite center required')
    contact = center + (np.array([-.080, 0., -.040]) if center[2] > 1.1
                        else np.array([-.065, 0., 0.]))
    if shape == 'thrust':
        # The high finite region is reachable near its lower/front edge, but
        # another 15 mm forward exceeds this arm's workspace. Follow down and
        # away there; do not force an unreachable extension or enlarge the goal.
        follow = contact + ([0., 0., -.025] if center[2] > 1.1 else [.015, 0., 0.])
        return [contact + [-.20, 0., 0.], contact, follow,
                contact + [-.20, 0., 0.]]
    d = np.asarray(direction, dtype=float)
    if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-9:
        raise ValueError('Finite nonzero cutting direction required')
    d /= np.linalg.norm(d)
    return [contact - .16*d + [-.07, 0., 0.], contact,
            contact + .13*d, contact + .13*d + [-.20, 0., 0.]]


def retimed_curve(nodes, speed_limit=2.):
    """Rest-to-rest reference, analytically retimed to the selected qdot cap."""
    q = np.asarray(nodes, dtype=float)
    if not np.isfinite(speed_limit) or speed_limit <= 0:
        raise ValueError('Positive finite reference speed required')
    durations = np.maximum(.30, 1.9*np.max(np.abs(np.diff(q, axis=0)), axis=1)/speed_limit)
    times = np.r_[0., np.cumsum(durations)]
    curve = C2JointSpline(times, q)
    peak = max(curve.analytic_extrema()['peak_speed_rad_s'])
    if peak > speed_limit:
        times *= peak / speed_limit * (1. + 1e-10)
        curve = C2JointSpline(times, q)
    return curve


def region_gaps(scene, geometry, owner):
    defender = 'a2' if owner == 'a1' else 'a1'
    blade = blade_sections_from_scene(scene)[owner]
    return {r.name: min(point_cylinder_distance(geometry.world_center(r.name, defender), b)[0]
                        for b in blade) - r.radius for r in geometry.regions}


def illegal_contacts(scene):
    result = []
    for contact in scene.data.contact:
        if contact.dist >= -1e-8:
            continue
        category = classify_contact(scene.model, contact, rules_version='whole_blade_first_touch_v2')
        if category.startswith('illegal_'):
            result.append({'category': category, 'depth_m': -float(contact.dist),
                           'geoms': [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(i))
                                     for i in (contact.geom1, contact.geom2)]})
    return result


def static_screen(scene, geometry, q, opponent):
    results = {}
    for owner, other in [('a1', 'a2'), ('a2', 'a1')]:
        scene.data.qpos[scene.agents[owner]['qpos_ids']] = q
        scene.data.qpos[scene.agents[other]['qpos_ids']] = opponent
        scene.data.qvel[:] = 0.
        mujoco.mj_forward(scene.model, scene.data)
        results[owner] = {'illegal_contacts': illegal_contacts(scene),
                          'region_gaps_m': region_gaps(scene, geometry, owner)}
    return results


def solve_tip_candidates(model, scene, geometry, target, reference, seeds, bounds, opts, opponent):
    """Position task + forward cone + small unwrapped travel regularizer.

    Rank feasible candidates by travel *after* fit and static collision gates.
    No original strike direction is a hard constraint. Each seed is reported.
    """
    data = mujoco.MjData(model)
    target, reference = np.asarray(target, float), np.asarray(reference, float)
    rows = []
    for seed in seeds:
        def residual(q):
            _, axis, tip = tool_fk(model, data, q)
            return np.r_[opts['position_weight'] * (tip-target),
                         opts['axis_cone_weight_m'] * max(0., opts['minimum_forward_axis']-axis[0]),
                         opts['joint_regularization'] * (q-reference)]
        fit = least_squares(residual, np.clip(seed, bounds[:, 0]+1e-9, bounds[:, 1]-1e-9),
                            bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=opts['max_nfev'])
        q = fit.x
        grip, axis, tip = tool_fk(model, data, q)
        error = float(np.linalg.norm(tip-target))
        check = static_screen(scene, geometry, q, opponent)
        feasible = error <= opts['tip_tolerance_m'] and axis[0] >= opts['minimum_forward_axis']-1e-3
        feasible = feasible and all(not r['illegal_contacts'] for r in check.values())
        rows.append({'qpos_rad': q.tolist(), 'tip_m': tip.tolist(), 'axis': axis.tolist(),
            'grip_m': grip.tolist(), 'tip_error_m': error,
            'weighted_travel_rad': travel_cost([reference, q], opts['j0_travel_weight']),
            'j0_delta_deg': float(np.rad2deg(q[0]-reference[0])),
            'solver_box_min_margin_rad': float(np.min(np.minimum(q-bounds[:, 0], bounds[:, 1]-q))),
            'physical_min_margin_rad': float(np.min(np.minimum(q-bounds[:, 0], bounds[:, 1]-q))) + opts['joint_margin_rad'],
            'static_screen': check, 'fit_pass': bool(feasible), 'nfev': fit.nfev})
    valid = [i for i, row in enumerate(rows) if row['fit_pass']]
    chosen = min(valid, key=lambda i: rows[i]['weighted_travel_rad']) if valid else None
    return {'target_tip_m': target.tolist(), 'trials': rows, 'selected_index': chosen,
            'fit_pass': chosen is not None}, None if chosen is None else np.array(rows[chosen]['qpos_rad'])


def audit_path(scene, geometry, curve, opponent, active_region, period, attack_indices=None):
    times = np.linspace(0., curve.duration_s, int(np.ceil(curve.duration_s/period))+1)
    q = curve.sample(times)
    ranges = model_joint_ranges_rad(load_joint_constraints())
    bounds_ok = bool(np.all(curve.bezier_controls >= ranges[:, 0]) and
                     np.all(curve.bezier_controls <= ranges[:, 1]))
    rows = {}
    for owner, other in [('a1', 'a2'), ('a2', 'a1')]:
        contacts, first, minima = {}, {}, {r.name: float('inf') for r in geometry.regions}
        opponent_entry = False
        for t, pose in zip(times, q):
            scene.data.qpos[scene.agents[owner]['qpos_ids']] = pose
            scene.data.qpos[scene.agents[other]['qpos_ids']] = opponent
            scene.data.qvel[:] = 0.
            mujoco.mj_forward(scene.model, scene.data)
            for c in illegal_contacts(scene):
                contacts.setdefault('|'.join(c['geoms']), dict(c, first_sample_s=float(t)))
            gaps = region_gaps(scene, geometry, owner)
            for name, gap in gaps.items():
                minima[name] = min(minima[name], gap)
                if gap < 0 and name not in first:
                    first[name] = float(t)
            opponent_entry |= any(gap < 0 for gap in region_gaps(scene, geometry, other).values())
        if active_region is None:
            entry_ok = not first
        else:
            start, end = curve.knot_time_s[list(attack_indices)]
            entry_ok = active_region in first and start <= first[active_region] <= end
            # A free duel would have ended at an earlier off-target entry.
            entry_ok &= not any(t < start for t in first.values())
        rows[owner] = {'illegal_contacts': list(contacts.values()), 'first_overlap_sample_s': first,
                       'minimum_region_gaps_m': minima, 'expected_entry_phase_pass': bool(entry_ok),
                       'fixed_opponent_entered_moving_agents_region': bool(opponent_entry)}
    passed = bounds_ok and all(not r['illegal_contacts'] and r['expected_entry_phase_pass'] and
                              not r['fixed_opponent_entered_moving_agents_region'] for r in rows.values())
    return {'sampled_reference_pass': bool(passed), 'bounds_proven_by_control_envelope': bounds_ok,
            'sample_period_s_max': float(np.max(np.diff(times))), 'slots': rows,
            'native_dynamics_tested': False, 'continuous_collision_certificate': False,
            'successful_defence': False}, {'time_s': times, 'qpos': q, 'qvel': curve.derivatives(times)[0]}


def build(output, config=DEFAULT, templates=None):
    output, config = Path(output).resolve(), Path(config).resolve()
    if output.exists():
        raise FileExistsError('Use a new output path; retain failed trials')
    cfg = json.loads(config.read_text())
    scoring_path, graph_path = ROOT/cfg['scoring'], ROOT/cfg['seed_graph']
    score, graph = json.loads(scoring_path.read_text()), json.loads(graph_path.read_text())
    if cfg['admitted_to_training'] or score['runtime_activation_allowed']:
        raise ValueError('Candidate-only specs required')
    geometry = TriangleGeometry.from_mapping({k: score[k] for k in ('base_distance', 'regions')})
    constraints = load_joint_constraints()
    ranges = model_joint_ranges_rad(constraints)
    bounds = ranges + np.array([cfg['solver']['joint_margin_rad'], -cfg['solver']['joint_margin_rad']])
    single = build_limited_floor(constraints)
    ready = np.array(graph['nodes']['ready'])
    scene = build_limited_duel(geometry.base_distance, initial_qpos=ready, constraints=constraints)
    base_seeds = [np.array(q) for q in graph['nodes'].values()]
    rng = np.random.default_rng(cfg['solver']['seed'])
    base_seeds += list(rng.uniform(bounds[:, 0], bounds[:, 1], size=(cfg['solver']['random_starts'], 7)))
    output.mkdir(parents=True)
    report = {'status': 'building_candidate_only', 'scoring': score,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [config, scoring_path, graph_path, Path(__file__), ROOT/'sim/triangle_task.py',
                                    ROOT/'sim/motion_spline.py', ROOT/'config/joint_constraints_observed.json']},
        'model_source_sha256': scene.source_hashes, 'guards': {}, 'endpoint_probes': {}, 'actions': {},
        'source': cfg['source'], 'native_dynamics_tested': False, 'admitted_to_training': False,
        'old_actor_compatible': False, 'trained_policy': False,
        'limitations': ['Sampled FK does not validate physical attacks, defensive timing, collisions between samples or tracking.',
                        'Guard and parry names are intentions, never score exemptions or defence certificates.',
                        'No live scorer, reward, observation, actor or controller was replaced.',
                        'Unwrapped branch search does not require reversing J0 when no legal shorter path exists.']}
    def save():
        (output/'report.json').write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2)+'\n')
    guards = {}
    for name, spec in cfg['guards'].items():
        trials = []
        if 'source_node' in spec:
            options = [np.array(graph['nodes'][spec['source_node']])]
        else:
            options = []
            for seed in base_seeds[:6]:
                result = solve_pose(single.model, spec['grip'], spec['axis'], bounds, initial_q=seed,
                                    max_iterations=160, position_tolerance_m=.008,
                                    angle_tolerance_rad=np.deg2rad(3.), tip_tolerance_m=.025)
                trials.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in result.items()})
                if result['pose_success']:
                    options.append(result['qpos'])
        clean = []
        for pose in options:
            check = static_screen(scene, geometry, pose, ready)
            if all(not r['illegal_contacts'] and min(r['region_gaps_m'].values()) > 0 for r in check.values()):
                clean.append(pose)
        chosen = min(clean, key=lambda q: travel_cost([ready, q])) if clean else None
        report['guards'][name] = {'trials': trials, 'static_reference_pass': chosen is not None,
                                 'qpos_rad': None if chosen is None else chosen.tolist(),
                                 'defence_validated': False}
        if chosen is not None:
            guards[name] = chosen
        save()
    if 'center' not in guards:
        raise RuntimeError('Shared center failed static gate; report retained')
    opts = cfg['solver']
    # Keep exact center requests separate: a reachable edge must not rewrite
    # the failed center result or become evidence of accurate center reaching.
    for region in geometry.regions:
        target = np.asarray(geometry.world_center(region.name, 'a2'))
        probe, _ = solve_tip_candidates(single.model, scene, geometry, target, ready,
                                        [ready]+base_seeds, bounds, opts, ready)
        report['endpoint_probes'][region.name+'_center'] = probe
        save()
    selected_templates = cfg['attack_templates'] if templates is None else {
        name: cfg['attack_templates'][name] for name in templates}
    for template, spec in selected_templates.items():
        center = np.asarray(geometry.world_center(spec['region'], 'a2'))
        points = attack_waypoints(center, spec['shape'], spec.get('attacker_direction'))
        for guard_name, start in guards.items():
            name = f'{template}__from_{guard_name}'
            nodes, fits = [start], []
            for index, point in enumerate(points):
                # Search many branches for entry. Stay near each prior result
                # for subsequent path samples; every attempted fit is retained.
                seeds = [nodes[-1]] + (base_seeds if index == 0 else base_seeds[:5])
                fit, chosen = solve_tip_candidates(single.model, scene, geometry, point, nodes[-1],
                                                   seeds, bounds, opts, ready)
                fits.append(fit)
                if chosen is None:
                    break
                nodes.append(chosen)
            if len(nodes) != len(points)+1:
                report['actions'][name] = {'kind': 'attack', 'template': template, 'target_region': spec['region'],
                    'start_guard': guard_name, 'waypoint_fits': fits,
                    'sampled_reference_pass': False, 'failure': 'task_waypoint_fit_failed',
                    'native_dynamics_tested': False, 'admitted_to_training': False}
                save(); print(name, 'waypoint_fit_failed', flush=True)
                continue
            # Check each exit, then select the least travel *valid whole path*.
            exits = []
            for end_name, end in guards.items():
                curve = retimed_curve(nodes+[end], cfg['reference_speed_limit_rad_s'])
                audit, arrays = audit_path(scene, geometry, curve, ready, spec['region'],
                                          cfg['sample_period_s'], attack_indices=(1, 3))
                exits.append((end_name, curve, audit, arrays))
            valid = [x for x in exits if x[2]['sampled_reference_pass']]
            end_name, curve, audit, arrays = min(valid or exits,
                key=lambda x: travel_cost(x[1].knot_qpos, opts['j0_travel_weight']))
            report['actions'][name] = {'kind': 'attack', 'template': template, 'target_region': spec['region'],
                'start_guard': guard_name, 'end_guard': end_name, 'waypoint_fits': fits,
                'alternate_exits': {n: {'audit': a, 'weighted_travel_rad': travel_cost(c.knot_qpos)}
                                    for n,c,a,_ in exits}, 'audit': audit,
                'sampled_reference_pass': audit['sampled_reference_pass'],
                'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
                'spline': curve.report(), 'weighted_travel_rad': travel_cost(curve.knot_qpos),
                'j0_total_travel_deg': float(np.rad2deg(np.abs(np.diff(curve.knot_qpos[:, 0])).sum())),
                'j0_initial_motion_deg': float(np.rad2deg(curve.knot_qpos[1,0]-curve.knot_qpos[0,0])),
                'native_dynamics_tested': False, 'admitted_to_training': False}
            np.savez_compressed(output/f'{name}.npz', **arrays,
                                knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
            save(); print(name, 'reference_pass='+str(audit['sampled_reference_pass']), flush=True)
    # Directed guard edges enable a graph. They remain non-admitted until
    # reactive paired attacks and native tracking have been tested.
    for source, start in guards.items():
        for dest, end in guards.items():
            if source == dest:
                continue
            name = f'guard_{source}_to_{dest}'
            curve = retimed_curve([start, end], cfg['reference_speed_limit_rad_s'])
            audit, arrays = audit_path(scene, geometry, curve, ready, None, cfg['sample_period_s'])
            report['actions'][name] = {'kind': 'guard_transition_not_verified_parry', 'start_guard': source,
                'end_guard': dest, 'audit': audit, 'sampled_reference_pass': audit['sampled_reference_pass'],
                'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
                'native_dynamics_tested': False, 'admitted_to_training': False}
            np.savez_compressed(output/f'{name}.npz', **arrays,
                                knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
            save()
    passed = [name for name,a in report['actions'].items() if a['sampled_reference_pass']]
    report.update(status='candidate_reference_screen_complete_not_training_ready',
                  sampled_reference_pass_names=passed,
                  attack_region_coverage={r.name: [n for n in passed if report['actions'][n].get('target_region') == r.name]
                                          for r in geometry.regions})
    (output/'scene.xml').write_text(scene.spec.to_xml())
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(DEFAULT))
    p.add_argument('--templates', nargs='+')
    args = p.parse_args()
    build(args.output, args.config, args.templates)


if __name__ == '__main__':
    main()
