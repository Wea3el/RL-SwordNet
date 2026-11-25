"""Bounded Plan 2 static keyframes, never motion/defence/training certificates.

Solve six target-position requests and three interception geometry requests.
The existing ready is preserved exactly. All failed fits and static overlaps
remain in the report. No production scorer, controller or skill bank changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from plan1_skills.matching import tool_fk
from .a1_flow import fit_named_pose
from .assembly import ROOT, _finite_json
from .duel_scene import classify_contact
from .fencing_rules import blade_sections_from_scene, point_cylinder_distance
from .joint_constraints import enforce_joint_trajectory, load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor


DEFAULT = ROOT / 'config/a1_plan2_keyframes_v1.json'


def tip_cell(point, defender_faces_positive_x):
    """Position label only, not a first-contact event or attack direction."""
    p = np.asarray(point, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Finite 3D point required')
    local_y = p[1] if defender_faces_positive_x else -p[1]
    height = 'low' if p[2] < .975 else 'mid' if p[2] < 1.125 else 'high'
    return height + ('_left' if local_y >= 0 else '_right')


def attack_requests(cfg, scoring, axis, length):
    axis = np.asarray(axis, float)
    if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1.) or length <= 0:
        raise ValueError('Measured unit sword axis and positive tool length required')
    center_x = scoring['targets']['a2'][0]['center_world_m'][0]
    result = {}
    for height, z in cfg['attack_heights_m'].items():
        for side, sign in [('left', -1.), ('right', 1.)]:
            # Attacker a1 faces +X; defender a2 faces -X, so its left is world -Y.
            tip = np.array([center_x-cfg['attack_tip_inset_toward_attacker_m'],
                            sign*cfg['attack_tip_lateral_offset_m'], z])
            name = f'contact_{height}_{side}'
            result[name] = {'kind': 'contact_configuration_candidate',
                'requested_score_cell': f'{height}_{side}',
                'requested_tip_m': tip.tolist(),
                'grip_m': (tip-length*axis).tolist(), 'axis': axis.tolist(),
                'intended_future_motion_direction': None,
                'motion_direction_note': 'Must be assigned and measured on a continuous path; blade axis is not cutting direction.'}
    return result


def static_dual_check(scene, q, ready, scoring):
    """Mirror the same q in both slots against unchanged ready opponents."""
    results = {}
    for owner, other in [('a1', 'a2'), ('a2', 'a1')]:
        data, model = scene.data, scene.model
        data.qpos[scene.agents[owner]['qpos_ids']] = q
        data.qpos[scene.agents[other]['qpos_ids']] = ready
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        contacts = []
        for c in data.contact:
            if c.dist >= -1e-9:
                continue
            category = classify_contact(model, c, rules_version='whole_blade_first_touch_v2')
            contacts.append({'category': category, 'depth_m': float(-c.dist),
                'geoms': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (c.geom1, c.geom2)]})
        blades = blade_sections_from_scene(scene)[owner]
        target = scoring['targets'][other]
        gaps = [min(point_cylinder_distance(np.asarray(s['center_world_m']), b)[0]
                    for b in blades)-s['radius_m'] for s in target]
        tip = data.site_xpos[scene.agents[owner]['sites']['shinai_tip']].copy()
        tip_inside = any(np.linalg.norm(tip-np.asarray(s['center_world_m'])) < s['radius_m'] for s in target)
        results[owner] = {'contacts': contacts,
            'no_illegal_static_overlap': not any(x['category'].startswith('illegal_') for x in contacts),
            'opponent_target_sphere_gaps_m': list(map(float, gaps)),
            'blade_overlaps_opponent_target_union': min(gaps) < 0,
            'actual_tip_world_m': tip.tolist(),
            'actual_tip_inside_target_union': bool(tip_inside),
            'actual_tip_cell': tip_cell(tip, other == 'a1') if tip_inside else None,
            'entire_blade_intersected_cells': None,
            'cell_note': 'Tip position is exact; sphere overlaps are not disjoint cells. Full blade/cell intersection and first-entry location remain unimplemented.'}
    return results


def build(output, config=DEFAULT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; retain failed candidates')
    config = Path(config).resolve()
    cfg = json.loads(config.read_text())
    scoring_path, seed_path = (ROOT/cfg[n] for n in ('scoring_proposal', 'seed_graph'))
    scoring, graph = (json.loads(p.read_text()) for p in (scoring_path, seed_path))
    if scoring['runtime_activation_allowed'] or cfg['admitted_to_training']:
        raise ValueError('This builder only accepts non-deployed candidate specifications')
    ready = np.asarray(graph['nodes']['ready'])
    constraints = load_joint_constraints()
    enforce_joint_trajectory(ready, constraints)
    single = build_limited_floor(constraints)
    dual = build_limited_duel(scoring['base_distance_m'], initial_qpos=ready, constraints=constraints)
    _, axis, tip = tool_fk(single.model, qpos=np.asarray(graph['nodes']['original_strike']))
    grip, _, _ = tool_fk(single.model, qpos=np.asarray(graph['nodes']['original_strike']))
    requests = attack_requests(cfg, scoring, axis, float(np.linalg.norm(tip-grip)))
    for name, request in cfg['interception_requests'].items():
        requests[name] = dict(request, kind='interception_geometry_candidate',
                             requested_score_cell=None, intended_future_motion_direction=None)
    bounds = model_joint_ranges_rad(constraints)+np.array([cfg['solver']['joint_margin_rad'], -cfg['solver']['joint_margin_rad']])
    output.mkdir(parents=True)
    source_files = [config, scoring_path, seed_path, ROOT/'config/joint_constraints_observed.json', Path(__file__)]
    report = {'schema_version': 1, 'status': 'building_static_candidates',
        'source': cfg['source'], 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'model_source_sha256': dual.source_hashes,
        'pose_request_config': cfg, 'scoring_proposal_id': scoring['proposal_id'],
        'continuous_path_tested': False, 'native_dynamics_tested': False,
        'attack_or_defence_validated': False, 'admitted_to_training': False,
        'measured_motion_direction': None, 'poses': {},
        'limitations': ['Independent IK solutions must not be switched directly in a live episode.',
            'Static intersection is not a legal hit, defence or measured cutting direction.',
            'No full-blade disjoint-cell/first-entry test, path, reaction timing, recovery or naturalness is certified.',
            'A2 remains free to use continuous commands under future common rules; these endpoints do not gate scoring.']}
    def save():
        (output/'report.json').write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2)+'\n')
    p, a, t = tool_fk(single.model, qpos=ready)
    report['poses']['ready'] = {'kind': 'existing_exact_shared_ready', 'qpos_rad': ready.tolist(),
        'actual_grip_m': p.tolist(), 'actual_axis': a.tolist(), 'actual_tip_m': t.tolist(),
        'dual_static': static_dual_check(dual, ready, ready, scoring),
        'note': 'Previously used ready; not re-certified dynamically under this scoring proposal.'}
    save()
    for name, request in requests.items():
        initial = np.asarray(graph['nodes']['original_strike'] if name.startswith('contact_') else graph['nodes']['original_glancing_intercept'])
        fit = fit_named_pose(single.model, request, initial, bounds, cfg['solver'])
        q = np.asarray(fit['qpos_rad'])
        enforce_joint_trajectory(q, constraints)
        p, a, t = tool_fk(single.model, qpos=q)
        checks = static_dual_check(dual, q, ready, scoring)
        okay = fit['pose_fit'] and fit['collision_free'] and all(v['no_illegal_static_overlap'] for v in checks.values())
        failures = []
        if not fit['pose_fit']: failures.append('task_space_fit_failed')
        if not fit['collision_free']: failures.append('single_arm_static_overlap')
        if not all(v['no_illegal_static_overlap'] for v in checks.values()): failures.append('dual_static_overlap')
        if name.startswith('contact_') and not all(v['actual_tip_cell'] == request['requested_score_cell'] for v in checks.values()):
            failures.append('requested_tip_cell_not_reached_in_both_slots'); okay = False
        if not name.startswith('contact_') and any(v['blade_overlaps_opponent_target_union'] for v in checks.values()):
            failures.append('interception_pose_already_occupies_opponent_target'); okay = False
        report['poses'][name] = dict(fit, request=request, actual_grip_m=p.tolist(),
            actual_axis=a.tolist(), actual_tip_m=t.tolist(), dual_static=checks,
            physical_joint_minimum_margin_rad=float(np.min(np.minimum(q-model_joint_ranges_rad(constraints)[:, 0], model_joint_ranges_rad(constraints)[:, 1]-q))),
            static_candidate_pass=bool(okay), failure_reasons=failures,
            measured_motion_direction=None, attack_or_defence_validated=False, admitted_to_training=False)
        save()
        print(name, 'static_candidate_pass='+str(bool(okay)), flush=True)
    report['status'] = 'static_keyframes_complete_not_motion_validated'
    report['static_candidate_pass_names'] = [n for n,v in report['poses'].items() if v.get('static_candidate_pass')]
    report['failed_request_names'] = [n for n,v in report['poses'].items() if v.get('failure_reasons')]
    (output/'scene.xml').write_text(dual.spec.to_xml())
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(DEFAULT))
    a = p.parse_args()
    build(a.output, a.config)


if __name__ == '__main__':
    main()
