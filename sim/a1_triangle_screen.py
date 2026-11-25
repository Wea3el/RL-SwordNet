"""Stricter offline screen of triangle references; never a game certificate.

Retain ALL clean exit alternatives, distinguish targeted from first-hit
coverage, and measure a material point's reference velocity instead of using
the template name as evidence of a cutting direction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .a1_triangle_candidates import illegal_contacts, region_gaps, retimed_curve, travel_cost
from .assembly import ROOT, _finite_json
from .fencing_rules import blade_sections_from_scene, point_cylinder_distance
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .triangle_task import TriangleGeometry


def first_region_checks(first_entries, target, attack_window):
    """Sample-time logic only. Ties are unresolved, not native double events."""
    start, end = attack_window
    targeted = target in first_entries and start <= first_entries[target] <= end
    if not first_entries:
        return {'targeted_entry_pass': False, 'unrestricted_first_region_pass': False,
                'first_regions_at_sample_resolution': []}
    earliest = min(first_entries.values())
    first = sorted(k for k,v in first_entries.items() if abs(v-earliest) <= 1e-12)
    return {'targeted_entry_pass': bool(targeted),
            'unrestricted_first_region_pass': bool(targeted and first == [target]),
            'first_regions_at_sample_resolution': first}


def audit(scene, geometry, curve, opponent, target, period, intended_direction=None):
    times = np.linspace(0, curve.duration_s, int(np.ceil(curve.duration_s/period))+1)
    poses = curve.sample(times); velocities = curve.derivatives(times)[0]
    bounds = model_joint_ranges_rad(load_joint_constraints())
    ranges_ok = bool(np.all(curve.bezier_controls >= bounds[:,0]) and np.all(curve.bezier_controls <= bounds[:,1]))
    attack_window = (float(curve.knot_time_s[1]), float(curve.knot_time_s[3])) if target else None
    rows = {}
    for owner, other in [('a1', 'a2'), ('a2', 'a1')]:
        contacts, entries, witnesses = {}, {}, {}
        minimum = {r.name: float('inf') for r in geometry.regions}
        opponent_entry = False
        for sample, (t, q, dq) in enumerate(zip(times, poses, velocities)):
            mapping = scene.agents[owner]
            scene.data.qpos[mapping['qpos_ids']] = q
            scene.data.qpos[scene.agents[other]['qpos_ids']] = opponent
            scene.data.qvel[:] = 0.
            scene.data.qvel[mapping['dof_ids']] = dq
            mujoco.mj_forward(scene.model, scene.data)
            for c in illegal_contacts(scene):
                contacts.setdefault('|'.join(c['geoms']), dict(c, time_s=float(t)))
            sections = blade_sections_from_scene(scene)
            for region in geometry.regions:
                center = np.asarray(geometry.world_center(region.name, other))
                choices = []
                for section in sections[owner]:
                    distance, point = point_cylinder_distance(center, section)
                    choices.append((distance-region.radius, point, section.name))
                gap, point, geom_name = min(choices, key=lambda row: row[0])
                minimum[region.name] = min(minimum[region.name], float(gap))
                if gap < 0 and region.name not in entries:
                    entries[region.name] = float(t)
                    gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                    jac = np.zeros((3, scene.model.nv))
                    mujoco.mj_jac(scene.model, scene.data, jac, None, point, int(scene.model.geom_bodyid[gid]))
                    world_velocity = jac @ scene.data.qvel
                    local_velocity = world_velocity * (np.ones(3) if owner == 'a1' else np.array([-1.,-1.,1.]))
                    cosine = None
                    if intended_direction is not None and np.linalg.norm(local_velocity) > 1e-8:
                        direction = np.asarray(intended_direction, float)
                        cosine = float(local_velocity @ direction/(np.linalg.norm(local_velocity)*np.linalg.norm(direction)))
                    witnesses[region.name] = {'entry_bracket_s': [float(times[max(0,sample-1)]),float(t)],
                        'blade_material_point_world_m': point.tolist(),
                        'reference_velocity_attacker_frame_m_s': local_velocity.tolist(),
                        'reference_direction_cosine': cosine,
                        'native_velocity_measurement': False}
            opponent_entry |= any(gap < 0 for gap in region_gaps(scene, geometry, other).values())
        checks = first_region_checks(entries, target, attack_window) if target else {
            'targeted_entry_pass': not entries, 'unrestricted_first_region_pass': not entries,
            'first_regions_at_sample_resolution': []}
        cosine = witnesses.get(target, {}).get('reference_direction_cosine')
        direction_ok = intended_direction is None or cosine is not None and cosine >= .35
        # This margin is a screen, not a tracking tolerance or contact certificate.
        penetration_ok = target is None or minimum[target] < -.005
        initial_no_overlap = not entries or min(entries.values()) > 0.
        rows[owner] = dict(checks, first_overlap_sample_s=entries, entry_witnesses=witnesses,
            minimum_region_gaps_m=minimum, illegal_contacts=list(contacts.values()),
            reference_direction_screen_pass=bool(direction_ok),
            reference_penetration_screen_pass=bool(penetration_ok),
            initial_no_overlap=bool(initial_no_overlap),
            fixed_opponent_entered_moving_agents_region=bool(opponent_entry))
    passed = ranges_ok and all(row['unrestricted_first_region_pass'] and
        row['reference_direction_screen_pass'] and row['reference_penetration_screen_pass'] and
        row['initial_no_overlap'] and not row['illegal_contacts'] and
        not row['fixed_opponent_entered_moving_agents_region'] for row in rows.values())
    return {'sampled_reference_pass': bool(passed), 'slots': rows,
        'analytic_joint_bounds_pass': ranges_ok,
        'true_min_joint_margin_rad': float(np.min(np.minimum(curve.bezier_controls-bounds[:,0],bounds[:,1]-curve.bezier_controls))),
        'sample_period_s_max': float(np.max(np.diff(times))),
        'native_dynamics_tested': False, 'native_direction_verified': False,
        'continuous_collision_certificate': False, 'successful_defence': False}, {
            'time_s': times, 'qpos': poses, 'qvel': velocities}


def screen(source, output, additional_sources=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous screens; choose a new output directory')
    paths = [source/'report.json', source/'scene.xml', ROOT/'config/a1_triangle_motion_v1.json', Path(__file__)]
    previous = json.loads(paths[0].read_text())
    previous = copy.deepcopy(previous)
    for extra in additional_sources:
        extra_path = Path(extra).resolve()/'report.json'
        replacement = json.loads(extra_path.read_text())
        if replacement['scoring'] != previous['scoring'] or replacement['guards'] != previous['guards']:
            raise ValueError('Additional candidate runs must have identical geometry and guard solutions')
        previous['actions'].update(replacement['actions'])
        paths.append(extra_path)
    cfg = json.loads(paths[2].read_text())
    score = previous['scoring']
    geometry = TriangleGeometry.from_mapping({k: score[k] for k in ('base_distance','regions')})
    guards = {n: np.array(g['qpos_rad']) for n,g in previous['guards'].items() if g['static_reference_pass']}
    scene = build_limited_duel(geometry.base_distance, initial_qpos=guards['center'])
    output.mkdir(parents=True)
    report = {'status': 'screening', 'scoring': score, 'guards': copy.deepcopy(previous['guards']),
        'endpoint_probes': copy.deepcopy(previous['endpoint_probes']), 'actions': {},
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'preserved_source_failures': {n: a for n,a in previous['actions'].items() if 'knot_qpos' not in a},
        'scope': 'ALL exit alternatives; sampled geometry and reference velocity only',
        'native_dynamics_tested': False, 'admitted_to_training': False,
        'old_actor_compatible': False, 'trained_policy': False,
        'coverage_note': 'Unique earliest region at sample resolution; not substep/native adjudication.'}
    def save():
        (output/'report.json').write_text(json.dumps(_finite_json(report),ensure_ascii=False,indent=2)+'\n')
    for name, original in previous['actions'].items():
        if 'knot_qpos' not in original:
            continue
        is_attack = original['kind'] == 'attack'
        nodes = np.asarray(original['knot_qpos'])
        variants = guards.items() if is_attack else [(original['end_guard'], nodes[-1])]
        target = original.get('target_region')
        template = cfg['attack_templates'].get(original.get('template'), {})
        direction = [1.,0.,0.] if template.get('shape') == 'thrust' else template.get('attacker_direction')
        for exit_name, exit_q in variants:
            variant = name+'__to_'+exit_name if is_attack else name
            curve = retimed_curve(list(nodes[:-1])+[exit_q], cfg['reference_speed_limit_rad_s'])
            result, arrays = audit(scene, geometry, curve, guards['center'], target,
                                   cfg['sample_period_s'], direction)
            report['actions'][variant] = {'kind': original['kind'], 'template': original.get('template'),
                'start_guard': original['start_guard'], 'end_guard': exit_name, 'target_region': target,
                'sampled_reference_pass': result['sampled_reference_pass'], 'audit': result,
                'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
                'duration_s': curve.duration_s, 'spline': curve.report(),
                'weighted_joint_travel_rad': travel_cost(curve.knot_qpos),
                'j0_initial_motion_deg': float(np.rad2deg(nodes[1,0]-nodes[0,0])),
                'direction_intent_attacker_frame': direction,
                'native_dynamics_tested': False, 'admitted_to_training': False}
            np.savez_compressed(output/f'{variant}.npz', **arrays,
                                knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
        save(); print('Screened',name,flush=True)
    passing = {n:a for n,a in report['actions'].items() if a['sampled_reference_pass']}
    report.update(status='screen_complete_not_training_ready',
        sampled_reference_pass_names=list(passing),
        attack_region_coverage={r.name: [n for n,a in passing.items() if a['target_region'] == r.name] for r in geometry.regions},
        guard_exit_coverage={g: [n for n,a in passing.items() if a['end_guard'] == g] for g in guards})
    (output/'scene.xml').write_text(scene.spec.to_xml())
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--additional-source', action='append', default=[])
    a=p.parse_args(); screen(a.source,a.output,a.additional_source)


if __name__ == '__main__':
    main()
