"""Review a user-authored, rounded upright-guard reference without training.

The screenshot pose and an explicit opponent-facing J0 variant remain distinct.
No automatic mirroring, IK replacement, wrapping, motion bridge, new referee or
controller is installed. Optional native holding uses the existing shared runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .a1_runtime import apply_runtime_profile, load_runtime_profile
from .a1_triangle_native import audit_recorded_triangle
from .a1_upright_audit import upright_metrics
from .constrained_trial import hold_reference, run_constrained_trial
from .duel_scene import duel_contact_records
from .fencing_rules import FirstTouchScorer, blade_sections_from_scene
from .j3_zero_inspector import display_to_cad
from .joint_constraints import JOINT_NAMES, enforce_joint_trajectory, load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .triangle_task import TriangleGeometry


DEFAULT_REFERENCE = ROOT / 'config/manual_guard_reference_v1.json'
SCORING_PATH = ROOT / 'config/scoring_triangle_v1_candidate.json'


def numeric_seven(value, label):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (7,) or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, float, np.integer, np.floating)) for v in raw):
        raise ValueError(label + ' must contain seven numeric values, not booleans')
    out = raw.astype(float)
    if not np.isfinite(out).all():
        raise ValueError(label + ' must be finite')
    return out


def load_reference(path, constraints=None):
    constraints = load_joint_constraints() if constraints is None else constraints
    raw = json.loads(Path(path).read_text())
    if (type(raw.get('schema_version')) is not int or raw['schema_version'] != 1
            or raw.get('status') != 'approximate_user_guard_reference_only'
            or raw.get('joint_names') != list(JOINT_NAMES)
            or raw.get('display_coordinate_system') != 'hardware_zero_radians_J3_zero_means_straight'):
        raise ValueError('Expected an explicitly approximate ordered hardware-zero reference')
    offset = numeric_seven(raw.get('mapping_offset_deg'), 'mapping_offset_deg')
    if not np.array_equal(offset, constraints.offset_deg):
        raise ValueError('Reference zero mapping is stale; no silent reinterpretation')
    display = numeric_seven(raw.get('display_rad'), 'display_rad')
    cad = choose_pose(display, constraints)
    return raw, display, cad


def choose_pose(display_rad, constraints, yaw_delta_rad=0.):
    display = numeric_seven(display_rad, 'display_rad').copy()
    if (isinstance(yaw_delta_rad, (bool, np.bool_))
            or not isinstance(yaw_delta_rad, (int, float, np.number))
            or not np.isfinite(yaw_delta_rad)):
        raise ValueError('Base yaw delta must be a finite number')
    display[0] += yaw_delta_rad
    cad = display_to_cad(display, constraints)
    enforce_joint_trajectory(cad, constraints)
    return cad


def static_screen(scene, geometry):
    model, data = scene.model, scene.data
    mujoco.mj_forward(model, data)
    contacts = duel_contact_records(scene, rules_version='whole_blade_first_touch_v2')
    result = {'slots': {}, 'contacts': contacts, 'initial_target_overlap': {},
              'native_motion_tested': False, 'defence_certified': False,
              'admitted_to_training': False}
    for slot, ids in scene.agents.items():
        grip = data.site_xpos[ids['sites']['shinai_grip']].copy()
        tip = data.site_xpos[ids['sites']['shinai_tip']].copy()
        origin = np.array([0., 0., 0.]) if slot == 'a1' else np.array([geometry.base_distance, 0., 0.])
        flip = np.array([1., 1., 1.]) if slot == 'a1' else np.array([-1., -1., 1.])
        local_grip = (grip - origin) * flip
        axis = (tip - grip) / np.linalg.norm(tip - grip)
        q = data.qpos[ids['qpos_ids']]
        bounds = model.jnt_range[ids['joint_ids']]
        aid, did = ids['actuator_ids'], ids['dof_ids']
        if (not np.array_equal(model.actuator_gear[aid], np.tile([1., 0., 0., 0., 0., 0.], (7, 1)))
                or not np.array_equal(model.actuator_trnid[aid, 0], ids['joint_ids'])):
            raise ValueError('Static torque report requires native unit-gear joint actuators')
        demand = data.qfrc_bias[did] - data.qfrc_passive[did]
        torque_bounds = model.actuator_forcerange[aid]
        ratios = np.where(demand >= 0, demand / torque_bounds[:, 1], demand / torque_bounds[:, 0])
        result['slots'][slot] = {
            'original_cad_rad': q.tolist(), 'grip_self_frame_m': local_grip.tolist(),
            'grip_world_m': grip.tolist(), 'tip_world_m': tip.tolist(),
            'sword_axis_self_frame': (axis * flip).tolist(),
            'sword_elevation_deg': float(np.rad2deg(np.arcsin(np.clip(axis[2], -1., 1.)))),
            'upright_style_screen': upright_metrics([grip], [tip]),
            'grip_in_front_of_base': bool(local_grip[0] > 0),
            'actual_joint_ranges_pass': bool(np.isfinite(q).all() and
                np.all(q >= bounds[:, 0]) and np.all(q <= bounds[:, 1])),
            'nearest_angle_margin_deg': np.rad2deg(np.minimum(q - bounds[:, 0], bounds[:, 1] - q)).tolist(),
            'static_torque_demand_nm': demand.tolist(), 'joint_torque_bounds_nm': torque_bounds.tolist(),
            'static_cap_ratios': ratios.tolist(), 'ideal_static_demand_within_caps': bool(np.all(ratios <= 1.)),
        }
    blade = blade_sections_from_scene(scene)
    for region in geometry.regions:
        scorer = FirstTouchScorer({'a1': geometry.world_center(region.name, 'a2'),
                                  'a2': geometry.world_center(region.name, 'a1')}, region.radius)
        scorer.update(blade, blade, 0., 0.)
        if scorer.failure:
            raise ValueError('Static triangle geometry check failed: ' + str(scorer.failure))
        result['initial_target_overlap'][region.name] = scorer.initial_overlap_events
    # Conservative static gate, not a referee: neutral contact at reset is also
    # excluded here so this cannot inherit a preloaded blade-contact advantage.
    result['candidate_static_gate'] = bool(not contacts and
        not any(result['initial_target_overlap'].values()) and all(
            row['actual_joint_ranges_pass'] and row['grip_in_front_of_base'] and row['upright_style_screen']['pass']
            and row['ideal_static_demand_within_caps'] for row in result['slots'].values()))
    result['scope'] = 'Nominal static pose only; not a reachable, robust or effective guard.'
    return result


def run(source, output, *, native_hold=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous reference reviews; use a fresh output directory')
    source_bytes = source.read_bytes()
    constraints = load_joint_constraints()
    raw, display, original = load_reference(source, constraints)
    scoring = json.loads(SCORING_PATH.read_text())
    geometry = TriangleGeometry.from_mapping({k: scoring[k] for k in ('base_distance', 'regions')})
    facing = choose_pose(display, constraints, raw['facing_variant']['base_yaw_delta_rad'])
    profile = load_runtime_profile(ROOT / 'config/a1_runtime_rk4_candidate.json')
    report = {'schema_version': 1, 'status': 'approximate_user_guard_review',
              'source_reference': str(source), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_is_exact_user_export': False, 'scoring': scoring,
              'original_screenshot_display_rad': display.tolist(),
              'j0_facing_delta_rad': raw['facing_variant']['base_yaw_delta_rad'],
              'variants': {}, 'admitted_to_training': False, 'defence_certified': False,
              'automatic_bridges_or_wraps_used': False,
              'required_next_checks': ['exact user pose confirmation', 'visual facing-variant approval',
                                       'reachable continuous entries/exits', 'full-window physical parries',
                                       'left/right transitions and multi-round chains']}
    output.mkdir(parents=True)
    for name, pose in (('screenshot', original), ('opponent_facing', facing)):
        scene = build_limited_duel(geometry.base_distance, initial_qpos=pose, constraints=constraints)
        apply_runtime_profile(scene, profile)
        report['variants'][name] = static_screen(scene, geometry)
        report['variants'][name]['both_slots_same_joint_vector'] = True
        # An explicit reset is saved for reproduction, not a route to the pose.
        (output / (name + '.xml')).write_text(scene.spec.to_xml())
    def save():
        report['source_unchanged'] = source.read_bytes() == source_bytes
        (output / 'report.json').write_text(json.dumps(_finite_json(report), indent=2) + '\n')
    save()
    if native_hold and report['variants']['opponent_facing']['candidate_static_gate']:
        programs = {s: hold_reference(facing, start_s=.75, duration_s=1.) for s in ('a1', 'a2')}
        hold = run_constrained_trial(programs, output / 'native_hold', duration_s=3., progress=print,
            metadata={'manual_reference_sha256': report['source_sha256'],
                      'reference_is_rounded_screenshot': True, 'explicit_j0_delta_rad': float(facing[0]-original[0]),
                      'scope': 'same-pose native hold only; not approach, parry or training'})
        report['native_hold'] = {k: hold[k] for k in ('status', 'failure', 'completed_time_s',
            'native_execution_pass', 'entry_ready', 'final_settled', 'maximum_joint_reference_error_rad',
            'maximum_actual_joint_violation_rad', 'maximum_penetration_m', 'source_unchanged')}
        if hold['physics_steps']:
            triangle = audit_recorded_triangle(output / 'native_hold', scoring, target_region=None)
            (output / 'hold_triangle_audit.json').write_text(json.dumps(_finite_json(triangle), indent=2) + '\n')
            report['native_hold']['triangle_no_entry'] = not (triangle['all_entries'] or
                triangle['initial_overlap'] or triangle['sweep_failures'])
        report['native_hold']['defence_certified'] = False
        save()
    elif native_hold:
        report['native_hold'] = {'status': 'not_run_static_gate_failed'}
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-hold', action='store_true')
    args = parser.parse_args()
    result = run(args.source, args.output, native_hold=args.native_hold)
    print(json.dumps({'output': str(args.output.resolve()),
                      'facing_static_gate': result['variants']['opponent_facing']['candidate_static_gate'],
                      'native_hold': result.get('native_hold'), 'defence_certified': False}, indent=2))


if __name__ == '__main__':
    main()
