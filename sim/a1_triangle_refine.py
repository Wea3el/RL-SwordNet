"""Bounded upper-motion refinement; reference evidence, not admitted skills.

Change preparation/follow-through geometry instead of enlarging the target or
silently relaxing the contact test.  All old failures and all new IK trials are
retained.  A passing nominal reference is explicitly separate from native
physics / actual defensive effectiveness. The .04 rad / .03 m robustness
thresholds are diagnostic engineering assumptions, not user-confirmed scoring
or hardware limits; neither result alone admits or rejects a training skill.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from plan1_skills.matching import solve_pose, tool_fk
from .a1_triangle_candidates import retimed_curve, static_screen, travel_cost
from .a1_triangle_screen import audit
from .assembly import ROOT
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .triangle_task import TriangleGeometry


TEMPLATES = ('thrust_upper', 'cut_up_upper_from_left',
             'cut_up_upper_from_right', 'cut_straight_down_upper')
ROBUST_MARGIN_RAD = .04
ROBUST_PENETRATION_M = .03


def refinement_waypoints(center, template):
    """Preserve the target; draw recovery towards the robot's own workspace.

    Upper cuts need not keep rising far beyond the finite scoring region.
    Straight-down preparation can be high and closer to the robot.  The
    contact witness is the same near/lower witness as the preceding run.
    Labels only request direction; the material-point screen measures it.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('finite three-dimensional region center required')
    contact = center + [-.080, 0., -.040]
    if template == 'thrust_upper':
        points = [contact+[-.20, 0., 0.], contact,
                  contact+[-.04, 0., -.020], contact+[-.24, 0., -.020]]
        direction = [1., 0., 0.]
    elif template in ('cut_up_upper_from_left', 'cut_up_upper_from_right'):
        sign = -1. if template.endswith('left') else 1.
        direction = np.array([0., sign, 1.])
        points = [contact-.16*direction/np.sqrt(2.)+[-.07, 0., 0.], contact,
                  contact+[-.07, sign*.04, .04], contact+[-.25, sign*.04, .02]]
    elif template == 'cut_straight_down_upper':
        direction = [0., 0., -1.]
        points = [contact+[-.15, 0., .16], contact,
                  contact+[-.07, 0., -.12], contact+[-.24, 0., -.12]]
    else:
        raise ValueError('unsupported refinement template')
    return np.asarray(points), np.asarray(direction)


def robustness_metrics(result, margin_rad=ROBUST_MARGIN_RAD,
                       penetration_m=ROBUST_PENETRATION_M, target='upper'):
    """Diagnostic sensitivity only: these values do not decide admission."""
    if not np.isfinite(margin_rad) or margin_rad < 0 or not np.isfinite(penetration_m) or penetration_m <= 0:
        raise ValueError('finite nonnegative margin and positive penetration required')
    margin = float(result['true_min_joint_margin_rad'])
    penetration = min(-float(row['minimum_region_gaps_m'][target])
                      for row in result['slots'].values())
    return {'minimum_required_joint_margin_rad': float(margin_rad),
            'minimum_required_penetration_m': float(penetration_m),
            'minimum_reference_joint_margin_rad': margin,
            'minimum_slot_penetration_m': penetration,
            'joint_margin_pass': margin >= margin_rad-1e-10,
            'penetration_pass': penetration >= penetration_m,
            'robustness_pass': bool(result['sampled_reference_pass'] and
                                   margin >= margin_rad-1e-10 and penetration >= penetration_m),
            'diagnostic_robustness_assumptions': True,
            'non_admission_guarantee': True,
            'scope': 'Reference geometry margins only; not a native tracking certificate'}


def _json_ready(value):
    """Keep failed solver arrays, scalar types and nonfinite evidence usable."""
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _distinct_seeds(seeds):
    unique = []
    for seed in seeds:
        q = np.asarray(seed, float)
        if q.shape == (7,) and np.isfinite(q).all() and not any(np.allclose(q, x, atol=1e-9, rtol=0) for x in unique):
            unique.append(q)
    return unique


def _solve_tip(model, scene, geometry, target, reference, seeds, bounds,
               opponent, max_nfev, contact_phase):
    data = mujoco.MjData(model)
    rows = []
    for seed in _distinct_seeds(seeds):
        def residual(q):
            _, axis, tip = tool_fk(model, data, q)
            # During preparation/recovery there is no invented forward-axis
            # requirement. Contact still has a soft non-backward preference.
            cone = .2*max(0., -axis[0]) if contact_phase else 0.
            return np.r_[tip-target, cone, .002*(q-reference)]
        fit = least_squares(residual, np.clip(seed, bounds[:, 0]+1e-9, bounds[:, 1]-1e-9),
                            bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=max_nfev)
        q = fit.x
        grip, axis, tip = tool_fk(model, data, q)
        error = float(np.linalg.norm(tip-target))
        check = static_screen(scene, geometry, q, opponent)
        passed = error <= .012 and (not contact_phase or axis[0] >= -1e-3)
        passed = passed and all(not row['illegal_contacts'] for row in check.values())
        rows.append({'qpos_rad': q.tolist(), 'tip_m': tip.tolist(), 'axis': axis.tolist(),
                     'grip_m': grip.tolist(), 'tip_error_m': error, 'fit_pass': bool(passed),
                     'weighted_travel_rad': travel_cost([reference, q]),
                     'minimum_solver_box_margin_rad': float(np.min(np.minimum(q-bounds[:, 0], bounds[:, 1]-q))),
                     'static_screen': check, 'nfev': int(fit.nfev), 'solver_success': bool(fit.success)})
    valid = [i for i, row in enumerate(rows) if row['fit_pass']]
    chosen = min(valid, key=lambda i: rows[i]['weighted_travel_rad']) if valid else None
    return {'target_tip_m': np.asarray(target).tolist(), 'contact_phase': contact_phase,
            'forward_preference': 'soft axis_x >= 0' if contact_phase else 'none',
            'trials': rows, 'selected_index': chosen, 'fit_pass': chosen is not None}, (
            None if chosen is None else np.asarray(rows[chosen]['qpos_rad']))


def refine(source, output, *, margin_rad=.01, max_nfev=110,
           start_guards=('center', 'right'), templates=TEMPLATES, period=.01):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve previous evidence; choose a new output directory')
    if not np.isfinite(margin_rad) or margin_rad <= 0 or margin_rad >= .2:
        raise ValueError('solver margin must be in (0, .2) rad; not a new hardware limit')
    if type(max_nfev) is not int or max_nfev < 10:
        raise ValueError('max_nfev must be an integer >= 10')
    if not np.isfinite(period) or period <= 0 or period > .025:
        raise ValueError('period must be in (0, .025] seconds')
    if not templates or any(t not in TEMPLATES for t in templates):
        raise ValueError('select at least one known upper template')
    previous = json.loads((source/'report.json').read_text())
    score = copy.deepcopy(previous['scoring'])
    if score.get('runtime_activation_allowed') or score.get('admitted_to_training'):
        raise ValueError('only inactive candidate geometry can be refined')
    geometry = TriangleGeometry.from_mapping({k: score[k] for k in ('base_distance', 'regions')})
    cfg_path = ROOT/'config/a1_triangle_motion_v1.json'
    cfg = json.loads(cfg_path.read_text())
    constraints = load_joint_constraints()
    physical_bounds = model_joint_ranges_rad(constraints)
    bounds = physical_bounds+np.array([margin_rad, -margin_rad])
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('margin leaves no feasible joint interval')
    source_guards = {name: np.asarray(row['qpos_rad'], float)
                     for name, row in previous['guards'].items() if row['static_reference_pass']}
    if 'center' not in source_guards:
        raise ValueError('a checked center reference is required')
    model = build_limited_floor(constraints).model
    scene = build_limited_duel(geometry.base_distance, initial_qpos=source_guards['center'], constraints=constraints)
    output.mkdir(parents=True)
    inputs = [source/'report.json', source/'scene.xml', cfg_path, Path(__file__),
              ROOT/'sim/a1_triangle_screen.py', ROOT/'sim/a1_triangle_candidates.py',
              ROOT/'sim/motion_spline.py', ROOT/'config/joint_constraints_observed.json']
    report = {'status': 'building_refinement_candidate_only', 'scoring': score,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'model_source_sha256': scene.source_hashes,
              'solver_margin_rad': margin_rad, 'robust_minimum_margin_rad': ROBUST_MARGIN_RAD,
              'robust_minimum_penetration_m': ROBUST_PENETRATION_M,
              'diagnostic_robustness_assumptions': True, 'non_admission_guarantee': True,
              'reference_speed_limit_rad_s': 2., 'sample_period_s': period,
              'source': 'Robot-task-space authored local path refinement; not ASE-trained motions.',
              'guards': {}, 'actions': {},
              'preserved_source_failures': {n: copy.deepcopy(a) for n,a in previous['actions'].items()
                                            if not a.get('sampled_reference_pass')},
              'preserved_source_nested_failures': copy.deepcopy(previous.get('preserved_source_failures', {})),
              'native_dynamics_tested': False, 'native_direction_verified': False,
              'admitted_to_training': False, 'trained_policy': False, 'old_actor_compatible': False,
              'limitations': ['Nominal sample-screen success is not robustness, dynamics or defence.',
                              'No scoring geometry, live reward, opponent observation or hardware limit changed.',
                              'Retained nominal upper contact is near the target edge; robust depth is separately gated.',
                              'Existing lower/side actions are not recertified by this upper-only refinement.']}
    def save():
        (output/'report.json').write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2)+'\n')
    save()
    guards = {}
    for name, original in source_guards.items():
        check = static_screen(scene, geometry, original, source_guards['center'])
        valid = bool(np.all(original >= bounds[:, 0]) and np.all(original <= bounds[:, 1]) and
                     all(not r['illegal_contacts'] and min(r['region_gaps_m'].values()) > 0 for r in check.values()))
        trials, candidates = [], [original] if valid else []
        spec = cfg['guards'][name]
        if not valid:
            data = mujoco.MjData(model)
            grip, axis, _ = tool_fk(model, data, original)
            grip, axis = spec.get('grip', grip), spec.get('axis', axis)
            for seed in _distinct_seeds([original, source_guards['center'], source_guards.get('high', original)]):
                result = solve_pose(model, grip, axis, bounds, initial_q=np.clip(seed, bounds[:, 0]+1e-9, bounds[:, 1]-1e-9),
                                    max_iterations=max_nfev, position_tolerance_m=.015,
                                    angle_tolerance_rad=np.deg2rad(7.), tip_tolerance_m=.06)
                q = np.asarray(result['qpos'])
                check2 = static_screen(scene, geometry, q, source_guards['center'])
                clean = bool(result['pose_success'] and np.all(q >= bounds[:, 0]) and np.all(q <= bounds[:, 1]) and
                             all(not r['illegal_contacts'] and min(r['region_gaps_m'].values()) > 0 for r in check2.values()))
                trials.append({'solver_result': result, 'static_screen': check2, 'fit_pass': clean})
                if clean:
                    candidates.append(q)
        chosen = min(candidates, key=lambda q: travel_cost([original, q])) if candidates else None
        report['guards'][name] = {'qpos_rad': None if chosen is None else chosen.tolist(),
            'original_qpos_rad': original.tolist(), 'original_retained': valid,
            'refinement_trials': trials, 'static_reference_pass': chosen is not None,
            'minimum_physical_joint_margin_rad': None if chosen is None else float(np.min(
                np.minimum(chosen-physical_bounds[:, 0], physical_bounds[:, 1]-chosen))),
            'defence_validated': False}
        if chosen is not None:
            guards[name] = chosen
        save()
    if 'center' not in guards or any(name not in guards for name in start_guards):
        report['status'] = 'guard_refinement_failed_not_training_ready'; save()
        return report
    opponent = guards['center']
    center = np.asarray(geometry.world_center('upper', 'a2'))
    for template in templates:
        points, direction = refinement_waypoints(center, template)
        for start_name in start_guards:
            prefix = template+'__from_'+start_name
            start, fits = guards[start_name], []
            nodes = [start]
            for index, point in enumerate(points):
                seeds = [nodes[-1], guards.get('high', opponent), opponent]
                # Old candidates are only IK seeds; no old certificate survives.
                old = next((a for a in previous['actions'].values() if a.get('template') == template
                            and a.get('start_guard') == start_name and 'knot_qpos' in a), None)
                if old is not None and index+1 < len(old['knot_qpos'])-1:
                    seeds.append(old['knot_qpos'][index+1])
                fit, q = _solve_tip(model, scene, geometry, point, nodes[-1], seeds, bounds,
                                    opponent, max_nfev, index == 1)
                fits.append(fit)
                if q is None:
                    break
                nodes.append(q)
            if len(nodes) != 5:
                report['actions'][prefix] = {'kind': 'attack', 'template': template, 'target_region': 'upper',
                    'start_guard': start_name, 'waypoint_fits': fits, 'failure': 'task_waypoint_fit_failed',
                    'sampled_reference_pass': False, 'robustness_pass': False,
                    'native_dynamics_tested': False, 'admitted_to_training': False}
                save(); print(prefix, 'waypoint_fit_failed', flush=True)
                continue
            for exit_name, exit_q in guards.items():
                name = prefix+'__to_'+exit_name
                curve = retimed_curve(nodes+[exit_q], 2.)
                result, arrays = audit(scene, geometry, curve, opponent, 'upper', period, direction)
                robustness = robustness_metrics(result)
                report['actions'][name] = {'kind': 'attack', 'template': template, 'target_region': 'upper',
                    'start_guard': start_name, 'end_guard': exit_name, 'waypoint_fits': fits,
                    'direction_intent_attacker_frame': direction.tolist(),
                    'sampled_reference_pass': result['sampled_reference_pass'], 'audit': result,
                    'robustness': robustness, 'robustness_pass': robustness['robustness_pass'],
                    'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
                    'duration_s': curve.duration_s, 'spline': curve.report(),
                    'weighted_joint_travel_rad': travel_cost(curve.knot_qpos),
                    'native_dynamics_tested': False, 'admitted_to_training': False}
                np.savez_compressed(output/(name+'.npz'), **arrays,
                                    knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
            save(); print(prefix, 'screened_all_guard_exits', flush=True)
    for start_name, start in guards.items():
        for exit_name, exit_q in guards.items():
            if start_name == exit_name:
                continue
            name = 'guard_'+start_name+'_to_'+exit_name
            curve = retimed_curve([start, exit_q], 2.)
            result, arrays = audit(scene, geometry, curve, opponent, None, period)
            report['actions'][name] = {'kind': 'guard_transition_not_verified_parry',
                'start_guard': start_name, 'end_guard': exit_name, 'target_region': None,
                'sampled_reference_pass': result['sampled_reference_pass'], 'audit': result,
                'knot_time_s': curve.knot_time_s.tolist(), 'knot_qpos': curve.knot_qpos.tolist(),
                'duration_s': curve.duration_s, 'native_dynamics_tested': False,
                'admitted_to_training': False, 'successful_defence': False}
            np.savez_compressed(output/(name+'.npz'), **arrays,
                                knot_time_s=curve.knot_time_s, knot_qpos=curve.knot_qpos)
    passed = [n for n,a in report['actions'].items() if a['sampled_reference_pass']]
    report.update(status='refinement_complete_not_training_ready', sampled_reference_pass_names=passed,
                  robust_reference_pass_names=[n for n,a in report['actions'].items() if a.get('robustness_pass')],
                  attack_region_coverage={r.name: [n for n in passed if report['actions'][n].get('target_region') == r.name]
                                          for r in geometry.regions})
    report['robust_reference_template_coverage'] = {template: [n for n in passed
        if report['actions'][n].get('template') == template and report['actions'][n].get('robustness_pass')]
        for template in templates}
    report['robustness_pass'] = bool(all(report['robust_reference_template_coverage'].values()) and
        all(g['minimum_physical_joint_margin_rad'] >= ROBUST_MARGIN_RAD-1e-10
            for g in report['guards'].values() if g['static_reference_pass']))
    (output/'scene.xml').write_text(scene.spec.to_xml())
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=str(ROOT/'outputs/a1_triangle_screen_v2'))
    p.add_argument('--output', required=True)
    p.add_argument('--margin-rad', type=float, default=.01)
    p.add_argument('--max-nfev', type=int, default=110)
    p.add_argument('--start-guards', nargs='+', default=['center', 'right'])
    p.add_argument('--templates', nargs='+', choices=TEMPLATES, default=list(TEMPLATES))
    p.add_argument('--period', type=float, default=.01)
    a = p.parse_args()
    refine(a.source, a.output, margin_rad=a.margin_rad, max_nfev=a.max_nfev,
           start_guards=a.start_guards, templates=a.templates, period=a.period)


if __name__ == '__main__':
    main()
