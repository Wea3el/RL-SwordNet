"""Read-only ideal free-space torque demand for candidate joint references.

Static gravity and a speed bound do not test acceleration or braking. This
screen computes M(q) qdd + bias(q, qd) - passive(q, qd), without using contact
or joint-limit assistance. It never integrates or changes the controller.
Sampled ideal tracking feasibility is NOT actual tracking or defence evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


def joint_torque_bounds(model):
    """Physical scalar actuator-force bounds mapped to independent joints."""
    if model.nq != 7 or model.nv != 7 or model.nu != 7:
        raise ValueError('This screen requires the independent seven-joint arm')
    bounds = np.zeros((7, 2))
    used = set()
    for aid in range(model.nu):
        if model.actuator_trntype[aid] != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError('Only direct joint actuators are supported')
        jid = int(model.actuator_trnid[aid, 0])
        if model.jnt_type[jid] != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Only hinge coordinates are supported')
        dof = int(model.jnt_dofadr[jid])
        if dof in used or not model.actuator_forcelimited[aid]:
            raise ValueError('One force-limited actuator per joint is required')
        used.add(dof)
        gear = float(model.actuator_gear[aid, 0])
        if not np.isfinite(gear) or gear == 0:
            raise ValueError('Finite nonzero transmission gear is required')
        bounds[dof] = np.sort(model.actuator_forcerange[aid] * gear)
    if used != set(range(7)) or np.any(bounds[:, 0] >= 0) or np.any(bounds[:, 1] <= 0):
        raise ValueError('Both signs of finite joint torque must be available')
    return bounds


def free_space_demand(model, data, q, qvel, qacc):
    """Return ideal actuator demand with no contact/limit force contribution."""
    for value in (q, qvel, qacc):
        if np.asarray(value).shape != (7,) or not np.isfinite(value).all():
            raise ValueError('Finite seven-joint q, qvel and qacc are required')
    data.qpos[:] = q
    data.qvel[:] = qvel
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    inertia = mass @ qacc
    return inertia + data.qfrc_bias - data.qfrc_passive, inertia


def screen_curve(model, curve, period_s=.02):
    if not isinstance(curve, C2JointSpline):
        raise TypeError('A validated C2 reference is required')
    if curve.knot_time_s[0] != 0.:
        raise ValueError('Use an explicit zero-based reference clock')
    if isinstance(period_s, bool) or not np.isfinite(period_s) or period_s <= 0:
        raise ValueError('A positive finite sample period is required')
    bounds = joint_torque_bounds(model)
    times = np.unique(np.r_[np.arange(0., curve.duration_s, period_s),
                            curve.knot_time_s, curve.duration_s])
    q = curve.sample(times)
    velocity, acceleration = curve.derivatives(times)
    if np.any(q < model.jnt_range[:, 0]) or np.any(q > model.jnt_range[:, 1]):
        raise ValueError('Reference exceeds the model joint ranges; no clipping')
    data = mujoco.MjData(model)
    demands, inertias = [], []
    for qi, vi, ai in zip(q, velocity, acceleration):
        demand, inertia = free_space_demand(model, data, qi, vi, ai)
        demands.append(demand.copy()); inertias.append(inertia.copy())
    demands, inertias = np.asarray(demands), np.asarray(inertias)
    ratios = np.where(demands >= 0, demands / bounds[:, 1], demands / bounds[:, 0])
    worst = np.unravel_index(np.argmax(ratios), ratios.shape)
    return {
        'schema_version': 1,
        'kind': 'sampled_ideal_free_space_torque_demand',
        'sample_count': len(times), 'sample_period_s': period_s,
        'sample_time_s': times.tolist(),
        'required_joint_torque_nm': demands.tolist(),
        'inertial_joint_torque_nm': inertias.tolist(),
        'physical_joint_torque_bounds_nm': bounds.tolist(),
        'peak_utilization_by_joint': np.max(ratios, axis=0).tolist(),
        'peak_required_abs_torque_nm': np.max(np.abs(demands), axis=0).tolist(),
        'worst_joint': f'J{worst[1]}', 'worst_time_s': float(times[worst[0]]),
        'worst_utilization': float(ratios[worst]),
        'ideal_demand_within_caps_at_samples': bool(np.all(ratios <= 1.)),
        'reference_extrema': curve.analytic_extrema(),
        'scope': 'Planning diagnostic only; does not integrate, enforce tracking, or validate a defence.',
        'contact_and_joint_limit_assistance_used': False,
        'limitations': [
            'Finite sampling can miss extrema of configuration-dependent torque.',
            'Uses physical actuator-force caps, not the tighter force achievable at a given bounded position command.',
            'Does not include impacts, feedback lag, moving opponents, controller saturation transients or motor calibration.',
            'Exceeding caps diagnoses infeasible exact tracking, not necessarily invalid approximate motion; native tests decide.',
            'Passing is not a collision, tracking, naturalness, defence, or training certificate.'
        ],
        'admitted_to_training': False,
    }


def run(source, action_name, output, period_s=.02):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Keep prior diagnostics; use a fresh output')
    source_bytes = source.read_bytes()
    report = json.loads(source_bytes)
    action = report['actions'][action_name]
    curve = C2JointSpline(action['knot_time_s'], action['knot_qpos'])
    assembly = build_limited_floor()
    result = screen_curve(assembly.model, curve, period_s)
    dependencies = [ROOT/name for name in (
        'sim/a1_dynamics_screen.py', 'sim/limited_model.py', 'sim/floor.py',
        'sim/assembly.py', 'sim/model.py', 'sim/shinai.py', 'sim/motion_spline.py',
        'sim/joint_constraints.py', 'config/assembly.json',
        'config/joint_constraints_observed.json')]
    result.update(action=action_name, source_report=str(source),
                  source_report_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_source_sha256=assembly.source_hashes,
                  dependency_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in dependencies},
                  source_unchanged=source.read_bytes() == source_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(result), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('action', 'worst_joint', 'worst_time_s',
          'worst_utilization', 'peak_utilization_by_joint')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--action', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--period', type=float, default=.02)
    args = parser.parse_args()
    run(args.source, args.action, args.output, args.period)
