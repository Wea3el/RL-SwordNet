"""Bounded, deterministic kinematic diagnostics, not physical execution.

The task has five independent dimensions: a grip position and a unit sword
direction. Sword roll is intentionally unconstrained. Every supplied frame is
retained, including unreachable targets and colliding interpolated paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def _site(model, name):
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if result < 0:
        raise ValueError(f"Required tool site is missing: {name}")
    return result


def tool_fk(model, data=None, qpos=None):
    """Return copied world grip, unit grip-to-tip direction, and tip position."""
    if data is None:
        data = mujoco.MjData(model)
    if qpos is not None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("qpos must be finite with shape (model.nq,)")
        data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    grip = data.site_xpos[_site(model, "shinai_grip")].copy()
    tip = data.site_xpos[_site(model, "shinai_tip")].copy()
    length = float(np.linalg.norm(tip - grip))
    if length < 1e-8:
        raise ValueError("Tool grip and tip must be distinct")
    return grip, (tip-grip)/length, tip


def _direction_error(current, target):
    """Spherical tangent residual, including a deterministic antipodal escape."""
    dot = float(np.clip(current @ target, -1., 1.))
    angle = float(np.arccos(dot))
    tangent = target-dot*current
    sine = float(np.linalg.norm(tangent))
    if sine > 1e-9:
        return tangent*(angle/sine), angle
    if dot >= 0:
        return np.zeros(3), angle
    basis = np.eye(3)[int(np.argmin(np.abs(current)))]
    tangent = basis-current*(basis @ current)
    return np.pi*tangent/np.linalg.norm(tangent), angle


def _jacobian(model, data, grip_site, tip_site, axis, length):
    grip_jac = np.zeros((3, model.nv))
    tip_jac = np.zeros_like(grip_jac)
    mujoco.mj_jacSite(model, data, grip_jac, None, grip_site)
    mujoco.mj_jacSite(model, data, tip_jac, None, tip_site)
    # Differentiate the normalized tip-minus-grip vector. This is equivalent
    # to -skew(axis) @ Jrot for a rigidly attached sword, without assuming site
    # local axis conventions or a particular fixed-body hierarchy.
    axis_jac = (np.eye(3)-np.outer(axis, axis)) @ (tip_jac-grip_jac)/length
    return grip_jac, axis_jac


def _check_model(model, bounds):
    if model.nq != model.nv or model.nq != model.njnt:
        raise ValueError("Matching currently requires scalar hinge/slide joints only")
    if np.any(~np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])):
        raise ValueError("Matching currently requires scalar hinge/slide joints only")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (model.nq, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Explicit finite ordered joint_ranges with shape (nq,2) are required")
    return bounds


def solve_pose(model, grip_target, axis_target, joint_ranges, *, initial_q=None,
               max_step_rad=.20, max_iterations=100, position_tolerance_m=.02,
               angle_tolerance_rad=np.deg2rad(5), orientation_weight_m=.30,
               tip_tolerance_m=None):
    """Solve one bounded pose using analytic site Jacobians and line search.

    max_step_rad bounds each optimization update (not physical speed). No
    angle wrapping occurs. Returned errors are recomputed at the returned q.
    """
    bounds = _check_model(model, joint_ranges)
    target = np.asarray(grip_target, dtype=float)
    axis_target = np.asarray(axis_target, dtype=float)
    if target.shape != (3,) or axis_target.shape != (3,) or not np.isfinite(np.r_[target, axis_target]).all() or not np.isclose(np.linalg.norm(axis_target), 1., atol=1e-6):
        raise ValueError("Targets must be finite 3-vectors and axis must be unit")
    if max_step_rad <= 0 or max_iterations < 1 or orientation_weight_m <= 0 or (tip_tolerance_m is not None and tip_tolerance_m <= 0):
        raise ValueError("Solver step, iterations, and orientation weight must be positive")
    q = np.asarray(model.qpos0 if initial_q is None else initial_q, dtype=float).copy()
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError("initial_q must be finite with shape (nq,)")
    q = np.clip(q, bounds[:, 0], bounds[:, 1])
    data = mujoco.MjData(model)
    grip_site, tip_site = _site(model, "shinai_grip"), _site(model, "shinai_tip")
    damping = 1e-4
    best_q, best_score = q.copy(), np.inf
    count = 0
    for count in range(1, max_iterations+1):
        grip, axis, tip = tool_fk(model, data, q)
        error, angle = _direction_error(axis, axis_target)
        position_error = float(np.linalg.norm(target-grip))
        length = float(np.linalg.norm(tip-grip))
        tip_error = float(np.linalg.norm(tip-(target+length*axis_target)))
        score = position_error**2+(orientation_weight_m*angle)**2
        if score < best_score:
            best_q, best_score = q.copy(), score
        if position_error <= position_tolerance_m and angle <= angle_tolerance_rad and (tip_tolerance_m is None or tip_error <= tip_tolerance_m):
            break
        jp, ja = _jacobian(model, data, grip_site, tip_site, axis, length)
        jac = np.vstack([jp, orientation_weight_m*ja])
        residual = np.r_[target-grip, orientation_weight_m*error]
        # Freeze joints whose unconstrained descent points outside the bounds,
        # rather than repeatedly clipping a large infeasible update.
        gradient = jac.T @ residual
        active = ((q <= bounds[:, 0]+1e-9) & (gradient < 0)) | ((q >= bounds[:, 1]-1e-9) & (gradient > 0))
        jac[:, active] = 0
        step = jac.T @ np.linalg.solve(jac@jac.T+damping*np.eye(6), residual)
        peak = np.max(np.abs(step))
        if peak > max_step_rad:
            step *= max_step_rad/peak
        improved = False
        for fraction in (1., .5, .25, .125, .0625, .03125):
            trial = np.clip(q+fraction*step, bounds[:, 0], bounds[:, 1])
            p, a, _ = tool_fk(model, data, trial)
            trial_angle = np.arccos(np.clip(a @ axis_target, -1., 1.))
            trial_score = np.sum((target-p)**2)+(orientation_weight_m*trial_angle)**2
            if trial_score < score-1e-14:
                q = trial
                damping = max(1e-6, damping*.5)
                improved = True
                if trial_score < best_score:
                    best_q, best_score = q.copy(), float(trial_score)
                break
        if not improved:
            damping = min(1e2, damping*10)
            if np.max(np.abs(step)) < 1e-8 or damping >= 1e2:
                break
    grip, axis, tip = tool_fk(model, data, best_q)
    position_error = float(np.linalg.norm(target-grip))
    angle = float(np.arccos(np.clip(axis @ axis_target, -1., 1.)))
    tip_error = float(np.linalg.norm(tip-(target+np.linalg.norm(tip-grip)*axis_target)))
    return {"qpos": best_q, "position_error_m": position_error,
            "angle_error_rad": angle, "tip_error_m": tip_error,
            "score": best_score, "iterations": count,
            "pose_success": position_error <= position_tolerance_m and angle <= angle_tolerance_rad and (tip_tolerance_m is None or tip_error <= tip_tolerance_m)}


def _path_contacts(model, data, start, end, step, threshold):
    subdivisions = max(1, int(np.ceil(np.max(np.abs(end-start))/step)))
    records = {}
    for alpha in np.linspace(0., 1., subdivisions+1):
        data.qpos[:] = start+alpha*(end-start)
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            if contact.dist >= -threshold:
                continue
            names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom_{g}" for g in (contact.geom1, contact.geom2))
            if names not in records or contact.dist < records[names]["distance_m"]:
                records[names] = {"geom1": names[0], "geom2": names[1],
                                  "distance_m": float(contact.dist), "path_fraction": float(alpha)}
    return list(records.values()), subdivisions+1


def screen_joint_path(model, start, end=None, *, interpolation_step_rad=.05,
                      collision_threshold_m=.001):
    """Audit static contacts and linearly interpolated joint poses; no dynamics."""
    start = np.asarray(start, dtype=float)
    end = start if end is None else np.asarray(end, dtype=float)
    if start.shape != (model.nq,) or end.shape != start.shape or not np.isfinite(np.r_[start, end]).all():
        raise ValueError("Path endpoints must be finite qpos vectors")
    if interpolation_step_rad <= 0 or collision_threshold_m <= 0:
        raise ValueError("Collision sampling step and threshold must be positive")
    return _path_contacts(model, mujoco.MjData(model), start, end,
                          interpolation_step_rad, collision_threshold_m)


@dataclass
class MatchResult:
    qpos: np.ndarray
    position_error_m: np.ndarray
    angle_error_rad: np.ndarray
    tip_error_m: np.ndarray
    joint_delta_rad: np.ndarray
    pose_success: np.ndarray
    collision_free: np.ndarray
    success: np.ndarray
    contacts: list
    iterations: np.ndarray
    collision_sample_count: np.ndarray

    def summary(self):
        # Row zero is solver seed -> initialization pose, NOT a sampled motion
        # interval. Never mix it into source-adjacent continuity statistics.
        adjacent_peak = float(np.abs(self.joint_delta_rad[1:]).max()) if len(self.qpos) > 1 else 0.
        return {"scope": "kinematic_qpos_snapshots_not_dynamics_or_defensive_effectiveness",
                "frames": len(self.qpos), "passed_frames": int(self.success.sum()),
                "pose_passed_frames": int(self.pose_success.sum()),
                "collision_free_frames": int(self.collision_free.sum()),
                "all_frames_passed": bool(self.success.all()),
                "max_position_error_m": float(self.position_error_m.max()),
                "max_angle_error_deg": float(np.rad2deg(self.angle_error_rad.max())),
                "max_tip_error_m": float(self.tip_error_m.max()),
                "initialization_delta_rad": self.joint_delta_rad[0].tolist(),
                "initialization_path_checked": False,
                "max_source_adjacent_step_rad": adjacent_peak,
                "max_joint_step_rad": adjacent_peak,
                "max_penetration_m": max((-c["distance_m"] for row in self.contacts for c in row), default=0.),
                "collision_path_samples": int(self.collision_sample_count.sum()),
                "failed_frame_indices": np.flatnonzero(~self.success).tolist()}


def match_sequence(model, grip_targets, axis_targets, joint_ranges, *, initial_q=None,
                   seed=0, max_step_rad=.20, max_iterations=100,
                   position_tolerance_m=.02, angle_tolerance_rad=np.deg2rad(5),
                   tip_tolerance_m=.05, collision_threshold_m=.001,
                   interpolation_step_rad=.05, first_frame_starts=12):
    """Match ALL frames; warm-start consecutive frames and audit joint paths.

    First-frame bounded deterministic multistart addresses local IK minima.
    Later frames use only the previous result, so failures cannot be hidden by
    jumping independently to an unrelated IK branch. The first frame is an
    initialization pose only; the path from initial_q to it is not validated.
    """
    bounds = _check_model(model, joint_ranges)
    positions, axes = np.asarray(grip_targets, dtype=float), np.asarray(axis_targets, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0 or axes.shape != positions.shape or not np.isfinite(positions).all() or not np.isfinite(axes).all() or not np.allclose(np.linalg.norm(axes, axis=1), 1., atol=1e-6):
        raise ValueError("Finite nonempty Nx3 positions and unit axes are required")
    if min(interpolation_step_rad, collision_threshold_m, tip_tolerance_m, position_tolerance_m, angle_tolerance_rad) <= 0 or first_frame_starts < 1:
        raise ValueError("Tolerances, interpolation step and start count must be positive")
    initial = np.asarray(model.qpos0 if initial_q is None else initial_q, dtype=float)
    if initial.shape != (model.nq,) or not np.isfinite(initial).all():
        raise ValueError("initial_q must be finite with shape (nq,)")
    initial = np.clip(initial, bounds[:, 0], bounds[:, 1])
    current = initial.copy()
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    grip, axis, tip = tool_fk(model, data, initial)
    tool_length = np.linalg.norm(tip-grip)
    qrows, perror, aerror, terror, deltas, poses, free, contacts, iterations, samples = ([] for _ in range(10))
    for index, (target, target_axis) in enumerate(zip(positions, axes)):
        seeds = [current]
        if index == 0:
            seeds += [.5*(bounds[:, 0]+bounds[:, 1])]
            seeds += [rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(max(0, first_frame_starts-2))]
            seeds = seeds[:first_frame_starts]
        best, best_contacts, best_samples, best_rank = None, None, None, None
        for start in seeds:
            result = solve_pose(model, target, target_axis, bounds, initial_q=start,
                                max_step_rad=max_step_rad, max_iterations=max_iterations,
                                position_tolerance_m=position_tolerance_m,
                                angle_tolerance_rad=angle_tolerance_rad,
                                tip_tolerance_m=tip_tolerance_m)
            # First-frame seed-to-pose path is not a physical transition and
            # should not disqualify a collision-free initialization pose.
            path_start = result["qpos"] if index == 0 else current
            found, count = _path_contacts(model, data, path_start, result["qpos"], interpolation_step_rad, collision_threshold_m)
            rank = (not result["pose_success"], bool(found), result["score"])
            if best_rank is None or rank < best_rank:
                best, best_contacts, best_samples, best_rank = result, found, count, rank
            if result["pose_success"] and not found:
                break
        grip, axis, tip = tool_fk(model, data, best["qpos"])
        tip_error = float(np.linalg.norm(tip-(target+tool_length*target_axis)))
        qrows.append(best["qpos"].copy())
        deltas.append(best["qpos"]-current)
        perror.append(best["position_error_m"])
        aerror.append(best["angle_error_rad"])
        terror.append(tip_error)
        poses.append(best["pose_success"] and tip_error <= tip_tolerance_m)
        free.append(not best_contacts)
        contacts.append(best_contacts)
        iterations.append(best["iterations"])
        samples.append(best_samples)
        current = best["qpos"].copy()
    poses, free = np.asarray(poses), np.asarray(free)
    return MatchResult(np.array(qrows), np.array(perror), np.array(aerror),
                       np.array(terror), np.array(deltas), poses, free, poses & free,
                       contacts, np.array(iterations), np.array(samples))
