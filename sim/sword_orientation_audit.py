"""Read-only, actor-frame sword orientation audit for candidate motion reports.

Blade axis means grip -> tip.  It is NOT the velocity of a cut, a target label,
or a camera direction.  These declared geometry checks grant no dynamic,
defence, training, or historical-fencing certificate.

Reports may declare ``orientation_requirements`` using the default schema below.
``guard_axes`` maps guard names to axis bounds. ``action_axes`` maps exact route
names to optional ``start``/``end`` bounds, ``anchors`` (phase, bounds), and
``intervals`` (phase=[begin,end], bounds). Supported bounds are x/y/z_min/max
and abs_x/y/z_max. Phases are normalized local action time, not skill labels.
``family_routes`` optionally maps a family to allowed [start_guard,end_guard]
pairs. Defaults deliberately do not impose a left/right shaft direction on all
named guards: the generator must explicitly declare that design choice.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .motion_spline import C2JointSpline


ATTACKS = ("cut_down_left_to_right", "cut_down_right_to_left",
           "cut_up_left_to_right", "cut_up_right_to_left", "cut_down_center",
           "thrust_center")
DEFENCES = ("parry_left", "parry_right", "parry_high")
DEFAULT_REQUIREMENTS = {
    "schema_version": 1,
    "lateral_threshold": .15,
    "neutral_guard": "center",
    "neutral_axis": {"x_min": .85, "abs_y_max": .2},
    "required_families": {"attack": list(ATTACKS), "defence": list(DEFENCES)},
    "required_coverage_groups": ["all", "attack", "defence"],
    "guard_axes": {}, "action_axes": {}, "family_routes": {},
}
_BOUND_NAMES = {axis + suffix for axis in "xyz" for suffix in ("_min", "_max")}
_BOUND_NAMES |= {"abs_" + axis + "_max" for axis in "xyz"}


def _number(value):
    return (not isinstance(value, (bool, str, bytes)) and np.isscalar(value)
            and np.isfinite(value))


def _axes(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite N x 3 direction array")
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("Zero vectors have no sword direction")
    return array / lengths[:, None]


def world_to_actor_axes(world_axes, actor_rotation):
    """Convert row-vector world axes using actor->world proper rotation."""
    rotation = np.asarray(actor_rotation, float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)):
        raise ValueError("Actor frame requires an orthonormal proper rotation")
    return _axes(world_axes) @ rotation


def axis_metrics(axes):
    axes = _axes(axes)
    heading = np.rad2deg(np.arctan2(axes[:, 1], axes[:, 0]))
    circular = np.sort(np.mod(heading, 360.))
    gaps = np.diff(np.r_[circular, circular[0] + 360.])
    pitch = np.rad2deg(np.arcsin(np.clip(axes[:, 2], -1., 1.)))
    forward_angle = np.rad2deg(np.arccos(np.clip(axes[:, 0], -1., 1.)))
    return {
        "sample_count": len(axes),
        "axis_component_min": axes.min(axis=0).tolist(),
        "axis_component_max": axes.max(axis=0).tolist(),
        "lateral_y_range": [float(axes[:, 1].min()), float(axes[:, 1].max())],
        "forward_x_range": [float(axes[:, 0].min()), float(axes[:, 0].max())],
        "heading_deg_range_wrapped": [float(heading.min()), float(heading.max())],
        "heading_circular_extent_deg": float(360. - gaps.max()),
        "pitch_deg_range": [float(pitch.min()), float(pitch.max())],
        "angle_from_forward_deg_range": [float(forward_angle.min()), float(forward_angle.max())],
        "maximum_pairwise_axis_angle_deg": float(np.rad2deg(np.arccos(
            np.clip((axes @ axes.T).min(), -1., 1.)))),
        "positive_y_fraction": float(np.mean(axes[:, 1] > 1e-9)),
        "negative_y_fraction": float(np.mean(axes[:, 1] < -1e-9)),
        "all_strictly_positive_y": bool(np.all(axes[:, 1] > 1e-9)),
        "all_strictly_negative_y": bool(np.all(axes[:, 1] < -1e-9)),
    }


def _validate_bounds(bounds):
    if not isinstance(bounds, dict) or not bounds or set(bounds) - _BOUND_NAMES:
        raise ValueError("Axis bounds must be a nonempty mapping of declared component bounds")
    if any(not _number(value) or not -1 <= value <= 1 for value in bounds.values()):
        raise ValueError("Axis bounds must be finite numbers in [-1,1]")
    for axis in "xyz":
        if bounds.get(axis + "_min", -1.) > bounds.get(axis + "_max", 1.):
            raise ValueError("Contradictory axis bounds")
        if bounds.get("abs_" + axis + "_max", 0.) < 0:
            raise ValueError("Absolute axis maxima cannot be negative")


def _phase(value):
    return _number(value) and 0 <= value <= 1


def requirements_config(declared=None):
    result = copy.deepcopy(DEFAULT_REQUIREMENTS)
    if declared is not None:
        if not isinstance(declared, dict) or set(declared) - set(result):
            raise ValueError("Unknown orientation requirement fields")
        result.update(copy.deepcopy(declared))
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("Orientation requirement schema 1 required")
    if not _number(result["lateral_threshold"]) or not 0 < result["lateral_threshold"] <= 1:
        raise ValueError("Positive lateral coverage threshold required")
    if not isinstance(result["neutral_guard"], str) or not result["neutral_guard"]:
        raise ValueError("An explicit neutral guard name is required")
    _validate_bounds(result["neutral_axis"])
    families = result["required_families"]
    if (not isinstance(families, dict) or set(families) != {"attack", "defence"}
            or any(not isinstance(names, list) or not names
                   or any(not isinstance(name, str) or not name for name in names)
                   for names in families.values())):
        raise ValueError("Declare nonempty attack and defence family lists")
    flat = sum(families.values(), [])
    if len(flat) != len(set(flat)):
        raise ValueError("Required family names must be unique")
    groups = result["required_coverage_groups"]
    if (not isinstance(groups, list) or not groups or len(groups) != len(set(groups))
            or any(group not in ("all", "attack", "defence") for group in groups)):
        raise ValueError("Explicit distinct coverage groups required")
    for key in ("guard_axes", "action_axes", "family_routes"):
        if not isinstance(result[key], dict) or any(not isinstance(name, str) or not name for name in result[key]):
            raise ValueError(f"{key} requires named declarations")
    for bounds in result["guard_axes"].values():
        _validate_bounds(bounds)
    for spec in result["action_axes"].values():
        if not isinstance(spec, dict) or set(spec) - {"start", "end", "anchors", "intervals"}:
            raise ValueError("Unknown per-action orientation rule")
        for endpoint in ("start", "end"):
            if endpoint in spec:
                _validate_bounds(spec[endpoint])
        for key in ("anchors", "intervals"):
            rows = spec.get(key, [])
            if not isinstance(rows, list):
                raise ValueError("Anchors and intervals must be lists")
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"phase", "bounds"}:
                    raise ValueError("Each anchor/interval needs phase and bounds")
                phase = row["phase"]
                if key == "anchors" and not _phase(phase):
                    raise ValueError("Anchor phase must lie in [0,1]")
                if key == "intervals" and (not isinstance(phase, list) or len(phase) != 2
                        or not all(_phase(p) for p in phase) or phase[0] > phase[1]):
                    raise ValueError("Ordered interval phases in [0,1] required")
                _validate_bounds(row["bounds"])
    for family, routes in result["family_routes"].items():
        if family not in flat or not isinstance(routes, list) or not routes or any(
                not isinstance(route, list) or len(route) != 2
                or any(not isinstance(name, str) or not name for name in route) for route in routes):
            raise ValueError("Allowed family routes need declared [start_guard,end_guard] pairs")
    return result


def _bounds_check(axes, bounds):
    axes = _axes(axes)
    checks = {}
    for name, limit in bounds.items():
        if name.startswith("abs_"):
            value = float(np.abs(axes[:, "xyz".index(name[4])]).max())
            passed = value <= limit + 1e-9
        else:
            values = axes[:, "xyz".index(name[0])]
            value = float(values.min() if name.endswith("_min") else values.max())
            passed = value >= limit - 1e-9 if name.endswith("_min") else value <= limit + 1e-9
        checks[name] = {"actual": value, "limit": limit, "pass": bool(passed)}
    return {"pass": all(check["pass"] for check in checks.values()), "bounds": checks}


def audit_axes(actions, guards, requirements=None):
    """Evaluate DECLARED geometry rules on actor-local sampled directions.

    ``actions[name]`` needs axes, family, start_guard, end_guard and an explicit
    boolean source_screen_pass. Only passed source references count toward bank
    coverage; rejected or absent families cannot be disguised by another route.
    This helper does not compute or certify the source screen itself.
    """
    config = requirements_config(requirements)
    if not isinstance(actions, dict) or not isinstance(guards, dict):
        raise ValueError("Actions and guards must be mappings")
    required = sum(config["required_families"].values(), [])
    role = {family: kind for kind, names in config["required_families"].items() for family in names}
    prepared = {}
    for name, action in actions.items():
        if not isinstance(name, str) or not isinstance(action, dict):
            raise ValueError("Named action samples required")
        if type(action.get("source_screen_pass")) is not bool or not isinstance(action.get("family"), str):
            raise ValueError("Each action needs a family and strict boolean source_screen_pass")
        prepared[name] = dict(action, axes=_axes(action["axes"]))
    guard_axes = {name: _axes([axis]) for name, axis in guards.items()}
    present = {action["family"] for action in prepared.values()}
    eligible = {name: action for name, action in prepared.items()
                if action["source_screen_pass"] and action["family"] in role}
    eligible_families = {action["family"] for action in eligible.values()}
    missing = sorted(set(required) - present)
    unscreened = sorted((set(required) & present) - eligible_families)
    coverage = {}
    for group in ("all", "attack", "defence"):
        names = [name for name, action in eligible.items()
                 if group == "all" or role[action["family"]] == group]
        axes = np.concatenate([eligible[name]["axes"] for name in names]) if names else None
        threshold = config["lateral_threshold"]
        coverage[group] = {
            "eligible_action_names": names, "threshold": threshold,
            "positive_side_covered": bool(axes is not None and np.any(axes[:, 1] >= threshold)),
            "negative_side_covered": bool(axes is not None and np.any(axes[:, 1] <= -threshold)),
        }
        coverage[group]["pass"] = coverage[group]["positive_side_covered"] and coverage[group]["negative_side_covered"]
    checks, missing_declarations = {}, []
    guard_rules = {config["neutral_guard"]: config["neutral_axis"]}
    # Both neutral and explicit guard rules are evaluated, never silently overwrite.
    for name, bounds in guard_rules.items():
        key = "neutral:" + name
        if name not in guard_axes:
            missing_declarations.append(key)
        else:
            checks[key] = _bounds_check(guard_axes[name], bounds)
    for name, bounds in config["guard_axes"].items():
        key = "guard:" + name
        if name not in guard_axes:
            missing_declarations.append(key)
        else:
            checks[key] = _bounds_check(guard_axes[name], bounds)
    for name, spec in config["action_axes"].items():
        if name not in prepared:
            missing_declarations.append("action:" + name)
            continue
        axes = prepared[name]["axes"]
        phases = np.linspace(0., 1., len(axes))
        for endpoint, index in (("start", 0), ("end", -1)):
            if endpoint in spec:
                checks[f"action:{name}:{endpoint}"] = _bounds_check(axes[[index]], spec[endpoint])
        for i, row in enumerate(spec.get("anchors", [])):
            index = int(np.argmin(abs(phases - row["phase"])))
            check = _bounds_check(axes[[index]], row["bounds"])
            check.update(requested_phase=row["phase"], sampled_phase=float(phases[index]))
            checks[f"action:{name}:anchor:{i}"] = check
        for i, row in enumerate(spec.get("intervals", [])):
            selected = (phases >= row["phase"][0] - 1e-12) & (phases <= row["phase"][1] + 1e-12)
            if not selected.any():
                missing_declarations.append(f"action:{name}:interval:{i}:no_sample")
            else:
                checks[f"action:{name}:interval:{i}"] = _bounds_check(axes[selected], row["bounds"])
    for name, action in prepared.items():
        allowed = config["family_routes"].get(action["family"])
        if allowed is not None:
            route = [action.get("start_guard"), action.get("end_guard")]
            checks["route:" + name] = {"actual": route, "allowed": allowed, "pass": route in allowed}
    failures = [name for name, check in checks.items() if not check["pass"]]
    failures += ["coverage:" + group for group in config["required_coverage_groups"] if not coverage[group]["pass"]]
    incomplete = bool(missing or unscreened or missing_declarations)
    return {
        "requirements": config, "coverage": coverage, "checks": checks,
        "missing_families": missing, "unscreened_or_failed_families": unscreened,
        "missing_declared_evidence": missing_declarations, "failed_checks": failures,
        "status": "incomplete" if incomplete else "fail" if failures else "pass",
        "orientation_geometry_candidate_pass": not incomplete and not failures,
    }


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON number: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _qpos(value):
    q = np.asarray(value, float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected finite ordered J0..J6 pose")
    return q


def audit_report(source, *, requirements=None, samples_per_action=301, scene_path=None):
    """Audit frozen reference poses, not integrated native trajectories."""
    import mujoco
    source = Path(source).resolve()
    if type(samples_per_action) is not int or not 3 <= samples_per_action <= 2001:
        raise ValueError("samples_per_action must be an integer from 3 to 2001")
    raw = source.read_bytes()
    report = _json(raw)
    if (not isinstance(report, dict) or not isinstance(report.get("actions"), dict)
            or not isinstance(report.get("guards"), dict) or not report["guards"]):
        raise ValueError("Geometry report requires actions and a nonempty guards mapping")
    config = requirements_config(requirements if requirements is not None else report.get("orientation_requirements"))
    scene = Path(scene_path).resolve() if scene_path is not None else source.parent / "scene.xml"
    paths = (source, scene, Path(__file__).resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    # The report is read exactly once for analysis; concurrent edits invalidate it.
    hashes[str(source)] = hashlib.sha256(raw).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    slots = {}
    for slot in ("a1", "a2"):
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{slot}_J{i}") for i in range(7)]
        sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{slot}_shinai_{kind}") for kind in ("grip", "tip")]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, slot + "_fixed_base")
        if min(joints + sites + [body]) < 0 or any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Scene needs both named fixed-base, seven-hinge sword arms")
        slots[slot] = {"qids": model.jnt_qposadr[joints], "sites": sites, "body": body}

    def fk(qs):
        local, world, grips = {s: [] for s in slots}, {s: [] for s in slots}, {s: [] for s in slots}
        for q in qs:
            for slot in slots.values():
                data.qpos[slot["qids"]] = q
            data.qvel[:] = 0.
            mujoco.mj_forward(model, data)
            for name, slot in slots.items():
                grip, tip = data.site_xpos[slot["sites"]]
                axis = _axes([tip - grip])[0]
                rotation = data.xmat[slot["body"]].reshape(3, 3)
                world[name].append(axis)
                local[name].append(world_to_actor_axes([axis], rotation)[0])
                grips[name].append(grip.copy())
        return ({s: np.asarray(v) for s, v in local.items()},
                {s: np.asarray(v) for s, v in world.items()},
                {s: np.asarray(v) for s, v in grips.items()})

    guard_axes, guard_rows, mirror_errors = {}, {}, []
    for name, row in report["guards"].items():
        if not isinstance(name, str) or not isinstance(row, dict) or "qpos_rad" not in row:
            raise ValueError("Every guard requires a named qpos_rad")
        local, world, _ = fk([_qpos(row["qpos_rad"])])
        error = float(np.linalg.norm(local["a1"] - local["a2"], axis=1).max())
        mirror_errors.append(error)
        guard_axes[name] = local["a1"][0]
        guard_rows[name] = {"axis_actor_local": local["a1"][0].tolist(),
                            "axis_world_by_slot": {s: world[s][0].tolist() for s in slots},
                            "mirror_slot_local_axis_error": error, **axis_metrics(local["a1"])}
    actions, action_rows = {}, {}
    for name, row in report["actions"].items():
        if not isinstance(name, str) or not isinstance(row, dict):
            raise ValueError("Every action must be named")
        try:
            curve = C2JointSpline(row["knot_time_s"], row["knot_qpos"])
            if curve.knot_qpos.ndim != 2 or curve.knot_qpos.shape[1] != 7:
                raise ValueError("Actions must have seven joints")
            for endpoint, index in (("start_guard", 0), ("end_guard", -1)):
                if row[endpoint] not in report["guards"] or not np.array_equal(
                        curve.knot_qpos[index], _qpos(report["guards"][row[endpoint]]["qpos_rad"])):
                    raise ValueError("Action endpoints must equal declared guard poses")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed action: {name}") from exc
        times = np.linspace(curve.knot_time_s[0], curve.knot_time_s[-1], samples_per_action)
        local, world, grips = fk(curve.sample(times))
        error = float(np.linalg.norm(local["a1"] - local["a2"], axis=1).max())
        mirror_errors.append(error)
        screen = row.get("candidate_pass", row.get("sampled_reference_pass", False))
        if type(screen) is not bool:
            raise ValueError("Source screen flags must be booleans")
        family = row.get("family", row.get("policy_family_id", name))
        actions[name] = {"axes": local["a1"], "family": family, "source_screen_pass": screen,
                         "start_guard": row["start_guard"], "end_guard": row["end_guard"]}
        action_rows[name] = {
            "family": family, "source_screen_pass": screen,
            "start_guard": row["start_guard"], "end_guard": row["end_guard"],
            "time_s": times.tolist(), "axis_actor_local": local["a1"].tolist(),
            "mirror_slot_local_axis_error": error,
            "grip_endpoint_displacement_actor_local_m":
                ((grips["a1"][-1] - grips["a1"][0]) @ data.xmat[slots["a1"]["body"]].reshape(3, 3)).tolist(),
            **axis_metrics(local["a1"]),
        }
    gate = audit_axes(actions, guard_axes, config)
    unchanged = all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in paths)
    mirror_error = max(mirror_errors, default=float("inf"))
    mirror_pass = bool(mirror_error <= 1e-9)
    return {
        "schema_version": 1, "audit_kind": "sampled_reference_sword_orientation",
        "coordinate_frame": "+X actor forward, +Y actor left, +Z up; grip-to-tip unit axis",
        "axis_is_not_cut_velocity_or_target_direction": True,
        "source_report": str(source), "source_sha256": hashes, "source_unchanged": unchanged,
        "requirements_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "guards": guard_rows, "actions": action_rows, "geometry_gate": gate,
        "mirrored_slot_invariance": {"max_actor_local_axis_error": mirror_error,
                                     "tolerance": 1e-9, "pass": mirror_pass},
        "orientation_geometry_candidate_pass": bool(gate["orientation_geometry_candidate_pass"] and mirror_pass and unchanged),
        "native_dynamics_tested": False, "defence_effectiveness_validated": False,
        "admitted_to_training": False,
        "limitations": ["Finite reference samples are not continuous orientation certificates.",
                        "No contact, torque, velocity, scoring, or moving-opponent validation is performed.",
                        "Candidate coverage is not policy usage or a guarantee of natural fencing.",
                        "Scene/report hashes do not independently freeze externally referenced mesh files."],
    }


def _actual_frame_indices(times, samples):
    """Nearest recorded frames on a bounded physical-time grid; no interpolation."""
    times = np.asarray(times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] < 0. or not np.all(np.diff(times) > 0.)):
        raise ValueError("Strictly increasing, nonnegative finite physical timestamps required")
    if type(samples) is not int or not 2 <= samples <= 301:
        raise ValueError("Native readback samples must be an integer from 2 to 301")
    if len(times) <= samples:
        return np.arange(len(times))
    desired = np.linspace(times[0], times[-1], samples)
    right = np.searchsorted(times, desired, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(times[left] - desired) <= abs(times[right] - desired), left, right)
    # Desired endpoints are exact recorded endpoints. Duplicates caused by a
    # nonuniform log are removed, never filled with interpolated poses.
    return np.unique(np.r_[0, indices, len(times) - 1]).astype(int)


def audit_native_trace(trial_dir, *, samples=301):
    """Read back actual native qpos in both actors' own fixed-base frames.

    This is a bounded diagnostic of one recorded trial, not a bank geometry
    gate, a new physics run, or proof that the source trial succeeded. Commands
    and desired-reference arrays in the archive are deliberately never used.
    """
    import mujoco
    trial = Path(trial_dir).resolve()
    scene, trace = trial / "scene.xml", trial / "trace.npz"
    paths = [scene, trace, Path(__file__).resolve()]
    summary_path = trial / "summary.json"
    if summary_path.is_file():
        paths.append(summary_path)
    raw = {path: path.read_bytes() for path in paths}
    hashes = {str(path): hashlib.sha256(value).hexdigest() for path, value in raw.items()}
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    try:
        with np.load(io.BytesIO(raw[trace]), allow_pickle=False) as archive:
            times = np.asarray(archive["time_s"], dtype=float).copy()
            qpos = np.asarray(archive["qpos"], dtype=float).copy()
    except (KeyError, TypeError) as exc:
        raise ValueError("Native trace needs actual qpos and time_s arrays") from exc
    indices = _actual_frame_indices(times, samples)
    if qpos.shape != (len(times), model.nq) or not np.isfinite(qpos).all():
        raise ValueError("Finite actual qpos must match each physical timestamp and model.nq")
    slots = {}
    for slot in ("a1", "a2"):
        sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{slot}_shinai_{kind}")
                 for kind in ("grip", "tip")]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, slot + "_fixed_base")
        if min(sites + [body]) < 0:
            raise ValueError("Native scene needs both named fixed-base sword actors")
        slots[slot] = {"sites": sites, "body": body, "local": [], "world": []}
    for index in indices:
        data.qpos[:] = qpos[index]  # Exact recorded state; never target_qpos.
        data.qvel[:] = 0.
        data.time = float(times[index])
        mujoco.mj_forward(model, data)  # Offline FK, not a new simulation step.
        for slot in slots.values():
            grip, tip = data.site_xpos[slot["sites"]]
            world = _axes([tip - grip])[0]
            rotation = data.xmat[slot["body"]].reshape(3, 3)
            slot["world"].append(world)
            slot["local"].append(world_to_actor_axes([world], rotation)[0])
    actors = {name: {
        "axis_actor_local": np.asarray(slot["local"]).tolist(),
        "axis_world": np.asarray(slot["world"]).tolist(),
        **axis_metrics(slot["local"]),
    } for name, slot in slots.items()}
    summary = _json(raw[summary_path]) if summary_path in raw else {}
    if not isinstance(summary, dict):
        raise ValueError("Optional native summary must be a JSON object")
    unchanged = all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)]
                    for path in paths)
    return {
        "schema_version": 1, "audit_kind": "recorded_native_sword_orientation",
        "trial_directory": str(trial), "source_sha256": hashes, "source_unchanged": unchanged,
        "state_source": "trace.npz:qpos", "commands_or_reference_used": False,
        "interpolation_used": False, "new_physics_steps_run": 0,
        "coordinate_frame": "+X actor forward, +Y actor left, +Z up; grip-to-tip unit axis",
        "axis_is_not_cut_velocity_or_target_direction": True,
        "recorded_frame_count": len(times), "selected_frame_count": len(indices),
        "maximum_selected_frames": samples, "sample_indices": indices.tolist(),
        "time_s": times[indices].tolist(), "sampled_actual_qpos": qpos[indices].tolist(),
        "selection_rule": "nearest actual frames on physical-time grid; unique indices; both endpoints retained",
        "actors": actors,
        "source_trial_status": summary.get("status"),
        "source_trial_native_execution_pass": summary.get("native_execution_pass"),
        "source_trial_source_unchanged": summary.get("source_unchanged"),
        "is_training_gate": False, "admitted_to_training": False,
        "defence_effectiveness_validated": False,
        "limitations": ["Readback geometry only: source failure/status is not overwritten.",
                        "At most 301 recorded frames are examined; extrema between selected frames may be missed.",
                        "Different actors execute different actual states; no equality of their axes is expected.",
                        "No new contact, torque, scoring, defence, or complete-bank coverage certificate is produced.",
                        "Scene/trace hashes do not independently freeze externally referenced mesh files."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="Candidate geometry report")
    mode.add_argument("--native", type=Path, help="Recorded trial directory containing scene.xml and trace.npz")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory; never overwritten")
    parser.add_argument("--requirements", type=Path, help="Explicit JSON overrides report declaration/defaults")
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior orientation evidence; choose a fresh output directory")
    if args.native and args.requirements:
        raise ValueError("Native readback does not apply a complete-bank requirement gate")
    req_bytes = args.requirements.read_bytes() if args.requirements else None
    if args.native:
        result = audit_native_trace(args.native, samples=args.samples)
    else:
        result = audit_report(args.source, requirements=_json(req_bytes) if req_bytes is not None else None,
                              samples_per_action=args.samples)
    if args.requirements:
        result["requirements_file"] = str(args.requirements.resolve())
        result["requirements_file_sha256"] = hashlib.sha256(req_bytes).hexdigest()
        result["requirements_file_unchanged"] = args.requirements.read_bytes() == req_bytes
        result["orientation_geometry_candidate_pass"] &= result["requirements_file_unchanged"]
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / ("native_orientation_audit.json" if args.native else "orientation_audit.json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path.resolve()), "source_unchanged": result["source_unchanged"],
                      "status": result["geometry_gate"]["status"] if "geometry_gate" in result
                      else "actual_native_readback_not_training_gate",
                      "orientation_geometry_candidate_pass": result.get("orientation_geometry_candidate_pass")}))


if __name__ == "__main__":
    main()
