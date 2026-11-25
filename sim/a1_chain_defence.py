"""Bounded physical defence search against the frozen negative-side down cut.

Future attack samples are used ONLY to author candidate geometry offline. The
paired judge and its unchanged controller/rules determine actual success. A
touch, reference intersection, or stopped video is never a defence certificate.
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

from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .duel_scene import classify_contact
from .motion_spline import C2JointSpline
from .multiguard_defence_probe import run as run_pair
from .multiguard_geometry import GripAxisSolver, defence_reference_audit, unit
from .triangle_task import TriangleGeometry
from plan1_skills.matching import tool_fk


def write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2, ensure_ascii=False) + "\n")


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh bounded-search output")
    source = ROOT / "outputs/orientation_v2_geometry/trim07/report.json"
    native = ROOT / "outputs/orientation_v2_native/down_negative_07/native"
    paths = [source, native / "trace.npz", native / "scene.xml", native / "summary.json",
             Path(__file__).resolve(), ROOT / "config/joint_constraints_observed.json"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original = json.loads(source.read_text())
    cons = load_joint_constraints()
    bounds = model_joint_ranges_rad(cons) + [.03, -.03]
    model = build_limited_floor(cons).model
    solver = GripAxisSolver(model, bounds)
    cap = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    geometry = TriangleGeometry.from_mapping({k: original["scoring"][k] for k in ("base_distance", "regions")})
    center = np.array(original["guards"]["center"]["qpos_rad"])
    attack = original["actions"]["cut_down_left_to_right"]
    attack_entry = np.array(attack["knot_qpos"][0])
    scene = build_limited_duel(geometry.base_distance, initial_qpos=center, constraints=cons)
    attack_model = mujoco.MjModel.from_xml_path(str(native / "scene.xml"))
    attack_data = mujoco.MjData(attack_model)
    grip_id, tip_id = [attack_model.site("a1_shinai_" + kind).id for kind in ("grip", "tip")]
    with np.load(native / "trace.npz", allow_pickle=False) as archive:
        native_times, native_q = archive["time_s"].copy(), archive["qpos"].copy()
    report = {
        "schema_version": 2, "status": "bounded_defence_geometry_search",
        "scoring": original["scoring"], "source_sha256": hashes,
        "guards": copy.deepcopy(original["guards"]),
        "actions": {"cut_down_left_to_right": copy.deepcopy(attack)}, "trials": [],
        "candidate_only": True, "native_dynamics_tested": False, "admitted_to_training": False,
        "limitations": ["Offline attack timing is for authoring only, not a policy observation.",
                        "Initial B remains exploratory and is not a final canonical neutral.",
                        "Only a matched full-window probe can qualify an exact defence and timing.",
                        "No chain transition, broader defence, or training readiness is certified."],
    }
    output.mkdir(parents=True, exist_ok=False)
    write(output / "report.json", report)
    (output / "scene.xml").write_text(scene.spec.to_xml())
    trial_id = 0
    for contact_time, attack_material in ((3.3, .6), (3.5, .75)):
        index = int(np.argmin(abs(native_times - contact_time)))
        attack_data.qpos[:] = native_q[index]
        mujoco.mj_forward(attack_model, attack_data)
        attack_grip, tip = attack_data.site_xpos[[grip_id, tip_id]]
        target_world = attack_grip + attack_material * unit(tip - attack_grip)
        target = (target_world - [geometry.base_distance, 0., 0.]) * [-1., -1., 1.]
        for axis_values in ([.55, -.75, .35], [.65, -.70, .25], [.40, -.85, .35], [.55, -.55, .63]):
            axis = unit(axis_values)
            for defence_material in (.45, .60):
                p = target - defence_material * axis
                block, fit = solver.solve(p, axis, center, regularization=.0015, iterations=160)
                solver.data.qpos[:] = block; solver.data.qvel[:] = 0.
                mujoco.mj_forward(model, solver.data)
                block_ratio = float(np.max(abs(solver.data.qfrc_bias) / cap))
                trial = {"id": trial_id, "contact_time_s": contact_time,
                         "native_sample_time_s": float(native_times[index]),
                         "attack_material_m": attack_material,
                         "defence_material_m": defence_material,
                         "requested_intercept_actor_local_m": target.tolist(),
                         "requested_grip_m": p.tolist(), "requested_axis": axis.tolist(),
                         "block_qpos_rad": block.tolist(), "block_fit": fit,
                         "block_static_torque_ratio": block_ratio, "variants": []}
                trial_id += 1
                if fit["fit_pass"] and block_ratio < 1.:
                    for variant, shift in enumerate(([0., -.10, .12], [.04, -.05, .20])):
                        sweep, sweep_fit = solver.solve(p + shift, axis, block, regularization=.0015, iterations=160)
                        solver.data.qpos[:] = sweep; solver.data.qvel[:] = 0.
                        mujoco.mj_forward(model, solver.data)
                        sweep_ratio = float(np.max(abs(solver.data.qfrc_bias) / cap))
                        name = f"parry_negative_{trial['id']:02d}_{variant}"
                        times = [0., 2., 2.4, 3.4, 6.4, 8.4]
                        curve = C2JointSpline(times, [center, block, block, sweep, sweep, center])
                        peak = float(max(curve.analytic_extrema()["peak_speed_rad_s"]))
                        idle, _ = defence_reference_audit(scene, geometry, curve, attack_entry, period=.04)
                        accepted = bool(sweep_fit["fit_pass"] and sweep_ratio < 1.
                                        and peak <= 2. and idle["sampled_reference_pass"])
                        trial["variants"].append({"name": name, "sweep_shift_m": shift,
                            "sweep_qpos_rad": sweep.tolist(), "sweep_fit": sweep_fit,
                            "sweep_static_torque_ratio": sweep_ratio, "max_reference_speed_rad_s": peak,
                            "idle_opponent_geometry": idle, "candidate_pass": accepted})
                        if accepted:
                            report["actions"][name] = {
                                "kind": "defence", "start_guard": "center", "end_guard": "center",
                                "target_region": "left", "candidate_pass": True,
                                "sampled_reference_pass": True, "admitted_to_training": False,
                                "native_dynamics_tested": False, "authoring_trial_id": trial["id"],
                                "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist(),
                                "description": "preposition, hold, sweep up/right, hold through attack, recover",
                            }
                report["trials"].append(trial)
                write(output / "report.json", report)
                print("defence_geometry", trial["id"], "fit", fit["fit_pass"],
                      "torque", round(block_ratio, 3), "eligible", [v["name"] for v in trial["variants"] if v["candidate_pass"]], flush=True)
    report["source_unchanged"] = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in paths)
    report["status"] = "bounded_geometry_complete_not_defence_validated"
    write(output / "report.json", report)
    return report


def rebase(source, output):
    """Recheck old controls and near-guard variants against a NEW attack."""
    source, output = Path(source).resolve(), Path(output).resolve()
    previous_path = ROOT / "outputs/a1_chain_defence/geometry01/report.json"
    if output.exists():
        raise FileExistsError("Keep old defence outcomes immutable")
    attack_bytes, old_bytes = source.read_bytes(), previous_path.read_bytes()
    attack_report, previous = json.loads(attack_bytes), json.loads(old_bytes)
    center = np.asarray(attack_report["guards"]["center"]["qpos_rad"])
    if (not np.array_equal(center, previous["guards"]["center"]["qpos_rad"])
            or attack_report["scoring"] != previous["scoring"]):
        raise ValueError("Rebasing requires the exact same B and scoring; no silent replacement")
    constraints = load_joint_constraints()
    model = build_limited_floor(constraints).model
    solver = GripAxisSolver(model, model_joint_ranges_rad(constraints) + [.03, -.03])
    cap = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    geometry = TriangleGeometry.from_mapping({k: attack_report["scoring"][k] for k in ("base_distance", "regions")})
    scene = build_limited_duel(geometry.base_distance, initial_qpos=center, constraints=constraints)
    attack = attack_report["actions"]["cut_down_left_to_right"]
    attack_curve = C2JointSpline(attack["knot_time_s"], attack["knot_qpos"])
    attack_entry = attack_curve.knot_qpos[0]
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "builder_snapshot.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    report = {
        "schema_version": 2, "scoring": attack_report["scoring"],
        "guards": copy.deepcopy(attack_report["guards"]),
        "actions": {"cut_down_left_to_right": copy.deepcopy(attack)},
        "source_sha256": {str(source): hashlib.sha256(attack_bytes).hexdigest(),
                          str(previous_path): hashlib.sha256(old_bytes).hexdigest(),
                          str(snapshot): hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        "status": "rebased_defence_candidates_not_native_verified", "trials": [],
        "candidate_only": True, "admitted_to_training": False, "native_dynamics_tested": False,
        "limitations": ["Old defence failures remain failures; NEW attack requires new paired evidence.",
                        "Near-guard contact is a leverage hypothesis, not a defence certificate.",
                        "Future reference geometry is used for offline authoring only.",
                        "Every defence ends at exact B via an explicit recovery; B remains exploratory."],
    }
    for name in ("parry_negative_05_0", "parry_negative_09_1"):
        row = copy.deepcopy(previous["actions"][name])
        curve = C2JointSpline(row["knot_time_s"], row["knot_qpos"])
        idle, _ = defence_reference_audit(scene, geometry, curve, attack_entry, period=.04)
        row.update(candidate_pass=idle["sampled_reference_pass"], sampled_reference_pass=idle["sampled_reference_pass"],
                   new_attack_idle_screen=idle, inherited_physical_certificate=False)
        report["actions"][name] = row

    def torque_ratio(q):
        solver.data.qpos[:] = q; solver.data.qvel[:] = 0.
        mujoco.mj_forward(model, solver.data)
        return float(np.max(abs(solver.data.qfrc_bias) / cap))

    candidate_id = 0
    for contact_time in (3.3, 3.5):
        attack_q = attack_curve.sample(contact_time - .75)
        grip, shaft, _ = tool_fk(model, qpos=attack_q)
        target = (grip + .75 * shaft - [geometry.base_distance, 0., 0.]) * [-1., -1., 1.]
        for axis_values in ([.4, -.85, .35], [.55, -.55, -.63]):
            axis = unit(axis_values)
            for along in (.18, .28):
                name = f"parry_near_guard_{candidate_id:02d}"
                candidate_id += 1
                p = target - along * axis
                q, fit = solver.solve(p, axis, center, regularization=.0015, iterations=180)
                pre, prefit = solver.solve(p + [0., 0., -.12], axis, q,
                                          regularization=.0015, iterations=180)
                sweep, sweepfit = solver.solve(p + [0., -.06, .16], axis, q,
                                              regularization=.0015, iterations=180)
                ratios = [torque_ratio(v) for v in (pre, q, sweep)]
                # Arrive BELOW the incoming blade, then rise through the interception.
                # The final hold covers the complete attack before explicit recovery.
                times = [0., 1.7, 2.55, 3.55, 6.4, 8.4]
                poses = [center, pre, q, sweep, sweep, center]
                curve = C2JointSpline(times, poses)
                speed = float(max(curve.analytic_extrema()["peak_speed_rad_s"]))
                idle, _ = defence_reference_audit(scene, geometry, curve, attack_entry, period=.04)
                passed = bool(all(f["fit_pass"] for f in (fit, prefit, sweepfit))
                              and max(ratios) < 1. and speed <= 2. and idle["sampled_reference_pass"])
                trial = {"name": name, "contact_time_s": contact_time,
                         "defending_material_m": along, "attacking_material_m": .75,
                         "target_defender_local_m": target.tolist(), "axis": axis.tolist(),
                         "fit": fit, "pre_fit": prefit, "sweep_fit": sweepfit,
                         "static_torque_ratios": ratios, "peak_reference_speed_rad_s": speed,
                         "idle_screen": idle, "candidate_pass": passed,
                         "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist()}
                report["trials"].append(trial)
                if passed:
                    report["actions"][name] = {
                        "kind": "defence", "start_guard": "center", "end_guard": "center",
                        "target_region": "left", "candidate_pass": True, "sampled_reference_pass": True,
                        "native_dynamics_tested": False, "admitted_to_training": False,
                        "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist(),
                        "description": "near-guard leverage candidate; approach from below, rise and hold, explicit recovery",
                    }
                # A separately named, less extended candidate: the upward
                # approach itself supplies the attempted deflection. Never
                # relabel the failed additional sweep as successful IK.
                rise_times = [0., 1.7, 2.55, 6.4, 8.4]
                rise = C2JointSpline(rise_times, [center, pre, q, q, center])
                rise_speed = float(max(rise.analytic_extrema()["peak_speed_rad_s"]))
                rise_idle, _ = defence_reference_audit(scene, geometry, rise, attack_entry, period=.04)
                rise_pass = bool(fit["fit_pass"] and prefit["fit_pass"] and max(ratios[:2]) < 1.
                                 and rise_speed <= 2. and rise_idle["sampled_reference_pass"])
                trial["rise_hold_candidate"] = {"candidate_pass": rise_pass,
                    "peak_reference_speed_rad_s": rise_speed, "idle_screen": rise_idle,
                    "knot_time_s": rise_times, "knot_qpos": rise.knot_qpos.tolist()}
                if rise_pass:
                    report["actions"][name + "_rise_hold"] = {
                        "kind": "defence", "start_guard": "center", "end_guard": "center",
                        "target_region": "left", "candidate_pass": True, "sampled_reference_pass": True,
                        "native_dynamics_tested": False, "admitted_to_training": False,
                        "knot_time_s": rise_times, "knot_qpos": rise.knot_qpos.tolist(),
                        "description": "near-guard approach from below and hold; NO rejected extra-sweep segment",
                    }
                print(name, "fit", [f["fit_pass"] for f in (fit, prefit, sweepfit)],
                      "torques", [round(v, 3) for v in ratios], "speed", round(speed, 3),
                      "pass", passed, "rise_hold", rise_pass, flush=True)
                write(output / "report.json", report)
    (output / "scene.xml").write_text(scene.spec.to_xml())
    report["source_unchanged"] = (source.read_bytes() == attack_bytes and previous_path.read_bytes() == old_bytes)
    write(output / "report.json", report)
    return report


def point_candidates(source, output, entry_guard="center"):
    """Free-axis, interior-joint material-point candidates; geometry only.

    A full-path first-contact-side screen is separate from fitting the chosen
    material point. It can reject an apparently upward endpoint movement whose
    earlier rotating shaft would actually push the attacker down.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve preceding failed contact geometry")
    raw = source.read_bytes(); old = json.loads(raw)
    cons = load_joint_constraints(); physical = model_joint_ranges_rad(cons)
    bounds = physical + [.14, -.14]
    model = build_limited_floor(cons).model; data = mujoco.MjData(model)
    cap = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    center = np.asarray(old["guards"][entry_guard]["qpos_rad"])
    if center[2] < np.pi:
        bounds[2, 1] = min(bounds[2, 1], np.pi)
    else:
        bounds[2, 0] = max(bounds[2, 0], np.pi)
    geom = TriangleGeometry.from_mapping({k: old["scoring"][k] for k in ("base_distance", "regions")})
    scene = build_limited_duel(geom.base_distance, initial_qpos=center, constraints=cons)
    attack = old["actions"]["cut_down_left_to_right"]
    attack_curve = C2JointSpline(attack["knot_time_s"], attack["knot_qpos"])
    self_clearance_pairs = [(gid,
                             model.geom(f"arm_proxy_{j}").id)
                            for gid in blade_collision_geom_ids(model, '') for j in range(7)]

    def self_clearances():
        return np.array([mujoco.mj_geomDistance(model, data, a, b, .05, None)
                         for a, b in self_clearance_pairs])

    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "builder_snapshot.py"; snapshot.write_bytes(Path(__file__).read_bytes())
    report = {"schema_version": 2, "scoring": old["scoring"], "guards": copy.deepcopy(old["guards"]),
        "actions": {"cut_down_left_to_right": copy.deepcopy(attack)}, "trials": [],
        "source_sha256": {str(source): hashlib.sha256(raw).hexdigest(),
                          str(snapshot): hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        "status": "interior_material_point_geometry_only", "candidate_only": True,
        "admitted_to_training": False, "native_dynamics_tested": False,
        "all_interior_poses_joint_margin_rad": .14,
        "requested_blade_to_arm_clearance_m": .010,
        "entry_exit_guard": entry_guard,
        "initial_final_guard_note": "Exact source guard; no implicit repositioning. Interior IK margin does not overwrite source endpoint.",
        "limitations": ["Reference first contact is only a screening hypothesis, not actual force evidence.",
                        "No candidate obtains defence credit until the unchanged full-window native judge passes.",
                        "A new robust attack source requires new exact held-entry baseline and pair tests."]}

    def solve_point(target, seed_q, along_seed, incoming, approach_velocity):
        target, incoming, approach_velocity = map(np.asarray, (target, incoming, approach_velocity))
        weights = np.array([2., 1., 2., 1., 1., 1., 1.])
        def residual(x):
            q, along = x[:7], x[7]
            grip, axis, _ = tool_fk(model, data, q)
            return np.r_[15. * (grip + along * axis - target),
                .8 * max(0., abs(axis @ incoming) - .55),
                .6 * max(0., abs(axis @ approach_velocity) - .50),
                8. * np.maximum(0., .010 - self_clearances()),
                .015 * weights * (q - seed_q), .05 * (along - along_seed)]
        start = np.r_[np.maximum(bounds[:, 0] + 1e-8, np.minimum(bounds[:, 1] - 1e-8, seed_q)), along_seed]
        fit = least_squares(residual, start, bounds=(np.r_[bounds[:, 0], .12], np.r_[bounds[:, 1], .72]),
                            max_nfev=180, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q, along = fit.x[:7], float(fit.x[7])
        grip, axis, _ = tool_fk(model, data, q)
        error = float(np.linalg.norm(grip + along * axis - target))
        data.qvel[:] = 0.; mujoco.mj_forward(model, data)
        ratio = float(np.max(abs(data.qfrc_bias) / cap))
        minimum_clearance = float(self_clearances().min())
        cross = float(abs(axis @ incoming)); slide = float(abs(axis @ approach_velocity))
        return q, {"along_m": along, "material_point_error_m": error,
            "actual_axis": axis.tolist(), "actual_grip_m": grip.tolist(),
            "incoming_axis_abs_dot": cross, "incoming_velocity_abs_dot": slide,
            "minimum_joint_margin_rad": float(np.minimum(q - physical[:, 0], physical[:, 1] - q).min()),
            "minimum_blade_to_arm_clearance_m": minimum_clearance,
            "static_torque_ratio": ratio, "nfev": int(fit.nfev),
            "fit_pass": bool(error <= .012 and cross <= .60 and slide <= .55 and ratio < 1.
                             and minimum_clearance >= .005)}

    def first_reference_contact(curve):
        found = None
        for absolute in np.arange(.75, 4.05, .02):
            scene.data.qpos[scene.agents["a1"]["qpos_ids"]] = attack_curve.sample(absolute - .75)
            scene.data.qpos[scene.agents["a2"]["qpos_ids"]] = curve.sample(absolute - .75)
            mujoco.mj_forward(scene.model, scene.data)
            for contact in scene.data.contact:
                if contact.dist >= 0 or classify_contact(scene.model, contact, rules_version="whole_blade_first_touch_v2") != "cross_blade_blade":
                    continue
                normal = contact.frame[:3].copy()
                if scene.model.geom(contact.geom1).name.startswith("a2_"):
                    normal *= -1.
                found = {"time_s": float(absolute), "normal_attacker_to_defender_world": normal.tolist(),
                         "depth_m": float(-contact.dist), "candidate_upward_normal": bool(normal[2] < -.10)}
                return found
        return found

    trial_id = 0
    for crossing_time in (3.15, 3.45):
        local_t = crossing_time - .75
        g, incoming_world, _ = tool_fk(model, data, attack_curve.sample(local_t))
        target = (g + .72 * incoming_world - [geom.base_distance, 0., 0.]) * [-1., -1., 1.]
        before_grip, before_axis, _ = tool_fk(model, qpos=attack_curve.sample(local_t - .01))
        after_grip, after_axis, _ = tool_fk(model, qpos=attack_curve.sample(local_t + .01))
        velocity = unit(((after_grip + .72 * after_axis) - (before_grip + .72 * before_axis)) * [-1., -1., 1.])
        incoming = incoming_world * [-1., -1., 1.]
        for preferred_along in (.25, .50):
            name = f"parry_point_interior_{trial_id:02d}"; trial_id += 1
            block, bf = solve_point(target + [0., 0., -.012], center, preferred_along, incoming, velocity)
            pre, pf = solve_point(target + [0., .02, -.15], block, bf["along_m"], incoming, velocity)
            sweep, sf = solve_point(target + [0., -.04, .12], block, bf["along_m"], incoming, velocity)
            cg, ca, _ = tool_fk(model, qpos=center)
            retreat, rf = solve_point(cg + .35 * ca + [-.18, .05, -.14], pre, .35, incoming, velocity)
            times = [0., 1.0, 1.9, 2.6, 3.5, 6.4, 8.4]
            curve = C2JointSpline(times, [center, retreat, pre, block, sweep, sweep, center])
            peak = float(max(curve.analytic_extrema()["peak_speed_rad_s"]))
            idle, _ = defence_reference_audit(scene, geom, curve, attack_curve.knot_qpos[0], period=.04)
            first = first_reference_contact(curve)
            fitpass = all(row["fit_pass"] for row in (bf, pf, sf, rf))
            passed = bool(fitpass and peak <= 2. and idle["sampled_reference_pass"]
                          and first is not None and first["candidate_upward_normal"])
            row = {"name": name, "crossing_time_s": crossing_time,
                "preferred_material_m": preferred_along, "target_defender_local_m": target.tolist(),
                "block": bf, "pre": pf, "sweep": sf, "retreat": rf, "fit_pass": fitpass,
                "idle_screen": idle, "peak_reference_speed_rad_s": peak,
                "first_reference_weapon_contact": first, "candidate_pass": passed,
                "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist()}
            report["trials"].append(row)
            if passed:
                report["actions"][name] = {"kind": "defence", "start_guard": entry_guard, "end_guard": entry_guard,
                    "target_region": "left", "candidate_pass": True, "sampled_reference_pass": True,
                    "admitted_to_training": False, "native_dynamics_tested": False,
                    "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist()}
            # A separately authored direct approach is allowed only if its own
            # full-shaft first-contact screen passes. It is not a skipped frame
            # of the rejected retreat trajectory, and has a new time law.
            direct_times = [0., 2., 2.6, 3.5, 6.4, 8.4]
            direct = C2JointSpline(direct_times, [center, pre, block, sweep, sweep, center])
            direct_peak = float(max(direct.analytic_extrema()["peak_speed_rad_s"]))
            direct_idle, _ = defence_reference_audit(scene, geom, direct, attack_curve.knot_qpos[0], period=.04)
            direct_first = first_reference_contact(direct)
            direct_pass = bool(all(item["fit_pass"] for item in (bf, pf, sf))
                and direct_peak <= 2. and direct_idle["sampled_reference_pass"]
                and direct_first is not None and direct_first["candidate_upward_normal"])
            row["direct_approach"] = {"candidate_pass": direct_pass,
                "peak_reference_speed_rad_s": direct_peak, "idle_screen": direct_idle,
                "first_reference_weapon_contact": direct_first,
                "knot_time_s": direct_times, "knot_qpos": direct.knot_qpos.tolist()}
            if direct_pass:
                report["actions"][name + "_direct"] = {
                    "kind": "defence", "start_guard": entry_guard, "end_guard": entry_guard,
                    "target_region": "left", "candidate_pass": True, "sampled_reference_pass": True,
                    "admitted_to_training": False, "native_dynamics_tested": False,
                    "knot_time_s": direct_times, "knot_qpos": direct.knot_qpos.tolist()}
            print(name, "fits", [row["fit_pass"] for row in (bf, pf, sf, rf)],
                  "speed", round(peak, 3), "firstcontact", first, "pass", passed,
                  "direct_pass", direct_pass, flush=True)
            write(output / "report.json", report)
    (output / "scene.xml").write_text(scene.spec.to_xml())
    report["source_unchanged"] = source.read_bytes() == raw
    write(output / "report.json", report)
    return report


def check_static_path(source, output):
    """Freeze a new report after sampling gravity demand on complete paths."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh path-gravity audit destination")
    raw = source.read_bytes(); report = json.loads(raw)
    model = build_limited_floor(load_joint_constraints()).model
    data = mujoco.MjData(model)
    cap = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "builder_snapshot.py"; snapshot.write_bytes(Path(__file__).read_bytes())
    report["parent_source_sha256"] = {str(source): hashlib.sha256(raw).hexdigest(),
                                      str(snapshot): hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    for name, action in report["actions"].items():
        if action.get("kind") != "defence":
            continue
        curve = C2JointSpline(action["knot_time_s"], action["knot_qpos"])
        times = np.linspace(0., curve.duration_s, int(np.ceil(curve.duration_s / .025)) + 1)
        ratios = []
        for q in curve.sample(times):
            data.qpos[:] = q; data.qvel[:] = 0.
            mujoco.mj_forward(model, data)
            ratios.append(abs(data.qfrc_bias) / cap)
        ratios = np.array(ratios)
        ti, joint = np.unravel_index(ratios.argmax(), ratios.shape)
        passed = bool(ratios.max() < 1.)
        action["path_static_gravity"] = {
            "sample_count": len(times), "max_ratio": float(ratios.max()),
            "worst_joint": "J" + str(joint), "worst_time_s": float(times[ti]),
            "max_ratio_per_joint": ratios.max(axis=0).tolist(), "pass": passed,
            "meaning": "Static zero-velocity gravity screen only; no dynamic/contact certificate."}
        action["candidate_pass"] = action.get("candidate_pass") is True and passed
        action["sampled_reference_pass"] = action["candidate_pass"]
        print(name, "whole_path_static_ratio", float(ratios.max()), "pass", action["candidate_pass"], flush=True)
    report["path_static_source_unchanged"] = source.read_bytes() == raw
    report["status"] = "point_geometry_and_path_gravity_checked_not_native_defence"
    (output / "scene.xml").write_bytes((source.parent / "scene.xml").read_bytes())
    write(output / "report.json", report)
    return report


def slow_approach(source, output, action_name, factor=1.2):
    """New clock for one existing path; never transfers its native outcome."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous timing experiments")
    raw = source.read_bytes(); original = json.loads(raw)
    old_action = original["actions"][action_name]
    old_times = np.asarray(old_action["knot_time_s"], float)
    if len(old_times) != 7 or not 1. < factor <= 1.5:
        raise ValueError("This bounded variant requires a seven-knot path and factor in (1,1.5]")
    times = old_times.copy()
    times[:5] *= factor
    times[5] = max(6.8, times[4] + .5)
    times[6] = times[5] + 2.
    curve = C2JointSpline(times, old_action["knot_qpos"])
    name = action_name + "_slow_" + str(factor).replace(".", "p")
    report = {k: copy.deepcopy(v) for k, v in original.items() if k not in ("actions", "trials")}
    report["actions"] = {"cut_down_left_to_right": copy.deepcopy(original["actions"]["cut_down_left_to_right"])}
    geometry = TriangleGeometry.from_mapping({k: report["scoring"][k] for k in ("base_distance", "regions")})
    entry = report["guards"][old_action["start_guard"]]["qpos_rad"]
    attack = report["actions"]["cut_down_left_to_right"]
    ac = C2JointSpline(attack["knot_time_s"], attack["knot_qpos"])
    scene = build_limited_duel(geometry.base_distance, initial_qpos=entry)
    idle, _ = defence_reference_audit(scene, geometry, curve, ac.knot_qpos[0], period=.04)
    first = None
    for absolute in np.arange(.75, 4.4, .02):
        scene.data.qpos[scene.agents["a1"]["qpos_ids"]] = ac.sample(absolute - .75)
        scene.data.qpos[scene.agents["a2"]["qpos_ids"]] = curve.sample(absolute - .75)
        mujoco.mj_forward(scene.model, scene.data)
        for contact in scene.data.contact:
            if contact.dist >= 0 or classify_contact(scene.model, contact, rules_version="whole_blade_first_touch_v2") != "cross_blade_blade":
                continue
            normal = contact.frame[:3].copy()
            if scene.model.geom(contact.geom1).name.startswith("a2_"):
                normal *= -1.
            first = {"time_s": float(absolute), "normal_attacker_to_defender_world": normal.tolist(),
                     "candidate_upward_normal": bool(normal[2] < -.1)}
            break
        if first is not None:
            break
    peak = float(max(curve.analytic_extrema()["peak_speed_rad_s"]))
    passed = bool(idle["sampled_reference_pass"] and peak <= 2. and first is not None
                  and first["candidate_upward_normal"])
    action = copy.deepcopy(old_action)
    action.update(knot_time_s=times.tolist(), candidate_pass=passed, sampled_reference_pass=passed,
        native_dynamics_tested=False, admitted_to_training=False,
        timing_revision={"factor_first_four_intervals": factor, "attack_start_s": .75,
                         "defence_start_s": .75, "source_action": action_name,
                         "first_reference_contact": first, "idle_screen": idle,
                         "peak_reference_speed_rad_s": peak, "native_certificate_inherited": False})
    report["actions"][name] = action
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "builder_snapshot.py"; snapshot.write_bytes(Path(__file__).read_bytes())
    report["timing_parent_source_sha256"] = {str(source): hashlib.sha256(raw).hexdigest(),
        str(snapshot): hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    report["status"] = "slower_final_approach_candidate_requires_new_native_pair"
    report["timing_source_unchanged"] = source.read_bytes() == raw
    (output / "scene.xml").write_text(scene.spec.to_xml())
    write(output / "report.json", report)
    print(name, "pass", passed, "first", first, "peak", peak, flush=True)
    return report


def lateral_candidates(source, output, entry_guard="left_guard_positive", lifted=False):
    """Bounded new contact-load hypothesis: mostly backward/lateral, not UP.

    The heuristic only authors geometry. It cannot admit a skill or override
    the unchanged actual-contact, joint-limit, and complete-window judges.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Lateral hypotheses require a fresh destination")
    raw = source.read_bytes(); old = json.loads(raw)
    cons = load_joint_constraints(); physical = model_joint_ranges_rad(cons)
    bounds = physical + [.14, -.14]
    # A previous raised-transit candidate overshot J0's upper stop before
    # any contact. Re-solve (never clip) with room for real tracking error.
    bounds[0, 1] = min(bounds[0, 1], np.deg2rad(330.))
    # Keep the wrist on the entry's positive J6 branch. Without this declared
    # corridor, independently fitted points flip -82 -> +82 degrees and are
    # not an executable connected parry at the available reaction time.
    bounds[6, 0] = max(bounds[6, 0], 0.)
    model = build_limited_floor(cons).model; data = mujoco.MjData(model)
    cap = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    entry = np.asarray(old["guards"][entry_guard]["qpos_rad"])
    bounds[2, 0] = max(bounds[2, 0], np.pi)
    geometry = TriangleGeometry.from_mapping({k: old["scoring"][k] for k in ("base_distance", "regions")})
    scene = build_limited_duel(geometry.base_distance, initial_qpos=entry, constraints=cons)
    attack = old["actions"]["cut_down_left_to_right"]
    ac = C2JointSpline(attack["knot_time_s"], attack["knot_qpos"])
    transform = np.array([-1., -1., 1.])
    pairs = [(gid, model.geom(f"arm_proxy_{j}").id)
             for gid in blade_collision_geom_ids(model, '') for j in range(7)]
    def clearance():
        return np.array([mujoco.mj_geomDistance(model, data, a, b, .05, None) for a, b in pairs])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "builder_snapshot.py"; snapshot.write_bytes(Path(__file__).read_bytes())
    report = {"schema_version": 2, "scoring": old["scoring"], "guards": copy.deepcopy(old["guards"]),
        "actions": {"cut_down_left_to_right": copy.deepcopy(attack)}, "trials": [],
        "source_sha256": {str(source): hashlib.sha256(raw).hexdigest(),
                          str(snapshot): hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        "candidate_only": True, "admitted_to_training": False, "native_dynamics_tested": False,
        "entry_exit_guard": entry_guard, "status": "lateral_contact_load_geometry_only",
        "interior_J0_upper_design_bound_deg": 330.,
        "interior_J6_lower_design_bound_deg": 0.,
        "design_change": "Near-vertical blade attempts a backward/lateral attacker reaction instead of mandatory upward force.",
        "limitations": ["A favorable reference normal or J5 load is not a physical defence result.",
                        "The unchanged full-window native paired judge is still required.",
                        "No midmotion policy reaction or broad timing corridor is certified."]}

    def fit_point(target, seed, along_seed, preferred_axis, normal_local):
        def residual(x):
            q, along = x[:7], x[7]
            grip, axis, _ = tool_fk(model, data, q)
            return np.r_[15. * (grip + along * axis - target),
                .6 * (axis - preferred_axis), .8 * (axis @ normal_local),
                8. * np.maximum(0., .010 - clearance()),
                .015 * np.array([2., 1., 2., 1., 1., 1., 1.]) * (q - seed),
                .05 * (along - along_seed)]
        x0 = np.r_[np.clip(seed, bounds[:, 0] + 1e-8, bounds[:, 1] - 1e-8), along_seed]
        fit = least_squares(residual, x0, bounds=(np.r_[bounds[:, 0], .12], np.r_[bounds[:, 1], .72]),
                            max_nfev=220, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q, along = fit.x[:7], float(fit.x[7]); grip, axis, _ = tool_fk(model, data, q)
        data.qvel[:] = 0.; mujoco.mj_forward(model, data)
        error = float(np.linalg.norm(grip + along * axis - target))
        ratio = float(np.max(abs(data.qfrc_bias) / cap)); clear = float(clearance().min())
        alignment = float(axis @ preferred_axis)
        return q, {"along_m": along, "actual_axis": axis.tolist(), "actual_grip_m": grip.tolist(),
            "material_point_error_m": error, "preferred_axis_dot": alignment,
            "static_torque_ratio": ratio, "minimum_blade_to_arm_clearance_m": clear,
            "minimum_joint_margin_rad": float(np.minimum(q - physical[:, 0], physical[:, 1] - q).min()),
            "fit_pass": bool(error <= .014 and alignment >= .82 and ratio < 1. and clear >= .005)}

    def first_contact(curve, desired_normal):
        for absolute in np.arange(.75, 4.4, .02):
            aq = ac.sample(absolute - .75)
            scene.data.qpos[scene.agents["a1"]["qpos_ids"]] = aq
            scene.data.qpos[scene.agents["a2"]["qpos_ids"]] = curve.sample(absolute - .75)
            mujoco.mj_forward(scene.model, scene.data)
            for contact in scene.data.contact:
                if contact.dist >= 0 or classify_contact(scene.model, contact, rules_version="whole_blade_first_touch_v2") != "cross_blade_blade":
                    continue
                normal = contact.frame[:3].copy()
                if scene.model.geom(contact.geom1).name.startswith("a2_"):
                    normal *= -1.
                point = contact.pos.copy()
                jp = np.zeros((3, scene.model.nv))
                a_geom = contact.geom1 if scene.model.geom(contact.geom1).name.startswith("a1_") else contact.geom2
                mujoco.mj_jac(scene.model, scene.data, jp, None, point, int(scene.model.geom_bodyid[a_geom]))
                j5 = int(scene.agents["a1"]["dof_ids"][5])
                load = float(jp[:, j5] @ (-normal))
                return {"time_s": float(absolute), "normal_attacker_to_defender_world": normal.tolist(),
                        "desired_normal_dot": float(normal @ desired_normal),
                        "attacker_J5_torque_per_unit_normal_force_m": load,
                        "candidate_load_direction_pass": bool(normal @ desired_normal >= .5 and load > .10)}
        return None

    for i, (crossing, axis_sign) in enumerate(((3.2, 1.), (3.2, -1.), (3.5, 1.), (3.5, -1.))):
        g, incoming, _ = tool_fk(model, data, ac.sample(crossing - .75))
        target = (g + .72 * incoming - [geometry.base_distance, 0., 0.]) * transform
        n = np.array([1., 1., -.3]); n = unit(n - incoming * (n @ incoming))
        preferred_axis = unit(np.cross(incoming, n)) * transform * axis_sign
        normal_local = n * transform
        block, bf = fit_point(target + .012 * normal_local, entry, .30, preferred_axis, normal_local)
        pre, pf = fit_point(target + .10 * normal_local, block, bf["along_m"], preferred_axis, normal_local)
        sweep, sf = fit_point(target - .10 * normal_local, block, bf["along_m"], preferred_axis, normal_local)
        extra_fits = []
        if lifted:
            high, hf = fit_point(target + .13 * normal_local + [0., 0., .40], pre,
                                 pf["along_m"], preferred_axis, normal_local)
            # The raised transit is explicit and returns via the same raised
            # waypoint. It is not an invisible reset or conditional teleport.
            times = [0., 1.68, 2.415, 3.15, 3.885, 6.8, 7.85, 9.53]
            curve = C2JointSpline(times, [entry, high, pre, block, sweep, sweep, high, entry])
            extra_fits = [hf]
        else:
            times = [0., 2.2, 2.85, 3.55, 6.8, 8.8]
            curve = C2JointSpline(times, [entry, pre, block, sweep, sweep, entry])
        peak = float(max(curve.analytic_extrema()["peak_speed_rad_s"]))
        idle, _ = defence_reference_audit(scene, geometry, curve, ac.knot_qpos[0], period=.035)
        first = first_contact(curve, n)
        accepted = bool(all(f["fit_pass"] for f in (pf,bf,sf,*extra_fits)) and peak<=2. and idle["sampled_reference_pass"]
                        and first is not None and first["candidate_load_direction_pass"])
        name = f"parry_lateral_load_{i:02d}" + ("_raised" if lifted else "")
        trial = {"name": name, "crossing_time_s": crossing, "preferred_axis": preferred_axis.tolist(),
            "desired_normal_attacker_to_defender_world": n.tolist(), "pre": pf, "block": bf, "sweep": sf,
            "raised_transit_fits": extra_fits,
            "first_reference_contact": first, "idle_screen": idle, "peak_reference_speed_rad_s": peak,
            "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist(), "candidate_pass": accepted}
        report["trials"].append(trial)
        if accepted:
            report["actions"][name] = {"kind": "defence", "start_guard": entry_guard, "end_guard": entry_guard,
                "target_region": "left", "candidate_pass": True, "sampled_reference_pass": True,
                "admitted_to_training": False, "native_dynamics_tested": False,
                "knot_time_s": times, "knot_qpos": curve.knot_qpos.tolist(),
                "contact_design": trial["first_reference_contact"]}
        print(name, "fits", [f["fit_pass"] for f in (pf,bf,sf)], "axis", preferred_axis,
              "peak", peak, "first", first, "idle", idle["sampled_reference_pass"], "pass", accepted, flush=True)
        write(output / "report.json", report)
    (output / "scene.xml").write_text(scene.spec.to_xml())
    report["source_unchanged"] = source.read_bytes() == raw
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--rebase", type=Path, help="NEW attack report; recheck old controls and generate near-guard variants")
    parser.add_argument("--point-source", type=Path, help="NEW material-point interior-margin geometry search")
    parser.add_argument("--lateral-source", type=Path, help="NEW bounded backward/lateral contact-load hypothesis")
    parser.add_argument("--lateral-lift", action="store_true", help="Explicit raised transit for lateral candidate")
    parser.add_argument("--static-source", type=Path, help="Freeze a new report after full-path gravity screening")
    parser.add_argument("--entry-guard", default="center", help="Exact source guard for material-point defence entry AND exit")
    parser.add_argument("--retime-source", type=Path, help="Freeze a slower final-approach candidate from an existing seven-knot defence")
    parser.add_argument("--retime-factor", type=float, default=1.2)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--defence")
    parser.add_argument("--offset", type=float, default=0.)
    parser.add_argument("--attacker", choices=("a1", "a2"), default="a1")
    args = parser.parse_args()
    if args.build is not None:
        build(args.build)
    elif args.rebase is not None and args.output is not None:
        rebase(args.rebase, args.output)
    elif args.point_source is not None and args.output is not None:
        point_candidates(args.point_source, args.output, args.entry_guard)
    elif args.lateral_source is not None and args.output is not None:
        lateral_candidates(args.lateral_source, args.output, args.entry_guard, args.lateral_lift)
    elif args.static_source is not None and args.output is not None:
        check_static_path(args.static_source, args.output)
    elif args.retime_source is not None and args.output is not None and args.defence:
        slow_approach(args.retime_source, args.output, args.defence, args.retime_factor)
    elif args.source is not None and args.output is not None and args.defence:
        run_pair(args.source, args.output, attack_name="cut_down_left_to_right",
                 defence_name=args.defence, defence_offset_s=args.offset, attacker=args.attacker)
    else:
        parser.error("Use --build DIR or --source REPORT --output DIR --defence NAME")


if __name__ == "__main__":
    main()
