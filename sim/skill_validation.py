"""Native shared-skill validation, not RL or validated hardware fencing.

Both arms use EXACTLY the same finite Guard/Attack/Parry paths and clocks.
Scripted paired controls diagnose causality; continuous policy trials diagnose
whether the game supports useful choices. Every failed run remains evidence.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .duel_scene import build_duel_scene, duel_contact_records
from .manual_duel import position_control
from .shared_control import EntryScorer, SkillController, ScriptedPolicy
from .shared_skills import build_shared_library


DEFAULT_CONFIG = {
    "schema_version": 1, "base_distance_m": 1.75,
    "physics_timestep_s": .002, "control_period_s": .01,
    "decision_period_s": .10, "scoring_tip_radius_m": .01,
    "illegal_contact_force_threshold_n": .5,
    "penetration_limit_m": .005, "hypothetical_limit_force_ceiling_nm": .1,
    "diagnostic_actual_speed_rad_s": 2., "diagnostic_tracking_error_rad": .5,
    "minimum_blade_force_n": .1, "minimum_blade_impulse_ns": .02,
    "initial_hold_s": .75, "final_hold_s": .75,
    "profile": "pose_and_game_feasibility_motor_speed_and_tracking_are_diagnostics_not_certification",
    "scoring": "one outside-entry tip touch per accepted Attack during thrust/extended_hold/retract; same-step double touch; targets unchanged",
}

# Explicit opt-in. Historical v1 runs, defaults, masks and thresholds are kept;
# v2 is a new game rule, not a relabelling of the old failed experiments.
V2_CONFIG = dict(DEFAULT_CONFIG,
    rules_version="whole_blade_first_touch_v2",
    scoring="native finite blade cylinders enter the unchanged virtual target; no skill/phase gate; first-touch round with same-physics-step double touch; shadow entries continue for whole-skill validation",
)


def save_json(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2, allow_nan=False)+"\n")


def freeze_experiment(output, library, scene, *, metadata=None, cfg=None, implementation_paths=()):
    """New immutable-evidence folder for standalone/policy validation drivers."""
    default_implementation_paths = (
        "sim/shared_skills.py", "sim/shared_control.py", "sim/skill_validation.py",
        "sim/duel_scene.py", "sim/manual_duel.py", "sim/fencing_rules.py")
    hashes = {}
    for supplied in (*default_implementation_paths, *(implementation_paths or ())):
        path = (ROOT/Path(supplied)).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("Implementation evidence must name existing repository files")
        hashes[str(path.relative_to(ROOT.resolve()))] = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    scene.spec.option.timestep = cfg["physics_timestep_s"]
    (folder/"scene.xml").write_text(scene.spec.to_xml())
    np.savez_compressed(folder/"shared_paths.npz", **{f"{k}_{field}": getattr(v, field) for k, v in library.skills.items() for field in ("qpos", "time_s")})
    frozen = {"runtime_config": cfg, "library_config": library.config, "library_reports": library.reports,
              "skill_segments": {k: v.segments for k,v in library.skills.items()},
              "source_hashes": scene.source_hashes, "metadata": metadata or {},
              "implementation_sha256": hashes}
    save_json(folder/"experiment.json", frozen)
    return folder


def observation(scene, owner, controllers, now):
    """Same observation schema for either player. No opponent program or phase."""
    other = "a2" if owner == "a1" else "a1"
    mapping, enemy = scene.agents[owner], scene.agents[other]
    # All fixed bases differ only by yaw pi. Vectors use the observer's frame.
    rotation = np.diag([1., 1., 1.]) if owner == "a1" else np.diag([-1., -1., 1.])
    tip = scene.data.site_xpos[enemy["sites"]["shinai_tip"]].copy()
    grip = scene.data.site_xpos[enemy["sites"]["shinai_grip"]].copy()
    target = np.asarray(scene.config["virtual_targets"][owner]["center_world_m"])
    jac = np.zeros((3, scene.model.nv))
    mujoco.mj_jacSite(scene.model, scene.data, jac, None, enemy["sites"]["shinai_tip"])
    return {"time_s": float(now), "own_skill": controllers[owner].state(now),
            "own_q_rad": scene.data.qpos[mapping["qpos_ids"]].copy(),
            "own_qvel_rad_s": scene.data.qvel[mapping["dof_ids"]].copy(),
            "opponent_tip_minus_own_target_local_m": rotation@(tip-target),
            "opponent_grip_minus_own_target_local_m": rotation@(grip-target),
            "opponent_tip_velocity_local_m_s": rotation@(jac@scene.data.qvel)}


def _controller_implementation_hashes(factory, controllers):
    """Hash actual controller/factory modules without exposing their contents."""
    result = {}
    for implementation in (factory, *(type(c) for c in controllers.values())):
        while isinstance(implementation, functools.partial):
            implementation = implementation.func
        try:
            filename = inspect.getsourcefile(implementation)
        except TypeError:
            filename = None
        if filename is None:
            raise ValueError("Custom controller/factory implementation source must be inspectable")
        path = Path(filename).resolve()
        label = str(path.relative_to(ROOT.resolve())) if path.is_relative_to(ROOT.resolve()) else str(path)
        result[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _controller_diagnostics(controller):
    """Optional controller-owned audit data, never added to opponent inputs."""
    provider = getattr(controller, "diagnostics", None)
    result = copy.deepcopy(provider()) if callable(provider) else {}
    if not isinstance(result, dict):
        raise ValueError("Controller diagnostics() must return a dictionary")
    for name in ("action_map", "feedback_gates", "ignored_busy_requests", "ignored_requests"):
        if name not in result and hasattr(controller, name):
            result[name] = copy.deepcopy(getattr(controller, name))
    result.setdefault("events", copy.deepcopy(controller.events))
    return result


def run_trial(scene, library, policies, duration_s, cfg=None, output=None, *, controller_factory=None):
    """Run native physics with the historical controller, or an explicit v2 one.

    A custom ``factory(library)`` is invoked independently for both slots. Its
    request/state/target/events interface matches SkillController; optional
    observe_own_state(q, qvel, now) receives copies of current OWN state only,
    once per physics frame before either policy's decision snapshot. A
    nonempty controller.failure invalidates the trial and prevents a new
    action from being accepted. Custom controllers cannot opt into v1's old
    animation-phase scoring. None preserves the historical default path.
    """
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    rules_version = cfg.get("rules_version", "tip_attack_v1")
    if rules_version not in {"tip_attack_v1", "whole_blade_first_touch_v2"}:
        raise ValueError("Unknown rules version")
    full_blade = rules_version == "whole_blade_first_touch_v2"
    if controller_factory is not None and not full_blade:
        raise ValueError("Custom controller_factory requires explicit whole_blade_first_touch_v2 rules")
    if controller_factory is not None and not callable(controller_factory):
        raise ValueError("controller_factory must be callable")
    if not np.isfinite(duration_s) or duration_s <= 0 or set(policies) != {"a1", "a2"}:
        raise ValueError("Positive duration and both symmetric policies required")
    if not library.reports["library"]["all_skills_pass_geometry"]:
        raise ValueError("Shared geometry preflight failed; failed rows remain in library reports")
    if not np.isclose(cfg["base_distance_m"], scene.config["base_distance_m"], rtol=0., atol=1e-12):
        raise ValueError("Runtime base distance must match the actual scene")
    model, data = scene.model, scene.data
    dt, period, decision = (cfg[k] for k in ("physics_timestep_s", "control_period_s", "decision_period_s"))
    if not all(np.isclose(x/dt, round(x/dt)) and x >= dt for x in (period, decision)):
        raise ValueError("Control and decision intervals must divide physics steps")
    model.opt.timestep = dt
    scene.spec.option.timestep = dt
    mujoco.mj_resetData(model, data)
    for mapping in scene.agents.values():
        data.qpos[mapping["qpos_ids"]] = library.guard_q
    mujoco.mj_forward(model, data)
    initial_q = data.qpos.copy()
    initial_contacts = duel_contact_records(scene, rules_version=rules_version)
    if np.any(model.body_gravcomp) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("No external force, torque or body gravity compensation")
    factory = SkillController if controller_factory is None else controller_factory
    controllers = {name: factory(library) for name in policies}
    if controllers["a1"] is controllers["a2"]:
        raise ValueError("Controller factory must create independent instances for both agents")
    controller_hashes = (_controller_implementation_hashes(factory, controllers)
                         if controller_factory is not None else None)
    controller_failures = {}

    def failed_controllers():
        for owner, controller in controllers.items():
            fault = getattr(controller, "failure", None)
            if fault:
                controller_failures.setdefault(owner, copy.deepcopy(fault))
        return bool(controller_failures)
    gravity = mujoco.MjData(model)
    centers = {"a1": scene.config["virtual_targets"]["a2"]["center_world_m"],
               "a2": scene.config["virtual_targets"]["a1"]["center_world_m"]}
    radius = scene.config["virtual_targets"]["a1"]["radius_m"]+(0. if full_blade else cfg["scoring_tip_radius_m"])
    if scene.config["virtual_targets"]["a1"]["radius_m"] != scene.config["virtual_targets"]["a2"]["radius_m"]:
        raise ValueError("Equal targets required")
    if full_blade:
        from .fencing_rules import FirstTouchScorer, blade_sections_from_scene
        scorer = FirstTouchScorer(centers, radius)
        previous_blades = blade_sections_from_scene(scene)
    else:
        scorer = EntryScorer(centers, radius)
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    peak_speed, peak_error, peak_torque = (np.zeros(model.nv) for _ in range(3))
    min_distance = {k: np.inf for k in policies}
    min_active_distance = dict(min_distance)
    force_peak, impulse, illegal_peak, penetration, limit_peak = 0., 0., 0., 0., 0.
    minimum_margin, first_blade_contact, failure = np.inf, None, None
    contact_pairs, illegal_pairs, neutral_pairs = {}, {}, {}
    neutral_peak = 0.
    hashes = {k: hashlib.sha256() for k in policies}
    traces = {k: [] for k in ("time_s", "qpos", "qvel", "target_qpos", "ctrl", "actuator_force_nm", "blade_normal_force_n")}
    decisions = []
    contact_impulse_samples = []
    steps = int(np.ceil(duration_s/dt))
    commands = {k: library.guard_q.copy() for k in policies}
    states = {k: controllers[k].state(0.) for k in policies}
    previous_tip = {k: data.site_xpos[m["sites"]["shinai_tip"]].copy() for k, m in scene.agents.items()}
    previous_time = 0.
    position_control(scene, gravity, commands)
    for step in range(steps+1):
        now = float(data.time)
        mujoco.mj_forward(model, data)
        # Score the just-completed step using the phase that commanded it.
        tips = {k: data.site_xpos[m["sites"]["shinai_tip"]].copy() for k, m in scene.agents.items()}
        if full_blade:
            blades = blade_sections_from_scene(scene)
            scorer.update(previous_blades, blades, previous_time, now)
            if scorer.failure:
                failure = "uncertified_scoring_geometry"
                break
        else:
            scorer.update(states, previous_tip, tips, previous_time, now)
        for owner, controller in controllers.items():
            feedback = getattr(controller, "observe_own_state", None)
            if callable(feedback):
                mapping = scene.agents[owner]
                feedback(data.qpos[mapping["qpos_ids"]].copy(), data.qvel[mapping["dof_ids"]].copy(), now)
        if failed_controllers():
            failure = "controller_failure"
            break
        if step % round(decision/dt) == 0:
            # Snapshot both observations BEFORE either policy changes a clock.
            obs = {k: observation(scene, k, controllers, now) for k in policies}
            choices = {k: policies[k].decide(obs[k]) for k in policies}
            for k, choice in choices.items():
                if failed_controllers():
                    break
                accepted = controllers[k].request(choice, now)
                if choice:
                    decisions.append({"time_s": now, "agent": k, "action_id": int(choice), "accepted": accepted})
        states = {k: controllers[k].state(now) for k in policies}
        if failed_controllers():
            failure = "controller_failure"
            break
        if step % round(period/dt) == 0:
            commands = {k: c.target(now) for k, c in controllers.items()}
            if failed_controllers():
                failure = "controller_failure"
                break
            for k in policies:
                hashes[k].update(np.asarray(commands[k], dtype="<f8").tobytes())
            position_control(scene, gravity, commands)
            # Keep recorded ctrl, actuator_force, contact forces and qacc at
            # the SAME state/control instant, not old-force/new-ctrl pairs.
            mujoco.mj_forward(model, data)
        full_target = np.empty(model.nq)
        for k, mapping in scene.agents.items():
            full_target[mapping["qpos_ids"]] = commands[k]
            distance = float(np.linalg.norm(tips[k]-centers[k]))
            min_distance[k] = min(min_distance[k], distance)
            if states[k]["strike_active"]:
                min_active_distance[k] = min(min_active_distance[k], distance)
        if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc, data.actuator_force)):
            failure = "nonfinite_state"
            break
        peak_speed = np.maximum(peak_speed, np.abs(data.qvel))
        peak_error = np.maximum(peak_error, np.abs(data.qpos-full_target))
        peak_torque = np.maximum(peak_torque, np.abs(data.actuator_force))
        minimum_margin = min(minimum_margin, float(np.minimum(data.qpos-model.jnt_range[:, 0], model.jnt_range[:, 1]-data.qpos).min()))
        limits = np.flatnonzero(data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        limit_peak = max(limit_peak, float(np.max(np.abs(data.efc_force[limits]), initial=0.)))
        blade_force = 0.
        for contact in duel_contact_records(scene, rules_version=rules_version):
            normal = max(0., contact["normal_force_n"])
            penetration = max(penetration, -contact["distance_m"])
            pair = " / ".join(sorted((contact["geom1"], contact["geom2"])))
            table = (contact_pairs if contact["category"] == "cross_blade_blade" else
                     neutral_pairs if contact["category"] == "cross_blade_guard" else illegal_pairs)
            item = table.setdefault(pair, {"category": contact["category"], "peak_force_n": 0., "minimum_distance_m": 0., "first_time_s": now})
            item["peak_force_n"] = max(item["peak_force_n"], normal)
            item["minimum_distance_m"] = min(item["minimum_distance_m"], contact["distance_m"])
            if contact["category"] == "cross_blade_blade":
                blade_force += normal
            elif contact["category"] == "cross_blade_guard":
                neutral_peak = max(neutral_peak, normal)
            else:
                illegal_peak = max(illegal_peak, normal)
        force_peak = max(force_peak, blade_force)
        impulse += blade_force*dt
        if blade_force > 0. and now < duration_s:
            contact_impulse_samples.append({"time_s": now, "normal_force_n": blade_force,
                                            "interval_s": min(dt, duration_s-now)})
        if blade_force > cfg["minimum_blade_force_n"] and first_blade_contact is None:
            first_blade_contact = now
        if step % round(period/dt) == 0 or step == steps:
            for key, value in (("time_s", now), ("qpos", data.qpos.copy()), ("qvel", data.qvel.copy()),
                               ("target_qpos", full_target.copy()), ("ctrl", data.ctrl.copy()),
                               ("actuator_force_nm", data.actuator_force.copy()), ("blade_normal_force_n", blade_force)):
                traces[key].append(value)
        if any(w.number for w in data.warning):
            failure = "mujoco_warning"
        elif np.any(np.abs(data.actuator_force) > caps+1e-6):
            failure = "native_actuator_cap_exceeded"
        elif illegal_peak > cfg["illegal_contact_force_threshold_n"]:
            failure = "illegal_contact_force"
        elif penetration > cfg["penetration_limit_m"]:
            failure = "excessive_proxy_penetration"
        elif limit_peak > cfg["hypothetical_limit_force_ceiling_nm"]:
            failure = "unidentified_joint_limit_assistance"
        elif minimum_margin < -.005:
            failure = "joint_left_hypothetical_range"
        if failure or step == steps:
            break
        previous_tip, previous_time = tips, now
        if full_blade:
            previous_blades = blades
        mujoco.mj_step(model, data)
    if full_blade:
        if failure:
            scorer.invalidate(failure, float(data.time))
        scorer.finish(float(data.time))
    result = {"status": "invalid" if failure else "completed", "failure": failure,
              "requested_duration_s": duration_s, "completed_time_s": float(data.time),
              "initial_qpos": initial_q.tolist(), "initial_contacts": initial_contacts,
              "qpos_resets_after_initialization": 0,
              "scoring_radius_m": radius, "targets_by_attacker": centers,
              # For effectiveness, ALL physical entries matter, even after
              # the actual first-touch game would have ended. Never erase a
              # late hit to make a contacted parry look successful.
              "hits": scorer.all_entries if full_blade else scorer.events,
              "inactive_target_entries": [] if full_blade else scorer.inactive_entries,
              "action_events": {k: c.events for k, c in controllers.items()}, "decisions": decisions,
              "command_sha256": {k: h.hexdigest() for k, h in hashes.items()},
              "minimum_tip_target_center_distance_m": min_distance,
              "minimum_active_tip_target_distance_m": min_active_distance,
              "first_positive_blade_contact_time_s": first_blade_contact,
              "blade_contact_peak_force_n": force_peak, "blade_contact_normal_impulse_ns": impulse,
              "blade_contact_impulse_samples": contact_impulse_samples,
              "blade_contact_pairs": contact_pairs, "illegal_contact_pairs": illegal_pairs,
              "neutral_weapon_contact_pairs": neutral_pairs,
              "neutral_weapon_contact_peak_force_n": neutral_peak,
              "illegal_contact_peak_force_n": illegal_peak, "maximum_contact_penetration_m": penetration,
              "peak_hypothetical_limit_constraint_force_nm": limit_peak,
              "minimum_actual_joint_margin_rad": minimum_margin,
              "peak_actual_joint_speed_rad_s": peak_speed.tolist(), "peak_joint_tracking_error_rad": peak_error.tolist(),
              "peak_actuator_force_nm": peak_torque.tolist(), "unchanged_force_caps_nm": caps.tolist(),
              "diagnostics": {"speed_over_2rad_s": bool(peak_speed.max() > cfg["diagnostic_actual_speed_rad_s"]),
                              "tracking_error_over_0_5rad": bool(peak_error.max() > cfg["diagnostic_tracking_error_rad"])},
              "runtime_config": cfg, "source_hashes": scene.source_hashes,
              "geometry_sha256": hashlib.sha256(b"".join(v.qpos.astype("<f8").tobytes()+v.time_s.astype("<f8").tobytes() for v in library.skills.values())).hexdigest(),
              "limitations": "Shared finite manual skills and native rigid proxies; no RL, measured hardware, dataset imitation or verified autonomous defence."}
    if full_blade:
        result.update(round_first_touch_events=scorer.events, round_result=scorer.round_result,
                      scoring_failure=scorer.failure,
                      initial_blade_target_overlap=scorer.initial_overlap_events,
                      minimum_blade_target_clearance_m=scorer.minimum_clearance_m,
                      validation_continues_after_round_end=True)
    if controller_factory is not None:
        result.update(controller_diagnostics={k: _controller_diagnostics(c) for k, c in controllers.items()},
                      controller_implementation_sha256=controller_hashes,
                      controller_failures=controller_failures)
    if output is not None:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/"trajectory.npz", **traces)
        save_json(folder/"summary.json", result)
    return result


def compare_defence(baseline, defended, attacker="a1", cfg=None):
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    hits0 = [h for h in baseline["hits"] if h["agent"] == attacker]
    hits1 = [h for h in defended["hits"] if h["agent"] == attacker]
    same = (baseline["command_sha256"][attacker] == defended["command_sha256"][attacker]
            and np.array_equal(baseline["initial_qpos"], defended["initial_qpos"]))
    # Independent comparisons must not silently compare different fixtures,
    # scoring thresholds, motor caps, authored paths or truncated windows.
    comparable = all(baseline[k] == defended[k] for k in (
        "runtime_config", "scoring_radius_m", "targets_by_attacker", "unchanged_force_caps_nm",
        "requested_duration_s", "source_hashes", "geometry_sha256"))
    full_window = all(abs(r["completed_time_s"]-r["requested_duration_s"]) <= r["runtime_config"]["physics_timestep_s"]+1e-7 for r in (baseline, defended))
    valid = same and comparable and full_window and baseline["status"] == defended["status"] == "completed"
    contact = defended["first_positive_blade_contact_time_s"]
    # Only impulse delivered BEFORE the no-defence hit can support the
    # interception claim. A tiny early graze plus a large late collision is
    # not evidence of stopping the original threat.
    cutoff = min((h["time_s"] for h in hits0), default=0.)
    samples = defended.get("blade_contact_impulse_samples")
    early_impulse = (sum(s["normal_force_n"]*max(0., min(s["interval_s"], cutoff-s["time_s"])) for s in samples)
                     if samples is not None else None)
    v2 = baseline["runtime_config"].get("rules_version") == "whole_blade_first_touch_v2"
    impulse_evidence = (early_impulse if early_impulse is not None else
                        (0. if v2 else defended["blade_contact_normal_impulse_ns"]))
    causal = bool(hits0 and contact is not None and contact < hits0[0]["time_s"]
                  and impulse_evidence >= cfg["minimum_blade_impulse_ns"])
    return {"valid_paired_trial": valid, "same_attack_commands_and_initial_state": same,
            "same_fixture_runtime_and_full_window": comparable and full_window,
            "baseline_hit": bool(hits0), "defended_hit": bool(hits1),
            "blade_impulse_before_baseline_hit_ns": early_impulse,
            "post_contact_target_entry": bool(contact is not None and any(h["time_s"] >= contact for h in hits1)),
            "blade_contact_before_baseline_hit": causal,
            "successful_physical_defence": bool(valid and hits0 and not hits1 and causal)}


def run_suite(output, offsets=(-2., -1., 0., 1., 2., 3., 4.), cfg=None, library_config=None):
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    library = build_shared_library(library_config)
    scene = build_duel_scene(cfg["base_distance_m"])
    attack = library.skills["attack"]
    start = cfg["initial_hold_s"]
    thrust_start = start+next(s["start_s"] for s in attack.segments if s["name"] == "thrust")
    duration = max(start+attack.duration_s, thrust_start+max(offsets)+library.skills["parry"].duration_s)+cfg["final_hold_s"]
    scene.spec.option.timestep = cfg["physics_timestep_s"]
    (folder/"scene.xml").write_text(scene.spec.to_xml())
    np.savez_compressed(folder/"shared_paths.npz", **{f"{k}_{field}": getattr(v, field) for k, v in library.skills.items() for field in ("qpos", "time_s")})
    report = {"status": "running", "scope": "shared_finite_skill_effectiveness_before_RL", "runtime_config": cfg,
              "library_config": library.config, "library_reports": library.reports,
              "skill_segments": {k: v.segments for k, v in library.skills.items()},
              "source_hashes": scene.source_hashes, "software_versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
              "implementation_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ("sim/shared_skills.py", "sim/shared_control.py", "sim/skill_validation.py", "sim/duel_scene.py", "sim/manual_duel.py", "sim/fencing_rules.py")},
              "baselines": {}, "parry_trials": [], "matrix": []}
    save_json(folder/"summary.json", report)
    if not library.reports["library"]["all_skills_pass_geometry"]:
        report.update(status="failed_geometry_preflight", may_start_RL=False, invalid_cases=["geometry_preflight"])
        save_json(folder/"summary.json", report)
        return report
    for attacker in ("a1", "a2"):
        defender = "a2" if attacker == "a1" else "a1"
        baseline = run_trial(scene, library, {attacker: ScriptedPolicy([(start, 1)]), defender: ScriptedPolicy()}, duration, cfg, folder/f"{attacker}_attack_guard")
        report["baselines"][attacker] = baseline
        save_json(folder/"summary.json", report)
        print(attacker, "Attack/Guard", baseline["status"], baseline["hits"], flush=True)
        for offset in offsets:
            parry_start = thrust_start+offset
            if parry_start < 0:
                continue
            label = f"{attacker}_parry_offset_{offset:+.2f}"
            trial = run_trial(scene, library, {attacker: ScriptedPolicy([(start, 1)]), defender: ScriptedPolicy([(parry_start, 2)])}, duration, cfg, folder/label)
            comparison = compare_defence(baseline, trial, attacker, cfg)
            report["parry_trials"].append({"label": label, "attacker": attacker, "offset_from_thrust_start_s": offset, "comparison": comparison, "trial": trial})
            save_json(folder/"summary.json", report)
            print(label, trial["status"], comparison, flush=True)
    # All 3x3 simultaneous choices plus Attack/Attack start offsets diagnose
    # sword-sword clashes and recovery interactions, not a fake RPS rewardtable.
    cases = [(a, b, 0.) for a in range(3) for b in range(3)]
    cases += [(1, 1, delay) for delay in (-4., -2., 2., 4.)]
    for a, b, delay in cases:
        t1, t2 = start+max(0., -delay), start+max(0., delay)
        end = max(t1+library.skills[{0:"guard", 1:"attack", 2:"parry"}[a]].duration_s,
                  t2+library.skills[{0:"guard", 1:"attack", 2:"parry"}[b]].duration_s)+cfg["final_hold_s"]
        label = f"matrix_{a}_{b}_offset_{delay:+.2f}"
        trial = run_trial(scene, library, {"a1": ScriptedPolicy([(t1, a)]), "a2": ScriptedPolicy([(t2, b)])}, end, cfg, folder/label)
        report["matrix"].append({"label": label, "actions": [a,b], "a2_start_offset_s": delay, "trial": trial})
        save_json(folder/"summary.json", report)
        print(label, trial["status"], trial["hits"], flush=True)
    good = [p for p in report["parry_trials"] if p["comparison"]["successful_physical_defence"]]
    invalid = [p["label"] for p in report["parry_trials"]+report["matrix"] if p["trial"]["status"] != "completed"]
    invalid += [f"{k}_attack_guard" for k, trial in report["baselines"].items() if trial["status"] != "completed"]
    report.update(status="completed_diagnostic_not_RL_readiness", successful_defence_cases=[p["label"] for p in good], invalid_cases=invalid,
                  may_start_RL=False, next_gate="Independent current-observation policies and continuous counterplay not yet validated")
    save_json(folder/"summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-2., -1., 0., 1., 2., 3., 4.])
    parser.add_argument("--library-config", type=Path)
    parser.add_argument("--rules", choices=("tip_attack_v1", "whole_blade_first_touch_v2"), default="tip_attack_v1")
    parser.add_argument("--decision-period-s", type=float)
    args = parser.parse_args()
    cfg = dict(V2_CONFIG if args.rules == "whole_blade_first_touch_v2" else DEFAULT_CONFIG)
    if args.decision_period_s is not None:
        cfg["decision_period_s"] = args.decision_period_s
    report = run_suite(args.output, args.offsets, cfg=cfg,
                       library_config=json.loads(args.library_config.read_text()) if args.library_config else None)
    return 2 if report.get("invalid_cases") else 0


if __name__ == "__main__":
    raise SystemExit(main())
