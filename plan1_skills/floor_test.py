"""Reproducible first-round ASE matching on the full-size, floor-fixed arm.

This entry point prepares verified numeric references, solves actual grip/axis
IK, and screens sampled configurations. qpos assignment here is diagnostic FK,
NEVER a physics controller or evidence of successful defence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .library import ROOT, DEFAULT_CACHE, SkillLibrary, build_library, load_manifest

DEFAULT_CONFIG = ROOT / "config/floor_match.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation_z(degrees):
    t = np.deg2rad(degrees)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.]])


def dense_reference(clip, substeps):
    """Include every original phase and the specified interpolated samples."""
    if isinstance(substeps, bool) or not isinstance(substeps, int) or not 1 <= substeps <= 20:
        raise ValueError("substeps must be an integer in [1,20]")
    phase = np.r_[np.concatenate([np.linspace(a, b, substeps+1)[:-1]
                                 for a, b in zip(clip.phase[:-1], clip.phase[1:])]), clip.phase[-1]]
    samples = [clip.sample(float(p)) for p in phase]
    return phase, np.array([x[0] for x in samples]), np.array([x[1] for x in samples])


def gravity_screen(model, qs):
    """Static gravity demand only: not inverse full motion or tracking."""
    data = mujoco.MjData(model)
    torque = []
    for q in qs:
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        torque.append(data.qfrc_bias.copy())
    torque = np.array(torque)
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    exceeded = np.any(np.abs(torque) > caps+1e-9, axis=1)
    return {"maximum_absolute_static_gravity_nm": np.abs(torque).max(0).tolist(),
            "configured_debug_caps_nm": caps.tolist(),
            "samples_exceeding_static_cap": int(exceeded.sum()),
            "sample_count": len(qs),
            "limitation": "Below caps is necessary, not sufficient: inertia, damping, contact, control error and real actuator ratings are unvalidated."}, torque


def contact_screen(model, qs, threshold):
    from sim.assembly import contact_records

    data = mujoco.MjData(model)
    bad = []
    worst = 0.
    pairs = {}
    for index, q in enumerate(qs):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        records = [x for x in contact_records(model, data) if x["distance_m"] < -threshold]
        if records:
            bad.append(index)
        for item in records:
            worst = max(worst, -item["distance_m"])
            pair = " / ".join(sorted([item["geom1"], item["geom2"]]))
            previous = pairs.setdefault(pair, {"samples": 0, "maximum_depth_m": 0.})
            previous["samples"] += 1
            previous["maximum_depth_m"] = max(previous["maximum_depth_m"], -item["distance_m"])
    return {"sample_count": len(qs), "samples_with_deep_contact": len(bad),
            "failed_sample_indices": bad, "maximum_depth_m": worst, "pairs": pairs,
            "scope": "Existing primitive proxies only; not full CAD clearance or continuous swept-volume proof."}


def metrics(result, tolerance):
    pos = np.asarray(result.position_error_m)
    angle = np.rad2deg(result.angle_error_rad)
    tip = np.asarray(result.tip_error_m)
    good = (pos <= tolerance["grip_error_m"]) & (angle <= tolerance["axis_error_deg"]) & (tip <= tolerance["tip_error_m"])
    delta = np.abs(np.diff(result.qpos, axis=0))
    jumps = np.flatnonzero(np.any(delta > tolerance["maximum_adjacent_joint_change_rad"], axis=1))+1 if len(delta) else np.array([], dtype=int)
    return {"samples": len(pos), "pose_accepted": int(good.sum()),
            "failed_pose_indices": np.flatnonzero(~good).tolist(),
            "maximum_grip_error_m": float(pos.max()), "maximum_axis_error_deg": float(angle.max()),
            "maximum_tip_error_m": float(tip.max()),
            "maximum_adjacent_joint_change_rad": float(delta.max()) if len(delta) else 0.,
            "joint_jump_indices": jumps.tolist(),
            "joint_min_rad": result.qpos.min(0).tolist(), "joint_max_rad": result.qpos.max(0).tolist(),
            "all_pose_samples_accepted": bool(good.all()), "joint_continuity_accepted": not len(jumps),
            "initialization_delta_rad": result.joint_delta_rad[0].tolist(),
            "initialization_path_checked": False,
            "interpolated_joint_path_contact_samples": int(result.collision_sample_count.sum()),
            "frames_with_interpolated_path_contacts": int((~result.collision_free).sum()),
            "interpolated_path_contacts": [{"frame": i, "records": row} for i, row in enumerate(result.contacts) if row]}


def save_preview(assembly, sequences, output):
    """Static FK snapshots, visibly labelled as such, not rollout video."""
    from PIL import Image, ImageDraw

    model, data = assembly.model, assembly.data
    model.vis.global_.offwidth, model.vis.global_.offheight = 900, 750
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.15, -.30, .85]
    camera.azimuth, camera.elevation, camera.distance = 120, -15, 3.2
    tiles = []
    with mujoco.Renderer(model, height=750, width=900) as renderer:
        for name, qs in sequences.items():
            # Animation clips may spend their midpoint in a wind-up/hold. Show
            # the greatest actual tool displacement rather than assume phase .5.
            from .matching import tool_fk
            if name == "guard":
                indices = [0]
            else:
                p0, a0, _ = tool_fk(model, data, qs[0])
                scores = []
                for q in qs:
                    p, a, _ = tool_fk(model, data, q)
                    scores.append(np.linalg.norm(p-p0)+.3*np.arccos(np.clip(a@a0, -1, 1)))
                indices = sorted(set([0, int(np.argmax(scores)), len(qs)-1]))
            for index in indices:
                data.qpos[:] = qs[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 900, 65), fill=(18, 25, 34))
                draw.text((20, 12), f"{name} | sample {index}/{len(qs)-1} | DIAGNOSTIC FK, NOT PHYSICS", fill="white", font_size=20)
                draw.text((20, 38), "Original-scale CAD arm + 1.080 m shinai; fixed floor base; no torso", fill="white", font_size=17)
                im.save(output / f"{name}_{index:04d}.png")
                tiles.append(im.resize((600, 500)))
    rows = (len(tiles)+2)//3
    sheet = Image.new("RGB", (1800, rows*500), (18, 25, 34))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3)*600, (index//3)*500))
    sheet.save(output / "matching_preview.png")


def main(argv=None):
    from sim.floor import build_floor_assembly
    from sim.assembly import export_scene
    from .matching import match_sequence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_CACHE / "sources")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/floor_matching_round1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new/empty output directory; previous evidence is preserved")
    cfg = json.loads(args.config.read_text())
    if cfg.get("schema_version") != 1 or cfg.get("status") != "exploratory_kinematic_screen_not_hardware_or_defence_validation":
        raise ValueError("Expected an explicitly labelled exploratory configuration")
    output.mkdir(parents=True, exist_ok=True)
    build_library(args.source_dir, output / "templates")
    library = SkillLibrary(output / "templates")
    rotation = rotation_z(cfg["world_from_skill_yaw_deg"])
    neutral = np.asarray(cfg["neutral_grip_world_m"], dtype=float)
    tolerance = cfg["acceptance"]
    report = {"status": "running", "scope": "Static actual-tool kinematic fitting and sampled proxy-contact/gravity screening only; NOT physical playback, trained imitation or validated defence.",
              "config": cfg, "config_sha256": digest(args.config), "source_manifest_sha256": digest(ROOT / "motion_skills/plan1/manifest.json"),
              "software_versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
              "implementation_sha256": {p: digest(ROOT / p) for p in ["sim/floor.py", "plan1_skills/matching.py", "plan1_skills/floor_test.py"]},
              "source_commit": load_manifest()["upstream"]["commit"], "source_verified_by_restricted_converter": True,
              "model_source_hashes": {}, "world_from_skill_rotation": rotation.tolist(),
              "tool_length_grip_to_tip_m": .79, "sword_total_length_m": 1.08,
              "joint_order": [f"J{i}" for i in range(7)], "profiles": {},
              "unchanged": "Original CAD meshes/XML, arm length/mass, calibrated fixture, sword geometry/mass. No torso and no learned policies are added.",
              "warnings": [cfg["limits_note"], cfg["collision_note"], cfg["timing_note"],
                           "Both clips use the SAME frozen mapping. Mapping was fitted on these clips, not held-out data.",
                           "5D tool IK leaves axial roll/redundancy unconstrained except continuity; collider screening does not certify clamp clearance."]}
    (output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    prepared_preview = None
    for profile in cfg["profiles"]:
        ranges = None if profile == "current_debug" else np.tile([-np.pi, np.pi], (7, 1))
        if profile not in {"current_debug", "hypothetical180"}:
            raise ValueError("Unknown joint profile")
        assembly = build_floor_assembly(joint_ranges=ranges)
        model = assembly.model
        ranges = model.jnt_range.copy()
        report["model_source_hashes"] = assembly.source_hashes
        folder = output / profile
        folder.mkdir()
        export_scene(assembly, folder / "scene.xml")
        profile_report = {"joint_ranges_rad": ranges.tolist(), "initial_pose_contacts": contact_screen(model, [assembly.home], tolerance["penetration_m"]),
                          "skills": {}, "transitions": {}, "physics_executed": False}
        report["profiles"][profile] = profile_report
        sequences = {}
        guard = None
        for name in ["guard", "stab", "parry_up"]:
            clip = library.clips[name]
            phase, delta, direction = dense_reference(clip, cfg["substeps_per_source_interval"])
            targets = neutral + delta @ rotation.T
            axes = direction @ rotation.T
            seed = np.clip(np.asarray(cfg["exploratory_guard_seed_rad"]), ranges[:, 0], ranges[:, 1]) if guard is None else guard
            result = match_sequence(model, targets, axes, ranges, initial_q=seed,
                                    seed=cfg["solver"]["seed"], max_iterations=cfg["solver"]["max_iterations"],
                                    position_tolerance_m=cfg["solver"]["internal_grip_tolerance_m"],
                                    angle_tolerance_rad=np.deg2rad(cfg["solver"]["internal_axis_tolerance_deg"]),
                                    tip_tolerance_m=cfg["solver"]["internal_tip_tolerance_m"],
                                    collision_threshold_m=tolerance["penetration_m"])
            qs = result.qpos
            if name == "guard":
                guard = qs[0].copy()
            sequences[name] = qs
            entry = metrics(result, tolerance)
            entry["original_source_frames"] = len(clip.phase)
            entry["contacts"] = contact_screen(model, qs, tolerance["penetration_m"])
            entry["static_gravity"], torque = gravity_screen(model, qs)
            entry["candidate_passes_this_screen"] = bool(entry["all_pose_samples_accepted"] and entry["joint_continuity_accepted"] and not entry["contacts"]["samples_with_deep_contact"] and not entry["frames_with_interpolated_path_contacts"] and not entry["static_gravity"]["samples_exceeding_static_cap"])
            with np.load(output / "templates" / f"{name}.npz", allow_pickle=False) as z:
                source_time = np.interp(phase, clip.phase, z["source_time_s"])
            dt = np.diff(source_time)
            if len(dt) and np.all(dt > 0):
                velocity = np.abs(np.diff(qs, axis=0) / dt[:, None]).max(0)
                entry["peak_joint_speed_at_unapproved_source_timing_rad_s"] = velocity.tolist()
                entry["minimum_uniform_time_stretch_for_debug_speed_only"] = max(1., float(velocity.max() / assembly.config["runtime"]["max_target_speed_rad_s"]))
            np.savez_compressed(folder / f"{name}.npz", phase=phase, source_time_s=source_time, qpos=qs,
                                target_grip_m=targets, target_axis=axes, position_error_m=result.position_error_m,
                                axis_error_rad=result.angle_error_rad, tip_error_m=result.tip_error_m,
                                static_gravity_nm=torque)
            profile_report["skills"][name] = entry
            print(json.dumps({"profile": profile, "skill": name, "samples": len(qs), "pose_accepted": entry["pose_accepted"],
                              "max_grip_m": entry["maximum_grip_error_m"], "max_axis_deg": entry["maximum_axis_error_deg"],
                              "contacts": entry["contacts"]["samples_with_deep_contact"], "gravity_over_cap": entry["static_gravity"]["samples_exceeding_static_cap"]}), flush=True)
        for name in ["stab", "parry_up"]:
            for label, a, b in [(f"guard_to_{name}", guard, sequences[name][0]), (f"{name}_to_guard", sequences[name][-1], guard)]:
                s = np.linspace(0, 1, cfg["transition_samples"])
                blend = 10*s**3-15*s**4+6*s**5
                q = a + blend[:, None]*(b-a)
                contacts = contact_screen(model, q, tolerance["penetration_m"])
                gravity, torque = gravity_screen(model, q)
                profile_report["transitions"][label] = {"contacts": contacts, "static_gravity": gravity,
                    "tool_path": "FK of a minimum-jerk joint interpolation; NOT independently matched Cartesian data.", "approved_duration_s": None}
                np.savez_compressed(folder / f"{label}.npz", phase=s, qpos=q, static_gravity_nm=torque)
        if profile == "hypothetical180":
            prepared_preview = (assembly, sequences, folder)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    report["status"] = "completed_diagnostic_not_robot_ready"
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.render and prepared_preview:
        save_preview(*prepared_preview)
    print(str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
