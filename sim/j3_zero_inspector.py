"""Static-only MuJoCo pose editor with J3 displayed in hardware coordinates.

The shared training model remains in its original CAD coordinates.  This opt-in
copy moves only the J3 display origin so that zero means a straight elbow.  It
does not turn a manually edited pose into a valid motion, controller or skill.
The interactive loop calls ``mj_forward`` but deliberately never ``mj_step``.
In this editor only, the visible Control sliders set joint poses directly;
they are not commands to the training controller.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .duel_scene import DuelScene
from .joint_constraints import (JOINT_NAMES, JointConstraints,
                                enforce_joint_trajectory,
                                load_joint_constraints, model_joint_ranges_rad)
from .limited_duel import SLOTS, build_limited_duel


STATUS = "static_pose_inspection_only_not_training_or_dynamics_validation"
CONTROL_MODE = "control_sliders_set_static_display_joint_radians_v1"
# Six familiar legacy-center coordinates, not an inherited guard certificate.
# J3 is supplied separately from the newly confirmed hardware zero.
DEFAULT_OTHER_CAD_RAD = np.array([
    4.537856055185257, .5061454830783556, 3.4033920413889427, 0.,
    2.7576202181510405, .7853981633974483, .6283185307179586,
])


def _offset(constraints):
    if not isinstance(constraints, JointConstraints) or not constraints.confirmed:
        raise ValueError("A confirmed JointConstraints mapping is required")
    if tuple(constraints.sign) != (1.,) * 7:
        raise ValueError("This J3-only inspector requires the current +1 signs")
    other = np.delete(np.asarray(constraints.offset_deg), 3)
    if np.any(other != 0):
        raise ValueError("Only the J3 display origin may be rebased")
    if constraints.offset_deg[3] == 0:
        raise ValueError("J3 hardware-straight zero offset has not been configured")
    return float(np.deg2rad(constraints.offset_deg[3]))


def _vector(value):
    raw = np.asarray(value)
    if raw.shape[-1:] != (7,) or raw.dtype.kind not in "fiu":
        raise ValueError("Expected finite numeric CAD/display radians ending in 7")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Pose values must be finite")
    return result.copy()


def cad_to_display(q_rad, constraints):
    """J0..J6 radians, with only J3 changed to its hardware-zero coordinate."""
    result = _vector(q_rad)
    result[..., 3] += _offset(constraints)
    return result


def display_to_cad(q_rad, constraints):
    """Exact inverse, without clipping, wrapping, or pose certification."""
    result = _vector(q_rad)
    result[..., 3] -= _offset(constraints)
    return result


def initial_cad_pose(constraints):
    q = DEFAULT_OTHER_CAD_RAD.copy()
    q[3] = -_offset(constraints)
    enforce_joint_trajectory(q, constraints)
    return q


def rebase_scene(source, constraints):
    """Copy a current CAD scene, never modify the shared source or its state.

    The unit-gear native position actuators have length=q. Adding the same
    offset to their control and length cancels in their affine force law.
    This identity is tested but is not a general transmission conversion.
    """
    offset = _offset(constraints)
    if 'display_rebase' in source.config:
        raise ValueError('Do not apply the display zero shift twice')
    expected_ranges = model_joint_ranges_rad(constraints)
    for ids in source.agents.values():
        if not np.array_equal(source.model.jnt_range[ids['joint_ids']], expected_ranges):
            raise ValueError('Source scene does not use this confirmed CAD-coordinate mapping')
    spec = source.spec.copy()
    for slot, ids in source.agents.items():
        name = f"{slot}_J3"
        aid = ids["actuator_ids"][3]
        if (source.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or source.model.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED
                or source.model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_AFFINE
                or not np.array_equal(source.model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
                or source.model.actuator_gainprm[aid, 0] != -source.model.actuator_biasprm[aid, 1]):
            raise ValueError("J3 actuator is not the tested unit-gear native position servo")
        joint, actuator = spec.joint(name), spec.actuator(name)
        joint.ref += offset
        joint.springref += offset
        joint.range = np.asarray(joint.range) + offset
        actuator.ctrlrange = np.asarray(actuator.ctrlrange) + offset
    for key in spec.keys:
        q, ctrl = np.asarray(key.qpos).copy(), np.asarray(key.ctrl).copy()
        for ids in source.agents.values():
            q[ids["qpos_ids"][3]] += offset
            ctrl[ids["actuator_ids"][3]] += offset
        key.qpos, key.ctrl = q, ctrl
    spec.modelname = "J3_HARDWARE_ZERO_STATIC_INSPECTION_ONLY"
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = source.data.qpos
    data.qvel[:] = source.data.qvel
    data.ctrl[:] = source.data.ctrl
    for ids in source.agents.values():
        data.qpos[ids["qpos_ids"][3]] += offset
        data.ctrl[ids["actuator_ids"][3]] += offset
    mujoco.mj_forward(model, data)
    config = copy.deepcopy(source.config)
    config["status"] = STATUS
    config["display_rebase"] = {
        "joint": "J3", "offset_rad": offset,
        "formula": "display_J3_rad = original_CAD_J3_rad + offset_rad",
        "original_training_coordinates_changed": False,
        "physics_stepping_permitted_by_this_tool": False,
    }
    return DuelScene(model, data, spec, config, source.source_hashes.copy(),
                     copy.deepcopy(source.agents))


def build_inspector(constraints=None, initial_qpos=None):
    constraints = load_joint_constraints() if constraints is None else constraints
    initial = initial_cad_pose(constraints) if initial_qpos is None else initial_qpos
    source = build_limited_duel(initial_qpos=initial, constraints=constraints)
    return source, rebase_scene(source, constraints)


def compare_coordinates(source, display, constraints):
    """Check the coordinate change at a nonsaturated nonzero servo error."""
    a, b = mujoco.MjData(source.model), mujoco.MjData(display.model)
    ranges = source.model.jnt_range
    a.qpos[:] = ranges[:, 0] * .4 + ranges[:, 1] * .6
    a.qvel[:] = np.linspace(-.07, .09, source.model.nv)
    a.ctrl[:] = a.qpos + np.linspace(-.0005, .0007, source.model.nu)
    b.qpos[:], b.qvel[:], b.ctrl[:] = a.qpos, a.qvel, a.ctrl
    expected_length = np.zeros(source.model.nu)
    for ids in source.agents.values():
        qid, aid = ids["qpos_ids"][3], ids["actuator_ids"][3]
        b.qpos[qid] += _offset(constraints)
        b.ctrl[aid] += _offset(constraints)
        expected_length[aid] = _offset(constraints)
    mujoco.mj_forward(source.model, a)
    mujoco.mj_forward(display.model, b)
    errors = {}
    for field in ("xpos", "xmat", "geom_xpos", "geom_xmat", "site_xpos",
                  "actuator_velocity", "actuator_force", "qfrc_actuator",
                  "qfrc_bias", "qfrc_passive", "qacc"):
        errors[field] = float(np.max(np.abs(getattr(a, field) - getattr(b, field))))
    errors["actuator_length_expected_shift"] = float(np.max(np.abs(
        b.actuator_length - a.actuator_length - expected_length)))
    # Forces may involve cancellation of two ~1000x values; tolerance remains
    # much smaller than any task-scale pose/force threshold.
    if any(value > 1e-8 for value in errors.values()):
        raise ValueError(f"J3 display rebase equivalence failed: {errors}")
    return {"tested_both_slots": True, "nonzero_pose_and_velocity": True,
            "nonsaturated_control_error": True, "maximum_absolute_errors": errors,
            "scope": "forward coordinate-equivalence check; not motion validation"}


def pose_record(scene, constraints):
    result = {"status": STATUS, "time_s": float(scene.data.time), "slots": {}}
    for slot, ids in scene.agents.items():
        displayed = scene.data.qpos[ids["qpos_ids"]].copy()
        cad = display_to_cad(displayed, constraints)
        result["slots"][slot] = {
            "joint_names": list(JOINT_NAMES), "display_rad": displayed.tolist(),
            "hardware_deg": np.rad2deg(displayed).tolist(), "original_cad_rad": cad.tolist(),
        }
    return result


class StaticPoseControls:
    """Bridge the passive viewer's Control UI to bounded manual joint poses.

    Passive viewing has no physics loop to follow actuator targets. Use this
    bridge only for the explicitly rebased inspection copy. Each UI value is
    an angle, not a force/servo command. Invalid fields are restored to their
    last accepted values (never clipped or wrapped). No collision, dynamics,
    trajectory or skill validity is implied by accepting an angle.
    """

    def __init__(self, scene):
        if (scene.config.get("status") != STATUS
                or scene.config.get("display_rebase", {}).get(
                    "physics_stepping_permitted_by_this_tool") is not False):
            raise ValueError("Static controls require the dedicated inspection copy")
        self.scene = scene
        entries = [(f"{slot}_{name}", int(qid), int(aid), int(jid))
                   for slot, ids in scene.agents.items()
                   for name, qid, aid, jid in zip(
                       JOINT_NAMES, ids["qpos_ids"], ids["actuator_ids"], ids["joint_ids"])]
        model = scene.model
        self.names = [entry[0] for entry in entries]
        self.qids = np.array([entry[1] for entry in entries])
        self.aids = np.array([entry[2] for entry in entries])
        jids = np.array([entry[3] for entry in entries])
        if (len(entries) != 14 or sorted(self.qids.tolist()) != list(range(model.nq))
                or sorted(self.aids.tolist()) != list(range(model.nu))
                or len(set(jids)) != len(entries)
                or np.any(model.jnt_type[jids] != mujoco.mjtJoint.mjJNT_HINGE)
                or np.any(model.actuator_trntype[self.aids] != mujoco.mjtTrn.mjTRN_JOINT)
                or not np.array_equal(model.actuator_trnid[self.aids, 0], jids)
                or not np.all(model.jnt_limited[jids])
                or not np.all(model.actuator_ctrllimited[self.aids])):
            raise ValueError("Expected exactly two seven-hinge inspection arms")
        self.ranges = model.jnt_range[jids].copy()
        if not np.array_equal(model.actuator_ctrlrange[self.aids], self.ranges):
            raise ValueError("Static slider bounds must equal the display joint bounds")
        self.accepted = scene.data.qpos[self.qids].copy()
        if (not np.isfinite(self.accepted).all()
                or np.any(self.accepted < self.ranges[:, 0])
                or np.any(self.accepted > self.ranges[:, 1])):
            raise ValueError("Initial manual pose must be finite and within joint bounds")
        # Replace the inherited gravity-compensated servo targets with the
        # actual displayed angles before the GUI takes its first snapshot.
        self._publish_pose()

    def _publish_pose(self):
        data = self.scene.data
        data.qpos[self.qids] = self.accepted
        data.ctrl[self.aids] = self.accepted
        data.qvel[:] = 0
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.scene.model, data)

    def apply(self):
        """Apply valid edited fields and return names of rejected fields."""
        requested = self.scene.data.ctrl[self.aids].copy()
        valid = (np.isfinite(requested) & (requested >= self.ranges[:, 0])
                 & (requested <= self.ranges[:, 1]))
        self.accepted[valid] = requested[valid]
        self._publish_pose()
        return [name for name, ok in zip(self.names, valid) if not ok]


def write_inspector(output, *, constraints=None):
    """Export into a fresh directory, preserving all historical scenes."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Existing output is preserved: {output}")
    constraints = load_joint_constraints() if constraints is None else constraints
    source, scene = build_inspector(constraints)
    check = compare_coordinates(source, scene, constraints)
    output.mkdir(parents=True, exist_ok=False)
    xml = scene.spec.to_xml()
    (output / "scene.xml").write_text(xml, encoding="utf-8")
    # XML output has finite decimal precision; check the exported file too.
    exported = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    metadata = {
        "status": STATUS, "for_training": False, "motion_or_hold_certified": False,
        "constraints_source": constraints.source,
        "mapping_offset_deg": list(constraints.offset_deg),
        "joint_table_hardware_bounds_deg": [list(x) for x in constraints.bounds_deg],
        "display_only_rebase": scene.config["display_rebase"],
        "original_model_j3_ranges_rad": {
            s: source.model.jnt_range[ids["joint_ids"][3]].tolist()
            for s, ids in source.agents.items()},
        "display_model_j3_ranges_rad": {
            s: scene.model.jnt_range[ids["joint_ids"][3]].tolist()
            for s, ids in scene.agents.items()},
        "equivalence_check": check,
        "exported_xml_qpos0_max_error_rad": float(np.max(np.abs(exported.qpos0 - scene.model.qpos0))),
        "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "initial_pose": pose_record(scene, constraints),
        "interactive_control_mode": CONTROL_MODE,
        "instructions": [
            "Use this module --inspect for static editing; it never integrates physics.",
            "In this module's static editor ONLY, Control sliders directly set joint angles in radians.",
            "Both a1_J3 and a2_J3: zero=straight, limits=-20..120 degrees.",
            "Invalid numeric fields are rejected; Clear all cannot set J5 below its 15-degree minimum.",
            "Pause/Run are disabled by design. Manual pose changes do not certify safe/feasible motion.",
            "Closing --inspect writes pose_on_close.json with exact original-CAD inverse coordinates.",
            "Do not use scene.xml as a training model or replace an old trial scene.",
        ],
    }
    (output / "mapping.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return scene, constraints, metadata


def inspect(scene, constraints, output):
    # Import only when explicitly requested; no GUI or live simulation at import.
    import mujoco.viewer
    controls = StaticPoseControls(scene)
    print("STATIC POSE EDITOR: use Control sliders (radians); J3=0 is straight.")
    print("Control directly edits poses in this tool ONLY. No physics steps; Run stays disabled.")
    print("Closing writes pose_on_close.json with display and original CAD coordinates.")
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = [.875, 0., .80]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 115., -20.
        viewer.set_texts((None, None,
                          "STATIC POSE EDITOR\nControl = joint angle (rad)\nJ3 = 0: straight\nNo physics / no motion validation", ""))
        while viewer.is_running():
            viewer.sync()
            with viewer.lock():
                rejected = controls.apply()
            if rejected:
                print("Out-of-range/non-finite fields restored: " + ", ".join(rejected))
            time.sleep(.02)
    path = Path(output) / "pose_on_close.json"
    record = pose_record(scene, constraints)
    record["interactive_control_mode"] = CONTROL_MODE
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pose: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--inspect", action="store_true", help="open a static-only interactive editor")
    args = parser.parse_args()
    scene, constraints, metadata = write_inspector(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "status": metadata["status"],
                      "J3_display_bounds_deg": np.rad2deg(scene.model.jnt_range[
                          scene.agents["a1"]["joint_ids"][3]]).tolist()}))
    if args.inspect:
        inspect(scene, constraints, args.output)


if __name__ == "__main__":
    main()
