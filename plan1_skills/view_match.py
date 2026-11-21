"""View saved matched poses without executing a physics trajectory.

The display duration is a visualization choice, NOT an approved robot speed.
Only qpos is interpolated and mj_forward is called; mj_step is never used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .library import ROOT

LABEL = "KINEMATIC PREVIEW, NOT PHYSICS; display timing not robot execution"


def load_run(run, profile, skill):
    """Load an explicit saved diagnostic run; numeric archives never use pickle."""
    import mujoco

    if profile not in {"current_debug", "hypothetical180"} or skill not in {"stab", "parry_up"}:
        raise ValueError("Unknown profile or skill")
    run = Path(run).resolve()
    report = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    order = report.get("joint_order")
    if order != [f"J{i}" for i in range(7)]:
        raise ValueError("Saved report must declare joint order J0 through J6")
    model = mujoco.MjModel.from_xml_path(str(run/profile/"scene.xml"))
    model_order = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if model_order != order or model.nq != 7 or model.nv != 7 or not np.array_equal(model.jnt_qposadr, np.arange(7)):
        raise ValueError("Scene joint order does not match the saved numeric qpos")
    with np.load(run/profile/f"{skill}.npz", allow_pickle=False) as archive:
        phase, qs = archive["phase"].copy(), archive["qpos"].copy()
    if phase.ndim != 1 or len(phase) < 2 or qs.shape != (len(phase), 7) or not np.isfinite(phase).all() or not np.isfinite(qs).all() or not np.all(np.diff(phase) > 0) or not np.isclose(phase[0], 0.) or not np.isclose(phase[-1], 1.):
        raise ValueError("Saved poses must be finite Nx7 on a strictly increasing [0,1] phase grid")
    if np.any(qs < model.jnt_range[:, 0]-1e-8) or np.any(qs > model.jnt_range[:, 1]+1e-8):
        raise ValueError("Saved poses exceed this scene's declared joint ranges")
    return model, phase, qs


def sample_pose(phase, qs, at):
    return np.array([np.interp(float(at), phase, qs[:, i]) for i in range(qs.shape[1])])


def configure_camera(camera):
    camera.lookat[:] = [.15, -.30, .85]
    camera.azimuth, camera.elevation, camera.distance = 120, -15, 3.2


def write_gif(model, phase, qs, path, duration, *, skill, profile):
    """Headless native render with a permanent non-physics warning overlay."""
    import mujoco
    from PIL import Image, ImageDraw

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite preview: {path}")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Display duration must be finite and positive")
    model.vis.global_.offwidth, model.vis.global_.offheight = 720, 600
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    configure_camera(camera)
    frames = []
    count = 64
    with mujoco.Renderer(model, height=600, width=720) as renderer:
        for index, at in enumerate(np.linspace(0., 1., count)):
            data.qpos[:] = sample_pose(phase, qs, at)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 720, 78), fill=(16, 23, 32))
            draw.text((12, 8), "KINEMATIC PREVIEW, NOT PHYSICS", fill=(255, 210, 90), font_size=21)
            draw.text((12, 33), "Display timing is NOT robot execution speed", fill="white", font_size=17)
            draw.text((12, 55), f"{profile} | {skill} | phase {at:.3f} | saved FK poses", fill="white", font_size=16)
            frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive open provides a final race-safe no-overwrite check.
    with path.open("xb") as stream:
        frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                       duration=max(10, round(1000*duration/count)), loop=0, disposal=2)
    return path


def view(model, phase, qs, duration):
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(model)
    data.qpos[:] = qs[0]
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer.cam)
        viewer.opt.geomgroup[3] = 0
        start = time.monotonic()
        while viewer.is_running():
            at = ((time.monotonic()-start)/duration) % 1.
            with viewer.lock():
                data.qpos[:] = sample_pose(phase, qs, at)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1/60.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"outputs/floor_matching_round1_refined")
    parser.add_argument("--profile", choices=["current_debug", "hypothetical180"], default="hypothetical180")
    parser.add_argument("--skill", choices=["stab", "parry_up"], default="stab")
    parser.add_argument("--duration", type=float, default=8., help="Seconds per display loop, not robot timing")
    parser.add_argument("--gif", type=Path, help="Write a NEW headless GIF instead of opening the viewer")
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Display duration must be finite and positive")
    if args.gif is not None and args.gif.exists():
        raise FileExistsError(f"Refusing to overwrite preview: {args.gif}")
    print(LABEL, flush=True)
    print("Joint poses are interpolated for display. No dynamics, collision response, or successful defence is demonstrated.", flush=True)
    if args.profile == "hypothetical180":
        print("WARNING: hypothetical180 uses assumed +/-180 degree limits, NOT measured hardware limits.", flush=True)
    model, phase, qs = load_run(args.run, args.profile, args.skill)
    if args.gif is not None:
        print(write_gif(model, phase, qs, args.gif, args.duration, skill=args.skill, profile=args.profile), flush=True)
    else:
        view(model, phase, qs, args.duration)


if __name__ == "__main__":
    main()
