"""View recorded physics states; replay never invents or recomputes outcomes.

The experiment must first be run with sim.manual_duel. Display interpolation
uses saved mj_step states, not commanded skill paths or a new physics run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

def sample_state(trace, time):
    return np.array([np.interp(time, trace["time_s"], trace["qpos"][:, i])
                     for i in range(trace["qpos"].shape[1])])


def render_pair(run, output, case="nominal", fps=20, start_s=0., end_s=None, azimuth=90.):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Existing replay is preserved: {output}")
    if not 1 <= fps <= 60:
        raise ValueError("Use 1 to 60 display frames per second")
    report = json.loads((run/"summary.json").read_text())
    pair = next(p for p in report["pairs"] if p["label"] == case)
    traces = [np.load(run/case/name/"trajectory.npz", allow_pickle=False)
              for name in ("no_defence", "parry")]
    # Use the exported experiment scene, never today's regenerated placement
    # or geometry. Mesh files must remain available at their exported paths.
    model = mujoco.MjModel.from_xml_path(str(run/"scene.xml"))
    if any(t["qpos"].ndim != 2 or t["qpos"].shape[1] != model.nq for t in traces):
        raise ValueError("Saved state dimensions must match the exported scene")
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 560
    states = [mujoco.MjData(model), mujoco.MjData(model)]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [report["manual_geometry_config"]["base_distance_m"]/2, 0., .85]
    camera.azimuth, camera.elevation, camera.distance = azimuth, -17., 3.25
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # inherited diagnostic proxy overlays, not physics
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
    duration = min(float(t["time_s"][-1]) for t in traces)
    end_s = duration if end_s is None else end_s
    if not np.isfinite([start_s, end_s, azimuth]).all() or not 0 <= start_s < end_s <= duration+1e-8:
        raise ValueError("Replay excerpt must lie within both recorded trials")
    is_ase = "ase_geometry_config" in report
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve real simulation time: one output frame per 1/fps second.
    with mujoco.Renderer(model, height=560, width=640) as renderer, imageio.get_writer(
            output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for time in np.arange(start_s, end_s+1e-9, 1/fps):
            panels = []
            for index, (trace, data, result) in enumerate(zip(traces, states, [pair["no_defence"], pair["parry"]])):
                data.qpos[:] = sample_state(trace, time)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 86), fill=(16, 23, 32))
                draw.text((10, 7), "RECORDED PHYSICS REPLAY | 1x time"+ (" | EXCERPT" if start_s or end_s < duration else ""), font=font, fill=(255, 220, 130))
                title = "NO DEFENCE" if index == 0 else ("SCHEDULED ASE PARRYUP" if is_ase else "SCHEDULED SIDE PARRY")
                draw.text((10, 31), f"{title}  |  t={time:.2f}s", font=font, fill="white")
                hit_at = result["first_target_hit_time_s"]
                hit = hit_at is not None and time >= hit_at
                contact_force = float(np.interp(time, trace["time_s"], trace["blade_normal_force_n"]))
                status = "TARGET HIT" if hit else "target not hit yet"
                color = (255, 100, 90) if hit else (120, 240, 170)
                draw.text((10, 56), f"{status} | blade force {contact_force:.1f} N", font=font, fill=color)
                if result["status"] != "completed":
                    draw.rectangle((0, 86, 640, 117), fill=(130, 20, 20))
                    draw.text((10, 91), f"INVALID TRIAL: {result['failure']}", font=font, fill="white")
                elif is_ase:
                    draw.text((10, 92), "ASE geometric retargeting; source rhythm not preserved", font=small, fill=(255, 220, 130))
                draw.rectangle((0, 523, 640, 560), fill=(16, 23, 32))
                draw.text((10, 527), "Rigid proxies, assumed joint ranges/debug motor caps.", font=small, fill="white")
                draw.text((10, 543), "No learning, perception, human target or hardware validation.", font=small, fill="white")
                panels.append(np.asarray(frame))
            writer.append_data(np.concatenate(panels, axis=1))
    for trace in traces:
        trace.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="nominal")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--start-s", type=float, default=0.)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--azimuth", type=float, default=90.)
    args = parser.parse_args()
    print(render_pair(args.run, args.output, args.case, args.fps, args.start_s, args.end_s, args.azimuth), flush=True)


if __name__ == "__main__":
    main()
