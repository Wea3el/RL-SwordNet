"""Compare two recorded shared-skill physics trials, without a new rollout.

Only logged actual joint positions are interpolated for display. Hit events,
contact forces and validity come from saved evidence, never recomputed commands
or a regenerated scene. A replay is not proof of autonomous defence or RL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import mujoco
import numpy as np


def sample_state(trace, time):
    """Interpolate recorded actual qpos by seconds, not authored skill phase."""
    if not np.isfinite(time):
        raise ValueError("Finite replay time required")
    return np.array([np.interp(time, trace["time_s"], trace["qpos"][:, i])
                     for i in range(trace["qpos"].shape[1])])


def hit_counts(result, time):
    counts = {"a1": 0, "a2": 0}
    double_touch = False
    for event in result.get("hits", []):
        if float(event["time_s"]) <= time+1e-10:
            counts[event["agent"]] += 1
            double_touch = double_touch or bool(event.get("double_touch", False))
    return counts, double_touch


def load_trial(directory, nq):
    directory = Path(directory)
    result = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    with np.load(directory/"trajectory.npz", allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in ("time_s", "qpos", "blade_normal_force_n")}
    time, q, force = (trace[key] for key in ("time_s", "qpos", "blade_normal_force_n"))
    if time.ndim != 1 or len(time) < 2 or q.shape != (len(time), nq) or force.shape != time.shape or not all(np.isfinite(x).all() for x in (time, q, force)) or not np.all(np.diff(time) > 0):
        raise ValueError("Finite recorded qpos/force on increasing times must match exported scene")
    for event in result.get("hits", []):
        if event.get("agent") not in {"a1", "a2"} or not np.isfinite(event.get("time_s", np.nan)):
            raise ValueError("Saved hit events must identify an agent and finite event time")
    return result, trace


def _trial_directory(run, selected):
    path = (run/selected).resolve()
    if not path.is_relative_to(run) or not path.is_dir():
        raise ValueError("Select an existing trial directory within --run")
    return path


def render_comparison(run, baseline, defended, output, *, start_s=0., end_s=None,
                      fps=15, azimuth=90., elevation=-17., distance=3.65):
    """Render two panels of saved physics evidence, never invoke mj_step."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Existing replay is preserved: {output}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 output filename")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("Use an integer display FPS in [1,60]")
    if not np.isfinite([start_s, azimuth, elevation, distance]).all() or start_s < 0 or distance <= 0:
        raise ValueError("Finite nonnegative excerpt start and valid camera settings required")
    directories = [_trial_directory(run, name) for name in (baseline, defended)]
    # The exact exported scene is mandatory; no current builder/defaults are
    # used to reconstruct old robot placement, proxies, masses or fixtures.
    model = mujoco.MjModel.from_xml_path(str(run/"scene.xml"))
    loaded = [load_trial(directory, model.nq) for directory in directories]
    results, traces = zip(*loaded)
    common_start = max(float(t["time_s"][0]) for t in traces)
    common_end = min(float(t["time_s"][-1]) for t in traces)
    end_s = common_end if end_s is None else end_s
    if not np.isfinite(end_s) or not common_start <= start_s < end_s <= common_end+1e-8:
        raise ValueError("Excerpt must lie within both recorded trial windows")
    suite = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    separation = float(suite.get("runtime_config", {}).get("base_distance_m", 1.75))
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Saved base separation must be positive")
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    width, height = 720, 640
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    states = [mujoco.MjData(model), mujoco.MjData(model)]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [separation/2, 0., .85]
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before allowing the encoder to write this new file.
    # A pre-existing user's video is never selected as an encoder destination.
    with output.open("xb"):
        pass
    excerpt = start_s > common_start or end_s < common_end-1e-8
    with mujoco.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
            output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for time in np.arange(start_s, end_s+1e-9, 1/fps):
            panels = []
            for index, (result, trace, data, directory) in enumerate(zip(results, traces, states, directories)):
                data.qpos[:] = sample_state(trace, time)
                data.qvel[:] = 0.
                mujoco.mj_forward(model, data)  # Render FK only; no mj_step.
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 122), fill=(16, 23, 32))
                title = "RECORDED PHYSICS | 1x time"+(" | EXCERPT" if excerpt else "")
                draw.text((12, 7), title, font=font, fill=(255, 220, 130))
                role = "BASELINE" if index == 0 else "DEFENCE CANDIDATE"
                name = textwrap.shorten(directory.name, width=36, placeholder="...")
                draw.text((12, 33), f"{role}: {name} | t={time:.2f}s", font=font, fill="white")
                counts, double = hit_counts(result, time)
                hit = bool(sum(counts.values()))
                status = f"Target hit yet: {'YES' if hit else 'NO'} | A1: {counts['a1']} / A2: {counts['a2']}"
                if double:
                    status += " | double touch"
                draw.text((12, 59), status, font=font, fill=(255, 110, 100) if hit else (120, 240, 170))
                force = float(np.interp(time, trace["time_s"], trace["blade_normal_force_n"]))
                draw.text((12, 84), f"Recorded blade force: {force:.2f} N | peak: {result.get('blade_contact_peak_force_n', 0.):.2f} N", font=font, fill="white")
                invalid = result.get("status") != "completed"
                if invalid:
                    draw.rectangle((0, 122, width, 170), fill=(135, 23, 23))
                    failure = str(result.get("failure") or result.get("status", "unknown"))
                    for line, text in enumerate(textwrap.wrap("INVALID TRIAL: "+failure, width=73)[:2]):
                        draw.text((12, 126+line*19), text, font=font, fill="white")
                else:
                    draw.rectangle((0, 122, width, 147), fill=(80, 61, 20))
                    draw.text((12, 126), "UNVALIDATED: shared-skill experiment, not autonomous proof", font=font, fill=(255, 225, 150))
                draw.rectangle((0, height-62, width, height), fill=(16, 23, 32))
                diagnostics = [name for name, value in result.get("diagnostics", {}).items() if value]
                note = "Diagnostics: "+(", ".join(diagnostics) if diagnostics else "no saved speed/tracking flags")
                draw.text((12, height-57), textwrap.shorten(note, width=95), font=small, fill=(255, 205, 130))
                draw.text((12, height-37), "Shared manual skills; rigid proxies; assumed joint ranges/debug motor caps.", font=small, fill="white")
                draw.text((12, height-19), "Interpolated logged qpos for display only. No RL or hardware validation.", font=small, fill="white")
                panels.append(np.asarray(frame))
            writer.append_data(np.concatenate(panels, axis=1))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", required=True, help="Trial directory relative to --run")
    parser.add_argument("--defended", required=True, help="Comparison trial directory relative to --run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=0.)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--azimuth", type=float, default=90.)
    parser.add_argument("--elevation", type=float, default=-17.)
    parser.add_argument("--distance", type=float, default=3.65)
    args = parser.parse_args(argv)
    print(render_comparison(args.run, args.baseline, args.defended, args.output,
                            start_s=args.start_s, end_s=args.end_s, fps=args.fps,
                            azimuth=args.azimuth, elevation=args.elevation, distance=args.distance), flush=True)


if __name__ == "__main__":
    main()
