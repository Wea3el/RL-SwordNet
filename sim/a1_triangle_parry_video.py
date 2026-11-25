"""Side-by-side replay of matched *recorded* triangle defence experiments.

This module never advances physics, interpolates a command, or reads the old
single-target score.  A partial trace stays partial: neither side is frozen to
manufacture a complete comparison window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .constrained_replay import load_recorded_states, probe_video, source_hashes
from .triangle_task import REGIONS, TriangleGeometry


def _times(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or not np.all(np.diff(values) > 0)):
        raise ValueError("Finite strictly increasing native timestamps required")
    return values


def nearest_indices(times, requested):
    """Choose actual native samples; ties select the earlier sample."""
    times = _times(times)
    requested = np.asarray(requested, dtype=float)
    if (requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all()
            or np.any(requested < times[0]) or np.any(requested > times[-1])):
        raise ValueError("Every display timestamp must lie in the native trace")
    right = np.searchsorted(times, requested, side="left").clip(1, len(times)-1)
    left = right-1
    return np.where(requested-times[left] <= times[right]-requested, left, right)


def paired_schedule(baseline_times, defended_times, expected_duration_s, *, fps=10):
    """Replay only the common evidence window, never padding a failed side."""
    bt, dt = _times(baseline_times), _times(defended_times)
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("Integer display FPS in [1,60] required")
    expected = float(expected_duration_s)
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError("Positive finite expected duration required")
    start, end = max(float(bt[0]), float(dt[0])), min(float(bt[-1]), float(dt[-1]), expected)
    if not start < end:
        raise ValueError("Two nonempty overlapping recorded windows required")
    requested = start + np.arange(int(np.ceil((end-start)*fps-1e-10)))/fps
    return {"requested_time_s": requested,
            "baseline_indices": nearest_indices(bt, requested),
            "defended_indices": nearest_indices(dt, requested),
            "common_start_s": start, "common_end_s": end,
            "expected_duration_s": expected,
            "truncated": bool(start > 1e-8 or end < expected-1e-8)}


def entry_counts(audit, actual_time_s):
    """Independent region entries up to the displayed native state, not points."""
    if not np.isfinite(actual_time_s):
        raise ValueError("Finite displayed native time required")
    counts = {owner: {region: 0 for region in REGIONS} for owner in ("a1", "a2")}
    for entry in audit["all_entries"]:
        owner, region, when = entry["agent"], entry["region"], float(entry["time_s"])
        if owner not in counts or region not in REGIONS or not np.isfinite(when) or when < 0:
            raise ValueError("Malformed independent triangle entry")
        # A failed sweep can retain diagnostic candidates, not valid entries.
        if not entry.get("step_invalid", False) and when <= actual_time_s+1e-10:
            counts[owner][region] += 1
    return counts


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pair(experiment, case):
    experiment = Path(experiment).resolve()
    if not case or Path(case).name != case or case in (".", ".."):
        raise ValueError("A single case directory name is required")
    comparison_path = experiment/(case+"_comparison.json")
    if case == 'defended' and not comparison_path.exists():
        comparison_path = experiment/'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    sides = {}
    for label, folder in (("baseline", experiment/"baseline"), ("defended", experiment/case)):
        hashes = source_hashes(folder)
        model = mujoco.MjModel.from_xml_path(str(folder/"scene.xml"))
        summary, timestamps, qpos = load_recorded_states(folder, model.nq)
        audit = comparison[label+"_triangle_audit"]
        if any(audit["native_trace_source_sha256"].get(name) != hashes[name]
               for name in ("summary.json", "trace.npz")):
            raise ValueError("Triangle audit does not belong to the recorded "+label)
        geometry = TriangleGeometry.from_mapping({k: audit["scoring_snapshot"][k]
                                                   for k in ("base_distance", "regions")})
        entry_counts(audit, float(timestamps[-1]))  # Validate all supplied entry evidence.
        sides[label] = dict(folder=folder, hashes=hashes, model=model, summary=summary,
                            times=timestamps, qpos=qpos, audit=audit, geometry=geometry)
    b, d = sides["baseline"], sides["defended"]
    if b["geometry"].to_mapping() != d["geometry"].to_mapping():
        raise ValueError("Comparison uses different triangle geometries")
    expected = float(b["summary"]["requested_duration_s"])
    if not np.isclose(expected, d["summary"]["requested_duration_s"], rtol=0, atol=1e-8):
        raise ValueError("Comparison requested different full windows")
    return comparison_path, comparison, sides, expected


def status_lines(comparison, sides, schedule):
    """Never display a conditional certificate for truncated/failed evidence."""
    status = str(comparison.get("status", "unknown"))
    failed = [name for name, side in sides.items()
              if side["summary"].get("status") != "completed"
              or not side["summary"].get("native_execution_pass")]
    if status == "qualified" and (schedule["truncated"] or failed):
        raise ValueError("Qualified result conflicts with incomplete/failed source evidence")
    if schedule["truncated"]:
        title = "TRUNCATED EVIDENCE - comparison window is incomplete"
    else:
        title = "FULL RECORDED WINDOW | result: "+status
    detail = ("LIMITED CONDITION ONLY - open-loop probe, not learned reaction"
              if status == "qualified" else "NOT CERTIFIED - open-loop probe, not a trained policy")
    if 'physical_trigger_events' in comparison:
        detail = ("CURRENT-OBSERVATION CUE | finite corridor baseline, NOT learned policy"
                  if status in ('qualified','defence_with_counter_candidate') else
                  "NOT CERTIFIED | observation-cued diagnostic, NOT learned policy")
    if failed:
        detail = "SOURCE FAILURE: "+", ".join(failed)+" | "+detail
    return title, detail


def render(experiment, case, output, *, fps=10, side_width=512, height=384, progress=None):
    """Render actual paired evidence and preserve existing output directories."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Existing replay is preserved: "+str(output))
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in (side_width, height))
            or side_width < 512 or height < 384 or side_width % 2 or height % 2):
        raise ValueError("Even side width >=512 and height >=384 required")
    comparison_path, comparison, sides, expected = load_pair(experiment, case)
    comparison_hash = _digest(comparison_path)
    schedule = paired_schedule(sides["baseline"]["times"], sides["defended"]["times"], expected, fps=fps)
    title, detail = status_lines(comparison, sides, schedule)
    output.mkdir(parents=True)
    video, partial = output/"paired_native_replay.mp4", output/"paired_native_replay.partial.mp4"
    manifest = {"status": "rendering", "case": case, "comparison": str(comparison_path),
                "comparison_sha256": comparison_hash, "comparison_status": comparison.get("status"),
                "source_sha256": {name: side["hashes"] for name, side in sides.items()},
                "source_directories": {name: str(side["folder"]) for name, side in sides.items()},
                "source_failures": {name: side["summary"].get("failure") for name, side in sides.items()},
                "sampling": "nearest actual native qpos; no interpolation, mj_step, or padding",
                "full_requested_duration_s": expected, "common_start_s": schedule["common_start_s"],
                "common_end_s": schedule["common_end_s"], "truncated_evidence": schedule["truncated"],
                "fps": fps, "width": 2*side_width, "height": height,
                "frames": len(schedule["requested_time_s"]), "not_trained_policy": True,
                "official_triangle_settlement": False,
                "entry_count_semantics": "independent swept geometry entries, not score; no block immunity"}
    manifest_path = output/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    colors = {"left": [.18, .67, 1., .23], "right": [.25, .9, .42, .23], "upper": [1., .57, .16, .23]}
    for side in sides.values():
        model = side["model"]
        model.vis.global_.offwidth, model.vis.global_.offheight = side_width, height
        side["data"] = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [side["geometry"].base_distance/2., 0., .80]
        camera.azimuth, camera.elevation, camera.distance = 115., -20., 3.6
        option = mujoco.MjvOption()
        option.geomgroup[3], option.sitegroup[:] = 0, 0  # Hide historical single-target fixtures.
        side.update(camera=camera, option=option)

    def panel(name, index):
        side = sides[name]
        data, model = side["data"], side["model"]
        data.qpos[:] = side["qpos"][index]
        data.time = float(side["times"][index])
        mujoco.mj_forward(model, data)
        renderer = side["renderer"]
        renderer.update_scene(data, camera=side["camera"], scene_option=side["option"])
        for defender in ("a1", "a2"):
            for region in side["geometry"].regions:
                i = renderer.scene.ngeom
                if i >= renderer.scene.maxgeom:
                    raise ValueError("Insufficient visualization geometry capacity")
                mujoco.mjv_initGeom(renderer.scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([region.radius, 0., 0.]),
                    np.asarray(side["geometry"].world_center(region.name, defender)),
                    np.eye(3).reshape(-1), np.asarray(colors[region.name], dtype=np.float32))
                renderer.scene.ngeom += 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        im = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 44, side_width, 88), fill=(17, 24, 33))
        draw.text((8, 47), name.upper()+f" | native t={data.time:.3f}s", font=font, fill="white")
        draw.text((8, 68), "ACTUAL RECORDED QPOS | NOT TRAINED | 1x", font=small, fill=(255, 215, 135))
        counts = entry_counts(side["audit"], data.time)
        draw.rectangle((0, height-76, side_width, height), fill=(17, 24, 33))
        draw.text((8, height-73), "Triangle geometry entries (L / R / U), not match score", font=small, fill="white")
        for row, owner in enumerate(("a1", "a2")):
            values = " / ".join(str(counts[owner][r]) for r in REGIONS)
            draw.text((8, height-55+row*17), f"{owner} attacking opponent: {values}", font=small, fill=(175, 220, 250))
        state = str(side["summary"].get("status", "unknown"))
        failure = ",".join(side["summary"].get("failure") or [])
        draw.text((8, height-20), ("source: "+state+(" | "+failure if failure else ""))[:77], font=small, fill=(255, 170, 170) if failure else "white")
        return im

    def paired_frame(b_index, d_index):
        im = Image.new("RGB", (2*side_width, height))
        im.paste(panel("baseline", b_index), (0, 0))
        im.paste(panel("defended", d_index), (side_width, 0))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 2*side_width, 43), fill=(60, 23, 23) if schedule["truncated"] else (17, 24, 33))
        draw.text((8, 3), title, font=font, fill=(255, 200, 130))
        draw.text((8, 24), detail, font=small, fill="white")
        draw.line((side_width, 44, side_width, height), fill=(220, 220, 220), width=2)
        return im

    start_wall = time.monotonic()
    try:
        with mujoco.Renderer(sides["baseline"]["model"], height=height, width=side_width) as br, \
             mujoco.Renderer(sides["defended"]["model"], height=height, width=side_width) as dr:
            sides["baseline"]["renderer"], sides["defended"]["renderer"] = br, dr
            selected = set(np.linspace(0, manifest["frames"]-1, 5).round().astype(int))
            previews = []
            with imageio.get_writer(str(partial), fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-threads", "2"]) as writer:
                for number, (bi, di) in enumerate(zip(schedule["baseline_indices"], schedule["defended_indices"])):
                    im = paired_frame(int(bi), int(di))
                    writer.append_data(np.asarray(im))
                    if number in selected:
                        name = f"frame_{number:04d}.png"
                        im.save(output/name)
                        previews.append(name)
                    if progress and (number % 30 == 0 or number == manifest["frames"]-1):
                        progress(f"Paired replay {number+1}/{manifest['frames']} frames", flush=True)
            terminal = [int(nearest_indices(sides[name]["times"], [schedule["common_end_s"]])[0])
                        for name in ("baseline", "defended")]
            paired_frame(*terminal).save(output/"common_terminal_state.png")
        manifest["ffprobe"] = probe_video(partial, width=2*side_width, height=height,
                                          frames=manifest["frames"], fps=fps)
        if _digest(comparison_path) != comparison_hash or any(
                source_hashes(side["folder"]) != side["hashes"] for side in sides.values()):
            raise RuntimeError("Recorded source or comparison changed during replay")
        partial.replace(video)
        manifest.update(status="completed", video=str(video), source_unchanged=True,
                        previews=previews, terminal_preview="common_terminal_state.png")
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_time_s"] = time.monotonic()-start_wall
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--side-width", type=int, default=512)
    p.add_argument("--height", type=int, default=384)
    a = p.parse_args()
    print(json.dumps(render(a.experiment, a.case, a.output, fps=a.fps,
                            side_width=a.side_width, height=a.height, progress=print), indent=2))


if __name__ == "__main__":
    main()
