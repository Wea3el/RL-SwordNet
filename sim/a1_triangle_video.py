"""Short, isolated candidate-motion previews; never a native fencing replay.

Each chapter uses its own checked reference endpoints. Chapter cuts are NOT
validated action transitions. No dynamics, learned policy or defence verdict is
rendered. Only sampled-reference-passing actions can appear in this video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .motion_spline import C2JointSpline
from .triangle_task import REGIONS, TriangleGeometry


REGION_COLORS = {
    "left": (0.18, 0.67, 1.0, 0.23),
    "right": (0.25, 0.9, 0.42, 0.23),
    "upper": (1.0, 0.57, 0.16, 0.23),
}


def display_times(duration: float, fps: int) -> np.ndarray:
    if (isinstance(duration, (bool, str, bytes)) or not np.isscalar(duration)
            or duration is None or not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("fps must be an integer in [1, 60]")
    return np.minimum(np.arange(int(np.ceil(duration * fps)) + 1) / fps, duration)


def geometry_from_report(report: dict) -> TriangleGeometry:
    value = report.get("geometry", report.get("scoring"))
    if not isinstance(value, dict):
        raise ValueError("report needs its own scoring or geometry snapshot")
    if not {"base_distance", "regions"}.issubset(value):
        raise ValueError("geometry snapshot needs base_distance and regions")
    return TriangleGeometry.from_mapping({key: value[key] for key in ("base_distance", "regions")})


def _duration(action: dict) -> float:
    try:
        times = np.asarray(action["knot_time_s"], dtype=float)
        value = float(times[-1] - times[0])
        return value if np.isfinite(value) and value > 0 else float("inf")
    except (KeyError, IndexError, TypeError, ValueError):
        return float("inf")


def select_chapters(report: dict, requested=None) -> dict:
    """One passing attack per region, then up to two distinct guard edges.

    Prefer shorter checked references for a compact overview, not higher combat
    performance. Missing regions are reported, not replaced by failed paths.
    """
    actions = report.get("actions")
    if not isinstance(actions, dict):
        raise ValueError("report needs an actions mapping")
    passed = {name: action for name, action in actions.items()
              if isinstance(action, dict) and action.get("sampled_reference_pass") is True}
    if requested is not None:
        if (isinstance(requested, (str, bytes)) or not requested
                or len(set(requested)) != len(requested)
                or any(name not in passed for name in requested)):
            raise ValueError("explicit chapters must be distinct passing reference names")
        selected_regions = {passed[name].get("target_region") for name in requested}
        missing = [name for name in REGIONS if name not in selected_regions]
        return {"selected_names": list(requested), "missing_attack_regions": missing,
                "all_three_regions_have_sampled_reference": not missing,
                "selection_rule": "explicit subset; all selected references passed; missing means not shown in this subset",
                "excluded_failed_actions": [name for name in actions if name not in passed],
                "passed_but_not_selected": [name for name in passed if name not in requested]}
    selected, missing = [], []
    for region in REGIONS:
        candidates = [name for name, action in passed.items()
                      if action.get("kind") == "attack" and action.get("target_region") == region]
        if candidates:
            selected.append(min(candidates, key=lambda name: (_duration(passed[name]), name)))
        else:
            missing.append(region)
    guards = [name for name, action in passed.items()
              if str(action.get("kind", "")).startswith("guard_transition")
              and isinstance(action.get("start_guard"), str)
              and isinstance(action.get("end_guard"), str)
              and action["start_guard"] != action["end_guard"]]
    seen = set()
    for name in sorted(guards, key=lambda name: (_duration(passed[name]), name)):
        edge = (passed[name]["start_guard"], passed[name]["end_guard"])
        if edge not in seen:
            selected.append(name)
            seen.add(edge)
        if len(seen) == 2:
            break
    return {
        "selected_names": selected,
        "missing_attack_regions": missing,
        "all_three_regions_have_sampled_reference": not missing,
        "selection_rule": "shortest passing reference per region, then up to two distinct passing guard edges",
        "excluded_failed_actions": [name for name in actions if name not in passed],
        "passed_but_not_selected": [name for name in passed if name not in selected],
    }


def load_curve(source: Path, name: str, action: dict) -> tuple[C2JointSpline, Path]:
    """Check the exact saved reference rather than resampling an unchecked path."""
    source = Path(source).resolve()
    if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name:
        raise ValueError("action name must be a simple filename stem")
    path = (source / f"{name}.npz").resolve()
    if path.parent != source:
        raise ValueError("action data must remain inside the source directory")
    if action.get("sampled_reference_pass") is not True:
        raise ValueError("failed or unchecked action cannot be rendered")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"knot_time_s", "knot_qpos", "time_s", "qpos"}
        if not required.issubset(arrays.files):
            raise ValueError("reference archive lacks checked arrays")
        curve = C2JointSpline(arrays["knot_time_s"], arrays["knot_qpos"])
        times, positions = np.asarray(arrays["time_s"]), np.asarray(arrays["qpos"])
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or not np.isclose(times[0], curve.knot_time_s[0], atol=1e-12, rtol=0)
                or not np.isclose(times[-1], curve.knot_time_s[-1], atol=1e-12, rtol=0)):
            raise ValueError("checked samples must cover the entire reference clock")
        if positions.shape != (len(times), 7) or not np.isfinite(positions).all():
            raise ValueError("checked qpos must be a finite Nx7 array")
        if not np.allclose(curve.sample(times), positions, atol=1e-12, rtol=0):
            raise ValueError("reconstructed reference differs from saved checked samples")
        for key, actual in (("knot_time_s", curve.knot_time_s), ("knot_qpos", curve.knot_qpos)):
            if key in action:
                expected = np.asarray(action[key], dtype=float)
                if expected.shape != actual.shape or not np.allclose(expected, actual, atol=1e-12, rtol=0):
                    raise ValueError("report and reference archive knots disagree")
    return curve, path


def render(source, output, *, fps=12, width=640, height=480, actions=None):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from .constrained_replay import probe_video

    display_times(1.0, fps)
    if type(width) is not int or type(height) is not int or width < 480 or height < 360:
        raise ValueError("integer dimensions of at least 480x360 are required")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("choose a new output path; preserve old evidence")
    report_path, scene_path = source / "report.json", source / "scene.xml"
    report = json.loads(report_path.read_text())
    if str(report.get("status", "")).startswith("building"):
        raise ValueError("source builder must finish before rendering")
    geometry = geometry_from_report(report)
    selection = select_chapters(report, actions)
    center = report.get("guards", {}).get("center", {})
    center_q = np.asarray(center.get("qpos_rad"), dtype=float)
    if center.get("static_reference_pass") is not True or center_q.shape != (7,) or not np.isfinite(center_q).all():
        raise ValueError("other arm requires a checked finite center guard")
    curves, paths = {}, [report_path, scene_path]
    for name in selection["selected_names"]:
        curves[name], path = load_curve(source, name, report["actions"][name])
        paths.append(path)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.mkdir(parents=True)
    manifest = {
        "status": "rendering", "kind": "isolated_kinematic_reference_NOT_native_replay",
        "native_dynamics_tested": False, "successful_defence_demonstrated": False,
        "trained_policy": False, "admitted_to_training": False,
        "source_sha256": hashes, "fps": fps, "width": width, "height": height,
        "selection": selection, "scoring_geometry_snapshot": geometry.to_mapping(),
        "other_arm": "a2 held at checked center guard", "chapters": [],
        "notes": [
            "All chapters are isolated previews; chapter cuts are not continuous combinations.",
            "All references retain their full clock at 1x speed, including their exact endpoint.",
            "Colored finite target spheres are display-only candidates, on both robots.",
            "Passing sampled FK references do not prove native tracking, attacks, or defence.",
            "Missing region coverage is reported and never filled using failed paths.",
        ],
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    save()
    if not curves:
        manifest["status"] = "no_passed_reference_to_render"
        save()
        raise ValueError("no passing attack or nontrivial guard transition; see manifest")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    joint_ids = {}
    for side in ("a1", "a2"):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_J{i}") for i in range(7)]
        if any(index < 0 for index in ids):
            raise ValueError("scene must contain a1_J0..J6 and a2_J0..J6")
        joint_ids[side] = model.jnt_qposadr[ids]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 14), ImageFont.truetype(font_path, 12)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [geometry.base_distance / 2, 0.0, 0.8]
    camera.azimuth, camera.elevation, camera.distance = 110, -18, 3.7
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    video = output / "a1_triangle_candidates.mp4"
    partial = output / "a1_triangle_candidates.partial.mp4"
    frame_count = 0

    def paint(renderer, name, curve, elapsed, chapter):
        data.qpos[joint_ids["a1"]] = curve.sample(curve.knot_time_s[0] + elapsed)
        data.qpos[joint_ids["a2"]] = center_q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)  # Deliberately no mj_step / physics integration.
        renderer.update_scene(data, camera=camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        for side in ("a1", "a2"):
            for region in geometry.regions:
                index = renderer.scene.ngeom
                if index >= renderer.scene.maxgeom:
                    raise ValueError("insufficient visualization geometry capacity")
                mujoco.mjv_initGeom(renderer.scene.geoms[index], mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([region.radius, 0.0, 0.0]), np.array(geometry.world_center(region.name, side)),
                    np.eye(3).reshape(-1), np.array(REGION_COLORS[region.name], dtype=np.float32))
                renderer.scene.ngeom += 1
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        action = report["actions"][name]
        target = action.get("target_region", "guard transition")
        draw.rectangle((0, 0, width, 95), fill=(15, 22, 30))
        draw.text((9, 4), "KINEMATIC REFERENCE | NOT TRAINED | NO DYNAMICS", font=font, fill=(255, 215, 130))
        draw.text((9, 24), "ISOLATED CHAPTER | NOT A VERIFIED DEFENCE", font=font, fill="white")
        draw.text((9, 44), f"{chapter}/{len(curves)}  {target} | {action.get('start_guard')} -> {action.get('end_guard')}", font=font, fill="white")
        draw.text((9, 66), f"1x reference time {elapsed:.2f}/{curve.duration_s:.2f}s | {name}", font=small, fill=(185, 225, 250))
        draw.rectangle((0, height - 54, width, height), fill=(15, 22, 30))
        draw.text((9, height - 49), "Other arm: FIXED CENTER GUARD. No score / defence proof.", font=small, fill="white")
        draw.text((9, height - 30), "Targets: blue LEFT | green RIGHT | orange UPPER (both robots)", font=small, fill=(185, 225, 250))
        return image

    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            with imageio.get_writer(str(partial), fps=fps, codec="libx264", quality=8,
                    macro_block_size=1, ffmpeg_params=["-threads", "2", "-movflags", "+faststart"]) as writer:
                for chapter, (name, curve) in enumerate(curves.items(), 1):
                    if chapter > 1:
                        separator = Image.new("RGB", (width, height), (15, 22, 30))
                        draw = ImageDraw.Draw(separator)
                        draw.text((20, height // 2 - 25), "NEXT ISOLATED REFERENCE", font=font, fill=(255, 215, 130))
                        draw.text((20, height // 2), "Chapter reset is not a validated motion transition.", font=small, fill="white")
                        for _ in range(max(1, round(0.35 * fps))):
                            writer.append_data(np.asarray(separator))
                            frame_count += 1
                    start = frame_count / fps
                    times = display_times(curve.duration_s, fps)
                    snapshots = []
                    for index, elapsed in enumerate(times):
                        image = paint(renderer, name, curve, float(elapsed), chapter)
                        writer.append_data(np.asarray(image))
                        frame_count += 1
                        if index in (0, len(times) - 1):
                            path = output / f"{chapter:02d}_{name}_{'start' if index == 0 else 'end'}.jpg"
                            image.save(path)
                            snapshots.append(str(path))
                    action = report["actions"][name]
                    manifest["chapters"].append({
                        "name": name, "kind": action["kind"], "target_region": action.get("target_region"),
                        "start_guard": action.get("start_guard"), "end_guard": action.get("end_guard"),
                        "isolated_preview": True, "inter_chapter_transition_validated": False,
                        "reference_duration_s": curve.duration_s, "video_start_s": start,
                        "video_end_s": frame_count / fps, "reference_frame_count": len(times),
                        "exact_endpoint_included": True, "snapshot_paths": snapshots,
                    })
                    save()
                    print(f"Rendered isolated candidate {chapter}/{len(curves)}: {name}", flush=True)
        manifest["ffprobe"] = probe_video(partial, width=width, height=height, frames=frame_count, fps=fps)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
            raise ValueError("source changed while rendering")
        partial.replace(video)
        manifest.update(status="completed", video=str(video), frame_count=frame_count,
                        duration_s=frame_count / fps, source_unchanged=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--actions", nargs="+", help="Explicit passing reference subset, in display order")
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output, fps=args.fps, width=args.width, height=args.height, actions=args.actions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
