"""Video of an exported native trial, using recorded actual qpos only.

No policy, target interpolation or physics stepping is performed. Display
frames use the nearest recorded physical state and retain a source hash audit.
An MP4 is only given its final name after the encoder closes and ffprobe passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np


def source_hashes(trial):
    trial = Path(trial)
    return {name: hashlib.sha256((trial/name).read_bytes()).hexdigest()
            for name in ("scene.xml", "trace.npz", "summary.json")}


def load_recorded_states(trial, nq):
    trial = Path(trial)
    summary = json.loads((trial/"summary.json").read_text(encoding="utf-8"))
    with np.load(trial/"trace.npz", allow_pickle=False) as archive:
        times, qpos = archive["time_s"].copy(), archive["qpos"].copy()
    if (times.ndim != 1 or len(times) < 2 or qpos.shape != (len(times), nq)
            or not np.isfinite(times).all() or not np.isfinite(qpos).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Finite recorded actual qpos on strictly increasing times required")
    return summary, times, qpos


def frame_schedule(times, *, fps=30, start_s=None, end_s=None):
    """Nearest actual samples at real-time video intervals, without interpolation."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Strictly increasing finite physical timestamps required")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("Integer display FPS in [1,60] required")
    start = float(times[0]) if start_s is None else float(start_s)
    end = float(times[-1]) if end_s is None else float(end_s)
    if not np.isfinite([start, end]).all() or not times[0] <= start < end <= times[-1]:
        raise ValueError("Replay excerpt must lie within the recorded time window")
    requested = start + np.arange(int(np.ceil((end-start)*fps-1e-10)))/fps
    right = np.searchsorted(times, requested, side="left").clip(1, len(times)-1)
    left = right-1
    indices = np.where(requested-times[left] <= times[right]-requested, left, right)
    return requested, indices


def probe_video(video, *, width, height, frames, fps):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate,duration",
        "-of", "json", str(video)], check=True, capture_output=True, text=True, timeout=30)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one decodable video stream")
    stream = streams[0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    if (stream["width"] != width or stream["height"] != height
            or int(stream["nb_read_frames"]) != frames
            or abs(numerator/denominator-fps) > 1e-6
            or abs(float(stream["duration"])-frames/fps) > .002):
        raise ValueError("Decoded video dimensions, duration, FPS or frame count disagree")
    return stream


def render_trial(trial, output, *, fps=30, width=960, height=640, start_s=None,
                 end_s=None, label="Native physical trial", progress=None):
    """Render saved evidence without simulation; preserve existing outputs."""
    trial, output = Path(trial).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Existing replay directory is preserved: {output}")
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in (width, height))
            or width < 320 or height < 240 or width % 2 or height % 2):
        raise ValueError("Even video dimensions of at least 320 x 240 required")
    if not isinstance(label, str) or not label or len(label) > 100:
        raise ValueError("A short descriptive label is required")
    hashes = source_hashes(trial)
    model = mujoco.MjModel.from_xml_path(str(trial/"scene.xml"))
    summary, times, qpos = load_recorded_states(trial, model.nq)
    requested, indices = frame_schedule(times, fps=fps, start_s=start_s, end_s=end_s)
    output.mkdir(parents=True)
    temporary, video = output/"native_replay.partial.mp4", output/"native_replay.mp4"
    manifest = {"status": "rendering", "source_trial": str(trial),
        "source_sha256": hashes, "label": label, "fps": fps,
        "width": width, "height": height, "frame_count": len(indices),
        "sampling": "nearest recorded actual qpos; no interpolation, target playback or mj_step",
        "source_start_s": float(times[indices[0]]), "source_end_s": float(times[indices[-1]]),
        "video_duration_s": len(indices)/fps,
        "maximum_sampling_error_s": float(np.max(np.abs(times[indices]-requested))),
        "source_trial_status": summary.get("status"), "not_trained_policy": True}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.875, 0., .80]
    camera.azimuth, camera.elevation, camera.distance = 115., -20., 3.6
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[0] = 1
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    hit_times = [float(e["time_s"]) for e in summary.get("all_target_entries", [])]
    start_wall = time.monotonic()

    def frame(renderer, index):
        data.qpos[:] = qpos[index]  # Exactly a logged state, never the command.
        data.time = float(times[index])
        mujoco.mj_forward(model, data)  # Offline geometry evaluation, no mj_step.
        renderer.update_scene(data, camera=camera, scene_option=option)
        # Shadows are display-only and expensive on software renderers.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        im = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, width, 66), fill=(17, 24, 33))
        draw.text((12, 6), "NATIVE REPLAY | ACTUAL RECORDED QPOS | NOT TRAINED", font=font, fill=(255, 215, 135))
        draw.text((12, 35), f"{label} | simulation t = {times[index]:.3f} s | 1x speed", font=small, fill="white")
        draw.rectangle((0, height-53, width, height), fill=(17, 24, 33))
        entries = sum(t <= times[index]+1e-10 for t in hit_times)
        draw.text((12, height-47), f"Recorded target entries so far: {entries} | source status: {summary.get('status', 'unknown')}", font=small, fill=(175, 220, 250))
        draw.text((12, height-25), "Replay is visual evidence only; full-window validity comes from the saved trial and paired tests.", font=small, fill="white")
        return im

    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            with imageio.get_writer(str(temporary), fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-threads", "2"]) as writer:
                preview_slots = set(np.linspace(0, len(indices)-1, 5).round().astype(int))
                previews = []
                for number, index in enumerate(indices):
                    im = frame(renderer, int(index))
                    writer.append_data(np.asarray(im))
                    if number in preview_slots:
                        name = f"frame_{number:04d}_{times[index]:06.3f}s.png"
                        im.save(output/name)
                        previews.append(name)
                    if progress is not None and (number % 30 == 0 or number == len(indices)-1):
                        progress(f"Replay: {number+1}/{len(indices)} frames, {time.monotonic()-start_wall:.1f}s wall", flush=True)
                # Include the exact terminal state in the stills, even though
                # the last video interval starts one display tick before it.
                terminal = frame(renderer, int(np.searchsorted(times, end_s if end_s is not None else times[-1])))
                terminal.save(output/"terminal_state.png")
        manifest["ffprobe"] = probe_video(temporary, width=width, height=height, frames=len(indices), fps=fps)
        if hashes != source_hashes(trial):
            raise RuntimeError("Source evidence changed during rendering")
        temporary.replace(video)
        manifest.update(status="completed", video=str(video), previews=previews,
                        terminal_preview="terminal_state.png", source_unchanged=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_time_s"] = time.monotonic()-start_wall
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial")
    parser.add_argument("output")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--label", default="Native physical trial")
    args = parser.parse_args()
    print(json.dumps(render_trial(args.trial, args.output, fps=args.fps, width=args.width,
        height=args.height, start_s=args.start, end_s=args.end, label=args.label, progress=print), indent=2))


if __name__ == "__main__":
    main()
