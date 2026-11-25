"""Clearly labelled, full-duration kinematic previews of A1 reference curves.

No mj_step, controller, learned policy, official score or defence verdict is
used. Native-replay rendering remains separate and is never reused as a label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT
from .constrained_replay import probe_video
from .motion_spline import C2JointSpline


LABELS = {
    'thrust_low_left': 'Low-left thrust',
    'thrust_low_right': 'Low-right thrust',
    'thrust_mid_right': 'Mid-right thrust',
    'sweep_low_left_to_right': 'Low sweep: attacker left to right',
    'sweep_mid_right_to_left': 'Mid sweep: attacker right to left',
    'intercept_left': 'Left interception candidate',
    'intercept_right': 'Right interception candidate',
    'intercept_center_high': 'Upper interception candidate',
}


def display_times(duration, fps):
    if not np.isfinite(duration) or duration <= 0 or type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError('Positive duration and integer FPS in [1,60] required')
    # Keep the exact endpoint, without ever speeding up the reference clock.
    return np.minimum(np.arange(int(np.ceil(duration*fps))+1)/fps, duration)


def render(source, output, *, fps=20, width=960, height=640):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new preview directory; preserve old evidence')
    report_path, scene_path = source/'report.json', source/'scene.xml'
    report = json.loads(report_path.read_text())
    accepted = [(n,a) for n,a in report['actions'].items() if a['continuous_reference_candidate_pass']]
    if not accepted or any(n not in LABELS for n,_ in accepted):
        raise ValueError('Known labelled reference candidates required')
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'a1_J{i}') for i in range(7)]
    qids = model.jnt_qposadr[joint_ids]
    other = model.jnt_qposadr[[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'a2_J{i}') for i in range(7)]]
    scoring_path = ROOT/'config/scoring_spec_v3_proposal.json'
    scoring = json.loads(scoring_path.read_text())
    output.mkdir(parents=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [report_path,scene_path,scoring_path]}
    manifest = {'status':'rendering','kind':'kinematic_reference_preview_NOT_native_replay',
        'native_dynamics_tested':False,'successful_defence_demonstrated':False,'trained_policy':False,
        'source_sha256':hashes,'fps':fps,'width':width,'height':height,'chapters':[],
        'excluded_failed_actions':[n for n,a in report['actions'].items() if not a['continuous_reference_candidate_pass']],
        'notes':['Each chapter is an isolated reference demonstration, not a self-selected combination.',
                 'Only a1 moves; a2 is held at ready. No moving attack tests interception.',
                 'Translucent regions are display-only candidates, not deployed scoring or official points.']}
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    save()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.875,0.,.82]
    camera.azimuth, camera.elevation, camera.distance = 110., -18., 3.7
    option = mujoco.MjvOption(); option.geomgroup[3] = 0; option.sitegroup[:] = 0
    video, partial = output/'a1_reference_overview.mp4', output/'a1_reference_overview.partial.mp4'
    frame_count = 0
    def paint(renderer, name, curve, t, chapter, phase):
        data.qpos[qids] = curve.sample(t)
        data.qpos[other] = curve.knot_qpos[0]
        data.qvel[:] = 0
        mujoco.mj_forward(model,data)  # Static FK only. Never physical integration.
        renderer.update_scene(data,camera=camera,scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        for sphere in scoring['targets']['a2']:
            index = renderer.scene.ngeom
            if index >= renderer.scene.maxgeom:
                raise ValueError('Insufficient visualization geometry capacity')
            mujoco.mjv_initGeom(renderer.scene.geoms[index],mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([sphere['radius_m'],0.,0.]),np.asarray(sphere['center_world_m'],float),
                np.eye(3).reshape(-1),np.array([.2,.8,.95,.20],dtype=np.float32))
            renderer.scene.ngeom += 1
        im = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0,0,width,78),fill=(15,22,30))
        draw.text((12,5),'KINEMATIC REFERENCE | NO DYNAMICS | NOT TRAINED',font=font,fill=(255,215,130))
        draw.text((12,31),f'{chapter}/{len(accepted)}  {LABELS[name]}',font=font,fill='white')
        draw.text((12,56),f'1x reference speed | t = {t:.2f} / {curve.duration_s:.2f} s | {phase}',font=small,fill=(180,225,250))
        draw.rectangle((0,height-59,width,height),fill=(15,22,30))
        draw.text((12,height-53),'Other arm: FIXED READY. Interception motion is NOT a verified block.',font=small,fill='white')
        draw.text((12,height-31),'Blue region: scoring proposal only. No points or defence success awarded.',font=small,fill=(180,225,250))
        return im
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            with imageio.get_writer(str(partial),fps=fps,codec='libx264',quality=8,
                    macro_block_size=1,ffmpeg_params=['-threads','2','-movflags','+faststart']) as writer:
                for chapter,(name,action) in enumerate(accepted,1):
                    path = source/f'{name}.npz'; hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as arrays:
                        curve = C2JointSpline(arrays['knot_time_s'],arrays['knot_qpos'])
                        if not np.allclose(curve.sample(arrays['time_s']),arrays['qpos'],atol=1e-12,rtol=0):
                            raise ValueError('Reference reconstruction differs from saved checked samples')
                    start = frame_count/fps; times = display_times(curve.duration_s,fps)
                    for index,t in enumerate(times):
                        phase = ('START' if t==0 else 'RETURNED TO READY' if t==curve.duration_s else 'REFERENCE MOTION')
                        im = paint(renderer,name,curve,float(t),chapter,phase)
                        writer.append_data(np.asarray(im)); frame_count += 1
                        if index in {0,int(np.argmin(np.abs(times-4.))),len(times)-1}:
                            im.save(output/f'{chapter:02d}_{name}_{index:04d}.jpg')
                    for _ in range(round(.4*fps)):
                        writer.append_data(np.asarray(im)); frame_count += 1
                    manifest['chapters'].append({'name':name,'label':LABELS[name],
                        'video_start_s':start,'video_end_s':frame_count/fps,'reference_duration_s':curve.duration_s,
                        'reference_frame_count':len(times),'end_hold_s':.4,'exact_endpoint_included':True})
                    save(); print(f'Rendered {chapter}/{len(accepted)}: {name}',flush=True)
        manifest['ffprobe'] = probe_video(partial,width=width,height=height,frames=frame_count,fps=fps)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
            raise ValueError('Source changed while rendering')
        partial.replace(video)
        manifest.update(status='completed',video=str(video),frame_count=frame_count,
                        duration_s=frame_count/fps,source_unchanged=True)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/a1_plan2_continuous_v1')
    p.add_argument('--output',required=True)
    a=p.parse_args(); print(json.dumps(render(a.source,a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
