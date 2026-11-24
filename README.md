# RL-SwordNet — Plan 1

## Report entry points

- [English preliminary report draft](docs/report_draft_en.md): repository, description, tools, algorithms, planned work/experiments, expected results, preliminary work and ethics.
- [当前中文项目框架与报告说明](docs/plan1_zh.md)：最新 A1/A2 双 self-play 主线及截至 12 月 3 日的计划。
- [Current validation status and limitations](docs/skill_validation_zh.md): distinguishes geometric candidates, physical controls and work still required before RL.

Updated 22 November 2026. Use the current documents for the report; the [21 November plan](docs/plan1_20261121_zh.md) is retained as history and its fixed-script/shared-three-skill comparison is superseded. No trained policy or multi-directional defensive coverage is claimed.

The confirmed research goal is **two separately self-play-trained policies** on unchanged floor-fixed arms: A1 selects and times finite predefined skills, while A2 outputs continuous joint-target increments. They must share physical observations, native low-level control and scoring rules, then be frozen for cross-play. This architecture is not yet implemented or cleared for RL.

The current prerequisite is full-window **physical attack/parry validation**. Opt-in `whole_blade_first_touch_v2` scores the actual three finite blade cylinders without skill/phase labels; blade-to-guard contact is neutral legal weapon contact. The first-touch game result is separate from later diagnostic target entries, so a contacted parry cannot hide a delayed hit. Historical v1 rules and failed evidence remain unchanged. The manually authored shared three-skill fixture and earlier ASE conversion below are controls, not the final A1 bank or trained policies. See [acceptance requirements, evidence and reproduction](docs/skill_validation_zh.md).

- [中文动作定义、成对物理验证、时序边界及复现方法](docs/manual_duel_zh.md)
- [共享技能有效性验证与连续反击阻塞项](docs/skill_validation_zh.md)
- [完整 ASE 直刺和上拨挡的双机测试与适配结果](docs/ase_duel_zh.md)
- [中文方案：范围、算法、实验和截至 12 月 3 日的计划](docs/plan1_zh.md)
- [Longsword pose library: ten guards, nine cuts, geometry screen only](docs/hema_poses.md) (`python -m sim.hema_skills`), a separate A1 candidate library, not physically validated.
- [Skill library setup, format, and source notice](motion_skills/plan1/README.md)
- [Pinned source manifest](motion_skills/plan1/manifest.json)

This branch provides a bounded single-arm baseline, earlier source conversion/matching tests, and versioned two-arm native-contact skill diagnostics. **The references are not hardware-ready trajectories or trained policies.** Full-CAD collision validation, measured hardware limits, contact-safe continuous transitions, a complete training environment/rewards, PPO training and self-play remain pending. Third-party motion binaries and generated trajectories are not redistributed in Git.

## Longsword pose states (separate A1 library)

Ten stationary guards (action ids 0 to 9) and nine cuts (ids 10 to 18) defined in `sim/hema_skills.py`, documented in [docs/hema_poses.md](docs/hema_poses.md). Geometry screen only; nothing here is physically validated.

```bash
python -m sim.hema_skills                                   # solve + screen, writes outputs/hema_poses/summary.json
python -m sim.hema_skills --render                          # also writes PNG strips of every pose and cut to outputs/hema_poses/renders
python -m sim.hema_skills --view                            # play every pose and cut in the MuJoCo viewer (kinematic, no physics)
python -m sim.hema_skills --view roof_left straight_down    # play only the named states
python -m sim.hema_skills --path-space task                 # straight-line task-space cuts; all fail the collision screen, kept as evidence
python -m sim.hema_skills --config overrides.json           # JSON keys override DEFAULT_CONFIG (pose targets, cut chains, timing)
python -m unittest discover -s tests -p "test_hema_skills.py" -v
```

To use the states from Python:

```python
from sim.hema_skills import build_hema_library
library = build_hema_library()
library.poses["roof_left"]                 # exact joint vector (7,) of a stationary state
library.holds["roof_left"].sample(t)       # zero-velocity hold as a SkillPath
library.cuts["straight_down"].sample(t)    # joint targets t seconds into a cut; ends at the exact end pose
library.transitions[("roof_left", "ox_left")]   # 1.5 s joint-space move between any two states
library.action_ids                         # name -> 0..18
library.reports["library"]                 # pass/fail summary of the geometry screen
```

## Shared skill validation before RL

Historical v1 diagnostic:

```bash
python -m sim.skill_validation --output outputs/shared_new --library-config config/shared_skills_staged_candidate.json --offsets -6 -5 -3 -2.5 -2 -1.5 0 2 3
python -m sim.policy_validation --output outputs/policies_new --library-config config/shared_skills_staged_candidate.json --duration-s 60
MUJOCO_GL=egl python -m sim.skill_validation_replay --run outputs/shared_new --baseline a1_attack_guard --defended a1_parry_offset_-2.00 --output outputs/shared_new/comparison.mp4 --start-s 8 --end-s 17
```

Explicit whole-blade v2 diagnostic, with a 20 ms decision tick and neighbouring parry start samples:

```bash
python -m sim.skill_validation --output outputs/whole_blade_v2_new --library-config config/shared_skills_staged_candidate.json --rules whole_blade_first_touch_v2 --decision-period-s .02 --offsets -5 -2.1 -2.05 -2 -1.95 -1.9 2
```

Even a successful sampled window is not a universal parry guarantee. Every final A1 skill must have a real hitting Guard control, full-window defence evidence, timing/direction negative controls, and tested entry/exit transitions. Merely choosing the matching direction never grants immunity.

Use new output paths. The staged manual candidate keeps the 1.080 m sword, original arm dimensions, fixed targets, collision masks and debug torque caps unchanged. It separates upright preparation from sword rotation. Speed/tracking are diagnostics in this pose-first stage, not hardware certification. The shared manual builder's default deliberately retains the earlier simultaneous-preparation candidate; select its staged configuration explicitly. Every output includes failed cases; a successful scheduled parry is not a validated continuous game or autonomous-policy comparison.

## A1 ASE derived five skill geometry pilot

`A1` denotes the finite-skill policy family, not the left `a1` scene slot. A1-family policies may occupy both robot slots during their own self-play; this candidate library does not constrain A2's continuous joint-command action space. Configs and future exports explicitly carry `policy_family=A1`. The present candidates do not establish multi-directional defence or a trainable self-play environment.

This pilot requires the five exact source files listed in `config/ase_pilot_v2.json`, obtained separately under the source terms and kept in a local cache. Use `--source-dir PATH` for that cache; missing/corrupt files are rejected without automatic download. The historical `plan1_skills fetch` command obtains only its two older sources, not the new five-source pilot.

```bash
python -m sim.ase_pilot --config config/ase_pilot_v2_calibrated_candidate.json --output outputs/ase_pilot_new --ik
```

This local-cache-only candidate contains Ready, Stab, a horizontal slash and left/right parry seeds, with pinned source hashes, selected frames, authored seams and retiming recorded. The calibrated common mapping passes the sampled actual-model IK and strict proxy-contact screen for all five candidates. **Dynamic attacks, interceptions, joint-loop recovery and continuous switching are not certified; no candidate is admitted to training.** An independent joint-return diagnostic found geometrically clear paths, but some move the sword tip substantially and must be included in full-window defence tests. The original failed mapping and all calibration attempts are retained; see the [current evidence](docs/skill_validation_zh.md).

## Earlier role-specific manual attack and parry test

```bash
python -m sim.manual_duel --output outputs/manual_duel_new --matrix
MUJOCO_GL=egl python -m sim.manual_duel_replay --run outputs/manual_duel_new --output outputs/manual_duel_new/comparison.mp4
```

Use a new output path. Physics-only testing needs the existing requirements; optional MP4 rendering also needs `imageio` and `imageio-ffmpeg`. The replay uses recorded physics states, not command trajectories.

The default attack hits without defence, while the correctly timed active side-parry prevents the hit throughout 11.85 simulated seconds. Mirrored roles and a 1 ms timestep repeat the result; a deliberately late parry fails. Timing/speed sensitivity tests also contain genuine failures: this is a controlled example, **not universal defence or an RL win rate**. Original dimensions/gains/torque caps are unchanged; joint travel remains hypothetical. See the [full evidence and limitations](docs/manual_duel_zh.md).

## Earlier floor-fixed ASE matching test

The later full-clip physical experiment is described in [ASE duel testing](docs/ase_duel_zh.md): `python -m sim.ase_duel --output outputs/ase_original_new`. Both full clips play once with separate guard connections. The original display mapping misses the target; common spatial calibration produces a real native-clip thrust hit, but tested ParryUp timings have not established valid defence. This is geometry-preserving retimed playback, not source-speed imitation or a trained policy.

The current experiment has **no humanoid torso**. The complete original CAD base is fixed to the world in its original upright orientation and translated upward by 8 mm so its measured bottom meets the floor. The arm, transverse near-guard fixture and 1.080 m shinai are not scaled. The actual grip-to-tip distance is 0.790 m. Original XML/meshes and CAD inertias are preserved.

```bash
python -m sim.floor --inspect
```

That command displays the CAD-zero geometry only, not an executed motion or a validated reset-to-guard path. The older shoulder-mounted experiment below is retained for comparison and is not used by the floor test.

After obtaining the verified sources as described in the [skill library instructions](motion_skills/plan1/README.md), run:

```bash
python -m plan1_skills.floor_test --output outputs/floor_matching_round1
```

Use a new/empty output directory for each run. With a Linux EGL renderer, add `--render` and prefix the command with `MUJOCO_GL=egl`. An existing verified source cache can be selected with `--source-dir PATH`; no third-party runtime or unsafe pickle loading is required. To view the generated joint configurations locally:

```bash
python -m plan1_skills.view_match --run outputs/floor_matching_round1 --skill stab
python -m plan1_skills.view_match --run outputs/floor_matching_round1 --skill parry_up
```

These are explicitly labelled **kinematic previews**, not physics playback. Display duration does not specify robot execution timing.

The test applies one common mapping from [config/floor_match.json](config/floor_match.json) to Guard, Stab and ParryUp, using the real mounted grip/tip sites rather than an assumed end-effector axis. It preserves all 128 Stab and 58 ParryUp source frames and adds interpolation (509 and 229 samples respectively). It retains failed samples, checks adjacent joint changes and interpolated proxy contacts, constructs return-to-Guard connections, and reports static gravity demand against the existing provisional torque caps. The source hand displacement is adapted by the pre-existing factor of 0.5; sword and robot dimensions are unchanged.

The first-round full-frame IK screen found a candidate mapping under **hypothetical ±180° joint bounds**; the original tiny debug windows cannot contain it. This is not confirmation of real joint travel. The original animation timing also produces excessive reference joint speeds and must not be used directly. Results in `summary.json` distinguish pose fitting, primitive-contact screening, gravity demand and untested physics. A successful ParryUp fit does not establish that it intercepts an opponent's sword. The start-up motion from CAD zero to Guard remains unvalidated.

For a separate, genuinely forward-stepped **slow diagnostic**, not a kinematic preview:

```bash
python -m sim.floor_playback --input outputs/floor_matching_round1/hypothetical180 --skill both --output outputs/floor_playback_round1
```

This resets directly to Guard, then uses native position actuators with model-gravity feedforward through the **same force caps**. Original gains are preserved (`[2000,2000,2000,2000,200,200,200]`), without external generalized forces or body gravity cancellation. Each saved joint interval is retimed with a stop-to-stop quintic at provisional target bounds of 0.05 rad/s and 0.1 rad/s². The first candidate takes approximately 345 s for Stab and 275 s for ParryUp, including connections and holds, in **simulation time**. This intentionally slow check is not an approved training/combat speed. Output `summary.json` and `trajectory.csv` record actual tracking, complete actuator output, contacts and any failure. Real joint/motor constraints and fast dynamic feasibility remain open.

Local round completed on 2026-11-22: all 509 Stab and 229 ParryUp kinematic samples met the configured error thresholds under the hypothetical range profile (maximum grip fit error 0.100 mm and 0.078 mm, respectively). Both slow forward-step checks completed, with maximum joint tracking errors 0.00582 rad / 0.00363 rad and no proxy penetration, torque saturation or MuJoCo warnings. The checked-in tiny debug-window profile failed pose matching and is not evidence against the data: it was only intended for small joint demos. These results do not validate real joint travel, source-speed execution, full CAD contacts, start-up travel or opponent interception.

## Single-arm baseline

A CAD-derived, fixed-base arm model with seven simulated hinge joints (`J0`–`J6`). The first milestone is a model audit and a bounded, low-speed single-arm demonstration. The long-term project goal is two fixed robots and sim-to-real evaluation.

![preview](preview.png)

This repository is not yet an RL environment. The manual two-arm fixture supplies controlled contact diagnostics, with explicit v1 tip and v2 whole-blade rules; a complete training game, hardware calibration and autonomous policies remain pending. See [model status and remaining inputs](docs/model_status.md) for the original baseline; the earlier [ASE Plan 1](docs/plan1_20261121_zh.md) remains documented for comparison.

## Earlier shoulder-mounted shinai assembly (not the current floor setup)

The complete CAD base is side-mounted to the right shoulder of a fixed humanoid upper body. A short collar grips the shinai 31 mm below the guard, with its axis **parallel to the measured four-hole face**, not inserted along the face normal. The original CAD XML/meshes and arm scale are preserved. This is a **simulation fixture, not a manufacturing-ready clamp**; there are still seven DOFs and no grasp actuator.

```bash
python -m sim.inspect_mount --output outputs/mount_inspection
python -m sim.assembly --headless --output outputs/assembly
```

For a static viewer that stays open until closed, run `python -m sim.assembly --inspect`. For an interactive physics check, run `python -m sim.assembly`. To save images on a Linux EGL setup, use `MUJOCO_GL=egl python -m sim.assembly --headless --output outputs/assembly --snapshot`.

The default check initializes a near-horizontal, forward-pointing sword, selects a model-gravity-balanced J0 initial angle and completes a 10-second hold/small-J6 out-and-back under the existing provisional joint and torque bounds. It writes an independent scene, trajectory and pass/fail evidence to `outputs/assembly/`; it does not validate the CAD-zero-to-home path or fencing skills. Static inspection is not a physics test. When opening the generated XML directly, select the `assembly_home` keyframe; the Python entry point handles this initialization and gravity-compensated control. [中文装配说明、测量依据与限制](docs/assembly_zh.md) distinguishes measured interface dimensions from assumed fixture, mass and runtime values.

## Run the audit and demo

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m sim.audit --output outputs/model_audit.json
python -m sim.demo --headless --output outputs/joint_demo
```

The audit inspects the original `scene.xml`; it does not apply the demo configuration. The demo applies `config/debug.json` to the loaded model in memory and leaves the source XML unchanged.

To save a rendered snapshot, add `--snapshot` (requires a working MuJoCo rendering backend):

```bash
python -m sim.demo --headless --output outputs/joint_demo --snapshot outputs/joint_demo/preview.png
```

On a Linux server without a display, select an available backend before Python starts, for example `MUJOCO_GL=egl python -m sim.demo --headless --snapshot outputs/joint_demo/preview.png`. The physics-only headless run does not require a graphics context.

To open the interactive viewer, omit `--headless`:

```bash
python -m sim.demo --config config/debug.json --output outputs/joint_demo
```

The default demo lasts 30 simulated seconds: one second of initial hold, a four-second cosine out-and-back motion for each of the seven joints, then one second of final hold. Each motion has an amplitude of `0.025 rad` relative to the CAD pose. Physics runs at `0.002 s` and control at `0.01 s`.

Outputs are `trajectory.csv` and `summary.json`. They record joint position, velocity, trajectory target, actual actuator torque at each joint, and the world position of the **`end_effector_block` body origin**. That body origin is not a sword tip or a calibrated tool frame.

The demo uses the existing position actuators and offsets their targets to compensate for gravity computed by the model. The same actuator torque caps constrain both tracking and gravity compensation. This checks the configured simulation; it does not establish that the real motors can perform the motion.

## Debug limits

All values below are development settings, **not measured hardware specifications**:

| Setting | Default |
| --- | --- |
| Joint software window about the CAD pose | J0: ±0.10 rad; J1–J6: ±0.15 rad |
| Actuator torque caps, J0–J6 | `[5, 80, 15, 20, 3, 3, 2]` Nm |
| Target speed limit | 0.05 rad/s |
| Measured joint speed monitoring threshold | 0.5 rad/s |
| Joint tracking-error monitoring threshold | 0.1 rad |

The raw XML has no joint range or actuator torque limits. Use the configured demo for the bounded motion check; opening the raw XML in a viewer does not apply these settings.

## Files and checks

- `robot.xml`: the original seven-joint arm and CAD inertial parameters.
- `scene.xml`: the original arm scene, floor, and lighting.
- `meshes/`: CAD STL assets; the repository describes their units as meters.
- `config/debug.json`: provisional runtime settings for the low-speed demo.
- `sim/`: model-loading, audit, and demonstration tools.
- `docs/model_status.md`: assumptions, limitations, and required project inputs.

```bash
python -m unittest discover -s tests -v
```

For the original viewer-only workflow:

```bash
python -m mujoco.viewer --mjcf=scene.xml
```

On Windows, `view.bat` opens that original scene. It does not run the configured demo.

## Current model limitations

- The XML models seven independently actuated joints; the real independent degrees of freedom and any mechanical coupling still need confirmation.
- Moving mass is `26.532377 kg`, derived from the model's CAD inertial parameters. Printed-part masses, centers of mass, and inertias need calibration.
- `qpos = 0` is the CAD pose, not a confirmed encoder zero or vertical-sword pose.
- All original robot meshes are visual only. The separate assembly adds coarse collision proxies; full-CAD contact validation and scoring are not implemented.
- Raw actuator gains, damping, and armature are placeholders. Transmission ratios and measured motor behavior are not modeled.
- The original scene has no sword/tool sites. The separate assembly adds a proposed grip/tool transform, not a calibrated real installation. Source URDF, Pinocchio IK, perception, RL training and trained-policy evaluation are not included.
