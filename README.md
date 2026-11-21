# Arm MuJoCo model

A CAD-derived, fixed-base arm model with seven simulated hinge joints (`J0`–`J6`). The first milestone is a model audit and a bounded, low-speed single-arm demonstration. The long-term project goal is two fixed robots and sim-to-real evaluation.

![preview](preview.png)

This repository is not yet a sword-fighting or RL environment. The sword, contact rules, hardware limits, and Cartesian/Pinocchio controller are pending. See [model status and remaining inputs](docs/model_status.md) for the confirmed state as of **2026-11-21**.

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
- All robot meshes are visual only. Robot self-contact, robot-floor contact, sword contact, and scoring are not implemented.
- Raw actuator gains, damping, and armature are placeholders. Transmission ratios and measured motor behavior are not modeled.
- No sword, calibrated tool frame, source URDF, Pinocchio IK, perception, RL training, or trained-policy evaluation is included.
