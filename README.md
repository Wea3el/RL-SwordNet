# Arm MuJoCo model

A 7-joint robot arm converted from the Onshape "Global Assembly" URDF export to MuJoCo.

![preview](preview.png)

## Run it

```
python -m pip install mujoco
python -m mujoco.viewer --mjcf=scene.xml
```

On Windows you can double-click `view.bat` instead.

```python
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
data.ctrl[:] = 0.2          # target angle (rad) for J0..J6
mujoco.mj_step(model, data)
```

## Files

- `scene.xml`: load this one. Floor and lighting around the robot.
- `robot.xml`: the arm.
- `meshes/`: STL files from the CAD, in meters.

## Model

- Hinge joints `J0` (base rotation) to `J6` (end effector), each with a position actuator of the same name.
- Moving mass 26.5 kg, masses and inertias from the CAD.
- `qpos = 0` is the CAD pose.

## Limits

- No joint range limits.
- Meshes are visual only: no collisions.
- Actuator gains, joint damping and armature are placeholders, not motor data.
- Belts, pulleys and capstans are frozen; gear ratios are not modelled.
