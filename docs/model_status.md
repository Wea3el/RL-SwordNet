# Model status and next inputs

Status date: **2026-11-21**. This document distinguishes properties of the checked-in model from assumptions about the physical robot. It does not report training results or hardware validation.

## Confirmed baseline

The repository contains a MuJoCo arm scene, CAD mesh assets, and a viewer preview. The original README describes a conversion from an Onshape "Global Assembly" URDF export. The source URDF is not included in this baseline, so that conversion history does not establish agreement between an available URDF and the current MJCF.

The original `scene.xml` and `robot.xml` were loaded successfully using MuJoCo 3.4.0 for a static check. This check established:

| Property | Checked-in model |
| --- | --- |
| Generalized positions / velocities / actuators | 7 / 7 / 7 |
| Joint names | J0, J1, J2, J3, J4, J5, J6 |
| Joint type | Seven hinge joints in a serial chain |
| Base | Fixed to the world; no floating-base joint |
| Equality constraints | None |
| Moving-body mass sum | 26.532377 kg |
| Robot mesh assets / geom instances | 125 / 140 |
| Mesh asset availability | All referenced mesh files present |
| Physics step at load | 0.002 s |
| Gravity at load | `[0, 0, -9.81]` m/s² |
| Integrator | `implicitfast` |
| Initial joint configuration | Zero joint coordinates, representing the CAD pose |
| Robot contact | Disabled for every robot geom |
| Joint ranges in the original XML | Unspecified |
| Actuator control / force limits in the original XML | Unspecified |
| Sword, tool sites, encoder sensors, cameras | Not present |

Every moving body has explicit mass, center of mass, and principal inertia data. The principal moments are positive and satisfy the inertia triangle inequalities. Those are mathematical consistency checks, not evidence that the values match the manufactured robot. Printed-part density assumptions remain to be corrected.

J0 is a hinge relative to the world, not a free base. In the CAD pose its world axis points along `-Z`. The other joint axes are specified in their local body frames; their signs and encoder conventions still need checking against the mechanism.

## Single-arm demonstration boundary

The first demonstration checks model loading, bounded joint commands, and low-speed tracking. It reads joint positions and velocities directly from MuJoCo as ideal state feedback. This is enough for the initial single-arm motion check; it does not simulate encoder noise, delay, or velocity estimation.

The runtime debug configuration adds software joint windows, actuator torque caps, command speed limits, and monitoring thresholds. These are provisional development choices. In particular, a low J0 debug cap is not a measurement of the weak J0 motor mentioned in the project discussion.

| Runtime parameter | Debug value |
| --- | --- |
| Joint window about zero | J0 ±0.10 rad; remaining joints ±0.15 rad |
| Joint torque caps, J0–J6 | `[5, 80, 15, 20, 3, 3, 2]` Nm |
| Excursion from zero | 0.025 rad, one joint at a time |
| Per-joint out-and-back duration | 4 simulated seconds |
| Initial / final hold | 1 simulated second each |
| Total duration | 30 simulated seconds |
| Physics / control step | 0.002 / 0.01 s |
| Target speed limit | 0.05 rad/s |
| Measured speed monitoring threshold | 0.5 rad/s |
| Tracking-error monitoring threshold | 0.1 rad |

The controller uses the native position actuators. It adds a target offset corresponding to the model's computed gravity torque, then applies the same output torque cap to the combined actuator command. Successful tracking therefore depends on this model and these chosen limits. It does not confirm motor sizing or sim-to-real transfer.

The audit command intentionally inspects the original XML without these runtime overrides. The demo changes the loaded model in memory and preserves the checked-in XML. This allows the raw export and the configured demonstration to be reviewed separately.

The logged endpoint is the world position of the `end_effector_block` body origin. There is no sword tip or calibrated tool center point in the model. All plots, reports, and downstream code should retain that distinction.

### Verified debug run on 2026-11-21

Using MuJoCo 3.4.0 and the checked-in `config/debug.json`, the 30-second headless joint demonstration completed with 3,001 logged control samples and no MuJoCo warnings. The maximum observed tracking error across all joints was approximately **0.002145 rad (0.123 degrees)**. The largest observed joint speed was approximately **0.021762 rad/s**, and the largest actuator torque was approximately **49.30 Nm at J1**, below its provisional 80 Nm debug cap. These are simulation measurements from the CAD-mass model with model-gravity feedforward; they do not validate the real actuators or mass properties.

The command `MUJOCO_GL=egl python -m sim.demo --headless --output outputs/joint_demo --snapshot outputs/joint_demo/preview.png` also produced an offscreen arm image. Logs and the generated image are stored under the ignored `outputs/` directory. Re-run the commands to obtain evidence for the current configuration rather than treating these values as permanent acceptance criteria.

Automated checks cover the full default motion, force-cap failure injection, command slew limiting without teleporting the joint state, configuration validation, joint-name mapping, and replacement of stale success evidence after an injected simulator error. They also verify that the source model still has no sword/tool sites or robot collision geometry, so those features cannot be inferred from a passing arm demonstration.

## Why actuator inputs still matter

The raw XML gives J0–J3 a position gain of 2000 and J4–J6 a gain of 200; all joints share damping 1 and armature 0.01. The README identifies these as placeholders. There is no modeled transmission ratio or actuator force cap in the original XML.

For example, the static model at zero position and velocity produces `[400, 400, 400, 400, 40, 40, 40]` Nm if all raw position-actuator targets are changed instantly to 0.2 rad. This is a consequence of the configured gains, not a motor specification. The demonstration instead uses small smooth excursions and explicit debug caps.

The current CAD inertia model also gives J1 a gravity bias of approximately -47.43 Nm at the zero configuration. That calculation is useful for checking the simulation setup, but depends on uncalibrated mass properties and should not be treated as a measured holding requirement.

## Inputs needed from the team

| Input | What needs confirmation | Work it enables |
| --- | --- | --- |
| Joint / DOF table | Real independent inputs, active and passive joints, coupling, axes, zero conventions, travel limits | Correct articulation, command bounds, encoder mapping |
| Actuator table | Motor models, transmission ratios, output-side continuous and peak torque, speed limits, control mode and update rate | Credible actuator behavior and hardware-feasible commands |
| Printed-part mass mapping | Leon's slicer estimates matched to the parts in each moving link, with consistent print settings | Revised link masses and corresponding inertia estimates |
| Remaining inertial data | Nonprinted component masses, link centers of mass, and inertia estimation method | More credible gravity and acceleration loads |
| Sword installation | Grip location, fixed mounting transform, dimensions, mass, center of mass, inertia estimate | A mounted sword body and calibrated tool sites |
| Initial sword pose | Desired world direction and a reachable joint configuration | A validated vertical-sword initial state |
| Source robot description | The matching URDF, mesh paths, joint naming and version | Pinocchio model loading and cross-checks against MuJoCo |
| Cartesian command contract | Position only or full pose; world/base frame; TCP; command rate and increments | An unambiguous policy-to-IK interface |
| Match layout and rules | Base separation and orientations, permitted contacts, target regions, episode and scoring rules | Two-robot scene and evaluation environment |

The minimum remaining mechanical inputs for a meaningful low-speed model are confirmation of the joint structure, conservative travel bounds, and the sword mounting transform. The unmodified arm can be audited while these are collected. Realistic training and hardware-feasibility claims require actuator and inertial calibration as well.

## Cartesian policy and Pinocchio IK

The planned interface is a Cartesian policy command converted by Pinocchio IK into a joint target. This controller is not implemented in the current baseline, and it is not required to validate the initial joint-space demo.

Before connecting it, both model representations must agree on joint order, joint signs, zero offsets, base frame, and tool frame. A fixed transform from `end_effector_block` to the actual grip or sword TCP must be defined. IK should use the current joint configuration as its initial solution, preserve motion continuity, respect limits, and handle unreachable targets and singular configurations explicitly.

A reachable IK solution is only a kinematic result. Velocity and actuator limits still determine whether the motion can be executed. The weak J0 observation makes that distinction particularly relevant, but no measured J0 limit is available yet.

For a later two-robot policy, own-arm joint feedback does not reveal the opponent's state. The team must choose whether early experiments use simulator ground truth for the opponent and how that interface will later be replaced by perception. No camera is required for the present single-arm import milestone.

## Next milestones

1. Keep a reproducible audit of the original export and run the bounded single-arm joint demo with logged state and actuator output.
2. Confirm actual DOFs, travel bounds, transmission behavior, and actuator specifications; replace the provisional limits and calibration assumptions.
3. Add the mounted sword, named grip/tip sites, a validated initial pose, and simplified collision geometry with suitable exclusions.
4. Validate Pinocchio forward kinematics against MuJoCo, then test low-speed Cartesian trajectories through IK.
5. Define the two fixed-robot layout, contact and scoring rules, observations, actions, and episode termination conditions.
6. Add and evaluate training only after those interfaces and model checks are established. Historical-motion imitation is an optional extension, not a dependency of the first environment.

## Preliminary report checklist

For the report due **2026-11-22**, document the repository link, current tools, proposed algorithms, planned experiments, expected results, and ethics discussion. Clearly label which parts have been implemented, which have only been proposed, and which depend on missing inputs.

The available baseline supports a description of the CAD-derived single-arm model and its audit. Any completed demo results should be quoted from the generated `summary.json` and trajectory data, including the runtime configuration used. Do not present planned RL training, successful sword contact, dual-robot performance, hardware transfer, or expected win rates as observed results.

The ethics discussion can state the project's educational and sporting purpose, the limitations of simulation-based evidence, and the dual-use concerns raised in the project meeting. A substantive research conclusion requires actual experimental evidence.
