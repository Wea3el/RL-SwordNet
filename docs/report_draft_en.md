# RL SwordNet preliminary report draft

Status: 22 November 2026. This proposal covers the intended work through the final report on 3 December 2026. Planned methods and expected outcomes are distinguished from preliminary results. See the [current Chinese framework](plan1_zh.md) for implementation boundaries.

## Repository

[RL-SwordNet, plan1 branch](https://github.com/Wea3el/RL-SwordNet/tree/plan1).

## Project description

We are developing a simulated interactive sword-training target for a sporting and educational setting. The environment contains two fixed-base, seven-joint robot arms, each holding a bamboo shinai. The current configuration has no humanoid torso, mobile base or visual observation pipeline.

Our research question is how a predefined motion prior and action abstraction affect learning efficiency, offensive and defensive performance, and flexibility under a limited training budget. Both policy families will be trained through self-play. A1 will learn to select and time finite, validated motion skills, whereas A2 will output continuous joint-target increments and learn movement as well as tactical behaviour. We will freeze the resulting policies and evaluate them against one another.

This is not a comparison between a fixed scripted opponent and a learning policy, or between a trained imitation-learning expert and pure RL. Since the action spaces and motion priors differ, results will concern those design choices rather than demonstrate the superiority of an RL algorithm or historical fencing system.

## Tools and existing code

We use [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) for articulated dynamics, native actuation and weapon contact, together with Python, NumPy and project-specific forward/inverse kinematics and validation utilities. Local simulation checks use the MuJoCo 3.4 series. GitHub tracks code, configurations and documentation. PPO and its training dependencies are planned additions, not a completed training pipeline.

The [ASE project](https://xbpeng.github.io/projects/ASE/index.html) provides the Reallusion sword-and-shield animation sources used as candidate pose references for A1. We are not reproducing ASE's full learning framework or transferring a humanoid pretrained policy. The source animations do not provide this robot's joint commands, paired opponent states or successful-block labels.

Our conversion verifies source hashes, extracts a right-hand grip proxy and sword direction, and explicitly records spatial calibration, displacement adaptation, selected keyframes, interpolation and new timing. Offline tool-space IK maps these references to the actual robot. The physical robot and shinai are not resized to match the animation. Motion binaries and generated trajectories are not redistributed in this repository.

## Algorithms to be implemented

We plan to use [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), updating policies and value functions from simulated interaction rollouts. A1 uses a finite skill-selection output; A2 uses seven continuous joint-target increments. Both adapters will feed the same native low-level controller, joint constraints and torque limits. No in-episode state teleportation or scripted blade deflection is permitted.

A1 self-play uses two instances of the A1 policy family, with frozen historical A1 policies sampled as opponents. A2 is trained separately against historical A2 policies. Opponents remain frozen within an episode. A1 and A2 denote policy families, not fixed left/right robot slots. Fixed or random policies are diagnostic baselines, not self-play.

The planned common observation contains current own joint/control state and relative opponent weapon geometry and motion. Simulation state is used without requiring cameras; future trajectories, hidden opponent skill IDs and script clocks are excluded. This does not imply that a real robot's own encoders can observe its opponent.

Rewards will primarily reflect legal scores and episode outcomes, with modest control-effort or action-change penalties. Selecting a defensive pose, touching the opponent's blade or producing a larger impact will not automatically earn defensive reward. The final reward, foul attribution, reset/step interface and normalization are still to be implemented and tested against exploitative or inactive behaviour.

An explicit v2 scoring component already detects entry of the physical blade into unchanged virtual targets independently of skill labels. Blade-to-guard contact is neutral legal weapon contact. First touch decides a round; same-physics-step double touches draw. Invalid numerical or physical trials are not ordinary zero-point draws. Skill validation continues after the first touch to expose delayed hits during recovery; this scorer alone is not a complete RL environment.

## Planned work and experiments

First, we will complete and validate A1's finite library. The current geometric pilot has Ready, Stab, one slash and two source parry candidates. We plan to extend directional coverage, initially considering thrust, left/right slashes, a downward cut and corresponding defensive options. Source direction names and IK success do not certify effective interception.

Each attack must hit a Guard control. Each defensive claim requires physical interception before that control would be hit and no target entry across the full attack, defensive recovery and permitted transition. Timing, direction, perturbation and late-hit negative controls will be retained. Independent continuous-joint attacks must also test A1 coverage; A2 will not be restricted to A1's library.

After these gates, we will implement the shared training environment and both action adapters, then run the two self-play pipelines. Target milestones are skill/transition validation on 23–25 November, environment and simple-policy checks on 25–27 November, training on 27–30 November, and frozen evaluation on 1–2 December. These dates are targets contingent on validation, not completed deliverables.

The main evaluation will cross-play frozen A1 and A2 policies with swapped positions and independent evaluation seeds. We will report wins, losses, draws, scoring and defensive performance, invalid/foul rates, and learning curves. Budgets will be compared in simulated time and physics steps rather than raw macro-action counts. We aim for multiple independent training seeds and will disclose the actual number and uncertainty if resources are limited. Historical-opponent pools versus latest-only opponents and unseen timing variations are planned ablations where feasible.

Defensive performance will include target touches received per valid round and full-window protection in controlled paired attack tests, not counts of defensive skill labels. Invalid trials will be reported separately from draws; training-time foul attribution and consequences must be fixed before learning to prevent score-avoidance exploits.

## Expected results

We hypothesize that predefined skills may improve A1's early learning efficiency, while A2's continuous action space may permit more flexible solutions at a higher exploration cost. Neither advantage is assumed. Intended deliverables include a reproducible simulation, validated skills, training/evaluation materials and an analysis of limitations or negative results. Failed prerequisites will be reported rather than bypassed with scoring changes, restricted opponents or scripted demonstrations labelled as learned behaviour.

## Preliminary work completed

The original-scale CAD arm has been fixed to the floor with a transverse, near-guard shinai mount. The model uses a 1.08 m shinai, a 0.79 m grip-to-tip distance and a provisional 0.436 kg mass. Two-arm contact scenes, actual-tool kinematics, source verification and versioned scoring/validation code have been developed. Hardware limits, CAD mass/inertia corrections and flexible contact remain uncalibrated.

Controlled native-dynamics tests of an earlier manually authored attack/parry pair established a real hitting attack and successful defensive timing samples on one attack line, including mirrored positions and an independent 1 ms repeat. They did not establish multi-directional defence. A paired failure also showed that a successful parry followed by immediate counterattack could expose the defender to a delayed hit, demonstrating why complete transitions must be tested.

The new five-skill ASE-derived A1 pilot passes sampled actual-model IK and strict proxy-geometry checks under a common calibration. It is not dynamically validated or admitted to training. End-effector return does not necessarily return all joints to the same Guard; geometrically clear joint-return candidates can still move the blade significantly. These portions must be included in subsequent physical attack/parry validation. Detailed status is recorded in [skill validation](skill_validation_zh.md); generated local evidence is not included as a public experiment archive in this commit.

No PPO/self-play policy has yet been trained. There are no comparative win rates or convergence claims.

## Ethics and limitations

The project is limited to educational and sporting simulation and acknowledges the dual-use nature and impact risks of an armed robot. A blunt shinai, an ArUco marker or a proposed impedance controller does not establish safe human contact. Physical deployment would require additional sensing, calibrated dynamics, force limits, emergency stops and validated compliance; these are not completed results.

The [Reallusion motion notice](https://github.com/nv-tlabs/ASE/blob/main/ase/data/motions/reallusion_sword_shield/README.txt) restricts the data to noncommercial use. We retain ASE/Reallusion attribution and do not infer motion redistribution rights from the code license.
