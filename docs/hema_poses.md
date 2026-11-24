# Longsword pose library (separate A1 candidate library)

Updated 22 November 2026. Module: `sim/hema_skills.py`; tests: `tests/test_hema_skills.py`.

This is a **separate** finite action library built from the pose list supplied on 22 November 2026. It does not replace, share code paths with, or inherit validation from the shared three-skill fixture (`sim/shared_skills.py`) or the ASE-derived five-skill pilot (`sim/ase_pilot.py`). Every pose and cut is solved on the actual floor-fixed seven-joint arm and shinai, then screened with the same sampled proxy-collision, joint-margin and static-gravity checks used by the other libraries. **Passing that screen is geometric evidence only.** No cut has been shown to hit, no guard has been shown to parry, and nothing here has been run in the two-arm contact scene.

## Frame and body references

All numbers are in the robot's own base frame: +x toward the opponent, +y the robot's left, -y the robot's right, +z up. The floor-fixed arm has no head or torso, so "head height", "torso height" and "left/right of torso" are virtual reference values chosen so the poses are reachable and collision-free:

| Reference | Value |
| --- | --- |
| Side offset ("left/right of torso/head") | ±0.25 m from the base centre line |
| Head height (roof grip) | 1.45 m |
| Upper height (ox grip) | 1.35 m |
| Torso height (plow grip) | 0.80 m |
| Low height (fool grip) | 0.65 m |
| Over-the-head height (kron grip) | 1.55 m |
| Long point grip | 0.55 m forward, centred, 1.00 m high |

The shinai is 0.79 m from grip to tip. The floor plane collides with the shinai, so a true "point down" fool guard is impossible without the tip going 0.14 m under the floor. Both fool poses therefore point 40 degrees below horizontal, leaving the tip 0.14 m above the floor.

## Stationary poses (velocity 0)

Action ids 0 to 9. Each is a zero-velocity hold of one exact joint vector.

| Id | Pose | Grip (x, y, z) m | Blade direction | Deviation from the source list |
| --- | --- | --- | --- | --- |
| 0 | roof_left | (0.25, +0.25, 1.45) | up, 11° back | back tilt added so the grip clears the arm |
| 1 | roof_right | (0.25, -0.25, 1.45) | up, 11° back | same |
| 2 | fool_left | (0.30, +0.25, 0.65) | 40° below horizontal, forward | floor clearance, see above |
| 3 | fool_right | (0.30, -0.25, 0.65) | 40° below horizontal, forward | same |
| 4 | plow_left | (0.30, +0.25, 0.80) | forward, 20° up | point raised toward the opponent's upper target |
| 5 | plow_right | (0.30, -0.25, 0.80) | forward, 20° up | the source list says "left of torso"; read as a copy error and mirrored |
| 6 | ox_left | (0.20, +0.25, 1.35) | forward, 15° down | none |
| 7 | ox_right | (0.20, -0.25, 1.35) | forward, 15° down | none |
| 8 | long_point | (0.55, 0.00, 1.00) | forward, horizontal | blade roll is unconstrained (5-DoF IK) |
| 9 | kron | (0.25, 0.00, 1.55) | up, 30° back | the source list gave no orientation; this is an assumption |

All ten solve with position error below 0.2 mm and axis error below 0.05°, are free of sampled proxy contacts, keep at least 0.59 rad from the hypothetical ±π joint limits, and hold statically within the unchanged debug torque caps. Those caps are not hardware specifications.

## Dynamic cuts

Action ids 10 to 18. A cut is a continuous motion from its start pose, through the exact long point pose where one is listed, to its exact end pose, followed by a short hold. The source list says "any up position" or "any down position" may start a cut, so each diagonal cut is built from its canonical guard and once more from the other guard on the same side:

| Id | Cut | Chain |
| --- | --- | --- |
| 10 | downward_cut_left_to_right | roof_left → long_point → fool_right |
| 11 | downward_cut_left_to_right_from_ox_left | ox_left → long_point → fool_right |
| 12 | downward_cut_right_to_left | roof_right → long_point → fool_left |
| 13 | downward_cut_right_to_left_from_ox_right | ox_right → long_point → fool_left |
| 14 | upward_cut_right_to_left | fool_right → long_point → roof_left |
| 15 | upward_cut_right_to_left_from_plow_right | plow_right → long_point → roof_left |
| 16 | upward_cut_left_to_right | fool_left → long_point → roof_right |
| 17 | upward_cut_left_to_right_from_plow_left | plow_left → long_point → roof_right |
| 18 | straight_down | kron → long_point |

The source list only says the downward cuts pass through long point. The upward cuts are routed through long point as well; that is an assumption and is recorded in the config.

**Path construction.** The default `cut_path_space` is `joint`: a piecewise-linear joint path through the exact waypoint poses with one continuous quintic timing profile over the whole cut, so the blade does not stop at long point. Every seam (start pose, long point row, end pose) is an exact joint equality; no mid-skill reset is inserted. The alternative `task` mode draws a straight grip line with shortest-arc blade rotation and solves it with warm-started IK. In task mode **all nine cuts fail the collision screen** because the 5-DoF IK lets the shinai handle swing through the forearm proxies; that result is kept as evidence (`python -m sim.hema_skills --path-space task`), not hidden.

**Velocity requirement.** The source list says "velocity > 1". This is read as a tip speed of at least 1 m/s while passing long point. With the authored 0.6 s per segment, the tip passes long point at 2.2 to 3.6 m/s and the peak tip speed is 2.6 to 8.4 m/s. These are authored command timings under a provisional 6 rad/s command-speed cap and 10 ms control rounding, not measured or torque-feasible execution speeds. Actual cut durations are 2.1 s (two-segment cuts) and 0.9 s (straight down), each including a 0.3 s end hold.

## Transitions between stationary poses

All 90 ordered pose pairs have a 1.5 s joint-space quintic transition with exact seams. All 90 pass the sampled collision and joint-margin screen, which is also why the joint-space cut paths are collision-free. A cut's exact end pose is the exact start of every transition and cut that begins at that pose, so a policy can chain hold → cut → hold → cut without discontinuities.

## What this library does not establish

- No cut has been run against a target in the two-arm scene, so there is no hit evidence under `whole_blade_first_touch_v2`.
- No stationary guard has been tested as a parry against any cut, in any timing window.
- The joint-space cut paths are collision-free in the single-arm screen only; the tip path between waypoints is whatever the joint interpolation produces, not an authored blade line.
- Joint limits remain hypothetical ±π with a 0.05 rad IK margin; torque caps are debug values.
- "Left/right of head" and "over the head" are virtual heights, not a model of a human opponent.

The next gate is the same as for every other candidate library: paired physical attack-versus-guard and guard-versus-attack runs in the dual-arm scene before any pose or cut enters an A1 training action space.

## How to run the states

All commands run from the repository root with `requirements.txt` installed (MuJoCo 3.4, NumPy, Pillow).

```bash
python -m sim.hema_skills                                   # solve + screen; writes outputs/hema_poses/summary.json
python -m sim.hema_skills --render                          # also writes PNG strips to outputs/hema_poses/renders
python -m sim.hema_skills --view                            # play every pose and cut in the MuJoCo viewer
python -m sim.hema_skills --view roof_left straight_down    # play only the named states, looping
python -m sim.hema_skills --path-space task --output outputs/hema_poses_task   # straight-line cuts; all fail, kept as evidence
python -m sim.hema_skills --config my_overrides.json        # JSON keys override DEFAULT_CONFIG
python -m unittest discover -s tests -p "test_hema_skills.py" -v
```

- `summary.json` records every target, the reached grip/axis/tip, the solved joint vectors, per-cut tip paths and speeds, and all screen results including failures.
- `--render` writes `poses_row1.png`, `poses_row2.png` (five poses each) and `cut_<name>.png` (five frames from start to end of each cut), all from one fixed three-quarter camera.
- `--view` and `--render` play joint targets kinematically. Physics does not advance, so they show geometry, not tracking or contact.
- A `--config` JSON may replace `poses`, `cuts`, `cut_path_space`, timing values or the IK seed. Unknown pose names in a cut, non-unit blade axes and non-positive durations are rejected with an explicit error.

### Using the states from code

```python
from sim.hema_skills import build_hema_library, POSE_ORDER, CUT_ORDER

library = build_hema_library()              # ~4 s; solves on the actual arm every time, nothing is cached
q = library.poses["roof_left"]              # exact (7,) joint vector of a stationary state
hold = library.holds["roof_left"]           # SkillPath: hold.sample(t) == q for every t
cut = library.cuts["straight_down"]         # SkillPath: cut.sample(t) are joint targets t seconds after start
cut.duration_s                              # includes the 0.3 s end hold; cut.sample(t >= duration) == end pose
cut.phase_at(t)                             # segment name, e.g. "kron_to_long_point", "hold_long_point", or "ready"
move = library.transitions[("fool_right", "ox_left")]   # 1.5 s joint-space move between any two states
library.action_ids                          # {"roof_left": 0, ..., "kron": 9, "downward_cut_left_to_right": 10, ..., "straight_down": 18}
library.reports["cuts"]["straight_down"]    # geometry screen, tip speeds, long point passage
library.reports["library"]                  # all_poses_pass_geometry, all_cuts_pass_geometry, failed_* lists
```

Every `SkillPath` here is the same class the shared library uses, so `sample(elapsed)` returns joint targets for the native position actuators and never writes `qpos` directly. Chaining is exact: a cut's last row equals its end pose, which equals the first row of any transition or cut starting there.
