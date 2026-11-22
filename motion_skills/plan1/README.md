# Historical Plan 1 reference templates

This directory documents the original three-template source converter, retained for reproducibility. It is not the current A1 training library and does not constrain A2's continuous joint actions. See the [current project framework](../../docs/plan1_zh.md) and [report draft](../../docs/report_draft_en.md). These tools prepare references; they do not train PPO or establish effective defence.

The newer five-source A1 geometry pilot is implemented in `sim/ase_pilot.py`, with pinned filenames, hashes and sizes in `config/ase_pilot_v2.json`. After reviewing [the source notice](SOURCES.md), obtain those exact files separately from the pinned ASE source directory into a local cache, then pass `--source-dir PATH` to `python -m sim.ase_pilot`. The historical fetch command below downloads only Stab and ParryUp. The five-source pilot remains unvalidated for dynamic attack/parry and is not admitted to training.

| Action ID | Skill | Definition |
| --- | --- | --- |
| 0 | `guard` | Hold the first Stab reference pose; an engineered reference, not a separate captured guard motion |
| 1 | `stab` | Thrust reference from `RL_Avatar_Atk_Stab_Motion.npy` |
| 2 | `parry_up` | Sword up-parry reference from `RL_Avatar_Sword_ParryUp_Motion.npy`; blocking effectiveness is unvalidated |

## Get and prepare the library

Run from the repository root after installing `requirements.txt`. Python 3.10 or newer is required. The skill tools use NumPy and the standard library; they do not require Isaac Gym, PyTorch, SciPy, or the upstream ASE runtime.

```bash
python -m pip install -r requirements.txt
python -m plan1_skills list
```

Read the [source and usage notice](SOURCES.md) before downloading. If your intended use is permitted by the applicable terms:

```bash
python -m plan1_skills fetch --acknowledge-source-notice
python -m plan1_skills prepare
python -m plan1_skills inspect --skill stab --phase 0.5
```

`fetch` retrieves only two files from a pinned upstream commit and checks their byte counts and SHA-256 digests against [manifest.json](manifest.json). The acknowledgement flag records that the caller reviewed the notice; it neither grants rights nor certifies license compliance. No upstream Python is downloaded or executed.

`prepare` verifies the sources again, decodes the audited numeric structures without executing pickle, performs skeleton forward kinematics, and writes:

```text
outputs/plan1_skills/
  sources/         # two verified upstream .npy files
  templates/
    library.json   # provenance, processing conventions, skill metadata
    guard.npz
    stab.npz
    parry_up.npz
```

All downloaded and generated data stay in the ignored `outputs/` directory. Existing corrupt source files are rejected, not replaced. Existing nonempty template directories are preserved: use `--output outputs/plan1_skills/templates_v2` for a new build. For offline use, place the exact files in a local directory and run `prepare --source-dir PATH --output NEW_PATH`; network access is only needed by `fetch`.

## Coordinates, timing, and numeric format

The source skeleton has 17 nodes. The grip proxy is the `right_hand` skeletal origin, not a measured clamp point. The source sword node is rigidly 0.74 m from that hand origin; it is **not** the endpoint of this project's shinai.

Each clip uses a fixed frame based on its first shoulder line: +X points from the right shoulder toward the left shoulder, projected horizontally; +Z is source-world up; +Y is +Z cross +X. This is a mathematical frame, not an asserted anatomical forward direction. Root translation changes are removed from grip positions. Torso rotation is not removed.

Both clips reference the first Stab grip position in their respective fixed frames. Grip displacement is scaled by **0.5**; sword-axis direction is not scaled. Guard has zero displacement and holds the first Stab axis. This produces candidate fixed-base adaptations, not exact reproductions of the human animation. No robot-base rotation, tool mounting transform, absolute robot grip position, or shinai length has yet been applied.

The generated `.npz` files contain numerical arrays only and are loaded with `allow_pickle=False`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `phase` | `(N,)` | Strictly increasing normalized phase from 0 to 1 |
| `grip_delta_m` | `(N, 3)` | Scaled grip displacement relative to the shared reference, in the skill frame |
| `blade_direction` | `(N, 3)` | Unit grip-to-source-sword direction in the skill frame |
| `source_time_s` | `(N,)` | Original sampling time for provenance, not an approved playback clock; zeros for the synthetic Guard hold |

Stab contains 128 source frames and ParryUp 58. Guard has two identical endpoint poses. Source rates are approximately 30 Hz; no robot execution duration is approved. The sampler linearly interpolates grip displacement and uses unit-direction spherical interpolation. It does not define rotation around the sword axis. Time scaling, roll choice, continuous IK, skill transitions, and finite-torque execution must be implemented and validated separately.

```python
from pathlib import Path
from plan1_skills import SkillLibrary

library = SkillLibrary(Path("outputs/plan1_skills/templates"))
grip_delta_m, blade_axis = library.sample("parry_up", phase=0.5)
```

## Validation boundary

The checked-in arm has seven position actuators, no mounted sword, no calibrated grip site, no measured joint limits, and disabled robot contacts. The small bounded joint demo does not validate these templates. Earlier sparse IK diagnostics used hypothetical limits and mounting assumptions, as detailed in [the proposal](../../docs/plan1_zh.md#8-现有证据与尚未完成事项); they are not an executable skill acceptance test.

The separate [floor-mounted experiment](../../README.md#current-setup-floor-fixed-arm-and-first-ase-matching-test) now builds the arm with the proposed near-guard clamp and original-size shinai. `python -m plan1_skills.floor_test` performs full-frame actual-grip/axis IK and sampled diagnostics without changing this source-template contract; `python -m sim.floor_playback` is a separate slow finite-torque check. These are exploratory tests under labelled hypothetical joint ranges, not confirmation of hardware limits or a learned policy.

Before training, confirm real joint/actuator limits and the clamp transform, validate all frames and transitions at the intended execution speed, and verify collisions under MuJoCo dynamics. A ParryUp label alone does not demonstrate a successful defense. The planned PPO policy selects skills only after those checks; no trained policy, self-play runner, reward function, or scoring environment is provided here.

Run the offline checks, including the original single-arm tests, with:

```bash
python -m unittest discover -s tests -v
```

The source checksum and restricted decoder prevent executing downloaded pickle objects in this workflow. They are not a general security guarantee for arbitrary data files, nor a physical-safety certification. Use the checked-in manifest, not hashes supplied by an untrusted file.
