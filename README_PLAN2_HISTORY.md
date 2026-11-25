# RL-SwordNet Plan 2

Two original-size, floor-fixed, seven-joint robot arms holding 1.080 m shinai. **A1 learns when to select validated continuous skills; A2 learns continuous joint-target increments.** Each family will train through its own self-play, then frozen policies will cross-play under one shared simulator, observation contract and referee.

**Status, 23 November 2026:** design and validation branch, not a trained A1/A2 release. Confirmed joint limits invalidate the old five-action bank's admission. The new motion candidates, rule proposal and A2 command audit are included; the multi-region referee, revised training integration and A2 native controller are still pending. No unattended long training should start from this branch's historical A1 entry point.

## Plan and final requirements

The report-ready [中文 Plan 2 设计与验收方案](docs/plan2_zh.md) explains the reasons, dependencies and failure gates for all four requirements:

| Final requirement | Design and evidence required |
| --- | --- |
| Natural, multi-direction, multi-location attack and defence | Data-informed continuous motion; distinct location/direction diagnostics **and** unrestricted duels. A larger target alone is not proof of coverage. |
| Test A2's independently learned flexibility | Held-out directions, timings and locations; legal attack/defence success and adaptation, not jitter or an assumed A2 win. |
| Fair observations and scoring | Same physical information, control limits, target geometry and event rules; simultaneous observations, swapped slots and frozen evaluation. |
| No obvious individual capability defect | A1 and A2 separately pass control, legal attack, physical defence, recovery and degeneracy gates before cross-play. |

The dependency order is: **hardware constraints → common scoring specification and feasibility → data/continuous skills → full-window paired validation → common-platform checks → separate self-play → frozen cross-play**. A failed stage is repaired before dependent stages are accepted.

## Read these first

- [Plan 2 rationale, experiments and admission gates](docs/plan2_zh.md)
- [Common scoring proposal](docs/scoring_spec_v3_proposal_zh.md) and [machine-readable candidate](config/scoring_spec_v3_proposal.json) — **not activated**
- [A2 control, observations, learning and fairness](docs/a2_system_design_zh.md)
- [Confirmed joint ranges](config/joint_constraints_observed.json), [requested poses](config/a1_pose_requirements_v1.json), [current action coverage and failures](docs/a1_action_coverage_zh.md)
- [Packaged evidence summary and original-report fingerprints](docs/plan2_verification.json)
- [Historical A1 setup/results](README_A1_HISTORY.md) and [source/data redistribution notice](motion_skills/plan1/SOURCES.md)

## Reproduce the offline checks

Run from the repository root. Tested on Linux with Python 3.13.11, MuJoCo 3.4.0, NumPy 2.5.4 and SciPy 1.16.3. Dependencies below are ranges, not a bit-reproducible lockfile.

```bash
git clone --branch plan2 --single-branch https://github.com/Wea3el/RL-SwordNet.git
cd RL-SwordNet
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plan2.txt
python -m unittest tests.test_joint_constraints tests.test_candidate_graph tests.test_rule_target_audit tests.test_scoring_reachability tests.test_a2_command_contract -q

# Choose a fresh output path each time; previous evidence is never overwritten.
python -m tools.audit_rule_contract --output outputs/plan2_check/event_contract
python -m tools.audit_a2_command_contract --output outputs/plan2_check/command_contract
python -m tools.audit_scoring_reachability --output outputs/plan2_check/reachability
python -m tools.refine_scoring_reachability --source outputs/plan2_check/reachability --output outputs/plan2_check/refinement
```

These checks need neither ASE motion binaries nor Torch. The reachability checks are **static geometry**, and the command audit is **offline**, not proof of dynamic attacks or a deployed A2 controller. Source-built scenes use the current checkout's mesh path. Archived local scenes contain absolute mesh paths and must not be treated as portable replay files.

To run the full suite, also install the historical learner dependencies:

```bash
python -m pip install -r requirements-a1.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -q
```

Full native trajectory/video replay needs additional local source artifacts and rendering dependencies; it is not part of this clean-clone offline recipe. `tools.audit_target_candidates` specifically requires seven historical native trace sets which are **not distributed**. Their measured outcomes, including failures, are summarized in the verification file. Other historical `outputs/...` links are local evidence paths, not files hosted on GitHub. Original/derived third-party motion arrays, checkpoints and full `outputs/` are deliberately excluded.

Release checks on 23 November 2026: the local full suite passed **525 tests**; a clean exported checkout with no ASE cache passed its suite (**521 run, 2 skipped**) and all four offline audit commands above. The quick subset passed 50 tests. These include software-contract and fixture tests, not trained fencing matches; passing them does not admit the new scoring proposal or either policy to final evaluation. The two skips are source-dependent ASE fixtures, not silently passed tests.

## What this branch does not claim

- The overlapping three-sphere target is not yet a validated multi-location game or an anatomical torso.
- Existing successful single-ball parries do not transfer automatically to the proposed larger region.
- Ten named guards/five cuts are requirements, not all implemented and admitted skills.
- A2's larger action space creates an opportunity, not a guarantee of better performance or natural motion.
- Debug force caps and reference-command tests are not calibrated motor limits or real-robot safety certification.

This is a simulation research platform for controlled training with blunt practice swords. It is not a controller approved for contact with people.
