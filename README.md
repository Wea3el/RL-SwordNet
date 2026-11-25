# RL-SwordNet — A1 motion-library baseline

Two original-size, floor-fixed, seven-joint robot arms with 1.080 m shinai. **A1 will learn to select and time finite continuous skills; A2 will learn continuous joint-target increments.** Each family trains by self-play before frozen cross-play under the same physics, observations and rules.

**Release status — 25 November 2026:** current motion materials, scoped validation evidence, a named-edge development adapter and training strategy. **Not a trained policy and not a new-bank PPO training release.** The retained five-action trainer is incompatible with this bank. Do not start unattended training from its historical commands.

中文完整说明：[A1 首版动作包、评分与训练策略](docs/a1_release_zh.md)。

## Included motion materials

| Family | Materials | Coverage claim |
| --- | --- | --- |
| Attack | Left/right downward cuts, left/right rising cuts, central downward cut, one thrust | Six distinct attack families, not three copies of one poke |
| Defence | Left, right and high guard families; six incoming-attack-specific variants | A matched full-window defence case for each of the six selected attacks |
| Preparation/recovery | Explicit guard nodes, transfers and selected interception/unloading stages | Exact-node command continuity; dynamic scope varies by edge |

The merged catalog has **34 edges, 13 named nodes (12 distinct joint poses)**. It contains **18 transfer candidates**: graph connectivity does not certify all transfers under attack. The named adapter keeps wait and every edge distinct; it does not select using an opponent's action label or silently create a bridge.

The result is **6/6 selected, matched single-attack defence cases**, not a 100% success rate against arbitrary attacks. The package index contains those nominal cases. Additional local evidence covers particular timing offsets, swapped slots, repeated rounds and an observed right-parry sequence; see [the separate scope index](docs/a1_validation_scope.json). Raw extension traces are not bundled.

Known weakness: a fast left downward-cut → immediate reverse rising-cut sequence can beat the current defence. An attempted replacement makes contact earlier but still concedes a hit at about 7.651 s. This failure is retained; the failed replacement is not a usable skill. The first release does not require every combination to be defensible.

Materials and fingerprints: [`assets/a1_pose_bank_v1/`](assets/a1_pose_bank_v1/). Large raw native traces remain local; packaged summaries/hashes are not a substitute for those raw files.

## Inspect the package

Use Python 3.11+; the development machine uses Python 3.13 and MuJoCo 3.4. Package verification needs NumPy and MuJoCo; rendering also needs Pillow and ffmpeg. Dependency ranges are not a reproducibility lockfile.

```bash
git clone --branch A1 --single-branch https://github.com/Wea3el/RL-SwordNet.git
cd RL-SwordNet
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m sim.a1_release inspect
python -m unittest tests.test_a1_release tests.test_a1_variant_bank \
  tests.test_joint_constraints tests.test_blade_geometry tests.test_triangle_task -q
```

On Windows use `.venv\Scripts\activate`. Inspection verifies packaged bytes, reference curves, exact endpoints and the confirmed limits; **it does not run a native duel or prove tactical performance**. New actor checkpoints must bind the catalog/order, controller, physical model, observation and referee identities. Old five-action weights cannot be relabelled or reused as the new actor.

To run one **fixed scheduled physics diagnostic**, rather than a trained policy, use fresh directories:

```bash
python -m sim.a1_release_case --case parry_right_descending \
  --variant baseline --output outputs/a1_repro_right_baseline
python -m sim.a1_release_case --case parry_right_descending \
  --variant defended --output outputs/a1_repro_right_defended
```

Each preserves the full packaged 16-second window and then audits all three target regions. A completed run or a sword contact is not automatic defence admission; compare the matched baseline, actual contact and full-window entries. The CLI refuses existing output directories and mismatched frozen inputs.

Release checks: 974-test full regression passed (2 external-data skips), followed by 36 final package/case-runner checks. A fresh packaged right-down defence ran all 16 s with valid three-region audit and zero entries. Exact scope and file fingerprints are in [the release verification record](docs/a1_release_verification.json); this extra smoke test is not a trained-policy evaluation or a new matched-pair certificate.

## Actual-state video examples

These are replayed native MuJoCo states, **not RL policy demonstrations**. The left example is a complete two-round record; the right example is a labelled excerpt from a complete 16-second trial. The source timestamps and hashes are in [`media/a1_v1/manifest.json`](media/a1_v1/manifest.json).

- [Left defence, recovery and second round (24 s)](media/a1_v1/left_two_rounds.mp4)
- [Right observed interception and staged recovery (10 s excerpt)](media/a1_v1/right_observed_stages.mp4)
- [Known fast-combination failure (6 s excerpt)](media/a1_v1/fast_combo_failure.mp4)

## Training strategy: attack and defend without action rewards

Reward **physical outcomes**, never the choice of an attack/defence label, a blade contact, or a switch. Keep all three target regions worth the same. Do not force an attack-defence alternation: sustained pressure or defence can be correct, and a hard timer creates a predictable artificial tactic.

1. Integrate the new catalog, real-state availability mask and shared online referee. Check complete rounds before any long run.
2. Supply both attack openings and validated defendable threats across directions, legal start states and timing. No hidden opponent labels, mid-round resets or artificially slowed attacks.
3. Train against frozen historical opponents, fixed for each round; alternate physical slots. Use passive, attacking and defensive diagnostic opponents to expose specialization, not as a substitute for self-play.
4. If a policy only defends or only attacks, first check reachability/observation, then rebalance training situations/opponents. Do not pay it for pressing the other button.
5. Admit a final checkpoint only after **separate attack, defence, recovery, legality and held-out tests**. Average reward alone is insufficient. Report timeouts and action repetition rather than silently penalizing them.

This reduces degeneracy risk; it cannot guarantee successful learning before training/evaluation. [`config/a1_training_strategy_v1.json`](config/a1_training_strategy_v1.json) is a documented plan, **not an active runtime configuration**.

## Shared scoring and fairness

Three non-overlapping spherical regions, each radius **0.10 m**, in the defender's base-local frame (+x toward opponent, +y left, +z up):

| Region | Center (m) | Legal first-hit reward |
| --- | --- | --- |
| Left | (0.32, +0.20, 0.93) | +1 / −1 |
| Right | (0.32, −0.20, 0.93) | +1 / −1 |
| Upper | (0.32, 0, 1.22) | +1 / −1 |

The whole finite blade can score. Blade/blade and blade/guard contact is neutral, not block immunity. Illegal handle/arm/base contact is attributed using the approved pre-contact point-motion heuristic: unilateral fault −1/+1, joint/unclear fault −1/−1, self-contact/ground fault belongs to the responsible side. Same-step fouls take priority; simultaneous legal hit and normal timeout are 0/0. Numerical/evidence failures are errors, not draws. Full-window motion audits continue past first contact; matches terminate at their first terminal event.

Both families must use the same physical state information, model, torque caps, target/referee, time budget and slot-swapped evaluation. Opponent action IDs, future trajectories and policy identity must not enter the actor. Simulator truth is available now; real perception and hardware transfer are not validated.

If A2 mainly wins by exploiting a known A1 library-coverage gap, report that interface limitation separately from a failure to select an available counter. Win rate alone would not establish better learned tactics.

Confirmed joint limits include the CAD-to-straight-elbow J3 offset. Shared simulation caps remain `[5, 80, 15, 20, 3, 3, 2] Nm`, particularly **J4 = 3 Nm**. The 2 rad/s reference ceiling is not a guarantee on collision-induced joint velocity. These are simulation assumptions, not a hardware safety certificate.

## Remaining integration gates

- The historical `sim.a1_env` / `sim.a1_train` uses five actions, a common guard and the old target. It is retained for history, not offered as a working entry point for this bank.
- The three-region contract and offline audit are present; the new online referee plus new actor/collector still need integration and testing.
- A2's common continuous-command adapter is pending. An example 0.1 rad tracking leash changes some validated A1 commands, so it cannot be silently adopted with inherited certificates.
- Not every transfer has native combat validation; rapid cross-side/attack-defence transitions require further scope-specific checks. Do not mark all catalog edges as admitted.
- No new trained checkpoint is included. Final A1-vs-A2 conclusions require independent ability gates and frozen cross-play.

See [current training compatibility audit](docs/a1_release_training_audit.json), [Chinese strategy and caveats](docs/a1_release_zh.md), [Plan 2 rationale](docs/plan2_zh.md), and [historical five-action notes](README_A1_HISTORY.md). [Earlier Plan 2 README](README_PLAN2_HISTORY.md) and older experiment docs are history, not current admission evidence.

## Motion-data notice

This package does not include ASE/Reallusion motion binaries or pretrained policies. Historical ASE workflows remain subject to their original terms; a public repository is not a grant of redistribution rights. See [source notice](motion_skills/plan1/SOURCES.md). Current hand-authored motion materials retain their explicit provenance; they are not claimed to be a validated HEMA dataset. Do not operate this simulation as a real sword-fighting safety controller.
