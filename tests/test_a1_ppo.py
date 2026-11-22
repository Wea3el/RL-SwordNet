"""Synthetic complete-round optimizer contracts; no native rollout or fencing claim."""
import copy
from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sim.a1_policy import A1FeatureEncoder, A1Policy
from sim.a1_ppo import (metadata_hash, ppo_batch_identity, ppo_update_once, weights_hash)
from sim.a1_rollout import DecisionEpisode, physical_time_gae


def public_observation(offset=0.):
    return {"own_skill": {"action_id": 0, "busy": False, "elapsed_s": 0.,
                          "action_mask": [True]*5},
        "own_q_rad": np.linspace(0., .3, 7)+offset, "own_qvel_rad_s": np.zeros(7),
        "opponent_tip_minus_own_target_local_m": np.array([.4+offset, .1, .2]),
        "opponent_grip_minus_own_target_local_m": np.array([.7, .1, .2]),
        "opponent_tip_velocity_local_m_s": np.array([-.02, 0., -.01])}


def episodes(policy, policy_id="learner"):
    result = []
    generator = torch.Generator().manual_seed(122)
    for i, reward in enumerate((1., -1., 0.)):
        slot = "a1" if i % 2 == 0 else "a2"
        buffer = DecisionEpisode({slot: policy_id})
        first = policy.act(public_observation(i*.1), generator=generator)
        second = policy.act(public_observation(i*.1+.03), generator=generator)
        buffer.start(slot, first.features, first.action_mask, first.action, first.log_prob,
                     first.value, policy_id=policy_id, time_s=0.)
        buffer.advance(time_s=.02, rewards={slot: 0.}, next_observations={slot: second.features},
            ready_by_slot={slot: True}, effective_actions={slot: first.action},
            accepted={slot: True})
        buffer.start(slot, second.features, second.action_mask, second.action, second.log_prob,
                     second.value, policy_id=policy_id, time_s=.02)
        # Synthetic elapsed30s tests physical credit without native physics.
        buffer.advance(time_s=30.02, rewards={slot: reward}, next_observations={slot: second.features},
            ready_by_slot={slot: False}, effective_actions={slot: second.action},
            accepted={slot: True}, episode_end=True)
        result.append(buffer.commit()[slot])
    return result


class SinglePPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.policy = A1Policy(A1FeatureEncoder((1., 27., 28.72, 30., 25.)), hidden_sizes=(8, 8))
        self.identity = ppo_batch_identity(self.policy, "learner")
        self.rows = episodes(self.policy)

    def update(self, rows=None, identity=None, **kwargs):
        return ppo_update_once(self.policy, self.rows if rows is None else rows,
            expected_policy_id="learner", batch_identity=self.identity if identity is None else identity,
            **kwargs)

    def assert_rejected_unchanged(self, rows=None, identity=None, **kwargs):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        before = weights_hash(self.policy)
        with self.assertRaises(ValueError): self.update(rows, identity, optimizer=optimizer, **kwargs)
        self.assertEqual(weights_hash(self.policy), before)
        self.assertEqual(optimizer.state, {})

    def test_single_explicit_update_changes_weights_and_reports_finite_metrics(self):
        before = weights_hash(self.policy)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        result = self.update(optimizer=optimizer)
        self.assertTrue(result["weights_changed"])
        self.assertEqual(result["weights_sha256_before"], before)
        self.assertEqual(result["weights_sha256_after"], weights_hash(self.policy))
        self.assertEqual(result["epochs"], 1)
        self.assertEqual(result["optimizer_steps_this_call"], 1)
        self.assertEqual(result["optimizer_step_count_before"], 0)
        self.assertEqual(result["optimizer_step_count_after"], 1)
        self.assertEqual(result["actor_choice_rows"], 6)
        self.assertTrue(result["finite_gradients"])
        self.assertFalse(result["learned_fencing_ability_demonstrated"])
        self.assertFalse(result["checkpoint_written"])
        self.assertAlmostEqual(result["ratio_before_min"], 1., places=6)
        self.assertAlmostEqual(result["ratio_before_max"], 1., places=6)
        self.assertEqual(result["clip_fraction_before"], 0.)
        json.dumps(result, allow_nan=False)

    def test_physical_MC_targets_not_normalized_and_population_advantage_stats(self):
        advantage, target = map(np.concatenate, zip(*(physical_time_gae(r) for r in self.rows)))
        old_values = np.array([r.value for e in self.rows for r in e])
        expected_mse = np.mean((old_values-target)**2)
        result = self.update()
        self.assertAlmostEqual(result["returns_min"], float(target.min()), places=6)
        self.assertAlmostEqual(result["returns_max"], float(target.max()), places=6)
        self.assertAlmostEqual(result["value_loss"], float(expected_mse), places=6)
        self.assertAlmostEqual(result["raw_advantage_mean"], float(advantage.mean()), places=6)
        self.assertAlmostEqual(result["raw_advantage_population_std"], float(advantage.std()), places=6)
        self.assertFalse(result["value_targets_normalized"])
        self.assertEqual(result["physical_time_lambda_per_second"], 1.)
        self.assertGreater(result["returns_max"], .77)

    def test_busy_forced_zero_invalid_mask_and_action_are_rejected_before_update(self):
        busy = self.rows[0][0].observation.copy(); busy[23] = 1.
        changes = ({"observation": busy}, {"action_mask": np.array([True, False, False, False, False])},
                   {"action_mask": np.ones(5, dtype=int)}, {"action": True}, {"action": 5})
        for change in changes:
            rows = list(self.rows)
            rows[0] = (replace(rows[0][0], **change), *rows[0][1:])
            self.assert_rejected_unchanged(rows)

    def test_stale_logprob_value_nonfinite_and_partial_round_are_rejected(self):
        for change in ({"log_prob": self.rows[0][0].log_prob-.01},
                       {"value": self.rows[0][0].value+.01}, {"log_prob": float("nan")},
                       {"discounted_reward": float("inf")}, {"policy_id": "opponent"}):
            rows = list(self.rows); rows[0] = (replace(rows[0][0], **change), *rows[0][1:])
            self.assert_rejected_unchanged(rows)
        rows = list(self.rows); rows[0] = (*rows[0][:-1], replace(rows[0][-1], episode_end=False))
        self.assert_rejected_unchanged(rows)
        rows = list(self.rows); rows[0] = rows[0][1:]
        self.assert_rejected_unchanged(rows)

    def test_feature_metadata_and_weight_identities_cannot_be_relabelled_current(self):
        for key, value in (("feature_version", "old_encoder"), ("policy_id", "another_actor"),
                           ("policy_metadata_sha256", "0"*64)):
            bad = dict(self.identity, **{key: value})
            self.assert_rejected_unchanged(identity=bad)
        with torch.no_grad(): self.policy.critic.bias.add_(.01)
        self.assert_rejected_unchanged()

    def test_explicit_environment_compatibility_binding_is_checked(self):
        metadata = {"bank": "frozen-bank", "deployed_rule_version": "training_v1"}
        identity = ppo_batch_identity(self.policy, "learner", metadata)
        self.assert_rejected_unchanged(identity=identity)
        self.assert_rejected_unchanged(identity=identity, expected_compatibility_metadata={"bank": "other"})
        result = self.update(identity=identity, expected_compatibility_metadata=metadata)
        self.assertTrue(result["environment_compatibility_checked"])

    def test_discontinuous_state_or_mixed_physical_discount_is_rejected(self):
        for change in ({"observation": self.rows[0][1].observation+.01},
                       {"discount": .9**self.rows[0][1].duration_s}):
            rows = list(self.rows); rows[0] = (rows[0][0], replace(rows[0][1], **change))
            self.assert_rejected_unchanged(rows)

    def test_nonfinite_gradient_never_calls_optimizer_step(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        before = weights_hash(self.policy)
        hook = self.policy.actor.bias.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        try:
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaises(ValueError): self.update(optimizer=optimizer)
                step.assert_not_called()
        finally:
            hook.remove()
        self.assertEqual(weights_hash(self.policy), before)
        self.assertEqual(optimizer.state, {})

    def test_step_exception_rolls_back_weights_and_optimizer(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        before = weights_hash(self.policy)
        actual = optimizer.step
        def broken_step():
            actual()
            raise RuntimeError("synthetic optimizer failure after partial mutation")
        with patch.object(optimizer, "step", side_effect=broken_step):
            with self.assertRaises(RuntimeError): self.update(optimizer=optimizer)
        self.assertEqual(weights_hash(self.policy), before)
        self.assertEqual(optimizer.state, {})

    def test_frozen_opponent_or_foreign_optimizer_cannot_update(self):
        other = copy.deepcopy(self.policy)
        optimizer = torch.optim.Adam(other.parameters(), lr=3e-4)
        with self.assertRaises(ValueError): self.update(optimizer=optimizer)
        for parameter in self.policy.parameters(): parameter.requires_grad_(False)
        with self.assertRaises(ValueError): self.update()

    def test_invalid_hyperparameters_reject_without_mutation(self):
        for settings in ({"learning_rate": float("nan")}, {"learning_rate": 0.},
                         {"clip_range": 1.}, {"value_coef": -.1}, {"entropy_coef": -.1},
                         {"max_grad_norm": 0.}):
            before = weights_hash(self.policy)
            with self.assertRaises(ValueError): self.update(**settings)
            self.assertEqual(weights_hash(self.policy), before)

    def test_canonical_helpers_match_selfplay_module(self):
        from sim.a1_selfplay import weights_hash as pool_weights, metadata_hash as pool_metadata
        self.assertEqual(weights_hash(self.policy), pool_weights(self.policy))
        self.assertEqual(metadata_hash(self.policy.metadata()), pool_metadata(self.policy.metadata()))


if __name__ == "__main__": unittest.main()
