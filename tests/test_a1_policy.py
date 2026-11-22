"""Candidate policy contracts only; no optimizer, training, native run or checkpoint."""
import copy
import hashlib
import json
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sim.a1_policy import (ACTION_NAMES, FEATURE_ORDER, A1FeatureEncoder,
                            A1Policy, FEATURE_VERSION)


def observation(*, busy=False, action=0, elapsed=0., mask=None):
    return {"time_s": 12.3,
        "own_skill": {"action_id": action, "busy": busy, "elapsed_s": elapsed,
            "action_mask": [True]*5 if mask is None else mask, "phase": "ignored",
            "attack_id": 999},
        "own_q_rad": np.arange(7)*.1, "own_qvel_rad_s": np.arange(7)*.2,
        "opponent_tip_minus_own_target_local_m": np.array([.2, -.1, .3]),
        "opponent_grip_minus_own_target_local_m": np.array([.4, -.2, .5]),
        "opponent_tip_velocity_local_m_s": np.array([.02, -.01, .03]),
        "opponent_skill_id": "not an observation feature", "opponent_future_phase": object()}


def weights(policy):
    return hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in policy.state_dict().values())).hexdigest()


class A1PolicyTests(unittest.TestCase):
    def setUp(self):
        self.encoder = A1FeatureEncoder((1., 27., 28.72, 30., 25.))
        self.policy = A1Policy(self.encoder)

    def test_versioned_public_feature_order_and_scales(self):
        x, mask, choice = self.encoder.encode(observation())
        self.assertEqual(x.shape, (30,)); self.assertEqual(x.dtype, np.float32)
        self.assertEqual(len(FEATURE_ORDER), 30); self.assertTrue(choice)
        np.testing.assert_allclose(x[:7], np.arange(7)*.1/math.pi, atol=1e-8)
        np.testing.assert_allclose(x[7:14], np.arange(7)*.2/4., atol=1e-8)
        np.testing.assert_array_equal(x[24:29], [1, 0, 0, 0, 0])
        self.assertEqual(x[23], 0.); self.assertEqual(x[29], 0.)
        self.assertEqual(self.encoder.metadata()["feature_version"], FEATURE_VERSION)
        self.assertEqual(self.encoder.metadata()["feature_order"], list(FEATURE_ORDER))
        json.dumps(self.policy.metadata(), allow_nan=False)

    def test_absolute_time_hidden_ids_phase_and_snapshot_identity_are_not_read(self):
        first = observation(); second = copy.deepcopy(first)
        del second["time_s"]
        second["opponent_skill_id"] = object(); second["opponent_future_phase"] = object()
        second["own_skill"]["phase"] = object(); second["own_skill"]["attack_id"] = -999
        second["opponent_snapshot_id"] = "secret_future_winner"
        a = self.policy.act(first); b = self.policy.act(second)
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        self.assertEqual(a.value, b.value)

    def test_hazard_initialization_and_two_to_three_second_survival(self):
        result = self.policy.act(observation())
        p = -math.expm1(-.25*.02)
        np.testing.assert_allclose(result.probabilities, [1-p]+[p/4]*4, rtol=1e-6)
        self.assertAlmostEqual(float(result.probabilities[0])**150, math.exp(-.25*3), places=5)
        self.assertTrue(result.actor_choice)
        self.assertIsNotNone(result.log_prob)
        self.assertAlmostEqual(sum(result.probabilities), 1., places=6)
        self.assertTrue(np.all(result.probabilities > 0))

    def test_initial_hazard_is_input_independent_but_mask_renormalizes(self):
        a = observation(); b = observation(); b["own_q_rad"] = np.ones(7)
        np.testing.assert_array_equal(self.policy.act(a).probabilities, self.policy.act(b).probabilities)
        partial = [True, False, True, False, True]
        result = self.policy.act(observation(mask=partial))
        prior = np.asarray(self.policy.initial_probabilities)
        expected = prior*np.asarray(partial); expected /= expected.sum()
        np.testing.assert_allclose(result.probabilities, expected, rtol=1e-6)
        self.assertEqual(result.probabilities[1], 0.); self.assertEqual(result.probabilities[3], 0.)

    def test_busy_and_unavailable_force_continue_without_actor_logprob(self):
        for obs in (observation(busy=True, action=1, elapsed=13.5),
                    observation(mask=[True, False, False, False, False])):
            result = self.policy.act(obs)
            self.assertEqual(result.action, 0); self.assertFalse(result.actor_choice)
            self.assertIsNone(result.log_prob); self.assertIsNone(result.entropy)
            np.testing.assert_array_equal(result.probabilities, [1, 0, 0, 0, 0])
            self.assertTrue(math.isfinite(result.value))
        result = self.policy.act(observation(busy=True, action=1, elapsed=13.5))
        self.assertEqual(result.features[23], 1.); self.assertEqual(result.features[29], .5)
        np.testing.assert_array_equal(result.features[24:29], [0, 1, 0, 0, 0])

    def test_ready_guard_is_a_real_wait_choice(self):
        result = self.policy.act(observation(), deterministic=True)
        self.assertEqual(result.action, 0); self.assertTrue(result.actor_choice)
        self.assertIsNotNone(result.log_prob); self.assertGreater(result.entropy, 0.)

    def test_seeded_sampling_reproducible_without_weight_update(self):
        before = weights(self.policy)
        a = torch.Generator().manual_seed(77); b = torch.Generator().manual_seed(77)
        actions_a = [self.policy.act(observation(), generator=a).action for _ in range(50)]
        actions_b = [self.policy.act(observation(), generator=b).action for _ in range(50)]
        self.assertEqual(actions_a, actions_b); self.assertEqual(weights(self.policy), before)
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.parameters()))

    def test_actor_evaluation_replays_logprob_value_and_has_gradients_without_update(self):
        result = self.policy.act(observation())
        before = weights(self.policy)
        logp, entropy, value = self.policy.evaluate_actions(result.features[None],
            [result.action], result.action_mask[None])
        self.assertAlmostEqual(float(logp.detach()[0]), result.log_prob, places=6)
        self.assertAlmostEqual(float(value.detach()[0]), result.value, places=6)
        self.assertTrue(torch.isfinite(entropy).all())
        (-logp.mean()+value.square().mean()).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.policy.parameters()))
        self.assertEqual(weights(self.policy), before)

    def test_actor_evaluation_rejects_busy_forced_or_invalid_action_rows(self):
        ready = self.policy.act(observation())
        busy = self.policy.act(observation(busy=True, action=4, elapsed=1.))
        for x, acts, masks in ((busy.features[None], [0], busy.action_mask[None]),
                              (ready.features[None], [1.], ready.action_mask[None]),
                              (ready.features[None], [True], ready.action_mask[None]),
                              (ready.features[None], [5], ready.action_mask[None]),
                              (ready.features[None], [1], np.array([[True, False, True, True, True]]))):
            with self.assertRaises(ValueError): self.policy.evaluate_actions(x, acts, masks)

    def test_bad_public_vectors_masks_progress_or_encoder_are_rejected(self):
        for field, value in (("own_q_rad", [0]*6), ("own_qvel_rad_s", [float("nan")]*7),
                             ("opponent_tip_velocity_local_m_s", [0, 0, float("inf")])):
            obs = observation(); obs[field] = value
            with self.assertRaises(ValueError): self.policy.act(obs)
        for field, value in (("busy", 1), ("action_id", True), ("action_id", 5),
                             ("elapsed_s", -.1), ("elapsed_s", float("nan")),
                             ("action_mask", [1]*5), ("action_mask", [False]*5)):
            obs = observation(); obs["own_skill"][field] = value
            with self.assertRaises(ValueError): self.policy.act(obs)
        for durations in ((1.,)*4, (1., 0., 1., 1., 1.), (1., float("nan"), 1., 1., 1.)):
            with self.assertRaises(ValueError): A1FeatureEncoder(durations)
        for kwargs in ({"guard_hazard_per_s": 0.}, {"decision_period_s": True}, {"hidden_sizes": (0,)}):
            with self.assertRaises(ValueError): A1Policy(self.encoder, **kwargs)

    def test_encoder_from_library_freezes_own_durations_and_rejects_id_mismatch(self):
        source = SimpleNamespace(skills={name: SimpleNamespace(name=name, action_id=i, duration_s=i+1.)
                                        for i, name in enumerate(ACTION_NAMES)})
        encoder = A1FeatureEncoder.from_library(source)
        source.skills["stab"].duration_s = 99.
        self.assertEqual(encoder.durations_s[1], 2.)
        source.skills["stab"].action_id = 3
        with self.assertRaises(ValueError): A1FeatureEncoder.from_library(source)
        source.skills["stab"].action_id = 1.
        with self.assertRaises(ValueError): A1FeatureEncoder.from_library(source)

    def test_nonfinite_network_output_never_becomes_valid_rollout_data(self):
        with torch.no_grad():
            self.policy.critic.bias.fill_(float("nan"))
        with self.assertRaises(ValueError): self.policy.act(observation())


if __name__ == "__main__":
    unittest.main()
