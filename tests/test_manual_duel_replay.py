"""Fast target-consistency and recorded-state replay checks, without rollouts."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim.manual_duel import run_experiment, run_trial, swept_sphere_fraction
from sim.manual_duel_replay import sample_state


class TargetFixtureValidationTests(unittest.TestCase):
    def test_scoring_radius_mismatch_rejected_before_reset_or_physics(self):
        targets = {owner: {"center_world_m": [1.43, 0., 1.05], "radius_m": .06}
                   for owner in ("a1", "a2")}
        scene = SimpleNamespace(model=object(), data=object(), agents={"a1": {}, "a2": {}},
                                config={"virtual_targets": targets})
        for mirrored in (False, True):
            with self.subTest(mirrored=mirrored), patch("sim.manual_duel.schedule", return_value={}), patch("sim.manual_duel.mujoco.mj_resetData") as reset, patch("sim.manual_duel.mujoco.mj_step") as step:
                with self.assertRaisesRegex(ValueError, "radius must match"):
                    run_trial(scene, object(), {"virtual_target_radius_m": .05},
                              parry_enabled=True, mirrored=mirrored)
                reset.assert_not_called()
                step.assert_not_called()

    def test_reference_target_and_runtime_radius_must_match_fixed_fixture(self):
        for field, wrong, message in (("target_world_m", [1.44, 0., 1.05], "target must match"),
                                      ("target_radius_m", .05, "radii must agree")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config = {"base_distance_m": 1.75, "target_world_m": [1.43, 0., 1.05], "target_radius_m": .06}
                config[field] = wrong
                library = SimpleNamespace(config=config)
                scene = SimpleNamespace(config={"virtual_targets": {"a2": {"center_world_m": [1.43, 0., 1.05]}}})
                output = Path(directory)/"rejected"
                with patch("sim.manual_duel.build_manual_library", return_value=library), patch("sim.manual_duel.build_duel_scene", return_value=scene), patch("sim.manual_duel.run_trial") as trial:
                    with self.assertRaisesRegex(ValueError, message):
                        run_experiment(output, {"virtual_target_radius_m": .06})
                    trial.assert_not_called()
                # An empty evidence directory may be reserved; no model,
                # trajectory or misleading partial success report is written.
                self.assertEqual(list(output.iterdir()), [])

    def test_just_outside_effective_radius_is_not_a_scored_hit(self):
        radius = .06+.01
        self.assertIsNone(swept_sphere_fraction([-1., radius+1e-8, 0.],
                                                [1., radius+1e-8, 0.], [0., 0., 0.], radius))
        self.assertAlmostEqual(swept_sphere_fraction([-1., radius, 0.],
                                                    [1., radius, 0.], [0., 0., 0.], radius), .5)


class RecordedReplayTests(unittest.TestCase):
    def setUp(self):
        self.trace = {"time_s": np.array([0., .2, 1.]),
                      "qpos": np.array([[0., 10.], [2., 14.], [10., 30.]]),
                      "phase": np.array([0., .9, 1.]),
                      "target_qpos": np.full((3, 2), 999.)}

    def test_interpolation_uses_recorded_seconds_not_phase_or_commands(self):
        np.testing.assert_allclose(sample_state(self.trace, .1), [1., 12.])
        np.testing.assert_allclose(sample_state(self.trace, .6), [6., 22.])
        for at, expected in zip(self.trace["time_s"], self.trace["qpos"]):
            np.testing.assert_array_equal(sample_state(self.trace, at), expected)

    def test_display_sampling_clamps_at_recorded_endpoints(self):
        np.testing.assert_array_equal(sample_state(self.trace, -1.), self.trace["qpos"][0])
        np.testing.assert_array_equal(sample_state(self.trace, 2.), self.trace["qpos"][-1])

    def test_sampling_preserves_every_joint_column_without_physics_calls(self):
        trace = {"time_s": np.array([0., 2.]),
                 "qpos": np.vstack([np.arange(14.), np.arange(14.)+4.])}
        with patch("sim.manual_duel_replay.mujoco.mj_step", side_effect=AssertionError("Replay is not a rollout")):
            result = sample_state(trace, .5)
        self.assertEqual(result.shape, (14,))
        np.testing.assert_allclose(result, np.arange(14.)+1.)


if __name__ == "__main__":
    unittest.main()
