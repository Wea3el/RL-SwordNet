"""Recorded shared-skill replay helpers; no video or physics generation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim.skill_validation_replay import hit_counts, load_trial, render_comparison, sample_state


class SharedReplayTests(unittest.TestCase):
    def test_actual_recorded_state_interpolation_uses_seconds_and_preserves_joints(self):
        trace = {"time_s": np.array([0., .2, 1.]),
                 "qpos": np.array([[0., 10.], [2., 14.], [10., 30.]]),
                 "target_qpos": np.full((3, 2), 999.)}
        with patch("sim.skill_validation_replay.mujoco.mj_step", side_effect=AssertionError("No new rollout")):
            np.testing.assert_allclose(sample_state(trace, .1), [1., 12.])
            np.testing.assert_allclose(sample_state(trace, .6), [6., 22.])
            np.testing.assert_array_equal(sample_state(trace, -1.), trace["qpos"][0])
            np.testing.assert_array_equal(sample_state(trace, 2.), trace["qpos"][-1])

    def test_saved_hit_events_are_not_shown_early_and_double_touch_is_not_winner(self):
        result = {"hits": [{"agent": "a1", "time_s": 2., "double_touch": False},
                           {"agent": "a1", "time_s": 3., "double_touch": True},
                           {"agent": "a2", "time_s": 3., "double_touch": True}]}
        self.assertEqual(hit_counts(result, 1.), ({"a1": 0, "a2": 0}, False))
        self.assertEqual(hit_counts(result, 2.5), ({"a1": 1, "a2": 0}, False))
        self.assertEqual(hit_counts(result, 3.), ({"a1": 2, "a2": 1}, True))

    def test_existing_video_is_preserved_before_scene_or_trace_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"existing.mp4"
            output.write_bytes(b"existing-user-video")
            with patch("sim.skill_validation_replay.mujoco.MjModel.from_xml_path") as scene:
                with self.assertRaises(FileExistsError):
                    render_comparison(Path(directory)/"missing", "a", "b", output)
                scene.assert_not_called()
            self.assertEqual(output.read_bytes(), b"existing-user-video")

    def test_numeric_trace_dimensions_and_times_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/"summary.json").write_text(json.dumps({"status": "invalid", "failure": "test", "hits": []}))
            np.savez_compressed(folder/"trajectory.npz", time_s=[0., 1.],
                                qpos=np.zeros((2, 14)), blade_normal_force_n=[0., 1.])
            result, trace = load_trial(folder, 14)
            self.assertEqual(result["status"], "invalid")
            self.assertEqual(trace["qpos"].shape, (2, 14))
            with self.assertRaises(ValueError):
                load_trial(folder, 7)
            np.savez_compressed(folder/"trajectory.npz", time_s=[1., 0.],
                                qpos=np.zeros((2, 14)), blade_normal_force_n=[0., 1.])
            with self.assertRaises(ValueError):
                load_trial(folder, 14)

    def test_invalid_display_settings_fail_without_renderer_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"new.mp4"
            with patch("sim.skill_validation_replay.mujoco.Renderer") as renderer:
                for settings in ({"fps": 0}, {"fps": True}, {"fps": 61},
                                 {"start_s": np.nan}, {"distance": -1.}):
                    with self.subTest(settings=settings), self.assertRaises(ValueError):
                        render_comparison(directory, "a", "b", output, **settings)
                renderer.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
