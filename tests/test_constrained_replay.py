import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim.constrained_replay import frame_schedule, load_recorded_states, probe_video, render_trial


class ConstrainedReplayTests(unittest.TestCase):
    def test_real_time_nearest_actual_sample_no_target_interpolation(self):
        times = np.arange(7501)/1000
        requested, indices = frame_schedule(times, fps=30)
        self.assertEqual(len(indices), 225)
        self.assertAlmostEqual(len(indices)/30, 7.5)
        self.assertLessEqual(float(np.max(np.abs(times[indices]-requested))), .0005)
        np.testing.assert_array_equal(indices[:4], [0, 33, 67, 100])

    def test_invalid_timing_settings_rejected(self):
        for kwargs in ({"fps": True}, {"fps": 0}, {"fps": 61},
                       {"start_s": -.1}, {"end_s": 2}, {"end_s": 0}, {"start_s": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                frame_schedule([0., 1.], **kwargs)
        with self.assertRaises(ValueError):
            frame_schedule([0., np.nan])

    def test_actual_qpos_is_loaded_not_desired_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/"summary.json").write_text(json.dumps({"status": "failed"}))
            np.savez(p/"trace.npz", time_s=[0., 1.], qpos=[[1., 2.], [3., 4.]],
                     desired_reference_qpos=np.full((2, 2), 999.))
            result, times, qpos = load_recorded_states(p, 2)
            self.assertEqual(result["status"], "failed")
            np.testing.assert_array_equal(qpos, [[1., 2.], [3., 4.]])
            with self.assertRaises(ValueError):
                load_recorded_states(p, 3)

    def test_existing_directory_is_never_overwritten_or_simulated(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sim.constrained_replay.mujoco.mj_step", side_effect=AssertionError("Never simulate")):
                with self.assertRaises(FileExistsError):
                    render_trial("nonexistent", directory)

    def test_probe_checks_decoded_frame_count_and_duration(self):
        content = {"streams": [{"width": 960, "height": 640, "nb_read_frames": "225",
                                "avg_frame_rate": "30/1", "duration": "7.500000"}]}
        with patch("sim.constrained_replay.subprocess.run") as run:
            run.return_value.stdout = json.dumps(content)
            self.assertEqual(probe_video("test.mp4", width=960, height=640, frames=225, fps=30), content["streams"][0])
            with self.assertRaises(ValueError):
                probe_video("test.mp4", width=960, height=640, frames=224, fps=30)


if __name__ == "__main__":
    unittest.main()
