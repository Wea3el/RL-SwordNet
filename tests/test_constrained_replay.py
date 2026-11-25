import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim.constrained_replay import (frame_schedule, load_recorded_states, probe_video,
                                    render_trial, load_triangle_overlay, replay_camera, replay_footer)


class ConstrainedReplayTests(unittest.TestCase):
    def test_aborted_trial_never_claims_full_recovery(self):
        summary = {'status': 'failed', 'completed_time_s': 5.904, 'requested_duration_s': 10.25}
        for triangle, qualified in ((False, False), (True, False), (True, True)):
            caption = replay_footer(summary, triangle=triangle, qualified=qualified)
            self.assertIn('5.904/10.250', caption)
            self.assertIn('recovery incomplete', caption)
            self.assertNotIn('full recovery is shown', caption)
            self.assertNotIn('Qualified exact pair', caption)

    def test_completed_and_failed_captions_remain_distinct(self):
        self.assertIn('full recovery', replay_footer({'status': 'completed'}, triangle=True))
        self.assertIn('Qualified exact pair', replay_footer({'status': 'completed'}, qualified=True))
        self.assertIn('Failed/incomplete', replay_footer({'status': 'failed'}))

    def test_excerpt_never_claims_to_show_complete_recovery(self):
        for qualified in (False, True):
            caption = replay_footer({'status': 'completed'}, triangle=True,
                                    qualified=qualified, full_window=False)
            self.assertIn('Excerpt only', caption)
            self.assertNotIn('full recovery is shown', caption)
            self.assertNotIn('Qualified exact pair', caption)

    def test_contact_view_uses_loaded_blade_contact_not_visual_guess(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'contacts.json'
            path.write_text(json.dumps([
                {'time_s': 1., 'contacts': [{'category': 'cross_blade_blade', 'normal_force_n': 0.}]},
                {'time_s': 2., 'contacts': [{'category': 'cross_blade_blade', 'normal_force_n': 1.,
                                          'position_world_m': [1., .1, .8]}]}]))
            view = replay_camera(directory, 'contact')
            self.assertEqual(view['lookat'], [1., .1, .8])
            self.assertEqual(view['recorded_contact_time_s'], 2.)
            self.assertEqual(len(view['contacts_sha256']), 64)
            path.write_text('[]')
            with self.assertRaises(ValueError):
                replay_camera(directory, 'contact')

    def test_camera_selection_is_display_only_and_rejects_unknown(self):
        self.assertEqual(replay_camera('not_read', 'overview')['distance'], 3.6)
        with self.assertRaises(ValueError):
            replay_camera('not_read', 'invented')

    def test_qualified_caption_requires_matched_exact_defended_evidence(self):
        hashes = {'summary.json': 's', 'trace.npz': 't', 'scene.xml': 'm'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.json'
            comparison = {'status': 'qualified', 'successful_physical_defence': True,
                          'matching_checks': {'same': True},
                          'defended_triangle_audit': {'native_trace_source_sha256': hashes}}
            path.write_text(json.dumps(comparison))
            overlay, _, _, qualified = load_triangle_overlay(hashes, defence_comparison=path)
            self.assertTrue(qualified)
            self.assertEqual(overlay, comparison['defended_triangle_audit'])
            with self.assertRaisesRegex(ValueError, 'different native evidence'):
                load_triangle_overlay(dict(hashes, **{'scene.xml': 'other'}), defence_comparison=path)
            comparison['matching_checks']['same'] = False
            path.write_text(json.dumps(comparison))
            with self.assertRaisesRegex(ValueError, 'matched qualified'):
                load_triangle_overlay(hashes, defence_comparison=path)

    def test_failed_defence_and_ambiguous_overlay_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.json'
            path.write_text(json.dumps({'status': 'failed', 'successful_physical_defence': False}))
            with self.assertRaises(ValueError):
                load_triangle_overlay({}, defence_comparison=path)
            with self.assertRaises(ValueError):
                load_triangle_overlay({}, triangle_audit=path, defence_comparison=path)

    def test_triangle_overlay_cannot_borrow_another_trial_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); trial=base/'trial';trial.mkdir()
            for name in ('summary.json','scene.xml','trace.npz'):
                (trial/name).write_bytes(b'placeholder')
            audit=base/'triangle.json'
            audit.write_text(json.dumps({'native_trace_source_sha256':{'summary.json':'wrong','trace.npz':'wrong'}}))
            with self.assertRaisesRegex(ValueError,'different native evidence'):
                render_trial(trial,base/'video',triangle_audit=audit)

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
