import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from sim.a1_upright_audit import audit, loaded_contact_indices, upright_metrics


class UprightAuditTests(unittest.TestCase):
    def test_tip_direction_is_signed_not_an_unsigned_line(self):
        grip = np.array([[0., 0., .6]])
        axis = np.array([[.5, 0., np.sqrt(.75)]])
        self.assertTrue(upright_metrics(grip, grip+axis)['pass'])
        self.assertFalse(upright_metrics(grip, grip-axis)['pass'])

    def test_slot_yaw_does_not_change_upward_elevation(self):
        grip = np.array([[0., .2, .6]])
        tip = np.array([[.2, .3, 1.2]])
        yaw = np.diag([-1., -1., 1.])
        self.assertEqual(upright_metrics(grip, tip), upright_metrics(grip@yaw, tip@yaw))

    def test_low_grip_cannot_pass_just_by_aiming_up(self):
        self.assertFalse(upright_metrics([[0., 0., .1]], [[0., 0., 1.]])['pass'])

    def test_all_frames_checked_not_average_orientation(self):
        grip = np.array([[0., 0., .6], [0., 0., .6]])
        result = upright_metrics(grip, grip+[[0., 0., 1.], [0., 0., -1.]])
        self.assertFalse(result['pass'])
        self.assertEqual(result['failed_local_indices'], [1])

    def test_loaded_contact_cannot_hide_outside_declared_phase(self):
        contacts = [{'time_s': 3., 'contacts': [
            {'category': 'cross_blade_guard', 'normal_force_n': .001}]}]
        np.testing.assert_array_equal(loaded_contact_indices(np.arange(5.), contacts), [3])
        with self.assertRaises(ValueError):
            loaded_contact_indices(np.arange(2.), contacts)

    def test_invalid_geometry_or_thresholds_rejected(self):
        for g, t in (([], []), ([[0, 0, 0]], [[0, 0, 0]]), ([[0, 0, np.nan]], [[0, 0, 1]])):
            with self.assertRaises(ValueError): upright_metrics(g, t)
        with self.assertRaises(ValueError):
            upright_metrics([[0, 0, .6]], [[0, 0, 1.]], min_elevation_deg=-25.)

    def test_native_readback_uses_actual_qpos_and_bound_pair(self):
        # Positive rotation about Y turns this local +Z sword forward/down.
        xml = ('<mujoco><compiler angle="radian"/><worldbody>'
               '<body pos="0 0 .6"><joint name="a2_J0" axis="0 1 0"/>'
               '<geom type="sphere" size=".01" mass=".1"/>'
               '<site name="a2_shinai_grip" pos="0 0 0"/>'
               '<site name="a2_shinai_tip" pos="0 0 1"/>'
               '</body></worldbody></mujoco>')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'scene.xml').write_text(xml)
            (root/'summary.json').write_text(json.dumps({'native_execution_pass': True}))
            (root/'contacts.json').write_text(json.dumps([{'time_s': .5, 'contacts': [
                {'category': 'cross_blade_blade', 'normal_force_n': 1.}]}]))
            np.savez(root/'trace.npz', time_s=[0., .5, 1.], qpos=[[0.], [0.], [0.]],
                     desired_reference_qpos=[[np.pi], [np.pi], [np.pi]])
            hashes = {n: hashlib.sha256((root/n).read_bytes()).hexdigest()
                      for n in ('scene.xml', 'trace.npz', 'summary.json')}
            pair = {'status': 'qualified', 'successful_physical_defence': True,
                    'attacker_slot': 'a1', 'matching_checks': {'fixture': True},
                    'defended_triangle_audit': {'native_trace_source_sha256': hashes}}
            (root/'pair.json').write_text(json.dumps(pair))
            result = audit(root, root/'audit1', defender='a2', active_start_s=0.,
                           active_end_s=1., comparison=root/'pair.json')
            self.assertTrue(result['qualified_upright_example'])
            # Corrupt the command-independent actual state while keeping the
            # old pair hash: it must fail both the orientation and binding.
            np.savez(root/'trace.npz', time_s=[0., .5, 1.], qpos=[[np.pi]]*3,
                     desired_reference_qpos=[[0.]]*3)
            result = audit(root, root/'audit2', defender='a2', active_start_s=0.,
                           active_end_s=1., comparison=root/'pair.json')
            self.assertFalse(result['actual_orientation_pass'])
            self.assertFalse(result['independent_qualified_defence_bound'])


if __name__ == '__main__': unittest.main()
