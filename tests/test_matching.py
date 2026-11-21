"""Synthetic kinematic tests: no source motion files or network required."""
import unittest

import mujoco
import numpy as np

from plan1_skills.matching import match_sequence, screen_joint_path, solve_pose, tool_fk


def fixture():
    # Cartesian translations plus two hinge rotations: exactly five task DOFs.
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint type="slide" axis="1 0 0"/><geom size=".01" mass="1"/>
      <body><joint type="slide" axis="0 1 0"/><geom size=".01" mass="1"/>
      <body><joint type="slide" axis="0 0 1"/><geom size=".01" mass="1"/>
      <body><joint axis="0 0 1"/><geom size=".01" mass="1"/>
      <body><joint axis="0 1 0"/><geom size=".01" mass="1"/>
      <site name="shinai_grip"/><site name="shinai_tip" pos=".79 0 0"/>
      </body></body></body></body></body></worldbody></mujoco>''')


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.model = fixture()
        self.bounds = np.array([[-2, 2]]*3+[[-np.pi, np.pi]]*2)

    def test_reachable_whole_sequence_and_bounds(self):
        q = np.array([[.1*i, -.02*i, .03*i, .04*i, -.03*i] for i in range(8)])
        targets = [tool_fk(self.model, qpos=row) for row in q]
        result = match_sequence(self.model, np.array([t[0] for t in targets]),
                                np.array([t[1] for t in targets]), self.bounds,
                                position_tolerance_m=1e-5, angle_tolerance_rad=1e-5)
        self.assertEqual(result.qpos.shape, q.shape)
        self.assertTrue(result.pose_success.all(), result.summary())
        self.assertTrue(np.all(result.qpos >= self.bounds[:, 0]))
        self.assertTrue(np.all(result.qpos <= self.bounds[:, 1]))

    def test_unreachable_frame_not_dropped(self):
        targets = np.array([[0., 0, 0], [10, 0, 0], [.1, 0, 0]])
        result = match_sequence(self.model, targets, np.tile([1., 0, 0], (3, 1)),
                                self.bounds, first_frame_starts=1)
        self.assertEqual(len(result.qpos), 3)
        self.assertFalse(result.pose_success[1])
        self.assertGreater(result.position_error_m[1], 7.)

    def test_antiparallel_is_not_false_success(self):
        result = solve_pose(self.model, [0., 0, 0], [-1., 0, 0], self.bounds,
                            max_iterations=150, angle_tolerance_rad=1e-5)
        self.assertTrue(result["pose_success"], result)
        self.assertLess(result["angle_error_rad"], 1e-5)

    def test_explicit_bounds_and_unit_axis_required(self):
        with self.assertRaises(ValueError):
            match_sequence(self.model, [[0., 0, 0]], [[2., 0, 0]], self.bounds)
        with self.assertRaises(ValueError):
            match_sequence(self.model, [[0., 0, 0]], [[1., 0, 0]], np.zeros((5, 2)))

    def test_deterministic_initialization(self):
        args = (self.model, [[.2, .1, .3]], [[0., 1, 0]], self.bounds)
        a = match_sequence(*args, seed=123)
        b = match_sequence(*args, seed=123)
        np.testing.assert_array_equal(a.qpos, b.qpos)

    def test_interpolated_path_checks_collision_between_clear_endpoints(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="obstacle" type="sphere" size=".1"/>
          <body><joint type="slide" axis="1 0 0"/>
            <geom name="moving" type="sphere" size=".1" mass="1"/>
          </body></worldbody></mujoco>''')
        for q in ([-1.], [1.]):
            self.assertEqual(screen_joint_path(model, q)[0], [])
        contacts, count = screen_joint_path(model, [-1.], [1.])
        self.assertGreater(count, 2)
        self.assertTrue(any(c["distance_m"] < -.001 for c in contacts))


if __name__ == "__main__":
    unittest.main()
