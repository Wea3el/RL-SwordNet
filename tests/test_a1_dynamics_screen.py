import unittest

import mujoco
import numpy as np

from sim.a1_dynamics_screen import free_space_demand, joint_torque_bounds, screen_curve
from sim.motion_spline import C2JointSpline


def independent_model():
    bodies = ''.join(f'''<body pos="{i * 2} 0 0"><joint name="J{i}" type="hinge"
        axis="0 0 1" range="-3 3" damping=".2"/><geom type="sphere"
        size=".1" pos=".2 0 0" mass="1" contype="0" conaffinity="0"/></body>'''
        for i in range(7))
    actuators = ''.join(f'<motor joint="J{i}" forcerange="-1 2"/>' for i in range(7))
    return mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
        <option gravity="0 0 0"/><worldbody>{bodies}</worldbody>
        <actuator>{actuators}</actuator></mujoco>''')


class DynamicsScreenTests(unittest.TestCase):
    def test_torque_matches_unconstrained_inverse_dynamics(self):
        model = independent_model(); data = mujoco.MjData(model)
        q = np.linspace(-.2, .2, 7); v = np.linspace(-.3, .4, 7); a = np.linspace(.8, -.8, 7)
        actual, _ = free_space_demand(model, data, q, v, a)
        data.qacc[:] = a
        mujoco.mj_inverse(model, data)
        np.testing.assert_allclose(actual, data.qfrc_inverse, atol=1e-12)

    def test_damping_requires_torque_in_direction_of_velocity(self):
        model = independent_model(); data = mujoco.MjData(model)
        demand, inertia = free_space_demand(model, data, np.zeros(7), np.ones(7), np.zeros(7))
        np.testing.assert_allclose(demand, .2); np.testing.assert_allclose(inertia, 0.)

    def test_asymmetric_caps_and_negative_gears(self):
        model = independent_model(); model.actuator_gear[0, 0] = -2.
        bounds = joint_torque_bounds(model)
        np.testing.assert_allclose(bounds[0], [-4., 2.])
        np.testing.assert_allclose(bounds[1], [-1., 2.])

    def test_hold_has_zero_demand_and_never_admits_training(self):
        result = screen_curve(independent_model(), C2JointSpline([0., 1.], np.zeros((2, 7))))
        self.assertEqual(result['worst_utilization'], 0.)
        self.assertTrue(result['ideal_demand_within_caps_at_samples'])
        self.assertFalse(result['admitted_to_training'])

    def test_speedup_increases_inertial_demand(self):
        model = independent_model(); q = np.zeros((2, 7)); q[1, 0] = 1.
        slow = screen_curve(model, C2JointSpline([0., 2.], q))
        fast = screen_curve(model, C2JointSpline([0., .1], q), .001)
        self.assertGreater(fast['worst_utilization'], slow['worst_utilization'])
        self.assertFalse(fast['ideal_demand_within_caps_at_samples'])

    def test_no_joint_clipping_or_invalid_period(self):
        model = independent_model()
        with self.assertRaises(ValueError):
            screen_curve(model, C2JointSpline([0., 1.], np.full((2, 7), 4.)))
        with self.assertRaises(ValueError):
            screen_curve(model, C2JointSpline([0., 1.], np.zeros((2, 7))), 0)
        with self.assertRaises(ValueError):
            screen_curve(model, C2JointSpline([2., 3.], np.zeros((2, 7))))


if __name__ == '__main__':
    unittest.main()
