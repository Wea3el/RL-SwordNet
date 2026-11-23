import copy
import unittest

import mujoco
import numpy as np

from sim.a1_flow import audit_flow, canonical_graph_nodes, compile_connected_route, load_requirements
from sim.joint_constraints import load_joint_constraints
from sim.motion_spline import C2JointSpline


class FlowTests(unittest.TestCase):
    def test_requirements_preserve_ten_named_guards_and_five_cuts(self):
        cfg = load_requirements()
        self.assertEqual(len(cfg['poses']), 10)
        self.assertEqual(len(cfg['cuts']), 5)
        self.assertIsNone(cfg['poses']['kron']['axis'])
        self.assertGreater(cfg['poses']['plow_left']['grip_m'][1], 0)
        self.assertLess(cfg['poses']['plow_right']['grip_m'][1], 0)
        self.assertIn('no unit', cfg['interpretations']['speed'])

    def test_route_passes_waypoint_without_return_or_hold(self):
        curve = C2JointSpline([0., 2., 4.], np.repeat([[0.], [.3], [.8]], 7, axis=1))
        np.testing.assert_array_equal(curve.sample(2.), np.full(7, .3))
        self.assertTrue(np.all(curve.derivatives(2.)[0] > 0))
        self.assertEqual(curve.duration_s, 4.)
        self.assertEqual(curve.report()['hold_duration_s'], 0.)

    def test_graph_nodes_never_use_an_independent_static_ik_branch(self):
        hub = np.full(7, .5)
        poses = {'long_point': {'qpos_rad': hub, 'usable_candidate': True},
                 'a': {'qpos_rad': np.full(7, 4.), 'usable_candidate': True},
                 'failed': {'qpos_rad': np.zeros(7), 'usable_candidate': False}}
        spoke = np.linspace(hub, np.ones(7), 5)
        nodes = canonical_graph_nodes(poses, {'a': spoke})
        np.testing.assert_array_equal(nodes['a']['qpos_rad'], spoke[-1])
        self.assertEqual(nodes['a']['independent_static_solution_max_delta_rad'], 3.)
        self.assertNotIn('failed', nodes)
        mismatch = spoke.copy(); mismatch[0, 0] += .001
        with self.assertRaises(ValueError): canonical_graph_nodes(poses, {'a': mismatch})
        with self.assertRaises(ValueError): canonical_graph_nodes(poses, {'failed': spoke})

    def test_connected_cut_has_exact_shared_hub_and_reversible_endpoints(self):
        hub = np.zeros(7)
        spokes = {'a': np.linspace(hub, np.ones(7), 5),
                  'b': np.linspace(hub, -np.ones(7), 5)}
        curve = compile_connected_route(spokes, ['a', 'long_point', 'b'], 2.)
        np.testing.assert_array_equal(curve.sample(2.), hub)
        self.assertEqual(curve.duration_s, 4.)
        reverse = compile_connected_route(spokes, ['b', 'long_point', 'a'], 2.)
        np.testing.assert_allclose(curve.sample(np.linspace(0, 4, 81)),
            reverse.sample(np.linspace(4, 0, 81)), atol=1e-12)
        bad = copy.deepcopy(spokes); bad['b'][0, 0] = .001
        with self.assertRaises(ValueError): compile_connected_route(bad, ['a', 'long_point', 'b'], 2.)

    def test_full_curve_audit_keeps_actual_limits_and_no_physics_claim(self):
        xml = '<mujoco><worldbody>'
        for i in range(7): xml += f'<body pos="0 0 .1"><joint name="J{i}" type="hinge" axis="0 0 1"/><geom size=".02" mass="1" contype="0" conaffinity="0"/>'
        xml += '<site name="shinai_tip" pos=".1 0 0"/>'+'</body>'*7+'</worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        q = np.array([.5, .5, .5, .5, .5, .5, 0.])
        report, trace = audit_flow(model, C2JointSpline([0., .1], [q, q+.01]), load_joint_constraints())
        self.assertTrue(report['angle_bounds_entire_curve'])
        self.assertFalse(report['native_physics_tested'])
        self.assertFalse(report['requested_velocity_gt_1_checked'])
        self.assertEqual(trace['qpos'].shape, (11, 7))


if __name__ == '__main__': unittest.main()
