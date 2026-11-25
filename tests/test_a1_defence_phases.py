import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from sim.a1_defence_phases import split_hold, merge_graphs
from sim.motion_spline import C2JointSpline


class DefencePhasesTests(unittest.TestCase):
    def test_only_actual_rest_hold_is_split_without_changing_motion(self):
        q = np.zeros((5, 7)); q[1] = .3; q[2:4] = .6
        curve = C2JointSpline([0., 1., 2., 4., 6.], q)
        enter, leave, hold, checks = split_hold(curve, 2)
        self.assertEqual(hold, 2.)
        self.assertTrue(all(checks.values()))
        np.testing.assert_array_equal(enter.knot_qpos[-1], leave.knot_qpos[0])
        self.assertEqual(leave.duration_s, 2.)

    def test_moving_knot_is_not_an_interrupt_point(self):
        q = np.zeros((4, 7)); q[1] = .3; q[2] = .6
        with self.assertRaises(ValueError):
            split_hold(C2JointSpline([0., 1., 2., 3.], q), 1)

    def test_initial_or_terminal_hold_is_not_a_complete_split(self):
        curve = C2JointSpline([0., 1., 2., 3.], np.zeros((4, 7)))
        for index in (0, 2, True):
            with self.assertRaises(ValueError): split_hold(curve, index)

    def test_merge_rejects_foreign_scoring_or_different_hold_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            left={'scoring':{'version':1}, 'guards':{'L':{'qpos_rad':[0.]*7}}, 'actions':{},
                  'phase_split':{'hold_node':'L','enter':'enterL','exit':'exitL'}}
            right={'scoring':{'version':1}, 'guards':{'R':{'qpos_rad':[1.]*7}}, 'actions':{},
                   'phase_split':{'hold_node':'R'}}
            bridge={'scoring':{'version':1}, 'guards':{'left_parry_hold':{'qpos_rad':[0.]*7},
                    'right_parry_hold':{'qpos_rad':[1.]*7}}, 'actions':{}}
            paths=[root/name for name in ('left.json','right.json','bridge.json')]
            for path,value in zip(paths,(left,right,bridge)):path.write_text(json.dumps(value))
            merged=merge_graphs(*paths,root/'valid')
            self.assertEqual(merged['bilateral_phase_graph']['hold_aliases_checked_exactly']['left_parry_hold'],'L')
            bridge['scoring']['version']=2;paths[2].write_text(json.dumps(bridge))
            with self.assertRaisesRegex(ValueError,'same scoring'):merge_graphs(*paths,root/'bad_rule')
            bridge['scoring']['version']=1;bridge['guards']['left_parry_hold']['qpos_rad'][0]=.001
            paths[2].write_text(json.dumps(bridge))
            with self.assertRaisesRegex(ValueError,'does not equal'):merge_graphs(*paths,root/'bad_pose')


if __name__ == '__main__':
    unittest.main()
