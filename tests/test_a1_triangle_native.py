import unittest

import numpy as np
import mujoco
from types import SimpleNamespace
from unittest.mock import patch

from sim.a1_triangle_native import scheduled_programs, event_velocity, checked_kinematic_updater


class TriangleNativeTests(unittest.TestCase):
    def test_geometry_only_update_matches_full_forward_on_both_slots(self):
        bodies=''.join('<body pos="'+str(i)+' 0 0"><joint type="hinge"/>'+''.join(
            f'<geom name="{slot}_shinai_blade_collision_{j}" type="cylinder" size=".01 .1" pos="0 0 {j*.2}"/>'
            for j in range(3))+'</body>' for i,slot in enumerate(('a1','a2')))
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies+'</worldbody></mujoco>')
        scene=SimpleNamespace(model=model,data=mujoco.MjData(model),agents={s:{'prefix':s+'_'} for s in ('a1','a2')})
        update,report=checked_kinematic_updater(scene,np.array([[0,0],[.3,-.2],[-.1,.8]]))
        self.assertIs(update,mujoco.mj_kinematics)
        self.assertEqual(report['kinematics_vs_full_forward_max_endpoint_difference_m'],0.)
        self.assertEqual(report['endpoint_equivalence_states_tested'],3)
        self.assertTrue(report['all_native_frames_still_scored'])

    def test_geometry_mismatch_uses_full_forward_not_a_fast_false_pass(self):
        scene=SimpleNamespace(model=object(),data=SimpleNamespace(qpos=np.zeros(1)))
        first=SimpleNamespace(start=np.zeros(3),end=np.zeros(3))
        second=SimpleNamespace(start=np.ones(3),end=np.zeros(3))
        snapshots=[{'a1':[first],'a2':[first]},{'a1':[second],'a2':[first]}]
        with patch('sim.a1_triangle_native.mujoco.mj_kinematics'),patch('sim.a1_triangle_native.mujoco.mj_forward') as full,\
             patch('sim.a1_triangle_native.blade_sections_from_scene',side_effect=snapshots):
            update,report=checked_kinematic_updater(scene,np.zeros((1,1)))
            self.assertIs(update,full)
            self.assertEqual(report['geometry_update'],'mj_forward_fallback')

    def test_measured_direction_rejects_bad_axis_before_loading(self):
        for direction in ([0,0,0],[1,0],[np.nan,0,1]):
            with self.assertRaises(ValueError):
                event_velocity('missing',{},direction)

    def test_retime_does_not_drop_recovery_or_settle_tail(self):
        nodes = np.array([np.ones(7),np.ones(7)+.2,np.ones(7)+.1])
        action={'knot_time_s':[0.,1.,2.], 'knot_qpos':nodes.tolist()}
        programs,duration=scheduled_programs(action,np.ones(7),time_scale=2.,settle_s=2.)
        self.assertGreaterEqual(duration, programs['a1'].end_s+2.)
        np.testing.assert_array_equal(programs['a1'].target(duration),nodes[-1])
        self.assertEqual(programs['a1'].curve.duration_s,4.)

    def test_invalid_budget_and_scale_rejected(self):
        action={'knot_time_s':[0.,2.], 'knot_qpos':[np.ones(7),np.ones(7)]}
        for scale in (0.,-1.,np.nan,np.inf,True):
            with self.assertRaises(ValueError):
                scheduled_programs(action,np.ones(7),time_scale=scale)
        with self.assertRaises(ValueError):
            scheduled_programs(action,np.ones(7),time_scale=30.,settle_s=2.)

    def test_hold_is_independent_of_moving_reference(self):
        ready=np.ones(7)
        action={'knot_time_s':[0.,1.], 'knot_qpos':[ready,ready+.2]}
        programs,_=scheduled_programs(action,ready)
        np.testing.assert_array_equal(programs['a2'].target(1.75),ready)
        np.testing.assert_array_equal(programs['a1'].target(1.75),ready+.2)


if __name__ == '__main__':
    unittest.main()
