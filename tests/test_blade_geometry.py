import unittest

import mujoco
import numpy as np

from sim.blade_geometry import (adaptive_cylinder_sections, blade_profile_points,
    blade_cylinder_sections, blade_geometry_metadata, blade_collision_geom_ids)
from sim.blade_geometry import blade_geometry_identity


class ErrorBoundedProfileTests(unittest.TestCase):
    def test_nonmonotone_profile_is_covered_without_erasing_corners(self):
        points=np.array([[0.,.012],[.2,.022],[.4,.009],[.41,.015],[.5,.01]])
        sections=adaptive_cylinder_sections(points,.00025)
        self.assertEqual(sections[0].start_m,points[0,0])
        self.assertEqual(sections[-1].end_m,points[-1,0])
        for a,b in zip(sections[:-1],sections[1:]):
            self.assertEqual(a.end_m,b.start_m)
        for section in sections:
            samples=np.linspace(section.start_m,section.end_m,19)
            radii=np.interp(samples,points[:,0],points[:,1])
            self.assertTrue(np.all(section.radius_m-radii>=-1e-14))
            self.assertLessEqual(float(np.max(section.radius_m-radii)),.00025+1e-14)
        boundaries={s.start_m for s in sections}|{sections[-1].end_m}
        self.assertTrue(set(points[:,0]).issubset(boundaries))

    def test_actual_profile_has_bounded_105_sections_and_no_guard_inflation(self):
        points=blade_profile_points();sections=blade_cylinder_sections()
        self.assertEqual(len(sections),105)
        self.assertAlmostEqual(sections[0].radius_m,.016572382916666667)
        self.assertLess(sections[0].radius_m,.02)
        self.assertEqual(sections[0].start_m,points[0,0])
        self.assertEqual(sections[-1].end_m,points[-1,0])
        report=blade_geometry_metadata()
        self.assertLessEqual(report['realized_max_radial_excess_m'],.00025+1e-14)
        self.assertLess(report['collision_to_visible_surface_radial_bound_m'],.000277)
        self.assertFalse(report['old_geometry_certificates_inherited'])

    def test_constant_radius_is_not_blindly_subdivided(self):
        sections=adaptive_cylinder_sections([[0,.01],[10,.01]],.00001)
        self.assertEqual(len(sections),1)

    def test_bad_profiles_budgets_and_cost_bound_fail_closed(self):
        for p in ([[0,.01],[0,.02]],[[1,.01],[0,.02]],[[0,.01],[1,0]],[[0,np.nan],[1,.02]]):
            with self.assertRaises(ValueError):adaptive_cylinder_sections(p)
        for budget in (0,-1,float('inf'),float('nan'),True,'0.01'):
            with self.assertRaises(ValueError):adaptive_cylinder_sections([[0,.01],[1,.02]],budget)
        with self.assertRaises(ValueError):
            adaptive_cylinder_sections([[0,.01],[1,.02]],.00001,max_sections=5)


class NativeEnumerationTests(unittest.TestCase):
    def model(self,n=12,missing=None,kind='cylinder'):
        names=[i for i in reversed(range(n))if i!=missing]
        geoms=''.join(f'<geom name="a1_shinai_blade_collision_{i}" type="{kind}" size=".01 .05" pos="0 0 {i*.1}"/>'for i in names)
        return mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+geoms+'</worldbody></mujoco>')

    def test_numeric_order_beyond_ten_and_legacy_three(self):
        for n in (3,12):
            model=self.model(n);ids=blade_collision_geom_ids(model,'a1_')
            self.assertEqual([model.geom(i).name for i in ids],
                [f'a1_shinai_blade_collision_{i}' for i in range(n)])
            self.assertIs(ids,blade_collision_geom_ids(model,'a1_'))

    def test_missing_and_wrong_types_fail(self):
        for model in (self.model(missing=2),self.model(kind='capsule'),self.model(n=0)):
            with self.assertRaises(ValueError):blade_collision_geom_ids(model,'a1_')

    def test_compiled_cylinders_share_end_planes(self):
        sections=blade_cylinder_sections()
        geoms=''.join(f'<geom name="shinai_blade_collision_{i}" type="cylinder" size="{s.radius_m} {s.half_length_m}" pos="0 0 {s.center_m}"/>'for i,s in enumerate(sections))
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+geoms+'</worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        ids=blade_collision_geom_ids(m)
        for a,b in zip(ids[:-1],ids[1:]):
            self.assertAlmostEqual(d.geom_xpos[a,2]+m.geom_size[a,1],
                                   d.geom_xpos[b,2]-m.geom_size[b,1],places=14)
        self.assertAlmostEqual(d.geom_xpos[ids[0],2]-m.geom_size[ids[0],1],sections[0].start_m)
        self.assertAlmostEqual(d.geom_xpos[ids[-1],2]+m.geom_size[ids[-1],1],sections[-1].end_m)

    def test_identity_strips_slot_prefix_and_ignores_world_pose(self):
        parts=[]
        for slot,x in [('a1',0),('a2',1.75)]:
            parts.append(f'<body name="{slot}" pos="{x} 0 0"><geom name="{slot}_shinai_blade_collision_0" type="cylinder" size=".01 .1" pos="0 0 .1"/></body>')
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(parts)+'</worldbody></mujoco>')
        a=blade_geometry_identity(m,'a1_');b=blade_geometry_identity(m,'a2_')
        self.assertEqual(a,b)
        m.geom_size[blade_collision_geom_ids(m,'a2_')[0],0]+=.001
        self.assertNotEqual(a['sha256'],blade_geometry_identity(m,'a2_')['sha256'])


if __name__=='__main__':unittest.main()
