"""Floor-mount geometry preservation and assumption checks, not hardware tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.assembly import ARM_BITS, ENV_BIT, TOOL_BIT, export_scene
from sim.floor import CAD_TRANSLATION_M, build_floor_assembly, floor_build_report


ROOT = Path(__file__).resolve().parents[1]


def object_id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise AssertionError(f"Missing {kind.name} {name}")
    return i


class FloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assembly = build_floor_assembly()
        cls.model, cls.data = cls.assembly.model, cls.assembly.data
        cls.source = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
        cls.source_data = mujoco.MjData(cls.source)
        mujoco.mj_forward(cls.source, cls.source_data)

    def test_fixed_zero_pose_no_humanoid_and_vertical_J0(self):
        m, d = self.model, self.data
        self.assertEqual((m.nq, m.nv, m.nu), (7, 7, 7))
        self.assertFalse(np.any(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"), -1)
        self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "head"), -1)
        self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mannequin_foot"), -1)
        np.testing.assert_array_equal(d.qpos, np.zeros(7))
        np.testing.assert_array_equal(self.assembly.home, np.zeros(7))
        np.testing.assert_allclose(m.opt.gravity, [0, 0, -9.81])
        j0 = object_id(m, mujoco.mjtObj.mjOBJ_JOINT, "J0")
        np.testing.assert_allclose(np.abs(d.xaxis[j0]), [0, 0, 1], atol=3e-6)
        site = object_id(m, mujoco.mjtObj.mjOBJ_SITE, "floor_mount")
        np.testing.assert_allclose(d.site_xpos[site], [0, 0, 0], atol=1e-12)

    def test_all_source_mesh_arrays_scales_instances_and_inertias_preserved(self):
        m, d, old, od = self.model, self.data, self.source, self.source_data
        for old_mesh in range(old.nmesh):
            name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_MESH, old_mesh)
            mesh = object_id(m, mujoco.mjtObj.mjOBJ_MESH, name)
            np.testing.assert_array_equal(m.mesh_scale[mesh], old.mesh_scale[old_mesh])
            for field, adr, number in (("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                                       ("mesh_face", "mesh_faceadr", "mesh_facenum")):
                a, n = int(getattr(old, adr)[old_mesh]), int(getattr(old, number)[old_mesh])
                b = int(getattr(m, adr)[mesh])
                self.assertEqual(n, int(getattr(m, number)[mesh]))
                np.testing.assert_array_equal(getattr(m, field)[b:b+n], getattr(old, field)[a:a+n])
        candidates = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        source_count = 0
        fixed_bottom = []
        for i in range(old.ngeom):
            if old.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            source_count += 1
            mesh_name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_MESH, int(old.geom_dataid[i]))
            expected = od.geom_xpos[i] + CAD_TRANSLATION_M
            matching = [j for j in candidates if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[j])) == mesh_name]
            self.assertTrue(matching, f"Missing geom instance for {mesh_name}")
            j = min(matching, key=lambda k: np.linalg.norm(d.geom_xpos[k]-expected))
            np.testing.assert_allclose(d.geom_xpos[j], expected, atol=1e-12)
            np.testing.assert_allclose(d.geom_xmat[j], od.geom_xmat[i], atol=1e-12)
            self.assertEqual(int(m.geom_contype[j]), int(old.geom_contype[i]))
            self.assertEqual(int(m.geom_conaffinity[j]), int(old.geom_conaffinity[i]))
            if int(old.geom_bodyid[i]) == 0:
                mesh = int(m.geom_dataid[j])
                a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                points = m.mesh_vert[a:a+n] @ d.geom_xmat[j].reshape(3, 3).T + d.geom_xpos[j]
                fixed_bottom.append(float(points[:, 2].min()))
                self.assertEqual(int(m.body_weldid[m.geom_bodyid[j]]), 0)
            candidates.remove(j)
        self.assertEqual(source_count, 140)
        self.assertAlmostEqual(min(fixed_bottom), 0., places=7)
        for i in range(1, old.nbody):
            name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_BODY, i)
            j = object_id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            for field in ("body_mass", "body_ipos", "body_inertia"):
                np.testing.assert_array_equal(getattr(m, field)[j], getattr(old, field)[i])
        self.assertAlmostEqual(float(m.body_mass.sum()-old.body_mass.sum()), .536, places=9)

    def test_sword_lengths_caps_and_debug_windows(self):
        m, d = self.model, self.data
        def pos(name):
            return d.site_xpos[object_id(m, mujoco.mjtObj.mjOBJ_SITE, name)]
        self.assertAlmostEqual(float(np.linalg.norm(pos("shinai_tip")-pos("shinai_pommel"))), 1.08, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(pos("shinai_tip")-pos("shinai_grip"))), .79, places=9)
        half = np.asarray(self.assembly.config["runtime"]["joint_half_windows_rad"])
        np.testing.assert_allclose(m.jnt_range, np.column_stack((-half, half)))
        caps = np.array([5, 80, 15, 20, 3, 3, 2])
        np.testing.assert_array_equal(m.actuator_forcerange, np.column_stack((-caps, caps)))
        self.assertTrue(np.all(m.actuator_forcelimited))

    def test_floor_and_base_masks_retain_moving_contact_eligibility(self):
        m = self.model
        floor = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        base = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "base_collision")
        for geom in (floor, base):
            self.assertEqual(int(m.geom_contype[geom]), ENV_BIT)
            self.assertEqual(int(m.geom_conaffinity[geom]), ARM_BITS | TOOL_BIT)
        for i in range(7):
            g = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"arm_proxy_{i}")
            self.assertTrue(m.geom_contype[g] & m.geom_conaffinity[floor])
            self.assertTrue(m.geom_conaffinity[g] & ENV_BIT)
            for j in range(max(0, i-1), min(7, i+2)):
                self.assertFalse(m.geom_conaffinity[g] & (1 << j))
        self.assertTrue(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))

    def test_hypothetical_ranges_are_labelled_and_source_limited(self):
        ranges = np.tile([-np.pi, np.pi], (7, 1))
        assembly = build_floor_assembly(ranges)
        np.testing.assert_allclose(assembly.model.jnt_range, ranges)
        self.assertIn("hypothetical", assembly.config["floor"]["joint_range_status"])
        # Exercise source intersection without editing the original XML file.
        from sim import floor
        original_builder = floor._floor_scene
        def source_limited(config):
            spec = original_builder(config)
            spec.joint("J1").limited = True
            spec.joint("J1").range = [-.2, .3]
            return spec
        with patch("sim.floor._floor_scene", side_effect=source_limited):
            constrained = build_floor_assembly(ranges)
        np.testing.assert_allclose(constrained.model.jnt_range[1], [-.2, .3])
        for invalid in (np.zeros((6, 2)), np.tile([1., 2.], (7, 1)), np.tile([np.nan, 1.], (7, 1))):
            with self.assertRaises(ValueError):
                build_floor_assembly(invalid)

    def test_report_and_export_are_not_a_dynamics_pass(self):
        report = floor_build_report(self.assembly)
        self.assertEqual(report["physics_validation"], "not_run")
        self.assertNotEqual(report["status"], "passed")
        self.assertEqual(report["initial_contacts"], [
            {"geom1": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
             "geom2": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)),
             "distance_m": float(c.dist)} for c in self.data.contact])
        self.assertTrue(all(report["source_body_mass_COM_inertia_unchanged"].values()))
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/"floor.xml"
            export_scene(self.assembly, filename)
            exported = mujoco.MjModel.from_xml_path(str(filename))
            self.assertEqual(exported.nq, 7)
            np.testing.assert_allclose(exported.jnt_range, self.model.jnt_range)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
