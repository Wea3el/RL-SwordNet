"""Small adapter regressions for first-material-contact episode boundaries.

Synthetic contact frames deliberately control manifold identity and force-gate
timing. They are not native fencing, force attribution calibration, or exact
impact-time evidence. A separate tiny XML checks the sphere surface-witness
convention used by the monitor; no long native rollout is performed here.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.training_contacts import (
    TrainingContactMonitor, TrainingEnvironmentError, settle_training_tick)


DT = .001
CFG = {"physics_timestep_s": DT,
       "illegal_contact_force_threshold_n": .5}


class ContactFrame:
    """Copyable test state: unlike a shallow namespace, arrays never alias."""
    def __init__(self, model):
        self.time = 0.
        self.contact = []
        self.xmat = np.tile(np.eye(3).reshape(1, 9), (model.nbody, 1))
        self.xpos = np.zeros((model.nbody, 3))

    def __copy__(self):
        result = object.__new__(type(self))
        result.time = self.time
        result.contact = copy.deepcopy(self.contact)
        result.xmat = self.xmat.copy()
        result.xpos = self.xpos.copy()
        return result


class EpisodeFixture:
    """Real geom/body IDs, synthetic contact data and controlled first Jv."""
    def __init__(self, *, initial_contacts=()):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="a1_arm_body"><geom name="a1_arm_proxy_0" size=".1"/></body>
          <body name="a2_arm_body"><geom name="a2_arm_proxy_0" size=".1"/></body>
          <body name="a1_weapon_body"><geom name="a1_shinai_blade_collision_0" size=".1"/></body>
          <body name="a2_weapon_body"><geom name="a2_shinai_guard_collision" size=".1"/></body>
        </worldbody></mujoco>''')
        self.scene = SimpleNamespace(model=model, data=ContactFrame(model))
        self.ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in ("a1_arm_proxy_0", "a2_arm_proxy_0",
                                 "a1_shinai_blade_collision_0",
                                 "a2_shinai_guard_collision")}
        self.arm_ids = (self.ids["a1_arm_proxy_0"], self.ids["a2_arm_proxy_0"])
        self.weapon_ids = (self.ids["a1_shinai_blade_collision_0"],
                           self.ids["a2_shinai_guard_collision"])
        self.arm_bodies = tuple(int(model.geom_bodyid[g]) for g in self.arm_ids)
        self.velocities = {self.arm_bodies[0]: np.array([.2, 0., 0.]),
                           self.arm_bodies[1]: np.zeros(3)}
        self.scene.data.contact = copy.deepcopy(list(initial_contacts))
        self.monitor = None

    def start(self):
        self.monitor = TrainingContactMonitor(self.scene, CFG)
        return self.monitor

    def contact(self, point=(0., 0., 0.), *, force=0., reverse=False,
                legal=False, distance=0.):
        geoms = self.weapon_ids if legal else self.arm_ids
        normal = np.array([1., 0., 0.])
        if reverse:
            geoms = geoms[::-1]
            normal = -normal
        return SimpleNamespace(geom1=geoms[0], geom2=geoms[1],
            dist=float(distance), pos=np.asarray(point, dtype=float).copy(),
            frame=np.r_[normal, [0., 1., 0., 0., 0., 1.]],
            test_normal_force=float(force))

    def force(self, _model, data, index, output):
        output[0] = data.contact[index].test_normal_force

    def velocity(self, _model, _before, _after, body, _point):
        return self.velocities.get(int(body), np.zeros(3)).copy()

    def advance(self, contacts, *, positions=None):
        self.monitor.capture_before_step(self.scene)
        self.scene.data.time += DT
        self.scene.data.contact = copy.deepcopy(list(contacts))
        if positions:
            for body, point in positions.items():
                self.scene.data.xpos[body] = point
        return self.monitor.inspect(self.scene, self.monitor._last_tick+1)


class ContactEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = EpisodeFixture()
        self.force_patch = patch("sim.training_contacts.mujoco.mj_contactForce",
                                 side_effect=self.fixture.force)
        self.velocity_patch = patch(
            "sim.training_contacts.previous_material_point_velocity",
            side_effect=self.fixture.velocity)
        self.force_patch.start()
        self.velocity_mock = self.velocity_patch.start()
        self.addCleanup(self.force_patch.stop)
        self.addCleanup(self.velocity_patch.stop)
        self.fixture.start()

    def assert_unusable(self, callback):
        with self.assertRaises(TrainingEnvironmentError) as context:
            callback()
        self.assertEqual(context.exception.details["status"], "evidence_error")
        self.assertEqual(context.exception.details["reason"],
                         "unusable_first_contact_manifold")
        return context.exception.details

    def test_one_a_then_a_b_then_only_b_cannot_reuse_disappeared_a(self):
        f = self.fixture
        self.assertFalse(f.advance([f.contact()])["verdicts"])
        self.assertFalse(f.advance([f.contact(), f.contact((0., .1, 0.))])["verdicts"])
        error = self.assert_unusable(lambda: f.advance([
            f.contact((0., .1, 0.), force=1.)]))
        self.assertEqual(error["physics_tick"], 3)
        self.assertTrue(error["episode"]["manifold_identity_lost"])

    def test_same_count_replacement_is_sticky_even_if_original_returns(self):
        f = self.fixture
        f.advance([f.contact()])
        f.advance([f.contact((0., .1, 0.))])
        error = self.assert_unusable(lambda: f.advance([f.contact(force=1.)]))
        self.assertEqual(error["episode"]["first_manifold_size"], 1)
        self.assertTrue(error["episode"]["manifold_identity_lost"])

    def test_first_manifold_point_reordering_retains_every_cached_point(self):
        f = self.fixture
        a, b = (0., -.1, 0.), (0., .1, 0.)
        f.advance([f.contact(a), f.contact(b)])
        first = f.monitor.summary()["first_contact_episodes"][0]["evidence"]
        result = f.advance([f.contact(b, force=1.), f.contact(a, force=1.)])
        self.assertEqual(len(result["verdicts"]), 2)
        self.assertEqual({v["contact_id"] for v in result["verdicts"]},
                         {v["contact_id"] for v in first})
        self.assertTrue(all(v["physics_tick"] == 2 for v in result["verdicts"]))
        self.assertTrue(all(v["responsible_owners"] == ["a1"]
                            for v in result["verdicts"]))
        self.assertEqual(self.velocity_mock.call_count, 4)

    def test_zero_force_then_strict_force_gate_uses_original_preimpact_tick(self):
        f = self.fixture
        f.advance([f.contact()])
        cached = f.monitor.summary()["first_contact_episodes"][0]["evidence"][0]
        f.velocities[f.arm_bodies[0]] = np.array([-3., 0., 0.])
        self.assertFalse(f.advance([f.contact(force=.5)])["verdicts"])
        result = f.advance([f.contact(force=.500001)])
        item = result["verdicts"][0]
        self.assertEqual(item["physics_tick"], 3)
        self.assertEqual(item["contact_id"], cached["contact_id"])
        self.assertEqual(cached["physics_tick"], 1)
        self.assertEqual(item["responsible_owners"], ["a1"])
        self.assertAlmostEqual(item["normal_contribution_m_s"]["geom_a"], .2)
        self.assertEqual(self.velocity_mock.call_count, 2)

    def test_separation_creates_new_episode_and_fresh_material_evidence(self):
        f = self.fixture
        f.advance([f.contact()])
        f.advance([])
        f.velocities[f.arm_bodies[0]] = np.zeros(3)
        f.velocities[f.arm_bodies[1]] = np.array([-.2, 0., 0.])
        result = f.advance([f.contact((0., .1, 0.), force=1.)])
        self.assertEqual(result["verdicts"][0]["responsible_owners"], ["a2"])
        self.assertTrue(result["verdicts"][0]["contact_id"].startswith("episode2:"))
        self.assertEqual(len(f.monitor.summary()["first_contact_episodes"]), 2)
        self.assertEqual(self.velocity_mock.call_count, 4)

    def test_geom_order_and_normal_flip_keep_canonical_material_identity(self):
        f = self.fixture
        f.advance([f.contact((0., .1, 0.))])
        result = f.advance([f.contact((0., .1, 0.), force=1., reverse=True)])
        self.assertEqual(result["contacts"][0]["geom1"], "a2_arm_proxy_0")
        item = result["verdicts"][0]
        self.assertEqual(item["owners"], ["a1", "a2"])
        self.assertEqual(item["responsible_owners"], ["a1"])
        self.assertEqual(self.velocity_mock.call_count, 2)

    def test_same_tick_legal_contact_order_cannot_change_ambiguous_foul(self):
        f = self.fixture
        results = []
        for reverse in (False, True):
            f.scene.data = ContactFrame(f.scene.model)
            f.start()
            contacts = [f.contact(force=1.), f.contact(legal=True, force=1.)]
            if reverse:
                contacts.reverse()
            report = f.advance(contacts)
            self.assertEqual(len(report["verdicts"]), 1)
            self.assertTrue(report["verdicts"][0]["prior_legal_weapon_influence"])
            result = settle_training_tick(1, contacts=report["verdicts"])
            results.append((result["status"], result["match_winner"], result["rl_reward"]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], ("ambiguous_foul", None, {"a1": -1., "a2": -1.}))

    def test_legal_influence_between_first_geometry_and_force_gate_is_retained(self):
        f = self.fixture
        f.advance([f.contact()])
        f.advance([f.contact(), f.contact(legal=True, force=1.)])
        result = f.advance([f.contact(force=1.)])
        self.assertEqual(result["verdicts"][0]["status"], "ambiguous_foul")
        self.assertTrue(result["verdicts"][0]["prior_legal_weapon_influence"])

    def test_reset_seeds_positive_legal_force_history(self):
        f = self.fixture
        f.scene.data.contact = [f.contact(legal=True, force=1.)]
        f.start()
        self.assertTrue(f.monitor.summary()["round_legal_influence"])
        result = f.advance([f.contact(force=1.)])
        self.assertEqual(result["verdicts"][0]["status"], "ambiguous_foul")

    def test_initial_overlap_cannot_acquire_invented_preimpact(self):
        f = self.fixture
        f.scene.data.contact = [f.contact(distance=-.0001)]
        f.start()
        error = self.assert_unusable(lambda: f.advance([
            f.contact(distance=-.0001, force=1.)]))
        self.assertTrue(error["episode"]["initial_overlap"])

    def test_normal_compression_preserves_surface_material_points(self):
        f = self.fixture
        a, b = f.arm_bodies
        # Unit-radius-independent sphere geometry: centres +/-0.1, radius0.1.
        f.scene.data.xpos[a] = [-.1, 0., 0.]
        f.scene.data.xpos[b] = [.1, 0., 0.]
        f.advance([f.contact()])
        result = f.advance([f.contact(distance=-.002, force=1.)],
                           positions={a: [-.099, 0., 0.], b: [.099, 0., 0.]})
        self.assertEqual(result["verdicts"][0]["status"], "single_foul")
        self.assertFalse(f.monitor._episodes[tuple(sorted(f.arm_ids))]
                         ["manifold_identity_lost"])

    def test_nonfinite_time_is_rejected_at_every_collection_boundary(self):
        f = self.fixture
        for boundary in ("reset", "capture", "inspect_now", "inspect_before", "last_time"):
            with self.subTest(boundary=boundary):
                f.scene.data = ContactFrame(f.scene.model)
                f.start()
                with self.assertRaises(TrainingEnvironmentError):
                    if boundary == "reset":
                        f.scene.data.time = np.nan
                        f.start()
                    elif boundary == "capture":
                        f.scene.data.time = np.nan
                        f.monitor.capture_before_step(f.scene)
                    elif boundary == "last_time":
                        f.monitor._last_time = np.nan
                        f.monitor.capture_before_step(f.scene)
                    else:
                        f.monitor.capture_before_step(f.scene)
                        f.scene.data.time = DT
                        if boundary == "inspect_now":
                            f.scene.data.time = np.nan
                        else:
                            f.monitor._before.time = np.nan
                        f.monitor.inspect(f.scene, 1)


class SurfaceWitnessConventionTests(unittest.TestCase):
    def test_native_sphere_midpoint_signed_normal_distance_recovers_surfaces(self):
        # Native primitive convex sphere contact, not arbitrary mesh manifolds.
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body pos="-.095 0 0"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".1" mass="1"/></body>
          <body pos=".095 0 0"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".1" mass="1"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 1)
        contact = data.contact[0]
        normal = np.asarray(contact.frame[:3])
        points = (contact.pos-.5*contact.dist*normal,
                  contact.pos+.5*contact.dist*normal)
        for geom, point in zip((contact.geom1, contact.geom2), points):
            self.assertAlmostEqual(float(np.linalg.norm(point-data.geom_xpos[geom])),
                                   float(model.geom_size[geom, 0]), places=12)
        np.testing.assert_allclose(points[1]-points[0], contact.dist*normal,
                                   rtol=0., atol=1e-15)

    def test_native_parallel_cylinder_nonunique_witness_is_rejected_not_relabelled(self):
        # Forward-only geometry fixture: a convex cylinder's line-contact
        # witness can move even during normal compression. The signed witness
        # formula does not imply unique or persistent material-point identity.
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0" timestep=".001"/><worldbody>
          <body pos="-.051 0 0"><joint type="slide" axis="1 0 0"/>
            <geom name="a1_arm_proxy_0" type="cylinder" size=".05 .2" mass="1"/></body>
          <body pos=".051 0 0"><joint type="slide" axis="1 0 0"/>
            <geom name="a2_arm_proxy_0" type="cylinder" size=".05 .2" mass="1"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:] = [.2, 0.]
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 0)
        scene = SimpleNamespace(model=model, data=data)
        monitor = TrainingContactMonitor(scene, CFG)
        level = [0.]

        def controlled_force(_model, _data, _index, output):
            output[0] = level[0]

        with patch("sim.training_contacts.mujoco.mj_contactForce",
                   side_effect=controlled_force):
            monitor.capture_before_step(scene)
            data.qpos[:] = [.0015, -.0015]
            data.time = DT
            mujoco.mj_forward(model, data)
            self.assertEqual(data.ncon, 1)
            self.assertFalse(monitor.inspect(scene, 1)["verdicts"])
            monitor.capture_before_step(scene)
            data.qpos[:] = [.001501, -.001501]
            data.time = 2*DT
            level[0] = 1.
            mujoco.mj_forward(model, data)
            self.assertEqual(data.ncon, 1)
            with self.assertRaises(TrainingEnvironmentError) as context:
                monitor.inspect(scene, 2)
        self.assertEqual(context.exception.details["reason"],
                         "unusable_first_contact_manifold")
        self.assertTrue(context.exception.details["episode"]["manifold_identity_lost"])


if __name__ == "__main__":
    unittest.main()
