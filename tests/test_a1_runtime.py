"""Numerical opt-in must not quietly change the game or either robot."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from sim.a1_runtime import (apply_runtime_profile, assert_runtime_profile,
                            load_runtime_profile)
from sim.assembly import ROOT
from sim.duel_scene import build_duel_scene
from sim.skill_validation import V2_CONFIG


PROFILE = ROOT / "config/a1_runtime_rk4_candidate.json"


def fake_scene():
    return SimpleNamespace(data=SimpleNamespace(time=0.),
        model=SimpleNamespace(opt=SimpleNamespace(integrator=3, timestep=.002)),
        spec=SimpleNamespace(option=SimpleNamespace(integrator=3, timestep=.002)))


class RuntimeProfileTests(unittest.TestCase):
    def test_explicit_load_does_not_change_old_defaults(self):
        old = dict(V2_CONFIG)
        p = load_runtime_profile(PROFILE)
        self.assertEqual(V2_CONFIG, old)
        self.assertEqual(V2_CONFIG["physics_timestep_s"], .002)
        cfg = p.runtime_config()
        self.assertEqual(cfg["physics_timestep_s"], .001)
        self.assertEqual(cfg["explicit_integrator"], "RK4")
        self.assertEqual(cfg["rules_version"], V2_CONFIG["rules_version"])
        cfg["penetration_limit_m"] = 100.
        self.assertEqual(V2_CONFIG["penetration_limit_m"], .005)

    def test_profile_applies_and_checks_both_live_and_exported_options(self):
        p = load_runtime_profile(PROFILE)
        scene = fake_scene()
        apply_runtime_profile(scene, p)
        assert_runtime_profile(scene, p)
        for option in (scene.model.opt, scene.spec.option):
            option.integrator = 3
            with self.assertRaises(ValueError): assert_runtime_profile(scene, p)
            option.integrator = int(mujoco.mjtIntegrator.mjINT_RK4)
            option.timestep = .002
            with self.assertRaises(ValueError): assert_runtime_profile(scene, p)
            option.timestep = .001

    def test_invalid_or_late_profile_cannot_partially_mutate_settings(self):
        p = load_runtime_profile(PROFILE)
        for changes in ({"integrator":"implicitfast"}, {"physics_timestep_s":True},
                        {"physics_timestep_s":float("nan")}, {"control_period_s":.0101},
                        {"decision_period_s":.015}, {"name":""}, {"source_sha256":""}):
            scene = fake_scene()
            with self.assertRaises(ValueError):apply_runtime_profile(scene, replace(p, **changes))
            self.assertEqual(scene.model.opt.timestep, .002)
            self.assertEqual(scene.spec.option.integrator, 3)
        scene = fake_scene(); scene.data.time = .001
        with self.assertRaises(ValueError):apply_runtime_profile(scene, p)
        self.assertEqual(scene.model.opt.timestep, .002)

    def test_unknown_physical_override_or_bad_schema_is_rejected(self):
        base = json.loads(PROFILE.read_text())
        for changes in ({"force_caps_nm":[100]*7}, {"schema_version":True},
                        {"control_period_s":".01"}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"profile.json"
                path.write_text(json.dumps(dict(base, **changes)))
                with self.assertRaises(ValueError):load_runtime_profile(path)

    def test_hash_is_bound_to_fields_not_just_well_formed(self):
        p = load_runtime_profile(PROFILE)
        for changes in ({"physics_timestep_s":.002}, {"control_period_s":.02},
                        {"name":"different"}, {"source_sha256":"0"*64},
                        {"source_bytes":b"{}"}):
            changed = replace(p, **changes)
            with self.assertRaises(ValueError):changed.runtime_config()
            with self.assertRaises(ValueError):apply_runtime_profile(fake_scene(), changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"half_ms.json"
            source = json.loads(PROFILE.read_text())
            source.update(name="explicit_half_ms_test", physics_timestep_s=.0005)
            path.write_text(json.dumps(source))
            half = load_runtime_profile(path)
            self.assertNotEqual(half.source_sha256, p.source_sha256)
            self.assertEqual(apply_runtime_profile(fake_scene(), half)["physics_timestep_s"], .0005)

    def test_real_scene_preserves_states_forces_geometry_and_game(self):
        scene = build_duel_scene(1.75)
        names = ("body_mass", "body_inertia", "geom_size", "geom_pos", "geom_contype",
                 "geom_conaffinity", "actuator_gainprm", "actuator_biasprm",
                 "actuator_forcerange", "dof_damping", "dof_armature", "jnt_range")
        before = {k:getattr(scene.model,k).copy() for k in names}
        q, v = scene.data.qpos.copy(), scene.data.qvel.copy()
        old_config = json.dumps(scene.config, sort_keys=True)
        cfg = apply_runtime_profile(scene, load_runtime_profile(PROFILE))
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(scene.model,name), value)
        np.testing.assert_array_equal(scene.data.qpos, q)
        np.testing.assert_array_equal(scene.data.qvel, v)
        self.assertEqual(json.dumps(scene.config, sort_keys=True), old_config)
        for key in ("penetration_limit_m", "illegal_contact_force_threshold_n",
                    "minimum_blade_force_n", "minimum_blade_impulse_ns"):
            self.assertEqual(cfg[key], V2_CONFIG[key])
        self.assertIn('integrator="RK4"', scene.spec.to_xml())


if __name__ == "__main__":unittest.main()
