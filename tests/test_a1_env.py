"""Short native stepping and synthetic settlement, not fencing admission."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.a1_env import A1DuelEnv
from sim.a1_runtime import load_runtime_profile
from sim.assembly import ROOT
from sim.foul_attribution import ContactEvidence, attribute_contact
from sim.shared_skills import SkillPath
from sim.skill_validation import V2_CONFIG
from sim.training_contacts import TrainingEnvironmentError


PROFILE = ROOT/'config/a1_runtime_rk4_candidate.json'


def library_fixture():
    guard = np.zeros(7)
    skills = {}
    for i, name in enumerate(('guard', 'stab', 'slash_left', 'parry_left', 'parry_right')):
        skills[name] = SkillPath(name, i, np.vstack((guard, guard+i*.001, guard)),
            np.array([0., .5, 1.]), ['out', 'return'],
            [{'name': 'out', 'start_s': 0., 'end_s': .5},
             {'name': 'return', 'start_s': .5, 'end_s': 1.}])
    return SimpleNamespace(guard_q=guard, skills=skills,
        reports={'library': {'all_skills_pass_geometry': True}}, config={})


def short_native_scene(contact_test=False):
    """Tiny native14-hinge model preserving the public adapter schema.

    All fixture geoms are noncolliding and targets are deliberately distant.
    This tests clock/control/contracts, NOT original robot skill validity.
    """
    bodies, actuators = [], []
    for slot, x in (('a1', 0.), ('a2', 2.5)):
        part = [f'<body name="{slot}_base" pos="{x} 0 0">']
        for i in range(7):
            part.append(f'<body name="{slot}_link{i}" pos="0 0 .08">'
                f'<joint name="{slot}_J{i}" axis="0 1 0" range="-3.14 3.14"/>'
                '<inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>')
            actuators.append(f'<position name="{slot}_J{i}" joint="{slot}_J{i}" kp="200" kv="10" '
                'forcerange="-5 5" ctrlrange="-3.14 3.14"/>')
        part.extend((f'<site name="{slot}_shinai_grip" pos="0 0 0"/>',
                     f'<site name="{slot}_shinai_tip" pos=".6 0 0"/>'))
        for i in range(3):
            mask = 1 if contact_test and slot == 'a1' and i == 0 else 0
            part.append(f'<geom name="{slot}_shinai_blade_collision_{i}" type="cylinder" '
                f'pos="{.1+.2*i} 0 0" size=".01 .1" quat=".7071067811865476 0 .7071067811865475 0" '
                f'contype="{mask}" conaffinity="{mask}"/>')
        mask = 1 if contact_test and slot == 'a2' else 0
        part.append(f'<geom name="{slot}_arm_proxy_0" type="cylinder" pos=".1 0 0" '
            f'size=".01 .1" quat=".7071067811865476 0 .7071067811865475 0" contype="{mask}" conaffinity="{mask}"/>')
        part.extend(['</body>']*8)
        bodies.append(''.join(part))
    spec = mujoco.MjSpec.from_string('<mujoco><compiler angle="radian"/>'
        '<option gravity="0 0 0"/><worldbody>'+''.join(bodies)+
        '</worldbody><actuator>'+''.join(actuators)+'</actuator></mujoco>')
    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    agents = {}
    for slot in ('a1', 'a2'):
        joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{slot}_J{i}') for i in range(7)])
        aids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{slot}_J{i}') for i in range(7)])
        agents[slot] = {'prefix': slot+'_', 'qpos_ids': model.jnt_qposadr[joints],
            'dof_ids': model.jnt_dofadr[joints], 'actuator_ids': aids,
            'sites': {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'{slot}_{name}')
                      for name in ('shinai_tip', 'shinai_grip')}}
    return SimpleNamespace(model=model, data=data, spec=spec, agents=agents,
        config={'base_distance_m': 1.75, 'virtual_targets': {
            'a1': {'center_world_m': [0., 0., 3.], 'radius_m': .06},
            'a2': {'center_world_m': [2.5, 0., 3.], 'radius_m': .06}}}, source_hashes={'fixture': 'short_native'})


def foul_verdict(tick=1, owner='a1'):
    return attribute_contact(ContactEvidence(tick, 'synthetic_self_contact',
        'illegal_self_contact', owner, owner, (0., 0., 0.), (0., 0., 0.),
        (1., 0., 0.), True, False))


class A1EnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        with patch('sim.a1_env.build_duel_scene', return_value=short_native_scene()):
            return A1DuelEnv(library_fixture(), load_runtime_profile(PROFILE),
                             development_mode=True, **kwargs)

    def test_explicit_development_and_numeric_optin_before_building(self):
        old = copy.deepcopy(V2_CONFIG)
        with patch('sim.a1_env.build_duel_scene') as build:
            for mode in (False, None, 1):
                with self.assertRaises(ValueError):
                    A1DuelEnv(library_fixture(), load_runtime_profile(PROFILE), development_mode=mode)
            with self.assertRaises(ValueError):
                A1DuelEnv(library_fixture(), None, development_mode=True)
            build.assert_not_called()
        self.assertEqual(V2_CONFIG, old)

    def test_guard_reset_independent_slots_same_public_schema_no_aliasing(self):
        env = self.make_env(); obs, info = env.reset()
        self.assertIsNot(env.controllers['a1'], env.controllers['a2'])
        self.assertEqual(info['policy_families'], {'a1': 'A1', 'a2': 'A1'})
        self.assertFalse(info['admitted_to_training'])
        self.assertEqual(set(obs['a1']), set(obs['a2']))
        self.assertEqual(set(obs['a1']), {'time_s', 'own_skill', 'own_q_rad', 'own_qvel_rad_s',
            'opponent_tip_minus_own_target_local_m', 'opponent_grip_minus_own_target_local_m',
            'opponent_tip_velocity_local_m_s'})
        np.testing.assert_array_equal(env.scene.data.qpos, np.zeros(14))
        obs['a1']['own_q_rad'][:] = 99.
        self.assertTrue(np.all(env.scene.data.qpos == 0.))
        self.assertEqual(int(env.scene.model.opt.integrator), int(mujoco.mjtIntegrator.mjINT_RK4))

    def test_joint_action_validation_is_atomic_and_no_one_sided_step(self):
        env = self.make_env(); env.reset()
        for actions in ({'a1': 1}, {'a1': 1, 'a2': 99}, {'a1': 1, 'a2': True}, {'a1': 1., 'a2': 0}):
            with self.assertRaises(ValueError): env.step(actions)
            self.assertEqual(env.controllers['a1'].events, [])
            self.assertEqual(env.controllers['a2'].events, [])
            self.assertEqual(env.scene.data.time, 0.)

    def test_twenty_native_ticks_ten_ms_control_and_feedback_every_step(self):
        env = self.make_env(); env.reset()
        a, b = env.controllers['a1'], env.controllers['a2']
        with patch.object(a, 'observe_own_state', wraps=a.observe_own_state) as fa, \
             patch.object(b, 'observe_own_state', wraps=b.observe_own_state) as fb, \
             patch.object(a, 'target', wraps=a.target) as target, \
             patch('sim.a1_env.mujoco.mj_resetData', wraps=mujoco.mj_resetData) as reset:
            obs, rewards, term, trunc, info = env.step({'a1': 1, 'a2': 2})
        self.assertEqual((fa.call_count, fb.call_count), (20, 20))
        np.testing.assert_allclose([c.args[0] for c in target.call_args_list], [0., .01], atol=1e-12)
        reset.assert_not_called()
        self.assertAlmostEqual(info['elapsed_s'], .02)
        self.assertEqual(info['physics_steps'], 20)
        self.assertEqual(rewards, {'a1': 0., 'a2': 0.})
        self.assertFalse(term or trunc)
        self.assertEqual(info['new_skill_accepted'], {'a1': True, 'a2': True})
        self.assertEqual([a.events[0]['time_s'], b.events[0]['time_s']], [0., 0.])
        self.assertTrue(obs['a1']['own_skill']['busy'])

    def test_busy_ticks_force_continue_without_queue_or_new_skill_choice(self):
        env = self.make_env(); env.reset(); env.step({'a1': 1, 'a2': 2})
        _, _, _, _, info = env.step({'a1': 4, 'a2': 3})
        self.assertEqual(info['choice_enabled'], {'a1': False, 'a2': False})
        self.assertEqual(info['effective_actions'], {'a1': 0, 'a2': 0})
        self.assertEqual(info['accepted'], {'a1': False, 'a2': False})
        self.assertEqual(info['new_skill_accepted'], {'a1': False, 'a2': False})
        self.assertEqual([len(c.events) for c in env.controllers.values()], [1, 1])

    def test_fractional_decision_window_truncates_at_actual_physical_time(self):
        env = self.make_env(max_duration_s=.025); env.reset()
        env.step({'a1': 0, 'a2': 0})
        _, rewards, term, trunc, info = env.step({'a1': 0, 'a2': 0})
        self.assertFalse(term); self.assertTrue(trunc)
        self.assertAlmostEqual(info['elapsed_s'], .005)
        self.assertEqual(info['physics_steps'], 5)
        self.assertEqual(rewards, {'a1': 0., 'a2': 0.})
        self.assertEqual(env.result()['round_result']['status'], 'timeout_draw')
        with self.assertRaises(RuntimeError): env.step({'a1': 0, 'a2': 0})
        for value in (True, -.1, .0005, float('nan')):
            with self.assertRaises(ValueError): self.make_env(max_duration_s=value)

    def test_target_touch_terminates_after_one_native_tick_not_full_twenty(self):
        env = self.make_env(); env.reset()
        with patch.object(env.scorer, 'update', return_value=[{'agent': 'a1', 'time_s': .0005}]):
            _, rewards, term, trunc, info = env.step({'a1': 0, 'a2': 0})
        self.assertTrue(term); self.assertFalse(trunc)
        self.assertEqual(info['physics_steps'], 1)
        self.assertAlmostEqual(info['elapsed_s'], .001)
        self.assertEqual(rewards, {'a1': 1., 'a2': -1.})
        self.assertEqual(env.result()['round_result']['status'], 'target_win')

    def test_same_tick_foul_precedes_hit_and_uses_official_foul_rewards(self):
        env = self.make_env(); env.reset()
        report = {'verdicts': [foul_verdict()], 'numerical_faults': [], 'contacts': []}
        with patch.object(env.monitor, 'inspect', return_value=report), \
             patch.object(env.scorer, 'update', return_value=[{'agent': 'a1', 'time_s': .0005}]):
            _, rewards, term, _, info = env.step({'a1': 0, 'a2': 0})
        self.assertTrue(term)
        self.assertEqual(rewards, {'a1': -1., 'a2': 1.})
        self.assertTrue(info['settlement']['foul_precedes_same_tick_hit'])
        self.assertEqual(env.result()['round_result']['match_winner'], 'a2')

    def test_collector_evidence_error_latches_invalid_and_preserves_failure_trace(self):
        for method in ('capture_before_step', 'inspect'):
            env = self.make_env(); env.reset()
            with patch.object(env.monitor, method, side_effect=TrainingEnvironmentError({'status': 'evidence_error'})):
                with self.assertRaises(TrainingEnvironmentError): env.step({'a1': 0, 'a2': 0})
            self.assertEqual(env.result()['status'], 'invalid')
            self.assertIsNone(env.result()['round_result'])
            self.assertGreaterEqual(len(env.trace()['time_s']), 2)
            with self.assertRaises(RuntimeError): env.step({'a1': 0, 'a2': 0})

    def test_actual_monitor_initial_contact_error_exports_invalid_not_draw(self):
        with patch('sim.a1_env.build_duel_scene', return_value=short_native_scene(contact_test=True)):
            env = A1DuelEnv(library_fixture(), load_runtime_profile(PROFILE), development_mode=True)
        env.reset()
        model = env.scene.model
        # Deliberately inject an invalid initial forbidden contact into the
        # tiny fixture (not a gameplay capability or original robot change).
        # The real monitor, not a mocked verdict, must reject its provenance.
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'a2_base')
        model.body_pos[body] = [.199, 0., 0.]
        for name in ('a1_shinai_blade_collision_0', 'a2_arm_proxy_0'):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            model.geom_contype[geom] = model.geom_conaffinity[geom] = 1
        mujoco.mj_forward(model, env.scene.data)
        env.monitor.reset(env.scene)
        with self.assertRaises(TrainingEnvironmentError):
            env.step({'a1': 0, 'a2': 0})
        result = env.result()
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['failure']['details']['reason'], 'unusable_first_contact_manifold')
        self.assertIsNone(result['round_result'])
        with tempfile.TemporaryDirectory() as directory:
            target = env.export(Path(directory)/'invalid')
            saved = json.loads((target/'summary.json').read_text())
            self.assertEqual(saved['status'], 'invalid')
            self.assertTrue(saved['failure']['details']['episode']['initial_overlap'])
            with np.load(target/'trajectory.npz', allow_pickle=False) as trace:
                self.assertGreaterEqual(len(trace['time_s']), 2)

    def test_numerical_fault_and_profile_change_never_become_draw_or_reward(self):
        env = self.make_env(); env.reset()
        env.scene.data.qfrc_applied[0] = 1.
        with self.assertRaises(TrainingEnvironmentError): env.step({'a1': 0, 'a2': 0})
        self.assertEqual(env.result()['status'], 'invalid')

    def test_nonfinite_native_time_is_a_latched_error_before_feedback(self):
        env = self.make_env(); env.reset(); env.scene.data.time = float('nan')
        with self.assertRaises(TrainingEnvironmentError): env.step({'a1': 0, 'a2': 0})
        self.assertEqual(env.result()['status'], 'invalid')
        env = self.make_env(); env.reset(); env.scene.model.opt.timestep = .002
        with self.assertRaises(TrainingEnvironmentError): env.step({'a1': 0, 'a2': 0})
        self.assertEqual(env.result()['status'], 'invalid')

    def test_result_and_trace_export_are_copies_and_never_overwrite(self):
        env = self.make_env(max_duration_s=.003); env.reset(); env.step({'a1': 0, 'a2': 0})
        result, trace = env.result(), env.trace()
        result['decisions'].clear(); trace['qpos'][:] = 99.
        self.assertEqual(len(env.result()['decisions']), 1)
        self.assertFalse(np.any(env.trace()['qpos'] == 99.))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'new'
            env.export(target)
            saved = json.loads((target/'summary.json').read_text())
            self.assertEqual(saved['status'], 'truncated')
            self.assertEqual((target/'runtime_profile.json').read_bytes(), PROFILE.read_bytes())
            self.assertEqual(len(saved['reproducibility']['bank_numeric_and_metadata_sha256']), 64)
            self.assertIn('sim/training_contacts.py', saved['reproducibility']['implementation_sha256'])
            self.assertTrue((target/'bank_snapshot.json').exists())
            with np.load(target/'bank_snapshot.npz', allow_pickle=False) as bank:
                np.testing.assert_array_equal(bank['stab__qpos'], env.library.skills['stab'].qpos)
            with np.load(target/'trajectory.npz', allow_pickle=False) as data:
                self.assertEqual(data['qpos'].shape, (2, 14))
            with self.assertRaises(FileExistsError): env.export(target)

    def test_source_library_mutation_cannot_change_owned_curves(self):
        source = library_fixture()
        with patch('sim.a1_env.build_duel_scene', return_value=short_native_scene()):
            env = A1DuelEnv(source, load_runtime_profile(PROFILE), development_mode=True)
        source.skills['stab'].qpos[1] = 99.
        env.reset()
        self.assertTrue(np.all(env.library.skills['stab'].qpos[1] == .001))


if __name__ == '__main__': unittest.main()
