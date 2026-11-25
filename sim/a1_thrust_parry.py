"""Bounded distal-blade parries for the right defensive family.

Author the approach, active lateral beat, protective hold and explicit exit
from a frozen baseline.  The opponent recording is an OFFLINE authoring input;
the exported action is a finite joint reference and has no future-state input.
Geometry/ideal-torque gates only nominate a native test.  They never certify a
defence, cancel an incoming hit or replace an already qualified action.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from plan1_skills.matching import tool_fk
from .a1_dynamics_screen import screen_curve
from .a1_upright_right import active_check
from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids, blade_geometry_identity
from .duel_scene import classify_contact
from .fencing_rules import _point_cylinder_distances
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_geometry import GripAxisSolver
from .triangle_task import TriangleGeometry


ACTION = 'parry_right_thrust'
DEFAULT_SOURCE = ROOT/'outputs/a1_profile_v2/bank07_current_candidates/report.json'
BASELINE = ROOT/'outputs/a1_profile_v2/native_thrust_nominal/thrust_center__parry_right_upright_early__a1__offset_+0.000/baseline'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), indent=2)+'\n')


def _scene(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    agents = {s: {'prefix': s+'_', 'qpos_ids': np.array([
        model.jnt_qposadr[model.joint(s+'_J'+str(i)).id] for i in range(7)])}
        for s in ('a1', 'a2')}
    return SimpleNamespace(model=model, data=mujoco.MjData(model), agents=agents)


def _gaps(scene, geometry, owner):
    ids = np.asarray(blade_collision_geom_ids(scene.model, owner+'_'))
    half = scene.model.geom_size[ids, 1, None]
    axes = scene.data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
    centers = scene.data.geom_xpos[ids]
    other = 'a2' if owner == 'a1' else 'a1'
    return {r.name: float(np.min(_point_cylinder_distances(
        geometry.world_center(r.name, other), centers-half*axes,
        centers+half*axes, scene.model.geom_size[ids, 0])[0])-r.radius)
        for r in geometry.regions}


def _contacts(scene):
    return [{'geoms': [scene.model.geom(int(c.geom1)).name, scene.model.geom(int(c.geom2)).name],
             'category': classify_contact(scene.model, c, rules_version='whole_blade_first_touch_v2'),
             'depth_m': float(-c.dist)} for c in scene.data.contact if c.dist < -1e-8]


def _set(scene, attacker_q, defender_q):
    for s, q in (('a1', attacker_q), ('a2', defender_q)):
        scene.data.qpos[scene.agents[s]['qpos_ids']] = q
    scene.data.qvel[:] = 0.
    mujoco.mj_forward(scene.model, scene.data)


def _self_screen(model, curve):
    data = mujoco.MjData(model)
    collisions = {}
    times = np.unique(np.r_[np.arange(0, curve.duration_s, .01), curve.knot_time_s, curve.duration_s])
    for t in times:
        data.qpos[:] = curve.sample(t)
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        for c in data.contact:
            if c.dist < -1e-8:
                names = [model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name]
                row = collisions.setdefault('|'.join(names), {'geoms': names, 'first_local_s': float(t), 'max_depth_m': 0.})
                row['max_depth_m'] = max(row['max_depth_m'], float(-c.dist))
    return {'sample_period_s': .01, 'sample_count': len(times), 'collisions': list(collisions.values()),
            'sampled_pass': not collisions, 'continuous_collision_certificate': False}


def _idle_screen(scene, geometry, curve, opponents):
    results = {}
    for label, opponent in opponents.items():
        contacts, entries = {}, {}
        for t in np.unique(np.r_[np.arange(0, curve.duration_s, .02), curve.knot_time_s, curve.duration_s]):
            _set(scene, opponent, curve.sample(t))
            for c in _contacts(scene):
                contacts.setdefault('|'.join(c['geoms']), dict(c, first_local_s=float(t)))
            for owner in ('a1', 'a2'):
                for name, gap in _gaps(scene, geometry, owner).items():
                    if gap < 0:
                        entries.setdefault(owner+'|'+name, {'owner': owner, 'region': name, 'first_local_s': float(t), 'gap_m': gap})
        results[label] = {'contacts': list(contacts.values()), 'target_entries': list(entries.values()),
            'legal_pass': not entries and not any(c['category'].startswith('illegal_') for c in contacts.values()),
            'neutral_weapon_contacts_are_disclosed_not_fouls': True}
    return results


def _recorded_prefix(scene, geometry, curve, times, qpos, hit_time):
    first = None
    illegal_before = []
    entries_before = []
    rows = []
    for t in np.unique(np.r_[np.arange(.75, hit_time, .01), hit_time-.001]):
        i = int(np.argmin(abs(times-t)))
        _set(scene, qpos[i, scene.agents['a1']['qpos_ids']], curve.sample(t-.75))
        contacts = _contacts(scene)
        gaps = {s: _gaps(scene, geometry, s) for s in ('a1', 'a2')}
        if first is None:
            illegal_before.extend(dict(c, time_s=float(t)) for c in contacts if c['category'].startswith('illegal_'))
            entries_before.extend({'owner': s, 'region': r, 'time_s': float(t), 'gap_m': gap}
                                  for s, rs in gaps.items() for r, gap in rs.items() if gap < 0)
            weapon = [c for c in contacts if c['category'] in ('cross_blade_blade', 'cross_blade_guard')]
            if weapon:
                first = dict(weapon[0], time_s=float(t))
        if contacts:
            rows.append({'time_s': float(t), 'contacts': contacts})
    return {'first_sampled_weapon_contact': first, 'illegal_before_first_weapon': illegal_before,
            'target_entries_before_first_weapon': entries_before,
            'sampled_pre_hit_opportunity': bool(first and not illegal_before and not entries_before),
            'contacts_through_baseline_hit': rows,
            'scope': 'Reference defender versus undisturbed recorded attacker; after first contact is NOT physical prediction.'}


def build(output, source=DEFAULT_SOURCE):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh directory; failed candidates stay intact')
    output.mkdir(parents=True)
    bank = json.loads(source.read_text())
    if ACTION in bank['actions']:
        raise ValueError('Specialized action must be explicitly added, never overwritten')
    old = bank['actions']['parry_right_upright_early']
    q0 = np.asarray(bank['guards']['right_low_negative_robust']['qpos_rad'])
    pre = np.asarray(old['knot_qpos'][1])
    block = np.asarray(old['knot_qpos'][2])
    attack = bank['actions']['thrust_center']
    attack_curve = C2JointSpline(attack['knot_time_s'], attack['knot_qpos'])
    baseline_summary = json.loads((BASELINE/'summary.json').read_text())
    if not baseline_summary['native_execution_pass'] or baseline_summary['failure'] is not None:
        raise ValueError('Complete legal baseline required')
    with np.load(BASELINE/'references.npz') as z:
        if not all(np.array_equal(z['a1_'+key], getattr(attack_curve, key)) for key in ('knot_time_s','knot_qpos','coefficients')):
            raise ValueError('The recorded attack is not this source attack')
    with np.load(BASELINE/'trace.npz') as z:
        recorded_t, recorded_q = z['time_s'].copy(), z['qpos'].copy()
    scene = _scene(BASELINE/'scene.xml')
    model = build_limited_floor().model
    geometry = TriangleGeometry.from_mapping({k: bank['scoring'][k] for k in ('base_distance','regions')})
    bounds = model_joint_ranges_rad(load_joint_constraints())
    solver = GripAxisSolver(model, bounds+np.deg2rad([1.,-1.]))
    attack_ids = scene.agents['a1']['qpos_ids']
    def point_at(global_time, along):
        i = int(np.argmin(abs(recorded_t-global_time)))
        grip, axis, _ = tool_fk(model, qpos=recorded_q[i, attack_ids])
        return grip+along*axis
    rotation = np.diag([-1.,-1.,1.])
    # Three physically distinct distal-control proposals, not a random grid.
    specs = [
        {'id':'distal_low_angle_beat','axis_local':[-.83348,-.12949,.53716], 'attacker_material_m':.73,
         'defender_material_m':.48, 'prepare_y_m':.055, 'sweep_y_m':-.12, 'intercept_local_s':1.90},
        {'id':'distal_steep_follow','axis_local':[-.62,-.23,.750], 'attacker_material_m':.74,
         'defender_material_m':.44, 'prepare_y_m':.045, 'sweep_y_m':-.09, 'intercept_local_s':1.90},
        {'id':'tip_side_bind','axis_local':[-.23,-.53,.815], 'attacker_material_m':.77,
         'defender_material_m':.46, 'prepare_y_m':.050, 'sweep_y_m':-.10, 'intercept_local_s':1.85}]
    files = [source, Path(__file__), BASELINE/'summary.json', BASELINE/'trace.npz', BASELINE/'scene.xml', BASELINE/'references.npz']
    files.extend(ROOT/p for p in ('sim/multiguard_geometry.py','sim/a1_dynamics_screen.py','sim/blade_geometry.py',
        'sim/fencing_rules.py','sim/assembly.py','sim/motion_spline.py','config/assembly.json','config/joint_constraints_observed.json','assets/shinai_profile.json'))
    report = {'schema_version':1,'status':'specialized_thrust_authoring_only','action':ACTION,
        'source_sha256':{str(p.relative_to(ROOT)):_sha(p) for p in files}, 'trials':[],
        'blade_geometry':blade_geometry_identity(model),'native_dynamics_tested':False,'admitted_to_training':False,
        'no_prior_certificates_inherited':True,
        'purpose':'Distal active lateral displacement sustained through peak thrust; no other attack coverage claimed.',
        'future_data':'Recorded opponent is used only for offline design. Exported joint splines read no opponent future.'}
    _write(output/'report.json',report)
    for spec in specs:
        axis = np.asarray(spec['axis_local']); axis /= np.linalg.norm(axis)
        world_points = [point_at(t, spec['attacker_material_m']) for t in (2.30, 2.65, 3.20)]
        for point, dy in zip(world_points, (spec['prepare_y_m'], -.005, spec['sweep_y_m'])):
            point[1] += dy
        knots = [q0,pre]
        fits = []
        seed = block
        for world_point in world_points:
            local_point = rotation@(world_point-np.array([1.75,0.,0.]))
            goal = local_point-spec['defender_material_m']*axis
            q, fit = solver.solve(goal, axis, seed, regularization=.002, iterations=220)
            fits.append(dict(fit, requested_grip_local_m=goal.tolist(), requested_axis_local=axis.tolist(),
                             requested_contact_world_m=world_point.tolist()))
            knots.append(q); seed=q
        # Protect until the original thrust has withdrawn; then explicit exit.
        knots.extend([knots[-1].copy(),q0])
        times = [0.,.65,1.55,spec['intercept_local_s'],2.45,4.05,6.55]
        curve = C2JointSpline(times,knots)
        fit_pass = all(f['position_error_m']<.003 and f['axis_error_deg']<1. for f in fits)
        row = dict(spec, fit_pass=fit_pass, fits=fits, knot_time_s=times,knot_qpos=curve.knot_qpos.tolist(),
                   analytic_bounds_pass=bool(np.all(curve.bezier_controls>=bounds[:,0]) and np.all(curve.bezier_controls<=bounds[:,1])))
        row['self_screen'] = _self_screen(model,curve)
        row['upright_screen'] = active_check(model,curve,.65,4.05)
        row['ideal_free_space_dynamics'] = screen_curve(model,curve,.01)
        row['idle_screens'] = _idle_screen(scene,geometry,curve,{'attack_entry':attack_curve.knot_qpos[0], 'attack_exit':attack_curve.knot_qpos[-1]})
        row['recorded_prefix'] = _recorded_prefix(scene,geometry,curve,recorded_t,recorded_q,2.7813654667155725)
        row['candidate_pass'] = bool(fit_pass and row['analytic_bounds_pass'] and row['self_screen']['sampled_pass']
            and row['upright_screen']['sampled_upright_pass'] and row['ideal_free_space_dynamics']['ideal_demand_within_caps_at_samples']
            and all(v['legal_pass'] for v in row['idle_screens'].values()) and row['recorded_prefix']['sampled_pre_hit_opportunity'])
        candidate = copy.deepcopy(bank)
        candidate['actions'][ACTION] = {'kind':'defence','family':'parry_right','start_guard':'right_low_negative_robust',
            'end_guard':'right_low_negative_robust','target_region':'right','knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist(),
            'candidate_pass':row['candidate_pass'],'sampled_reference_pass':row['candidate_pass'],
            'active_interval_s':[.65,4.05], 'min_elevation_deg':25.,'minimum_grip_height_m':.35,
            'stages':[{'name':'prepare','start_s':0.,'end_s':1.55},{'name':'intercept_and_lateral_beat','start_s':1.55,'end_s':2.45},
                      {'name':'protective_hold','start_s':2.45,'end_s':4.05},{'name':'explicit_exit','start_s':4.05,'end_s':6.55}],
            'native_dynamics_tested':False,'admitted_to_training':False,'prior_certificate_inherited':False,
            'specialized_for':'thrust_center, nominal start/entry only until further validated',
            'authoring_provenance':{'candidate':spec['id'],'source_bank_sha256':_sha(source),'builder_sha256':_sha(__file__)}}
        candidate.update(status='specialized_thrust_candidate_only',action_space_expanded=True,requires_runtime_action_migration=True)
        folder=output/spec['id'];folder.mkdir();_write(folder/'report.json',candidate)
        row['candidate_bank_path']=str((folder/'report.json').relative_to(ROOT));row['candidate_bank_sha256']=_sha(folder/'report.json')
        report['trials'].append(row);_write(output/'report.json',report)
        print(json.dumps({'id':spec['id'],'fit':fit_pass,'self':row['self_screen']['sampled_pass'],
            'upright':row['upright_screen']['sampled_upright_pass'],'ideal':row['ideal_free_space_dynamics']['worst_utilization'],
            'idle':all(v['legal_pass'] for v in row['idle_screens'].values()),
            'first_contact':row['recorded_prefix']['first_sampled_weapon_contact'],'candidate_pass':row['candidate_pass']}),flush=True)
    report['all_source_inputs_unchanged']=all(_sha(ROOT/p)==h for p,h in report['source_sha256'].items())
    _write(output/'report.json',report)
    return report


def retime_low_angle(source_report, output, *, fine=False):
    """Two bounded clock revisions; retain the exact low-angle path.

    The first active interception knot stays at local1.90/global2.65s.  Only
    the preparation split and the remaining lateral-sweep/recovery duration
    change.  Failed preceding clocks remain separate immutable evidence.
    """
    source_report, output = Path(source_report).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier clock attempts')
    output.mkdir(parents=True)
    old = json.loads(source_report.read_text())
    original = next(r for r in old['trials'] if r['id']=='distal_low_angle_beat')
    parent_bank = ROOT/original['candidate_bank_path']
    bank = json.loads(parent_bank.read_text())
    scene = _scene(BASELINE/'scene.xml')
    model = build_limited_floor().model
    geometry = TriangleGeometry.from_mapping({k:bank['scoring'][k] for k in ('base_distance','regions')})
    attack = bank['actions']['thrust_center']
    attack_curve = C2JointSpline(attack['knot_time_s'], attack['knot_qpos'])
    with np.load(BASELINE/'trace.npz') as z:
        rt, rq = z['time_s'].copy(), z['qpos'].copy()
    report = {'schema_version':1,'status':'bounded_low_angle_clock_revision','trials':[],
        'native_dynamics_tested':False,'admitted_to_training':False,
        'source_sha256':{str(p.relative_to(ROOT)):_sha(p) for p in (source_report,parent_bank,Path(__file__))},
        'same_geometry_candidate':'distal_low_angle_beat','preserved_active_intercept_global_s':2.65}
    clocks = (
        ('low_angle_clock_a',[0.,.75,1.55,1.9,2.65,4.25,6.75]),
        ('low_angle_clock_b',[0.,.80,1.55,1.9,2.75,4.40,7.0]))
    if fine:
        clocks = (('low_angle_clock_c',[0.,.68,1.55,1.9,2.65,4.25,6.75]),)
    for identifier, times in clocks:
        curve = C2JointSpline(times,original['knot_qpos'])
        row = {'id':identifier,'exact_joint_waypoints_preserved':bool(np.array_equal(curve.knot_qpos,original['knot_qpos'])),
            'knot_time_s':times,'knot_qpos':curve.knot_qpos.tolist()}
        row['self_screen'] = _self_screen(model,curve)
        row['upright_screen'] = active_check(model,curve,times[1],times[-2])
        row['ideal_free_space_dynamics'] = screen_curve(model,curve,.01)
        row['idle_screens'] = _idle_screen(scene,geometry,curve,{'attack_entry':attack_curve.knot_qpos[0],'attack_exit':attack_curve.knot_qpos[-1]})
        row['recorded_prefix'] = _recorded_prefix(scene,geometry,curve,rt,rq,2.7813654667155725)
        row['candidate_pass'] = bool(row['self_screen']['sampled_pass'] and row['upright_screen']['sampled_upright_pass']
            and row['ideal_free_space_dynamics']['ideal_demand_within_caps_at_samples']
            and all(v['legal_pass'] for v in row['idle_screens'].values())
            and row['recorded_prefix']['sampled_pre_hit_opportunity'])
        candidate = copy.deepcopy(bank)
        action = candidate['actions'][ACTION]
        action.update(knot_time_s=times,candidate_pass=row['candidate_pass'],sampled_reference_pass=row['candidate_pass'],
            active_interval_s=[times[1],times[-2]],
            stages=[{'name':'prepare','start_s':0.,'end_s':1.55},
                    {'name':'intercept_and_lateral_beat','start_s':1.55,'end_s':times[4]},
                    {'name':'protective_hold','start_s':times[4],'end_s':times[-2]},
                    {'name':'explicit_exit','start_s':times[-2],'end_s':times[-1]}],
            authoring_provenance={'candidate':identifier,'geometry_source_sha256':_sha(source_report),
                'builder_sha256':_sha(__file__),'same_waypoints':True,'not_unconditional_uniform_delay':True})
        folder=output/identifier;folder.mkdir();_write(folder/'report.json',candidate)
        row['candidate_bank_path']=str((folder/'report.json').relative_to(ROOT));row['candidate_bank_sha256']=_sha(folder/'report.json')
        report['trials'].append(row);_write(output/'report.json',report)
        print(json.dumps({'id':identifier,'ideal':row['ideal_free_space_dynamics']['worst_utilization'],
            'self':row['self_screen']['sampled_pass'],'upright':row['upright_screen']['sampled_upright_pass'],
            'first_contact':row['recorded_prefix']['first_sampled_weapon_contact'],'candidate_pass':row['candidate_pass']}),flush=True)
    report['all_source_inputs_unchanged']=all(_sha(ROOT/p)==h for p,h in report['source_sha256'].items())
    _write(output/'report.json',report)
    return report


def repair_low_angle_grip(source_report, output):
    """Keep the distal attacker points but put them nearer our blade's base.

    This retracts the handle away from the opponent's arm.  The unrelated old
    pre waypoint is removed and the whole new entry is checked, not assumed.
    """
    source_report, output = Path(source_report).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the earlier handle-collision evidence')
    output.mkdir(parents=True)
    old = json.loads(source_report.read_text())
    original = next(r for r in old['trials'] if r['id']=='distal_low_angle_beat')
    parent_bank = ROOT/original['candidate_bank_path']
    bank = json.loads(parent_bank.read_text())
    model = build_limited_floor().model
    bounds = model_joint_ranges_rad(load_joint_constraints())
    solver = GripAxisSolver(model,bounds+np.deg2rad([1.,-1.]))
    axis = np.asarray(original['axis_local']); axis /= np.linalg.norm(axis)
    q0 = np.asarray(original['knot_qpos'][0]); seed = np.asarray(original['knot_qpos'][2])
    fits=[]; nodes=[q0]
    rotation=np.diag([-1.,-1.,1.]); material=.24
    for previous in original['fits']:
        world_point=np.asarray(previous['requested_contact_world_m'])
        local_point=rotation@(world_point-np.array([1.75,0.,0.]))
        goal=local_point-material*axis
        q,fit=solver.solve(goal,axis,seed,regularization=.002,iterations=240)
        fits.append(dict(fit,requested_grip_local_m=goal.tolist(),requested_contact_world_m=world_point.tolist()))
        nodes.append(q); seed=q
    nodes.extend([nodes[-1].copy(),q0])
    times=[0.,1.55,1.9,2.65,4.25,6.75]
    curve=C2JointSpline(times,nodes)
    scene=_scene(BASELINE/'scene.xml')
    geometry=TriangleGeometry.from_mapping({k:bank['scoring'][k] for k in ('base_distance','regions')})
    attack=bank['actions']['thrust_center'];ac=C2JointSpline(attack['knot_time_s'],attack['knot_qpos'])
    with np.load(BASELINE/'trace.npz')as z: rt,rq=z['time_s'].copy(),z['qpos'].copy()
    row={'id':'low_angle_proximal_contact','fits':fits,'defender_material_m':material,
         'knot_qpos':curve.knot_qpos.tolist(),'knot_time_s':times,
         'removed_old_pre_waypoint':True,'attacker_distal_points_unchanged':True,
         'fit_pass':all(f['position_error_m']<.003 and f['axis_error_deg']<1. for f in fits)}
    row['self_screen']=_self_screen(model,curve)
    row['upright_screen']=active_check(model,curve,1.2,4.25)
    row['ideal_free_space_dynamics']=screen_curve(model,curve,.01)
    row['idle_screens']=_idle_screen(scene,geometry,curve,{'attack_entry':ac.knot_qpos[0],'attack_exit':ac.knot_qpos[-1]})
    row['recorded_prefix']=_recorded_prefix(scene,geometry,curve,rt,rq,2.7813654667155725)
    row['candidate_pass']=bool(row['fit_pass'] and row['self_screen']['sampled_pass'] and row['upright_screen']['sampled_upright_pass']
        and row['ideal_free_space_dynamics']['ideal_demand_within_caps_at_samples']
        and all(v['legal_pass'] for v in row['idle_screens'].values()) and row['recorded_prefix']['sampled_pre_hit_opportunity'])
    candidate=copy.deepcopy(bank);action=candidate['actions'][ACTION]
    action.update(knot_time_s=times,knot_qpos=curve.knot_qpos.tolist(),candidate_pass=row['candidate_pass'],
        sampled_reference_pass=row['candidate_pass'],active_interval_s=[1.2,4.25],
        stages=[{'name':'prepare','start_s':0.,'end_s':1.55},{'name':'intercept_and_lateral_beat','start_s':1.55,'end_s':2.65},
                {'name':'protective_hold','start_s':2.65,'end_s':4.25},{'name':'explicit_exit','start_s':4.25,'end_s':6.75}],
        authoring_provenance={'candidate':row['id'],'builder_sha256':_sha(__file__),
                             'source_geometry_report_sha256':_sha(source_report),'defender_material_m':material})
    folder=output/row['id'];folder.mkdir();_write(folder/'report.json',candidate)
    row['candidate_bank_path']=str((folder/'report.json').relative_to(ROOT));row['candidate_bank_sha256']=_sha(folder/'report.json')
    report={'schema_version':1,'status':'handle_clearance_repair_same_low_angle_candidate','trials':[row],
            'native_dynamics_tested':False,'admitted_to_training':False,
            'source_sha256':{str(p.relative_to(ROOT)):_sha(p) for p in (source_report,parent_bank,Path(__file__))}}
    _write(output/'report.json',report)
    print(json.dumps({'id':row['id'],'fit':row['fit_pass'],'self':row['self_screen']['sampled_pass'],
        'upright':row['upright_screen']['sampled_upright_pass'],'ideal':row['ideal_free_space_dynamics']['worst_utilization'],
        'first_contact':row['recorded_prefix']['first_sampled_weapon_contact'],
        'illegal_pre_count':len(row['recorded_prefix']['illegal_before_first_weapon']),'pass':row['candidate_pass']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(DEFAULT_SOURCE))
    parser.add_argument('--output',required=True)
    parser.add_argument('--retime-report')
    parser.add_argument('--fine-clock', action='store_true')
    parser.add_argument('--repair-grip-report')
    args=parser.parse_args()
    if args.repair_grip_report:
        repair_low_angle_grip(args.repair_grip_report,args.output)
    elif args.retime_report:
        retime_low_angle(args.retime_report,args.output,fine=args.fine_clock)
    else:
        build(args.output,args.source)
