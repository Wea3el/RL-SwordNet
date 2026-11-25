"""Bounded small-J0 defence authoring; physical and policy gates stay separate.

Attack trajectory samples are offline authoring inputs only. No future attack
state is supplied to a policy. All candidate outputs require fresh exact-entry
paired native tests; no endpoint or reference intersection is a certificate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .a1_dynamics_screen import screen_curve
from .duel_scene import classify_contact
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .multiguard_defence_probe import run as run_pair
from .multiguard_geometry import defence_reference_audit, unit
from .triangle_task import TriangleGeometry
from plan1_skills.matching import tool_fk


def save(path, value):
    Path(path).write_text(json.dumps(_finite_json(value), ensure_ascii=False, indent=2) + "\n")


def build(source, output, max_j0_degrees=15.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Keep preceding candidate and failed native evidence")
    raw = source.read_bytes(); original = json.loads(raw)
    guard_name = "left_guard_positive"
    entry = np.asarray(original["guards"][guard_name]["qpos_rad"])
    cons = load_joint_constraints(); physical = model_joint_ranges_rad(cons)
    bounds = physical + [.15, -.15]
    bounds[0] = np.maximum(bounds[0, 0], entry[0] - np.deg2rad(max_j0_degrees)), np.minimum(bounds[0, 1], entry[0] + np.deg2rad(max_j0_degrees))
    bounds[2, 0] = max(bounds[2, 0], np.pi)
    model = build_limited_floor(cons).model; data = mujoco.MjData(model)
    caps = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    geometry = TriangleGeometry.from_mapping({k: original["scoring"][k] for k in ("base_distance", "regions")})
    scene = build_limited_duel(geometry.base_distance, initial_qpos=entry, constraints=cons)
    attack = original["actions"]["cut_down_left_to_right"]
    ac = C2JointSpline(attack["knot_time_s"], attack["knot_qpos"])
    transform = np.array([-1., -1., 1.])
    pairs = [(gid, model.geom(f"arm_proxy_{j}").id)
             for gid in blade_collision_geom_ids(model, '') for j in range(7)]
    def distances():
        return np.array([mujoco.mj_geomDistance(model, data, a, b, .05, None) for a,b in pairs])

    def solve(point, incoming, normal, seed, along_seed):
        def residual(x):
            q, along = x[:7], x[7]
            grip, axis, _ = tool_fk(model, data, q)
            return np.r_[15. * (grip + along * axis - point),
                .7 * (axis @ normal), .5 * max(0., abs(axis @ incoming) - .60),
                8. * np.maximum(0., .010 - distances()),
                .025 * np.array([4.,1.,1.,1.,1.,1.,1.]) * (q - seed),
                .025 * (along - along_seed)]
        x = np.r_[np.clip(seed, bounds[:,0]+1e-8, bounds[:,1]-1e-8), along_seed]
        result = least_squares(residual,x,bounds=(np.r_[bounds[:,0],.12],np.r_[bounds[:,1],.78]),
                               max_nfev=180, ftol=1e-8,xtol=1e-8,gtol=1e-8)
        q, along = result.x[:7],float(result.x[7]); grip,axis,_=tool_fk(model,data,q)
        error=float(np.linalg.norm(grip+along*axis-point)); clearance=float(distances().min())
        data.qvel[:]=0.;mujoco.mj_forward(model,data)
        ratio=float(np.max(abs(data.qfrc_bias)/caps)); cross=float(abs(axis@incoming))
        return q,{"material_point_error_m":error,"along_m":along,"actual_axis":axis.tolist(),
            "actual_grip_m":grip.tolist(),"normal_plane_error":float(abs(axis@normal)),
            "incoming_axis_abs_dot":cross,"minimum_blade_arm_clearance_m":clearance,
            "gravity_ratio":ratio,"nfev":int(result.nfev),
            "fit_pass":bool(error<=.015 and cross<=.72 and clearance>=.005 and ratio<1.)}

    def first_contact(curve, desired_normal):
        for time in np.arange(.75,4.4,.02):
            scene.data.qpos[scene.agents["a1"]["qpos_ids"]]=ac.sample(time-.75)
            scene.data.qpos[scene.agents["a2"]["qpos_ids"]]=curve.sample(time-.75)
            mujoco.mj_forward(scene.model,scene.data)
            for contact in scene.data.contact:
                if contact.dist>=0 or classify_contact(scene.model,contact,rules_version="whole_blade_first_touch_v2")!="cross_blade_blade":continue
                normal=contact.frame[:3].copy(); a_first=scene.model.geom(contact.geom1).name.startswith("a1_")
                if not a_first:normal*=-1.
                moments={}
                for slot,geom,force in (("a1",contact.geom1 if a_first else contact.geom2,-normal),
                                        ("a2",contact.geom2 if a_first else contact.geom1,normal)):
                    jac=np.zeros((3,scene.model.nv));mujoco.mj_jac(scene.model,scene.data,jac,None,contact.pos,int(scene.model.geom_bodyid[geom]))
                    moments[slot]=(jac[:,scene.agents[slot]["dof_ids"]].T@force).tolist()
                return {"time_s":float(time),"normal_attacker_to_defender_world":normal.tolist(),
                        "desired_normal_dot":float(normal@desired_normal),"joint_normal_moment_per_n":moments,
                        "favorable_loading_hypothesis":bool(normal@desired_normal>.35 and moments["a1"][5]>.05)}
        return None

    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/"builder_snapshot.py";snapshot.write_bytes(Path(__file__).read_bytes())
    report={"schema_version":2,"status":"small_J0_geometry_candidates_not_native_verified",
        "candidate_only":True,"admitted_to_training":False,"native_dynamics_tested":False,
        "guards":copy.deepcopy(original["guards"]),"scoring":copy.deepcopy(original["scoring"]),
        "actions":{"cut_down_left_to_right":copy.deepcopy(attack)},"trials":[],
        "source_sha256":{str(source):hashlib.sha256(raw).hexdigest(),str(snapshot):hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        "J0_design_interval_degrees":np.rad2deg(bounds[0]).tolist(),"entry_exit_guard":guard_name,
        "limitations":["One incoming negative-side down cut only; not bilateral defence coverage.",
            "Future attack samples used only for offline geometry authoring.","No native or training certificate inherited from geometry.",
            "Whole-window success, timing/slot controls and callable defence/attack switching remain mandatory."]}
    seed_alt=entry.copy();seed_alt[2:]=np.deg2rad([270.,45.,260.,95.,20.])
    seed_mid=entry.copy();seed_mid[1:]=np.deg2rad([40.,330.,85.,200.,65.,30.])
    for idx,(crossing,attack_material) in enumerate(((3.3,.72),(3.6,.72),(3.3,.76),(3.6,.76))):
        grip,incoming_world,tip=tool_fk(model,qpos=ac.sample(crossing-.75))
        physical_length=float(np.linalg.norm(tip-grip))
        if attack_material > physical_length:
            raise ValueError("Authoring contact point must lie on the physical blade")
        point=(grip+attack_material*incoming_world-[geometry.base_distance,0.,0.])*transform
        desired=np.array([1.,1.,-.3]);desired=unit(desired-incoming_world*(desired@incoming_world))
        incoming=incoming_world*transform;normal=desired*transform
        choices=[]
        for seed_i,seed in enumerate((entry,seed_alt,seed_mid)):
            q,fit=solve(point+.01*normal,incoming,normal,seed,.55)
            choices.append((q,fit,seed_i))
        valid=[item for item in choices if item[1]["fit_pass"]]
        trial={"trial":idx,"crossing_time_s":crossing,"attack_material_m":attack_material,
               "physical_grip_to_tip_length_m":physical_length,"target_on_physical_blade":True,
               "target_defender_local_m":point.tolist(),"desired_normal_world":desired.tolist(),
               "block_solutions":[{"seed":i,"qpos":q.tolist(),**fit} for q,fit,i in choices],"path_variants":[]}
        # Keep at most the best two distinct solutions per target; the bounded
        # native budget is separate and never launches every geometric fit.
        valid.sort(key=lambda item:item[1]["material_point_error_m"]+.01*item[1]["gravity_ratio"])
        for qi,(block,bf,seed_i) in enumerate(valid[:2]):
            pre,pf=solve(point+.10*normal,incoming,normal,block,bf["along_m"])
            sweep,sf=solve(point-.10*normal,incoming,normal,block,bf["along_m"])
            times=[0.,1.8975,2.875,3.91,6.5,8.685]
            curve=C2JointSpline(times,[entry,pre,block,sweep,sweep,entry])
            peak=float(max(curve.analytic_extrema()["peak_speed_rad_s"]))
            idle,_=defence_reference_audit(scene,geometry,curve,ac.knot_qpos[0],period=.035)
            first=first_contact(curve,desired)
            samples=np.linspace(0.,curve.duration_s,337);ratios=[]
            for q in curve.sample(samples):
                data.qpos[:]=q;data.qvel[:]=0.;mujoco.mj_forward(model,data);ratios.append(float(np.max(abs(data.qfrc_bias)/caps)))
            gravity_max=float(max(ratios))
            dynamics=screen_curve(model,curve,period_s=.025)
            passed=bool(pf["fit_pass"] and sf["fit_pass"] and peak<=1.6 and gravity_max<1.
                        and idle["sampled_reference_pass"] and first is not None and first["favorable_loading_hypothesis"])
            name=f"parry_small_J0_{idx:02d}_{qi}"
            variant={"name":name,"seed":seed_i,"pre":pf,"block":bf,"sweep":sf,"candidate_pass":passed,
                "peak_reference_speed_rad_s":peak,"whole_path_gravity_max":gravity_max,
                "free_space_dynamics":dynamics,
                "idle_screen":idle,"first_reference_contact":first,
                "knot_time_s":times,"knot_qpos":curve.knot_qpos.tolist()}
            trial["path_variants"].append(variant)
            if passed:
                report["actions"][name]={"kind":"defence","start_guard":guard_name,"end_guard":guard_name,
                    "target_region":"left","candidate_pass":True,"sampled_reference_pass":True,
                    "admitted_to_training":False,"native_dynamics_tested":False,
                    "knot_time_s":times,"knot_qpos":curve.knot_qpos.tolist(),"authoring_trial":idx,"authoring_seed":seed_i}
            print(name,"fits",[bf["fit_pass"],pf["fit_pass"],sf["fit_pass"]],"peak",round(peak,3),
                  "gravity",round(gravity_max,3),"first",first,"idle",idle["sampled_reference_pass"],"pass",passed,flush=True)
        report["trials"].append(trial);save(output/"report.json",report)
        print("target",idx,"valid_blocks",len(valid),flush=True)
    report["source_unchanged"]=source.read_bytes()==raw
    (output/"scene.xml").write_text(scene.spec.to_xml());save(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",type=Path)
    parser.add_argument("--source",type=Path,default=ROOT/"outputs/a1_chain_geometry/chain12/report.json")
    parser.add_argument("--max-j0-degrees",type=float,default=15.)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--defence")
    parser.add_argument("--offset",type=float,default=0.)
    parser.add_argument("--attacker",choices=("a1","a2"),default="a1")
    args=parser.parse_args()
    if args.build:
        build(args.source,args.build,args.max_j0_degrees)
    elif args.output and args.defence:
        run_pair(args.source,args.output,attack_name="cut_down_left_to_right",defence_name=args.defence,
                 defence_offset_s=args.offset,attacker=args.attacker)
    else:parser.error("Use --build DIR or --source REPORT --output DIR --defence NAME")


if __name__=="__main__":main()
