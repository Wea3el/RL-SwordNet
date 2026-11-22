"""New training-only adapter for the explicitly approved foul convention.

Old physical validation is unchanged. First-contact evidence uses a complete
pre-step MjData copy and the first contact's material point Jacobian. This is
a one-physics-step approximation, NOT exact impact time or causal blame.
Geometry-pair episodes retain ALL first-manifold points. A changed manifold
at ANY earlier frame is a sticky evidence error at qualification, not a made-up winner. A prior or
same-step positive legal weapon force conservatively makes cross blame
ambiguous for the rest of the round; apparent rest does not clear history.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass, replace

import mujoco
import numpy as np

from .duel_scene import classify_contact
from .foul_attribution import (ContactEvidence, LEGAL_CONTACTS, RULE_VERSION,
                               attribute_contact, settle_tick)


DEPLOYED_RULE_VERSION = "motion_foul_training_v1"


def _serializable(value):
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {key:_serializable(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


class TrainingEnvironmentError(RuntimeError):
    """An unusable transition: never substitute a reward or a timeout draw."""
    def __init__(self, details):
        self.details = _serializable(copy.deepcopy(details))
        super().__init__(str(details))


def settle_training_tick(physics_tick, *, contacts=(), hit_agents=(), numerical_faults=()):
    result = settle_tick(physics_tick, contacts=contacts, hit_agents=hit_agents,
                         numerical_faults=numerical_faults)
    source_metadata={key:result[key] for key in ("rule_version","candidate_only","live_runner_changed")}
    result.update(rule_version=DEPLOYED_RULE_VERSION,
                  candidate_only=False,live_runner_changed=True,
                  source_offline_result_metadata=source_metadata,
                  deployed_rule_version=DEPLOYED_RULE_VERSION,
                  source_offline_rule_version=RULE_VERSION,
                  deployment="explicitly_approved_new_training_environment_only")
    if not result["valid_for_reward"]:
        raise TrainingEnvironmentError(result)
    return result


def _owner(name):
    if name == "floor":
        return None
    if name.startswith(("a1_", "a2_")):
        return name[:2]
    raise ValueError(f"Unknown contact owner: {name}")


def previous_material_point_velocity(model, before, after, body, current_point):
    """Map the present point into this body's previous rigid frame, then Jv."""
    point = np.asarray(current_point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Finite world contact point required")
    current_rotation = after.xmat[body].reshape(3, 3)
    local = current_rotation.T @ (point-after.xpos[body])
    previous_point = before.xpos[body]+before.xmat[body].reshape(3, 3) @ local
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, before, jac, None, previous_point, int(body))
    velocity = jac @ before.qvel
    if not np.isfinite(velocity).all():
        raise ValueError("Nonfinite previous material-point velocity")
    return velocity


def _same_material_manifold(first, current):
    """Match unordered material-point pairs, not unstable contact indices.

    1nm is solely roundoff slack for rigid-frame transforms, not a calibrated
    sliding-contact tolerance. Untracked sliding/replaced points are rejected.
    """
    if len(first) != len(current):
        return False
    unused = list(range(len(current)))
    for point in first:
        match = next((i for i in unused if np.max(np.abs(point-current[i])) <= 1e-9), None)
        if match is None:
            return False
        unused.remove(match)
    return True


class TrainingContactMonitor:
    """Read-only evidence collector with reset-local contact episodes."""
    def __init__(self, scene, cfg):
        self.dt = float(cfg["physics_timestep_s"])
        self.force_threshold = float(cfg["illegal_contact_force_threshold_n"])
        if not np.isfinite([self.dt, self.force_threshold]).all() \
                or self.dt <= 0 or self.force_threshold < 0:
            raise ValueError("Finite positive step and nonnegative existing force gate required")
        self.reset(scene)

    def reset(self, scene):
        self._model = scene.model
        self._before = None
        self._last_time = float(scene.data.time)
        if not np.isfinite(self._last_time):
            raise TrainingEnvironmentError({"status":"numerical_fault", "reason":"nonfinite_reset_time"})
        self._last_tick = 0
        self._episodes = {}
        self._episode_serial = 0
        self._legal_influence = False
        self._audit = []
        # An initial overlap has no unperturbed prior state. If it later
        # triggers a forbidden force gate it must not invent attribution.
        for index, contact in enumerate(scene.data.contact):
            if classify_contact(scene.model,contact,rules_version="whole_blade_first_touch_v2") in LEGAL_CONTACTS:
                force=np.zeros(6);mujoco.mj_contactForce(scene.model,scene.data,index,force)
                if force[0] > 0.: self._legal_influence=True
            if contact.dist <= 0:
                key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                self._episodes[key] = {"evidence": [], "initial_overlap": True}

    def capture_before_step(self, scene):
        if scene.model is not self._model or self._before is not None \
                or not np.isfinite([scene.data.time,self._last_time]).all() \
                or abs(float(scene.data.time)-self._last_time) > 1e-9:
            raise TrainingEnvironmentError({"status":"evidence_error", "reason":"noncontiguous_pre_step_capture"})
        self._before = copy.copy(scene.data)

    def inspect(self, scene, physics_tick):
        before, self._before = self._before, None
        now = float(scene.data.time)
        if (scene.model is not self._model or before is None
                or not np.isfinite([now,self._last_time,before.time]).all()
                or abs(float(before.time)-self._last_time) > 1e-9
                or type(physics_tick) is not int or physics_tick != self._last_tick+1
                or abs(now-self._last_time-self.dt) > 1e-9):
            raise TrainingEnvironmentError({"status":"evidence_error", "reason":"missing_or_noncontiguous_native_frame"})
        model, data = scene.model, scene.data
        groups, records, faults = {}, [], []
        for index, contact in enumerate(data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            g1, g2 = int(contact.geom1), int(contact.geom2)
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in (g1,g2)]
            category = classify_contact(model, contact, rules_version="whole_blade_first_touch_v2")
            record = dict(geom1=names[0], geom2=names[1], category=category,
                distance_m=float(contact.dist), normal_force_n=float(force[0]),
                position_world_m=np.array(contact.pos).tolist(),
                normal_geom1_to_geom2_world=np.array(contact.frame[:3]).tolist())
            records.append(record)
            if not np.isfinite([contact.dist, *contact.pos, *contact.frame[:3], force[0]]).all():
                faults.append("nonfinite_native_contact")
                continue
            if category in LEGAL_CONTACTS and force[0] > 0.:
                self._legal_influence = True
            # Positive-margin, not-yet-touching contacts have no first impact.
            if contact.dist <= 0:
                key = tuple(sorted((g1,g2)))
                groups.setdefault(key, []).append((contact, record))
            elif category not in LEGAL_CONTACTS and force[0] > self.force_threshold:
                faults.append("forbidden_force_without_geometric_contact")

        # All contacts were scanned first: legal/illegal iteration order must
        # not change attribution of a same-step mixed contact.
        verdicts = []
        for key, points in groups.items():
            episode = self._episodes.get(key)
            material_points = []
            for contact, _ in points:
                # Canonical geom order also canonicalizes the two material
                # points when MuJoCo reverses geom1/geom2 presentation.
                pair_points = []
                # MuJoCo stores the midpoint between two surface witnesses.
                # Separate witnesses avoid interpreting normal compression
                # of that midpoint as a different material contact point.
                normal=np.asarray(contact.frame[:3])
                witnesses={int(contact.geom1):np.asarray(contact.pos)-.5*contact.dist*normal,
                           int(contact.geom2):np.asarray(contact.pos)+.5*contact.dist*normal}
                for geom in key:
                    body = int(model.geom_bodyid[geom])
                    pair_points.extend(data.xmat[body].reshape(3,3).T @
                                       (witnesses[geom]-data.xpos[body]))
                material_points.append(np.asarray(pair_points))
            if episode is None:
                self._episode_serial += 1
                evidence, errors = [], []
                for point_index, (contact, record) in enumerate(points):
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    try:
                        normal=np.asarray(contact.frame[:3])
                        witnesses=(np.asarray(contact.pos)-.5*contact.dist*normal,
                                   np.asarray(contact.pos)+.5*contact.dist*normal)
                        velocities = [previous_material_point_velocity(model, before, data,
                            int(model.geom_bodyid[g]), point) for g,point in zip((g1,g2),witnesses)]
                        item = ContactEvidence(physics_tick,
                            f"episode{self._episode_serial}:geom{key[0]}-{key[1]}:point{point_index}",
                            record["category"], _owner(record["geom1"]), _owner(record["geom2"]),
                            tuple(velocities[0]), tuple(velocities[1]),
                            tuple(np.array(contact.frame[:3])), True, self._legal_influence)
                        evidence.append(item)
                    except ValueError as error:
                        errors.append(str(error))
                episode = {"evidence":evidence, "initial_overlap":False,
                           "errors":errors, "first_time_s":now,
                           "first_manifold_size":len(points),
                           "first_material_points":material_points,
                           "manifold_identity_lost":False}
                self._audit.append({"first_time_s":now, "geom_pair":list(key),
                    "evidence":[asdict(item) for item in evidence], "errors":errors})
            elif not episode["initial_overlap"]:
                episode["manifold_identity_lost"] = bool(episode["manifold_identity_lost"] or
                    not _same_material_manifold(episode["first_material_points"], material_points))
            forbidden = any(r["category"] not in LEGAL_CONTACTS and
                            r["normal_force_n"] > self.force_threshold for _,r in points)
            if forbidden:
                if episode["initial_overlap"] or episode.get("errors") \
                        or not episode["evidence"] \
                        or episode["manifold_identity_lost"]:
                    raise TrainingEnvironmentError({"status":"evidence_error",
                        "reason":"unusable_first_contact_manifold", "physics_tick":physics_tick,
                        "geom_pair":list(key), "episode":copy.deepcopy(episode)})
                for cached in episode["evidence"]:
                    item = replace(cached, physics_tick=physics_tick,
                        prior_legal_weapon_influence=bool(
                            cached.prior_legal_weapon_influence or self._legal_influence))
                    verdicts.append(attribute_contact(item))
            groups[key] = episode
        # Separation clears only the pair episode, never round legal history.
        self._episodes = groups
        self._last_time, self._last_tick = now, physics_tick
        return {"verdicts":verdicts, "numerical_faults":faults, "contacts":records}

    def summary(self):
        return copy.deepcopy({"deployed_rule_version":DEPLOYED_RULE_VERSION,
            "source_offline_rule_version":RULE_VERSION,
            "preimpact_sampling":"immediately previous physics frame; first geom-pair manifold; not exact TOI",
            "same_or_prior_positive_legal_force_makes_cross_blame_ambiguous":True,
            "changed_or_sliding_first_material_manifold_is_evidence_error":True,
            "round_legal_influence":self._legal_influence,
            "last_physics_tick":self._last_tick, "first_contact_episodes":self._audit})
