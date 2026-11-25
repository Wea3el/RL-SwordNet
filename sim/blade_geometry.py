"""Error-bounded finite blade cylinders, shared by collision and scoring.

Each original piecewise-linear profile interval is subdivided until the
constant cylinder radius exceeds the circular profile by at most the declared
budget.  End planes are shared exactly: there are no axial gaps or capsule
extensions.  This is still an approximation, not a tessellation of CAD slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import weakref

import numpy as np

from .shinai import load_profile


BLADE_GEOMETRY_VERSION = "adaptive_flat_cylinder_v2"
DEFAULT_MAX_RADIAL_EXCESS_M = .00025
DEFAULT_MAX_SECTIONS = 256
_GEOM_CACHE = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class BladeCylinder:
    start_m: float
    end_m: float
    radius_m: float
    radial_excess_bound_m: float
    profile_interval: int

    @property
    def center_m(self):
        return .5*(self.start_m+self.end_m)

    @property
    def half_length_m(self):
        return .5*(self.end_m-self.start_m)


def _positive_real(value, name):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite positive real")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive real") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive real")
    return number


def adaptive_cylinder_sections(points, max_radial_excess_m=DEFAULT_MAX_RADIAL_EXCESS_M,
                               max_sections=DEFAULT_MAX_SECTIONS):
    """Cover a strictly increasing positive-radius linear profile from outside.

    The proof is interval-local: a linear radius lies between its two endpoint
    radii, while the cylinder uses their maximum.  Subdivision bounds their
    difference.  Profile corners, including changes of slope sign, are never
    silently replaced by a chord spanning several intervals.
    """
    try:
        raw = np.asarray(points)
        if raw.dtype.kind not in 'fiu':
            raise ValueError("Profile coordinates must be real numbers")
        values = np.array(raw, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Finite N by 2 profile required") from exc
    if (values.ndim != 2 or values.shape[1] != 2 or len(values)<2
            or not np.isfinite(values).all() or np.any(np.diff(values[:,0])<=0)
            or np.any(values[:,1]<=0)):
        raise ValueError("Strictly increasing finite axial stations and positive radii required")
    budget=_positive_real(max_radial_excess_m, 'max_radial_excess_m')
    if type(max_sections) is not int or max_sections<1:
        raise ValueError("max_sections must be a positive integer")
    counts=[max(1,math.ceil(abs(b[1]-a[1])/budget)) for a,b in zip(values[:-1],values[1:])]
    if sum(counts)>max_sections:
        raise ValueError(f"Requested radial budget needs {sum(counts)} sections, exceeding {max_sections}")
    sections=[]
    for interval,((a,ra),(b,rb),count) in enumerate(zip(values[:-1],values[1:],counts)):
        axial=np.linspace(a,b,count+1)
        radial=np.linspace(ra,rb,count+1)
        for j in range(count):
            radius=float(max(radial[j],radial[j+1]))
            excess=float(abs(radial[j+1]-radial[j]))
            if excess>budget+2e-15:
                raise ArithmeticError("Floating-point subdivision violates radial budget")
            sections.append(BladeCylinder(float(axial[j]),float(axial[j+1]),radius,excess,interval))
    return tuple(sections)


def blade_profile_points(profile=None):
    """Discard the guard's same-station outer step, never include it as blade."""
    profile=load_profile() if profile is None else profile
    try:
        values=np.asarray(profile['profile_s_r_m'],dtype=float)
        start=float(profile['dimensions']['guard_far_m'])
        end=float(profile['dimensions']['tip_from_grip_m'])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Measured blade profile with explicit guard/tip stations required") from exc
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():
        raise ValueError("Finite N by 2 profile required")
    matches=np.flatnonzero(np.isclose(values[:,0],start,rtol=0,atol=1e-12))
    if not len(matches):
        raise ValueError("No blade-side guard station in profile")
    result=values[matches[-1]:].copy()
    if (not len(result) or not np.isclose(result[-1,0],end,rtol=0,atol=1e-12)
            or np.any(np.diff(result[:,0])<=0) or np.any(result[:,1]<=0)):
        raise ValueError("Blade profile must end at tip and increase strictly after guard")
    return result


def blade_cylinder_sections(profile=None, max_radial_excess_m=DEFAULT_MAX_RADIAL_EXCESS_M,
                            max_sections=DEFAULT_MAX_SECTIONS):
    return adaptive_cylinder_sections(blade_profile_points(profile),max_radial_excess_m,max_sections)


def blade_geometry_metadata(profile=None, max_radial_excess_m=DEFAULT_MAX_RADIAL_EXCESS_M,
                            radial_segments=64):
    profile=load_profile() if profile is None else profile
    points=blade_profile_points(profile)
    sections=adaptive_cylinder_sections(points,max_radial_excess_m)
    if type(radial_segments) is not int or radial_segments<8:
        raise ValueError("Visual radial segments must be an integer >= 8")
    chord=float(points[:,1].max()*(1-math.cos(math.pi/radial_segments)))
    profile_hash=hashlib.sha256(json.dumps({'dimensions':profile['dimensions'],
        'profile_s_r_m':profile['profile_s_r_m']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'version':BLADE_GEOMETRY_VERSION,'section_count':len(sections),
        'profile_geometry_sha256':profile_hash,
        'profile_hash_encoding':'canonical JSON of dimensions and profile_s_r_m, UTF-8',
        'declared_max_radial_excess_m':float(max_radial_excess_m),
        'realized_max_radial_excess_m':max(s.radial_excess_bound_m for s in sections),
        'circular_profile_to_visual_polygon_max_inset_m':chord,
        'collision_to_visible_surface_radial_bound_m':float(max_radial_excess_m)+chord,
        'visual_radial_segments':radial_segments,'source_profile_interval_count':len(points)-1,
        'guard_far_station_m':float(points[0,0]),'tip_station_m':float(points[-1,0]),
        'axial_gaps_m':0.,'caps_extend_beyond_blade':False,
        'collision_scoring_shape':'identical_native_flat_ended_cylinders',
        'error_scope':'Radial envelope of measured axisymmetric piecewise-linear profile, not true slotted CAD or compliance.',
        'old_geometry_certificates_inherited':False}


def blade_collision_geom_ids(model, prefix=''):
    """Strict numeric enumeration, compatible with old three-cylinder scenes.

    Compiled model topology/names are immutable during a trial. Cache by a weak
    model key so per-step scoring does not rescan every geom or retain models.
    """
    import mujoco
    if not isinstance(prefix,str):
        raise ValueError("An explicit string prefix is required")
    cache=_GEOM_CACHE.setdefault(model,{})
    if prefix in cache:
        return cache[prefix]
    stem=prefix+'shinai_blade_collision_'
    found={}
    for i in range(model.ngeom):
        name=model.geom(i).name or ''
        if not name.startswith(stem):continue
        suffix=name[len(stem):]
        if not re.fullmatch(r'0|[1-9][0-9]*',suffix):
            raise ValueError(f"Invalid numbered blade geometry: {name}")
        index=int(suffix)
        if index in found:
            raise ValueError(f"Duplicate blade section index: {index}")
        if model.geom_type[i]!=int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise ValueError(f"Expected finite flat-ended native cylinder: {name}")
        if np.any(~np.isfinite(model.geom_size[i,:2])) or np.any(model.geom_size[i,:2]<=0):
            raise ValueError(f"Invalid blade cylinder dimensions: {name}")
        found[index]=i
    if not found or set(found)!=set(range(len(found))):
        raise ValueError(f"Missing or noncontiguous native blade sections for {prefix!r}")
    result=tuple(found[index] for index in range(len(found)))
    cache[prefix]=result
    return result


def blade_geometry_identity(model, prefix=''):
    """Compiled local collision geometry identity, independent of slot pose."""
    rows=[]
    for gid in blade_collision_geom_ids(model,prefix):
        rows.append({'name':model.geom(gid).name[len(prefix):],
                     'type':int(model.geom_type[gid]),
                     'size':model.geom_size[gid].tolist(),
                     'pos':model.geom_pos[gid].tolist(),
                     'quat':model.geom_quat[gid].tolist()})
    encoded=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return {'sha256':hashlib.sha256(encoded).hexdigest(),'sections':rows,
            'section_count':len(rows)}
