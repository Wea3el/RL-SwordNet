"""Explicit shared numerical profile, not a change to historical defaults.

Future A1 and A2 environments must opt into the SAME profile. This module
changes only integration and clocks, never geometry, forces or scoring gates.
Applying a profile does not admit a skill bank or start training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .skill_validation import V2_CONFIG


_FIELDS = {"schema_version", "name", "integrator", "physics_timestep_s",
           "control_period_s", "decision_period_s"}


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    integrator: str
    physics_timestep_s: float
    control_period_s: float
    decision_period_s: float
    source_sha256: str
    source_bytes: bytes = field(repr=False)

    def runtime_config(self):
        """An independent v2 configuration with explicit numeric provenance."""
        _validate(self)
        return dict(V2_CONFIG,
            physics_timestep_s=self.physics_timestep_s,
            control_period_s=self.control_period_s,
            decision_period_s=self.decision_period_s,
            explicit_integrator=self.integrator,
            numerical_profile=self.name,
            numerical_profile_sha256=self.source_sha256)


def _validate(profile):
    if not isinstance(profile, RuntimeProfile):
        raise ValueError("An explicit loaded RuntimeProfile is required")
    if not isinstance(profile.name, str) or not profile.name.strip():
        raise ValueError("Profile name must be nonempty")
    if profile.integrator != "RK4":
        raise ValueError("Only the explicitly tested RK4 candidate is supported here")
    clocks = (profile.physics_timestep_s, profile.control_period_s,
              profile.decision_period_s)
    for clock in clocks:
        if isinstance(clock, (bool, str)) or not isinstance(clock, (int, float)) \
                or not np.isfinite(clock) or clock <= 0:
            raise ValueError("Clocks must be finite positive numbers")
    dt, control, decision = clocks
    if not all(x >= dt and np.isclose(x/dt, round(x/dt), rtol=0., atol=1e-9)
               for x in (control, decision)):
        raise ValueError("Control and decision periods must divide physics steps")
    if not np.isclose(decision/control, round(decision/control), rtol=0., atol=1e-9):
        raise ValueError("Decision period must divide control ticks")
    digest = profile.source_sha256
    if not isinstance(digest, str) or len(digest) != 64 \
            or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("A numeric profile must retain its source SHA256")
    if not isinstance(profile.source_bytes, bytes) \
            or hashlib.sha256(profile.source_bytes).hexdigest() != digest:
        raise ValueError("Profile source bytes must match the recorded SHA256")
    source = json.loads(profile.source_bytes)
    if not isinstance(source, dict) or set(source) != _FIELDS \
            or type(source["schema_version"]) is not int or source["schema_version"] != 1 \
            or any(source[k] != getattr(profile, k) for k in _FIELDS-{"schema_version"}):
        raise ValueError("Changed numeric fields require a new explicit source JSON")


def load_runtime_profile(path):
    """Load a numeric-only JSON; unknown fields cannot override physical rules."""
    source = Path(path).read_bytes()
    content = json.loads(source)
    if not isinstance(content, dict) or set(content) != _FIELDS \
            or type(content["schema_version"]) is not int or content["schema_version"] != 1:
        raise ValueError("Expected schema 1 and only the explicit numeric profile fields")
    profile = RuntimeProfile(**{k: content[k] for k in _FIELDS if k != "schema_version"},
                             source_sha256=hashlib.sha256(source).hexdigest(),
                             source_bytes=source)
    _validate(profile)
    return profile


def assert_runtime_profile(scene, profile):
    """Check live model AND exported spec, not merely a JSON label."""
    _validate(profile)
    expected = int(mujoco.mjtIntegrator.mjINT_RK4)
    for option in (scene.model.opt, scene.spec.option):
        if int(option.integrator) != expected \
                or not np.isclose(option.timestep, profile.physics_timestep_s,
                                  rtol=0., atol=1e-15):
            raise ValueError("Live model and frozen spec must match the explicit profile")


def apply_runtime_profile(scene, profile):
    """Opt in before an episode; no mid-episode setting or state rewrites."""
    _validate(profile)
    if not np.isfinite(scene.data.time) or scene.data.time != 0.:
        raise ValueError("Apply numerical settings only before an episode starts")
    for option in (scene.model.opt, scene.spec.option):
        option.integrator = mujoco.mjtIntegrator.mjINT_RK4
        option.timestep = profile.physics_timestep_s
    assert_runtime_profile(scene, profile)
    return profile.runtime_config()
