"""Range-preserving C2 joint references for a candidate motion/pose graph.

This is only a numerical reference generator, not an IK, collision, tracking,
motor-limit or fencing-effectiveness certificate.  It does not alter the old A1
bank.  Consecutive waypoints may be passed with nonzero velocity; a reversal or
an explicit hold stops the affected joint.  Endpoints are always at rest.
"""
from __future__ import annotations

import math

import numpy as np


def _real_array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numeric values")
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    return result


def _readonly(array):
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _bezier(controls, u):
    """Stable de Casteljau evaluation; convex combinations preserve the range."""
    values = np.array(controls, copy=True)
    for remaining in range(len(values)-1, 0, -1):
        values[:remaining] = (1.-u)*values[:remaining]+u*values[1:remaining+1]
    return values[0].copy()


def _power_coefficients(controls):
    degree = len(controls)-1
    # Removing the common offset avoids cancellation of large absolute joint
    # coordinates in the higher-order coefficients.
    relative = controls-controls[0]
    coeff = np.zeros_like(controls)
    for k in range(degree+1):
        for i in range(k+1):
            coeff[k] += (math.comb(degree, i)*math.comb(degree-i, k-i)
                         * (-1.)**(k-i)*relative[i])
    coeff[0] += controls[0]
    return coeff


def _critical_points(coefficients):
    """Real roots of d/du, including both endpoints of the unit interval."""
    derivative = np.polynomial.polynomial.polyder(coefficients)
    points = [0., 1.]
    scale = float(np.max(np.abs(derivative))) if len(derivative) else 0.
    if scale:
        # Normalization avoids changing roots while reducing scale sensitivity.
        roots = np.polynomial.polynomial.polyroots(derivative/scale)
        for root in roots:
            if abs(root.imag) <= 1e-9 and -1e-12 <= root.real <= 1.+1e-12:
                points.append(float(np.clip(root.real, 0., 1.)))
    return points


class C2JointSpline:
    """Piecewise quintic, seven-joint spline with a proven endpoint envelope.

    For an internal knot, each joint's velocity is the signed minimum of its
    adjacent secants if both have the same nonzero sign, otherwise zero.  All
    knot accelerations are zero.  For an interval of length h and displacement
    d, the Bezier control points are

        q0, q0+h*v0/5, q0+2*h*v0/5,
        q1-2*h*v1/5, q1-h*v1/5, q1.

    Since abs(v0), abs(v1) <= abs(d/h), those controls are ordered between the
    endpoints.  Convex-hull containment thus applies to EVERY interpolated
    point, not just the sampled waypoints.  This only preserves the supplied
    joint-coordinate intervals; those intervals still need physical validation.
    """

    def __init__(self, time_s, qpos):
        times, q = _real_array(time_s, "time_s"), _real_array(qpos, "qpos")
        if times.ndim != 1 or len(times) < 2 or q.shape != (len(times), 7):
            raise ValueError("An increasing N-vector and matching Nx7 qpos are required")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            h, delta = np.diff(times), np.diff(q, axis=0)
            secants = delta/h[:, None]
        if (not np.isfinite(h).all() or np.any(h <= 0.)
                or not np.isfinite(delta).all() or not np.isfinite(secants).all()
                or not np.isfinite(times[-1]-times[0])):
            raise ValueError("Strictly increasing representable times and finite secants required")
        velocities = np.zeros_like(q)
        left, right = secants[:-1], secants[1:]
        same = (np.sign(left) == np.sign(right)) & (left != 0.) & (right != 0.)
        velocities[1:-1] = np.where(
            same, np.sign(left)*np.minimum(np.abs(left), np.abs(right)), 0.)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            outgoing = h[:, None]*velocities[:-1]/5.
            incoming = h[:, None]*velocities[1:]/5.
            controls = np.stack((q[:-1], q[:-1]+outgoing, q[:-1]+2.*outgoing,
                q[1:]-2.*incoming, q[1:]-incoming, q[1:]), axis=1)
            coefficients = np.asarray([_power_coefficients(c) for c in controls])
            derived = [controls]
            for order in range(1, 4):
                derived.append((6-order)*np.diff(derived[-1], axis=1)/h[:, None, None])
        if any(not np.isfinite(a).all() for a in (*derived, coefficients)):
            raise ValueError("Spline polynomial/derivatives exceed finite numeric precision")
        self.knot_time_s, self.knot_qpos = _readonly(times), _readonly(q)
        self.knot_velocities = _readonly(velocities)
        self.bezier_controls, self.coefficients = _readonly(controls), _readonly(coefficients)
        self._durations = _readonly(h)
        self._derived_controls = tuple(_readonly(a) for a in derived)

    @property
    def duration_s(self):
        return float(self.knot_time_s[-1]-self.knot_time_s[0])

    def _evaluate_one(self, time, order):
        times = self.knot_time_s
        if time <= times[0]:
            return self.knot_qpos[0].copy() if order == 0 else np.zeros(7)
        if time >= times[-1]:
            return self.knot_qpos[-1].copy() if order == 0 else np.zeros(7)
        index = int(np.searchsorted(times, time, side="right"))-1
        if time == times[index]:
            if order == 0:
                return self.knot_qpos[index].copy()
            if order == 1:
                return self.knot_velocities[index].copy()
            if order == 2:
                return np.zeros(7)
        u = float((time-times[index])/self._durations[index])
        return _bezier(self._derived_controls[order][index], u)

    def _evaluate(self, time_s, order):
        times = _real_array(time_s, "sample time")
        values = [self._evaluate_one(float(t), order) for t in times.ravel()]
        return np.asarray(values).reshape(times.shape+(7,))

    def sample(self, time_s):
        """Return q, holding the rest endpoints outside the specified clock."""
        return self._evaluate(time_s, 0)

    def derivatives(self, time_s):
        """Return (dq/dt, d2q/dt2); time arrays retain their shape plus 7."""
        return self._evaluate(time_s, 1), self._evaluate(time_s, 2)

    def analytic_extrema(self):
        """Polynomial critical-point extrema, not a finite sampling claim."""
        minima, maxima = [], []
        derivative_peaks = {order: np.zeros(7) for order in (1, 2, 3)}
        for i, coeff in enumerate(self.coefficients):
            lower, upper = np.empty(7), np.empty(7)
            for joint in range(7):
                for order in range(4):
                    polynomial = np.polynomial.polynomial.polyder(coeff[:, joint], m=order)
                    points = _critical_points(polynomial)
                    values = [_bezier(self._derived_controls[order][i], u)[joint] for u in points]
                    if order == 0:
                        lower[joint], upper[joint] = min(values), max(values)
                    else:
                        derivative_peaks[order][joint] = max(
                            derivative_peaks[order][joint], float(np.max(np.abs(values))))
            minima.append(lower); maxima.append(upper)
        return {"interval_min_rad": np.asarray(minima).tolist(),
            "interval_max_rad": np.asarray(maxima).tolist(),
            "position_min_rad": np.min(minima, axis=0).tolist(),
            "position_max_rad": np.max(maxima, axis=0).tolist(),
            "peak_speed_rad_s": derivative_peaks[1].tolist(),
            "peak_acceleration_rad_s2": derivative_peaks[2].tolist(),
            "peak_jerk_rad_s3": derivative_peaks[3].tolist(),
            "method": "unit-interval polynomial derivative roots plus endpoints"}

    def range_report(self):
        extrema = self.analytic_extrema()
        lower = np.minimum(self.knot_qpos[:-1], self.knot_qpos[1:])
        upper = np.maximum(self.knot_qpos[:-1], self.knot_qpos[1:])
        actual_lower, actual_upper = np.asarray(extrema["interval_min_rad"]), np.asarray(extrema["interval_max_rad"])
        controls = self.bezier_controls
        violation = max(0., float(np.max(lower-actual_lower)), float(np.max(actual_upper-upper)))
        control_violation = max(0., float(np.max(lower[:, None]-controls)),
                                float(np.max(controls-upper[:, None])))
        return {"interval_endpoint_min_rad": lower.tolist(), "interval_endpoint_max_rad": upper.tolist(),
            "maximum_analytic_overshoot_rad": violation,
            "maximum_control_envelope_violation_rad": control_violation,
            "position_min_rad": extrema["position_min_rad"], "position_max_rad": extrema["position_max_rad"],
            "containment_basis": "monotone Bezier controls bounded by adjacent waypoint coordinates",
            "physical_joint_bounds_supplied": False}

    def continuity_report(self):
        jumps = []
        for order in range(3):
            c = self._derived_controls[order]
            jumps.append(float(np.max(np.abs(c[:-1, -1]-c[1:, 0]))) if len(c) > 1 else 0.)
        return {"maximum_position_jump_rad": jumps[0], "maximum_velocity_jump_rad_s": jumps[1],
            "maximum_acceleration_jump_rad_s2": jumps[2],
            "endpoint_velocity_max_rad_s": float(np.max(np.abs(self.knot_velocities[[0, -1]]))),
            "endpoint_acceleration_max_rad_s2": float(np.max(np.abs(self._derived_controls[2][[0, -1]][[0, 1], [0, -1]]))),
            "continuity": "C2 reference; jerk continuity is not promised"}

    def report(self):
        holds = np.all(self.knot_qpos[1:] == self.knot_qpos[:-1], axis=1)
        return {"schema_version": 1, "kind": "range_preserving_c2_joint_reference",
            "waypoint_count": len(self.knot_time_s), "duration_s": self.duration_s,
            "start_time_s": float(self.knot_time_s[0]), "end_time_s": float(self.knot_time_s[-1]),
            "hold_duration_s": float(np.sum(self._durations[holds])),
            "hold_intervals": [{"start_s": float(self.knot_time_s[i]),
                "end_s": float(self.knot_time_s[i+1])} for i in np.flatnonzero(holds)],
            "maximum_adjacent_waypoint_delta_rad": np.max(np.abs(np.diff(self.knot_qpos, axis=0)), axis=0).tolist(),
            "range": self.range_report(), "continuity": self.continuity_report(),
            "extrema": self.analytic_extrema(), "physical_validation_passed": False,
            "scope": "Reference geometry/timing only; IK, collision, dynamics, tracking and fencing validation remain required"}
