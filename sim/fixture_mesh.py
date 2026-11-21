"""Small procedural fixture visuals, in meters, using only NumPy.

These meshes illustrate holes and hollow collars; they are not manufacturing
drawings or collision/inertia models. A plate is assembled from four closed,
touching single-hole quadrants, with coincident internal walls left in place.
"""
from __future__ import annotations

import numpy as np


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _segments(value):
    if type(value) is not int or not 8 <= value <= 512:
        raise ValueError("segments must be an integer between 8 and 512")
    return value


def _validate_mesh(vertices, faces):
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.isfinite(vertices).all()
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Malformed fixture mesh arrays")
    triangle = vertices[faces]
    cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    scale = float(np.max(np.ptp(vertices, axis=0)))
    tolerance = 64*np.finfo(float).eps*scale*scale
    if np.any(np.linalg.norm(cross, axis=1) <= tolerance):
        raise ValueError("Degenerate fixture triangle")
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    orientation = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    if np.any(counts != 2) or np.any(np.bincount(inverse, weights=orientation) != 0):
        raise ValueError("Fixture mesh is not closed with consistent winding")
    volume = np.einsum("ij,ij->i", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])).sum()/6
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Fixture mesh is not outward oriented")
    return vertices, faces


def _annular_extrusion(inner_xy, outer_xy, z_low, z_high):
    """Extrude corresponding CCW inner/outer loops, leaving the center open."""
    inner_xy, outer_xy = np.asarray(inner_xy), np.asarray(outer_xy)
    if (inner_xy.ndim != 2 or inner_xy.shape[1] != 2
            or outer_xy.shape != inner_xy.shape or len(inner_xy) < 8):
        raise ValueError("Invalid fixture boundary loops")
    count = len(inner_xy)
    vertices = np.concatenate([
        np.column_stack((inner_xy, np.full(count, z_low))),
        np.column_stack((outer_xy, np.full(count, z_low))),
        np.column_stack((inner_xy, np.full(count, z_high))),
        np.column_stack((outer_xy, np.full(count, z_high))),
    ]).astype(np.float64)
    faces = []
    for i in range(count):
        j = (i+1) % count
        ib, jb = i, j
        ob, oj = count+i, count+j
        it, jt = 2*count+i, 2*count+j
        ot, tj = 3*count+i, 3*count+j
        faces.extend((
            (ib, jb, ob), (jb, oj, ob),       # lower annulus, -Z
            (it, ot, jt), (ot, tj, jt),       # upper annulus, +Z
            (ob, oj, ot), (oj, tj, ot),       # external side
            (ib, it, jb), (jb, it, jt),       # bore wall, toward hole
        ))
    return _validate_mesh(vertices, np.asarray(faces, dtype=np.int32))


def _rectangle_angles(center, bounds, segments):
    xmin, xmax, ymin, ymax = bounds
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    corner_vectors = corners-center
    corner_angles = np.mod(np.arctan2(corner_vectors[:, 1], corner_vectors[:, 0]), 2*np.pi)
    angles = np.sort(np.r_[np.arange(segments)*(2*np.pi/segments), corner_angles])
    # Including corner rays makes the rectangle exact. Merge roundoff-level
    # duplicates to avoid skinny zero-area triangles at coincident rays.
    unique = [float(angles[0])]
    for angle in angles[1:]:
        if angle-unique[-1] > 1e-10:
            unique.append(float(angle))
    if len(unique) > 1 and 2*np.pi-unique[-1]+unique[0] <= 1e-10:
        unique.pop()
    return np.asarray(unique)


def perforated_plate(width_m=.05, thickness_m=.006, pitch_m=.028284271247,
                     hole_radius_m=.00265, segments=48):
    """Return a square plate with four actual through-holes at (+/-p/2,+/-p/2).

    Plate X/Y are centered at zero; mount-frame Z spans 0..thickness_m. The
    quadrant construction intentionally keeps coincident internal partition
    walls and is intended for rendering, not a manufacturing-grade union mesh.
    """
    width = _positive(width_m, "width_m")
    thickness = _positive(thickness_m, "thickness_m")
    pitch = _positive(pitch_m, "pitch_m")
    radius = _positive(hole_radius_m, "hole_radius_m")
    segments = _segments(segments)
    if pitch >= width or radius >= min(pitch, width-pitch)/2:
        raise ValueError("Holes must fit strictly inside their plate quadrants")
    vertices, faces, offset = [], [], 0
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        center = np.array([sx*pitch/2, sy*pitch/2])
        xmin, xmax = sorted((0., sx*width/2))
        ymin, ymax = sorted((0., sy*width/2))
        bounds = xmin, xmax, ymin, ymax
        angles = _rectangle_angles(center, bounds, segments)
        direction = np.column_stack((np.cos(angles), np.sin(angles)))
        dx, dy = direction[:, 0], direction[:, 1]
        tx, ty = np.full(len(angles), np.inf), np.full(len(angles), np.inf)
        for component, positive, negative, distances in (
            (dx, xmax-center[0], xmin-center[0], tx),
            (dy, ymax-center[1], ymin-center[1], ty),
        ):
            plus, minus = component > 1e-14, component < -1e-14
            distances[plus] = positive/component[plus]
            distances[minus] = negative/component[minus]
        outer_distance = np.minimum(tx, ty)
        if not np.isfinite(outer_distance).all() or np.any(outer_distance <= radius):
            raise ValueError("Invalid radial intersection with plate rectangle")
        inner = center+radius*direction
        outer = center+outer_distance[:, None]*direction
        part_vertices, part_faces = _annular_extrusion(inner, outer, 0., thickness)
        vertices.append(part_vertices)
        faces.append(part_faces+offset)
        offset += len(part_vertices)
    return _validate_mesh(np.concatenate(vertices), np.concatenate(faces))


def collar_mesh(inner_radius_m, outer_radius_m, width_m, segments=48):
    """Return an actual hollow annular collar, centered on Z=0, in meters."""
    inner = _positive(inner_radius_m, "inner_radius_m")
    outer = _positive(outer_radius_m, "outer_radius_m")
    width = _positive(width_m, "width_m")
    segments = _segments(segments)
    if inner >= outer:
        raise ValueError("inner_radius_m must be smaller than outer_radius_m")
    angles = np.arange(segments)*(2*np.pi/segments)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    return _annular_extrusion(inner*circle, outer*circle, -width/2, width/2)
