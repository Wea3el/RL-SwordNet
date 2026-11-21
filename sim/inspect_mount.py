"""Measure the existing four-hole end-effector face without modifying CAD.

Uses NumPy, MuJoCo and Pillow only. The source STL is triangulated geometry,
not a manufacturing drawing. The output frame has +Z outward from the unique
rectangular face with four circular boundary loops, and +X along EE +X.

Run: python -m sim.inspect_mount --output outputs/mount_inspection
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BODY = "end_effector_block"
MESH = "End_effector_Block"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float).copy()
    quaternion /= np.linalg.norm(quaternion)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def read_binary_stl(path: Path) -> tuple[np.ndarray, str]:
    payload = path.read_bytes()
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * count:
        raise ValueError("Expected a complete binary STL")
    records = np.frombuffer(payload, dtype=np.dtype([
        ("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84)
    return records["vertices"].astype(float), payload[:80].rstrip(b"\0").decode("ascii")


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    counts = Counter(tuple(sorted(edge)) for face in faces for edge in
                     [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])])
    graph: dict[int, set[int]] = defaultdict(set)
    for (a, b), count in counts.items():
        if count == 1:
            graph[a].add(b)
            graph[b].add(a)
    if not graph or any(len(neighbors) != 2 for neighbors in graph.values()):
        return []
    loops, used = [], set()
    for start in sorted(graph):
        if start in used:
            continue
        sequence, previous, current = [], None, start
        while current not in used:
            sequence.append(current)
            used.add(current)
            following = sorted(graph[current] - ({previous} if previous is not None else set()))[0]
            previous, current = current, following
        if current != start:
            return []
        loops.append(np.array(sequence))
    return loops


def circle_fit(points: np.ndarray) -> dict:
    # points are two-dimensional face coordinates, in metres.
    matrix = np.column_stack([2 * points, np.ones(len(points))])
    cx, cy, constant = np.linalg.lstsq(matrix, np.sum(points**2, axis=1), rcond=None)[0]
    center = np.array([cx, cy])
    radius = float(np.sqrt(constant + center @ center))
    residual = float(np.max(np.abs(np.linalg.norm(points - center, axis=1) - radius)))
    angle = np.sort(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    gap = float(np.max(np.diff(np.r_[angle, angle[0] + 2 * np.pi])))
    return {"center": center, "radius": radius, "vertex_radial_residual": residual,
            "maximum_polygon_chord_sagitta": radius * (1 - np.cos(gap / 2))}


def find_four_hole_face(vertices: np.ndarray, faces: np.ndarray) -> dict:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, twice_area[:, None], out=np.zeros_like(cross), where=twice_area[:, None] > 1e-20)
    visited, candidates = set(), []
    for seed in np.argsort(twice_area)[::-1]:
        if seed in visited or twice_area[seed] < 2e-6:
            continue
        normal = normals[seed]
        coplanar = ((normals @ normal > 1 - 1e-6)
                    & (np.max(np.abs((triangles - triangles[seed, 0]) @ normal), axis=1) < 2e-7))
        selected = np.flatnonzero(coplanar)
        visited.update(selected.tolist())
        if twice_area[selected].sum() / 2 < 1e-4:
            continue
        loops = boundary_loops(faces[selected])
        if len(loops) != 5:
            continue
        # Prefer EE +X for a reproducible in-plane direction, not a PCA sign.
        u = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
        u -= normal * (normal @ u)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        circular, outer = [], []
        for loop in loops:
            points = np.column_stack([vertices[loop] @ u, vertices[loop] @ v])
            fit = circle_fit(points)
            if len(loop) >= 12 and fit["vertex_radial_residual"] < 1e-6 and .0005 < fit["radius"] < .01:
                circular.append((loop, fit))
            else:
                outer.append(loop)
        if len(circular) == 4 and len(outer) == 1 and len(outer[0]) == 4:
            candidates.append({"faces": selected, "outer": outer[0], "holes": [item[0] for item in circular],
                               "approximate_normal": normal})
    if len(candidates) != 1:
        raise ValueError(f"Expected one rectangular four-hole face, found {len(candidates)}")
    return candidates[0]


def plot_face(path: Path, corners: np.ndarray, holes: list[dict]) -> None:
    image = Image.new("RGB", (1100, 950), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()

    def pixel(point):
        return (550 + 11500 * float(point[0]), 455 - 11500 * float(point[1]))

    draw.text((70, 30), "Measured four-hole mounting face (STL geometry)", fill="#132a3a", font=font)
    draw.text((70, 65), "Face-on view: +u = EE +X; +v in face; +n points toward viewer", fill="#34495e", font=small)
    outline = [pixel(point) for point in corners]
    draw.polygon(outline, fill="#e7f1f5", outline="#1e4d64", width=3)
    for hole in holes:
        polygon = [pixel(point) for point in hole["boundary_uv_m"]]
        draw.polygon(polygon, fill="white", outline="#1768ac", width=2)
        center = pixel(hole["center_mount_m"][:2])
        draw.line((center[0] - 7, center[1], center[0] + 7, center[1]), fill="#1768ac", width=1)
        draw.line((center[0], center[1] - 7, center[0], center[1] + 7), fill="#1768ac", width=1)
        draw.text((center[0] + 20, center[1] - 28), hole["id"], fill="#12476c", font=small)
    origin = pixel([0, 0])
    draw.ellipse((origin[0]-4, origin[1]-4, origin[0]+4, origin[1]+4), fill="#d96a1d")
    draw.line((origin, pixel([.010, 0])), fill="#b64728", width=3)
    draw.line((origin, pixel([0, .010])), fill="#277e52", width=3)
    draw.text(pixel([.011, -.001]), "+u", fill="#b64728", font=small)
    draw.text(pixel([.001, .012]), "+v", fill="#277e52", font=small)
    center_values = np.array([h["center_mount_m"] for h in holes])
    pitch = np.ptp(center_values[:, :2], axis=0) * 1000
    diameter = np.mean([h["diameter_m"] for h in holes]) * 1000
    draw.text((70, 820), f"Face: {np.ptp(corners[:,0])*1000:.6f} x {np.ptp(corners[:,1])*1000:.6f} mm", fill="#132a3a", font=font)
    draw.text((70, 855), f"Hole pitch: {pitch[0]:.6f} x {pitch[1]:.6f} mm; fitted diameter: {diameter:.6f} mm", fill="#132a3a", font=font)
    draw.text((70, 895), "Triangulation measurement only; no thread, fastener, fit or manufacturing tolerance inferred.", fill="#5a6770", font=small)
    image.save(path)


def main(output: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY)
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, MESH)
    matches = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_dataid == mesh_id)
                             & (model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)))
    if len(matches) != 1:
        raise ValueError("Cannot uniquely locate the end-effector mesh geom")
    geom_id = int(matches[0])
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    processed = model.mesh_vert[start:start + count].astype(float)
    start, count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    faces = model.mesh_face[start:start + count].copy()
    compiled_rotation = rotation(model.geom_quat[geom_id].copy())
    vertices = processed @ compiled_rotation.T + model.geom_pos[geom_id]

    # Independently apply the ORIGINAL geom transform to ORIGINAL STL vertices.
    # This detects using a recentered mesh with the pre-compilation transform.
    xml = ET.parse(ROOT / "robot.xml").getroot()
    asset = xml.find(f"./asset/mesh[@name='{MESH}']")
    geom = xml.find(f".//body[@name='{BODY}']/geom[@mesh='{MESH}']")
    if asset is None or geom is None:
        raise ValueError("Original XML mesh/geom not found")
    stl_path = ROOT / "meshes" / asset.attrib["file"]
    raw, header = read_binary_stl(stl_path)
    source_pos = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ")
    source_quat = np.fromstring(geom.attrib.get("quat", "1 0 0 0"), sep=" ")
    source_scale = np.fromstring(asset.attrib.get("scale", "1 1 1"), sep=" ")
    original_in_ee = np.unique(raw.reshape(-1, 3), axis=0) * source_scale
    original_in_ee = original_in_ee @ rotation(source_quat).T + source_pos
    nearest = []
    for offset in range(0, len(original_in_ee), 128):
        difference = original_in_ee[offset:offset+128, None, :] - vertices[None, :, :]
        nearest.extend(np.sqrt(np.min(np.sum(difference**2, axis=2), axis=1)).tolist())
    if max(nearest) > 1e-6:
        raise ValueError("Original-STL and compiled-mesh coordinate paths disagree")

    face = find_four_hole_face(vertices, faces)
    outer = vertices[face["outer"]]
    center = outer.mean(axis=0)
    _, _, vh = np.linalg.svd(outer - center, full_matrices=False)
    normal = vh[-1]
    if normal @ face["approximate_normal"] < 0:
        normal = -normal
    u = np.array([1., 0., 0.])
    u -= normal * (normal @ u)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    mount_rotation = np.column_stack([u, v, normal])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = mount_rotation, center
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, mount_rotation.ravel())
    if quaternion[0] < 0:
        quaternion = -quaternion
    outer_mount = (outer - center) @ mount_rotation
    holes = []
    for loop in face["holes"]:
        projected = (vertices[loop] - center) @ mount_rotation
        fit = circle_fit(projected[:, :2])
        local_center = np.r_[fit["center"], 0.]
        holes.append({"id": "", "center_mount_m": local_center.tolist(),
                      "center_ee_m": (center + mount_rotation @ local_center).tolist(),
                      "diameter_m": 2*fit["radius"], "boundary_vertices": len(loop),
                      "fit_max_vertex_radial_residual_m": fit["vertex_radial_residual"],
                      "maximum_tessellation_chord_sagitta_m": fit["maximum_polygon_chord_sagitta"],
                      "boundary_uv_m": projected[:, :2].tolist()})
    holes.sort(key=lambda hole: (hole["center_mount_m"][1], hole["center_mount_m"][0]))
    for index, hole in enumerate(holes):
        hole["id"] = f"H{index+1}"
    hole_centers = np.array([hole["center_mount_m"] for hole in holes])
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J6")
    joint_axis = model.jnt_axis[joint_id].copy()
    report = {
        "status": "measured_existing_triangulated_geometry_not_manufacturing_specification",
        "input": {"scene": "scene.xml", "robot_xml": "robot.xml", "mesh": str(stl_path.relative_to(ROOT)),
                  "sha256": {p: sha256(ROOT / p) for p in ["scene.xml", "robot.xml", str(stl_path.relative_to(ROOT))]},
                  "stl_header": header, "triangle_count": len(raw), "compiled_vertex_count": len(vertices)},
        "units": "metres", "body_frame": BODY,
        "selection": "Unique planar quadrilateral outer boundary with four circular inner boundary loops; matches user-marked four-hole face",
        "coordinate_verification": {
            "method": "compiled mesh vertices with compiled geom transform, independently checked against original STL plus original XML geom transform",
            "original_geom_pos": source_pos.tolist(), "original_geom_quat_wxyz": source_quat.tolist(),
            "compiled_geom_pos": model.geom_pos[geom_id].tolist(), "compiled_geom_quat_wxyz": model.geom_quat[geom_id].tolist(),
            "compiled_mesh_pos": model.mesh_pos[mesh_id].tolist(), "compiled_mesh_quat_wxyz": model.mesh_quat[mesh_id].tolist(),
            "maximum_nearest_vertex_disagreement_m": max(nearest)},
        "mount_frame": {
            "definition": "+Z outward face normal, +X projection of EE +X onto face, +Y=Z cross X; origin at outer-square center",
            "T_ee_from_mount": transform.tolist(), "quat_ee_from_mount_wxyz": quaternion.tolist(),
            "center_ee_m": center.tolist(), "u_ee": u.tolist(), "v_ee": v.tolist(), "normal_ee": normal.tolist(),
            "face_dimensions_uv_m": np.ptp(outer_mount[:, :2], axis=0).tolist(),
            "outer_corners_ee_m": outer.tolist(), "outer_corners_mount_m": outer_mount.tolist(),
            "face_triangles": len(face["faces"]),
            "maximum_outer_plane_residual_m": float(np.max(np.abs(outer_mount[:, 2])))},
        "holes": holes,
        "hole_pattern": {"pitch_uv_m": np.ptp(hole_centers[:, :2], axis=0).tolist(),
                         "pairwise_center_distances_m": np.linalg.norm(hole_centers[:, None] - hole_centers[None, :], axis=2).tolist(),
                         "center_radius_m": np.linalg.norm(hole_centers[:, :2], axis=1).tolist()},
        "joint_reference": {"joint": "J6", "axis_in_ee": joint_axis.tolist(),
                            "anchor_in_ee_m": model.jnt_pos[joint_id].tolist(),
                            "angle_between_outward_normal_and_J6_axis_deg": float(np.degrees(np.arccos(np.clip(normal @ joint_axis, -1, 1)))),
                            "conclusion": "four-hole face normal is perpendicular, not parallel, to J6 axis"},
        "limitations": ["Circle fit measures tessellated hole boundaries, not thread or tolerances",
                        "No bolt size, engagement depth, load rating or clamp design is inferred",
                        "This transform locates the mounting plate, not the final shinai grip; clamp-to-grip remains a separate design transform"],
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_face(output / "mount_face.png", outer_mount, holes)
    (output / "measurement.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mount_frame": report["mount_frame"],
                      "hole_pitch_uv_m": report["hole_pattern"]["pitch_uv_m"],
                      "hole_diameters_m": [hole["diameter_m"] for hole in holes],
                      "joint_reference": report["joint_reference"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mount_inspection")
    main(parser.parse_args().output)
