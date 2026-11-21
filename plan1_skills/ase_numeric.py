"""Read checksum-verified ASE sword motions without executing their pickle.

This is a narrow, data-only interpreter for the audited ASE SkeletonMotion NPY
layout, not a general-purpose safe pickle loader. No upstream Python is imported
or executed. GLOBAL is represented as a symbol and REDUCE is interpreted only
for the explicitly supported numeric/container layouts, never called.

Source schema: nv-tlabs/ASE, commit
6f9b4f1f289603eee6a4f45d082bc6e6d83fecec, under
ase/data/motions/reallusion_sword_shield. Its motion README says "strictly for
noncommercial use"; this module does not grant redistribution rights. Download
and derived motion arrays are intentionally not included here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import pickletools
import re

import numpy as np


UPSTREAM_REPOSITORY = "https://github.com/nv-tlabs/ASE"
UPSTREAM_COMMIT = "6f9b4f1f289603eee6a4f45d082bc6e6d83fecec"
UPSTREAM_MOTION_DIRECTORY = "ase/data/motions/reallusion_sword_shield"
_NODES = (
    "pelvis", "torso", "head", "right_upper_arm", "right_lower_arm",
    "right_hand", "sword", "left_upper_arm", "left_lower_arm", "shield",
    "left_hand", "right_thigh", "right_shin", "right_foot", "left_thigh",
    "left_shin", "left_foot",
)
_PARENTS = (-1, 0, 1, 1, 3, 4, 5, 1, 7, 8, 8, 0, 11, 12, 0, 14, 15)
_SYMBOLS = {
    "numpy.core.multiarray _reconstruct", "numpy ndarray", "numpy dtype",
    "collections OrderedDict", "numpy.core.multiarray scalar",
}
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_ARRAY_ELEMENTS = 2_000_000


@dataclass
class _DType:
    code: str
    order: str = "="


@dataclass
class _Array:
    value: object = None


def _unwrap(value, active=None, depth=0):
    """Remove interpreter wrappers; reject cycles rather than following them."""
    if depth > 100:
        raise ValueError("ASE data nesting is too deep")
    if active is None:
        active = set()
    if not isinstance(value, (_Array, dict, list, tuple)):
        return value
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic ASE data is unsupported")
    active.add(identity)
    try:
        if isinstance(value, _Array):
            return _unwrap(value.value, active, depth + 1)
        if isinstance(value, dict):
            return {k: _unwrap(v, active, depth + 1) for k, v in value.items()}
        result = [_unwrap(v, active, depth + 1) for v in value]
        return tuple(result) if isinstance(value, tuple) else result
    finally:
        active.remove(identity)


def _decode_verified_bytes(raw):
    stream = io.BytesIO(raw)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("only the audited NPY 1.0 format is supported")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    if shape != () or fortran or dtype != np.dtype("O"):
        raise ValueError("expected the audited scalar object wrapper")
    payload = stream.read()
    stack, memo = [], {}
    mark = object()

    def pop_mark():
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is mark:
                values = stack[i + 1:]
                del stack[i:]
                return values
        raise ValueError("missing data marker")

    for count, (op, arg, position) in enumerate(pickletools.genops(payload)):
        if count > 100_000 or len(stack) > 10_000 or len(memo) > 10_000:
            raise ValueError("ASE data structure exceeds interpreter bounds")
        name = op.name
        if name == "PROTO":
            if arg not in (2, 3, 4):
                raise ValueError("unsupported data protocol")
        elif name == "FRAME":
            pass
        elif name == "GLOBAL":
            if arg not in _SYMBOLS:
                raise ValueError(f"unsupported GLOBAL symbol: {arg}")
            stack.append(("symbol", arg))
        elif name in ("BINPUT", "LONG_BINPUT"):
            if not stack or not 0 <= arg <= 10_000:
                raise ValueError("invalid memo entry")
            memo[arg] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
        elif name in (
            "BININT", "BININT1", "BININT2", "LONG1", "LONG4", "BINFLOAT",
            "BINUNICODE", "SHORT_BINUNICODE", "BINBYTES", "SHORT_BINBYTES",
            "BINBYTES8",
        ):
            stack.append(arg)
        elif name == "MARK":
            stack.append(mark)
        elif name in ("NEWTRUE", "NEWFALSE", "NONE"):
            stack.append({"NEWTRUE": True, "NEWFALSE": False, "NONE": None}[name])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "TUPLE":
            stack.append(tuple(pop_mark()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            size = int(name[-1])
            if len(stack) < size:
                raise ValueError("invalid tuple stack")
            values = stack[-size:]
            del stack[-size:]
            stack.append(tuple(values))
        elif name == "APPEND":
            value = stack.pop()
            stack[-1].append(value)
        elif name == "APPENDS":
            values = pop_mark()
            stack[-1].extend(values)
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            stack[-1][key] = value
        elif name == "SETITEMS":
            values = pop_mark()
            if len(values) % 2:
                raise ValueError("odd key/value list")
            for key, value in zip(values[::2], values[1::2]):
                stack[-1][key] = value
        elif name == "REDUCE":
            args, symbol = stack.pop(), stack.pop()
            if symbol == ("symbol", "numpy.core.multiarray _reconstruct"):
                if args != (("symbol", "numpy ndarray"), (0,), b"b"):
                    raise ValueError("unsupported ndarray structure")
                stack.append(_Array())
            elif symbol == ("symbol", "numpy dtype"):
                if (not isinstance(args, tuple) or len(args) != 3
                        or args[0] not in ("O8", "f8", "f4", "i8", "i4")
                        or args[1:] != (False, True)):
                    raise ValueError("unsupported dtype structure")
                stack.append(_DType(args[0]))
            elif symbol == ("symbol", "collections OrderedDict") and args == ():
                stack.append({})
            elif symbol == ("symbol", "numpy.core.multiarray scalar"):
                if not isinstance(args, tuple) or len(args) != 2:
                    raise ValueError("invalid numeric scalar")
                dt, contents = args
                if not isinstance(dt, _DType) or dt.code == "O8":
                    raise ValueError("only numeric scalars are supported")
                numeric_dtype = np.dtype(dt.order + dt.code)
                if not isinstance(contents, bytes) or len(contents) != numeric_dtype.itemsize:
                    raise ValueError("invalid numeric scalar bytes")
                stack.append(np.frombuffer(contents, dtype=numeric_dtype, count=1)[0].item())
            else:
                raise ValueError(f"unsupported REDUCE symbol: {symbol!r}")
        elif name == "BUILD":
            state = stack.pop()
            instance = stack[-1]
            if isinstance(instance, _DType):
                if (not isinstance(state, tuple) or len(state) != 8
                        or state[0] != 3 or state[1] not in ("<", ">", "|", "=")
                        or state[2:5] != (None, None, None)):
                    raise ValueError("unsupported dtype state")
                instance.order = state[1]
            elif isinstance(instance, _Array):
                if not isinstance(state, tuple) or len(state) != 5:
                    raise ValueError("invalid ndarray state")
                version, array_shape, array_dtype, order, contents = state
                if (version != 1 or not isinstance(array_shape, tuple)
                        or len(array_shape) > 4
                        or any(type(d) is not int or d < 0 for d in array_shape)
                        or math.prod(array_shape) > _MAX_ARRAY_ELEMENTS
                        or not isinstance(array_dtype, _DType) or type(order) is not bool):
                    raise ValueError("unsupported ndarray dimensions or dtype")
                if array_dtype.code == "O8":
                    if (array_shape != () or not isinstance(contents, list)
                            or len(contents) != 1 or not isinstance(contents[0], dict)):
                        raise ValueError("only one scalar dictionary wrapper is supported")
                    instance.value = contents[0]
                else:
                    numeric_dtype = np.dtype(array_dtype.order + array_dtype.code)
                    if (not isinstance(contents, bytes)
                            or len(contents) != math.prod(array_shape) * numeric_dtype.itemsize):
                        raise ValueError("invalid ndarray byte count")
                    instance.value = np.frombuffer(contents, dtype=numeric_dtype).reshape(
                        array_shape, order="F" if order else "C"
                    ).copy()
            else:
                raise ValueError("unsupported BUILD target")
        elif name == "STOP":
            if len(stack) != 1 or not isinstance(stack[0], _Array) or position + 1 != len(payload):
                raise ValueError("invalid final ASE data structure")
            result = _unwrap(stack[0])
            if not isinstance(result, dict):
                raise ValueError("motion must be a dictionary")
            return result
        else:
            raise ValueError(f"unsupported data opcode: {name}")
    raise ValueError("missing data terminator")


def _tensor(value, name, shape=None, integer=False):
    if isinstance(value, dict) and "arr" in value:
        value = value["arr"]
    if (not isinstance(value, np.ndarray)
            or value.dtype.kind not in ("iu" if integer else "f")
            or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all()):
        raise ValueError(f"invalid numeric array: {name}")
    return value


def _xyzw_to_matrix(quaternions):
    """NumPy quaternion rotation matrices; inputs are normalized by caller."""
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    result = np.empty(quaternions.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y*y + z*z)
    result[..., 0, 1] = 2 * (x*y - z*w)
    result[..., 0, 2] = 2 * (x*z + y*w)
    result[..., 1, 0] = 2 * (x*y + z*w)
    result[..., 1, 1] = 1 - 2 * (x*x + z*z)
    result[..., 1, 2] = 2 * (y*z - x*w)
    result[..., 2, 0] = 2 * (x*z - y*w)
    result[..., 2, 1] = 2 * (y*z + x*w)
    result[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def read_verified_motion(path, expected_sha256, expected_bytes):
    """Return meter-scale tool trajectories from a verified ASE motion file.

    The expected size/hash must come from the project's trusted manifest, not
    values supplied by the downloaded file. Returned grip is the right-hand
    skeletal origin, NOT a calibrated real robot clamp. Direction is grip-to-tip.
    Sword and body trajectories still need fixed-base retargeting, constraint
    validation and time scaling; this function does not make them executable.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character SHA256 digest")
    if type(expected_bytes) is not int or not 0 < expected_bytes <= _MAX_FILE_BYTES:
        raise ValueError("expected_bytes must be a positive bounded integer")
    source = Path(path)
    if source.stat().st_size != expected_bytes:
        raise ValueError("motion file size does not match trusted manifest")
    # Bound the read too: a changed file between stat and read must not cause an
    # unbounded allocation before the size/hash mismatch is detected.
    with source.open("rb") as stream:
        raw = stream.read(expected_bytes + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_bytes or digest != expected_sha256.lower():
        raise ValueError("motion SHA256 does not match trusted manifest")
    try:
        data = _decode_verified_bytes(raw)
        if data.get("__name__") != "SkeletonMotion" or data.get("is_local") is not True:
            raise ValueError("expected a local-rotation SkeletonMotion")
        tree = data["skeleton_tree"]
        if not isinstance(tree, dict) or tuple(tree["node_names"]) != _NODES:
            raise ValueError("unexpected ASE skeleton node names or order")
        parents = _tensor(tree["parent_indices"], "parent_indices", (17,), integer=True)
        if not np.array_equal(parents, _PARENTS):
            raise ValueError("unexpected ASE skeleton hierarchy")
        offsets = _tensor(tree["local_translation"], "local_translation", (17, 3))
        rotations = _tensor(data["rotation"], "rotation")
        if rotations.ndim != 3 or rotations.shape[1:] != (17, 4) or not 2 <= rotations.shape[0] <= 20_000:
            raise ValueError("unexpected motion frame/rotation dimensions")
        frames = int(rotations.shape[0])
        root = _tensor(data["root_translation"], "root_translation", (frames, 3))
        for key in ("global_velocity", "global_angular_velocity"):
            _tensor(data[key], key, (frames, 17, 3))
        norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            raise ValueError("invalid unit quaternion norms")
        rotations = rotations.astype(np.float64) / norms
        if (not np.allclose(offsets[6], [.74, 0, 0], rtol=0, atol=1e-6)
                or np.max(np.linalg.norm(rotations[:, 6, :3], axis=-1)) > 1e-6):
            raise ValueError("expected the audited rigid +X 0.74m sword")
        fps_value = data["fps"]
        if isinstance(fps_value, (bool, str, bytes)) or not np.isscalar(fps_value):
            raise ValueError("invalid scalar frame rate")
        fps = float(fps_value)
        if not math.isfinite(fps) or not 29.0 <= fps <= 31.0:
            raise ValueError("expected the approximately 30Hz ASE motion rate")
        time = np.arange(frames, dtype=np.float64) / fps
        if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
            raise ValueError("invalid motion timestamps")
        local_matrices = _xyzw_to_matrix(rotations)
        global_matrices = np.empty_like(local_matrices)
        positions = np.empty((frames, 17, 3), dtype=np.float64)
        for joint, parent in enumerate(_PARENTS):
            if parent < 0:
                positions[:, joint] = root
                global_matrices[:, joint] = local_matrices[:, joint]
            else:
                parent_matrices = global_matrices[:, parent]
                positions[:, joint] = positions[:, parent] + np.einsum(
                    "nij,j->ni", parent_matrices, offsets[joint]
                )
                global_matrices[:, joint] = parent_matrices @ local_matrices[:, joint]
        grip, tip = positions[:, 5], positions[:, 6]
        displacement = tip - grip
        lengths = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(positions).all() or not np.allclose(lengths, .74, rtol=0, atol=1e-6):
            raise ValueError("nonfinite FK or inconsistent rigid sword length")
        direction = displacement / lengths[:, None]
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("unsupported or malformed ASE motion structure") from error
    return {
        "time": time, "grip_m": grip.copy(), "direction": direction,
        "shoulder_l_m": positions[:, 7].copy(),
        "shoulder_r_m": positions[:, 3].copy(),
        "root_m": root.astype(np.float64, copy=True), "source_tip_m": tip.copy(),
        "frames": frames, "fps": fps,
        "provenance": {
            "repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT,
            "motion_directory": UPSTREAM_MOTION_DIRECTORY,
            "local_filename": source.name, "sha256": digest, "bytes": expected_bytes,
            "decoder": "restricted-data-opcode-interpreter; no upstream execution",
            "grip_definition": "right_hand skeletal origin; not a measured clamp",
            "units": "meters; official MJCF-compatible skeleton offsets",
            "motion_license": "strictly noncommercial; check upstream terms",
        },
    }
