"""Shared math, scan, and body shaping helpers"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .reader import BoneTransform, G1MModel, Quaternion, Vector3


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def wrap_angle_degrees(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    return 0.0 if abs(wrapped) <= 1e-6 else wrapped


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    length = math.sqrt(
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + quaternion.w * quaternion.w
    )
    if length <= 1e-8:
        return Quaternion(0.0, 0.0, 0.0, 1.0)
    return Quaternion(
        quaternion.x / length,
        quaternion.y / length,
        quaternion.z / length,
        quaternion.w / length,
    )


def quaternion_from_euler_degrees(pitch: float, yaw: float, roll: float) -> Quaternion:
    """Build a quaternion from XYZ Euler degrees

    Pitch maps to X, yaw maps to Y, and roll maps to Z
    """

    pitch_radians = math.radians(pitch) * 0.5
    yaw_radians = math.radians(yaw) * 0.5
    roll_radians = math.radians(roll) * 0.5

    sx, cx = math.sin(pitch_radians), math.cos(pitch_radians)
    sy, cy = math.sin(yaw_radians), math.cos(yaw_radians)
    sz, cz = math.sin(roll_radians), math.cos(roll_radians)

    return normalize_quaternion(
        Quaternion(
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz,
        )
    )


def euler_degrees_from_quaternion(quaternion: Quaternion) -> tuple[float, float, float]:
    """Convert a quaternion to XYZ Euler degrees

    Returns pitch(X), yaw(Y), roll(Z)
    """

    normalized = normalize_quaternion(quaternion)
    x = normalized.x
    y = normalized.y
    z = normalized.z
    w = normalized.w

    m11 = 1.0 - 2.0 * (y * y + z * z)
    m12 = 2.0 * (x * y - z * w)
    m13 = 2.0 * (x * z + y * w)
    m22 = 1.0 - 2.0 * (x * x + z * z)
    m23 = 2.0 * (y * z - x * w)
    m32 = 2.0 * (y * z + x * w)
    m33 = 1.0 - 2.0 * (x * x + y * y)

    yaw = math.asin(clamp(m13, -1.0, 1.0))
    if abs(m13) < 0.9999999:
        pitch = math.atan2(-m23, m33)
        roll = math.atan2(-m12, m11)
    else:
        pitch = math.atan2(m32, m22)
        roll = 0.0

    return (
        wrap_angle_degrees(math.degrees(pitch)),
        wrap_angle_degrees(math.degrees(yaw)),
        wrap_angle_degrees(math.degrees(roll)),
    )


@dataclass(frozen=True, slots=True)
class BodyShapeControls:
    bust_volume: float = 0.0
    bust_lift: float = 0.0
    bust_width: float = 0.0
    bust_separation: float = 0.0
    upper_chest: float = 0.0
    waist: float = 0.0
    torso_taper: float = 0.0
    hip_width: float = 0.0
    seat_projection: float = 0.0
    shoulder_width: float = 0.0
    limb_thickness: float = 0.0
    arm_length: float = 0.0
    leg_length: float = 0.0
    posture: float = 0.0

    def is_neutral(self, *, tolerance: float = 1e-6) -> bool:
        return all(
            abs(value) <= tolerance
            for value in (
                self.bust_volume,
                self.bust_lift,
                self.bust_width,
                self.bust_separation,
                self.upper_chest,
                self.waist,
                self.torso_taper,
                self.hip_width,
                self.seat_projection,
                self.shoulder_width,
                self.limb_thickness,
                self.arm_length,
                self.leg_length,
                self.posture,
            )
        )


@dataclass(frozen=True, slots=True)
class BodyShapeProfile:
    height: float
    half_width: float
    depth: float
    torso_center: tuple[int, ...]
    upper_chest: tuple[int, ...]
    waist_center: tuple[int, ...]
    hip_center: tuple[int, ...]
    bust_left: tuple[int, ...]
    bust_right: tuple[int, ...]
    hip_left: tuple[int, ...]
    hip_right: tuple[int, ...]
    shoulder_left: tuple[int, ...]
    shoulder_right: tuple[int, ...]
    upper_arm_left: tuple[int, ...]
    upper_arm_right: tuple[int, ...]
    forearm_left: tuple[int, ...]
    forearm_right: tuple[int, ...]
    thigh_left: tuple[int, ...]
    thigh_right: tuple[int, ...]
    calf_left: tuple[int, ...]
    calf_right: tuple[int, ...]

    @property
    def all_bone_ids(self) -> tuple[int, ...]:
        ordered: list[int] = []
        for group in (
            self.torso_center,
            self.upper_chest,
            self.waist_center,
            self.hip_center,
            self.bust_left,
            self.bust_right,
            self.hip_left,
            self.hip_right,
            self.shoulder_left,
            self.shoulder_right,
            self.upper_arm_left,
            self.upper_arm_right,
            self.forearm_left,
            self.forearm_right,
            self.thigh_left,
            self.thigh_right,
            self.calf_left,
            self.calf_right,
        ):
            for bone_id in group:
                if bone_id not in ordered:
                    ordered.append(bone_id)
        return tuple(ordered)

    @property
    def has_shape_targets(self) -> bool:
        return bool(self.all_bone_ids)


def build_global_matrices(model: G1MModel, *, current: bool) -> list[list[float]]:
    matrices: list[list[float] | None] = [None] * len(model.bones)

    def resolve(bone_index: int) -> list[float]:
        cached = matrices[bone_index]
        if cached is not None:
            return cached

        bone = model.bones[bone_index]
        transform = bone.current_transform if current else bone.original_transform
        local_matrix = compose_local_matrix(transform)
        if 0 <= bone.parent < len(model.bones):
            cached = multiply_matrix(resolve(bone.parent), local_matrix)
        else:
            cached = local_matrix
        matrices[bone_index] = cached
        return cached

    return [resolve(index) for index in range(len(model.bones))]


def build_body_shape_profile(model: G1MModel) -> BodyShapeProfile:
    if not model.bones:
        return BodyShapeProfile(
            height=0.0,
            half_width=0.0,
            depth=0.0,
            torso_center=(),
            upper_chest=(),
            waist_center=(),
            hip_center=(),
            bust_left=(),
            bust_right=(),
            hip_left=(),
            hip_right=(),
            shoulder_left=(),
            shoulder_right=(),
            upper_arm_left=(),
            upper_arm_right=(),
            forearm_left=(),
            forearm_right=(),
            thigh_left=(),
            thigh_right=(),
            calf_left=(),
            calf_right=(),
        )

    matrices = build_global_matrices(model, current=False)
    positions = {
        bone_index: Vector3(matrix[3], matrix[7], matrix[11])
        for bone_index, matrix in enumerate(matrices)
    }
    xs = [position.x for position in positions.values()]
    ys = [position.y for position in positions.values()]
    zs = [position.z for position in positions.values()]
    max_y = max(ys) if ys else 0.0
    min_z = min(zs) if zs else 0.0
    max_z = max(zs) if zs else 0.0
    height = max(max_y, 1.0)
    half_width = max(max(abs(min(xs, default=0.0)), abs(max(xs, default=0.0))), 1.0)
    depth = max(max_z - min_z, 1.0)
    center_threshold = max(half_width * 0.055, 3.0)
    torso_side_min = center_threshold * 1.15
    torso_side_max = max(half_width * 0.32, torso_side_min + 2.0)
    side_hip_max = max(half_width * 0.24, torso_side_min + 2.0)
    front_bias = min_z + depth * 0.55
    children: dict[int, list[int]] = {bone.index: [] for bone in model.bones}
    for bone in model.bones:
        if 0 <= bone.parent < len(model.bones):
            children[bone.parent].append(bone.index)

    def is_center(bone_id: int) -> bool:
        return abs(positions[bone_id].x) <= center_threshold

    def is_helper_cluster_root(bone_id: int) -> bool:
        child_ids = children.get(bone_id, [])
        if len(child_ids) < 8:
            return False
        leaf_children = sum(1 for child_id in child_ids if not children.get(child_id))
        return leaf_children >= int(len(child_ids) * 0.75)

    def position_distance(left_id: int, right_id: int) -> float:
        left = positions[left_id]
        right = positions[right_id]
        return math.sqrt(
            (left.x - right.x) * (left.x - right.x)
            + (left.y - right.y) * (left.y - right.y)
            + (left.z - right.z) * (left.z - right.z)
        )

    def select(
        *,
        side: str | None = None,
        min_y_ratio: float = 0.0,
        max_y_ratio: float = 1.0,
        min_abs_x: float = 0.0,
        max_abs_x: float | None = None,
        max_abs_z: float | None = None,
        min_z: float | None = None,
        max_z: float | None = None,
        center_only: bool = False,
    ) -> list[int]:
        selected: list[int] = []
        for bone_id, position in positions.items():
            if is_helper_cluster_root(bone_id):
                continue
            y_ratio = position.y / height if height > 0.0 else 0.0
            if not min_y_ratio <= y_ratio <= max_y_ratio:
                continue
            abs_x = abs(position.x)
            if abs_x < min_abs_x:
                continue
            if max_abs_x is not None and abs_x > max_abs_x:
                continue
            if max_abs_z is not None and abs(position.z) > max_abs_z:
                continue
            if center_only and not is_center(bone_id):
                continue
            if side == "left" and position.x <= center_threshold:
                continue
            if side == "right" and position.x >= -center_threshold:
                continue
            if min_z is not None and position.z < min_z:
                continue
            if max_z is not None and position.z > max_z:
                continue
            selected.append(bone_id)
        return selected

    def dedupe(indices: Iterable[int], *, limit: int | None = None, key=None) -> tuple[int, ...]:
        unique: list[int] = []
        seen: set[int] = set()
        ordered = list(indices)
        if key is not None:
            ordered.sort(key=key)
        for bone_id in ordered:
            if bone_id in seen:
                continue
            unique.append(bone_id)
            seen.add(bone_id)
            if limit is not None and len(unique) >= limit:
                break
        return tuple(unique)

    def side_by_x(candidates: list[int], *, side: str, inner_ratio: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
        side_candidates = [bone_id for bone_id in candidates if (positions[bone_id].x > 0 if side == "left" else positions[bone_id].x < 0)]
        if not side_candidates:
            return (), ()
        ordered = sorted(side_candidates, key=lambda bone_id: abs(positions[bone_id].x))
        pivot = max(1, int(math.ceil(len(ordered) * inner_ratio)))
        return dedupe(ordered[:pivot]), dedupe(ordered[pivot:])

    def side_by_y(candidates: list[int], *, side: str, upper_ratio: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
        side_candidates = [bone_id for bone_id in candidates if (positions[bone_id].x > 0 if side == "left" else positions[bone_id].x < 0)]
        if not side_candidates:
            return (), ()
        ordered = sorted(side_candidates, key=lambda bone_id: positions[bone_id].y, reverse=True)
        pivot = max(1, int(math.ceil(len(ordered) * upper_ratio)))
        return dedupe(ordered[:pivot]), dedupe(ordered[pivot:])

    def side_matches(bone_id: int, side: str, *, min_abs_x: float = 0.0) -> bool:
        position = positions[bone_id]
        if abs(position.x) < min_abs_x:
            return False
        return position.x > center_threshold if side == "left" else position.x < -center_threshold

    def helper_matches(base_ids: Iterable[int], used_main_ids: set[int], side: str) -> tuple[int, ...]:
        base_list = tuple(base_ids)
        if not base_list:
            return ()
        tolerance = max(height * 0.0125, 0.5)
        helpers: list[int] = []
        for bone_id in positions:
            if bone_id in used_main_ids or children.get(bone_id):
                continue
            if not side_matches(bone_id, side, min_abs_x=center_threshold):
                continue
            parent = model.bones[bone_id].parent
            if parent in used_main_ids:
                continue
            if any(position_distance(bone_id, base_id) <= tolerance for base_id in base_list):
                helpers.append(bone_id)
        return dedupe(helpers, key=lambda bone_id: positions[bone_id].y)

    def side_cluster_helpers(reference_ids: Iterable[int], used_main_ids: set[int], side: str) -> tuple[int, ...]:
        references = tuple(reference_ids)
        if not references:
            return ()
        reference_y = sum(positions[bone_id].y for bone_id in references) / len(references)
        helpers: list[int] = []
        for bone_id in positions:
            if bone_id in used_main_ids or children.get(bone_id):
                continue
            if not side_matches(bone_id, side, min_abs_x=center_threshold):
                continue
            parent = model.bones[bone_id].parent
            if not (0 <= parent < len(model.bones) and is_helper_cluster_root(parent)):
                continue
            if abs(positions[bone_id].y - reference_y) > height * 0.08:
                continue
            if abs(positions[bone_id].x) > max(half_width * 0.38, center_threshold + 2.0):
                continue
            helpers.append(bone_id)
        return dedupe(helpers, key=lambda bone_id: (positions[bone_id].y, abs(positions[bone_id].z)))

    def primary_arm_chain(side: str) -> tuple[int, ...]:
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for bone_id, position in positions.items():
            y_ratio = position.y / height if height > 0.0 else 0.0
            if not (0.52 <= y_ratio <= 0.90):
                continue
            if not side_matches(bone_id, side, min_abs_x=center_threshold):
                continue
            parent = model.bones[bone_id].parent
            if 0 <= parent < len(model.bones) and abs(positions[parent].x) > abs(position.x):
                continue

            path = [bone_id]
            cursor = bone_id
            while len(path) < 5:
                if len(children.get(cursor, ())) > 2 and len(path) >= 3:
                    break
                cursor_position = positions[cursor]
                child_candidates = []
                for child_id in children.get(cursor, []):
                    child_position = positions[child_id]
                    child_y_ratio = child_position.y / height if height > 0.0 else 0.0
                    if not (0.45 <= child_y_ratio <= 0.96):
                        continue
                    if not side_matches(child_id, side, min_abs_x=center_threshold):
                        continue
                    if abs(child_position.x) + center_threshold * 0.15 < abs(cursor_position.x):
                        continue
                    if abs(child_position.y - cursor_position.y) > height * 0.20:
                        continue
                    child_candidates.append(child_id)
                if not child_candidates:
                    break
                cursor = max(child_candidates, key=lambda child_id: abs(positions[child_id].x))
                if cursor in path:
                    break
                path.append(cursor)

            if len(path) < 3:
                continue
            x_span = abs(positions[path[-1]].x) - abs(positions[path[0]].x)
            if x_span <= half_width * 0.08:
                continue
            score = x_span - abs(positions[path[0]].x) * 0.12 + len(path)
            candidates.append((score, tuple(path)))

        if not candidates:
            return ()
        return max(candidates, key=lambda item: item[0])[1]

    def primary_leg_chain(side: str) -> tuple[int, ...]:
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for bone_id, position in positions.items():
            y_ratio = position.y / height if height > 0.0 else 0.0
            if not (0.30 <= y_ratio <= 0.74):
                continue
            if not side_matches(bone_id, side, min_abs_x=center_threshold * 0.65):
                continue
            if abs(position.x) > max(half_width * 0.34, center_threshold + 2.0):
                continue

            path = [bone_id]
            cursor = bone_id
            while len(path) < 4:
                cursor_position = positions[cursor]
                child_candidates = []
                for child_id in children.get(cursor, []):
                    child_position = positions[child_id]
                    if not side_matches(child_id, side, min_abs_x=center_threshold * 0.50):
                        continue
                    if child_position.y >= cursor_position.y - height * 0.05:
                        continue
                    if abs(child_position.x - cursor_position.x) > max(half_width * 0.18, center_threshold + 2.0):
                        continue
                    child_candidates.append(child_id)
                if not child_candidates:
                    break
                cursor = min(child_candidates, key=lambda child_id: positions[child_id].y)
                if cursor in path:
                    break
                path.append(cursor)

            if len(path) < 3:
                continue
            vertical_drop = positions[path[0]].y - positions[path[-1]].y
            if vertical_drop <= height * 0.25:
                continue
            score = vertical_drop - abs(positions[path[0]].x) * 0.25 + len(path)
            candidates.append((score, tuple(path)))

        if not candidates:
            return ()
        return max(candidates, key=lambda item: item[0])[1]

    torso_center = dedupe(
        select(center_only=True, min_y_ratio=0.50, max_y_ratio=0.82, max_abs_z=depth * 0.28),
        key=lambda bone_id: positions[bone_id].y,
    )
    upper_chest = dedupe(
        select(center_only=True, min_y_ratio=0.62, max_y_ratio=0.84, max_abs_z=depth * 0.28),
        key=lambda bone_id: positions[bone_id].y,
    )
    waist_center = dedupe(
        select(center_only=True, min_y_ratio=0.48, max_y_ratio=0.68, max_abs_z=depth * 0.28),
        key=lambda bone_id: positions[bone_id].y,
    )
    hip_center = dedupe(
        select(center_only=True, min_y_ratio=0.36, max_y_ratio=0.58, max_abs_z=depth * 0.35),
        key=lambda bone_id: positions[bone_id].y,
    )
    if not waist_center:
        waist_center = dedupe(
            bone_id for bone_id in torso_center if positions[bone_id].y <= height * 0.72
        )
    if not hip_center:
        hip_center = dedupe(
            bone_id for bone_id in waist_center if positions[bone_id].y <= height * 0.60
        )

    bust_left = dedupe(
        select(
            side="left",
            min_y_ratio=0.64,
            max_y_ratio=0.86,
            min_abs_x=torso_side_min,
            max_abs_x=torso_side_max,
            min_z=front_bias,
        ),
        key=lambda bone_id: (positions[bone_id].y, -abs(positions[bone_id].x)),
    )
    bust_right = dedupe(
        select(
            side="right",
            min_y_ratio=0.64,
            max_y_ratio=0.86,
            min_abs_x=torso_side_min,
            max_abs_x=torso_side_max,
            min_z=front_bias,
        ),
        key=lambda bone_id: (positions[bone_id].y, -abs(positions[bone_id].x)),
    )
    if not bust_left:
        bust_left = dedupe(
            select(
                side="left",
                min_y_ratio=0.64,
                max_y_ratio=0.86,
                min_abs_x=torso_side_min,
                max_abs_x=torso_side_max,
            ),
            key=lambda bone_id: (positions[bone_id].y, abs(positions[bone_id].x)),
        )
    if not bust_right:
        bust_right = dedupe(
            select(
                side="right",
                min_y_ratio=0.64,
                max_y_ratio=0.86,
                min_abs_x=torso_side_min,
                max_abs_x=torso_side_max,
            ),
            key=lambda bone_id: (positions[bone_id].y, abs(positions[bone_id].x)),
        )

    hip_left = dedupe(
        select(
            side="left",
            min_y_ratio=0.34,
            max_y_ratio=0.62,
            min_abs_x=torso_side_min,
            max_abs_x=side_hip_max,
        ),
        key=lambda bone_id: positions[bone_id].y,
    )
    hip_right = dedupe(
        select(
            side="right",
            min_y_ratio=0.34,
            max_y_ratio=0.62,
            min_abs_x=torso_side_min,
            max_abs_x=side_hip_max,
        ),
        key=lambda bone_id: positions[bone_id].y,
    )

    shoulder_candidates = select(
        min_y_ratio=0.62,
        max_y_ratio=0.86,
        min_abs_x=torso_side_max * 0.85,
        max_abs_x=half_width * 0.48,
    )
    shoulder_left, upper_arm_left = side_by_x(shoulder_candidates, side="left", inner_ratio=0.45)
    shoulder_right, upper_arm_right = side_by_x(shoulder_candidates, side="right", inner_ratio=0.45)

    arm_candidates = select(
        min_y_ratio=0.60,
        max_y_ratio=0.95,
        min_abs_x=half_width * 0.35,
    )
    upper_left_arm, fore_left_arm = side_by_x(arm_candidates, side="left", inner_ratio=0.45)
    upper_right_arm, fore_right_arm = side_by_x(arm_candidates, side="right", inner_ratio=0.45)
    upper_arm_left = dedupe((*upper_arm_left, *upper_left_arm))
    upper_arm_right = dedupe((*upper_arm_right, *upper_right_arm))
    forearm_left = dedupe(fore_left_arm)
    forearm_right = dedupe(fore_right_arm)
    if not shoulder_left:
        shoulder_left = dedupe(upper_arm_left[: max(1, len(upper_arm_left) // 4 or 1)])
    if not shoulder_right:
        shoulder_right = dedupe(upper_arm_right[: max(1, len(upper_arm_right) // 4 or 1)])

    leg_candidates = select(
        min_y_ratio=0.02,
        max_y_ratio=0.68,
        min_abs_x=center_threshold * 0.75,
        max_abs_x=max(half_width * 0.24, center_threshold + 2.0),
    )
    thigh_left, calf_left = side_by_y(leg_candidates, side="left", upper_ratio=0.5)
    thigh_right, calf_right = side_by_y(leg_candidates, side="right", upper_ratio=0.5)

    left_arm_chain = primary_arm_chain("left")
    right_arm_chain = primary_arm_chain("right")
    if left_arm_chain:
        used = set(left_arm_chain)
        if len(left_arm_chain) >= 4:
            shoulder_core = left_arm_chain[:2]
            upper_core = left_arm_chain[2:3]
            forearm_core = left_arm_chain[3:4]
        else:
            shoulder_core = left_arm_chain[:1]
            upper_core = left_arm_chain[1:2]
            forearm_core = left_arm_chain[2:3]
        shoulder_left = dedupe((*shoulder_core, *helper_matches(shoulder_core, used, "left")))
        upper_arm_left = dedupe((*upper_core, *helper_matches(upper_core, used, "left")))
        forearm_left = dedupe(forearm_core)
    if right_arm_chain:
        used = set(right_arm_chain)
        if len(right_arm_chain) >= 4:
            shoulder_core = right_arm_chain[:2]
            upper_core = right_arm_chain[2:3]
            forearm_core = right_arm_chain[3:4]
        else:
            shoulder_core = right_arm_chain[:1]
            upper_core = right_arm_chain[1:2]
            forearm_core = right_arm_chain[2:3]
        shoulder_right = dedupe((*shoulder_core, *helper_matches(shoulder_core, used, "right")))
        upper_arm_right = dedupe((*upper_core, *helper_matches(upper_core, used, "right")))
        forearm_right = dedupe(forearm_core)

    left_leg_chain = primary_leg_chain("left")
    right_leg_chain = primary_leg_chain("right")
    if left_leg_chain:
        used = set(left_leg_chain)
        thigh_core = left_leg_chain[:1]
        calf_core = left_leg_chain[1:2]
        hip_left = dedupe((*thigh_core, *side_cluster_helpers(thigh_core, used, "left")))
        thigh_left = dedupe((*thigh_core, *helper_matches(thigh_core, used, "left")))
        calf_left = dedupe((*calf_core, *helper_matches(calf_core, used, "left")))
    if right_leg_chain:
        used = set(right_leg_chain)
        thigh_core = right_leg_chain[:1]
        calf_core = right_leg_chain[1:2]
        hip_right = dedupe((*thigh_core, *side_cluster_helpers(thigh_core, used, "right")))
        thigh_right = dedupe((*thigh_core, *helper_matches(thigh_core, used, "right")))
        calf_right = dedupe((*calf_core, *helper_matches(calf_core, used, "right")))
    if left_leg_chain and right_leg_chain:
        left_parent = model.bones[left_leg_chain[0]].parent
        right_parent = model.bones[right_leg_chain[0]].parent
        if left_parent == right_parent and 0 <= left_parent < len(model.bones):
            hip_center = dedupe((left_parent, *hip_center))

    return BodyShapeProfile(
        height=height,
        half_width=half_width,
        depth=depth,
        torso_center=torso_center,
        upper_chest=upper_chest,
        waist_center=waist_center,
        hip_center=hip_center,
        bust_left=bust_left,
        bust_right=bust_right,
        hip_left=hip_left,
        hip_right=hip_right,
        shoulder_left=shoulder_left,
        shoulder_right=shoulder_right,
        upper_arm_left=upper_arm_left,
        upper_arm_right=upper_arm_right,
        forearm_left=forearm_left,
        forearm_right=forearm_right,
        thigh_left=thigh_left,
        thigh_right=thigh_right,
        calf_left=calf_left,
        calf_right=calf_right,
    )


def apply_body_shape_controls(
    model: G1MModel,
    overrides: dict[int, BoneTransform],
    profile: BodyShapeProfile,
    controls: BodyShapeControls,
) -> dict[int, BoneTransform]:
    if controls.is_neutral() or not profile.has_shape_targets:
        return overrides

    width_unit = profile.half_width * 0.035
    height_unit = profile.height * 0.025
    depth_unit = profile.depth * 0.055
    base_global_matrices = build_global_matrices(model, current=True)
    inverse_parent_matrices: dict[int, list[float]] = {}

    def ensure(bone_id: int) -> BoneTransform:
        transform = overrides.get(bone_id)
        if transform is None:
            transform = model.bones[bone_id].current_transform.copy()
            overrides[bone_id] = transform
        return transform

    def scale_group(bone_ids: Iterable[int], *, x: float = 1.0, y: float = 1.0, z: float = 1.0) -> None:
        for bone_id in bone_ids:
            transform = ensure(bone_id)
            transform.scale = Vector3(
                clamp(transform.scale.x * x, 0.05, 8.0),
                clamp(transform.scale.y * y, 0.05, 8.0),
                clamp(transform.scale.z * z, 0.05, 8.0),
            )

    length_axis_cache: dict[int, str] = {}

    def dominant_length_axis(bone_id: int) -> str:
        cached = length_axis_cache.get(bone_id)
        if cached is not None:
            return cached

        child_offsets = [
            model.bones[child_id].original_transform.position
            for child_id, bone in enumerate(model.bones)
            if bone.parent == bone_id
        ]
        if child_offsets:
            vector = max(
                child_offsets,
                key=lambda offset: offset.x * offset.x + offset.y * offset.y + offset.z * offset.z,
            )
        else:
            vector = model.bones[bone_id].original_transform.position

        components = {"x": abs(vector.x), "y": abs(vector.y), "z": abs(vector.z)}
        axis = max(components, key=components.get)
        if components[axis] <= 1e-5:
            axis = "x"
        length_axis_cache[bone_id] = axis
        return axis

    def scale_limb_group(
        bone_ids: Iterable[int],
        *,
        length: float = 1.0,
        thickness: float = 1.0,
    ) -> None:
        for bone_id in bone_ids:
            axis = dominant_length_axis(bone_id)
            factors = {"x": thickness, "y": thickness, "z": thickness}
            factors[axis] = length
            scale_group(
                (bone_id,),
                x=factors["x"],
                y=factors["y"],
                z=factors["z"],
            )

    def move_group(
        bone_ids: Iterable[int],
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        for bone_id in bone_ids:
            transform = ensure(bone_id)
            transform.position = transform.position.add(Vector3(x, y, z))

    def move_group_model(
        bone_ids: Iterable[int],
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        world_delta = Vector3(x, y, z)
        for bone_id in bone_ids:
            transform = ensure(bone_id)
            local_delta = world_delta_to_local_translation(
                model,
                bone_id,
                world_delta,
                base_global_matrices,
                inverse_parent_matrices,
            )
            transform.position = transform.position.add(local_delta)

    def move_side(
        left_ids: Iterable[int],
        right_ids: Iterable[int],
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        move_group(left_ids, x=x, y=y, z=z)
        move_group(right_ids, x=-x, y=y, z=z)

    def move_side_model(
        left_ids: Iterable[int],
        right_ids: Iterable[int],
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        move_group_model(left_ids, x=x, y=y, z=z)
        move_group_model(right_ids, x=-x, y=y, z=z)

    def rotate_group(
        bone_ids: Iterable[int],
        *,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        for bone_id in bone_ids:
            transform = ensure(bone_id)
            current_pitch, current_yaw, current_roll = euler_degrees_from_quaternion(transform.rotation)
            transform.rotation = quaternion_from_euler_degrees(
                current_pitch + pitch,
                current_yaw + yaw,
                current_roll + roll,
            )

    bust_scale_x = 1.0 + controls.bust_volume * 0.18
    bust_scale_y = 1.0 + controls.bust_volume * 0.08
    bust_scale_z = 1.0 + controls.bust_volume * 0.22
    scale_group(profile.bust_left, x=bust_scale_x, y=bust_scale_y, z=bust_scale_z)
    scale_group(profile.bust_right, x=bust_scale_x, y=bust_scale_y, z=bust_scale_z)
    move_group_model(profile.bust_left, y=controls.bust_lift * height_unit, z=controls.bust_volume * depth_unit * 0.35)
    move_group_model(profile.bust_right, y=controls.bust_lift * height_unit, z=controls.bust_volume * depth_unit * 0.35)
    move_side_model(
        profile.bust_left,
        profile.bust_right,
        x=(controls.bust_width + controls.bust_separation * 1.15) * width_unit * 0.55,
    )
    scale_group(
        profile.bust_left,
        x=1.0 + controls.bust_width * 0.12,
        z=1.0 + controls.bust_width * 0.06,
    )
    scale_group(
        profile.bust_right,
        x=1.0 + controls.bust_width * 0.12,
        z=1.0 + controls.bust_width * 0.06,
    )

    scale_group(
        profile.upper_chest,
        x=1.0 + controls.upper_chest * 0.12 + controls.torso_taper * 0.08,
        y=1.0 + controls.upper_chest * 0.04,
        z=1.0 + controls.upper_chest * 0.10 + controls.torso_taper * 0.05,
    )
    move_group_model(profile.upper_chest, z=controls.upper_chest * depth_unit * 0.20)

    waist_scale = 1.0 + controls.waist * 0.18 - controls.torso_taper * 0.12
    scale_group(profile.waist_center, x=waist_scale, z=1.0 + controls.waist * 0.12 - controls.torso_taper * 0.08)

    move_side_model(profile.hip_left, profile.hip_right, x=controls.hip_width * width_unit * 0.7)
    scale_group(profile.hip_center, x=1.0 + controls.hip_width * 0.10, z=1.0 + controls.seat_projection * 0.06)
    scale_group(profile.hip_left, x=1.0 + controls.hip_width * 0.10, z=1.0 + controls.seat_projection * 0.08)
    scale_group(profile.hip_right, x=1.0 + controls.hip_width * 0.10, z=1.0 + controls.seat_projection * 0.08)
    move_group_model(profile.hip_center, z=-controls.seat_projection * depth_unit * 0.45)
    move_group_model(profile.hip_left, z=-controls.seat_projection * depth_unit * 0.45)
    move_group_model(profile.hip_right, z=-controls.seat_projection * depth_unit * 0.45)

    move_side_model(profile.shoulder_left, profile.shoulder_right, x=controls.shoulder_width * width_unit * 0.65)
    move_side_model(profile.upper_arm_left, profile.upper_arm_right, x=controls.shoulder_width * width_unit * 0.35)

    limb_scale_x = 1.0 + controls.limb_thickness * 0.14
    limb_scale_z = 1.0 + controls.limb_thickness * 0.16
    for group in (
        profile.upper_arm_left,
        profile.upper_arm_right,
        profile.forearm_left,
        profile.forearm_right,
        profile.thigh_left,
        profile.thigh_right,
        profile.calf_left,
        profile.calf_right,
    ):
        scale_limb_group(group, thickness=(limb_scale_x + limb_scale_z) * 0.5)

    arm_length_scale = 1.0 + controls.arm_length * 0.10
    scale_limb_group(profile.upper_arm_left, length=arm_length_scale)
    scale_limb_group(profile.upper_arm_right, length=arm_length_scale)
    scale_limb_group(profile.forearm_left, length=1.0 + controls.arm_length * 0.08)
    scale_limb_group(profile.forearm_right, length=1.0 + controls.arm_length * 0.08)

    leg_length_scale = 1.0 + controls.leg_length * 0.12
    scale_limb_group(profile.thigh_left, length=leg_length_scale)
    scale_limb_group(profile.thigh_right, length=leg_length_scale)
    scale_limb_group(profile.calf_left, length=1.0 + controls.leg_length * 0.10)
    scale_limb_group(profile.calf_right, length=1.0 + controls.leg_length * 0.10)

    rotate_group(profile.hip_center, pitch=controls.posture * 3.5)
    rotate_group(profile.torso_center, pitch=-controls.posture * 4.5)
    rotate_group(profile.upper_chest, pitch=-controls.posture * 7.0)

    return overrides


def compose_local_matrix(transform: BoneTransform) -> list[float]:
    quat = normalize_quaternion(transform.rotation)
    xx = quat.x * quat.x
    yy = quat.y * quat.y
    zz = quat.z * quat.z
    xy = quat.x * quat.y
    xz = quat.x * quat.z
    yz = quat.y * quat.z
    wx = quat.w * quat.x
    wy = quat.w * quat.y
    wz = quat.w * quat.z

    return [
        (1.0 - 2.0 * (yy + zz)) * transform.scale.x,
        (2.0 * (xy - wz)) * transform.scale.y,
        (2.0 * (xz + wy)) * transform.scale.z,
        transform.position.x,
        (2.0 * (xy + wz)) * transform.scale.x,
        (1.0 - 2.0 * (xx + zz)) * transform.scale.y,
        (2.0 * (yz - wx)) * transform.scale.z,
        transform.position.y,
        (2.0 * (xz - wy)) * transform.scale.x,
        (2.0 * (yz + wx)) * transform.scale.y,
        (1.0 - 2.0 * (xx + yy)) * transform.scale.z,
        transform.position.z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def multiply_matrix(left: list[float], right: list[float]) -> list[float]:
    return [
        left[0] * right[0] + left[1] * right[4] + left[2] * right[8] + left[3] * right[12],
        left[0] * right[1] + left[1] * right[5] + left[2] * right[9] + left[3] * right[13],
        left[0] * right[2] + left[1] * right[6] + left[2] * right[10] + left[3] * right[14],
        left[0] * right[3] + left[1] * right[7] + left[2] * right[11] + left[3] * right[15],
        left[4] * right[0] + left[5] * right[4] + left[6] * right[8] + left[7] * right[12],
        left[4] * right[1] + left[5] * right[5] + left[6] * right[9] + left[7] * right[13],
        left[4] * right[2] + left[5] * right[6] + left[6] * right[10] + left[7] * right[14],
        left[4] * right[3] + left[5] * right[7] + left[6] * right[11] + left[7] * right[15],
        left[8] * right[0] + left[9] * right[4] + left[10] * right[8] + left[11] * right[12],
        left[8] * right[1] + left[9] * right[5] + left[10] * right[9] + left[11] * right[13],
        left[8] * right[2] + left[9] * right[6] + left[10] * right[10] + left[11] * right[14],
        left[8] * right[3] + left[9] * right[7] + left[10] * right[11] + left[11] * right[15],
        left[12] * right[0] + left[13] * right[4] + left[14] * right[8] + left[15] * right[12],
        left[12] * right[1] + left[13] * right[5] + left[14] * right[9] + left[15] * right[13],
        left[12] * right[2] + left[13] * right[6] + left[14] * right[10] + left[15] * right[14],
        left[12] * right[3] + left[13] * right[7] + left[14] * right[11] + left[15] * right[15],
    ]


def invert_affine_matrix(matrix: list[float]) -> list[float]:
    a00, a01, a02 = matrix[0], matrix[1], matrix[2]
    a10, a11, a12 = matrix[4], matrix[5], matrix[6]
    a20, a21, a22 = matrix[8], matrix[9], matrix[10]
    tx, ty, tz = matrix[3], matrix[7], matrix[11]

    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if abs(det) <= 1e-8:
        return [
            1.0, 0.0, 0.0, -tx,
            0.0, 1.0, 0.0, -ty,
            0.0, 0.0, 1.0, -tz,
            0.0, 0.0, 0.0, 1.0,
        ]

    inv_det = 1.0 / det
    i00 = (a11 * a22 - a12 * a21) * inv_det
    i01 = (a02 * a21 - a01 * a22) * inv_det
    i02 = (a01 * a12 - a02 * a11) * inv_det
    i10 = (a12 * a20 - a10 * a22) * inv_det
    i11 = (a00 * a22 - a02 * a20) * inv_det
    i12 = (a02 * a10 - a00 * a12) * inv_det
    i20 = (a10 * a21 - a11 * a20) * inv_det
    i21 = (a01 * a20 - a00 * a21) * inv_det
    i22 = (a00 * a11 - a01 * a10) * inv_det

    return [
        i00, i01, i02, -(i00 * tx + i01 * ty + i02 * tz),
        i10, i11, i12, -(i10 * tx + i11 * ty + i12 * tz),
        i20, i21, i22, -(i20 * tx + i21 * ty + i22 * tz),
        0.0, 0.0, 0.0, 1.0,
    ]


def world_delta_to_local_translation(
    model: G1MModel,
    bone_id: int,
    world_delta: Vector3,
    base_global_matrices: list[list[float]],
    inverse_parent_matrices: dict[int, list[float]],
) -> Vector3:
    bone = model.bones[bone_id]
    if not 0 <= bone.parent < len(base_global_matrices):
        return world_delta.copy()

    inverse_parent = inverse_parent_matrices.get(bone.parent)
    if inverse_parent is None:
        inverse_parent = invert_affine_matrix(base_global_matrices[bone.parent])
        inverse_parent_matrices[bone.parent] = inverse_parent

    return Vector3(
        inverse_parent[0] * world_delta.x + inverse_parent[1] * world_delta.y + inverse_parent[2] * world_delta.z,
        inverse_parent[4] * world_delta.x + inverse_parent[5] * world_delta.y + inverse_parent[6] * world_delta.z,
        inverse_parent[8] * world_delta.x + inverse_parent[9] * world_delta.y + inverse_parent[10] * world_delta.z,
    )
