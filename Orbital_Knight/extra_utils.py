"""Experimental mesh region sculpting and vertex morph helpers for G1M models"""

from __future__ import annotations

from dataclasses import dataclass
import json, math
from pathlib import Path
from struct import Struct
from typing import Iterable

from .preview_scene import (
    FACE_TYPE_QUAD,
    FACE_TYPE_TRIANGLE,
    FACE_TYPE_TRIANGLE_STRIP,
    G1MG_SUBSECTION_HEADER_STRUCT,
    RESOURCE_HEADER_STRUCT,
    SEMANTIC_BONE_INDEX,
    SEMANTIC_BONE_WEIGHT,
    SEMANTIC_NORMAL,
    SEMANTIC_POSITION,
    SEMANTIC_TANGENT,
    SEMANTIC_UV,
    SUPPORTED_INDEX_TYPES,
    SUPPORTED_WEIGHT_TYPES,
    VERTEX_ATTRIBUTE_STRUCT,
    VERTEX_BUFFER_HEADER_STRUCT,
    VERTEX_TYPE_FLOAT2,
    VERTEX_TYPE_FLOAT3,
    VERTEX_TYPE_FLOAT4,
    VertexAttribute,
    VertexAttributeSet,
    read_attribute_components,
    resolve_rigid_bone_id,
    resolve_vertex_influences,
)
from .reader import G1MModel, G1MParseError, Submesh, Vector3
from .utils import build_global_matrices

FLOAT2 = Struct("<2f")
FLOAT3 = Struct("<3f")
FLOAT4 = Struct("<4f")


@dataclass(frozen=True, slots=True)
class RegionSculptControls:
    bone_id: int
    body_part_index: int | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    radius: float = 25.0
    min_weight: float = 0.10
    falloff_power: float = 1.0
    bone_filter: frozenset[int] | None = None

    def is_neutral(self, *, tolerance: float = 1e-6) -> bool:
        return (
            abs(self.offset[0]) <= tolerance
            and abs(self.offset[1]) <= tolerance
            and abs(self.offset[2]) <= tolerance
            and abs(self.scale[0] - 1.0) <= tolerance
            and abs(self.scale[1] - 1.0) <= tolerance
            and abs(self.scale[2] - 1.0) <= tolerance
        )


@dataclass(frozen=True, slots=True)
class VertexSculptControls:
    """Live vertex sculpt controls used by the Cleanrot UI

    Bones are used as vertex weight anchors and falloff centers

    sculpt_mode:
        scale_offset, scale/offset around weighted center
        inflate, move vertices along their normals by inflate_amount
        smooth, relax selected vertices toward mesh neighbor average
    """

    target_bones: frozenset[int]
    body_part_index: int | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    radius: float = 25.0
    min_weight: float = 0.10
    falloff_power: float = 1.0
    weld_vertices: bool = False
    weld_threshold: float = 1e-3
    sculpt_mode: str = "scale_offset"
    inflate_amount: float = 0.0
    smooth_strength: float = 0.0

    def is_neutral(self, *, tolerance: float = 1e-6) -> bool:
        if not self.target_bones:
            return True
        mode = (self.sculpt_mode or "scale_offset").lower()
        if mode == "inflate":
            return abs(self.inflate_amount) <= tolerance
        if mode == "smooth":
            return abs(self.smooth_strength) <= tolerance
        return (
            abs(self.offset[0]) <= tolerance
            and abs(self.offset[1]) <= tolerance
            and abs(self.offset[2]) <= tolerance
            and abs(self.scale[0] - 1.0) <= tolerance
            and abs(self.scale[1] - 1.0) <= tolerance
            and abs(self.scale[2] - 1.0) <= tolerance
        )


@dataclass(frozen=True, slots=True)
class RegionSculptResult:
    patched_vertex_count: int
    visited_vertex_count: int
    skipped_vertex_count: int
    affected_submesh_count: int
    unsupported_submesh_count: int


@dataclass(slots=True)
class _WritableVertexBuffer:
    index: int
    stride: int
    count: int
    raw_data: bytes
    data_offset: int


@dataclass(slots=True)
class _WritableIndexBuffer:
    index: int
    bit_width: int
    count: int
    indices: list[int]
    restart_index: int
    data_offset: int = 0


@dataclass
class VertexDelta:
    vbo_index: int
    vertex_index: int
    dx: float
    dy: float
    dz: float


@dataclass
class MorphTarget:
    name: str
    bone_hint: int | None
    source_file: str
    deltas: list[VertexDelta]


@dataclass
class MorphPreset:
    sliders: dict[str, float]


def apply_region_sculpt(model: G1MModel, controls: RegionSculptControls) -> RegionSculptResult:
    patched_bytes, result = sculpt_g1m_region_bytes(model, controls)
    if result.patched_vertex_count:
        model.raw_data = patched_bytes
    return result


def apply_vertex_sculpt(model: G1MModel, controls: VertexSculptControls) -> RegionSculptResult:
    patched_bytes, result = sculpt_g1m_vertex_bytes(model, controls)
    if result.patched_vertex_count:
        model.raw_data = patched_bytes
    return result


def sculpt_g1m_vertex_bytes(
    model: G1MModel,
    controls: VertexSculptControls,
) -> tuple[bytes, RegionSculptResult]:
    """Apply a vertex weight sculpt directly to G1MG positions

    Target bones are used as weight filters and spatial anchors
    """

    if controls.is_neutral():
        return model.raw_data, RegionSculptResult(0, 0, 0, 0, 0)

    target_bones = frozenset(int(bone_id) for bone_id in controls.target_bones)
    bad_bones = [bone_id for bone_id in target_bones if not 0 <= bone_id < len(model.bones)]
    if bad_bones:
        bad_text = ", ".join(str(bone_id) for bone_id in sorted(bad_bones))
        raise G1MParseError(f"Vertex sculpt target bone(s) outside the skeleton: {bad_text}.")

    vertex_buffers, attribute_sets, index_buffers = parse_writable_preview_resources(model)
    submesh_indices = target_submesh_indices(model, controls.body_part_index, None)
    if not submesh_indices:
        return model.raw_data, RegionSculptResult(0, 0, 0, 0, 0)

    original_globals = build_global_matrices(model, current=False)
    bone_centers = {
        bone_id: Vector3(original_globals[bone_id][3], original_globals[bone_id][7], original_globals[bone_id][11])
        for bone_id in target_bones
    }
    output = bytearray(model.raw_data)

    pending_positions: dict[tuple[int, int], tuple[_WritableVertexBuffer, VertexAttribute, Vector3, Vector3]] = {}
    visited_vertices: set[tuple[int, int]] = set()
    affected_submeshes: set[int] = set()
    skipped_count = 0
    unsupported_submeshes = 0

    for submesh_index in submesh_indices:
        if not 0 <= submesh_index < len(model.submeshes):
            unsupported_submeshes += 1
            continue
        submesh = model.submeshes[submesh_index]
        if not 0 <= submesh.vbo_index < len(attribute_sets):
            unsupported_submeshes += 1
            continue

        attribute_set = attribute_sets[submesh.vbo_index]
        position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
        if position_attribute is None or position_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            unsupported_submeshes += 1
            continue

        try:
            position_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)]
        except (G1MParseError, IndexError):
            unsupported_submeshes += 1
            continue

        weight_attribute = attribute_set.find_attribute(SEMANTIC_BONE_WEIGHT)
        bone_index_attribute = attribute_set.find_attribute(SEMANTIC_BONE_INDEX)
        if weight_attribute and weight_attribute.data_type not in SUPPORTED_WEIGHT_TYPES:
            weight_attribute = None
        if bone_index_attribute and bone_index_attribute.data_type not in SUPPORTED_INDEX_TYPES:
            bone_index_attribute = None

        bind_set = (
            model.bone_bind_sets[submesh.bone_table_index]
            if 0 <= submesh.bone_table_index < len(model.bone_bind_sets)
            else None
        )
        rigid_bone_id = resolve_rigid_bone_id(model, submesh.index)
        submesh_changed = False

        for local_vertex_index in range(submesh.vertex_count):
            vertex_index = submesh.vertex_offset + local_vertex_index
            vertex_key = (position_buffer.index, vertex_index)
            if vertex_key in pending_positions:
                continue
            visited_vertices.add(vertex_key)

            try:
                position_components = read_attribute_components(position_buffer, position_attribute, vertex_index)
                bone_ids, bone_weights = resolve_vertex_influences(
                    model,
                    bind_set,
                    rigid_bone_id,
                    vertex_buffers,
                    attribute_set,
                    vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
            except (G1MParseError, IndexError):
                skipped_count += 1
                continue

            matching = [
                (int(bone_id), max(float(weight), 0.0))
                for bone_id, weight in zip(bone_ids, bone_weights)
                if int(bone_id) in target_bones and float(weight) > 0.0
            ]
            influence = sum(weight for _bone_id, weight in matching)
            if influence < max(float(controls.min_weight), 0.0):
                continue

            center = weighted_center_from_bones(matching, bone_centers)
            if center is None:
                continue

            position = Vector3(
                float(position_components[0]),
                float(position_components[1]),
                float(position_components[2]),
            )
            factor = vertexsculpt_factor(position, center, influence, controls)
            if factor <= 1e-6:
                continue

            mode = (controls.sculpt_mode or "scale_offset").lower()
            if mode == "inflate":
                normal = vertex_normal_for_sculpt(
                    output,
                    vertex_buffers,
                    attribute_set,
                    submesh,
                    index_buffers,
                    vertex_index,
                    position,
                    center,
                )
                amount = float(controls.inflate_amount) * factor
                sculpted = Vector3(
                    position.x + normal.x * amount,
                    position.y + normal.y * amount,
                    position.z + normal.z * amount,
                )
            elif mode == "smooth":
                average = neighbor_average_for_vertex(
                    output,
                    vertex_buffers,
                    attribute_set,
                    submesh,
                    index_buffers,
                    vertex_index,
                    position_attribute,
                    position_buffer,
                )
                if average is None:
                    skipped_count += 1
                    continue
                strength = min(max(float(controls.smooth_strength), 0.0), 1.0) * factor
                sculpted = Vector3(
                    position.x + (average.x - position.x) * strength,
                    position.y + (average.y - position.y) * strength,
                    position.z + (average.z - position.z) * strength,
                )
            else:
                target = vertexsculpted_position(position, center, controls)
                sculpted = Vector3(
                    position.x + (target.x - position.x) * factor,
                    position.y + (target.y - position.y) * factor,
                    position.z + (target.z - position.z) * factor,
                )
            pending_positions[vertex_key] = (position_buffer, position_attribute, position, sculpted)
            submesh_changed = True

        if submesh_changed:
            affected_submeshes.add(submesh.index)

    if controls.weld_vertices and pending_positions:
        weld_map = build_position_weld_map(
            vertex_buffers,
            attribute_sets,
            model.submeshes,
            threshold=float(controls.weld_threshold),
        )
        expanded = dict(pending_positions)
        for key, (position_buffer, position_attribute, old_position, sculpted) in pending_positions.items():
            delta = Vector3(sculpted.x - old_position.x, sculpted.y - old_position.y, sculpted.z - old_position.z)
            for twin in weld_map.get(key, ()):
                if twin in expanded:
                    continue
                twin_buffer = next((buffer for buffer in vertex_buffers if buffer.index == twin[0]), None)
                if twin_buffer is None or not 0 <= twin[1] < twin_buffer.count:
                    continue
                try:
                    twin_attr = position_attribute_for_vertex_buffer(twin_buffer.index, attribute_sets, vertex_buffers)
                    if twin_attr is None:
                        continue
                    twin_position = Vector3(*readFLOAT3_from_output(output, twin_buffer, twin_attr, twin[1]))
                except G1MParseError:
                    continue
                expanded[twin] = (
                    twin_buffer,
                    twin_attr,
                    twin_position,
                    Vector3(twin_position.x + delta.x, twin_position.y + delta.y, twin_position.z + delta.z),
                )
        pending_positions = expanded

    for vertex_key, (position_buffer, position_attribute, _old_position, sculpted) in pending_positions.items():
        write_float_position(output, position_buffer, position_attribute, vertex_key[1], sculpted)

    if pending_positions:
        recompute_normals(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)
        recompute_tangents(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)

    return bytes(output), RegionSculptResult(
        patched_vertex_count=len(pending_positions),
        visited_vertex_count=len(visited_vertices),
        skipped_vertex_count=skipped_count,
        affected_submesh_count=len(affected_submeshes),
        unsupported_submesh_count=unsupported_submeshes,
    )


def sculpt_g1m_region_bytes(
    model: G1MModel,
    controls: RegionSculptControls,
) -> tuple[bytes, RegionSculptResult]:
    if controls.is_neutral():
        return model.raw_data, RegionSculptResult(0, 0, 0, 0, 0)
    if not 0 <= controls.bone_id < len(model.bones):
        raise G1MParseError(f"Region sculpt target bone {controls.bone_id} is outside the skeleton.")

    vertex_buffers, attribute_sets, index_buffers = parse_writable_preview_resources(model)
    submesh_indices = target_submesh_indices(model, controls.body_part_index, controls.bone_filter)
    if not submesh_indices:
        return model.raw_data, RegionSculptResult(0, 0, 0, 0, 0)

    original_globals = build_global_matrices(model, current=False)
    center_matrix = original_globals[controls.bone_id]
    center = Vector3(center_matrix[3], center_matrix[7], center_matrix[11])
    output = bytearray(model.raw_data)

    visited_vertices: set[tuple[int, int]] = set()
    patched_vertices: set[tuple[int, int]] = set()
    affected_submeshes: set[int] = set()
    skipped_count = 0
    unsupported_submeshes = 0

    for submesh_index in submesh_indices:
        if not 0 <= submesh_index < len(model.submeshes):
            unsupported_submeshes += 1
            continue
        submesh = model.submeshes[submesh_index]
        if not 0 <= submesh.vbo_index < len(attribute_sets):
            unsupported_submeshes += 1
            continue

        attribute_set = attribute_sets[submesh.vbo_index]
        position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
        if position_attribute is None or position_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            unsupported_submeshes += 1
            continue

        position_buffer_index = attribute_set.vertex_buffer_index_for(position_attribute)
        if not 0 <= position_buffer_index < len(vertex_buffers):
            unsupported_submeshes += 1
            continue
        position_buffer = vertex_buffers[position_buffer_index]

        weight_attribute = attribute_set.find_attribute(SEMANTIC_BONE_WEIGHT)
        bone_index_attribute = attribute_set.find_attribute(SEMANTIC_BONE_INDEX)
        if weight_attribute and weight_attribute.data_type not in SUPPORTED_WEIGHT_TYPES:
            weight_attribute = None
        if bone_index_attribute and bone_index_attribute.data_type not in SUPPORTED_INDEX_TYPES:
            bone_index_attribute = None

        bind_set = (
            model.bone_bind_sets[submesh.bone_table_index]
            if 0 <= submesh.bone_table_index < len(model.bone_bind_sets)
            else None
        )
        rigid_bone_id = resolve_rigid_bone_id(model, submesh.index)

        submesh_changed = False
        for local_vertex_index in range(submesh.vertex_count):
            vertex_index = submesh.vertex_offset + local_vertex_index
            vertex_key = (position_buffer.index, vertex_index)
            if vertex_key in patched_vertices:
                continue
            visited_vertices.add(vertex_key)

            try:
                position_components = read_attribute_components(
                    position_buffer,
                    position_attribute,
                    vertex_index,
                )
                bone_ids, bone_weights = resolve_vertex_influences(
                    model,
                    bind_set,
                    rigid_bone_id,
                    vertex_buffers,
                    attribute_set,
                    vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
            except (G1MParseError, IndexError):
                skipped_count += 1
                continue

            influence = sum(
                weight
                for bone_id, weight in zip(bone_ids, bone_weights)
                if bone_id == controls.bone_id
            )
            if influence < max(controls.min_weight, 0.0):
                continue

            position = Vector3(
                float(position_components[0]),
                float(position_components[1]),
                float(position_components[2]),
            )
            factor = sculpt_factor(position, center, influence, controls)
            if factor <= 1e-6:
                continue

            target = sculpted_position(position, center, controls)
            sculpted = Vector3(
                position.x + (target.x - position.x) * factor,
                position.y + (target.y - position.y) * factor,
                position.z + (target.z - position.z) * factor,
            )
            write_float_position(output, position_buffer, position_attribute, vertex_index, sculpted)
            patched_vertices.add(vertex_key)
            submesh_changed = True

        if submesh_changed:
            affected_submeshes.add(submesh.index)

    if patched_vertices:
        recompute_normals(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)
        recompute_tangents(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)

    return bytes(output), RegionSculptResult(
        patched_vertex_count=len(patched_vertices),
        visited_vertex_count=len(visited_vertices),
        skipped_vertex_count=skipped_count,
        affected_submesh_count=len(affected_submeshes),
        unsupported_submesh_count=unsupported_submeshes,
    )


def save_morph(morph: MorphTarget, path: str | Path) -> None:
    payload = {
        "name": morph.name,
        "bone_hint": morph.bone_hint,
        "source_file": morph.source_file,
        "deltas": [[d.vbo_index, d.vertex_index, d.dx, d.dy, d.dz] for d in morph.deltas],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_morph(path: str | Path) -> MorphTarget:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    deltas = [
        VertexDelta(int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]))
        for row in payload.get("deltas", [])
    ]
    bone_hint = payload.get("bone_hint")
    return MorphTarget(
        name=str(payload.get("name") or Path(path).stem),
        bone_hint=int(bone_hint) if bone_hint is not None else None,
        source_file=str(payload.get("source_file") or ""),
        deltas=deltas,
    )


def capture_deltas(
    original_data: bytes,
    patched_data: bytes,
    vertex_buffers: list[_WritableVertexBuffer],
    name: str,
    bone_hint: int | None = None,
    source_file: str = "",
    threshold: float = 1e-4,
) -> MorphTarget:
    if len(original_data) != len(patched_data):
        raise G1MParseError("Cannot capture morph deltas from buffers with different sizes.")

    deltas: list[VertexDelta] = []
    for vertex_buffer in vertex_buffers:
        for vertex_index in range(vertex_buffer.count):
            offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride
            ensure_range(original_data, offset, 12, "original vertex position")
            ensure_range(patched_data, offset, 12, "patched vertex position")
            ox, oy, oz = FLOAT3.unpack_from(original_data, offset)
            px, py, pz = FLOAT3.unpack_from(patched_data, offset)
            dx = float(px - ox)
            dy = float(py - oy)
            dz = float(pz - oz)
            if abs(dx) > threshold or abs(dy) > threshold or abs(dz) > threshold:
                deltas.append(VertexDelta(vertex_buffer.index, vertex_index, dx, dy, dz))

    return MorphTarget(name=name, bone_hint=bone_hint, source_file=source_file, deltas=deltas)


def apply_morphs(
    model: G1MModel,
    morphs: list[tuple[MorphTarget, float]],
) -> bytes:
    vertex_buffers, attribute_sets, index_buffers = parse_writable_preview_resources(model)
    vertex_buffer_by_index = {buffer.index: buffer for buffer in vertex_buffers}

    accumulated: dict[tuple[int, int], list[float]] = {}
    for morph, weight in morphs:
        if abs(weight) <= 1e-8:
            continue
        scalar = float(weight)
        for delta in morph.deltas:
            key = (int(delta.vbo_index), int(delta.vertex_index))
            buffer = vertex_buffer_by_index.get(key[0])
            if buffer is None or not 0 <= key[1] < buffer.count:
                continue
            slot = accumulated.setdefault(key, [0.0, 0.0, 0.0])
            slot[0] += float(delta.dx) * scalar
            slot[1] += float(delta.dy) * scalar
            slot[2] += float(delta.dz) * scalar

    if not accumulated:
        return model.raw_data

    weld_map = build_position_weld_map(vertex_buffers, attribute_sets, model.submeshes)
    expanded = dict(accumulated)
    for key, delta in accumulated.items():
        for twin in weld_map.get(key, ()):
            if twin not in expanded:
                expanded[twin] = list(delta)

    output = bytearray(model.raw_data)
    for (vbo_index, vertex_index), (dx, dy, dz) in expanded.items():
        buffer = vertex_buffer_by_index.get(vbo_index)
        if buffer is None or not 0 <= vertex_index < buffer.count:
            continue
        entry_offset = buffer.data_offset + vertex_index * buffer.stride
        ensure_range(output, entry_offset, 12, "morph vertex position")
        x, y, z = FLOAT3.unpack_from(model.raw_data, entry_offset)
        FLOAT3.pack_into(output, entry_offset, x + dx, y + dy, z + dz)

    recompute_normals(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)
    recompute_tangents(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)
    return bytes(output)


def recompute_all_normals_tangents(model: G1MModel) -> bytes:
    vertex_buffers, attribute_sets, index_buffers = parse_writable_preview_resources(model)
    output = bytearray(model.raw_data)
    recompute_normals(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)
    recompute_tangents(output, vertex_buffers, attribute_sets, index_buffers, model.submeshes)
    return bytes(output)


def strip_to_triangles(
    indices: list[int],
    restart_index: int,
) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    strip: list[int] = []
    for index in indices:
        if index == restart_index:
            strip.clear()
            continue
        strip.append(index)
        if len(strip) < 3:
            continue
        a, b, c = strip[-3:]
        if a == b or b == c or a == c:
            continue
        if (len(strip) - 3) % 2:
            a, b = b, a
        triangles.append((a, b, c))
    return triangles


def indices_to_triangles(indices: list[int], face_type: int, restart_index: int) -> list[tuple[int, int, int]]:
    if face_type == FACE_TYPE_TRIANGLE_STRIP:
        return strip_to_triangles(indices, restart_index)
    if face_type == FACE_TYPE_TRIANGLE:
        triangles: list[tuple[int, int, int]] = []
        limit = len(indices) - (len(indices) % 3)
        for cursor in range(0, limit, 3):
            a, b, c = indices[cursor : cursor + 3]
            if restart_index in (a, b, c) or a == b or b == c or a == c:
                continue
            triangles.append((a, b, c))
        return triangles
    if face_type == FACE_TYPE_QUAD:
        triangles = []
        limit = len(indices) - (len(indices) % 4)
        for cursor in range(0, limit, 4):
            a, b, c, d = indices[cursor : cursor + 4]
            if restart_index in (a, b, c, d):
                continue
            triangles.append((a, b, c))
            triangles.append((a, c, d))
        return triangles
    return []


def recompute_normals(
    output: bytearray,
    vertex_buffers: list[_WritableVertexBuffer],
    attribute_sets: list[VertexAttributeSet],
    index_buffers: list[_WritableIndexBuffer],
    submeshes: list[Submesh],
) -> None:
    accumulators: dict[tuple[int, int], list[float]] = {}
    normal_targets: dict[tuple[int, int], tuple[_WritableVertexBuffer, VertexAttribute]] = {}

    for submesh in submeshes:
        if not 0 <= submesh.vbo_index < len(attribute_sets):
            continue
        if not 0 <= submesh.ib_index < len(index_buffers):
            continue
        attribute_set = attribute_sets[submesh.vbo_index]
        position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
        normal_attribute = attribute_set.find_attribute(SEMANTIC_NORMAL)
        if position_attribute is None or normal_attribute is None:
            continue
        if position_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            continue
        if normal_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            continue
        try:
            position_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)]
            normal_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(normal_attribute)]
        except (G1MParseError, IndexError):
            continue

        index_buffer = index_buffers[submesh.ib_index]
        raw_indices = index_buffer.indices[submesh.face_offset : submesh.face_offset + submesh.face_count]
        for a, b, c in indices_to_triangles(raw_indices, submesh.face_type, index_buffer.restart_index):
            if not indices_in_submesh((a, b, c), submesh):
                continue
            try:
                p0 = readFLOAT3_from_output(output, position_buffer, position_attribute, a)
                p1 = readFLOAT3_from_output(output, position_buffer, position_attribute, b)
                p2 = readFLOAT3_from_output(output, position_buffer, position_attribute, c)
            except G1MParseError:
                continue
            e1 = vec_sub(p1, p0)
            e2 = vec_sub(p2, p0)
            face_normal = vec_cross(e1, e2)
            if vec_length(face_normal) <= 1e-12:
                continue
            for vertex_index in (a, b, c):
                key = (normal_buffer.index, vertex_index)
                slot = accumulators.setdefault(key, [0.0, 0.0, 0.0])
                slot[0] += face_normal[0]
                slot[1] += face_normal[1]
                slot[2] += face_normal[2]
                normal_targets[key] = (normal_buffer, normal_attribute)

    for (buffer_index, vertex_index), normal in accumulators.items():
        target = normal_targets.get((buffer_index, vertex_index))
        if target is None:
            continue
        normal_buffer, normal_attribute = target
        nx, ny, nz = vec_normalize(normal)
        writeFLOAT3_or_4(output, normal_buffer, normal_attribute, vertex_index, (nx, ny, nz), 0.0)


def recompute_tangents(
    output: bytearray,
    vertex_buffers: list[_WritableVertexBuffer],
    attribute_sets: list[VertexAttributeSet],
    index_buffers: list[_WritableIndexBuffer],
    submeshes: list[Submesh],
) -> None:
    tangent_accum: dict[tuple[int, int], list[float]] = {}
    bitangent_accum: dict[tuple[int, int], list[float]] = {}
    tangent_targets: dict[tuple[int, int], tuple[_WritableVertexBuffer, VertexAttribute, _WritableVertexBuffer, VertexAttribute]] = {}

    for submesh in submeshes:
        if not 0 <= submesh.vbo_index < len(attribute_sets):
            continue
        if not 0 <= submesh.ib_index < len(index_buffers):
            continue
        attribute_set = attribute_sets[submesh.vbo_index]
        position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
        uv_attribute = attribute_set.find_attribute(SEMANTIC_UV)
        normal_attribute = attribute_set.find_attribute(SEMANTIC_NORMAL)
        tangent_attribute = attribute_set.find_attribute(SEMANTIC_TANGENT)
        if None in (position_attribute, uv_attribute, normal_attribute, tangent_attribute):
            continue
        if position_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            continue
        if uv_attribute.data_type != VERTEX_TYPE_FLOAT2:
            continue
        if normal_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            continue
        if tangent_attribute.data_type != VERTEX_TYPE_FLOAT4:
            continue
        try:
            position_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)]
            uv_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(uv_attribute)]
            normal_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(normal_attribute)]
            tangent_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(tangent_attribute)]
        except (G1MParseError, IndexError):
            continue

        index_buffer = index_buffers[submesh.ib_index]
        raw_indices = index_buffer.indices[submesh.face_offset : submesh.face_offset + submesh.face_count]
        for a, b, c in indices_to_triangles(raw_indices, submesh.face_type, index_buffer.restart_index):
            if not indices_in_submesh((a, b, c), submesh):
                continue
            try:
                p0 = readFLOAT3_from_output(output, position_buffer, position_attribute, a)
                p1 = readFLOAT3_from_output(output, position_buffer, position_attribute, b)
                p2 = readFLOAT3_from_output(output, position_buffer, position_attribute, c)
                uv0 = readFLOAT2_from_output(output, uv_buffer, uv_attribute, a)
                uv1 = readFLOAT2_from_output(output, uv_buffer, uv_attribute, b)
                uv2 = readFLOAT2_from_output(output, uv_buffer, uv_attribute, c)
            except G1MParseError:
                continue
            edge1 = vec_sub(p1, p0)
            edge2 = vec_sub(p2, p0)
            duv1 = (uv1[0] - uv0[0], uv1[1] - uv0[1])
            duv2 = (uv2[0] - uv0[0], uv2[1] - uv0[1])
            determinant = duv1[0] * duv2[1] - duv2[0] * duv1[1]
            if abs(determinant) <= 1e-12:
                continue
            r = 1.0 / determinant
            tangent = (
                (edge1[0] * duv2[1] - edge2[0] * duv1[1]) * r,
                (edge1[1] * duv2[1] - edge2[1] * duv1[1]) * r,
                (edge1[2] * duv2[1] - edge2[2] * duv1[1]) * r,
            )
            bitangent = (
                (edge2[0] * duv1[0] - edge1[0] * duv2[0]) * r,
                (edge2[1] * duv1[0] - edge1[1] * duv2[0]) * r,
                (edge2[2] * duv1[0] - edge1[2] * duv2[0]) * r,
            )
            for vertex_index in (a, b, c):
                key = (tangent_buffer.index, vertex_index)
                t_slot = tangent_accum.setdefault(key, [0.0, 0.0, 0.0])
                b_slot = bitangent_accum.setdefault(key, [0.0, 0.0, 0.0])
                t_slot[0] += tangent[0]
                t_slot[1] += tangent[1]
                t_slot[2] += tangent[2]
                b_slot[0] += bitangent[0]
                b_slot[1] += bitangent[1]
                b_slot[2] += bitangent[2]
                tangent_targets[key] = (tangent_buffer, tangent_attribute, normal_buffer, normal_attribute)

    for key, tangent in tangent_accum.items():
        target = tangent_targets.get(key)
        if target is None:
            continue
        tangent_buffer, tangent_attribute, normal_buffer, normal_attribute = target
        vertex_index = key[1]
        try:
            normal = readFLOAT3_from_output(output, normal_buffer, normal_attribute, vertex_index)
        except G1MParseError:
            continue
        n = vec_normalize(normal)
        t = vec_sub(tangent, vec_mul(n, vec_dot(n, tangent)))
        t = vec_normalize(t)
        if vec_length(t) <= 1e-12:
            continue
        handedness = read_existing_tangent_w(output, tangent_buffer, tangent_attribute, vertex_index)
        if abs(handedness) <= 1e-6:
            bitangent = bitangent_accum.get(key, [0.0, 0.0, 0.0])
            handedness = -1.0 if vec_dot(vec_cross(n, t), bitangent) < 0.0 else 1.0
        writeFLOAT4(output, tangent_buffer, tangent_attribute, vertex_index, (t[0], t[1], t[2], handedness))


def build_position_weld_map(
    vertex_buffers: list[_WritableVertexBuffer],
    attribute_sets: list[VertexAttributeSet],
    submeshes: list[Submesh],
    threshold: float = 1e-3,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    if threshold <= 0.0:
        threshold = 1e-3
    buckets: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    seen: set[tuple[int, int]] = set()

    for submesh in submeshes:
        if not 0 <= submesh.vbo_index < len(attribute_sets):
            continue
        attribute_set = attribute_sets[submesh.vbo_index]
        position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
        if position_attribute is None or position_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            continue
        try:
            position_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)]
        except (G1MParseError, IndexError):
            continue
        for local_vertex_index in range(submesh.vertex_count):
            vertex_index = submesh.vertex_offset + local_vertex_index
            key = (position_buffer.index, vertex_index)
            if key in seen:
                continue
            seen.add(key)
            try:
                x, y, z = read_attribute_components(position_buffer, position_attribute, vertex_index)[:3]
            except G1MParseError:
                continue
            bucket_key = (round(float(x) / threshold), round(float(y) / threshold), round(float(z) / threshold))
            buckets.setdefault(bucket_key, []).append(key)

    weld_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for entries in buckets.values():
        if len(entries) <= 1:
            continue
        for entry in entries:
            weld_map[entry] = [other for other in entries if other != entry]
    return weld_map


def submeshes_for_bones(model: G1MModel, bone_ids: set[int]) -> list[int]:
    if not bone_ids:
        return []
    matches: list[int] = []
    wanted = set(int(bone_id) for bone_id in bone_ids)
    for submesh in model.submeshes:
        if not 0 <= submesh.bone_table_index < len(model.bone_bind_sets):
            continue
        bind_set = model.bone_bind_sets[submesh.bone_table_index]
        present: set[int] = set()
        for bind in bind_set.binds:
            present.add(int(bind.bone_id))
            present.add(int(bind.reference_bone_id))
        if present.intersection(wanted):
            matches.append(submesh.index)
    return matches


def parse_writable_preview_resources(
    model: G1MModel,
) -> tuple[list[_WritableVertexBuffer], list[VertexAttributeSet], list[_WritableIndexBuffer]]:
    geometry_section = next((section for section in model.sections if section.magic == "G1MG"), None)
    if geometry_section is None:
        raise G1MParseError("Region sculpting requires a G1MG geometry section.")

    data = model.raw_data
    payload_offset = geometry_section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 36, "G1MG header")
    subsection_count = int.from_bytes(data[payload_offset + 32 : payload_offset + 36], "little", signed=True)
    try:
        g1mg_version = int(geometry_section.version)
    except ValueError:
        g1mg_version = 0

    vertex_buffers: list[_WritableVertexBuffer] = []
    attribute_sets: list[VertexAttributeSet] = []
    index_buffers: list[_WritableIndexBuffer] = []
    cursor = payload_offset + 36
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, G1MG_SUBSECTION_HEADER_STRUCT.size, f"G1MG subsection {subsection_index}")
        subsection_type, unknown, size = G1MG_SUBSECTION_HEADER_STRUCT.unpack_from(data, cursor)
        ensure_range(data, cursor, size, f"G1MG subsection {subsection_index}")
        payload = cursor + G1MG_SUBSECTION_HEADER_STRUCT.size
        subsection_end = cursor + size
        ensure_range(data, payload, 4, f"G1MG subsection {subsection_index} entry count")
        count = int.from_bytes(data[payload : payload + 4], "little", signed=False)
        payload += 4

        if subsection_type == 4:
            for _ in range(count):
                vb_header_size = VERTEX_BUFFER_HEADER_STRUCT.size if g1mg_version > 40 else 12
                ensure_range(data, payload, vb_header_size, "vertex buffer header")
                if vb_header_size == VERTEX_BUFFER_HEADER_STRUCT.size:
                    unknown0, stride, vertex_count, unknown1 = VERTEX_BUFFER_HEADER_STRUCT.unpack_from(data, payload)
                else:
                    unknown0, stride, vertex_count = Struct("<3i").unpack_from(data, payload)
                    unknown1 = 0
                payload += vb_header_size
                data_size = stride * vertex_count
                ensure_range(data, payload, data_size, "vertex buffer data")
                vertex_buffers.append(
                    _WritableVertexBuffer(
                        index=len(vertex_buffers),
                        stride=stride,
                        count=vertex_count,
                        raw_data=data[payload : payload + data_size],
                        data_offset=payload,
                    )
                )
                payload += data_size

        elif subsection_type == 5:
            for _ in range(count):
                ensure_range(data, payload, 4, "vertex attribute set")
                vertex_buffer_slot_count = int.from_bytes(data[payload : payload + 4], "little", signed=True)
                payload += 4
                ensure_range(data, payload, vertex_buffer_slot_count * 4, "vertex attribute indices")
                vertex_buffer_indices = list(
                    Struct(f"<{vertex_buffer_slot_count}i").unpack_from(data, payload)
                    if vertex_buffer_slot_count
                    else ()
                )
                payload += vertex_buffer_slot_count * 4

                ensure_range(data, payload, 4, "vertex attribute count")
                attribute_count = int.from_bytes(data[payload : payload + 4], "little", signed=True)
                payload += 4

                attributes: list[VertexAttribute] = []
                for _ in range(attribute_count):
                    ensure_range(data, payload, VERTEX_ATTRIBUTE_STRUCT.size, "vertex attribute")
                    buffer_slot, attr_offset, data_type, semantic, layer = VERTEX_ATTRIBUTE_STRUCT.unpack_from(
                        data,
                        payload,
                    )
                    attributes.append(
                        VertexAttribute(
                            buffer_slot=buffer_slot,
                            offset=attr_offset,
                            data_type=data_type,
                            semantic=semantic,
                            layer=layer,
                        )
                    )
                    payload += VERTEX_ATTRIBUTE_STRUCT.size

                attribute_sets.append(
                    VertexAttributeSet(
                        index=len(attribute_sets),
                        vertex_buffer_indices=vertex_buffer_indices,
                        attributes=attributes,
                    )
                )

        elif subsection_type == 7:
            index_buffers = parse_writable_index_buffers(data, payload, count, subsection_end, g1mg_version)

        cursor += size

    return vertex_buffers, attribute_sets, index_buffers


def parse_writable_index_buffers(
    data: bytes,
    payload: int,
    count: int,
    subsection_end: int,
    g1mg_version: int,
) -> list[_WritableIndexBuffer]:
    def parse_with(header_size: int) -> list[_WritableIndexBuffer] | None:
        cursor = payload
        buffers: list[_WritableIndexBuffer] = []
        for index in range(count):
            ensure_range(data, cursor, header_size, "index buffer header")
            if header_size == 8:
                index_count, bit_width = Struct("<2I").unpack_from(data, cursor)
            elif header_size == 12:
                index_count, bit_width, _pad = Struct("<3I").unpack_from(data, cursor)
            else:
                _flags, index_count, bit_width, _pad = Struct("<4I").unpack_from(data, cursor)
            if bit_width not in {8, 16, 32}:
                return None
            byte_width = bit_width // 8
            cursor += header_size
            ensure_range(data, cursor, int(index_count) * byte_width, "index buffer data")
            if bit_width == 8:
                fmt = f"<{index_count}B"
                restart = 0xFF
            elif bit_width == 16:
                fmt = f"<{index_count}H"
                restart = 0xFFFF
            else:
                fmt = f"<{index_count}I"
                restart = 0xFFFFFFFF
            indices = list(Struct(fmt).unpack_from(data, cursor) if index_count else ())
            data_offset = cursor
            cursor += int(index_count) * byte_width
            cursor = (cursor + 3) & ~3
            buffers.append(
                _WritableIndexBuffer(
                    index=index,
                    bit_width=int(bit_width),
                    count=int(index_count),
                    indices=indices,
                    restart_index=restart,
                    data_offset=data_offset,
                )
            )
        if cursor != subsection_end:
            return None
        return buffers

    # Runtime G1MG handler uses 12 byte index buffer headers for versions > 0040
    # Older versions seem to use 8 byte headers
    # A 16 byte fallback is kept for other KT/Omega Force branches
    header_order = (12, 8, 16) if g1mg_version > 40 else (8, 12, 16)
    for header_size in header_order:
        parsed = parse_with(header_size)
        if parsed is not None:
            return parsed
    raise G1MParseError("Unable to parse G1MG index buffers as 12, 8, or 16 byte headers.")


def target_submesh_indices(
    model: G1MModel,
    body_part_index: int | None,
    bone_filter: frozenset[int] | None = None,
) -> tuple[int, ...]:
    if body_part_index is None:
        candidates = tuple(submesh.index for submesh in model.submeshes)
    elif not 0 <= body_part_index < len(model.body_parts):
        candidates = ()
    else:
        candidates = tuple(model.body_parts[body_part_index].submesh_indices)

    if not bone_filter:
        return candidates
    allowed = set(submeshes_for_bones(model, set(bone_filter)))
    return tuple(index for index in candidates if index in allowed)


@dataclass(frozen=True, slots=True)
class DisableRegionResult:
    disabled_submesh_count: int
    zeroed_index_count: int
    skipped_submesh_count: int


def submeshes_for_body_parts(model: G1MModel, part_indices: Iterable[int]) -> set[int]:
    submesh_indices: set[int] = set()
    for part_index in part_indices:
        if 0 <= int(part_index) < len(model.body_parts):
            submesh_indices.update(model.body_parts[int(part_index)].submesh_indices)
    return submesh_indices


def body_parts_covered_by_submeshes(model: G1MModel, submesh_indices: set[int]) -> set[int]:
    """Body parts whose geometry is entirely inside the given submesh set

    Several body part entries can point at one submesh, blanking a region also
    blanks every other region built from the same faces
    """

    covered: set[int] = set()
    for part in model.body_parts:
        if part.submesh_indices and set(part.submesh_indices) <= submesh_indices:
            covered.add(part.index)
    return covered


def disable_g1m_submeshes(
    model: G1MModel,
    submesh_indices: Iterable[int],
) -> tuple[bytes, DisableRegionResult]:
    """Blank the face indices of the given submeshes

    The G1MG index range for each submesh is overwritten with zeroes, which turns
    its triangles degenerate so nothing rasterizes

    Vertex data, section sizes, and total file length are left untouched

    This makes extracting outfits from characters (some G1Ms have unique outfits that nevfer became selectable for custom characters) easier but also
    probably makes nude mods a reality for gooners, you dirty fucks will probably enjoy this
    """

    wanted = sorted({int(index) for index in submesh_indices})
    if not wanted:
        return model.raw_data, DisableRegionResult(0, 0, 0)

    _vertex_buffers, _attribute_sets, index_buffers = parse_writable_preview_resources(model)
    output = bytearray(model.raw_data)

    zeroed_indices = 0
    disabled_submeshes = 0
    skipped = 0
    for submesh_index in wanted:
        if not 0 <= submesh_index < len(model.submeshes):
            skipped += 1
            continue
        submesh = model.submeshes[submesh_index]
        if not 0 <= submesh.ib_index < len(index_buffers):
            skipped += 1
            continue
        index_buffer = index_buffers[submesh.ib_index]
        byte_width = index_buffer.bit_width // 8
        start = int(submesh.face_offset)
        count = int(submesh.face_count)
        if count <= 0 or start < 0 or start + count > index_buffer.count:
            skipped += 1
            continue
        begin = index_buffer.data_offset + start * byte_width
        ensure_range(output, begin, count * byte_width, f"submesh {submesh_index} index range")
        output[begin : begin + count * byte_width] = bytes(count * byte_width)
        zeroed_indices += count
        disabled_submeshes += 1

    return bytes(output), DisableRegionResult(
        disabled_submesh_count=disabled_submeshes,
        zeroed_index_count=zeroed_indices,
        skipped_submesh_count=skipped,
    )


def disable_g1m_body_parts(
    model: G1MModel,
    part_indices: Iterable[int],
) -> tuple[bytes, DisableRegionResult]:
    """Blank every submesh belonging to the given body parts"""

    return disable_g1m_submeshes(model, submeshes_for_body_parts(model, part_indices))


def weighted_center_from_bones(
    matching: list[tuple[int, float]],
    bone_centers: dict[int, Vector3],
) -> Vector3 | None:
    total = sum(weight for bone_id, weight in matching if bone_id in bone_centers)
    if total <= 1e-8:
        return None
    x = y = z = 0.0
    for bone_id, weight in matching:
        center = bone_centers.get(bone_id)
        if center is None:
            continue
        x += center.x * weight
        y += center.y * weight
        z += center.z * weight
    return Vector3(x / total, y / total, z / total)


def vertexsculpt_factor(
    position: Vector3,
    center: Vector3,
    influence: float,
    controls: VertexSculptControls,
) -> float:
    factor = min(max(float(influence), 0.0), 1.0)
    radius = max(float(controls.radius), 0.0)
    if radius > 1e-6:
        distance = math.sqrt(
            (position.x - center.x) * (position.x - center.x)
            + (position.y - center.y) * (position.y - center.y)
            + (position.z - center.z) * (position.z - center.z)
        )
        if distance >= radius:
            return 0.0
        falloff = 1.0 - (distance / radius)
        factor *= math.pow(falloff, max(float(controls.falloff_power), 0.01))
    return min(max(factor, 0.0), 1.0)


def vertex_normal_for_sculpt(
    output: bytes | bytearray,
    vertex_buffers: list[_WritableVertexBuffer],
    attribute_set: VertexAttributeSet,
    submesh: Submesh,
    index_buffers: list[_WritableIndexBuffer],
    vertex_index: int,
    position: Vector3,
    center: Vector3,
) -> Vector3:
    normal_attribute = attribute_set.find_attribute(SEMANTIC_NORMAL)
    if normal_attribute is not None and normal_attribute.data_type in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
        try:
            normal_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(normal_attribute)]
            nx, ny, nz = readFLOAT3_from_output(output, normal_buffer, normal_attribute, vertex_index)
            normal = vec_normalize((float(nx), float(ny), float(nz)))
            return Vector3(float(normal[0]), float(normal[1]), float(normal[2]))
        except (G1MParseError, IndexError):
            pass

    fallback = vec_normalize((position.x - center.x, position.y - center.y, position.z - center.z))
    return Vector3(float(fallback[0]), float(fallback[1]), float(fallback[2]))


def neighbor_average_for_vertex(
    output: bytes | bytearray,
    vertex_buffers: list[_WritableVertexBuffer],
    attribute_set: VertexAttributeSet,
    submesh: Submesh,
    index_buffers: list[_WritableIndexBuffer],
    vertex_index: int,
    position_attribute: VertexAttribute,
    position_buffer: _WritableVertexBuffer,
) -> Vector3 | None:
    if not 0 <= submesh.buffer_index < len(index_buffers):
        return None
    index_buffer = index_buffers[submesh.buffer_index]
    raw_indices = index_buffer.indices[submesh.face_offset : submesh.face_offset + submesh.face_count]
    triangles = indices_to_triangles(raw_indices, submesh.face_type, index_buffer.restart_index)
    neighbor_indices: set[int] = set()
    for a, b, c in triangles:
        tri = (a, b, c)
        if vertex_index not in tri:
            continue
        for value in tri:
            if value != vertex_index and submesh.vertex_offset <= value < submesh.vertex_offset + submesh.vertex_count:
                neighbor_indices.add(value)
    if not neighbor_indices:
        return None

    x = y = z = 0.0
    count = 0
    for neighbor_index in neighbor_indices:
        if not 0 <= neighbor_index < position_buffer.count:
            continue
        try:
            nx, ny, nz = readFLOAT3_from_output(output, position_buffer, position_attribute, neighbor_index)
        except G1MParseError:
            continue
        x += float(nx)
        y += float(ny)
        z += float(nz)
        count += 1
    if count <= 0:
        return None
    return Vector3(x / count, y / count, z / count)


def vertexsculpted_position(position: Vector3, center: Vector3, controls: VertexSculptControls) -> Vector3:
    return Vector3(
        center.x + (position.x - center.x) * controls.scale[0] + controls.offset[0],
        center.y + (position.y - center.y) * controls.scale[1] + controls.offset[1],
        center.z + (position.z - center.z) * controls.scale[2] + controls.offset[2],
    )


def position_attribute_for_vertex_buffer(
    vertex_buffer_index: int,
    attribute_sets: list[VertexAttributeSet],
    vertex_buffers: list[_WritableVertexBuffer],
) -> VertexAttribute | None:
    for attribute_set in attribute_sets:
        position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
        if position_attribute is None or position_attribute.data_type not in {VERTEX_TYPE_FLOAT3, VERTEX_TYPE_FLOAT4}:
            continue
        try:
            if vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)].index == vertex_buffer_index:
                return position_attribute
        except (G1MParseError, IndexError):
            continue
    return None


def sculpt_factor(
    position: Vector3,
    center: Vector3,
    influence: float,
    controls: RegionSculptControls,
) -> float:
    factor = min(max(influence, 0.0), 1.0)
    radius = max(float(controls.radius), 0.0)
    if radius > 1e-6:
        distance = math.sqrt(
            (position.x - center.x) * (position.x - center.x)
            + (position.y - center.y) * (position.y - center.y)
            + (position.z - center.z) * (position.z - center.z)
        )
        if distance >= radius:
            return 0.0
        falloff = 1.0 - (distance / radius)
        factor *= math.pow(falloff, max(float(controls.falloff_power), 0.01))
    return min(max(factor, 0.0), 1.0)


def sculpted_position(position: Vector3, center: Vector3, controls: RegionSculptControls) -> Vector3:
    return Vector3(
        center.x + (position.x - center.x) * controls.scale[0] + controls.offset[0],
        center.y + (position.y - center.y) * controls.scale[1] + controls.offset[1],
        center.z + (position.z - center.z) * controls.scale[2] + controls.offset[2],
    )


def write_float_position(
    output: bytearray,
    vertex_buffer: _WritableVertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
    position: Vector3,
) -> None:
    entry_offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride + attribute.offset
    if attribute.data_type == VERTEX_TYPE_FLOAT3:
        ensure_range(output, entry_offset, 12, "float3 position")
        FLOAT3.pack_into(output, entry_offset, position.x, position.y, position.z)
        return
    if attribute.data_type == VERTEX_TYPE_FLOAT4:
        ensure_range(output, entry_offset, 12, "float4 position")
        FLOAT3.pack_into(output, entry_offset, position.x, position.y, position.z)
        return
    raise G1MParseError(f"Region sculpt can't write position type 0x{attribute.data_type:04X}.")


def readFLOAT3_from_output(
    output: bytes | bytearray,
    vertex_buffer: _WritableVertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
) -> tuple[float, float, float]:
    if not 0 <= vertex_index < vertex_buffer.count:
        raise G1MParseError(f"Vertex index {vertex_index} is out of range for vertex buffer {vertex_buffer.index}.")
    entry_offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride + attribute.offset
    ensure_range(output, entry_offset, 12, "float3 attribute")
    return FLOAT3.unpack_from(output, entry_offset)


def readFLOAT2_from_output(
    output: bytes | bytearray,
    vertex_buffer: _WritableVertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
) -> tuple[float, float]:
    if not 0 <= vertex_index < vertex_buffer.count:
        raise G1MParseError(f"Vertex index {vertex_index} is out of range for vertex buffer {vertex_buffer.index}.")
    entry_offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride + attribute.offset
    ensure_range(output, entry_offset, 8, "float2 attribute")
    return FLOAT2.unpack_from(output, entry_offset)


def writeFLOAT3_or_4(
    output: bytearray,
    vertex_buffer: _WritableVertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
    xyz: tuple[float, float, float],
    w: float,
) -> None:
    entry_offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride + attribute.offset
    if attribute.data_type == VERTEX_TYPE_FLOAT3:
        ensure_range(output, entry_offset, 12, "float3 write")
        FLOAT3.pack_into(output, entry_offset, *xyz)
        return
    if attribute.data_type == VERTEX_TYPE_FLOAT4:
        ensure_range(output, entry_offset, 16, "float4 write")
        FLOAT4.pack_into(output, entry_offset, xyz[0], xyz[1], xyz[2], w)
        return


def writeFLOAT4(
    output: bytearray,
    vertex_buffer: _WritableVertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
    xyzw: tuple[float, float, float, float],
) -> None:
    entry_offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride + attribute.offset
    ensure_range(output, entry_offset, 16, "float4 write")
    FLOAT4.pack_into(output, entry_offset, *xyzw)


def read_existing_tangent_w(
    output: bytes | bytearray,
    vertex_buffer: _WritableVertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
) -> float:
    entry_offset = vertex_buffer.data_offset + vertex_index * vertex_buffer.stride + attribute.offset
    ensure_range(output, entry_offset, 16, "tangent")
    return float(FLOAT4.unpack_from(output, entry_offset)[3])


def indices_in_submesh(indices: Iterable[int], submesh: Submesh) -> bool:
    start = submesh.vertex_offset
    end = start + submesh.vertex_count
    return all(start <= index < end for index in indices)


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_mul(a, scalar: float):
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def vec_dot(a, b) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_length(a) -> float:
    return math.sqrt(vec_dot(a, a))


def vec_normalize(a):
    length = vec_length(a)
    if length <= 1e-12:
        return (0.0, 1.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def ensure_range(data: bytes | bytearray, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise G1MParseError(f"{label} extends past the end of the file.")
