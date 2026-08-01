"""Preview scene extraction and CPU skinning helpers for G1M models"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math, struct
from struct import Struct

from .cloth_shader import ClothShaderInput, Float4, make_cloth_constant_buffer, transform_cloth_vertex2
from .reader import G1MModel, G1MParseError, Vector3


RESOURCE_HEADER_STRUCT = Struct("<4s4sI")
ARRAY_SECTION_HEADER_STRUCT = Struct("<HHII")
G1MG_SUBSECTION_HEADER_STRUCT = Struct("<HHI")
VERTEX_BUFFER_HEADER_STRUCT = Struct("<4i")
INDEX_BUFFER_HEADER_STRUCT = Struct("<3i")
VERTEX_ATTRIBUTE_STRUCT = Struct("<hhHBB")

SEMANTIC_POSITION = 0
SEMANTIC_BONE_WEIGHT = 1
SEMANTIC_BONE_INDEX = 2
SEMANTIC_NORMAL = 3
SEMANTIC_POINT_SIZE = 4
SEMANTIC_UV = 5
SEMANTIC_TANGENT = 6
SEMANTIC_BINORMAL = 7
SEMANTIC_COLOR = 10
SEMANTIC_FOG = 11

FACE_TYPE_QUAD = 1
FACE_TYPE_TRIANGLE = 3
FACE_TYPE_TRIANGLE_STRIP = 4

VERTEX_TYPE_FLOAT1 = 0x0000
VERTEX_TYPE_FLOAT2 = 0x0001
VERTEX_TYPE_FLOAT3 = 0x0002
VERTEX_TYPE_FLOAT4 = 0x0003
VERTEX_TYPE_BYTE4 = 0x0005
VERTEX_TYPE_USHORT4 = 0x0007
VERTEX_TYPE_HALF2 = 0x000A
VERTEX_TYPE_HALF4 = 0x000B
VERTEX_TYPE_NORMALIZED_BYTE4 = 0x000D
VERTEX_TYPE_UINT4 = 0x0009

SUPPORTED_WEIGHT_TYPES = {
    VERTEX_TYPE_FLOAT1,
    VERTEX_TYPE_FLOAT2,
    VERTEX_TYPE_FLOAT3,
    VERTEX_TYPE_FLOAT4,
    VERTEX_TYPE_NORMALIZED_BYTE4,
}
SUPPORTED_INDEX_TYPES = {
    VERTEX_TYPE_BYTE4,
    VERTEX_TYPE_USHORT4,
    VERTEX_TYPE_UINT4,
    VERTEX_TYPE_NORMALIZED_BYTE4,
}
SUPPORTED_POSITION_TYPES = {
    VERTEX_TYPE_FLOAT3,
    VERTEX_TYPE_FLOAT4,
    VERTEX_TYPE_HALF4,
}
PATCH_CONTROL_INDEX_ATTRIBUTE_KEYS = (
    (SEMANTIC_BONE_INDEX, 0),
    (SEMANTIC_POINT_SIZE, 0),
    (SEMANTIC_FOG, 0),
    (SEMANTIC_UV, 5),
)


def ensure_range(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise G1MParseError(f"{label} extends past the end of the file.")


@dataclass(slots=True)
class VertexBuffer:
    index: int
    stride: int
    count: int
    raw_data: bytes


@dataclass(slots=True)
class VertexAttribute:
    buffer_slot: int
    offset: int
    data_type: int
    semantic: int
    layer: int


@dataclass(slots=True)
class VertexAttributeSet:
    index: int
    vertex_buffer_indices: list[int]
    attributes: list[VertexAttribute]

    def find_attribute(self, semantic: int, layer: int | None = 0) -> VertexAttribute | None:
        for attribute in self.attributes:
            if attribute.semantic == semantic and (layer is None or attribute.layer == layer):
                return attribute
        if layer is not None:
            for attribute in self.attributes:
                if attribute.semantic == semantic:
                    return attribute
        return None

    def vertex_buffer_index_for(self, attribute: VertexAttribute) -> int:
        if not 0 <= attribute.buffer_slot < len(self.vertex_buffer_indices):
            raise G1MParseError(
                f"Vertex attribute references buffer slot {attribute.buffer_slot}, "
                f"but only {len(self.vertex_buffer_indices)} vertex buffer slots are present."
            )
        return self.vertex_buffer_indices[attribute.buffer_slot]


@dataclass(slots=True)
class IndexBuffer:
    index: int
    width_bits: int
    indices: list[int]


@dataclass(slots=True)
class PreviewVertex:
    rest_position: Vector3
    bone_ids: list[int]
    bone_weights: list[float]


@dataclass(slots=True)
class PreviewMesh:
    submesh_index: int
    body_part_indices: list[int]
    mesh_names: list[str]
    bone_ids: list[int]
    vertices: list[PreviewVertex]
    triangles: list[tuple[int, int, int]]
    is_cloth_related: bool = False

    def skinned_positions(self, bone_delta_matrices: list[list[float]]) -> list[tuple[float, float, float]]:
        positions: list[tuple[float, float, float]] = []
        for vertex in self.vertices:
            if not vertex.bone_ids:
                positions.append((vertex.rest_position.x, vertex.rest_position.y, vertex.rest_position.z))
                continue

            x = 0.0
            y = 0.0
            z = 0.0
            for bone_id, weight in zip(vertex.bone_ids, vertex.bone_weights):
                tx, ty, tz = transform_point(bone_delta_matrices[bone_id], vertex.rest_position)
                x += tx * weight
                y += ty * weight
                z += tz * weight
            positions.append((x, y, z))
        return positions


@dataclass(slots=True)
class PreviewScene:
    meshes: list[PreviewMesh]
    unsupported_submeshes: list[tuple[int, str]]
    bounds_min: Vector3
    bounds_max: Vector3

    def skinned_mesh_positions(self, model: G1MModel) -> list[list[tuple[int, int, int]] | list[tuple[float, float, float]]]:
        bone_delta_matrices = build_bone_delta_matrices(model)
        return [mesh.skinned_positions(bone_delta_matrices) for mesh in self.meshes]

    @property
    def center(self) -> Vector3:
        return Vector3(
            (self.bounds_min.x + self.bounds_max.x) * 0.5,
            (self.bounds_min.y + self.bounds_max.y) * 0.5,
            (self.bounds_min.z + self.bounds_max.z) * 0.5,
        )

    @property
    def span(self) -> Vector3:
        return Vector3(
            self.bounds_max.x - self.bounds_min.x,
            self.bounds_max.y - self.bounds_min.y,
            self.bounds_max.z - self.bounds_min.z,
        )


def build_preview_scene(model: G1MModel) -> PreviewScene:
    vertex_buffers, attribute_sets, index_buffers = parse_preview_resources(model)
    original_global_matrices = build_global_matrices(model, current=False)
    submesh_to_parts: dict[int, list[int]] = {}
    for body_part in model.body_parts:
        for submesh_index in body_part.submesh_indices:
            submesh_to_parts.setdefault(submesh_index, []).append(body_part.index)

    meshes: list[PreviewMesh] = []
    unsupported_submeshes: list[tuple[int, str]] = []

    for submesh in model.submeshes:
        try:
            meshes.append(
                build_preview_mesh(
                    model,
                    submesh.index,
                    vertex_buffers,
                    attribute_sets,
                    index_buffers,
                    original_global_matrices,
                    submesh_to_parts.get(submesh.index, []),
                )
            )
        except G1MParseError as exc:
            unsupported_submeshes.append((submesh.index, str(exc)))

    bounds_min, bounds_max = compute_bounds(meshes)
    return PreviewScene(
        meshes=meshes,
        unsupported_submeshes=unsupported_submeshes,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def parse_preview_resources(
    model: G1MModel,
) -> tuple[list[VertexBuffer], list[VertexAttributeSet], list[IndexBuffer]]:
    geometry_section = next((section for section in model.sections if section.magic == "G1MG"), None)
    if geometry_section is None:
        raise G1MParseError("Preview rendering requires a G1MG geometry section.")

    data = model.raw_data
    payload_offset = geometry_section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 36, "G1MG header")
    subsection_count = int.from_bytes(data[payload_offset + 32 : payload_offset + 36], "little", signed=True)
    try:
        g1mg_version = int(geometry_section.version)
    except ValueError:
        g1mg_version = 0

    vertex_buffers: list[VertexBuffer] = []
    attribute_sets: list[VertexAttributeSet] = []
    index_buffers: list[IndexBuffer] = []

    cursor = payload_offset + 36
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, G1MG_SUBSECTION_HEADER_STRUCT.size, f"G1MG subsection {subsection_index}")
        subsection_type, unknown, size = G1MG_SUBSECTION_HEADER_STRUCT.unpack_from(data, cursor)
        ensure_range(data, cursor, size, f"G1MG subsection {subsection_index}")
        payload = cursor + G1MG_SUBSECTION_HEADER_STRUCT.size
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
                    VertexBuffer(
                        index=len(vertex_buffers),
                        stride=stride,
                        count=vertex_count,
                        raw_data=data[payload : payload + data_size],
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
            index_buffers = parse_preview_index_buffers(data, payload, count, cursor + size, g1mg_version)

        cursor += size

    return vertex_buffers, attribute_sets, index_buffers



def parse_preview_index_buffers(
    data: bytes,
    payload: int,
    count: int,
    subsection_end: int,
    g1mg_version: int = 0,
) -> list[IndexBuffer]:
    def parse_with(header_size: int) -> list[IndexBuffer] | None:
        cursor = payload
        buffers: list[IndexBuffer] = []
        for index in range(count):
            ensure_range(data, cursor, header_size, "index buffer header")
            if header_size == 8:
                index_count, width_bits = Struct("<2I").unpack_from(data, cursor)
            elif header_size == 12:
                index_count, width_bits, unknown0 = Struct("<3I").unpack_from(data, cursor)
            else:
                flags, index_count, width_bits, unknown0 = Struct("<4I").unpack_from(data, cursor)
            if width_bits not in {8, 16, 32}:
                return None
            if width_bits == 8:
                byte_width = 1
                fmt = f"<{index_count}B"
            elif width_bits == 16:
                byte_width = 2
                fmt = f"<{index_count}H"
            else:
                byte_width = 4
                fmt = f"<{index_count}I"
            cursor += header_size
            ensure_range(data, cursor, int(index_count) * byte_width, "index buffer data")
            indices = list(Struct(fmt).unpack_from(data, cursor) if index_count else ())
            cursor += int(index_count) * byte_width
            cursor = (cursor + 3) & ~3
            buffers.append(IndexBuffer(index=index, width_bits=int(width_bits), indices=indices))
        if cursor != subsection_end:
            return None
        return buffers

    header_order = (12, 8, 16) if g1mg_version > 40 else (8, 12, 16)
    for header_size in header_order:
        parsed = parse_with(header_size)
        if parsed is not None:
            return parsed
    raise G1MParseError("Unable to parse G1MG index buffers as 12-, 8-, or 16-byte headers.")


def build_preview_mesh(
    model: G1MModel,
    submesh_index: int,
    vertex_buffers: list[VertexBuffer],
    attribute_sets: list[VertexAttributeSet],
    index_buffers: list[IndexBuffer],
    original_global_matrices: list[list[float]],
    body_part_indices: list[int],
) -> PreviewMesh:
    submesh = model.submeshes[submesh_index]
    if not 0 <= submesh.vbo_index < len(attribute_sets):
        raise G1MParseError(f"Submesh {submesh.index} references missing vertex attribute set {submesh.vbo_index}.")
    if not 0 <= submesh.buffer_index < len(index_buffers):
        raise G1MParseError(f"Submesh {submesh.index} references missing index buffer {submesh.buffer_index}.")

    attribute_set = attribute_sets[submesh.vbo_index]
    position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
    if position_attribute is None:
        raise G1MParseError(f"Submesh {submesh.index} has no position attribute.")
    if position_attribute.data_type not in SUPPORTED_POSITION_TYPES:
        raise G1MParseError(
            f"Submesh {submesh.index} uses unsupported position type 0x{position_attribute.data_type:04X}."
        )

    weight_attribute = attribute_set.find_attribute(SEMANTIC_BONE_WEIGHT)
    bone_index_attribute = attribute_set.find_attribute(SEMANTIC_BONE_INDEX)

    if weight_attribute and weight_attribute.data_type not in SUPPORTED_WEIGHT_TYPES:
        weight_attribute = None
    if bone_index_attribute and bone_index_attribute.data_type not in SUPPORTED_INDEX_TYPES:
        bone_index_attribute = None
    is_cloth_layout = looks_like_cloth_layout(attribute_set)
    cloth_control_index_attributes = resolve_cloth_control_index_attributes(attribute_set, bone_index_attribute)
    uses_cloth_patch_layout = len(cloth_control_index_attributes) > 1 and weight_attribute is not None

    bind_set = (
        model.bone_bind_sets[submesh.bone_table_index]
        if 0 <= submesh.bone_table_index < len(model.bone_bind_sets)
        else None
    )
    cloth_entry = resolve_cloth_entry(model, body_part_indices, bind_set, submesh, is_cloth_layout)
    cloth_anchor_bone_id = (
        cloth_entry.parent_bone_id
        if cloth_entry is not None and 0 <= cloth_entry.parent_bone_id < len(model.bones)
        else resolve_cloth_anchor_bone_id(model, bind_set, submesh.bone_index, is_cloth_layout)
    )
    uses_weighted_cloth_bind_space = bool(
        is_cloth_layout or (cloth_entry is not None and cloth_entry.subsection_type == 0x00030002)
    )
    bind_space_transforms = resolve_bind_space_transforms(
        model,
        bind_set,
        original_global_matrices,
        uses_weighted_cloth_bind_space,
    )
    cloth_control_positions = resolve_cloth_control_positions(
        model,
        cloth_entry,
        cloth_control_index_attributes,
        vertex_buffers,
        attribute_set,
        submesh,
        original_global_matrices,
    )
    cloth_patch_shell_offsets = resolve_cloth_patch_shell_offsets(
        cloth_entry,
        cloth_control_index_attributes,
        vertex_buffers,
        attribute_set,
        submesh,
        weight_attribute,
        original_global_matrices,
    )
    cloth_constant_buffer = prepare_cloth_constant_buffer(cloth_entry, is_cloth_layout)
    rigid_bone_id = resolve_rigid_bone_id(model, submesh.index)

    vertices: list[PreviewVertex] = []
    for local_vertex_index in range(submesh.vertex_count):
        global_vertex_index = submesh.vertex_offset + local_vertex_index
        position_components = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)],
            position_attribute,
            global_vertex_index,
        )
        position = Vector3(
            float(position_components[0]),
            float(position_components[1]),
            float(position_components[2]),
        )
        if cloth_control_positions is not None and bone_index_attribute is not None:
            cloth_position = None
            if uses_cloth_patch_layout:
                cloth_position = resolve_cloth_patch_vertex_position(
                    cloth_control_positions,
                    cloth_control_index_attributes,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                )
            else:
                cloth_position = resolve_cloth_control_vertex_position(
                    cloth_control_positions,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
            if cloth_position is not None:
                position = cloth_position
                shell_offset = cloth_patch_shell_offsets.get(global_vertex_index)
                if shell_offset is not None:
                    position = position.add(shell_offset)
                if cloth_anchor_bone_id is not None:
                    bone_ids = [cloth_anchor_bone_id]
                    bone_weights = [1.0]
                else:
                    bone_ids = []
                    bone_weights = []
            elif cloth_anchor_bone_id is not None:
                x, y, z = transform_point(original_global_matrices[cloth_anchor_bone_id], position)
                position = Vector3(x, y, z)
                bone_ids = [cloth_anchor_bone_id]
                bone_weights = [1.0]
            else:
                bone_ids, bone_weights = resolve_vertex_influences(
                    model,
                    bind_set,
                    rigid_bone_id,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
        elif cloth_constant_buffer is not None:
            cloth_shader_position = resolve_cloth_shader_vertex_position(
                cloth_constant_buffer,
                vertex_buffers,
                attribute_set,
                global_vertex_index,
            )
            if cloth_shader_position is not None:
                position = cloth_shader_position
                if cloth_anchor_bone_id is not None:
                    bone_ids = [cloth_anchor_bone_id]
                    bone_weights = [1.0]
                else:
                    bone_ids = []
                    bone_weights = []
            elif bind_space_transforms is not None and weight_attribute is not None and bone_index_attribute is not None:
                if cloth_entry is not None and cloth_entry.subsection_type == 0x00030002 and cloth_anchor_bone_id is not None:
                    x, y, z = transform_point(original_global_matrices[cloth_anchor_bone_id], position)
                    position = Vector3(x, y, z)
                    bone_ids, bone_weights = resolve_vertex_influences(
                        model,
                        bind_set,
                        rigid_bone_id,
                        vertex_buffers,
                        attribute_set,
                        global_vertex_index,
                        weight_attribute,
                        bone_index_attribute,
                    )
                else:
                    bind_space_position = resolve_bind_space_vertex_position(
                        bind_space_transforms,
                        vertex_buffers,
                        attribute_set,
                        global_vertex_index,
                        weight_attribute,
                        bone_index_attribute,
                    )
                    if bind_space_position is not None:
                        position = bind_space_position
                    bone_ids, bone_weights = resolve_vertex_influences(
                        model,
                        bind_set,
                        rigid_bone_id,
                        vertex_buffers,
                        attribute_set,
                        global_vertex_index,
                        weight_attribute,
                        bone_index_attribute,
                    )
            elif cloth_anchor_bone_id is not None:
                x, y, z = transform_point(original_global_matrices[cloth_anchor_bone_id], position)
                position = Vector3(x, y, z)
                bone_ids = [cloth_anchor_bone_id]
                bone_weights = [1.0]
            else:
                bone_ids, bone_weights = resolve_vertex_influences(
                    model,
                    bind_set,
                    rigid_bone_id,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
        elif bind_space_transforms is not None and weight_attribute is not None and bone_index_attribute is not None:
            if cloth_entry is not None and cloth_entry.subsection_type == 0x00030002 and cloth_anchor_bone_id is not None:
                x, y, z = transform_point(original_global_matrices[cloth_anchor_bone_id], position)
                position = Vector3(x, y, z)
                bone_ids, bone_weights = resolve_vertex_influences(
                    model,
                    bind_set,
                    rigid_bone_id,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
            else:
                bind_space_position = resolve_bind_space_vertex_position(
                    bind_space_transforms,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
                if bind_space_position is not None:
                    position = bind_space_position
                bone_ids, bone_weights = resolve_vertex_influences(
                    model,
                    bind_set,
                    rigid_bone_id,
                    vertex_buffers,
                    attribute_set,
                    global_vertex_index,
                    weight_attribute,
                    bone_index_attribute,
                )
        elif cloth_anchor_bone_id is not None:
            x, y, z = transform_point(original_global_matrices[cloth_anchor_bone_id], position)
            position = Vector3(x, y, z)
            bone_ids = [cloth_anchor_bone_id]
            bone_weights = [1.0]
        else:
            bone_ids, bone_weights = resolve_vertex_influences(
                model,
                bind_set,
                rigid_bone_id,
                vertex_buffers,
                attribute_set,
                global_vertex_index,
                weight_attribute,
                bone_index_attribute,
            )
        vertices.append(PreviewVertex(rest_position=position, bone_ids=bone_ids, bone_weights=bone_weights))

    index_buffer = index_buffers[submesh.buffer_index]
    raw_indices = index_buffer.indices[submesh.face_offset : submesh.face_offset + submesh.face_count]
    triangles = triangulate_submesh_indices(
        raw_indices,
        submesh.face_type,
        submesh.vertex_offset,
        submesh.vertex_count,
        0xFFFF if index_buffer.width_bits == 16 else 0xFFFFFFFF,
    )
    if not triangles:
        raise G1MParseError(f"Submesh {submesh.index} did not produce any preview triangles.")

    mesh_names = [model.body_parts[part_index].name for part_index in body_part_indices]
    if not mesh_names:
        mesh_names = [f"Submesh {submesh.index}"]

    bone_ids = {bone_id for vertex in vertices for bone_id in vertex.bone_ids}
    if bind_set is not None:
        bone_ids.update(
            bind.bone_id
            for bind in bind_set.binds
            if 0 <= bind.bone_id < len(model.bones)
        )
        bone_ids.update(
            bind.reference_bone_id
            for bind in bind_set.binds
            if 0 <= bind.reference_bone_id < len(model.bones)
        )
    if cloth_entry is not None:
        if 0 <= cloth_entry.parent_bone_id < len(model.bones):
            bone_ids.add(cloth_entry.parent_bone_id)
        bone_ids.update(
            physics_bone.bone_id
            for physics_bone in cloth_entry.physics_bones
            if 0 <= physics_bone.bone_id < len(model.bones)
        )
    body_parts = [
        model.body_parts[part_index]
        for part_index in body_part_indices
        if 0 <= part_index < len(model.body_parts)
    ]
    is_cloth_related = bool(
        cloth_entry is not None
        or is_cloth_layout
        or uses_cloth_patch_layout
        or any((part.cloth_type_id & 0xF) != 0 or model.cloth_entry_for_body_part(part) is not None for part in body_parts)
    )
    return PreviewMesh(
        submesh_index=submesh.index,
        body_part_indices=body_part_indices,
        mesh_names=mesh_names,
        bone_ids=sorted(bone_ids),
        vertices=vertices,
        triangles=triangles,
        is_cloth_related=is_cloth_related,
    )


def resolve_rigid_bone_id(model: G1MModel, submesh_index: int) -> int | None:
    submesh = model.submeshes[submesh_index]
    if 0 <= submesh.bone_table_index < len(model.bone_bind_sets):
        bind_set = model.bone_bind_sets[submesh.bone_table_index]
        reference_bone_id = bind_set.reference_bone_id
        if reference_bone_id is not None and 0 <= reference_bone_id < len(model.bones):
            return reference_bone_id
        if len(bind_set.binds) == 1:
            bone_id = bind_set.binds[0].bone_id
            if 0 <= bone_id < len(model.bones):
                return bone_id

    if 0 <= submesh.bone_index < len(model.bones):
        return submesh.bone_index
    return None


def resolve_cloth_entry(
    model: G1MModel,
    body_part_indices: list[int],
    bind_set,
    submesh,
    is_cloth_layout: bool,
):
    if is_cloth_layout:
        cloth_entry = resolve_cloth_entry_from_joint_map(model, bind_set, submesh.bone_table_index)
        if cloth_entry is not None:
            return cloth_entry

    for part_index in body_part_indices:
        if not 0 <= part_index < len(model.body_parts):
            continue
        cloth_entry = model.cloth_entry_for_body_part(model.body_parts[part_index])
        if cloth_entry is not None:
            return cloth_entry
    return None


def resolve_cloth_entry_from_joint_map(model: G1MModel, bind_set, joint_map_index: int):
    if bind_set is None or joint_map_index < 0:
        return None

    signature = bind_set.joint_map_signature
    direct_match = None
    fallback_match = None
    for entry in iter_cloth_entries(model):
        if entry.joint_map_index is None:
            continue
        if signature is not None and entry.association_signature is not None and entry.association_signature != signature:
            continue
        if entry.joint_map_index == joint_map_index and direct_match is None:
            direct_match = entry
        elif entry.joint_map_index == joint_map_index + 1 and fallback_match is None:
            fallback_match = entry

    return direct_match or fallback_match


def iter_cloth_entries(model: G1MModel):
    for entry_map in (
        model.cloth_library.nuno_entries,
        model.cloth_library.nunv_entries,
        model.cloth_library.nuns_entries,
    ):
        for entries in entry_map.values():
            for entry in entries:
                yield entry


def prepare_cloth_constant_buffer(cloth_entry, is_cloth_layout: bool):
    if (
        not is_cloth_layout
        or cloth_entry is None
        or not cloth_entry.control_points
        or cloth_entry.subsection_type == 0x00030005
    ):
        return None
    return make_cloth_constant_buffer(cloth_entry.control_points)


def resolve_effective_cloth_sources(model: G1MModel, cloth_entry):
    control_points = cloth_entry.control_points
    influences = cloth_entry.influences

    if cloth_entry.subsection_type != 0x00030005:
        return control_points, influences
    if cloth_entry.subset_parent_entry_index is None:
        return control_points, influences

    nuno5_entries = model.cloth_library.nuno_entries.get(0x00030005, [])
    if not 0 <= cloth_entry.subset_parent_entry_index < len(nuno5_entries):
        return control_points, influences

    parent_entry = nuno5_entries[cloth_entry.subset_parent_entry_index]
    mapping = cloth_entry.subset_control_point_map or ()
    if not mapping:
        return control_points, influences

    remapped_points: list[tuple[float, float, float, float]] = []
    remapped_influences = []
    for index, point in enumerate(control_points):
        parent_index = mapping[index] if index < len(mapping) else -1
        if 0 <= parent_index < len(parent_entry.control_points):
            remapped_points.append(parent_entry.control_points[parent_index])
            if 0 <= parent_index < len(parent_entry.influences):
                remapped_influences.append(parent_entry.influences[parent_index])
            elif index < len(influences):
                remapped_influences.append(influences[index])
        else:
            remapped_points.append(point)
            if index < len(influences):
                remapped_influences.append(influences[index])

    return remapped_points, remapped_influences

def resolve_cloth_shader_vertex_position(
    cloth_constant_buffer,
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
) -> Vector3 | None:
    shader_input = build_cloth_shader_input(vertex_buffers, attribute_set, global_vertex_index)
    if shader_input is None:
        return None
    shader_output = transform_cloth_vertex2(shader_input, cloth_constant_buffer)
    if shader_output is None:
        return None
    return Vector3(shader_output.o0.x, shader_output.o0.y, shader_output.o0.z)

def build_cloth_shader_input(
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
) -> ClothShaderInput | None:
    position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
    weight_attribute = attribute_set.find_attribute(SEMANTIC_BONE_WEIGHT)
    bone_index_attribute = attribute_set.find_attribute(SEMANTIC_BONE_INDEX)
    if position_attribute is None or weight_attribute is None or bone_index_attribute is None:
        return None

    def read(attribute: VertexAttribute | None) -> tuple[float, ...] | None:
        if attribute is None:
            return None
        try:
            buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(attribute)]
            values = read_attribute_components(buffer, attribute, global_vertex_index)
        except (G1MParseError, IndexError, struct.error):
            return None
        return tuple(float(value) for value in values)

    return ClothShaderInput(
        v0=float4_position(read(position_attribute)),
        v1=float4_blend_weights(read(weight_attribute)),
        v2=float4_vector(read(attribute_set.find_attribute(SEMANTIC_BINORMAL))),
        v3=float4_colorish(read(attribute_set.find_attribute(SEMANTIC_COLOR, 1))),
        v4=float4_indices(read(bone_index_attribute)),
        v5=float4_indices(read(attribute_set.find_attribute(SEMANTIC_POINT_SIZE))),
        v6=float4_indices(read(attribute_set.find_attribute(SEMANTIC_FOG))),
        v7=float4_indices(read(attribute_set.find_attribute(SEMANTIC_UV, 5))),
        v8=float4_vector(read(attribute_set.find_attribute(SEMANTIC_NORMAL))),
        v9=float4_vector(read(attribute_set.find_attribute(SEMANTIC_TANGENT))),
        v10=float4_uv(read(attribute_set.find_attribute(SEMANTIC_UV, 0))),
        v11=float4_uv(read(attribute_set.find_attribute(SEMANTIC_UV, 1))),
        v12=float4_uv(read(attribute_set.find_attribute(SEMANTIC_UV, 2))),
    )


def float4_position(values: tuple[float, ...] | None) -> Float4:
    if not values:
        return Float4()
    if len(values) >= 4:
        return Float4(values[0], values[1], values[2], values[3])
    if len(values) >= 3:
        return Float4(values[0], values[1], values[2], 1.0)
    return Float4(values[0], values[1] if len(values) > 1 else 0.0, 0.0, 1.0)


def float4_vector(values: tuple[float, ...] | None) -> Float4:
    if not values:
        return Float4()
    if len(values) >= 4:
        return Float4(values[0], values[1], values[2], values[3])
    if len(values) >= 3:
        return Float4(values[0], values[1], values[2], 1.0)
    if len(values) == 2:
        return Float4(values[0], values[1], 0.0, 1.0)
    return Float4(values[0], 0.0, 0.0, 1.0)


def float4_blend_weights(values: tuple[float, ...] | None) -> Float4:
    if not values:
        return Float4()
    expanded = expand_weights(values)
    if len(expanded) < 4:
        expanded.extend([0.0] * (4 - len(expanded)))
    return Float4(expanded[0], expanded[1], expanded[2], expanded[3])


def float4_colorish(values: tuple[float, ...] | None) -> Float4:
    if not values:
        return Float4()
    if len(values) >= 4:
        return Float4(values[0], values[1], values[2], values[3])
    if len(values) == 3:
        return Float4(values[0], values[1], values[2], 1.0)
    if len(values) == 2:
        return Float4(values[0], values[1], 0.0, 1.0)
    return Float4(values[0], 0.0, 0.0, 1.0)


def float4_indices(values: tuple[float, ...] | None) -> Float4:
    if not values:
        return Float4()
    padded = list(values[:4])
    padded.extend([0.0] * (4 - len(padded)))
    return Float4(padded[0], padded[1], padded[2], padded[3])


def float4_uv(values: tuple[float, ...] | None) -> Float4:
    if not values:
        return Float4()
    padded = list(values[:2])
    padded.extend([0.0] * (2 - len(padded)))
    return Float4(padded[0], padded[1], 0.0, 0.0)


def resolve_cloth_control_index_attributes(
    attribute_set: VertexAttributeSet,
    bone_index_attribute: VertexAttribute | None,
) -> list[VertexAttribute]:
    attributes: list[VertexAttribute] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    def append(attribute: VertexAttribute | None) -> None:
        if attribute is None or attribute.data_type not in {VERTEX_TYPE_BYTE4, VERTEX_TYPE_USHORT4, VERTEX_TYPE_UINT4}:
            return
        key = (
            attribute.buffer_slot,
            attribute.offset,
            attribute.data_type,
            attribute.semantic,
            attribute.layer,
        )
        if key in seen:
            return
        seen.add(key)
        attributes.append(attribute)

    append(bone_index_attribute)
    for semantic, layer in PATCH_CONTROL_INDEX_ATTRIBUTE_KEYS:
        append(attribute_set.find_attribute(semantic, layer))
    return attributes


def looks_like_cloth_layout(attribute_set: VertexAttributeSet) -> bool:
    blend_weight_semantics = {0x0100, 0x0600, 0x0700, 0x0A01}
    blend_index_semantics = {0x0200, 0x0400, 0x0505, 0x0B00}

    weight_count = 0
    index_count = 0
    for attribute in attribute_set.attributes:
        semantic_code = (attribute.semantic << 8) | (attribute.layer & 0xFF)
        if semantic_code in blend_weight_semantics:
            weight_count += 1
        elif semantic_code in blend_index_semantics and attribute.data_type in {VERTEX_TYPE_BYTE4, VERTEX_TYPE_USHORT4}:
            index_count += 1

    return weight_count == 4 and index_count == 4


def resolve_cloth_anchor_bone_id(model: G1MModel, bind_set, fallback_bone_id: int, is_cloth_layout: bool) -> int | None:
    if bind_set is None or not is_cloth_layout:
        return None

    reference_bone_id = bind_set.reference_bone_id
    if reference_bone_id is not None and 0 <= reference_bone_id < len(model.bones):
        return reference_bone_id

    for bind in bind_set.binds:
        if 0 <= bind.bone_id < len(model.bones):
            return bind.bone_id

    if 0 <= fallback_bone_id < len(model.bones):
        return fallback_bone_id
    return None

def resolve_bind_space_transforms(
    model: G1MModel,
    bind_set,
    original_global_matrices: list[list[float]],
    uses_weighted_cloth_bind_space: bool,
) -> list[list[float]] | None:
    if bind_set is None or not bind_set.binds or not uses_weighted_cloth_bind_space:
        return None
    if not model.bind_matrices:
        return None

    bind_space_transforms: list[list[float]] = []
    for bind in bind_set.binds:
        if not 0 <= bind.bone_id < len(original_global_matrices):
            return None

        bone_matrix = original_global_matrices[bind.bone_id]
        bind_matrix = (
            model.bind_matrices[bind.matrix_id]
            if 0 <= bind.matrix_id < len(model.bind_matrices)
            else None
        )
        bind_space_transforms.append(
            compose_bind_space_matrix(bone_matrix, bind_matrix)
        )
    return bind_space_transforms


def resolve_cloth_control_positions(
    model: G1MModel,
    cloth_entry,
    control_index_attributes: list[VertexAttribute],
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    submesh,
    original_global_matrices: list[list[float]],
) -> list[Vector3] | None:
    if cloth_entry is None or cloth_entry.subsection_type not in {0x00030001, 0x00030003, 0x00030005, 0x00050001, 0x00060001}:
        return None
    if not control_index_attributes or not cloth_entry.control_points:
        return None

    source_control_points, source_influences = resolve_effective_cloth_sources(model, cloth_entry)
    if not source_control_points:
        return None

    max_raw_index = -1
    for index_attribute in control_index_attributes:
        index_buffer = vertex_buffers[attribute_set.vertex_buffer_index_for(index_attribute)]
        for local_vertex_index in range(submesh.vertex_count):
            global_vertex_index = submesh.vertex_offset + local_vertex_index
            raw_indices = read_attribute_components(index_buffer, index_attribute, global_vertex_index)
            max_raw_index = max(max_raw_index, *(int(value) for value in raw_indices))

    if max_raw_index < 0:
        return None

    control_points = choose_cloth_control_points(
        source_control_points,
        source_influences,
        max_raw_index=max_raw_index,
    )
    if control_points is None:
        return None

    if 0 <= cloth_entry.parent_bone_id < len(original_global_matrices):
        parent_matrix = original_global_matrices[cloth_entry.parent_bone_id]
        transformed_points: list[Vector3] = []
        for index, point in enumerate(control_points):
            point_vector = Vector3(point[0], point[1], point[2])
            influence = source_influences[index] if index < len(source_influences) else None
            if cloth_entry.subsection_type == 0x00030005 and influence is not None and influence.extra_value == 0:
                transformed_points.append(point_vector)
            else:
                transformed_points.append(Vector3(*transform_point(parent_matrix, point_vector)))
        return transformed_points

    return [Vector3(point[0], point[1], point[2]) for point in control_points]

def choose_cloth_control_points(
    source_control_points: list[tuple[float, float, float, float]],
    source_influences,
    *,
    max_raw_index: int,
) -> list[tuple[float, float, float]] | None:
    control_points = [(point[0], point[1], point[2]) for point in source_control_points]
    if not control_points:
        return None

    candidates: list[tuple[int, int, list[tuple[float, float, float]]]] = []
    if max_raw_index < len(control_points):
        candidates.append((len(control_points), 1, control_points))

    width = infer_cloth_control_width(source_control_points, source_influences)
    if width is not None:
        for row_group_size in (2, 4):
            candidate = average_cloth_control_rows(control_points, width, row_group_size)
            if candidate is None or max_raw_index >= len(candidate):
                continue
            candidates.append((len(candidate), row_group_size, candidate))

    if not candidates:
        return None

    _, _, candidate = min(candidates)
    return candidate

def infer_cloth_control_width(
    source_control_points: list[tuple[float, float, float, float]],
    source_influences,
) -> int | None:
    if not source_influences:
        return None

    width_counts: Counter[int] = Counter()
    for index, influence in enumerate(source_influences):
        for neighbor in influence.neighbors:
            if neighbor < 0:
                continue
            difference = abs(neighbor - index)
            if difference > 1:
                width_counts[difference] += 1

    if not width_counts:
        return None

    width, _count = min(
        width_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if width <= 0 or width > len(source_control_points):
        return None
    return width

def average_cloth_control_rows(
    control_points: list[tuple[float, float, float]],
    width: int,
    row_group_size: int,
) -> list[tuple[float, float, float]] | None:
    row_count = len(control_points) // width
    reduced_row_count = row_count // row_group_size
    if reduced_row_count <= 0:
        return None

    averaged: list[tuple[float, float, float]] = []
    for group_index in range(reduced_row_count):
        first_row = (group_index * row_count) // reduced_row_count
        last_row = ((group_index + 1) * row_count) // reduced_row_count
        if last_row <= first_row:
            last_row = first_row + 1
        span = last_row - first_row
        for column in range(width):
            x = 0.0
            y = 0.0
            z = 0.0
            for row_index in range(first_row, last_row):
                point = control_points[row_index * width + column]
                x += point[0]
                y += point[1]
                z += point[2]
            averaged.append((x / span, y / span, z / span))
    return averaged

def resolve_cloth_patch_shell_offsets(
    cloth_entry,
    control_index_attributes: list[VertexAttribute],
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    submesh,
    weight_attribute: VertexAttribute | None,
    original_global_matrices: list[list[float]],
) -> dict[int, Vector3]:
    if cloth_entry is None or weight_attribute is None or len(control_index_attributes) < 2:
        return {}

    normal_attribute = attribute_set.find_attribute(SEMANTIC_NORMAL)
    shell_attribute = attribute_set.find_attribute(SEMANTIC_TANGENT)
    if normal_attribute is None or shell_attribute is None:
        return {}
    parent_matrix = (
        original_global_matrices[cloth_entry.parent_bone_id]
        if 0 <= cloth_entry.parent_bone_id < len(original_global_matrices)
        else None
    )
    grouped_vertices: dict[tuple[object, ...], list[int]] = {}
    shell_offsets: dict[int, Vector3] = {}

    for local_vertex_index in range(submesh.vertex_count):
        global_vertex_index = submesh.vertex_offset + local_vertex_index
        signature = resolve_cloth_patch_signature(
            control_index_attributes,
            vertex_buffers,
            attribute_set,
            global_vertex_index,
            weight_attribute,
        )
        if signature is None:
            continue

        normal_components = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(normal_attribute)],
            normal_attribute,
            global_vertex_index,
        )
        shell_components = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(shell_attribute)],
            shell_attribute,
            global_vertex_index,
        )
        shell_distance = math.sqrt(
            float(shell_components[0]) * float(shell_components[0])
            + float(shell_components[1]) * float(shell_components[1])
            + float(shell_components[2]) * float(shell_components[2])
        )
        if shell_distance <= 1e-5:
            continue

        local_normal = Vector3(
            float(normal_components[0]),
            float(normal_components[1]),
            float(normal_components[2]),
        )
        transformed_normal = (
            transform_direction(parent_matrix, local_normal)
            if parent_matrix is not None
            else local_normal
        )
        normalized_normal = normalize_vector3(transformed_normal)
        if normalized_normal is None:
            continue

        grouped_vertices.setdefault(signature, []).append(global_vertex_index)
        shell_offsets[global_vertex_index] = Vector3(
            normalized_normal.x * shell_distance,
            normalized_normal.y * shell_distance,
            normalized_normal.z * shell_distance,
        )

    return {
        global_vertex_index: shell_offsets[global_vertex_index]
        for group in grouped_vertices.values()
        if len(group) > 1
        for global_vertex_index in group
    }

def resolve_cloth_patch_signature(
    control_index_attributes: list[VertexAttribute],
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
    weight_attribute: VertexAttribute | None,
) -> tuple[object, ...] | None:
    if weight_attribute is None or len(control_index_attributes) < 2:
        return None

    position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
    if position_attribute is None:
        return None

    column_components = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)],
        position_attribute,
        global_vertex_index,
    )
    row_components = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(weight_attribute)],
        weight_attribute,
        global_vertex_index,
    )
    if not looks_like_patch_basis_weights(column_components):
        return None
    if not looks_like_patch_basis_weights(row_components):
        return None

    column_weights = tuple(round(float(value), 6) for value in column_components[:4])
    row_weights = tuple(round(float(value), 6) for value in row_components[: len(control_index_attributes)])
    control_rows = tuple(
        tuple(
            int(value)
            for value in read_attribute_components(
                vertex_buffers[attribute_set.vertex_buffer_index_for(index_attribute)],
                index_attribute,
                global_vertex_index,
            )
        )
        for index_attribute in control_index_attributes
    )
    return (column_weights, row_weights, control_rows)

def resolve_cloth_patch_vertex_position(
    cloth_control_positions: list[Vector3],
    control_index_attributes: list[VertexAttribute],
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
    weight_attribute: VertexAttribute | None,
) -> Vector3 | None:
    if weight_attribute is None or len(control_index_attributes) < 2:
        return None

    position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
    if position_attribute is None:
        return None

    column_components = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)],
        position_attribute,
        global_vertex_index,
    )
    row_components = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(weight_attribute)],
        weight_attribute,
        global_vertex_index,
    )
    if not looks_like_patch_basis_weights(column_components):
        return None
    if not looks_like_patch_basis_weights(row_components):
        return None

    column_weights = [float(value) for value in column_components[:4]]
    row_weights = [float(value) for value in row_components[: len(control_index_attributes)]]
    x = 0.0
    y = 0.0
    z = 0.0
    total_weight = 0.0
    for row_weight, index_attribute in zip(row_weights, control_index_attributes):
        if abs(row_weight) <= 1e-5:
            continue
        row_indices = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(index_attribute)],
            index_attribute,
            global_vertex_index,
        )
        for column_weight, raw_index in zip(column_weights, row_indices):
            if abs(column_weight) <= 1e-5:
                continue
            control_index = int(raw_index)
            if not 0 <= control_index < len(cloth_control_positions):
                continue
            weight = row_weight * column_weight
            control_position = cloth_control_positions[control_index]
            x += control_position.x * weight
            y += control_position.y * weight
            z += control_position.z * weight
            total_weight += weight

    if abs(total_weight) <= 1e-5:
        return None
    return Vector3(x, y, z)

def resolve_cloth_control_vertex_position(
    cloth_control_positions: list[Vector3],
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
    weight_attribute: VertexAttribute | None,
    bone_index_attribute: VertexAttribute,
) -> Vector3 | None:
    raw_indices = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(bone_index_attribute)],
        bone_index_attribute,
        global_vertex_index,
    )
    if not raw_indices:
        return None

    raw_weights = resolve_cloth_control_weights(
        vertex_buffers,
        attribute_set,
        global_vertex_index,
        weight_attribute,
        len(raw_indices),
    )
    if not raw_weights:
        return None

    x = 0.0
    y = 0.0
    z = 0.0
    total_weight = 0.0
    for raw_index, raw_weight in zip(raw_indices, raw_weights):
        if abs(raw_weight) <= 1e-5:
            continue
        control_index = int(raw_index)
        if not 0 <= control_index < len(cloth_control_positions):
            continue
        control_position = cloth_control_positions[control_index]
        x += control_position.x * raw_weight
        y += control_position.y * raw_weight
        z += control_position.z * raw_weight
        total_weight += raw_weight

    if abs(total_weight) <= 1e-5:
        return None
    return Vector3(x, y, z)

def resolve_cloth_control_weights(
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
    weight_attribute: VertexAttribute | None,
    component_count: int,
) -> list[float]:
    position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
    if position_attribute is not None:
        position_components = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)],
            position_attribute,
            global_vertex_index,
        )
        if looks_like_cloth_control_weights(position_components):
            return [float(value) for value in position_components[:component_count]]

    if weight_attribute is None:
        return []

    weight_components = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(weight_attribute)],
        weight_attribute,
        global_vertex_index,
    )
    raw_weights = [float(value) for value in weight_components]
    if len(raw_weights) == 1:
        raw_weights.append(max(0.0, 1.0 - raw_weights[0]))
    elif len(raw_weights) == 2:
        raw_weights.append(max(0.0, 1.0 - raw_weights[0] - raw_weights[1]))
    elif len(raw_weights) == 3:
        raw_weights.append(max(0.0, 1.0 - raw_weights[0] - raw_weights[1] - raw_weights[2]))
    return raw_weights[:component_count]

def looks_like_cloth_control_weights(components: tuple[float, ...] | tuple[int, ...]) -> bool:
    if len(components) < 3:
        return False

    values = [float(value) for value in components[:4]]
    total = sum(values)
    max_abs = max(abs(value) for value in values)
    non_zero = sum(1 for value in values if abs(value) > 1e-5)
    return non_zero >= 2 and abs(total - 1.0) <= 0.05 and max_abs <= 1.5

def looks_like_patch_basis_weights(components: tuple[float, ...] | tuple[int, ...]) -> bool:
    if len(components) < 2:
        return False

    values = [float(value) for value in components[:4]]
    total = sum(values)
    max_abs = max(abs(value) for value in values)
    non_zero = sum(1 for value in values if abs(value) > 1e-5)
    return non_zero >= 1 and abs(total - 1.0) <= 0.05 and max_abs <= 1.5

def resolve_bind_space_vertex_position(
    bind_space_transforms: list[list[float]],
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
    weight_attribute: VertexAttribute,
    bone_index_attribute: VertexAttribute,
) -> Vector3 | None:
    raw_weights = expand_weights(
        read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(weight_attribute)],
            weight_attribute,
            global_vertex_index,
        )
    )
    raw_indices = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(bone_index_attribute)],
        bone_index_attribute,
        global_vertex_index,
    )
    position_attribute = attribute_set.find_attribute(SEMANTIC_POSITION)
    if position_attribute is None:
        return None
    position_components = read_attribute_components(
        vertex_buffers[attribute_set.vertex_buffer_index_for(position_attribute)],
        position_attribute,
        global_vertex_index,
    )
    position = Vector3(
        float(position_components[0]),
        float(position_components[1]),
        float(position_components[2]),
    )

    x = 0.0
    y = 0.0
    z = 0.0
    total_weight = 0.0
    for raw_index, raw_weight in zip(raw_indices, raw_weights):
        if raw_weight <= 1e-5:
            continue
        bind_index = int(raw_index) // 3
        if not 0 <= bind_index < len(bind_space_transforms):
            continue
        tx, ty, tz = transform_point(bind_space_transforms[bind_index], position)
        x += tx * raw_weight
        y += ty * raw_weight
        z += tz * raw_weight
        total_weight += raw_weight

    if total_weight <= 1e-5:
        return None
    return Vector3(x / total_weight, y / total_weight, z / total_weight)

def resolve_vertex_influences(
    model: G1MModel,
    bind_set,
    rigid_bone_id: int | None,
    vertex_buffers: list[VertexBuffer],
    attribute_set: VertexAttributeSet,
    global_vertex_index: int,
    weight_attribute: VertexAttribute | None,
    bone_index_attribute: VertexAttribute | None,
) -> tuple[list[int], list[float]]:
    bone_ids: list[int] = []
    bone_weights: list[float] = []

    raw_weights: list[float] = []
    if weight_attribute is not None:
        weight_components = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(weight_attribute)],
            weight_attribute,
            global_vertex_index,
        )
        raw_weights = expand_weights(weight_components)
        if any(weight < -0.0001 for weight in raw_weights):
            raw_weights = []

    raw_indices: list[int] = []
    if bone_index_attribute is not None:
        bone_index_components = read_attribute_components(
            vertex_buffers[attribute_set.vertex_buffer_index_for(bone_index_attribute)],
            bone_index_attribute,
            global_vertex_index,
        )
        raw_indices = [int(value) for value in bone_index_components]

    if raw_indices:
        if not raw_weights:
            raw_weights = [1.0]
        if len(raw_weights) < len(raw_indices):
            raw_weights = raw_weights + [0.0] * (len(raw_indices) - len(raw_weights))
        if len(raw_weights) > len(raw_indices):
            raw_weights = raw_weights[: len(raw_indices)]

        for raw_index, raw_weight in zip(raw_indices, raw_weights):
            if raw_weight <= 0.0001:
                continue
            bone_id = map_bone_index(raw_index, bind_set, len(model.bones))
            if bone_id is None:
                continue
            bone_ids.append(bone_id)
            bone_weights.append(raw_weight)

    if not bone_ids and rigid_bone_id is not None:
        return [rigid_bone_id], [1.0]

    if not bone_ids:
        return [], []

    total = sum(bone_weights)
    if total <= 0.0:
        if rigid_bone_id is not None:
            return [rigid_bone_id], [1.0]
        return [], []

    normalized_weights = [weight / total for weight in bone_weights]
    return bone_ids, normalized_weights

def map_bone_index(raw_index: int, bind_set, bone_count: int) -> int | None:
    if bind_set is not None:
        bind_index = raw_index // 3
        if 0 <= bind_index < len(bind_set.binds):
            bone_id = bind_set.binds[bind_index].bone_id
            if 0 <= bone_id < bone_count:
                return bone_id
    if 0 <= raw_index < bone_count:
        return raw_index
    return None


def expand_weights(components: tuple[float, ...]) -> list[float]:
    weights = [float(value) for value in components]
    if len(weights) == 1:
        weights.append(max(0.0, 1.0 - weights[0]))
    elif len(weights) == 2:
        weights.append(max(0.0, 1.0 - weights[0] - weights[1]))
    elif len(weights) == 3:
        weights.append(max(0.0, 1.0 - weights[0] - weights[1] - weights[2]))
    return weights


def read_attribute_components(
    vertex_buffer: VertexBuffer,
    attribute: VertexAttribute,
    vertex_index: int,
) -> tuple[float, ...] | tuple[int, ...]:
    if not 0 <= vertex_index < vertex_buffer.count:
        raise G1MParseError(
            f"Vertex index {vertex_index} is out of range for vertex buffer {vertex_buffer.index}."
        )

    entry_offset = vertex_index * vertex_buffer.stride + attribute.offset
    data = vertex_buffer.raw_data

    if attribute.data_type == VERTEX_TYPE_FLOAT1:
        ensure_range(data, entry_offset, 4, "float vertex attribute")
        return Struct("<f").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_FLOAT2:
        ensure_range(data, entry_offset, 8, "float2 vertex attribute")
        return Struct("<2f").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_FLOAT3:
        ensure_range(data, entry_offset, 12, "float3 vertex attribute")
        return Struct("<3f").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_FLOAT4:
        ensure_range(data, entry_offset, 16, "float4 vertex attribute")
        return Struct("<4f").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_HALF2:
        ensure_range(data, entry_offset, 4, "half2 vertex attribute")
        return Struct("<2e").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_HALF4:
        ensure_range(data, entry_offset, 8, "half4 vertex attribute")
        return Struct("<4e").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_BYTE4 or attribute.data_type == VERTEX_TYPE_NORMALIZED_BYTE4:
        ensure_range(data, entry_offset, 4, "byte4 vertex attribute")
        raw = Struct("<4B").unpack_from(data, entry_offset)
        if attribute.data_type == VERTEX_TYPE_NORMALIZED_BYTE4:
            return tuple(component / 255.0 for component in raw)
        return raw
    if attribute.data_type == VERTEX_TYPE_USHORT4:
        ensure_range(data, entry_offset, 8, "ushort4 vertex attribute")
        return Struct("<4H").unpack_from(data, entry_offset)
    if attribute.data_type == VERTEX_TYPE_UINT4:
        ensure_range(data, entry_offset, 16, "uint4 vertex attribute")
        return Struct("<4I").unpack_from(data, entry_offset)

    raise G1MParseError(f"Unsupported vertex attribute type 0x{attribute.data_type:04X}.")


def triangulate_submesh_indices(
    raw_indices: list[int],
    face_type: int,
    vertex_offset: int,
    vertex_count: int,
    restart_value: int,
) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []

    def localize(index: int) -> int:
        local = index - vertex_offset
        if not 0 <= local < vertex_count:
            raise G1MParseError(
                f"Index {index} is outside the vertex window [{vertex_offset}, {vertex_offset + vertex_count})."
            )
        return local

    if face_type == FACE_TYPE_TRIANGLE:
        limit = len(raw_indices) - (len(raw_indices) % 3)
        for cursor in range(0, limit, 3):
            a, b, c = raw_indices[cursor : cursor + 3]
            if a == restart_value or b == restart_value or c == restart_value:
                continue
            triangles.append((localize(a), localize(b), localize(c)))
        return triangles

    if face_type == FACE_TYPE_QUAD:
        limit = len(raw_indices) - (len(raw_indices) % 4)
        for cursor in range(0, limit, 4):
            a, b, c, d = raw_indices[cursor : cursor + 4]
            if restart_value in (a, b, c, d):
                continue
            la, lb, lc, ld = localize(a), localize(b), localize(c), localize(d)
            triangles.append((la, lb, lc))
            triangles.append((la, lc, ld))
        return triangles

    if face_type == FACE_TYPE_TRIANGLE_STRIP:
        strip: list[int] = []
        for index in raw_indices:
            if index == restart_value:
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
            triangles.append((localize(a), localize(b), localize(c)))
        return triangles

    raise G1MParseError(f"Unsupported submesh face type {face_type}.")


def compute_bounds(meshes: list[PreviewMesh]) -> tuple[Vector3, Vector3]:
    if not meshes:
        origin = Vector3(0.0, 0.0, 0.0)
        return origin, origin

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    for mesh in meshes:
        for vertex in mesh.vertices:
            min_x = min(min_x, vertex.rest_position.x)
            min_y = min(min_y, vertex.rest_position.y)
            min_z = min(min_z, vertex.rest_position.z)
            max_x = max(max_x, vertex.rest_position.x)
            max_y = max(max_y, vertex.rest_position.y)
            max_z = max(max_z, vertex.rest_position.z)

    return Vector3(min_x, min_y, min_z), Vector3(max_x, max_y, max_z)


def build_bone_delta_matrices(model: G1MModel) -> list[list[float]]:
    original_globals = build_global_matrices(model, current=False)
    current_globals = build_global_matrices(model, current=True)
    return [
        multiply_matrix(current_globals[index], invert_affine_matrix(original_globals[index]))
        for index in range(len(model.bones))
    ]


def build_global_matrices(model: G1MModel, *, current: bool) -> list[list[float]]:
    matrices: list[list[float] | None] = [None] * len(model.bones)

    def resolve(bone_index: int) -> list[float]:
        cached = matrices[bone_index]
        if cached is not None:
            return cached

        bone = model.bones[bone_index]
        transform = bone.current_transform if current else bone.original_transform
        local_matrix = compose_local_matrix(
            transform.scale.x,
            transform.scale.y,
            transform.scale.z,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
            transform.position.x,
            transform.position.y,
            transform.position.z,
        )
        if 0 <= bone.parent < len(model.bones):
            cached = multiply_matrix(resolve(bone.parent), local_matrix)
        else:
            cached = local_matrix
        matrices[bone_index] = cached
        return cached

    return [resolve(index) for index in range(len(model.bones))]


def compose_local_matrix(
    scale_x: float,
    scale_y: float,
    scale_z: float,
    quat_x: float,
    quat_y: float,
    quat_z: float,
    quat_w: float,
    pos_x: float,
    pos_y: float,
    pos_z: float,
) -> list[float]:
    length = math.sqrt(quat_x * quat_x + quat_y * quat_y + quat_z * quat_z + quat_w * quat_w)
    if length <= 0.0:
        quat_x = quat_y = quat_z = 0.0
        quat_w = 1.0
    else:
        quat_x /= length
        quat_y /= length
        quat_z /= length
        quat_w /= length

    xx = quat_x * quat_x
    yy = quat_y * quat_y
    zz = quat_z * quat_z
    xy = quat_x * quat_y
    xz = quat_x * quat_z
    yz = quat_y * quat_z
    wx = quat_w * quat_x
    wy = quat_w * quat_y
    wz = quat_w * quat_z

    return [
        (1.0 - 2.0 * (yy + zz)) * scale_x,
        (2.0 * (xy - wz)) * scale_y,
        (2.0 * (xz + wy)) * scale_z,
        pos_x,
        (2.0 * (xy + wz)) * scale_x,
        (1.0 - 2.0 * (xx + zz)) * scale_y,
        (2.0 * (yz - wx)) * scale_z,
        pos_y,
        (2.0 * (xz - wy)) * scale_x,
        (2.0 * (yz + wx)) * scale_y,
        (1.0 - 2.0 * (xx + yy)) * scale_z,
        pos_z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

def compose_bind_space_matrix(
    bone_matrix: list[float],
    bind_matrix: list[float] | None,
) -> list[float]:
    if bind_matrix is None:
        return bone_matrix

    return multiply_matrix(bone_matrix, invert_affine_matrix(bind_matrix))


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


def transform_point(matrix: list[float], vector: Vector3) -> tuple[float, float, float]:
    return (
        matrix[0] * vector.x + matrix[1] * vector.y + matrix[2] * vector.z + matrix[3],
        matrix[4] * vector.x + matrix[5] * vector.y + matrix[6] * vector.z + matrix[7],
        matrix[8] * vector.x + matrix[9] * vector.y + matrix[10] * vector.z + matrix[11],
    )


def transform_direction(matrix: list[float], vector: Vector3) -> Vector3:
    return Vector3(
        matrix[0] * vector.x + matrix[1] * vector.y + matrix[2] * vector.z,
        matrix[4] * vector.x + matrix[5] * vector.y + matrix[6] * vector.z,
        matrix[8] * vector.x + matrix[9] * vector.y + matrix[10] * vector.z,
    )


def normalize_vector3(vector: Vector3) -> Vector3 | None:
    length = math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    if length <= 1e-8:
        return None
    return Vector3(vector.x / length, vector.y / length, vector.z / length)
