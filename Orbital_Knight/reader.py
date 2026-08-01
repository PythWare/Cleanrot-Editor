"""G1M parsing and patch helpers for Cleanrot Editor"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json, struct
from typing import Any, Iterable

from .g1m_engine_reference import (
    ENGINE_G1M_FILE_MAGIC,
    ENGINE_G1M_FILE_VERSIONS,
    ENGINE_KNOWN_G1M_SECTIONS,
)
from .g1m_sections import SectionCoverageReport, parse_section_coverage


RESOURCE_HEADER_STRUCT = struct.Struct("<4s4sI")
ARRAY_SECTION_HEADER_STRUCT = struct.Struct("<HHII")
G1MG_SUBSECTION_HEADER_STRUCT = struct.Struct("<HHI")
VECTOR3_STRUCT = struct.Struct("<3f")
QUATERNION_STRUCT = struct.Struct("<4f")
BONE_STRUCT = struct.Struct("<3fi4f3ff")
BONE_BIND_STRUCT = struct.Struct("<III")
SUBMESH_STRUCT = struct.Struct("<14i")
MESH_GROUP_STRUCT = struct.Struct("<9i")
MESH_GROUP_LEGACY_STRUCT = struct.Struct("<5i")
MESH_GROUP_OLD_STRUCT = struct.Struct("<3i")
MESH_ENTRY_META_STRUCT = struct.Struct("<3i")
CLOTH_CONTROL_POINT_STRUCT = struct.Struct("<4f")
CLOTH_INFLUENCE_STRUCT = struct.Struct("<4i2f")
NUNO2_BONE_STRUCT = struct.Struct("<HH3I3fI")
NUNS_INFLUENCE_STRUCT = struct.Struct("<4i2f2i")
SOFT_NODE_ENTRY_HEADER_STRUCT = struct.Struct("<13I")
SOFT_NODE_ENTRY_NODE_HEADER_STRUCT = struct.Struct("<I3f3fI4BI")
SOFT_NODE_ENTRY_NODE_INFLUENCE_STRUCT = struct.Struct("<If")
SOFT_NODE_ENTRY_NODE_DATA_STRUCT = struct.Struct("<3I3f")


class G1MParseError(RuntimeError):
    """Raised when a G1M file does not match the expected structure"""


def ensure_range(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise G1MParseError(f"{label} extends past the end of the file.")


def decode_magic(raw: bytes) -> str:
    return raw.decode("ascii", "replace")[::-1]


def decode_version(raw: bytes) -> str:
    value = raw.decode("ascii", "replace")[::-1]
    return value if value.isdigit() else raw.decode("ascii", "replace")


def read_fixed_string(data: bytes, offset: int, length: int) -> str:
    ensure_range(data, offset, length, "fixed string")
    raw = data[offset : offset + length]
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()


@dataclass(slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def copy(self) -> "Vector3":
        return Vector3(self.x, self.y, self.z)

    def multiply(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)

    def add(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vector3":
        x, y, z = list(values)
        return cls(float(x), float(y), float(z))


@dataclass(slots=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.w]

    def copy(self) -> "Quaternion":
        return Quaternion(self.x, self.y, self.z, self.w)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Quaternion":
        x, y, z, w = list(values)
        return cls(float(x), float(y), float(z), float(w))


@dataclass(slots=True)
class BoneTransform:
    scale: Vector3
    rotation: Quaternion
    position: Vector3
    length: float

    def copy(self) -> "BoneTransform":
        return BoneTransform(
            scale=self.scale.copy(),
            rotation=self.rotation.copy(),
            position=self.position.copy(),
            length=self.length,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.to_list(),
            "rotation": self.rotation.to_list(),
            "position": self.position.to_list(),
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoneTransform":
        return cls(
            scale=Vector3.from_iterable(payload["scale"]),
            rotation=Quaternion.from_iterable(payload["rotation"]),
            position=Vector3.from_iterable(payload["position"]),
            length=float(payload["length"]),
        )


@dataclass(slots=True)
class Bone:
    index: int
    parent: int
    file_offset: int
    original_transform: BoneTransform
    current_transform: BoneTransform

    @property
    def changed(self) -> bool:
        return self.original_transform.to_dict() != self.current_transform.to_dict()


@dataclass(slots=True)
class SectionInfo:
    index: int
    magic: str
    version: str
    offset: int
    size: int

    @property
    def label(self) -> str:
        return f"{self.magic} v{self.version}"


@dataclass(slots=True)
class SkeletonInfo:
    version: str
    data_offset: int
    uses_internal_boneset: int
    total_bone_count: int
    bone_table_count: int
    bone_set_count: int
    bone_indices: list[int]
    local_to_global: dict[int, int]
    global_to_local: dict[int, int]
    is_internal: bool | None
    is_unordered: bool


@dataclass(slots=True)
class GeometryInfo:
    version: str
    model_type: str
    subsection_count: int


@dataclass(slots=True)
class MatrixInfo:
    version: str
    matrix_count: int


@dataclass(slots=True)
class BoneBind:
    matrix_id: int
    signature_high: int
    reference_bone_id: int
    reserved_high: int
    bone_id: int

    @property
    def cloth_id(self) -> int:
        return self.reference_bone_id

    @property
    def joint_map_signature(self) -> int:
        return ((self.signature_high & 0xFFFF) << 16) | (self.reference_bone_id & 0xFFFF)


@dataclass(slots=True)
class BoneBindSet:
    index: int
    binds: list[BoneBind]

    @property
    def joint_map_signature(self) -> int | None:
        if not self.binds:
            return None
        return self.binds[0].joint_map_signature

    @property
    def reference_bone_id(self) -> int | None:
        for bind in self.binds:
            if bind.reference_bone_id >= 0:
                return bind.reference_bone_id
        return None


@dataclass(slots=True)
class Submesh:
    index: int
    flags: int
    vbo_index: int
    bone_table_index: int
    bone_index: int
    material_index: int
    texture_index: int
    ib_index: int
    buffer_index: int
    face_type: int
    vertex_offset: int
    vertex_count: int
    face_offset: int
    face_count: int


@dataclass(slots=True)
class MeshEntry:
    index: int
    group_index: int
    lod: int
    group: int
    name: str
    cloth_type_id: int
    nun_section_id: int
    submesh_indices: list[int]


@dataclass(slots=True)
class ClothPhysicsBone:
    bone_id: int
    position: Vector3


@dataclass(slots=True)
class ClothInfluence:
    neighbors: tuple[int, int, int, int]
    distances: tuple[float, float]
    extra_value: int | None = None


@dataclass(slots=True)
class ClothEntry:
    source_section: str
    subsection_type: int
    entry_index: int
    parent_bone_id: int
    control_points: list[tuple[float, float, float, float]]
    influences: list[ClothInfluence]
    physics_bones: list[ClothPhysicsBone]
    association_signature: int | None = None
    joint_map_index: int | None = None
    entry_id: int | None = None
    subset_parent_entry_index: int | None = None
    subset_control_point_map: tuple[int, ...] | None = None


@dataclass(slots=True)
class SoftNodeInfluence:
    node_id: int
    weight: float


@dataclass(slots=True)
class SoftNode:
    node_id: int
    position: Vector3
    rotation: Vector3
    influence_flags: tuple[int, int, int, int]
    influences: list[SoftNodeInfluence]


@dataclass(slots=True)
class SoftEntry:
    entry_index: int
    entry_id: int
    root_bone_id: int
    nodes: list[SoftNode]


@dataclass(slots=True)
class SoftSubsection:
    subsection_type: int
    size: int
    entries: list[SoftEntry]
    raw_payload: bytes


@dataclass(slots=True)
class SoftLibrary:
    subsections: list[SoftSubsection]

    def subsections_for_type(self, subsection_type: int) -> list[SoftSubsection]:
        return [subsection for subsection in self.subsections if subsection.subsection_type == subsection_type]


@dataclass(slots=True)
class HairSubsection:
    subsection_type: int
    entry_count: int
    size: int
    raw_payload: bytes


@dataclass(slots=True)
class HairLibrary:
    subsections: list[HairSubsection]

    def subsections_for_type(self, subsection_type: int) -> list[HairSubsection]:
        return [subsection for subsection in self.subsections if subsection.subsection_type == subsection_type]


@dataclass(slots=True)
class CollisionSubsection:
    subsection_type: int
    entry_count: int
    size: int
    raw_payload: bytes


@dataclass(slots=True)
class CollisionLibrary:
    subsections: list[CollisionSubsection]

    def subsections_for_type(self, subsection_type: int) -> list[CollisionSubsection]:
        return [subsection for subsection in self.subsections if subsection.subsection_type == subsection_type]


@dataclass(slots=True)
class ClothLibrary:
    nuno_entries: dict[int, list[ClothEntry]]
    nunv_entries: dict[int, list[ClothEntry]]
    nuns_entries: dict[int, list[ClothEntry]]

    @staticmethod
    def entry(entries: list[ClothEntry] | None, index: int) -> ClothEntry | None:
        if entries is None or not 0 <= index < len(entries):
            return None
        return entries[index]

    def resolve(self, cloth_type_id: int, nun_section_id: int) -> ClothEntry | None:
        if nun_section_id < 0:
            return None

        cloth_kind = cloth_type_id & 0xF
        if cloth_kind == 1:
            entry = self.entry(self.nuno_entries.get(0x00030001), nun_section_id)
            if entry is not None:
                return entry

            if nun_section_id >= 10000:
                entry = self.entry(self.nunv_entries.get(0x00050001), nun_section_id - 10000)
                if entry is not None:
                    return entry

            if nun_section_id >= 20000:
                entry = self.entry(self.nuno_entries.get(0x00030003), nun_section_id - 20000)
                if entry is not None:
                    return entry
                entry = self.entry(self.nuno_entries.get(0x00030005), nun_section_id - 20000)
                if entry is not None:
                    return entry

            entry = self.entry(self.nuno_entries.get(0x00030003), nun_section_id)
            if entry is not None:
                return entry
            return self.entry(self.nuno_entries.get(0x00030005), nun_section_id)

        if cloth_kind == 2:
            return self.entry(self.nuno_entries.get(0x00030002), nun_section_id)
        if cloth_kind == 3:
            entry = self.entry(self.nuno_entries.get(0x00030003), nun_section_id)
            if entry is not None:
                return entry
            return self.entry(self.nuno_entries.get(0x00030005), nun_section_id)
        if cloth_kind == 4:
            return self.entry(self.nuno_entries.get(0x00030004), nun_section_id)
        if cloth_kind == 5:
            entry = self.entry(self.nuno_entries.get(0x00030005), nun_section_id)
            if entry is not None:
                return entry
            return self.entry(self.nuno_entries.get(0x00030003), nun_section_id)
        if cloth_kind == 6:
            return self.entry(self.nuns_entries.get(0x00060001), nun_section_id)
        return None


@dataclass(slots=True)
class BodyPart:
    index: int
    name: str
    mesh_name: str
    group_index: int
    mesh_index: int
    lod: int
    cloth_type_id: int
    nun_section_id: int
    submesh_indices: list[int]
    bone_ids: list[int]


@dataclass(slots=True)
class G1MModel:
    source_path: Path | None
    raw_data: bytes
    file_magic: str
    file_version: str
    sections: list[SectionInfo]
    skeleton_info: SkeletonInfo
    geometry_info: GeometryInfo | None
    matrix_info: MatrixInfo | None
    bones: list[Bone]
    bind_matrices: list[list[float]]
    body_parts: list[BodyPart]
    bone_bind_sets: list[BoneBindSet]
    submeshes: list[Submesh]
    mesh_entries: list[MeshEntry]
    collision_library: CollisionLibrary
    cloth_library: ClothLibrary
    soft_library: SoftLibrary
    hair_library: HairLibrary
    section_coverage: SectionCoverageReport
    parse_warnings: list[str]

    @classmethod
    def from_path(cls, path: str | Path) -> "G1MModel":
        target = Path(path)
        return cls.from_bytes(target.read_bytes(), source_path=target)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        source_path: str | Path | None = None,
    ) -> "G1MModel":
        ensure_range(data, 0, 24, "G1M header")
        raw_magic, raw_version, _file_size = RESOURCE_HEADER_STRUCT.unpack_from(data, 0)
        file_magic = decode_magic(raw_magic)
        file_version = decode_version(raw_version)
        if file_magic != ENGINE_G1M_FILE_MAGIC:
            raise G1MParseError(f"Expected {ENGINE_G1M_FILE_MAGIC} file magic, got {file_magic!r}.")

        parse_warnings: list[str] = []
        if file_version not in ENGINE_G1M_FILE_VERSIONS:
            parse_warnings.append(
                f"G1M version {file_version} was not in the runtime-confirmed set "
                f"{', '.join(sorted(ENGINE_G1M_FILE_VERSIONS))}; loading conservatively."
            )

        data_start, _reserved1, section_count = struct.unpack_from("<III", data, 12)
        sections = parse_sections(data, data_start, section_count)
        section_coverage = parse_section_coverage(data, sections)
        parse_warnings.extend(section_coverage.warnings)

        skeleton_section = next((section for section in sections if section.magic == "G1MS"), None)
        if skeleton_section is None:
            raise G1MParseError("File does not contain a G1MS skeleton section.")

        collision_section = next((section for section in sections if section.magic == "COLL"), None)
        matrix_section = next((section for section in sections if section.magic == "G1MM"), None)
        geometry_section = next((section for section in sections if section.magic == "G1MG"), None)
        soft_section = next((section for section in sections if section.magic == "SOFT"), None)
        hair_section = next((section for section in sections if section.magic == "HAIR"), None)

        skeleton_info, bones = parse_skeleton_section(data, skeleton_section)
        matrix_info, bind_matrices = parse_matrix_section(data, matrix_section)
        geometry_info, bone_bind_sets, submeshes, mesh_entries = parse_geometry_section(
            data,
            geometry_section,
        )
        collision_library = parse_optional(
            "COLL", parse_warnings, CollisionLibrary(subsections=[]), parse_collision_section, data, collision_section
        )
        cloth_library = parse_optional(
            "cloth", parse_warnings, ClothLibrary({}, {}, {}), parse_cloth_sections, data, sections
        )
        soft_library = parse_optional(
            "SOFT", parse_warnings, SoftLibrary(subsections=[]), parse_soft_section, data, soft_section
        )
        hair_library = parse_optional(
            "HAIR", parse_warnings, HairLibrary(subsections=[]), parse_hair_section, data, hair_section
        )
        body_parts = build_body_parts(mesh_entries, submeshes, bone_bind_sets, len(bones))

        return cls(
            source_path=Path(source_path) if source_path else None,
            raw_data=data,
            file_magic=file_magic,
            file_version=file_version,
            sections=sections,
            skeleton_info=skeleton_info,
            geometry_info=geometry_info,
            matrix_info=matrix_info,
            bones=bones,
            bind_matrices=bind_matrices,
            body_parts=body_parts,
            bone_bind_sets=bone_bind_sets,
            submeshes=submeshes,
            mesh_entries=mesh_entries,
            collision_library=collision_library,
            cloth_library=cloth_library,
            soft_library=soft_library,
            hair_library=hair_library,
            section_coverage=section_coverage,
            parse_warnings=parse_warnings,
        )

    @property
    def changed_bones(self) -> list[Bone]:
        return [bone for bone in self.bones if bone.changed]

    def body_parts_for_bone(self, bone_index: int) -> list[BodyPart]:
        return [part for part in self.body_parts if bone_index in part.bone_ids]

    def cloth_entry_for_body_part(self, body_part: BodyPart) -> ClothEntry | None:
        return self.cloth_library.resolve(body_part.cloth_type_id, body_part.nun_section_id)

    @property
    def global_bone_ids(self) -> tuple[int, ...]:
        if self.skeleton_info.global_to_local:
            return tuple(sorted(self.skeleton_info.global_to_local))
        return tuple(range(len(self.bones)))

    @property
    def compatible_bone_ids(self) -> tuple[int, ...]:
        bone_ids = set(self.global_bone_ids)
        bone_ids.update(range(len(self.bones)))
        return tuple(sorted(bone_ids))

    @property
    def unknown_sections(self):
        return self.section_coverage.unknown_sections

    @property
    def section_warnings(self) -> tuple[str, ...]:
        return tuple(self.parse_warnings)

    def section_summary(self) -> str:
        return self.section_coverage.summary()

    def set_bone_transform(
        self,
        bone_index: int,
        *,
        scale: Vector3 | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        length: float | None = None,
    ) -> Bone:
        bone = self.bones[bone_index]
        updated = bone.current_transform.copy()
        if scale is not None:
            updated.scale = scale
        if position is not None:
            updated.position = position
        if rotation is not None:
            updated.rotation = rotation
        if length is not None:
            updated.length = float(length)
        bone.current_transform = updated
        return bone

    def apply_adjustment_to_bones(
        self,
        bone_indices: Iterable[int],
        *,
        scale_multiplier: Vector3 | None = None,
        position_offset: Vector3 | None = None,
    ) -> None:
        for bone_index in bone_indices:
            bone = self.bones[bone_index]
            updated = bone.current_transform.copy()
            if scale_multiplier is not None:
                updated.scale = updated.scale.multiply(scale_multiplier)
            if position_offset is not None:
                updated.position = updated.position.add(position_offset)
            bone.current_transform = updated

    def reset_bone(self, bone_index: int) -> None:
        self.bones[bone_index].current_transform = self.bones[bone_index].original_transform.copy()

    def reset_bones(self, bone_indices: Iterable[int]) -> None:
        for bone_index in bone_indices:
            self.reset_bone(bone_index)

    def reset_all(self) -> None:
        for bone in self.bones:
            bone.current_transform = bone.original_transform.copy()

    def preset_payload(self) -> dict[str, Any]:
        return {
            "format": "marylcian-preset-1",
            "source_file": self.source_path.name if self.source_path else None,
            "source_bone_count": len(self.bones),
            "bones": {
                str(bone.index): bone.current_transform.to_dict()
                for bone in self.changed_bones
            },
        }

    def save_preset(self, path: str | Path) -> None:
        target = Path(path)
        payload = json.dumps(self.preset_payload(), indent=2)
        target.write_text(payload, encoding="utf-8")

    def apply_preset(self, payload: dict[str, Any]) -> list[int]:
        self.reset_all()
        applied: list[int] = []
        for raw_index, transform_payload in payload.get("bones", {}).items():
            bone_index = int(raw_index)
            if 0 <= bone_index < len(self.bones):
                self.bones[bone_index].current_transform = BoneTransform.from_dict(transform_payload)
                applied.append(bone_index)
        return applied

    def load_preset(self, path: str | Path) -> list[int]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.apply_preset(payload)

    def patched_bytes(self) -> bytes:
        buffer = bytearray(self.raw_data)
        for bone in self.bones:
            transform = bone.current_transform
            BONE_STRUCT.pack_into(
                buffer,
                bone.file_offset,
                transform.scale.x,
                transform.scale.y,
                transform.scale.z,
                bone.parent,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
                transform.position.x,
                transform.position.y,
                transform.position.z,
                transform.length,
            )
        return bytes(buffer)

    def export_patched_copy(self, path: str | Path) -> None:
        Path(path).write_bytes(self.patched_bytes())


def parse_optional(label: str, warnings: list[str], fallback, parser, *args):
    """Parse optional/non-critical section families without blocking model edits"""

    try:
        return parser(*args)
    except G1MParseError as exc:
        warnings.append(f"{label}: {exc}, raw bytes preserved but decoded view disabled.")
        return fallback

def parse_sections(data: bytes, data_start: int, section_count: int) -> list[SectionInfo]:
    sections: list[SectionInfo] = []
    offset = data_start
    for index in range(section_count):
        ensure_range(data, offset, RESOURCE_HEADER_STRUCT.size, f"section {index} header")
        raw_magic, raw_version, size = RESOURCE_HEADER_STRUCT.unpack_from(data, offset)
        if size < RESOURCE_HEADER_STRUCT.size:
            raise G1MParseError(f"Section {index} reports an invalid size: {size}.")
        ensure_range(data, offset, size, f"section {index}")
        sections.append(
            SectionInfo(
                index=index,
                magic=decode_magic(raw_magic),
                version=decode_version(raw_version),
                offset=offset,
                size=size,
            )
        )
        offset += size
    return sections

def parse_skeleton_section(data: bytes, section: SectionInfo) -> tuple[SkeletonInfo, list[Bone]]:
    version = section_version_number(section)
    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 16, "G1MS header")
    data_offset, uses_internal_boneset = struct.unpack_from("<II", data, payload_offset)
    total_bone_count, bone_table_count = struct.unpack_from("<HH", data, payload_offset + 8)
    bone_set_count = struct.unpack_from("<I", data, payload_offset + 12)[0]

    bone_indices_offset = payload_offset + 16
    bone_indices_size = bone_table_count * 2
    ensure_range(data, bone_indices_offset, bone_indices_size, "G1MS bone table")
    raw_bone_indices = list(
        struct.unpack_from(f"<{bone_table_count}H", data, bone_indices_offset)
        if bone_table_count
        else ()
    )
    bone_indices = [index if index != 0xFFFF else -1 for index in raw_bone_indices]

    bones_offset = section.offset + data_offset
    ensure_range(data, bones_offset, total_bone_count * BONE_STRUCT.size, "G1MS bone data")

    local_to_global: dict[int, int] = {}
    global_to_local: dict[int, int] = {}
    is_unordered = version < 32
    if is_unordered:
        for local_id in range(total_bone_count):
            local_to_global[local_id] = local_id
            global_to_local[local_id] = local_id
    else:
        for global_id, local_id in enumerate(bone_indices):
            if local_id < 0:
                continue
            local_to_global[int(local_id)] = int(global_id)
            global_to_local[int(global_id)] = int(local_id)
        if not local_to_global:
            for local_id in range(total_bone_count):
                local_to_global[local_id] = local_id
                global_to_local[local_id] = local_id

    is_internal: bool | None = None
    if total_bone_count:
        first_parent_raw = struct.unpack_from("<I", data, bones_offset + 12)[0]
        is_internal = first_parent_raw != 0x80000000

    bones: list[Bone] = []
    for bone_index in range(total_bone_count):
        bone_offset = bones_offset + bone_index * BONE_STRUCT.size
        unpacked = BONE_STRUCT.unpack_from(data, bone_offset)
        original = BoneTransform(
            scale=Vector3(*unpacked[0:3]),
            rotation=Quaternion(*unpacked[4:8]),
            position=Vector3(*unpacked[8:11]),
            length=unpacked[11],
        )
        bones.append(
            Bone(
                index=bone_index,
                parent=unpacked[3],
                file_offset=bone_offset,
                original_transform=original,
                current_transform=original.copy(),
            )
        )

    info = SkeletonInfo(
        version=section.version,
        data_offset=data_offset,
        uses_internal_boneset=uses_internal_boneset,
        total_bone_count=total_bone_count,
        bone_table_count=bone_table_count,
        bone_set_count=bone_set_count,
        bone_indices=bone_indices,
        local_to_global=local_to_global,
        global_to_local=global_to_local,
        is_internal=is_internal,
        is_unordered=is_unordered,
    )
    return info, bones

def parse_matrix_section(data: bytes, section: SectionInfo | None) -> tuple[MatrixInfo | None, list[list[float]]]:
    if section is None:
        return None, []

    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "G1MM header")
    matrix_count = struct.unpack_from("<i", data, payload_offset)[0]
    if matrix_count < 0:
        raise G1MParseError(f"G1MM reports an invalid matrix count: {matrix_count}.")

    matrices_offset = payload_offset + 4
    ensure_range(data, matrices_offset, matrix_count * 64, "G1MM matrix data")
    matrices = [
        list(struct.unpack_from("<16f", data, matrices_offset + matrix_index * 64))
        for matrix_index in range(matrix_count)
    ]
    info = MatrixInfo(version=section.version, matrix_count=matrix_count)
    return info, matrices


def parse_geometry_section(
    data: bytes,
    section: SectionInfo | None,
) -> tuple[GeometryInfo | None, list[BoneBindSet], list[Submesh], list[MeshEntry]]:
    if section is None:
        return None, [], [], []

    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 36, "G1MG header")
    model_type = read_fixed_string(data, payload_offset, 4)
    subsection_count = struct.unpack_from("<I", data, payload_offset + 32)[0]
    geometry_info = GeometryInfo(
        version=section.version,
        model_type=model_type,
        subsection_count=subsection_count,
    )
    g1mg_version = section_version_number(section)

    bone_bind_sets: list[BoneBindSet] = []
    raw_submesh_fields: list[tuple[int, ...]] = []
    submeshes: list[Submesh] = []
    mesh_entries: list[MeshEntry] = []
    index_buffer_counts: list[int] = []

    cursor = payload_offset + 36
    group_index = 0
    mesh_index = 0
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, G1MG_SUBSECTION_HEADER_STRUCT.size, f"G1MG subsection {subsection_index}")
        magic_id, _unknown, size = G1MG_SUBSECTION_HEADER_STRUCT.unpack_from(data, cursor)
        if size < G1MG_SUBSECTION_HEADER_STRUCT.size:
            raise G1MParseError(f"G1MG subsection {subsection_index} reports an invalid size.")
        ensure_range(data, cursor, size, f"G1MG subsection {subsection_index}")

        payload = cursor + G1MG_SUBSECTION_HEADER_STRUCT.size
        subsection_end = cursor + size
        ensure_range(data, payload, 4, f"G1MG subsection {subsection_index} entry count")
        count = struct.unpack_from("<I", data, payload)[0]
        payload += 4

        if magic_id == 6:
            for _ in range(count):
                ensure_range(data, payload, 4, "bone bind count")
                bind_count = struct.unpack_from("<I", data, payload)[0]
                payload += 4
                binds: list[BoneBind] = []
                for _ in range(bind_count):
                    ensure_range(data, payload, BONE_BIND_STRUCT.size, "bone bind entry")
                    matrix_id, packed_signature, packed_mapping = BONE_BIND_STRUCT.unpack_from(data, payload)
                    binds.append(
                        BoneBind(
                            matrix_id=int(matrix_id),
                            signature_high=int((packed_signature >> 16) & 0xFFFF),
                            reference_bone_id=int(packed_signature & 0xFFFF),
                            reserved_high=int((packed_mapping >> 16) & 0xFFFF),
                            bone_id=int(packed_mapping & 0xFFFF),
                        )
                    )
                    payload += BONE_BIND_STRUCT.size
                bone_bind_sets.append(BoneBindSet(index=len(bone_bind_sets), binds=binds))

        elif magic_id == 7:
            try:
                index_buffer_counts = parse_g1mg_index_buffer_counts(
                    data,
                    payload,
                    count,
                    subsection_end,
                    g1mg_version,
                )
            except G1MParseError:
                index_buffer_counts = []

        elif magic_id == 8:
            for _ in range(count):
                ensure_range(data, payload, 0x24, "surface entry header")
                header = tuple(int(value) for value in struct.unpack_from("<9i", data, payload))
                draw_window_count = max(0, int(header[8]))
                payload += 0x24
                if draw_window_count:
                    ensure_range(data, payload, draw_window_count * 0x14, "surface draw windows")
                    first_window = tuple(int(value) for value in struct.unpack_from("<5i", data, payload))
                    payload += draw_window_count * 0x14
                else:
                    first_window = (0, 0, 0, 0, 0)
                raw_submesh_fields.append(header + first_window)

        elif magic_id == 9:
            parsed_entries, mesh_index, group_index = parse_g1mg_mesh_groups(
                data=data,
                payload=payload,
                subsection_end=subsection_end,
                group_count=count,
                g1mg_version=g1mg_version,
                first_mesh_index=mesh_index,
                first_group_index=group_index,
                submesh_limit=len(raw_submesh_fields),
            )
            mesh_entries.extend(parsed_entries)

        cursor += size

    ib_field_index = choose_submesh_ib_field(raw_submesh_fields, index_buffer_counts)
    for fields in raw_submesh_fields:
        ib_index = int(fields[ib_field_index]) if ib_field_index is not None else int(fields[7])
        submeshes.append(
            Submesh(
                index=len(submeshes),
                flags=int(fields[0]),
                vbo_index=int(fields[1]),
                bone_table_index=int(fields[2]),
                bone_index=-1,
                material_index=int(fields[3]),
                texture_index=int(fields[6]),
                ib_index=ib_index,
                buffer_index=ib_index,
                face_type=int(fields[9]),
                vertex_offset=int(fields[10]),
                vertex_count=int(fields[11]),
                face_offset=int(fields[12]),
                face_count=int(fields[13]),
            )
        )

    return geometry_info, bone_bind_sets, submeshes, mesh_entries

def parse_g1mg_mesh_groups(
    data: bytes,
    payload: int,
    subsection_end: int,
    group_count: int,
    g1mg_version: int,
    first_mesh_index: int,
    first_group_index: int,
    submesh_limit: int,
) -> tuple[list[MeshEntry], int, int]:
    def parse_with(layout: str) -> tuple[list[MeshEntry], int, int] | None:
        cursor = payload
        mesh_index = first_mesh_index
        group_index = first_group_index
        entries: list[MeshEntry] = []

        if layout == "old12" and group_count != 1:
            return None

        for group_number in range(group_count):
            if layout == "new36":
                ensure_range(data, cursor, MESH_GROUP_STRUCT.size, "mesh group")
                fields = MESH_GROUP_STRUCT.unpack_from(data, cursor)
                lod, group, unknown1, submesh_count, unknown_count, lod_start, lod_end, unknown3, unknown4 = fields
                cursor += MESH_GROUP_STRUCT.size
                entry_count = submesh_count + unknown_count
            elif layout == "legacy20":
                ensure_range(data, cursor, MESH_GROUP_LEGACY_STRUCT.size, "legacy mesh group")
                lod, group, unknown1, submesh_count, unknown_count = MESH_GROUP_LEGACY_STRUCT.unpack_from(data, cursor)
                cursor += MESH_GROUP_LEGACY_STRUCT.size
                entry_count = submesh_count + unknown_count
            else:
                ensure_range(data, cursor, MESH_GROUP_OLD_STRUCT.size, "old mesh group")
                lod, group, old_count = MESH_GROUP_OLD_STRUCT.unpack_from(data, cursor)
                cursor += MESH_GROUP_OLD_STRUCT.size
                entry_count = None

            if entry_count is not None and (entry_count < 0 or entry_count > 100000):
                return None

            entry_number = 0
            while entry_count is None or entry_number < entry_count:
                if entry_count is None and cursor >= subsection_end:
                    break
                if cursor + 28 > subsection_end:
                    return None

                name = read_fixed_string(data, cursor, 16)
                cloth_type_id, nun_section_id, index_count = MESH_ENTRY_META_STRUCT.unpack_from(data, cursor + 16)
                cursor += 28

                if index_count < 0:
                    return None
                index_bytes = index_count * 4
                if cursor + index_bytes > subsection_end:
                    return None

                indices: list[int] = []
                if index_count:
                    indices = list(struct.unpack_from(f"<{index_count}i", data, cursor))
                    if submesh_limit and any(index < 0 or index >= submesh_limit for index in indices):
                        return None
                cursor += index_bytes

                entries.append(
                    MeshEntry(
                        index=mesh_index,
                        group_index=group_index,
                        lod=lod,
                        group=group,
                        name=name or f"mesh_{mesh_index}",
                        cloth_type_id=cloth_type_id,
                        nun_section_id=nun_section_id,
                        submesh_indices=indices,
                    )
                )
                mesh_index += 1
                entry_number += 1

            group_index += 1

        if cursor != subsection_end:
            return None
        return entries, mesh_index, group_index

    if g1mg_version <= 30:
        layouts = ("old12", "legacy20", "new36")
    elif g1mg_version > 41:
        layouts = ("new36", "legacy20", "old12")
    else:
        layouts = ("legacy20", "new36", "old12")

    for layout in layouts:
        parsed = parse_with(layout)
        if parsed is not None:
            return parsed

    raise G1MParseError("Unable to parse G1MG mesh group entries as 36, 20, or 12 byte headers.")

def parse_g1mg_index_buffer_counts(
    data: bytes,
    payload: int,
    count: int,
    subsection_end: int,
    g1mg_version: int = 0,
) -> list[int]:
    def parse_with(header_size: int) -> list[int] | None:
        cursor = payload
        counts: list[int] = []
        for _ in range(count):
            ensure_range(data, cursor, header_size, "index buffer header")
            if header_size == 8:
                index_count, bit_width = struct.unpack_from("<2I", data, cursor)
            elif header_size == 12:
                index_count, bit_width, _pad = struct.unpack_from("<3I", data, cursor)
            else:
                _flags, index_count, bit_width, _pad = struct.unpack_from("<4I", data, cursor)
            if bit_width not in {8, 16, 32}:
                return None
            byte_width = bit_width // 8
            cursor += header_size
            ensure_range(data, cursor, int(index_count) * byte_width, "index buffer data")
            cursor += int(index_count) * byte_width
            cursor = (cursor + 3) & ~3
            counts.append(int(index_count))
        if cursor != subsection_end:
            return None
        return counts

    header_order = (12, 8, 16) if g1mg_version > 40 else (8, 12, 16)
    for header_size in header_order:
        parsed = parse_with(header_size)
        if parsed is not None:
            return parsed
    raise G1MParseError("Unable to parse G1MG index buffer headers as 12, 8, or 16 byte entries.")


def choose_submesh_ib_field(
    raw_submesh_fields: list[tuple[int, ...]],
    index_buffer_counts: list[int],
) -> int | None:
    if not raw_submesh_fields or not index_buffer_counts:
        return 7

    def valid_count(field_index: int) -> int:
        valid = 0
        for fields in raw_submesh_fields:
            ib_index = int(fields[field_index])
            face_offset = int(fields[12])
            face_count = int(fields[13])
            if 0 <= ib_index < len(index_buffer_counts) and 0 <= face_offset <= index_buffer_counts[ib_index]:
                if face_offset + max(face_count, 0) <= index_buffer_counts[ib_index]:
                    valid += 1
        return valid

    field5_valid = valid_count(5)
    field7_valid = valid_count(7)
    if field7_valid > 0:
        return 7
    if field5_valid > 0:
        return 5
    return 7

def parse_cloth_sections(data: bytes, sections: list[SectionInfo]) -> ClothLibrary:
    nuno_section = next((section for section in sections if section.magic == "NUNO"), None)
    nunv_section = next((section for section in sections if section.magic == "NUNV"), None)
    nuns_section = next((section for section in sections if section.magic == "NUNS"), None)

    return ClothLibrary(
        nuno_entries=parse_nuno_section(data, nuno_section),
        nunv_entries=parse_nunv_section(data, nunv_section),
        nuns_entries=parse_nuns_section(data, nuns_section),
    )


def parse_collision_section(data: bytes, section: SectionInfo | None) -> CollisionLibrary:
    if section is None:
        return CollisionLibrary(subsections=[])

    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "COLL header")
    subsection_count = struct.unpack_from("<I", data, payload_offset)[0]

    subsections: list[CollisionSubsection] = []
    cursor = payload_offset + 4
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, 12, f"COLL subsection {subsection_index}")
        subsection_type, subsection_size, entry_count = struct.unpack_from("<III", data, cursor)
        if subsection_size < 12:
            raise G1MParseError(f"COLL subsection {subsection_index} reports an invalid size.")
        ensure_range(data, cursor, subsection_size, f"COLL subsection {subsection_index}")
        payload = cursor + 12
        subsections.append(
            CollisionSubsection(
                subsection_type=int(subsection_type),
                entry_count=int(entry_count),
                size=int(subsection_size),
                raw_payload=bytes(data[payload : cursor + subsection_size]),
            )
        )
        cursor += subsection_size

    return CollisionLibrary(subsections=subsections)

def parse_soft_section(data: bytes, section: SectionInfo | None) -> SoftLibrary:
    if section is None:
        return SoftLibrary(subsections=[])

    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "SOFT header")
    subsection_count = struct.unpack_from("<I", data, payload_offset)[0]

    subsections: list[SoftSubsection] = []
    cursor = payload_offset + 4
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, 8, f"SOFT subsection {subsection_index}")
        subsection_type, subsection_size = struct.unpack_from("<II", data, cursor)
        if subsection_size < 8:
            raise G1MParseError(f"SOFT subsection {subsection_index} reports an invalid size.")
        ensure_range(data, cursor, subsection_size, f"SOFT subsection {subsection_index}")

        payload = cursor + 8
        payload_end = cursor + subsection_size
        entries: list[SoftEntry] = []
        if subsection_type == 0x00080001:
            ensure_range(data, payload, 4, "SOFT 0x00080001 entry count")
            entry_count = struct.unpack_from("<I", data, payload)[0]
            payload += 4
            for entry_index in range(entry_count):
                ensure_range(data, payload, SOFT_NODE_ENTRY_HEADER_STRUCT.size, "SOFT entry header")
                (
                    entry_id,
                    len1,
                    z2,
                    z3,
                    u4,
                    len2,
                    len3,
                    root_bone_id,
                    u6,
                    u7,
                    z8,
                    o9,
                    len4,
                ) = SOFT_NODE_ENTRY_HEADER_STRUCT.unpack_from(data, payload)
                payload += SOFT_NODE_ENTRY_HEADER_STRUCT.size

                ensure_range(data, payload, 24 * 4, "SOFT entry unknown block")
                payload += 24 * 4

                nodes: list[SoftNode] = []
                for _node_index in range(len1):
                    ensure_range(
                        data,
                        payload,
                        SOFT_NODE_ENTRY_NODE_HEADER_STRUCT.size,
                        "SOFT node header",
                    )
                    (
                        node_id,
                        px,
                        py,
                        pz,
                        rx,
                        ry,
                        rz,
                        unk,
                        flag0,
                        flag1,
                        flag2,
                        flag3,
                        influence_count,
                    ) = SOFT_NODE_ENTRY_NODE_HEADER_STRUCT.unpack_from(data, payload)
                    payload += SOFT_NODE_ENTRY_NODE_HEADER_STRUCT.size

                    influences: list[SoftNodeInfluence] = []
                    for _ in range(influence_count + 1):
                        ensure_range(
                            data,
                            payload,
                            SOFT_NODE_ENTRY_NODE_INFLUENCE_STRUCT.size,
                            "SOFT node influence",
                        )
                        influence_id, influence_weight = SOFT_NODE_ENTRY_NODE_INFLUENCE_STRUCT.unpack_from(
                            data,
                            payload,
                        )
                        influences.append(
                            SoftNodeInfluence(
                                node_id=int(influence_id),
                                weight=float(influence_weight),
                            )
                        )
                        payload += SOFT_NODE_ENTRY_NODE_INFLUENCE_STRUCT.size

                    ensure_range(data, payload, SOFT_NODE_ENTRY_NODE_DATA_STRUCT.size, "SOFT node data")
                    payload += SOFT_NODE_ENTRY_NODE_DATA_STRUCT.size
                    nodes.append(
                        SoftNode(
                            node_id=int(node_id),
                            position=Vector3(float(px), float(py), float(pz)),
                            rotation=Vector3(float(rx), float(ry), float(rz)),
                            influence_flags=(int(flag0), int(flag1), int(flag2), int(flag3)),
                            influences=influences,
                        )
                    )

                ensure_range(data, payload, u4 * 4, "SOFT list1")
                payload += u4 * 4
                ensure_range(data, payload, len1 * 4, "SOFT list2")
                payload += len1 * 4
                ensure_range(data, payload, u6 * 4, "SOFT list3")
                payload += u6 * 4
                ensure_range(data, payload, len3 * 12, "SOFT list4")
                payload += len3 * 12

                ensure_range(data, payload, 8, "SOFT tail header")
                unk2, len5 = struct.unpack_from("<II", data, payload)
                payload += 8
                if len5 < 8:
                    raise G1MParseError(f"SOFT entry {entry_index} reports an invalid tail size.")
                ensure_range(data, payload, len5 - 8, "SOFT tail payload")
                payload += len5 - 8

                entries.append(
                    SoftEntry(
                        entry_index=entry_index,
                        entry_id=int(entry_id),
                        root_bone_id=int(root_bone_id),
                        nodes=nodes,
                    )
                )

        subsections.append(
            SoftSubsection(
                subsection_type=int(subsection_type),
                size=int(subsection_size),
                entries=entries,
                raw_payload=bytes(data[cursor + 8 : payload_end]),
            )
        )
        cursor += subsection_size

    return SoftLibrary(subsections=subsections)

def parse_hair_section(data: bytes, section: SectionInfo | None) -> HairLibrary:
    if section is None:
        return HairLibrary(subsections=[])

    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "HAIR header")
    subsection_count = struct.unpack_from("<I", data, payload_offset)[0]

    subsections: list[HairSubsection] = []
    cursor = payload_offset + 4
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, 12, f"HAIR subsection {subsection_index}")
        subsection_type, subsection_size, entry_count = struct.unpack_from("<III", data, cursor)
        if subsection_size < 12:
            raise G1MParseError(f"HAIR subsection {subsection_index} reports an invalid size.")
        ensure_range(data, cursor, subsection_size, f"HAIR subsection {subsection_index}")
        payload = cursor + 12
        subsections.append(
            HairSubsection(
                subsection_type=int(subsection_type),
                entry_count=int(entry_count),
                size=int(subsection_size),
                raw_payload=bytes(data[payload : cursor + subsection_size]),
            )
        )
        cursor += subsection_size

    return HairLibrary(subsections=subsections)


def section_version_number(section: SectionInfo | None) -> int:
    if section is None:
        return 0
    try:
        return int(section.version)
    except ValueError:
        return 0


def read_cloth_control_points(
    data: bytes,
    offset: int,
    count: int,
    label: str,
) -> tuple[list[tuple[float, float, float, float]], int]:
    control_points: list[tuple[float, float, float, float]] = []
    for _ in range(count):
        ensure_range(data, offset, CLOTH_CONTROL_POINT_STRUCT.size, label)
        control_points.append(CLOTH_CONTROL_POINT_STRUCT.unpack_from(data, offset))
        offset += CLOTH_CONTROL_POINT_STRUCT.size
    return control_points, offset


def read_cloth_influences(
    data: bytes,
    offset: int,
    count: int,
    label: str,
) -> tuple[list[ClothInfluence], int]:
    influences: list[ClothInfluence] = []
    for _ in range(count):
        ensure_range(data, offset, CLOTH_INFLUENCE_STRUCT.size, label)
        neighbor0, neighbor1, neighbor2, neighbor3, distance0, distance1 = CLOTH_INFLUENCE_STRUCT.unpack_from(
            data,
            offset,
        )
        influences.append(
            ClothInfluence(
                neighbors=(int(neighbor0), int(neighbor1), int(neighbor2), int(neighbor3)),
                distances=(float(distance0), float(distance1)),
            )
        )
        offset += CLOTH_INFLUENCE_STRUCT.size
    return influences, offset


NUNO5_CONTROL_POINT_STRUCT = struct.Struct("<6f5i")


def read_nuns_influences(
    data: bytes,
    offset: int,
    count: int,
    label: str,
) -> tuple[list[ClothInfluence], int]:
    influences: list[ClothInfluence] = []
    for _ in range(count):
        ensure_range(data, offset, NUNS_INFLUENCE_STRUCT.size, label)
        neighbor0, neighbor1, neighbor2, neighbor3, distance0, distance1, _extra0, _extra1 = (
            NUNS_INFLUENCE_STRUCT.unpack_from(data, offset)
        )
        influences.append(
            ClothInfluence(
                neighbors=(int(neighbor0), int(neighbor1), int(neighbor2), int(neighbor3)),
                distances=(float(distance0), float(distance1)),
            )
        )
        offset += NUNS_INFLUENCE_STRUCT.size
    return influences, offset


def read_nuno5_control_points(
    data: bytes,
    offset: int,
    count: int,
    label: str,
) -> tuple[list[tuple[float, float, float, float]], list[ClothInfluence], int]:
    control_points: list[tuple[float, float, float, float]] = []
    influences: list[ClothInfluence] = []
    for _ in range(count):
        ensure_range(data, offset, NUNO5_CONTROL_POINT_STRUCT.size, label)
        x, y, z, _ux, _uy, _uz, neighbor0, neighbor1, parent_id, neighbor3, extra_value = (
            NUNO5_CONTROL_POINT_STRUCT.unpack_from(data, offset)
        )
        control_points.append((float(x), float(y), float(z), 1.0))
        influences.append(
            ClothInfluence(
                neighbors=(int(neighbor0), int(neighbor1), int(parent_id), int(neighbor3)),
                distances=(float(extra_value), 0.0),
                extra_value=int(extra_value),
            )
        )
        offset += NUNO5_CONTROL_POINT_STRUCT.size
    return control_points, influences, offset


def build_nuno5_subset_map(
    parent_control_points: list[tuple[float, float, float, float]],
    child_control_points: list[tuple[float, float, float, float]],
) -> tuple[int, ...] | None:
    if not parent_control_points or not child_control_points:
        return None

    exact_lookup = {
        (point[0], point[1], point[2]): index
        for index, point in enumerate(parent_control_points)
    }
    summed_lookup = {
        (point[0] + point[0] + point[1] + point[2]): index
        for index, point in enumerate(parent_control_points)
    }

    mapping: list[int] = []
    matched_any = False
    for point in child_control_points:
        key = (point[0], point[1], point[2])
        parent_index = exact_lookup.get(key)
        if parent_index is None:
            parent_index = summed_lookup.get(point[0] + point[0] + point[1] + point[2], -1)
        if parent_index >= 0:
            matched_any = True
        mapping.append(int(parent_index))

    return tuple(mapping) if matched_any else None


def parse_nuno_section(data: bytes, section: SectionInfo | None) -> dict[int, list[ClothEntry]]:
    if section is None:
        return {}

    version = section_version_number(section)
    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "NUNO header")
    subsection_count = struct.unpack_from("<I", data, payload_offset)[0]

    entries_by_flag: dict[int, list[ClothEntry]] = {}
    cursor = payload_offset + 4
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, 12, f"NUNO subsection {subsection_index}")
        flag, subsection_size, entry_count = struct.unpack_from("<III", data, cursor)
        ensure_range(data, cursor, subsection_size, f"NUNO subsection {subsection_index}")
        payload = cursor + 12

        entries: list[ClothEntry] = []
        if flag == 0x00030001:
            for entry_index in range(entry_count):
                ensure_range(data, payload, 36, "NUNO 0x00030001 entry header")
                signature, control_point_count, unknown_section_count, skip1, skip2, skip3 = (
                    struct.unpack_from("<6I", data, payload)
                )
                joint_map_index = struct.unpack_from("<I", data, payload + 32)[0]
                payload += 24 + 0x4C
                if version >= 25:
                    payload += 0x10

                control_points, payload = read_cloth_control_points(
                    data,
                    payload,
                    control_point_count,
                    "NUNO 0x00030001 control point",
                )
                influences, payload = read_cloth_influences(
                    data,
                    payload,
                    control_point_count,
                    "NUNO 0x00030001 influences",
                )
                payload += unknown_section_count * 0x30
                payload += (skip1 + skip2 + skip3) * 4

                entries.append(
                    ClothEntry(
                        source_section="NUNO",
                        subsection_type=flag,
                        entry_index=entry_index,
                        parent_bone_id=int(signature & 0xFFFF),
                        control_points=control_points,
                        influences=influences,
                        physics_bones=[],
                        association_signature=int(signature),
                        joint_map_index=int(joint_map_index),
                    )
                )

        elif flag == 0x00030002:
            for entry_index in range(entry_count):
                ensure_range(data, payload, 16, "NUNO 0x00030002 entry header")
                parent_bone_id, _unknown1, physics_bone_count, skip1 = struct.unpack_from("<4I", data, payload)
                payload += 16 + (9 * 4) + 4 + (6 * 4)
                if version >= 29:
                    payload += 3 * 4

                physics_bones: list[ClothPhysicsBone] = []
                for _ in range(physics_bone_count):
                    ensure_range(data, payload, NUNO2_BONE_STRUCT.size, "NUNO 0x00030002 physics bone")
                    bone_index, _padding, _r0, _r1, _r2, x, y, z, _padding2 = NUNO2_BONE_STRUCT.unpack_from(
                        data,
                        payload,
                    )
                    physics_bones.append(
                        ClothPhysicsBone(
                            bone_id=int(bone_index),
                            position=Vector3(x, y, z),
                        )
                    )
                    payload += NUNO2_BONE_STRUCT.size

                payload += skip1 * 4
                entries.append(
                    ClothEntry(
                        source_section="NUNO",
                        subsection_type=flag,
                        entry_index=entry_index,
                        parent_bone_id=int(parent_bone_id),
                        control_points=[],
                        influences=[],
                        physics_bones=physics_bones,
                    )
                )

        elif flag == 0x00030003:
            for entry_index in range(entry_count):
                ensure_range(data, payload, 36, "NUNO 0x00030003 entry header")
                (
                    parent_bone_id,
                    control_point_count,
                    unknown_section_count,
                    skip1,
                    _unknown1,
                    skip2,
                    skip3,
                    skip4,
                    _unknown2,
                ) = struct.unpack_from("<9I", data, payload)
                payload += 36

                if version < 30:
                    payload += 0xA8
                    if version >= 25:
                        payload += 0x10
                else:
                    payload += 0x8
                    ensure_range(data, payload, 4, "NUNO 0x00030003 offset")
                    offset_value = struct.unpack_from("<I", data, payload)[0]
                    if offset_value < 4:
                        raise G1MParseError("NUNO 0x00030003 reported an invalid offset.")
                    payload += offset_value

                control_points, payload = read_cloth_control_points(
                    data,
                    payload,
                    control_point_count,
                    "NUNO 0x00030003 control point",
                )
                influences, payload = read_cloth_influences(
                    data,
                    payload,
                    control_point_count,
                    "NUNO 0x00030003 influences",
                )
                payload += unknown_section_count * 0x30
                payload += skip1 * 4
                payload += skip2 * 8
                payload += skip3 * 12
                payload += skip4 * 8

                entries.append(
                    ClothEntry(
                        source_section="NUNO",
                        subsection_type=flag,
                        entry_index=entry_index,
                        parent_bone_id=int(parent_bone_id),
                        control_points=control_points,
                        influences=influences,
                        physics_bones=[],
                    )
                )

        elif flag == 0x00030005:
            if version >= 35:
                ensure_range(data, payload, 4, "NUNO 0x00030005 reserved")
                payload += 4

            entry_id_to_entry_index: dict[int, int] = {}
            for entry_index in range(entry_count):
                ensure_range(data, payload, 0x24, "NUNO 0x00030005 entry header")
                parent_bone_id, _unknown0, lod_count, _unknown1, _unknown2 = struct.unpack_from("<5I", data, payload)
                entry_id, subset_id = struct.unpack_from("<HH", data, payload + 20)
                payload += 0x24

                subset_parent_entry_index = (
                    entry_id_to_entry_index.get(int(entry_id))
                    if subset_id & 0x7FF
                    else None
                )

                control_points: list[tuple[float, float, float, float]] = []
                influences: list[ClothInfluence] = []
                for lod_index in range(lod_count):
                    ensure_range(data, payload, 0x30, "NUNO 0x00030005 LOD header")
                    values = struct.unpack_from("<12I", data, payload)
                    control_point_count = int(values[0])
                    flags = int(values[1])
                    skip_counts = [int(value) for value in values[2:11]]
                    uses_skip9 = int(values[11])
                    payload += 0x30

                    skip9_size = 0
                    skip9_count = 0
                    if uses_skip9:
                        ensure_range(data, payload, 8, "NUNO 0x00030005 extra skip header")
                        skip9_size, skip9_count = struct.unpack_from("<2I", data, payload)
                        payload += 8

                    ensure_range(data, payload, 4, "NUNO 0x00030005 control point offset")
                    control_point_offset = struct.unpack_from("<I", data, payload)[0]
                    if control_point_offset < 4:
                        raise G1MParseError("NUNO 0x00030005 reported an invalid control point offset.")
                    payload += control_point_offset

                    if lod_index == 0:
                        control_points, influences, payload = read_nuno5_control_points(
                            data,
                            payload,
                            control_point_count,
                            "NUNO 0x00030005 control point",
                        )
                    else:
                        ensure_range(
                            data,
                            payload,
                            control_point_count * NUNO5_CONTROL_POINT_STRUCT.size,
                            "NUNO 0x00030005 lower LOD control point data",
                        )
                        payload += control_point_count * NUNO5_CONTROL_POINT_STRUCT.size

                    if flags & 0x1:
                        payload += 0x20 * control_point_count
                    if flags & 0x2:
                        payload += 0x18 * control_point_count

                    payload += (
                        skip_counts[0] * 0x4
                        + skip_counts[1] * 0xC
                        + skip_counts[2] * 0x10
                        + skip_counts[3] * 0xC
                        + skip_counts[4] * 0x8
                        + skip_counts[5] * 0x30
                        + skip_counts[6] * 0x48
                        + skip_counts[7] * 0x20
                    )
                    if flags & 0x4:
                        payload += 0x4 * control_point_count
                    for _ in range(skip_counts[8]):
                        ensure_range(data, payload, 0x10, "NUNO 0x00030005 skip block")
                        temp_count = struct.unpack_from("<I", data, payload)[0]
                        payload += 0x10 + temp_count * 0x4
                    payload += skip9_size * skip9_count

                entries.append(
                    ClothEntry(
                        source_section="NUNO",
                        subsection_type=flag,
                        entry_index=entry_index,
                        parent_bone_id=int(parent_bone_id),
                        control_points=control_points,
                        influences=influences,
                        physics_bones=[],
                        entry_id=int(entry_id),
                        subset_parent_entry_index=subset_parent_entry_index,
                    )
                )
                entry_id_to_entry_index.setdefault(int(entry_id), entry_index)

            for entry in entries:
                if entry.subset_parent_entry_index is None:
                    continue
                if not 0 <= entry.subset_parent_entry_index < len(entries):
                    continue
                parent_entry = entries[entry.subset_parent_entry_index]
                entry.subset_control_point_map = build_nuno5_subset_map(
                    parent_entry.control_points,
                    entry.control_points,
                )

        if entries:
            entries_by_flag[flag] = entries
        cursor += subsection_size

    return entries_by_flag


def parse_nunv_section(data: bytes, section: SectionInfo | None) -> dict[int, list[ClothEntry]]:
    if section is None:
        return {}

    version = section_version_number(section)
    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "NUNV header")
    subsection_count = struct.unpack_from("<I", data, payload_offset)[0]

    entries_by_flag: dict[int, list[ClothEntry]] = {}
    cursor = payload_offset + 4
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, 12, f"NUNV subsection {subsection_index}")
        flag, subsection_size, entry_count = struct.unpack_from("<III", data, cursor)
        ensure_range(data, cursor, subsection_size, f"NUNV subsection {subsection_index}")
        payload = cursor + 12

        entries: list[ClothEntry] = []
        if flag == 0x00050001:
            for entry_index in range(entry_count):
                ensure_range(data, payload, 20, "NUNV 0x00050001 entry header")
                signature, control_point_count, unknown_section_count, skip1, joint_map_index = struct.unpack_from(
                    "<5I",
                    data,
                    payload,
                )
                payload += 16 + 0x54
                if version > 10:
                    payload += 0x10

                control_points, payload = read_cloth_control_points(
                    data,
                    payload,
                    control_point_count,
                    "NUNV 0x00050001 control point",
                )
                influences, payload = read_cloth_influences(
                    data,
                    payload,
                    control_point_count,
                    "NUNV 0x00050001 influences",
                )
                payload += unknown_section_count * 0x30
                payload += skip1 * 4

                entries.append(
                    ClothEntry(
                        source_section="NUNV",
                        subsection_type=flag,
                        entry_index=entry_index,
                        parent_bone_id=int(signature & 0xFFFF),
                        control_points=control_points,
                        influences=influences,
                        physics_bones=[],
                        association_signature=int(signature),
                        joint_map_index=int(joint_map_index),
                    )
                )

        if entries:
            entries_by_flag[flag] = entries
        cursor += subsection_size

    return entries_by_flag


def parse_nuns_section(data: bytes, section: SectionInfo | None) -> dict[int, list[ClothEntry]]:
    if section is None:
        return {}

    payload_offset = section.offset + RESOURCE_HEADER_STRUCT.size
    ensure_range(data, payload_offset, 4, "NUNS header")
    subsection_count = struct.unpack_from("<I", data, payload_offset)[0]

    entries_by_flag: dict[int, list[ClothEntry]] = {}
    cursor = payload_offset + 4
    for subsection_index in range(subsection_count):
        ensure_range(data, cursor, 12, f"NUNS subsection {subsection_index}")
        flag, subsection_size, entry_count = struct.unpack_from("<III", data, cursor)
        ensure_range(data, cursor, subsection_size, f"NUNS subsection {subsection_index}")
        payload = cursor + 12

        entries: list[ClothEntry] = []
        if flag == 0x00060001:
            for entry_index in range(entry_count):
                ensure_range(data, payload, 28, "NUNS 0x00060001 entry header")
                parent_bone_le, _parent_bone_be, control_point_count, _unk1, _unk2, _unk3, _unk4, skip1 = (
                    struct.unpack_from("<HH6I", data, payload)
                )
                payload += 28 + 0xA4

                control_points, payload = read_cloth_control_points(
                    data,
                    payload,
                    control_point_count,
                    "NUNS 0x00060001 control point",
                )
                influences, payload = read_nuns_influences(
                    data,
                    payload,
                    control_point_count,
                    "NUNS 0x00060001 influences",
                )
                payload += skip1

                ensure_range(data, payload, 12, "NUNS BLWO header")
                _magic, _unknown, size = struct.unpack_from("<4sII", data, payload)
                payload += 12 + size + 0xC

                entries.append(
                    ClothEntry(
                        source_section="NUNS",
                        subsection_type=flag,
                        entry_index=entry_index,
                        parent_bone_id=int(parent_bone_le),
                        control_points=control_points,
                        influences=influences,
                        physics_bones=[],
                    )
                )

        if entries:
            entries_by_flag[flag] = entries
        cursor += subsection_size

    return entries_by_flag


def build_body_parts(
    mesh_entries: list[MeshEntry],
    submeshes: list[Submesh],
    bone_bind_sets: list[BoneBindSet],
    bone_count: int,
) -> list[BodyPart]:
    if not mesh_entries:
        return []

    name_counts = Counter(mesh.name for mesh in mesh_entries)
    display_counts: Counter[str] = Counter()
    body_parts: list[BodyPart] = []

    for mesh in mesh_entries:
        bone_ids: set[int] = set()
        submesh_indices: set[int] = set()
        for submesh_index in mesh.submesh_indices:
            if not 0 <= submesh_index < len(submeshes):
                continue
            submesh_indices.add(submesh_index)
            submesh = submeshes[submesh_index]
            if 0 <= submesh.bone_table_index < len(bone_bind_sets):
                for bind in bone_bind_sets[submesh.bone_table_index].binds:
                    if 0 <= bind.reference_bone_id < bone_count:
                        bone_ids.add(bind.reference_bone_id)
                    if 0 <= bind.bone_id < bone_count:
                        bone_ids.add(bind.bone_id)
            if 0 <= submesh.bone_index < bone_count:
                bone_ids.add(submesh.bone_index)

        display_name = mesh.name
        if name_counts[mesh.name] > 1:
            display_name = f"{mesh.name} [group {mesh.group_index}]"
        display_counts[display_name] += 1
        if display_counts[display_name] > 1:
            display_name = f"{display_name} #{display_counts[display_name]}"

        body_parts.append(
            BodyPart(
                index=len(body_parts),
                name=display_name,
                mesh_name=mesh.name,
                group_index=mesh.group_index,
                mesh_index=mesh.index,
                lod=mesh.lod,
                cloth_type_id=mesh.cloth_type_id,
                nun_section_id=mesh.nun_section_id,
                submesh_indices=sorted(submesh_indices),
                bone_ids=sorted(bone_ids),
            )
        )

    return body_parts
