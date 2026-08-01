"""G1M section/subsection diagnostics and preservation helpers"""

from __future__ import annotations
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .g1m_engine_reference import ENGINE_KNOWN_G1M_SECTIONS, G1MG_SUBSECTION_HANDLERS


RESOURCE_HEADER_STRUCT = struct.Struct("<4s4sI")
ARRAY_SECTION_HEADER_STRUCT = struct.Struct("<HHII")
G1MG_SUBSECTION_HEADER_STRUCT = struct.Struct("<HHI")
NUN_SECTION_HEADER_STRUCT = struct.Struct("<III")
SOFT_SECTION_HEADER_STRUCT = struct.Struct("<II")

SUBSECTION_SECTION_MAGICS = frozenset({"G1MG", "COLL", "NUNO", "NUNV", "NUNS", "HAIR", "SOFT"})


@dataclass(frozen=True, slots=True)
class RawSubsectionInfo:
    section_magic: str
    index: int
    subsection_type: int
    offset: int
    size: int
    entry_count: int | None
    header_size: int
    payload_offset: int
    payload_size: int
    engine_address: int | None = None
    note: str | None = None

    @property
    def label(self) -> str:
        count = "?" if self.entry_count is None else str(self.entry_count)
        addr = "" if self.engine_address is None else f" handler=0x{self.engine_address:08X}"
        return (
            f"{self.section_magic}[{self.index}] type=0x{self.subsection_type:08X} "
            f"entries={count} size={self.size}{addr}"
        )

@dataclass(frozen=True, slots=True)
class RawSectionInfo:
    index: int
    magic: str
    version: str
    offset: int
    size: int
    known_by_engine: bool
    subsection_count: int | None
    subsections: tuple[RawSubsectionInfo, ...]
    warning: str | None = None

    @property
    def payload_offset(self) -> int:
        return self.offset + RESOURCE_HEADER_STRUCT.size

    @property
    def payload_size(self) -> int:
        return max(0, self.size - RESOURCE_HEADER_STRUCT.size)

    @property
    def label(self) -> str:
        known = "engine-known" if self.known_by_engine else "unknown/preserved"
        if self.subsection_count is None:
            return f"{self.magic} v{self.version} size={self.size} {known}"
        return f"{self.magic} v{self.version} size={self.size} {known} subsections={self.subsection_count}"

@dataclass(frozen=True, slots=True)
class SectionCoverageReport:
    sections: tuple[RawSectionInfo, ...]

    @property
    def section_magics(self) -> tuple[str, ...]:
        return tuple(section.magic for section in self.sections)

    @property
    def unknown_sections(self) -> tuple[RawSectionInfo, ...]:
        return tuple(section for section in self.sections if not section.known_by_engine)

    @property
    def warnings(self) -> tuple[str, ...]:
        values: list[str] = []
        for section in self.sections:
            if section.warning:
                values.append(f"{section.magic}: {section.warning}")
        return tuple(values)

    def sections_for_magic(self, magic: str) -> tuple[RawSectionInfo, ...]:
        return tuple(section for section in self.sections if section.magic == magic)

    def subsections_for_magic(self, magic: str) -> tuple[RawSubsectionInfo, ...]:
        values: list[RawSubsectionInfo] = []
        for section in self.sections_for_magic(magic):
            values.extend(section.subsections)
        return tuple(values)

    def summary(self) -> str:
        pieces: list[str] = []
        counts = Counter(section.magic for section in self.sections)
        for magic, count in sorted(counts.items()):
            sections = self.sections_for_magic(magic)
            sub_count = sum(len(section.subsections) for section in sections)
            marker = "" if all(section.known_by_engine for section in sections) else "*"
            if sub_count:
                pieces.append(f"{magic}{marker} x{count} / {sub_count} sub")
            else:
                pieces.append(f"{magic}{marker} x{count}")
        if self.unknown_sections:
            pieces.append("* unknown preserved")
        if self.warnings:
            pieces.append(f"{len(self.warnings)} warning(s)")
        return " | ".join(pieces)

def parse_section_coverage(data: bytes, sections: Iterable[object]) -> SectionCoverageReport:
    raw_sections: list[RawSectionInfo] = []
    for section in sections:
        index = int(getattr(section, "index"))
        magic = str(getattr(section, "magic"))
        version = str(getattr(section, "version"))
        offset = int(getattr(section, "offset"))
        size = int(getattr(section, "size"))
        known = magic in ENGINE_KNOWN_G1M_SECTIONS
        subsection_count: int | None = None
        subsections: tuple[RawSubsectionInfo, ...] = ()
        warning: str | None = None

        if magic in SUBSECTION_SECTION_MAGICS:
            try:
                subsection_count, subsections = parse_raw_subsections(data, magic, offset, size)
            except Exception as exc:
                warning = str(exc)

        raw_sections.append(
            RawSectionInfo(
                index=index,
                magic=magic,
                version=version,
                offset=offset,
                size=size,
                known_by_engine=known,
                subsection_count=subsection_count,
                subsections=subsections,
                warning=warning,
            )
        )
    return SectionCoverageReport(sections=tuple(raw_sections))

def parse_raw_subsections(
    data: bytes,
    magic: str,
    section_offset: int,
    section_size: int,
) -> tuple[int, tuple[RawSubsectionInfo, ...]]:
    payload_offset = section_offset + RESOURCE_HEADER_STRUCT.size
    section_end = section_offset + section_size
    if magic == "G1MG":
        ensure_range(data, payload_offset, 36, "G1MG header")
        subsection_count = struct.unpack_from("<I", data, payload_offset + 32)[0]
        cursor = payload_offset + 36
    else:
        ensure_range(data, payload_offset, 4, f"{magic} subsection count")
        subsection_count = struct.unpack_from("<I", data, payload_offset)[0]
        cursor = payload_offset + 4

    subsections: list[RawSubsectionInfo] = []
    for index in range(subsection_count):
        if magic == "G1MG":
            header_size = G1MG_SUBSECTION_HEADER_STRUCT.size
            ensure_range(data, cursor, header_size, f"{magic} subsection {index} header")
            raw_type, _unknown, size = G1MG_SUBSECTION_HEADER_STRUCT.unpack_from(data, cursor)
            subsection_type = int(raw_type)
            if cursor + header_size + 4 <= section_end:
                entry_count = int(struct.unpack_from("<I", data, cursor + header_size)[0])
            else:
                entry_count = None
            handler = G1MG_SUBSECTION_HANDLERS.get(subsection_type)
            engine_address = handler.address if handler else None
            note = handler.note if handler else None
        elif magic == "SOFT":
            header_size = SOFT_SECTION_HEADER_STRUCT.size
            ensure_range(data, cursor, header_size, f"{magic} subsection {index} header")
            subsection_type, size = SOFT_SECTION_HEADER_STRUCT.unpack_from(data, cursor)
            entry_count = None
            engine_address = None
            note = None
        else:
            header_size = NUN_SECTION_HEADER_STRUCT.size
            ensure_range(data, cursor, header_size, f"{magic} subsection {index} header")
            subsection_type, size, count = NUN_SECTION_HEADER_STRUCT.unpack_from(data, cursor)
            entry_count = int(count)
            engine_address = None
            note = None

        size = int(size)
        if size < header_size:
            raise ValueError(f"subsection {index} reports invalid size {size}")
        ensure_range(data, cursor, size, f"{magic} subsection {index}")
        if cursor + size > section_end:
            raise ValueError(f"subsection {index} extends past {magic} section")

        subsections.append(
            RawSubsectionInfo(
                section_magic=magic,
                index=index,
                subsection_type=int(subsection_type),
                offset=cursor,
                size=size,
                entry_count=entry_count,
                header_size=header_size,
                payload_offset=cursor + header_size,
                payload_size=size - header_size,
                engine_address=engine_address,
                note=note,
            )
        )
        cursor += size

    if cursor > section_end:
        raise ValueError(f"subsection table extends past {magic} section")
    return int(subsection_count), tuple(subsections)

def ensure_range(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(f"{label} extends past the end of the file")
