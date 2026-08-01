"""Runtime confirmed G1M reference facts for Cleanrot"""

from __future__ import annotations

from dataclasses import dataclass


ENGINE_G1M_FILE_MAGIC = "G1M_"
ENGINE_G1M_FILE_VERSIONS = frozenset({"0033", "0034", "0035", "0036"})
ENGINE_G1T_FILE_MAGIC = "G1TG"
ENGINE_G1T_FILE_VERSIONS = frozenset({"0020", "0030", "0040", "0050", "0060", "0061"})

ENGINE_KNOWN_G1M_SECTIONS = frozenset(
    {
        "G1MF",
        "G1MS",
        "G1MG",
        "G1MM",
        "COLL",
        "EXTR",
        "NUNO",
        "NUNV",
        "NUNS",
        "HAIR",
        "SOFT",
    }
)


@dataclass(frozen=True, slots=True)
class EngineHandler:
    magic_or_subsection: str
    address: int
    note: str


G1M_SECTION_HANDLERS = {
    "G1MS": EngineHandler("G1MS", 0x00C7CA10, "skeleton/bone table parser"),
    "G1MM": EngineHandler("G1MM", 0x00C7CB90, "matrix/bind matrix parser"),
    "G1MG": EngineHandler("G1MG", 0x00C7E850, "geometry subsection dispatcher"),
    "COLL": EngineHandler("COLL", 0x00C7DA50, "collision subsection parser"),
    "EXTR": EngineHandler("EXTR", 0x00C7F120, "extra data subsection parser"),
    "NUNO": EngineHandler("NUNO", 0x00C7DB80, "cloth/soft body subsection parser"),
    "NUNV": EngineHandler("NUNV", 0x00C7DFF0, "cloth/soft body subsection parser"),
    "NUNS": EngineHandler("NUNS", 0x00C7E260, "cloth/soft body subsection parser"),
    "HAIR": EngineHandler("HAIR", 0x00C7E5F0, "hair subsection parser"),
}

G1MG_SUBSECTION_HANDLERS = {
    1: EngineHandler("G1MG:1", 0x00C7CC20, "64 byte records, currently preserved/diagnostic"),
    2: EngineHandler("G1MG:2", 0x00C7CC90, "0x10 header/variable payload records, preserved/diagnostic"),
    3: EngineHandler("G1MG:3", 0x00C7CD90, "pointer table/resource records, preserved/diagnostic"),
    4: EngineHandler("G1MG:4", 0x00C7CF60, "vertex buffer data"),
    5: EngineHandler("G1MG:5", 0x00C7D150, "vertex attribute/declaration sets"),
    6: EngineHandler("G1MG:6", 0x00C7D2F0, "bone bind/joint map sets"),
    7: EngineHandler("G1MG:7", 0x00C7D3C0, "index buffers, generic header is 8 bytes, IB record headers are 8 byte for <=0040, 12 byte for >0040"),
    8: EngineHandler("G1MG:8", 0x00C7D560, "surface records, count dword, pointer table, 0x24 byte surface header, then N 0x14 byte draw windows"),
    9: EngineHandler("G1MG:9", 0x00C7D630, "mesh groups/mesh entries"),
}
