from __future__ import annotations

from dataclasses import dataclass, field


CONSTANT_BUFFER_SIZE = 0x200
CONTROL_POINT_BASE = 96
MODELVIEW_BASE = 500
CV0_INDEX = 504
CV1_INDEX = 505
CV2_INDEX = 506


@dataclass(slots=True)
class Float4:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    def copy(self) -> "Float4":
        return Float4(self.x, self.y, self.z, self.w)

    def __neg__(self) -> "Float4":
        return Float4(-self.x, -self.y, -self.z, -self.w)


@dataclass(slots=True)
class Address4:
    x: int = 0
    y: int = 0
    z: int = 0
    w: int = 0


@dataclass(slots=True)
class ClothShaderInput:
    v0: Float4 = field(default_factory=Float4)
    v1: Float4 = field(default_factory=Float4)
    v2: Float4 = field(default_factory=Float4)
    v3: Float4 = field(default_factory=Float4)
    v4: Float4 = field(default_factory=Float4)
    v5: Float4 = field(default_factory=Float4)
    v6: Float4 = field(default_factory=Float4)
    v7: Float4 = field(default_factory=Float4)
    v8: Float4 = field(default_factory=Float4)
    v9: Float4 = field(default_factory=Float4)
    v10: Float4 = field(default_factory=Float4)
    v11: Float4 = field(default_factory=Float4)
    v12: Float4 = field(default_factory=Float4)


@dataclass(slots=True)
class ClothShaderOutput:
    o0: Float4 = field(default_factory=Float4)
    o1: Float4 = field(default_factory=Float4)
    o2: Float4 = field(default_factory=Float4)
    o3: Float4 = field(default_factory=Float4)
    o4: Float4 = field(default_factory=Float4)


def make_cloth_constant_buffer(
    control_points: list[tuple[float, float, float, float]],
) -> list[Float4] | None:
    if CONTROL_POINT_BASE + len(control_points) > MODELVIEW_BASE:
        return None

    constants = [Float4() for _ in range(CONSTANT_BUFFER_SIZE)]

    for base in range(0, 0x20, 4):
        constants[base + 0] = Float4(1.0, 0.0, 0.0, 0.0)
        constants[base + 1] = Float4(0.0, 1.0, 0.0, 0.0)
        constants[base + 2] = Float4(0.0, 0.0, 1.0, 0.0)
        constants[base + 3] = Float4(0.0, 0.0, 0.0, 1.0)

    constants[MODELVIEW_BASE + 0] = Float4(1.0, 0.0, 0.0, 0.0)
    constants[MODELVIEW_BASE + 1] = Float4(0.0, 1.0, 0.0, 0.0)
    constants[MODELVIEW_BASE + 2] = Float4(0.0, 0.0, 1.0, 0.0)
    constants[MODELVIEW_BASE + 3] = Float4(0.0, 0.0, 0.0, 1.0)
    constants[CV0_INDEX] = Float4(0.0, 0.0, 0.0, 0.0)

    for index, point in enumerate(control_points):
        constants[CONTROL_POINT_BASE + index] = Float4(
            float(point[0]),
            float(point[1]),
            float(point[2]),
            float(point[3]),
        )

    return constants


def round_nearest(value: float) -> int:
    return int(value + 0.5) if value > 0.0 else 0


def expandswizzle(swizzle: str) -> str:
    if not swizzle:
        return "xyzw"
    if len(swizzle) == 1:
        return swizzle * 4
    if len(swizzle) == 2:
        return swizzle[0] + swizzle[1] * 3
    if len(swizzle) == 3:
        return swizzle + swizzle[2]
    return swizzle[:4]


def component(source: Float4, selector: str) -> float:
    if selector == "x":
        return source.x
    if selector == "y":
        return source.y
    if selector == "z":
        return source.z
    return source.w


def swizzle(source: Float4, swizzle: str) -> Float4:
    usage = expandswizzle(swizzle)
    return Float4(
        component(source, usage[0]),
        component(source, usage[1]),
        component(source, usage[2]),
        component(source, usage[3]),
    )


def mask_float4(destination: Float4, source: Float4, mask: str) -> None:
    if not mask:
        destination.x = source.x
        destination.y = source.y
        destination.z = source.z
        destination.w = source.w
        return
    for selector in mask:
        if selector == "x":
            destination.x = source.x
        elif selector == "y":
            destination.y = source.y
        elif selector == "z":
            destination.z = source.z
        elif selector == "w":
            destination.w = source.w


def maskaddress4(destination: Address4, source: Address4, mask: str) -> None:
    if not mask:
        destination.x = source.x
        destination.y = source.y
        destination.z = source.z
        destination.w = source.w
        return
    for selector in mask:
        if selector == "x":
            destination.x = source.x
        elif selector == "y":
            destination.y = source.y
        elif selector == "z":
            destination.z = source.z
        elif selector == "w":
            destination.w = source.w


def add(destination: Float4, mask: str, src0: Float4, swizzle0: str, src1: Float4, swizzle1: str) -> None:
    s0 = swizzle(src0, swizzle0)
    s1 = swizzle(src1, swizzle1)
    mask_float4(
        destination,
        Float4(s0.x + s1.x, s0.y + s1.y, s0.z + s1.z, s0.w + s1.w),
        mask,
    )


def dp4(destination: Float4, mask: str, src0: Float4, swizzle0: str, src1: Float4, swizzle1: str) -> None:
    s0 = swizzle(src0, swizzle0)
    s1 = swizzle(src1, swizzle1)
    value = s0.x * s1.x + s0.y * s1.y + s0.z * s1.z + s0.w * s1.w
    mask_float4(destination, Float4(value, value, value, value), mask)


def lrp(
    destination: Float4,
    mask: str,
    src0: Float4,
    swizzle0: str,
    src1: Float4,
    swizzle1: str,
    src2: Float4,
    swizzle2: str,
) -> None:
    s0 = swizzle(src0, swizzle0)
    s1 = swizzle(src1, swizzle1)
    s2 = swizzle(src2, swizzle2)
    mask_float4(
        destination,
        Float4(
            s0.x * (s1.x - s2.x) + s2.x,
            s0.y * (s1.y - s2.y) + s2.y,
            s0.z * (s1.z - s2.z) + s2.z,
            s0.w * (s1.w - s2.w) + s2.w,
        ),
        mask,
    )


def mov(destination: Float4, mask: str, source: Float4, swizzle_pattern: str) -> None:
    mask_float4(destination, swizzle(source, swizzle_pattern), mask)


def mova(destination: Address4, mask: str, source: Float4, swizzle_pattern: str) -> None:
    swizzled = swizzle(source, swizzle_pattern)
    maskaddress4(
        destination,
        Address4(
            round_nearest(swizzled.x),
            round_nearest(swizzled.y),
            round_nearest(swizzled.z),
            round_nearest(swizzled.w),
        ),
        mask,
    )


def mad(
    destination: Float4,
    mask: str,
    src0: Float4,
    swizzle0: str,
    src1: Float4,
    swizzle1: str,
    src2: Float4,
    swizzle2: str,
) -> None:
    s0 = swizzle(src0, swizzle0)
    s1 = swizzle(src1, swizzle1)
    s2 = swizzle(src2, swizzle2)
    mask_float4(
        destination,
        Float4(
            s0.x * s1.x + s2.x,
            s0.y * s1.y + s2.y,
            s0.z * s1.z + s2.z,
            s0.w * s1.w + s2.w,
        ),
        mask,
    )


def mul(destination: Float4, mask: str, src0: Float4, swizzle0: str, src1: Float4, swizzle1: str) -> None:
    s0 = swizzle(src0, swizzle0)
    s1 = swizzle(src1, swizzle1)
    mask_float4(
        destination,
        Float4(s0.x * s1.x, s0.y * s1.y, s0.z * s1.z, s0.w * s1.w),
        mask,
    )


def slt(destination: Float4, mask: str, src0: Float4, swizzle0: str, src1: Float4, swizzle1: str) -> None:
    s0 = swizzle(src0, swizzle0)
    s1 = swizzle(src1, swizzle1)
    mask_float4(
        destination,
        Float4(
            1.0 if s0.x < s1.x else 0.0,
            1.0 if s0.y < s1.y else 0.0,
            1.0 if s0.z < s1.z else 0.0,
            1.0 if s0.w < s1.w else 0.0,
        ),
        mask,
    )


def constant(constants: list[Float4], base: int, index: int) -> Float4:
    return constants[base + index]


def transform_cloth_vertex2(
    shader_input: ClothShaderInput,
    constants: list[Float4],
) -> ClothShaderOutput | None:
    if len(constants) < 511:
        return None

    constants[CV1_INDEX].x = 0.0
    constants[CV1_INDEX].y = 1.0
    constants[CV1_INDEX].z = 0.0
    constants[CV1_INDEX].w = 0.0
    constants[CV2_INDEX].x = 0.0
    constants[CV2_INDEX].y = -3.0
    constants[CV2_INDEX].z = -6.0
    constants[CV2_INDEX].w = -9.0

    v0 = shader_input.v0
    v1 = shader_input.v1
    v2 = shader_input.v2
    v3 = shader_input.v3
    v4 = shader_input.v4
    v5 = shader_input.v5
    v6 = shader_input.v6
    v7 = shader_input.v7
    v8 = shader_input.v8
    v10 = shader_input.v10
    v11 = shader_input.v11
    v12 = shader_input.v12

    registers = [Float4() for _ in range(32)]
    address = [Address4()]

    dp4(registers[0], "x", v5, "", v5, "")
    slt(registers[0], "x", constants[CV1_INDEX], "x", registers[0], "x")
    mov(registers[0], "yzw", v4, "xxyz")
    mova(address[0], "w", registers[0], "z")
    mul(registers[1], "xyz", v0, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw")
    mova(address[0], "w", registers[0], "y")
    mad(registers[1], "xyz", v0, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[1], "")
    mova(address[0], "w", registers[0], "w")
    mad(registers[1], "xyz", v0, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[1], "")
    mov(registers[1], "w", v4, "w")
    mova(address[0], "w", registers[1], "w")
    mad(registers[1], "xyz", v0, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[1], "")

    mov(registers[2], "", v5, "")
    mova(address[0], "w", registers[2], "y")
    mul(registers[3], "xyz", v0, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw")
    mova(address[0], "w", registers[2], "x")
    mad(registers[3], "xyz", v0, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[3], "")
    mova(address[0], "w", registers[2], "z")
    mad(registers[3], "xyz", v0, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[3], "")
    mova(address[0], "w", registers[2], "w")
    mad(registers[3], "xyz", v0, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[3], "")

    mov(registers[3], "w", v6, "x")
    mov(registers[4], "x", v6, "y")
    mova(address[0], "w", registers[4], "x")
    mul(registers[4], "yzw", v0, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "xzxy")
    mova(address[0], "w", registers[3], "w")
    mad(registers[4], "yzw", v0, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "xzxy", registers[4], "")
    mov(registers[5], "xy", v6, "zwzw")
    mova(address[0], "w", registers[5], "x")
    mad(registers[4], "yzw", v0, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "xzxy", registers[4], "")
    mova(address[0], "w", registers[5], "y")
    mad(registers[4], "yzw", v0, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "xzxy", registers[4], "")

    mov(registers[5], "zw", v7, "xyxy")
    mova(address[0], "w", registers[5], "w")
    mul(registers[6], "xyz", v0, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw")
    mova(address[0], "w", registers[5], "z")
    mad(registers[6], "xyz", v0, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[6], "")
    mov(registers[6], "w", v7, "z")
    mova(address[0], "w", registers[6], "w")
    mad(registers[6], "xyz", v0, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[6], "")
    mov(registers[7], "x", v7, "w")
    mova(address[0], "w", registers[7], "x")
    mad(registers[6], "xyz", v0, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "zxyw", registers[6], "")

    mul(registers[7], "yzw", registers[3], "xyzx", v1, "y")
    mad(registers[7], "yzw", v1, "x", registers[1], "xyzx", registers[7], "")
    mad(registers[7], "yzw", v1, "z", registers[4], "xzwy", registers[7], "")
    mad(registers[7], "yzw", v1, "w", registers[6], "xyzx", registers[7], "")
    mul(registers[3], "xyz", registers[3], "", v3, "y")
    mad(registers[1], "xyz", v3, "x", registers[1], "", registers[3], "")
    mad(registers[1], "xyz", v3, "z", registers[4], "yzww", registers[1], "")
    mad(registers[1], "xyz", v3, "w", registers[6], "", registers[1], "")

    mova(address[0], "w", registers[0], "z")
    mul(registers[3], "xyz", v2, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw")
    mova(address[0], "w", registers[0], "y")
    mad(registers[3], "xyz", v2, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[3], "")
    mova(address[0], "w", registers[0], "w")
    mad(registers[0], "yzw", v2, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "xyzx", registers[3], "xxyz")
    mova(address[0], "w", registers[1], "w")
    mad(registers[0], "yzw", v2, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "xyzx", registers[0], "")

    mova(address[0], "w", registers[2], "y")
    mul(registers[3], "xyz", v2, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw")
    mova(address[0], "w", registers[2], "x")
    mad(registers[3], "xyz", v2, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[3], "")
    mova(address[0], "w", registers[2], "z")
    mad(registers[2], "xyz", v2, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[3], "")
    mova(address[0], "w", registers[2], "w")
    mad(registers[2], "xyz", v2, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[2], "")

    mova(address[0], "w", registers[4], "x")
    mul(registers[3], "xyz", v2, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw")
    mova(address[0], "w", registers[3], "w")
    mad(registers[3], "xyz", v2, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[3], "")
    mova(address[0], "w", registers[5], "x")
    mad(registers[3], "xyz", v2, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[3], "")
    mova(address[0], "w", registers[5], "y")
    mad(registers[3], "xyz", v2, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[3], "")

    mova(address[0], "w", registers[5], "w")
    mul(registers[4], "xyz", v2, "y", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw")
    mova(address[0], "w", registers[5], "z")
    mad(registers[4], "xyz", v2, "x", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[4], "")
    mova(address[0], "w", registers[6], "w")
    mad(registers[4], "xyz", v2, "z", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[4], "")
    mova(address[0], "w", registers[7], "x")
    mad(registers[4], "xyz", v2, "w", constant(constants, CONTROL_POINT_BASE, address[0].w), "yzxw", registers[4], "")

    mul(registers[2], "xyz", registers[2], "", v1, "y")
    mad(registers[0], "yzw", v1, "x", registers[0], "", registers[2], "xxyz")
    mad(registers[0], "yzw", v1, "z", registers[3], "xxyz", registers[0], "")
    mad(registers[0], "yzw", v1, "w", registers[4], "xxyz", registers[0], "")
    mul(registers[2], "xyz", registers[0], "yzww", registers[1], "")
    mad(registers[0], "yzw", registers[1], "xzxy", registers[0], "xzwy", -registers[2], "xxyz")
    mad(registers[1], "xyz", v8, "w", registers[0], "yzww", registers[7], "yzww")

    mov(registers[1], "w", constants[CV1_INDEX], "y")
    lrp(registers[2], "", registers[0], "x", registers[1], "", v0, "")
    mad(registers[1], "", registers[0], "x", -v4, "", v4, "")
    lrp(registers[3], "xyz", registers[0], "x", constants[CV1_INDEX], "yxxw", v1, "")
    mov(registers[0], "x", constants[CV1_INDEX], "x")
    slt(registers[0], "x", registers[0], "x", constants[CV0_INDEX], "x")
    add(registers[1], "", registers[1], "", constants[CV2_INDEX], "")
    mul(registers[0], "", registers[0], "x", registers[1], "")
    mova(address[0], "w", registers[0], "x")
    dp4(registers[1], "x", registers[2], "", constant(constants, 0, address[0].w), "")
    mova(address[0], "w", registers[0], "x")
    dp4(registers[1], "y", registers[2], "", constant(constants, 1, address[0].w), "")
    mova(address[0], "w", registers[0], "x")
    dp4(registers[1], "z", registers[2], "", constant(constants, 2, address[0].w), "")
    mul(registers[4], "xyz", registers[3], "x", registers[1], "")

    mova(address[0], "w", registers[0], "y")
    dp4(registers[5], "x", registers[2], "", constant(constants, 3, address[0].w), "")
    mova(address[0], "w", registers[0], "y")
    dp4(registers[5], "y", registers[2], "", constant(constants, 4, address[0].w), "")
    mova(address[0], "w", registers[0], "y")
    dp4(registers[5], "z", registers[2], "", constant(constants, 5, address[0].w), "")
    mad(registers[0], "xyw", registers[3], "y", registers[5], "xyzz", registers[4], "xyzz")
    add(registers[1], "w", -registers[3], "x", constants[CV1_INDEX], "y")
    add(registers[1], "w", -registers[3], "y", registers[1], "w")
    mova(address[0], "w", registers[0], "z")
    dp4(registers[3], "x", registers[2], "", constant(constants, 6, address[0].w), "")
    mova(address[0], "w", registers[0], "z")
    dp4(registers[3], "y", registers[2], "", constant(constants, 7, address[0].w), "")
    mova(address[0], "w", registers[0], "z")
    dp4(registers[3], "z", registers[2], "", constant(constants, 8, address[0].w), "")
    mad(registers[6], "xyz", registers[1], "w", registers[3], "", registers[0], "xyww")
    mov(registers[2], "xyz", registers[1], "")

    dp4(registers[0], "x", registers[2], "", constants[MODELVIEW_BASE + 0], "")
    dp4(registers[0], "y", registers[2], "", constants[MODELVIEW_BASE + 1], "")
    dp4(registers[0], "z", registers[2], "", constants[MODELVIEW_BASE + 2], "")
    dp4(registers[0], "w", registers[2], "", constants[MODELVIEW_BASE + 3], "")

    return ClothShaderOutput(
        o0=registers[0].copy(),
        o1=registers[0].copy(),
        o2=Float4(v10.x, v10.y, 0.0, 0.0),
        o3=Float4(v11.x, v11.y, 0.0, 0.0),
        o4=Float4(v12.x, v12.y, 0.0, 0.0),
    )
