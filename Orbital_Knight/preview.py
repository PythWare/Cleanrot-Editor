"""Standalone pyglet preview window for live G1M skinning feedback"""

from __future__ import annotations

import json, math, sys
from pathlib import Path

from .preview_scene import PreviewScene, build_preview_scene
from .reader import G1MModel, G1MParseError


VERTEX_SHADER_SOURCE = """#version 150 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

void main()
{
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    vertex_colors = colors;
}
"""

FRAGMENT_SHADER_SOURCE = """#version 150 core
in vec4 vertex_colors;
out vec4 final_color;

void main()
{
    final_color = vertex_colors;
}
"""


def safe_name(path: str | None) -> str:
    return Path(path).name if path else "No file"


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("Usage: python -m Orbital_Knight.preview <state-file>", file=sys.stderr)
        return 2

    state_path = Path(args[0])

    try:
        import pyglet

        pyglet.options["shadow_window"] = False
        import pyglet.gl as gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        from pyglet.math import Mat4, Vec3
    except ModuleNotFoundError:
        print(
            "pyglet is not installed for this Python interpreter. "
            "Install it into the interpreter that launches Cleanrot Editor.",
            file=sys.stderr,
        )
        return 1

    key = pyglet.window.key
    mouse = pyglet.window.mouse

    def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = vector
        length = math.sqrt(x * x + y * y + z * z)
        if length <= 1e-8:
            return (0.0, 0.0, 1.0)
        return (x / length, y / length, z / length)

    def subtract(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (left[0] - right[0], left[1] - right[1], left[2] - right[2])

    def cross(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    class PreviewWindow(pyglet.window.Window):
        def __init__(self, source_state_path: Path) -> None:
            config = None
            try:
                config = gl.Config(double_buffer=True, depth_size=24)
            except Exception:
                config = None

            kwargs = {
                "width": 1280,
                "height": 900,
                "caption": "Cleanrot Preview",
                "resizable": True,
            }
            if config is not None:
                kwargs["config"] = config

            try:
                super().__init__(**kwargs)
            except Exception:
                kwargs.pop("config", None)
                super().__init__(**kwargs)

            self.state_path = source_state_path
            self.state_mtime: float | None = None
            self.model_path: str | None = None
            self.model_mtime: float | None = None
            self.base_model: G1MModel | None = None
            self.scene: PreviewScene | None = None
            self.mesh_positions: list[list[tuple[float, float, float]]] = []
            self.selected_part_index: int | None = None
            self.selected_bone_index: int | None = None
            self.show_cloth_related = True
            self.show_noncloth_related = True
            self.selected_only = False
            self.hidden_part_indices: set[int] = set()
            self.status_message = "Waiting for preview state"
            self.hud_label = pyglet.text.Label(
                "",
                x=16,
                y=self.height - 16,
                anchor_x="left",
                anchor_y="top",
                multiline=True,
                width=max(self.width - 32, 100),
                color=(235, 240, 244, 255),
            )

            self.yaw = -25.0
            self.pitch = -12.0
            self.distance = 400.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.center = (0.0, 0.0, 0.0)
            self.radius = 100.0

            self.shader_program = ShaderProgram(
                Shader(VERTEX_SHADER_SOURCE, "vertex"),
                Shader(FRAGMENT_SHADER_SOURCE, "fragment"),
            )
            self.axis_vertex_list = None
            self.mesh_vertex_list = None

            gl.glClearColor(0.08, 0.09, 0.11, 1.0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            self.rebuild_axis_geometry()
            pyglet.clock.schedule_interval(self.poll_state, 0.1)
            self.load_state(force=True)

        def poll_state(self, _dt: float) -> None:
            self.load_state(force=False)

        def load_state(self, *, force: bool) -> None:
            if not self.state_path.exists():
                self.status_message = "Preview state file does not exist yet."
                self.clear_mesh_geometry()
                return

            stat = self.state_path.stat()
            if not force and self.state_mtime == stat.st_mtime:
                return
            self.state_mtime = stat.st_mtime

            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.status_message = f"Could not read preview state: {exc}"
                self.clear_mesh_geometry()
                return

            model_path = payload.get("model_path")
            if not model_path:
                self.status_message = "Preview state did not contain a model path."
                self.clear_mesh_geometry()
                return

            try:
                model_mtime = Path(model_path).stat().st_mtime
                if self.base_model is None or self.model_path != model_path or self.model_mtime != model_mtime:
                    self.base_model = G1MModel.from_path(model_path)
                    self.scene = build_preview_scene(self.base_model)
                    self.model_path = model_path
                    self.model_mtime = model_mtime
                    self.reset_camera()

                assert self.base_model is not None
                assert self.scene is not None
                self.mesh_positions = self.scene.skinned_mesh_positions(self.base_model)
                self.selected_part_index = payload.get("selected_part_index")
                self.selected_bone_index = payload.get("selected_bone_index")
                visibility = payload.get("visibility", {}) if isinstance(payload.get("visibility", {}), dict) else {}
                self.show_cloth_related = bool(visibility.get("show_cloth_related", True))
                self.show_noncloth_related = bool(visibility.get("show_noncloth_related", True))
                self.selected_only = bool(visibility.get("selected_only", False))
                self.hidden_part_indices = {
                    int(value)
                    for value in visibility.get("hidden_part_indices", [])
                    if isinstance(value, int) or (isinstance(value, str) and value.lstrip("-").isdigit())
                }
                self.status_message = self.build_status_text()
                self.set_caption(f"Cleanrot Preview {safe_name(model_path)}")
                self.rebuild_mesh_geometry()
            except (G1MParseError, OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
                self.status_message = f"Preview load failed: {exc}"
                self.clear_mesh_geometry()

        def build_status_text(self) -> str:
            if self.base_model is None or self.scene is None:
                return self.status_message

            selected_region = "All G1M submeshes"
            if self.selected_part_index is not None and 0 <= self.selected_part_index < len(self.base_model.body_parts):
                selected_region = self.base_model.body_parts[self.selected_part_index].name

            selected_bone = "None"
            if self.selected_bone_index is not None:
                selected_bone = str(self.selected_bone_index)

            visible_meshes = sum(1 for index in range(len(self.scene.meshes)) if self.mesh_visible(index))
            cloth_meshes = sum(1 for mesh in self.scene.meshes if mesh.is_cloth_related)
            noncloth_meshes = len(self.scene.meshes) - cloth_meshes
            visibility = (
                f"Visible: {visible_meshes}/{len(self.scene.meshes)} | "
                f"cloth linked {cloth_meshes}|non-cloth {noncloth_meshes}|hidden regions {len(self.hidden_part_indices)}"
            )
            lines = [
                f"{safe_name(self.model_path)}\n"
                f"Rendered submeshes: {len(self.scene.meshes)}|Unsupported: {len(self.scene.unsupported_submeshes)}\n"
                f"{visibility}\n"
                f"Selected region: {selected_region}\n"
                f"Selected bone anchor: {selected_bone}"
            ]
            lines.append("\nControls: Left drag rotate | Right drag pan | Wheel zoom | R reset camera")
            return "".join(lines)

        def delete_vertex_list(self, attribute_name: str) -> None:
            vertex_list = getattr(self, attribute_name)
            if vertex_list is None:
                return
            vertex_list.delete()
            setattr(self, attribute_name, None)

        def reset_camera(self) -> None:
            if self.scene is None:
                self.center = (0.0, 0.0, 0.0)
                self.radius = 100.0
                self.distance = 400.0
                self.rebuild_axis_geometry()
                return

            center = self.scene.center
            span = self.scene.span
            self.center = (center.x, center.y, center.z)
            self.radius = max(span.x, span.y, span.z, 1.0) * 0.6
            self.distance = max(self.radius * 2.5, 10.0)
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.rebuild_axis_geometry()

        def rebuild_axis_geometry(self) -> None:
            axis = max(self.radius * 0.5, 10.0)
            positions = [
                0.0, 0.0, 0.0,
                axis, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, axis, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, axis,
            ]
            colors = [
                0.85, 0.28, 0.25, 1.0,
                0.85, 0.28, 0.25, 1.0,
                0.22, 0.72, 0.40, 1.0,
                0.22, 0.72, 0.40, 1.0,
                0.25, 0.52, 0.90, 1.0,
                0.25, 0.52, 0.90, 1.0,
            ]

            self.delete_vertex_list("axis_vertex_list")
            self.axis_vertex_list = self.shader_program.vertex_list(
                6,
                gl.GL_LINES,
                position=("f", positions),
                colors=("f", colors),
            )

        def clear_mesh_geometry(self) -> None:
            self.delete_vertex_list("mesh_vertex_list")

        def mesh_visible(self, mesh_index: int) -> bool:
            assert self.scene is not None
            mesh = self.scene.meshes[mesh_index]
            if self.selected_only:
                if self.selected_part_index is None or self.selected_part_index not in mesh.body_part_indices:
                    return False
            if any(part_index in self.hidden_part_indices for part_index in mesh.body_part_indices):
                return False
            if mesh.is_cloth_related and not self.show_cloth_related:
                return False
            if not mesh.is_cloth_related and not self.show_noncloth_related:
                return False
            return True

        def mesh_color(self, mesh_index: int) -> tuple[float, float, float]:
            assert self.scene is not None
            mesh = self.scene.meshes[mesh_index]
            if self.selected_part_index is not None and self.selected_part_index in mesh.body_part_indices:
                return (0.91, 0.48, 0.24)
            if self.selected_bone_index is not None and self.selected_bone_index in mesh.bone_ids:
                return (0.26, 0.71, 0.73)
            if mesh.is_cloth_related:
                return (0.68, 0.54, 0.86)
            return (0.74, 0.78, 0.82)

        def rebuild_mesh_geometry(self) -> None:
            if self.scene is None:
                self.clear_mesh_geometry()
                return

            positions_flat: list[float] = []
            colors_flat: list[float] = []
            light = normalize((0.3, 0.7, 0.5))
            ambient = 0.34

            for mesh_index, mesh in enumerate(self.scene.meshes):
                if mesh_index >= len(self.mesh_positions) or not self.mesh_visible(mesh_index):
                    continue

                positions = self.mesh_positions[mesh_index]
                base_color = self.mesh_color(mesh_index)
                vertex_normals = [(0.0, 0.0, 0.0) for _ in positions]

                for a, b, c in mesh.triangles:
                    if not (
                        0 <= a < len(positions)
                        and 0 <= b < len(positions)
                        and 0 <= c < len(positions)
                    ):
                        continue

                    v0 = positions[a]
                    v1 = positions[b]
                    v2 = positions[c]
                    face_normal = cross(subtract(v1, v0), subtract(v2, v0))
                    for vertex_index in (a, b, c):
                        current = vertex_normals[vertex_index]
                        vertex_normals[vertex_index] = (
                            current[0] + face_normal[0],
                            current[1] + face_normal[1],
                            current[2] + face_normal[2],
                        )

                vertex_normals = [normalize(normal) for normal in vertex_normals]

                def shaded_color(vertex_index: int) -> tuple[float, float, float]:
                    if not 0 <= vertex_index < len(vertex_normals):
                        return base_color
                    normal = vertex_normals[vertex_index]
                    intensity = max(ambient, abs(normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]))
                    return (
                        min(base_color[0] * intensity, 1.0),
                        min(base_color[1] * intensity, 1.0),
                        min(base_color[2] * intensity, 1.0),
                    )

                for a, b, c in mesh.triangles:
                    if not (
                        0 <= a < len(positions)
                        and 0 <= b < len(positions)
                        and 0 <= c < len(positions)
                    ):
                        continue

                    v0 = positions[a]
                    v1 = positions[b]
                    v2 = positions[c]
                    shaded0 = shaded_color(a)
                    shaded1 = shaded_color(b)
                    shaded2 = shaded_color(c)

                    positions_flat.extend((
                        v0[0], v0[1], v0[2],
                        v1[0], v1[1], v1[2],
                        v2[0], v2[1], v2[2],
                    ))
                    colors_flat.extend((
                        shaded0[0], shaded0[1], shaded0[2], 1.0,
                        shaded1[0], shaded1[1], shaded1[2], 1.0,
                        shaded2[0], shaded2[1], shaded2[2], 1.0,
                    ))

            self.clear_mesh_geometry()
            if not positions_flat:
                return

            self.mesh_vertex_list = self.shader_program.vertex_list(
                len(positions_flat) // 3,
                gl.GL_TRIANGLES,
                position=("f", positions_flat),
                colors=("f", colors_flat),
            )

        def on_draw(self) -> None:
            self.clear()
            gl.glViewport(0, 0, self.width, self.height)

            self.projection = Mat4.perspective_projection(
                max(self.width, 1) / max(self.height, 1),
                max(self.radius * 0.01, 0.1),
                max(self.radius * 20.0, 100.0),
                fov=45.0,
            )
            self.view = (
                Mat4()
                .translate(Vec3(self.pan_x, self.pan_y, -self.distance))
                .rotate(math.radians(self.pitch), Vec3(1.0, 0.0, 0.0))
                .rotate(math.radians(self.yaw), Vec3(0.0, 1.0, 0.0))
                .translate(Vec3(-self.center[0], -self.center[1], -self.center[2]))
            )

            self.shader_program.use()
            self.draw_axes()
            self.draw_meshes()
            ShaderProgram.stop()

            self.projection = Mat4.orthogonal_projection(0, self.width, 0, self.height, -1.0, 1.0)
            self.view = Mat4()
            self.draw_hud()

        def draw_axes(self) -> None:
            if self.axis_vertex_list is None:
                return
            try:
                gl.glLineWidth(2.0)
            except Exception:
                pass
            self.axis_vertex_list.draw(gl.GL_LINES)

        def draw_meshes(self) -> None:
            if self.mesh_vertex_list is None:
                return
            self.mesh_vertex_list.draw(gl.GL_TRIANGLES)

        def draw_hud(self) -> None:
            gl.glDisable(gl.GL_DEPTH_TEST)

            self.hud_label.width = max(self.width - 32, 100)
            self.hud_label.x = 16
            self.hud_label.y = self.height - 16
            self.hud_label.text = self.status_message
            self.hud_label.draw()

            gl.glEnable(gl.GL_DEPTH_TEST)

        def on_mouse_drag(self, _x: int, _y: int, dx: int, dy: int, buttons: int, _modifiers: int) -> None:
            if buttons & mouse.LEFT:
                self.yaw += dx * 0.35
                self.pitch = max(-89.0, min(89.0, self.pitch - dy * 0.35))
            elif buttons & mouse.RIGHT:
                pan_scale = max(self.radius * 0.002, 0.01)
                self.pan_x += dx * pan_scale
                self.pan_y -= dy * pan_scale

        def on_mouse_scroll(self, _x: int, _y: int, _scroll_x: int, scroll_y: int) -> None:
            self.distance = max(self.radius * 0.2, self.distance * (0.92 ** scroll_y))

        def on_key_press(self, symbol: int, _modifiers: int) -> None:
            if symbol == key.R:
                self.reset_camera()

        def on_close(self) -> None:
            self.delete_vertex_list("axis_vertex_list")
            self.delete_vertex_list("mesh_vertex_list")
            self.shader_program.delete()
            super().on_close()

    window = PreviewWindow(state_path)
    pyglet.app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
