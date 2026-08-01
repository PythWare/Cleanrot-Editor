"""Tkinter GUI for the Cleanrot G1M vertex body editor"""

from __future__ import annotations

import json, os, site, subprocess, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from .reader import G1MModel, G1MParseError
from .extra_utils import (
    MorphTarget,
    VertexSculptControls,
    apply_morphs,
    body_parts_covered_by_submeshes,
    capture_deltas,
    disable_g1m_submeshes,
    load_morph,
    recompute_all_normals_tangents,
    save_morph,
    sculpt_g1m_vertex_bytes,
    parse_writable_preview_resources,
)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#fffaf3")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Card.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self.on_content_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel, add="+")

    def on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def on_mousewheel(self, event) -> None:
        if not self.winfo_ismapped():
            return
        delta = -1 * int(event.delta / 120) if event.delta else 0
        self.canvas.yview_scroll(delta, "units")


class ToolTip:
    """Tiny hover tooltip for compact UI help text"""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.after_id: str | None = None
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event=None) -> None:
        self.cancel()
        self.after_id = self.widget.after(self.delay_ms, self.show)

    def cancel(self) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def show(self) -> None:
        if self.tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except tk.TclError:
            return
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            wraplength=360,
            background="#fff8d7",
            foreground="#1f2427",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
        )
        label.pack()

    def hide(self, _event=None) -> None:
        self.cancel()
        if self.tip is not None:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


class CleanrotEditorApp(tk.Tk):
    """G1M vertex editor"""

    def __init__(self) -> None:
        super().__init__()
        self.title("Cleanrot Editor, G1M Body Studio")
        self.geometry("1320x1000")
        self.repo_root = Path(__file__).resolve().parents[1]
        self.sample_dir = self.repo_root
        self.model: G1MModel | None = None
        self.loaded_raw_data: bytes | None = None
        self.preview_sync_pending = False
        self.suspend_preview_traces = False
        self.preview_exit_reported = False
        self.preview_process: subprocess.Popen[str] | None = None
        self.preview_state_path = self.repo_root / "cleanrot_preview_state.json"
        self.preview_model_path = self.repo_root / "cleanrot_live_vertex_preview.g1m"
        self.preview_log_path = self.repo_root / "cleanrot_preview.log"
        self.last_preview_result = None

        self.file_var = tk.StringVar(value="No G1M file loaded.")
        self.counts_var = tk.StringVar(value="Sections 0|Bones 0|Submeshes 0|Vertex edits 0")
        self.status_var = tk.StringVar(value="Open a .g1m file to begin live vertex editing.")
        self.sections_var = tk.StringVar(value="Open a file to inspect G1M sections.")
        self.part_filter_var = tk.StringVar()

        self.target_bones_var = tk.StringVar(value="0,1")
        self.radius_var = tk.DoubleVar(value=25.0)
        self.min_weight_var = tk.DoubleVar(value=0.10)
        self.falloff_var = tk.DoubleVar(value=1.0)
        self.inflate_amount_var = tk.DoubleVar(value=0.0)
        self.smooth_strength_var = tk.DoubleVar(value=0.0)
        self.sculpt_mode_var = tk.StringVar(value="scale_offset")
        self.weld_vertices_var = tk.BooleanVar(value=False)
        self.auto_preview_var = tk.BooleanVar(value=True)
        self.show_cloth_related_var = tk.BooleanVar(value=True)
        self.show_noncloth_related_var = tk.BooleanVar(value=True)
        self.selected_only_var = tk.BooleanVar(value=False)
        self.hidden_part_indices: set[int] = set()
        self.disabled_submesh_indices: set[int] = set()
        self.visibility_status_var = tk.StringVar(value="Visibility: all regions shown.")
        self.scale_vars = {axis: tk.DoubleVar(value=1.0) for axis in "xyz"}
        self.offset_vars = {axis: tk.DoubleVar(value=0.0) for axis in "xyz"}

        self.loaded_morphs: list[tuple[Path | None, MorphTarget, tk.DoubleVar]] = []
        self.morph_status_var = tk.StringVar(value="No sliders loaded.")
        self.section_details_text: tk.Text | None = None

        self.configure_style()
        self.build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.part_filter_var.trace_add("write", self.refresh_region_tree)
        for variable in [
            self.target_bones_var,
            self.radius_var,
            self.min_weight_var,
            self.falloff_var,
            self.inflate_amount_var,
            self.smooth_strength_var,
            self.sculpt_mode_var,
            self.weld_vertices_var,
            self.show_cloth_related_var,
            self.show_noncloth_related_var,
            self.selected_only_var,
            *self.scale_vars.values(),
            *self.offset_vars.values(),
        ]:
            variable.trace_add("write", self.on_live_control_changed)

    def configure_style(self) -> None:
        self.palette = {
            "shell": "#f4efe6",
            "panel": "#fffaf3",
            "accent": "#1f5b63",
            "accent_soft": "#d9e8e8",
            "ink": "#1f2427",
            "muted": "#586166",
            "edge": "#cfc5b4",
        }
        style = ttk.Style(self)
        style.theme_use("clam")
        self.configure(background=self.palette["shell"])
        style.configure(".", background=self.palette["shell"], foreground=self.palette["ink"])
        style.configure("Card.TFrame", background=self.palette["panel"], relief="solid", borderwidth=1)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20), foreground=self.palette["accent"])
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground=self.palette["muted"])
        style.configure("PanelTitle.TLabel", font=("Segoe UI Semibold", 11), foreground=self.palette["accent"])
        style.configure("Body.TLabel", background=self.palette["panel"], foreground=self.palette["ink"])
        style.configure("Muted.TLabel", background=self.palette["panel"], foreground=self.palette["muted"])
        style.configure("Hint.TLabel", background=self.palette["panel"], foreground=self.palette["accent"], font=("Segoe UI Semibold", 9))
        style.configure("Accent.TButton", background=self.palette["accent"], foreground="white", padding=(12, 7))
        style.map("Accent.TButton", background=[("active", "#2c7580"), ("pressed", "#184850")])
        style.configure("Tool.TButton", padding=(10, 6))
        style.configure("TLabelframe", background=self.palette["panel"], bordercolor=self.palette["edge"])
        style.configure("TLabelframe.Label", background=self.palette["panel"], foreground=self.palette["accent"])
        style.configure(
            "Treeview",
            background="white",
            fieldbackground="white",
            foreground=self.palette["ink"],
            rowheight=24,
            bordercolor=self.palette["edge"],
        )
        style.configure("Treeview.Heading", background=self.palette["accent_soft"], foreground=self.palette["ink"])
        style.map("Treeview", background=[("selected", "#c6ddde")], foreground=[("selected", self.palette["ink"])])

    def build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Cleanrot Editor, G1M Body Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="G1M body modding Editor for characters, weapons, items, buildings, etc.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 12))
        ttk.Button(toolbar, text="Open G1M", command=self.open_g1m, style="Accent.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Preview 3D", command=self.open_preview, style="Tool.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Export Shape", command=self.export_current_shape, style="Tool.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Commit Shape", command=self.commit_current_shape, style="Tool.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Recompute N/T", command=self.recompute_normals_tangents, style="Tool.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Reset Live Controls", command=self.reset_live_controls, style="Tool.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Reset Model", command=self.reset_model, style="Tool.TButton").pack(side="left")

        summary = ttk.Frame(outer, style="Card.TFrame", padding=14)
        summary.pack(fill="x", pady=(0, 12))
        ttk.Label(summary, textvariable=self.file_var, style="Body.TLabel").pack(anchor="w")
        ttk.Label(summary, textvariable=self.counts_var, style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(summary, textvariable=self.sections_var, style="Muted.TLabel", wraplength=1280, justify="left").pack(anchor="w", pady=(4, 0))

        panes = ttk.Panedwindow(outer, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes, style="Card.TFrame", padding=12)
        right = ttk.Frame(panes, style="Card.TFrame", padding=0)
        panes.add(left, weight=3)
        panes.add(right, weight=4)

        self.build_region_panel(left)
        self.build_editor_panel(right)

        status = ttk.Frame(outer, style="Card.TFrame", padding=10)
        status.pack(fill="x", pady=(12, 0))
        ttk.Label(status, textvariable=self.status_var, style="Muted.TLabel", wraplength=1280).pack(anchor="w")

    def build_region_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Mesh Regions", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text=(
                "Pick a body/mesh region to limit sculpting. Expand a region to reach its individual "
                "submeshes, which can be disabled one layer at a time. Use All G1M submeshes when "
                "uncertain then tighten with target bones and min weight."
            ),
            style="Muted.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        ttk.Entry(parent, textvariable=self.part_filter_var).pack(fill="x", pady=(0, 8))
        columns = ("kind", "visible", "bones", "submeshes")
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True)
        self.part_tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", selectmode="browse")
        self.part_tree.heading("#0", text="Region/Submesh")
        self.part_tree.heading("kind", text="Kind")
        self.part_tree.heading("visible", text="Visible")
        self.part_tree.heading("bones", text="Bones")
        self.part_tree.heading("submeshes", text="Sub/Faces")
        self.part_tree.column("#0", width=250, minwidth=170, anchor="w", stretch=True)
        self.part_tree.column("kind", width=78, minwidth=64, anchor="center", stretch=False)
        self.part_tree.column("visible", width=74, minwidth=64, anchor="center", stretch=False)
        self.part_tree.column("bones", width=58, minwidth=50, anchor="center", stretch=False)
        self.part_tree.column("submeshes", width=78, minwidth=62, anchor="center", stretch=False)
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.part_tree.yview)
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.part_tree.xview)
        self.part_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.part_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.part_tree.bind("<<TreeviewSelect>>", self.on_region_selected)

        visibility = ttk.LabelFrame(parent, text="Preview Visibility", padding=8)
        visibility.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            visibility,
            text="Show non-cloth/body meshes",
            variable=self.show_noncloth_related_var,
            command=self.on_visibility_changed,
        ).pack(anchor="w")
        ttk.Checkbutton(
            visibility,
            text="Show cloth/NUN-linked meshes",
            variable=self.show_cloth_related_var,
            command=self.on_visibility_changed,
        ).pack(anchor="w")
        ttk.Checkbutton(
            visibility,
            text="Selected region only",
            variable=self.selected_only_var,
            command=self.on_visibility_changed,
        ).pack(anchor="w")
        action_row = ttk.Frame(visibility)
        action_row.pack(fill="x", pady=(6, 0))
        ttk.Button(action_row, text="Hide Selected", command=self.hide_selected_region, style="Tool.TButton").pack(side="left")
        ttk.Button(action_row, text="Show All", command=self.show_all_regions, style="Tool.TButton").pack(side="left", padx=(8, 0))
        disable_button = ttk.Button(
            action_row,
            text="Disable",
            command=self.toggle_disable_selected_region,
            style="Tool.TButton",
        )
        disable_button.pack(side="left", padx=(8, 0))
        ToolTip(
            disable_button,
            "Blank the selected row's faces so it stops rendering ingame. Select a region to blank "
            "all of it or expand the region and select a single submesh to drop just that layer. "
            "Click again to re-enable. File size is unchanged & the change is "
            "included in Export Shape and Commit Sculpt.",
        )
        ttk.Label(visibility, textvariable=self.visibility_status_var, style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(6, 0))

    def build_editor_panel(self, parent: ttk.Frame) -> None:
        scroll = ScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)
        content = scroll.content
        content.configure(padding=8)

        sculpt = ttk.LabelFrame(content, text="Live Vertex Sculpt", padding=8)
        sculpt.pack(fill="x", pady=(0, 12))
        sculpt_intro = ttk.Label(
            sculpt,
            text="G1MG vertex sculpting.",
            style="Muted.TLabel",
            justify="left",
        )
        sculpt_intro.grid(column=0, row=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.wrap_label_to_parent(sculpt_intro, sculpt, min_width=240)

        target_label = ttk.Label(sculpt, text="Target Bones", style="Body.TLabel")
        target_label.grid(column=0, row=1, sticky="w", pady=2)
        ToolTip(target_label, "Comma separated bone IDs. Cleanrot edits only vertices weighted to these bones above Min Weight.")
        target_row = ttk.Frame(sculpt)
        target_row.grid(column=1, row=1, sticky="w", padx=(8, 0), pady=2)
        ttk.Entry(target_row, textvariable=self.target_bones_var, width=18).pack(side="left")
        target_hint = ttk.Label(target_row, text="?", style="Hint.TLabel", cursor="question_arrow")
        target_hint.pack(side="left", padx=(6, 0))
        ToolTip(target_hint, "i.e, 0,1 if modding multple areas, type only 1 number if modding 1 area only")

        mode_label = ttk.Label(sculpt, text="Mode", style="Body.TLabel")
        mode_label.grid(column=0, row=3, sticky="w", pady=2)
        ToolTip(mode_label, "scale_offset = scaling/offset, inflate = push along normals, smooth = relax vertices.")
        mode_row = ttk.Frame(sculpt)
        mode_row.grid(column=1, row=3, sticky="w", padx=(8, 0), pady=2)
        mode_box = ttk.Combobox(
            mode_row,
            textvariable=self.sculpt_mode_var,
            state="readonly",
            values=("scale_offset", "inflate", "smooth"),
            width=14,
        )
        mode_box.pack(side="left")
        mode_hint = ttk.Label(mode_row, text="?", style="Hint.TLabel", cursor="question_arrow")
        mode_hint.pack(side="left", padx=(6, 0))
        ToolTip(mode_hint, "scale_offset uses Scale/Offset. Inflate uses normals. Smooth averages targeted vertices toward neighbors.")

        self.build_scalar_control(sculpt, "Radius", self.radius_var, 0.0, 120.0, 1.0, 4)
        self.build_scalar_control(sculpt, "Min Weight", self.min_weight_var, 0.0, 1.0, 0.01, 5)
        self.build_scalar_control(sculpt, "Falloff", self.falloff_var, 0.1, 4.0, 0.1, 6)
        self.build_scalar_control(sculpt, "Inflate Amount", self.inflate_amount_var, -30.0, 30.0, 0.1, 7)
        self.build_scalar_control(sculpt, "Smooth Strength", self.smooth_strength_var, 0.0, 1.0, 0.01, 8)
        self.build_vector_editor(sculpt, "Scale Around Weighted Center", self.scale_vars, 9, 0.25, 2.5, 0.01)
        self.build_vector_editor(sculpt, "Offset", self.offset_vars, 13, -20.0, 20.0, 0.1)

        options = ttk.Frame(sculpt)
        options.grid(column=0, row=17, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(options, text="Auto update preview", variable=self.auto_preview_var, command=self.schedule_preview_sync).pack(side="left")
        ttk.Checkbutton(
            options,
            text="Move welded duplicate vertices too",
            variable=self.weld_vertices_var,
            command=self.schedule_preview_sync,
        ).pack(side="left", padx=(18, 0))

        actions = ttk.Frame(sculpt)
        actions.grid(column=0, row=18, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Refresh Preview", command=self.force_preview_sync, style="Tool.TButton").pack(side="left")
        ttk.Button(actions, text="Commit Sculpt", command=self.commit_current_shape, style="Accent.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Capture Slider", command=self.capture_live_sculpt_as_slider, style="Tool.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Reset Controls", command=self.reset_live_controls, style="Tool.TButton").pack(side="left", padx=(8, 0))
        sculpt.columnconfigure(0, weight=0, minsize=125)
        sculpt.columnconfigure(1, weight=0, minsize=250)
        sculpt.columnconfigure(2, weight=1)

        sliders = ttk.LabelFrame(content, text="Vertex Slider Stack", padding=10)
        sliders.pack(fill="x", pady=(0, 12))
        ttk.Label(
            sliders,
            text="Sliders are loaded/created inside Cleanrot and blended live. JSON files are optional.",
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        slider_actions = ttk.Frame(sliders)
        slider_actions.pack(fill="x")
        ttk.Button(slider_actions, text="Load Slider File", command=self.load_slider_file, style="Tool.TButton").pack(side="left")
        ttk.Button(slider_actions, text="Save Selected Slider", command=self.save_selected_slider, style="Tool.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(slider_actions, text="Bake Slider Mix to Model", command=self.commit_current_shape, style="Accent.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(slider_actions, text="Clear Sliders", command=self.clear_sliders, style="Tool.TButton").pack(side="left", padx=(8, 0))
        self.slider_list_frame = ttk.Frame(sliders)
        self.slider_list_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(sliders, textvariable=self.morph_status_var, style="Muted.TLabel", wraplength=720).pack(anchor="w", pady=(8, 0))

        coverage = ttk.LabelFrame(content, text="G1M Coverage/Engine Reference", padding=10)
        coverage.pack(fill="both", expand=False, pady=(0, 12))
        ttk.Label(
            coverage,
            text=(
                "Unsupported optional chunks are preserved byte for byte and surfaced here."
            ),
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        coverage_body = ttk.Frame(coverage)
        coverage_body.pack(fill="both", expand=True)
        self.section_details_text = tk.Text(
            coverage_body,
            height=9,
            wrap="word",
            relief="solid",
            borderwidth=1,
            background="white",
            foreground=self.palette["ink"],
        )
        coverage_scroll = ttk.Scrollbar(coverage_body, orient="vertical", command=self.section_details_text.yview)
        self.section_details_text.configure(yscrollcommand=coverage_scroll.set)
        self.section_details_text.grid(row=0, column=0, sticky="nsew")
        coverage_scroll.grid(row=0, column=1, sticky="ns")
        coverage_body.rowconfigure(0, weight=1)
        coverage_body.columnconfigure(0, weight=1)
        self.section_details_text.insert("1.0", "Open a G1M to inspect section coverage.")
        self.section_details_text.configure(state="disabled")

        notes = ttk.LabelFrame(content, text="Workflow", padding=10)
        notes.pack(fill="x", expand=False)
        workflow_label = ttk.Label(
            notes,
            text=(
                "Open G1M, Preview 3D, pick a mesh region or use All, set Target Bones and sculpt mode, "
                "adjust sliders, and export. Commit Sculpt bakes the current live result as the new base. "
                "Keep welded duplicates off for isolated body edits, enable it when closing seams."
            ),
            style="Muted.TLabel",
            justify="left",
        )
        workflow_label.pack(anchor="w", fill="x")
        self.wrap_label_to_parent(workflow_label, notes, min_width=240)

    def wrap_label_to_parent(self, label: ttk.Label, parent: tk.Widget, *, padding: int = 22, min_width: int = 220) -> None:
        def sync(_event=None) -> None:
            try:
                width = parent.winfo_width()
                if width <= 1:
                    width = label.winfo_width()
                label.configure(wraplength=max(min_width, width - padding))
            except tk.TclError:
                pass

        parent.bind("<Configure>", sync, add="+")
        label.bind("<Configure>", sync, add="+")
        self.after_idle(sync)

    def control_hint_text(self, label: str) -> str:
        return {
            "Radius": "Spatial falloff radius around the weighted bone center. Set 0 to disable distance falloff.",
            "Min Weight": "Minimum total weight to the target bones before a vertex can be edited.",
            "Falloff": "Controls how quickly the effect fades across the radius. Higher values make a tighter effect.",
            "Inflate Amount": "Used by inflate mode. Positive pushes outward along normals, negative pulls inward.",
            "Smooth Strength": "Used by smooth mode. 0 does nothing, 1 moves targeted vertices fully toward neighbor average.",
            "Scale Around Weighted Center": "Used by scale_offset mode. Scales targeted vertices around the weighted target bone center.",
            "Offset": "Used by scale_offset mode. Adds a direct X/Y/Z offset after scaling.",
        }.get(label, "")

    def build_scalar_control(self, parent: ttk.LabelFrame, label: str, variable: tk.DoubleVar, low: float, high: float, inc: float, row: int) -> None:
        label_widget = ttk.Label(parent, text=label, style="Body.TLabel")
        label_widget.grid(column=0, row=row, sticky="w", pady=2)
        hint_text = self.control_hint_text(label)
        if hint_text:
            ToolTip(label_widget, hint_text)
        control_row = ttk.Frame(parent)
        control_row.grid(column=1, row=row, sticky="w", padx=(8, 0), pady=2)
        slider = ttk.Scale(control_row, from_=low, to=high, orient="horizontal", variable=variable, length=165)
        slider.pack(side="left")
        ttk.Spinbox(control_row, from_=low, to=high, increment=inc, textvariable=variable, width=7).pack(side="left", padx=(6, 0))
        if hint_text:
            hint = ttk.Label(control_row, text="?", style="Hint.TLabel", cursor="question_arrow")
            hint.pack(side="left", padx=(6, 0))
            ToolTip(hint, hint_text)

    def build_vector_editor(self, parent: ttk.LabelFrame, label: str, variables: dict[str, tk.DoubleVar], start_row: int, low: float, high: float, inc: float) -> None:
        title_row = ttk.Frame(parent)
        title_row.grid(column=0, row=start_row, columnspan=2, sticky="w", pady=(6, 2))
        title_label = ttk.Label(title_row, text=label, style="Body.TLabel")
        title_label.pack(side="left")
        hint_text = self.control_hint_text(label)
        if hint_text:
            ToolTip(title_label, hint_text)
            hint = ttk.Label(title_row, text="?", style="Hint.TLabel", cursor="question_arrow")
            hint.pack(side="left", padx=(6, 0))
            ToolTip(hint, hint_text)
        for offset, axis in enumerate("xyz", start=1):
            row = start_row + offset
            ttk.Label(parent, text=axis.upper(), style="Body.TLabel").grid(column=0, row=row, sticky="w", pady=2)
            control_row = ttk.Frame(parent)
            control_row.grid(column=1, row=row, sticky="w", padx=(8, 0), pady=2)
            ttk.Scale(control_row, from_=low, to=high, orient="horizontal", variable=variables[axis], length=165).pack(side="left")
            ttk.Spinbox(control_row, from_=low, to=high, increment=inc, textvariable=variables[axis], width=7).pack(side="left", padx=(6, 0))

    def default_open_dir(self) -> Path:
        return self.sample_dir if self.sample_dir.exists() else self.repo_root

    def open_g1m(self) -> None:
        path = filedialog.askopenfilename(
            title="Open G1M file",
            initialdir=self.default_open_dir(),
            filetypes=[("G1M models", "*.g1m"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.model = G1MModel.from_path(path)
        except G1MParseError as exc:
            messagebox.showerror("Unable to open file", str(exc), parent=self)
            return
        self.loaded_raw_data = self.model.raw_data
        self.sample_dir = Path(path).parent
        self.loaded_morphs.clear()
        self.hidden_part_indices.clear()
        self.disabled_submesh_indices.clear()
        self.refresh_all()
        self.status_var.set("Loaded G1M.")
        self.force_preview_sync()

    def selected_row(self) -> str | None:
        selection = self.part_tree.selection() if getattr(self, "part_tree", None) else ()
        return selection[0] if selection else None

    def selected_part(self):
        """Body part for the selected row, a submesh row resolves to its parent region"""

        if not self.model:
            return None
        item = self.selected_row()
        if item is None or item == "all":
            return None
        if item.startswith("sub:"):
            item = f"part:{item.split(':')[1]}"
        if item.startswith("part:"):
            index = int(item.split(":", 1)[1])
            if 0 <= index < len(self.model.body_parts):
                return self.model.body_parts[index]
        return None

    def selected_submesh_index(self) -> int | None:
        """Submesh index when a submesh row is selected, otherwise None"""

        if not self.model:
            return None
        item = self.selected_row()
        if item is None or not item.startswith("sub:"):
            return None
        submesh_index = int(item.split(":")[2])
        if 0 <= submesh_index < len(self.model.submeshes):
            return submesh_index
        return None

    def selected_part_index(self) -> int | None:
        part = self.selected_part()
        return part.index if part is not None else None

    def parse_target_bones(self) -> frozenset[int]:
        text = self.target_bones_var.get().strip()
        if not text:
            return frozenset()
        bones: set[int] = set()
        for piece in text.replace(";", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                bones.add(int(piece, 0))
            except ValueError as exc:
                raise ValueError(f"Invalid bone id: {piece!r}") from exc
        return frozenset(bones)

    def live_sculpt_controls(self) -> VertexSculptControls | None:
        try:
            target_bones = self.parse_target_bones()
        except ValueError:
            return None
        if not target_bones:
            return None
        return VertexSculptControls(
            target_bones=target_bones,
            body_part_index=self.selected_part_index(),
            offset=(self.offset_vars["x"].get(), self.offset_vars["y"].get(), self.offset_vars["z"].get()),
            scale=(self.scale_vars["x"].get(), self.scale_vars["y"].get(), self.scale_vars["z"].get()),
            radius=self.radius_var.get(),
            min_weight=self.min_weight_var.get(),
            falloff_power=self.falloff_var.get(),
            weld_vertices=self.weld_vertices_var.get(),
            sculpt_mode=self.sculpt_mode_var.get(),
            inflate_amount=self.inflate_amount_var.get(),
            smooth_strength=self.smooth_strength_var.get(),
        )

    def active_morph_pairs(self) -> list[tuple[MorphTarget, float]]:
        pairs: list[tuple[MorphTarget, float]] = []
        for _path, morph, variable in self.loaded_morphs:
            value = float(variable.get())
            if abs(value) > 1e-8:
                pairs.append((morph, value))
        return pairs

    def working_bytes(self) -> bytes:
        if not self.model:
            raise G1MParseError("No model loaded.")
        data = self.model.raw_data
        morph_pairs = self.active_morph_pairs()
        if morph_pairs:
            temp = G1MModel.from_bytes(data, source_path=self.model.source_path)
            data = apply_morphs(temp, morph_pairs)
        controls = self.live_sculpt_controls()
        self.last_preview_result = None
        if controls and not controls.is_neutral():
            temp = G1MModel.from_bytes(data, source_path=self.model.source_path)
            data, self.last_preview_result = sculpt_g1m_vertex_bytes(temp, controls)
        if self.disabled_submesh_indices:
            temp = G1MModel.from_bytes(data, source_path=self.model.source_path)
            data, _disable_result = disable_g1m_submeshes(temp, self.disabled_submesh_indices)
        return data

    def on_live_control_changed(self, *_args) -> None:
        if self.suspend_preview_traces or not self.auto_preview_var.get():
            return
        self.schedule_preview_sync()

    def schedule_preview_sync(self) -> None:
        if self.preview_sync_pending:
            return
        self.preview_sync_pending = True
        self.after_idle(self.flush_preview_sync)

    def force_preview_sync(self) -> None:
        self.preview_sync_pending = False
        self.sync_preview_state()

    def flush_preview_sync(self) -> None:
        self.preview_sync_pending = False
        self.sync_preview_state()

    def sync_preview_state(self) -> None:
        if not self.model or not self.model.source_path:
            return
        try:
            data = self.working_bytes()
            self.preview_model_path.write_bytes(data)
            payload = {
                "format": "marylcian-preview-state-vertex-1",
                "model_path": str(self.preview_model_path),
                "selected_part_index": self.selected_part_index(),
                "selected_bone_index": None,
                "preset": {"format": "marylcian-preset-1", "bones": {}},
                "visibility": {
                    "show_cloth_related": bool(self.show_cloth_related_var.get()),
                    "show_noncloth_related": bool(self.show_noncloth_related_var.get()),
                    "selected_only": bool(self.selected_only_var.get()),
                    "hidden_part_indices": sorted(self.hidden_part_indices),
                },
            }
            self.preview_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.refresh_counts()
            if self.last_preview_result is not None:
                result = self.last_preview_result
                self.status_var.set(
                    f"Live preview updated: {result.patched_vertex_count} vertices, "
                    f"{result.affected_submesh_count} submeshes."
                )
        except Exception as exc:
            self.status_var.set(f"Live preview update failed: {exc}")

    def commit_current_shape(self) -> None:
        if not self.model:
            return
        try:
            data = self.working_bytes()
        except Exception as exc:
            messagebox.showerror("Unable to commit shape", str(exc), parent=self)
            return
        self.model = G1MModel.from_bytes(data, source_path=self.model.source_path)
        disabled_count = len(self.disabled_submesh_indices)
        self.disabled_submesh_indices.clear()
        self.clear_live_shape_without_preview()
        self.refresh_all()
        if disabled_count:
            self.status_var.set(
                f"Committed current vertex shape plus {disabled_count} blanked submesh(es) into the loaded G1M state."
            )
        else:
            self.status_var.set("Committed current vertex shape into the loaded G1M state.")
        self.force_preview_sync()

    def export_current_shape(self) -> None:
        if not self.model:
            messagebox.showinfo("No model loaded", "Open a G1M file before exporting.", parent=self)
            return
        default_name = "vertex_patched.g1m"
        if self.model.source_path:
            default_name = f"{self.model.source_path.stem}_vertex_patched.g1m"
        path = filedialog.asksaveasfilename(
            title="Export current vertex-shaped G1M",
            initialdir=self.default_open_dir(),
            initialfile=default_name,
            defaultextension=".g1m",
            filetypes=[("G1M models", "*.g1m")],
        )
        if not path:
            return
        try:
            Path(path).write_bytes(self.working_bytes())
        except Exception as exc:
            messagebox.showerror("Unable to export", str(exc), parent=self)
            return
        if self.disabled_submesh_indices:
            self.status_var.set(
                f"Exported current vertex shape plus {len(self.disabled_submesh_indices)} blanked submesh(es) to {path}."
            )
            return
        self.status_var.set(f"Exported current vertex shape to {path}.")

    def recompute_normals_tangents(self) -> None:
        if not self.model:
            return
        try:
            data = recompute_all_normals_tangents(self.model)
            self.model = G1MModel.from_bytes(data, source_path=self.model.source_path)
        except Exception as exc:
            messagebox.showerror("Unable to recompute normals/tangents", str(exc), parent=self)
            return
        self.refresh_all()
        self.status_var.set("Recomputed normals and tangents on the committed model state.")
        self.force_preview_sync()

    def reset_model(self) -> None:
        if not self.model or self.loaded_raw_data is None:
            return
        if not messagebox.askyesno("Reset model", "Reset the loaded G1M back to the file as opened?", parent=self):
            return
        self.model = G1MModel.from_bytes(self.loaded_raw_data, source_path=self.model.source_path)
        self.hidden_part_indices.clear()
        self.disabled_submesh_indices.clear()
        self.clear_live_shape_without_preview()
        self.refresh_all()
        self.status_var.set("Reset model back to the originally opened bytes.")
        self.force_preview_sync()

    def reset_live_controls(self) -> None:
        self.suspend_preview_traces = True
        try:
            self.radius_var.set(25.0)
            self.min_weight_var.set(0.10)
            self.falloff_var.set(1.0)
            self.inflate_amount_var.set(0.0)
            self.smooth_strength_var.set(0.0)
            self.sculpt_mode_var.set("scale_offset")
            self.weld_vertices_var.set(False)
            for axis in "xyz":
                self.scale_vars[axis].set(1.0)
                self.offset_vars[axis].set(0.0)
        finally:
            self.suspend_preview_traces = False
        self.status_var.set("Reset live sculpt controls.")
        self.force_preview_sync()

    def clear_live_shape_without_preview(self) -> None:
        self.suspend_preview_traces = True
        try:
            for _path, _morph, variable in self.loaded_morphs:
                variable.set(0.0)
            self.inflate_amount_var.set(0.0)
            self.smooth_strength_var.set(0.0)
            self.sculpt_mode_var.set("scale_offset")
            for axis in "xyz":
                self.scale_vars[axis].set(1.0)
                self.offset_vars[axis].set(0.0)
            self.weld_vertices_var.set(False)
        finally:
            self.suspend_preview_traces = False
        self.refresh_slider_list()

    def capture_live_sculpt_as_slider(self) -> None:
        if not self.model:
            return
        controls = self.live_sculpt_controls()
        if controls is None or controls.is_neutral():
            messagebox.showinfo("No live sculpt", "Set a non-neutral live sculpt before capturing a slider.", parent=self)
            return
        name = simpledialog.askstring("Slider name", "Name this vertex slider:", parent=self)
        if not name:
            return
        try:
            vertex_buffers, _attribute_sets, _index_buffers = parse_writable_preview_resources(self.model)
            sculpted, result = sculpt_g1m_vertex_bytes(self.model, controls)
            morph = capture_deltas(
                self.model.raw_data,
                sculpted,
                vertex_buffers,
                name=name.strip(),
                bone_hint=next(iter(sorted(controls.target_bones)), None),
                source_file=self.model.source_path.name if self.model.source_path else "",
            )
        except Exception as exc:
            messagebox.showerror("Unable to capture slider", str(exc), parent=self)
            return
        if not morph.deltas:
            messagebox.showinfo("No vertices captured", "The current sculpt did not move any vertices.", parent=self)
            return
        variable = tk.DoubleVar(value=1.0)
        variable.trace_add("write", self.on_live_control_changed)
        self.loaded_morphs.append((None, morph, variable))
        self.reset_live_controls()
        self.refresh_slider_list()
        self.status_var.set(f"Captured slider '{morph.name}' with {len(morph.deltas)} vertex deltas ({result.patched_vertex_count} moved).")
        self.force_preview_sync()

    def load_slider_file(self) -> None:
        if not self.model:
            messagebox.showinfo("No model loaded", "Open a G1M file before loading vertex sliders.", parent=self)
            return
        paths = filedialog.askopenfilenames(
            title="Load Cleanrot vertex slider file(s)",
            initialdir=self.default_open_dir(),
            filetypes=[("Cleanrot vertex sliders", "*.json"), ("All files", "*.*")],
        )
        if not paths:
            return
        loaded = 0
        for raw_path in paths:
            path = Path(raw_path)
            try:
                morph = load_morph(path)
            except Exception as exc:
                messagebox.showerror("Unable to load slider", f"{path}: {exc}", parent=self)
                continue
            variable = tk.DoubleVar(value=0.0)
            variable.trace_add("write", self.on_live_control_changed)
            self.loaded_morphs.append((path, morph, variable))
            loaded += 1
        self.refresh_slider_list()
        self.status_var.set(f"Loaded {loaded} vertex slider file(s).")
        self.force_preview_sync()

    def save_selected_slider(self) -> None:
        if not self.loaded_morphs:
            messagebox.showinfo("No sliders", "Create or load a slider first.", parent=self)
            return
        path, morph, variable = self.loaded_morphs[-1]
        default_name = f"{morph.name or 'cleanrot_slider'}.json"
        path = filedialog.asksaveasfilename(
            title="Save last vertex slider",
            initialdir=self.default_open_dir(),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("Cleanrot vertex sliders", "*.json")],
        )
        if not path:
            return
        try:
            save_morph(morph, path)
        except OSError as exc:
            messagebox.showerror("Unable to save slider", str(exc), parent=self)
            return
        self.loaded_morphs[-1] = (Path(path), morph, self.loaded_morphs[-1][2])
        self.refresh_slider_list()
        self.status_var.set(f"Saved slider '{morph.name}' to {path}.")

    def clear_sliders(self) -> None:
        self.loaded_morphs.clear()
        self.refresh_slider_list()
        self.status_var.set("Cleared all loaded vertex sliders.")
        self.force_preview_sync()

    def refresh_slider_list(self) -> None:
        for child in self.slider_list_frame.winfo_children():
            child.destroy()
        if not self.loaded_morphs:
            self.morph_status_var.set("No sliders loaded.")
            return
        for index, (_path, morph, variable) in enumerate(self.loaded_morphs):
            row = ttk.Frame(self.slider_list_frame)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=morph.name, style="Body.TLabel", width=24).pack(side="left")
            ttk.Scale(row, from_=-1.0, to=1.0, orient="horizontal", variable=variable).pack(side="left", fill="x", expand=True, padx=(10, 8))
            ttk.Spinbox(row, from_=-1.0, to=1.0, increment=0.01, textvariable=variable, width=8).pack(side="left")
            ttk.Label(row, text=f"{len(morph.deltas)} verts", style="Muted.TLabel", width=12).pack(side="left", padx=(8, 0))
            ttk.Button(row, text="Remove", command=lambda i=index: self.remove_slider(i), style="Tool.TButton").pack(side="left", padx=(8, 0))
        self.morph_status_var.set(f"{len(self.loaded_morphs)} slider(s) loaded. Export uses current slider values.")

    def remove_slider(self, index: int) -> None:
        if 0 <= index < len(self.loaded_morphs):
            removed = self.loaded_morphs.pop(index)[1]
            self.refresh_slider_list()
            self.status_var.set(f"Removed slider '{removed.name}'.")
            self.force_preview_sync()

    def refresh_all(self) -> None:
        self.refresh_counts()
        self.refresh_region_tree()
        self.refresh_slider_list()
        self.refresh_section_details()
        self.refresh_visibility_status()

    def refresh_counts(self) -> None:
        if not self.model:
            self.counts_var.set("Sections 0|Bones 0|Submeshes 0|Vertex edits 0")
            self.sections_var.set("Open a file to inspect G1M sections.")
            return
        live_count = 0
        for path, morph, variable in self.loaded_morphs:
            if abs(float(variable.get())) > 1e-8:
                live_count += len(morph.deltas)
        if self.last_preview_result is not None:
            live_count += int(self.last_preview_result.patched_vertex_count)
        cloth_parts = sum(1 for part in self.model.body_parts if self.part_is_cloth_related(part))
        self.counts_var.set(
            f"Sections {len(self.model.sections)} | Bones {len(self.model.bones)} | "
            f"Submeshes {len(self.model.submeshes)} | Live vertex edits {live_count} | "
            f"cloth-linked regions {cloth_parts}"
        )
        section_text = self.model.section_summary() if hasattr(self.model, "section_summary") else " | ".join(section.label for section in self.model.sections)
        self.sections_var.set(section_text)
        if self.model.source_path:
            self.file_var.set(str(self.model.source_path))

    def refresh_section_details(self) -> None:
        if self.section_details_text is None:
            return
        self.section_details_text.configure(state="normal")
        self.section_details_text.delete("1.0", "end")
        if not self.model:
            self.section_details_text.insert("1.0", "Open a G1M to inspect section coverage.")
            self.section_details_text.configure(state="disabled")
            return
        lines: list[str] = []
        lines.append(f"File magic/version: {self.model.file_magic} v{self.model.file_version}")
        lines.append(f"Sections: {len(self.model.sections)}|Bones: {len(self.model.bones)}|Submeshes: {len(self.model.submeshes)}")
        lines.append("")
        for section in self.model.section_coverage.sections:
            lines.append(section.label)
            for subsection in section.subsections:
                note = f" {subsection.note}" if subsection.note else ""
                lines.append(f"  {subsection.label}{note}")
            if section.warning:
                lines.append(f"  warning: {section.warning}")
        if self.model.section_warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  {warning}" for warning in self.model.section_warnings)
        self.section_details_text.insert("1.0", "\n".join(lines))
        self.section_details_text.configure(state="disabled")

    def part_is_cloth_related(self, part) -> bool:
        if (part.cloth_type_id & 0xF) != 0:
            return True
        if self.model is not None:
            try:
                return self.model.cloth_entry_for_body_part(part) is not None
            except Exception:
                return False
        return False

    def on_visibility_changed(self, *_args) -> None:
        self.refresh_region_tree()
        self.refresh_visibility_status()
        self.schedule_preview_sync()

    def hide_selected_region(self) -> None:
        part = self.selected_part()
        if part is None:
            self.status_var.set("Select a specific region before hiding it.")
            return
        self.hidden_part_indices.add(part.index)
        self.on_visibility_changed()
        self.status_var.set(f"Hidden region in preview: {part.name}")

    def show_all_regions(self) -> None:
        self.hidden_part_indices.clear()
        self.show_cloth_related_var.set(True)
        self.show_noncloth_related_var.set(True)
        self.selected_only_var.set(False)
        self.on_visibility_changed()
        if self.disabled_submesh_indices:
            self.status_var.set(
                f"Preview visibility reset. {len(self.disabled_submesh_indices)} blanked submesh(es) stay disabled; "
                "select one and click Disable to re-enable it."
            )
            return
        self.status_var.set("Preview visibility reset: all regions shown.")

    def effective_disabled_parts(self) -> set[int]:
        if not self.model or not self.disabled_submesh_indices:
            return set()
        return body_parts_covered_by_submeshes(self.model, self.disabled_submesh_indices)

    def toggle_disable_selected_region(self) -> None:
        if not self.model:
            self.status_var.set("Open a G1M before disabling a region.")
            return
        submesh_index = self.selected_submesh_index()
        if submesh_index is not None:
            self.toggle_disable_submeshes([submesh_index], f"submesh {submesh_index}")
            return
        part = self.selected_part()
        if part is None:
            self.status_var.set("Select a region or one of its submeshes before disabling it.")
            return
        self.toggle_disable_submeshes(part.submesh_indices, f"region {part.name}")

    def toggle_disable_submeshes(self, submesh_indices, label: str) -> None:
        wanted = {index for index in submesh_indices if 0 <= index < len(self.model.submeshes)}
        if not wanted:
            self.status_var.set(f"Nothing to disable on {label}.")
            return
        if wanted <= self.disabled_submesh_indices:
            self.disabled_submesh_indices -= wanted
            self.on_visibility_changed()
            self.status_var.set(f"Re-enabled {label}.")
            return
        added = wanted - self.disabled_submesh_indices
        self.disabled_submesh_indices |= wanted
        try:
            self.working_bytes()
        except Exception as exc:
            self.disabled_submesh_indices -= added
            messagebox.showerror("Unable to disable", str(exc), parent=self)
            return
        self.on_visibility_changed()
        self.status_var.set(
            f"Disabled {label} ({len(added)} submesh(es) blanked). "
            f"{self.shared_geometry_note(added)}Export or commit to bake it in."
        )

    def shared_geometry_note(self, submesh_indices: set[int]) -> str:
        """Warn when the blanked submeshes also belong to other regions"""

        others = {
            part.name
            for part in self.model.body_parts
            if submesh_indices & set(part.submesh_indices)
        }
        selected = self.selected_part()
        if selected is not None:
            others.discard(selected.name)
        if not others:
            return ""
        names = ", ".join(sorted(others)[:3])
        if len(others) > 3:
            names += f", +{len(others) - 3} more"
        return f"Those faces are shared with {names}, so they go too. "

    def refresh_visibility_status(self) -> None:
        if not self.model:
            self.visibility_status_var.set("Visibility: open a G1M to inspect regions.")
            return
        cloth = sum(1 for part in self.model.body_parts if self.part_is_cloth_related(part))
        noncloth = len(self.model.body_parts) - cloth
        self.visibility_status_var.set(
            f"Visibility: {noncloth} mesh/{cloth} cloth linked regions, "
            f"{len(self.hidden_part_indices)} manually hidden, "
            f"{len(self.effective_disabled_parts())} disabled (faces blanked on export)."
        )

    def refresh_region_tree(self, *_args) -> None:
        current = self.selected_row()
        expanded = {
            iid for iid in self.part_tree.get_children("")
            if iid.startswith("part:") and self.part_tree.item(iid, "open")
        } if getattr(self, "part_tree", None) else set()
        self.part_tree.delete(*self.part_tree.get_children())
        if not self.model:
            return
        self.part_tree.insert(
            "",
            "end",
            iid="all",
            text="All G1M submeshes",
            values=("mixed", "yes", len(self.model.bones), len(self.model.submeshes)),
        )
        filter_text = self.part_filter_var.get().strip().lower()
        disabled = self.effective_disabled_parts()
        for part in self.model.body_parts:
            kind = "cloth/NUN" if self.part_is_cloth_related(part) else "mesh"
            haystack = f"{part.name} {kind} bones {' '.join(str(b) for b in part.bone_ids)}".lower()
            if filter_text and filter_text not in haystack:
                continue
            part_id = f"part:{part.index}"
            self.part_tree.insert(
                "",
                "end",
                iid=part_id,
                text=part.name,
                open=part_id in expanded,
                values=(
                    kind,
                    self.part_visibility_label(part, disabled),
                    len(part.bone_ids),
                    len(part.submesh_indices),
                ),
            )
            for submesh_index in part.submesh_indices:
                if not 0 <= submesh_index < len(self.model.submeshes):
                    continue
                submesh = self.model.submeshes[submesh_index]
                self.part_tree.insert(
                    part_id,
                    "end",
                    iid=f"sub:{part.index}:{submesh_index}",
                    text=f"submesh {submesh_index} (material {submesh.material_index})",
                    values=(
                        f"vbo {submesh.vbo_index}",
                        "disabled" if submesh_index in self.disabled_submesh_indices else "yes",
                        submesh.vertex_count,
                        submesh.face_count,
                    ),
                )
        if current and self.part_tree.exists(current):
            self.part_tree.selection_set(current)
            self.part_tree.focus(current)
            self.part_tree.see(current)
        elif self.part_tree.exists("all"):
            self.part_tree.selection_set("all")
            self.part_tree.focus("all")

    def part_visibility_label(self, part, disabled_parts: set[int]) -> str:
        submeshes = set(part.submesh_indices)
        blanked = submeshes & self.disabled_submesh_indices
        if submeshes and blanked == submeshes:
            return "disabled"
        if blanked:
            return f"partial {len(blanked)}/{len(submeshes)}"
        if part.index in disabled_parts:
            return "disabled"
        if part.index in self.hidden_part_indices:
            return "no"
        return "yes"

    def on_region_selected(self, _event=None) -> None:
        self.schedule_preview_sync()

    def find_preview_interpreter(self) -> list[str] | None:
        env = self.preview_env()
        probe = (
            "import sys, site; "
            "user_site = site.getusersitepackages(); "
            "sys.path.append(user_site) if user_site and user_site not in sys.path else None; "
            "import pyglet"
        )
        candidates: list[str] = []
        for candidate in [sys.executable, "py", "python"]:
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            probe_candidate = self.normalize_preview_interpreter(candidate)
            try:
                result = subprocess.run(
                    [probe_candidate, "-c", probe],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    env=env,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0:
                return [self.preview_launch_interpreter(candidate)]
        return None

    @staticmethod
    def normalize_preview_interpreter(candidate: str) -> str:
        candidate_path = Path(candidate)
        if candidate_path.name.lower() == "pythonw.exe":
            console_python = candidate_path.with_name("python.exe")
            if console_python.exists():
                return str(console_python)
        return candidate

    @staticmethod
    def preview_launch_interpreter(candidate: str) -> str:
        candidate_path = Path(candidate)
        if candidate_path.name.lower() == "python.exe":
            windowed_python = candidate_path.with_name("pythonw.exe")
            if windowed_python.exists():
                return str(windowed_python)
        return candidate

    @staticmethod
    def preview_creation_flags() -> int:
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0

    def preview_env(self) -> dict[str, str]:
        env = os.environ.copy()
        user_site = site.getusersitepackages()
        python_path_entries = [entry for entry in [user_site, env.get("PYTHONPATH")] if entry]
        if python_path_entries:
            env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
        return env

    def open_preview(self) -> None:
        if not self.model or not self.model.source_path:
            messagebox.showinfo("No model loaded", "Open a G1M file before launching the 3D preview.", parent=self)
            return
        interpreter = self.find_preview_interpreter()
        if interpreter is None:
            suggested_python = self.normalize_preview_interpreter(sys.executable)
            messagebox.showerror(
                "pyglet not available",
                "Cleanrot is ready to launch the preview, but the Python interpreters on PATH do not have pyglet installed.\n\n"
                f"Interpreter: {suggested_python}\nSuggested command:\n{suggested_python} -m pip install pyglet",
                parent=self,
            )
            return
        self.force_preview_sync()
        if self.preview_process and self.preview_process.poll() is None:
            self.status_var.set("3D preview is already running. Synced latest live vertex state.")
            return
        try:
            self.preview_log_path.write_text("", encoding="utf-8")
        except OSError:
            pass
        try:
            with self.preview_log_path.open("w", encoding="utf-8") as preview_log:
                self.preview_process = subprocess.Popen(
                    interpreter + ["-m", "Orbital_Knight.preview", str(self.preview_state_path)],
                    cwd=self.repo_root,
                    env=self.preview_env(),
                    stdout=preview_log,
                    stderr=subprocess.STDOUT,
                    creationflags=self.preview_creation_flags(),
                )
        except OSError as exc:
            messagebox.showerror("Unable to launch preview", str(exc), parent=self)
            return
        self.preview_exit_reported = False
        self.after(800, self.watch_preview_process)
        self.status_var.set("Launched live vertex preview window.")

    def watch_preview_process(self) -> None:
        if self.preview_process is None or self.preview_exit_reported:
            return
        returncode = self.preview_process.poll()
        if returncode is None:
            self.after(800, self.watch_preview_process)
            return
        self.preview_exit_reported = True
        if returncode == 0:
            self.status_var.set("3D preview closed.")
            return
        detail = f"The preview process exited with code {returncode}.\nPreview log: {self.preview_log_path}"
        excerpt = self.preview_log_excerpt()
        if excerpt:
            detail += f"\n\nLast log lines:\n{excerpt}"
        self.status_var.set(f"3D preview closed unexpectedly (exit code {returncode}).")
        messagebox.showerror("3D preview closed unexpectedly", detail, parent=self)

    def preview_log_excerpt(self) -> str:
        try:
            log_text = self.preview_log_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        if not log_text:
            return ""
        return "\n".join(log_text.splitlines()[-14:])

    def on_close(self) -> None:
        if self.preview_process and self.preview_process.poll() is None:
            self.preview_process.terminate()
        self.destroy()


def main() -> None:
    app = CleanrotEditorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
