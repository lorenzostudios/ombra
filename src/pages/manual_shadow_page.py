"""
MIRA - Editor Manuale Ombre con Keyframing (ManualShadowPage)
Consente di posizionare, ruotare, scalare ed interpolare ombre ellittiche fotogramma
per fotogramma mediante keyframe visivi, maniglie di controllo interattive e salvataggio del progetto.
"""

import json
import math
import queue
import shutil
import pathlib
import datetime
import threading
from dataclasses import asdict
import numpy as np
import cv2
from tkinter import filedialog, messagebox
import customtkinter as ctk

from ..constants import (
    APP_TITLE,
    BG_DARK,
    PANEL_BG,
    CARD_BG,
    CARD_BORDER,
    ACCENT,
    ACCENT_HOVER,
    BLUE,
    BLUE_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARN,
    DANGER,
    DANGER_HOVER,
    SHADOW_OUTPUT_FOLDER,
)
from ..utils import load_video_frames, open_path
from ..components.shadow_canvas import ShadowCanvas
from ..shadow.auto_shadow import (
    SHADOW_UI_ACCENT_BGR,
    SHADOW_WHITE_BGR,
)
from ..shadow.manual_shadow import (
    MANUAL_DEFAULT_COLOR,
    MANUAL_DEFAULT_PARAMS,
    ManualKeyframe,
    _manual_kf_to_params,
    _manual_signed_angle,
    manual_get_interpolated_params,
    manual_draw_shadow,
    manual_handle_positions,
    manual_find_project,
    MANUAL_PROJECT_SUFFIX,
)

class ManualShadowPage(ctk.CTkFrame):
    """
    Interfaccia grafica per l'editing manuale di ombre con maniglie su tela (ShadowCanvas).
    Supporta creazione, modifica, spostamento ed eliminazione di keyframe,
    interpolazione continua di posizione/scala/rotazione/opacità ed esportazione video.
    """

    HANDLE_HIT_PX = 12

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app

        self.video_path = None
        self.frames = []
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.frame_count = 0

        self.current_frame = 0
        self.keyframes = {}
        self.current_params = dict(MANUAL_DEFAULT_PARAMS)
        self.clipboard = None
        self.easing = False
        self.show_shadow_preview = True

        self.project_path = None
        self.auto_import_enabled = True

        self._undo_stack = []
        self._redo_stack = []

        self._drag_mode = None

        self._load_queue = queue.Queue()
        self._export_queue = queue.Queue()
        self._exporting = False
        self._updating_ui = False

        self._build()

    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.banner = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        self.banner.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        self.banner.grid_columnconfigure(0, weight=1)
        self.lbl_status_title = ctk.CTkLabel(
            self.banner,
            text="Genera video con ombra",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_status_title.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 0))
        self.lbl_status_sub = ctk.CTkLabel(
            self.banner,
            text="Seleziona un video per iniziare.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_status_sub.grid(row=1, column=0, sticky="w", padx=16, pady=(2, 12))
        self.lbl_frame_info = ctk.CTkLabel(
            self.banner, text="", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED
        )
        self.lbl_frame_info.grid(row=0, column=1, rowspan=2, sticky="e", padx=16)

        body = ctk.CTkFrame(self, fg_color=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, weight=0)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = ctk.CTkFrame(parent, width=260, fg_color=PANEL_BG, corner_radius=12)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left.grid_propagate(False)

        ctk.CTkLabel(
            left,
            text="Video",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(16, 6))
        ctk.CTkButton(
            left,
            text="Seleziona video",
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.select_video,
        ).pack(fill="x", padx=16)
        self.lbl_video_name = ctk.CTkLabel(
            left,
            text="Nessun video selezionato",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=220,
            justify="left",
        )
        self.lbl_video_name.pack(anchor="w", padx=16, pady=(6, 14))

        sep = ctk.CTkFrame(left, height=1, fg_color=CARD_BORDER)
        sep.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(
            left,
            text="Keyframe",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(0, 6))
        self.kf_list = ctk.CTkScrollableFrame(left, fg_color="transparent", height=150)
        self.kf_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        kf_nav = ctk.CTkFrame(left, fg_color="transparent")
        kf_nav.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(
            kf_nav, text="< Precedente (J)", command=lambda: self.jump_keyframe(-1)
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(
            kf_nav, text="Successivo (K) >", command=lambda: self.jump_keyframe(1)
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        sep2 = ctk.CTkFrame(left, height=1, fg_color=CARD_BORDER)
        sep2.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(
            left,
            text="Progetto",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(0, 6))
        proj_row = ctk.CTkFrame(left, fg_color="transparent")
        proj_row.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(
            proj_row, text="Salva (P)", command=self.save_project_dialog
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(
            proj_row, text="Carica (L)", command=self.load_project_dialog
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.lbl_project_info = ctk.CTkLabel(
            left,
            text="Nessun progetto caricato.",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            wraplength=220,
            justify="left",
        )
        self.lbl_project_info.pack(anchor="w", padx=16, pady=(0, 6))
        self.switch_auto_import = ctk.CTkSwitch(
            left,
            text="Import automatico progetto",
            font=ctk.CTkFont(size=11),
            command=self._on_toggle_auto_import,
        )
        self.switch_auto_import.select()
        self.switch_auto_import.pack(anchor="w", padx=16, pady=(0, 12))

        undo_row = ctk.CTkFrame(left, fg_color="transparent")
        undo_row.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkButton(
            undo_row,
            text="Undo (Z)",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.undo,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(
            undo_row,
            text="Redo (Y)",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.redo,
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

    def _build_center(self, parent):
        mid = ctk.CTkFrame(parent, fg_color=BG_DARK)
        mid.grid(row=0, column=1, sticky="nsew")
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1)

        self.canvas = ShadowCanvas(
            mid,
            on_press=self._on_canvas_press,
            on_drag=self._on_canvas_drag,
            on_release=self._on_canvas_release,
            on_resize=self._refresh_display,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        nav = ctk.CTkFrame(mid, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        nav.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(
            nav, text="< -1 (A)", width=64, command=lambda: self.step_frame(-1)
        ).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkButton(
            nav, text="+1 (D) >", width=64, command=lambda: self.step_frame(1)
        ).grid(row=0, column=1, padx=4)
        self.slider = ctk.CTkSlider(
            nav, from_=0, to=1, number_of_steps=1, command=self._on_frame_slider
        )
        self.slider.set(0)
        self.slider.grid(row=0, column=2, sticky="ew", padx=8)

        self.switch_easing = ctk.CTkSwitch(
            nav, text="Easing (T)", command=self._on_toggle_easing
        )
        self.switch_easing.grid(row=0, column=3, padx=4)
        self.switch_preview = ctk.CTkSwitch(
            nav, text="Preview (H)", command=self._on_toggle_preview
        )
        self.switch_preview.select()
        self.switch_preview.grid(row=0, column=4, padx=(4, 0))

        self.lbl_nav_info = ctk.CTkLabel(
            mid,
            text="Frame 0/0   0.00s",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_nav_info.grid(row=2, column=0, sticky="w", pady=(4, 0))

        kf_actions = ctk.CTkFrame(mid, fg_color="transparent")
        kf_actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.btn_add_kf = ctk.CTkButton(
            kf_actions,
            text="Aggiungi/Aggiorna keyframe (N)",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            command=self.add_or_update_current_keyframe,
        )
        self.btn_add_kf.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_del_kf = ctk.CTkButton(
            kf_actions,
            text="Elimina keyframe (X)",
            fg_color=DANGER,
            hover_color=DANGER_HOVER,
            command=self.delete_current_keyframe,
        )
        self.btn_del_kf.pack(side="left", expand=True, fill="x", padx=(6, 0))

        ctk.CTkLabel(
            mid,
            text="Trascina l'ombra per spostarla. Maniglia laterale = larghezza, "
            "maniglia inferiore = aspect ratio, maniglia in alto = rotazione.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).grid(row=4, column=0, sticky="w", pady=(6, 0))

    def _build_right(self, parent):
        right = ctk.CTkScrollableFrame(
            parent, width=300, fg_color=PANEL_BG, corner_radius=12
        )
        right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))

        ctk.CTkLabel(
            right,
            text="Parametri ombra",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(16, 10))

        self.lbl_pos = ctk.CTkLabel(
            right,
            text="Posizione: (0, 0)",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_LIGHT,
        )
        self.lbl_pos.pack(anchor="w", padx=16, pady=(0, 10))

        self.sliders = {}
        specs = [
            ("width", "Larghezza", 2, 400, 398),
            (
                "aspect",
                "Aspect ratio",
                0.1,
                1.0,
                90,
            ),
            ("angle", "Rotazione", -89, 90, 180),
            ("opacity", "Opacità", 0, 100, 100),
            ("blur", "Sfocatura", 1, 101, 100),
        ]
        for key, label, lo, hi, steps in specs:
            head = ctk.CTkFrame(right, fg_color="transparent")
            head.pack(fill="x", padx=16)
            ctk.CTkLabel(
                head, text=label, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED
            ).pack(side="left")
            ctk.CTkButton(
                head,
                text="ALL",
                width=54,
                height=20,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=CARD_BG,
                hover_color="#2a2a34",
                text_color=TEXT_LIGHT,
                command=lambda k=key: self.apply_param_to_all_keyframes(k),
            ).pack(side="right")
            s = ctk.CTkSlider(
                right,
                from_=lo,
                to=hi,
                number_of_steps=steps,
                command=lambda v, k=key: self._on_param_slider(k, v),
            )
            s.pack(fill="x", padx=16, pady=(2, 10))
            self.sliders[key] = s

        sep = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep.pack(fill="x", padx=16, pady=(0, 12))

        cv_row = ctk.CTkFrame(right, fg_color="transparent")
        cv_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(cv_row, text="Copia (C)", command=self.copy_params).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ctk.CTkButton(cv_row, text="Incolla (V)", command=self.paste_params).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        ctk.CTkButton(
            right,
            text="Reset parametri (R)",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.reset_params,
        ).pack(fill="x", padx=16, pady=(0, 16))

        sep2 = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep2.pack(fill="x", padx=16, pady=(0, 12))

        self.btn_export = ctk.CTkButton(
            right,
            text="Esporta video (S)",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            command=self.export_video,
            state="disabled",
        )
        self.btn_export.pack(fill="x", padx=16, pady=(0, 8))
        self.progress = ctk.CTkProgressBar(right)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=16, pady=(0, 6))
        self.lbl_export_status = ctk.CTkLabel(
            right, text="", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        )
        self.lbl_export_status.pack(anchor="w", padx=16, pady=(0, 10))

        ctk.CTkButton(
            right,
            text="Apri cartella output",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.open_output_folder,
        ).pack(fill="x", padx=16, pady=(0, 10))
        self.lbl_last_output = ctk.CTkLabel(
            right,
            text="Nessun video generato ancora.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=260,
            justify="left",
        )
        self.lbl_last_output.pack(anchor="w", padx=16, pady=(0, 8))
        self.btn_use_in_dataset = ctk.CTkButton(
            right,
            text="Usa nel dataset (video alterato)",
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            command=self.use_last_in_dataset,
            state="disabled",
        )
        self.btn_use_in_dataset.pack(fill="x", padx=16, pady=(0, 16))

    def on_show(self):
        self.refresh_last_output()

    def free_memory_if_idle(self):
        if self._exporting or self.keyframes:
            return
        if not self.frames:
            return
        self._reset_state()

    def _reset_state(self):
        self.frames = []
        self.frame_count = 0
        self.current_frame = 0
        self.video_path = None
        self.project_path = None
        self.btn_export.configure(state="disabled")
        self.lbl_video_name.configure(text="Nessun video selezionato")
        self._update_project_label()
        self.canvas.release()
        self._scroll_kf_to_top()

    def _scroll_kf_to_top(self):
        try:
            if hasattr(self, "kf_list") and hasattr(self.kf_list, "_parent_canvas"):
                self.kf_list._parent_canvas.yview_moveto(0.0)
        except Exception:
            pass

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Seleziona il video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
        )
        if not path:
            return
        self.video_path = pathlib.Path(path)
        self.lbl_video_name.configure(text=self.video_path.name)
        self.keyframes = {}
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.clipboard = None
        self.current_frame = 0
        self.project_path = None
        self._update_project_label()
        self._scroll_kf_to_top()
        self.app.set_status(f"Caricamento {self.video_path.name}...")
        self.lbl_status_sub.configure(text="Caricamento video...")

        def worker():
            try:
                frames, fps = load_video_frames(self.video_path)
                self._load_queue.put(("ok", frames, fps))
            except Exception as e:  # noqa: BLE001
                self._load_queue.put(("error", str(e), None))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_load()

    def _poll_load(self):
        try:
            kind, a, b = self._load_queue.get_nowait()
        except queue.Empty:
            self.after(50, self._poll_load)
            return
        if kind == "error":
            messagebox.showerror(APP_TITLE, f"Impossibile caricare il video:\n{a}")
            self.app.set_status("Errore caricamento video.", "error")
            return

        self.frames = a
        self.fps = b or 25.0
        self.frame_count = len(self.frames)
        self.height, self.width = self.frames[0].shape[:2]
        self.current_frame = 0

        self.current_params = dict(MANUAL_DEFAULT_PARAMS)
        self.current_params["x"] = self.width / 2.0
        self.current_params["y"] = self.height / 2.0
        self.current_params["color"] = list(MANUAL_DEFAULT_COLOR)

        n_max = max(0, self.frame_count - 1)
        self.slider.configure(from_=0, to=max(n_max, 1), number_of_steps=max(n_max, 1))
        self.slider.set(0)
        self.btn_export.configure(state="normal")

        self.app.set_status(
            f"Video caricato ({self.frame_count} frame, {self.fps:.1f} fps)", "success"
        )
        self._auto_load_project()
        self.load_params_for_current_frame()
        self._refresh_display()
        self._scroll_kf_to_top()
        self.after_idle(self._scroll_kf_to_top)

    def _on_toggle_auto_import(self):
        self.auto_import_enabled = bool(self.switch_auto_import.get())

    def _auto_load_project(self):
        if not self.auto_import_enabled or self.video_path is None:
            self._update_project_label()
            return False
        found = manual_find_project(self.video_path)
        if found is None:
            self._update_project_label(auto_missing=True)
            return False
        try:
            self.load_project(found, auto=True)
        except Exception as e:  # noqa: BLE001
            self.app.set_status(
                f"Progetto '{pathlib.Path(found).name}' non caricabile: {e}", "warn"
            )
            self._update_project_label(auto_missing=True)
            return False
        return True

    def _update_project_label(self, auto_missing=False, auto=False):
        if self.project_path is not None:
            prefix = "Importato automaticamente: " if auto else "Progetto: "
            self.lbl_project_info.configure(
                text=f"{prefix}{pathlib.Path(self.project_path).name} "
                f"({len(self.keyframes)} keyframe)",
                text_color=ACCENT if auto else TEXT_LIGHT,
            )
        elif auto_missing:
            self.lbl_project_info.configure(
                text="Nessun progetto trovato accanto al video.",
                text_color=TEXT_MUTED,
            )
        else:
            self.lbl_project_info.configure(
                text="Nessun progetto caricato.", text_color=TEXT_MUTED
            )

    def step_frame(self, delta):
        if not self.frames:
            return
        self.current_frame = max(
            0, min(self.frame_count - 1, self.current_frame + delta)
        )
        self._updating_ui = True
        self.slider.set(self.current_frame)
        self._updating_ui = False
        self.load_params_for_current_frame()
        self._refresh_display()

    def _on_frame_slider(self, value):
        if self._updating_ui or not self.frames:
            return
        self.current_frame = max(0, min(self.frame_count - 1, int(float(value))))
        self.load_params_for_current_frame()
        self._refresh_display()

    def jump_keyframe(self, direction):
        if not self.keyframes:
            return
        frames_sorted = sorted(self.keyframes.keys())
        if direction < 0:
            candidates = [f for f in frames_sorted if f < self.current_frame]
            if candidates:
                self.current_frame = candidates[-1]
        else:
            candidates = [f for f in frames_sorted if f > self.current_frame]
            if candidates:
                self.current_frame = candidates[0]
        self._updating_ui = True
        self.slider.set(self.current_frame)
        self._updating_ui = False
        self.load_params_for_current_frame()
        self._refresh_display()

    def handle_shortcut(self, keysym):
        mapping_simple = {
            "a": lambda: self.step_frame(-1),
            "d": lambda: self.step_frame(1),
            "j": lambda: self.jump_keyframe(-1),
            "k": lambda: self.jump_keyframe(1),
            "n": self.add_or_update_current_keyframe,
            "x": self.delete_current_keyframe,
            "c": self.copy_params,
            "v": self.paste_params,
            "r": self.reset_params,
            "t": lambda: self.switch_easing.toggle(),
            "z": self.undo,
            "y": self.redo,
            "h": lambda: self.switch_preview.toggle(),
            "p": self.save_project_dialog,
            "l": self.load_project_dialog,
            "s": self.export_video,
        }
        fn = mapping_simple.get(keysym.lower())
        if fn:
            fn()

    def load_params_for_current_frame(self):
        kf = self.keyframes.get(self.current_frame)
        if kf is not None:
            params = _manual_kf_to_params(kf)
        else:
            params = manual_get_interpolated_params(
                self.keyframes, self.current_frame, self.easing, self.current_params
            )
        self.current_params = params
        self._sync_sliders()

    def _sync_sliders(self):
        self._updating_ui = True
        p = self.current_params
        self.sliders["width"].set(p["width"])
        self.sliders["aspect"].set(self._current_aspect())
        self.sliders["angle"].set(self._angle_to_slider_value(p["angle"]))
        self.sliders["opacity"].set(p["opacity"] * 100)
        self.sliders["blur"].set(p["blur"])
        self.lbl_pos.configure(text=f"Posizione: ({p['x']:.0f}, {p['y']:.0f})")
        self._updating_ui = False

    def _current_aspect(self) -> float:
        p = self.current_params
        w = max(1e-6, p.get("width", 1))
        return max(0.1, min(1.0, p.get("height", w) / w))

    @staticmethod
    def _angle_to_slider_value(angle: float) -> float:
        return _manual_signed_angle(angle)

    @staticmethod
    def _slider_value_to_angle(value: float) -> float:
        return value % 180

    def _on_param_slider(self, key, value):
        if self._updating_ui or not self.frames:
            return
        value = float(value)
        if key == "width":
            aspect = self._current_aspect()
            new_width = max(2, value)
            self.current_params["width"] = new_width
            self.current_params["height"] = max(2.0, new_width * aspect)
        elif key == "aspect":
            aspect = max(0.1, min(1.0, value))
            self.current_params["height"] = max(
                2.0, self.current_params["width"] * aspect
            )
        elif key == "angle":
            self.current_params["angle"] = self._slider_value_to_angle(value)
        elif key == "opacity":
            self.current_params["opacity"] = value / 100.0
        elif key == "blur":
            v = max(1, int(value))
            if v % 2 == 0:
                v += 1
            self.current_params["blur"] = v
        self._refresh_display()

    def apply_param_to_all_keyframes(self, key):
        if not self.frames:
            return
        if not self.keyframes:
            self.app.set_status(
                "Nessun keyframe su cui applicare il parametro.", "warn"
            )
            return

        nearest_frame = min(self.keyframes.keys(), key=lambda f: abs(f - self.current_frame))
        self.current_frame = nearest_frame
        self._updating_ui = True
        self.slider.set(self.current_frame)
        self._updating_ui = False
        self.load_params_for_current_frame()

        p = self.current_params
        self._push_undo()
        for kf in self.keyframes.values():
            if key == "width":
                aspect = max(0.1, min(1.0, kf.height / max(1e-6, kf.width)))
                kf.width = max(2.0, float(p["width"]))
                kf.height = max(2.0, kf.width * aspect)
            elif key == "aspect":
                aspect = self._current_aspect()
                kf.height = max(2.0, kf.width * aspect)
            elif key == "angle":
                kf.angle = float(p["angle"]) % 180
            elif key == "opacity":
                kf.opacity = float(np.clip(p["opacity"], 0.0, 1.0))
            elif key == "blur":
                b = int(p["blur"])
                if b % 2 == 0:
                    b += 1
                kf.blur = max(1, b)
        labels = {
            "width": "Larghezza",
            "aspect": "Aspect ratio",
            "angle": "Rotazione",
            "opacity": "Opacità",
            "blur": "Sfocatura",
        }
        self.app.set_status(
            f"{labels.get(key, key)} applicata a {len(self.keyframes)} keyframe.",
            "success",
        )
        self.load_params_for_current_frame()
        self._refresh_display()

    def _on_toggle_easing(self):
        self.easing = bool(self.switch_easing.get())
        self.load_params_for_current_frame()
        self._refresh_display()

    def _on_toggle_preview(self):
        self.show_shadow_preview = bool(self.switch_preview.get())
        self._refresh_display()

    def _hit_test_handle(self, fx, fy):
        if not self.frames:
            return None
        scale = max(self.canvas.get_scale(), 1e-6)
        threshold = self.HANDLE_HIT_PX / scale
        handles = manual_handle_positions(self.current_params)
        best_key, best_dist = None, threshold
        for key, (hx, hy) in handles.items():
            dist = math.hypot(fx - hx, fy - hy)
            if dist <= best_dist:
                best_key, best_dist = key, dist
        return best_key

    def _on_canvas_press(self, fx, fy):
        if not self.frames:
            return
        mode = self._hit_test_handle(fx, fy)
        if mode is None:
            mode = "move"
            self.current_params["x"] = max(0.0, min(self.width, float(fx)))
            self.current_params["y"] = max(0.0, min(self.height, float(fy)))
        self._drag_mode = mode
        self._refresh_display()

    def _on_canvas_drag(self, fx, fy):
        if not self.frames or self._drag_mode is None:
            return
        p = self.current_params
        mode = self._drag_mode

        if mode == "move":
            p["x"] = max(0.0, min(self.width, float(fx)))
            p["y"] = max(0.0, min(self.height, float(fy)))
        elif mode == "resize_w":
            dx, dy = fx - p["x"], fy - p["y"]
            rad = math.radians(p["angle"])
            wx, wy = math.cos(rad), math.sin(rad)
            proj = dx * wx + dy * wy
            new_width = max(4.0, abs(proj) * 2)
            aspect = self._current_aspect()
            p["width"] = new_width
            p["height"] = max(4.0, new_width * aspect)
        elif mode == "resize_h":
            dx, dy = fx - p["x"], fy - p["y"]
            rad = math.radians(p["angle"])
            hx, hy = -math.sin(rad), math.cos(rad)
            proj = dx * hx + dy * hy
            new_height = max(4.0, abs(proj) * 2)
            p["height"] = min(new_height, p["width"])
        elif mode == "rotate":
            dx, dy = fx - p["x"], fy - p["y"]
            ang = math.degrees(math.atan2(dx, -dy)) % 180
            p["angle"] = ang

        self._sync_sliders()
        self._refresh_display()

    def _on_canvas_release(self, fx, fy):
        self._drag_mode = None

    def add_or_update_current_keyframe(self):
        if not self.frames:
            return
        self._push_undo()
        params = self.current_params
        blur = int(params["blur"])
        if blur % 2 == 0:
            blur += 1
        kf = ManualKeyframe(
            frame=self.current_frame,
            x=float(params["x"]),
            y=float(params["y"]),
            width=max(2.0, float(params["width"])),
            height=max(2.0, float(params["height"])),
            angle=float(params["angle"]) % 181,
            opacity=float(np.clip(params["opacity"], 0.0, 1.0)),
            blur=max(1, blur),
            color=list(params.get("color", MANUAL_DEFAULT_COLOR)),
        )
        self.keyframes[self.current_frame] = kf
        self.app.set_status(
            f"Keyframe registrato al frame {self.current_frame}", "success"
        )
        self._refresh_display()

    def delete_current_keyframe(self):
        if self.current_frame in self.keyframes:
            self._push_undo()
            del self.keyframes[self.current_frame]
            self.load_params_for_current_frame()
            self.app.set_status(
                f"Keyframe eliminato al frame {self.current_frame}", "info"
            )
            self._refresh_display()

    def copy_params(self):
        self.clipboard = dict(self.current_params)
        self.app.set_status("Parametri copiati.", "info")

    def paste_params(self):
        if not self.clipboard or not self.frames:
            return
        self.current_params = dict(self.clipboard)
        self._sync_sliders()
        self.add_or_update_current_keyframe()

    def reset_params(self):
        if not self.frames:
            return
        self.current_params = dict(MANUAL_DEFAULT_PARAMS)
        self.current_params["x"] = self.width / 2.0
        self.current_params["y"] = self.height / 2.0
        self.current_params["color"] = list(MANUAL_DEFAULT_COLOR)
        self._sync_sliders()
        self._refresh_display()

    def _snapshot(self):
        return {f: ManualKeyframe(**asdict(kf)) for f, kf in self.keyframes.items()}

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self.keyframes = self._undo_stack.pop()
        self.load_params_for_current_frame()
        self._refresh_display()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self.keyframes = self._redo_stack.pop()
        self.load_params_for_current_frame()
        self._refresh_display()

    def _draw_handles(self, img):
        p = self.current_params
        cx, cy = int(round(p["x"])), int(round(p["y"]))
        cv2.drawMarker(
            img, (cx, cy), SHADOW_UI_ACCENT_BGR, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA
        )
        axes = (max(1, int(round(p["width"] / 2))), max(1, int(round(p["height"] / 2))))
        cv2.ellipse(
            img,
            (cx, cy),
            axes,
            p["angle"],
            0,
            360,
            SHADOW_UI_ACCENT_BGR,
            1,
            cv2.LINE_AA,
        )

        handles = manual_handle_positions(p)
        hcol = {
            "resize_w": (255, 200, 0),
            "resize_h": (255, 200, 0),
            "rotate": (255, 120, 255),
        }
        active = self._drag_mode
        for key, (hx, hy) in handles.items():
            hx, hy = int(round(hx)), int(round(hy))
            if key == "rotate":
                cv2.line(img, (cx, cy), (hx, hy), (120, 120, 120), 1, cv2.LINE_AA)
                cv2.circle(img, (hx, hy), 7, hcol[key], -1)
                cv2.circle(img, (hx, hy), 7, SHADOW_WHITE_BGR, 1)
            else:
                size = 6
                col = hcol[key]
                thick = 2 if active == key else 1
                cv2.rectangle(
                    img, (hx - size, hy - size), (hx + size, hy + size), col, -1
                )
                cv2.rectangle(
                    img,
                    (hx - size, hy - size),
                    (hx + size, hy + size),
                    SHADOW_WHITE_BGR,
                    thick,
                )

    def _compose_frame(self, idx):
        img = self.frames[idx].copy()
        if self.show_shadow_preview:
            img = manual_draw_shadow(img, self.current_params)
        self._draw_handles(img)
        return img

    def _refresh_display(self):
        if not self.frames:
            return
        img = self._compose_frame(self.current_frame)
        self.canvas.render(img)

        t = self.current_frame / self.fps if self.fps else 0
        self.lbl_nav_info.configure(
            text=f"Frame {self.current_frame}/{self.frame_count - 1}   {t:.2f}s"
        )
        self.lbl_frame_info.configure(
            text=f"{self.video_path.name if self.video_path else ''}   |   "
            f"{self.fps:.1f} fps   |   {self.frame_count} frame   |   KF: {len(self.keyframes)}"
        )

        is_kf = self.current_frame in self.keyframes
        if is_kf:
            status, col = "KEYFRAME", ACCENT
        elif self.keyframes:
            status, col = "INTERPOLATO", BLUE
        else:
            status, col = "NESSUN KEYFRAME", WARN
        self.lbl_status_title.configure(text="Ombra manuale (keyframe)")
        self.lbl_status_sub.configure(
            text=f"{status}   |   easing: {'on' if self.easing else 'off'}   |   "
            f"preview: {'on' if self.show_shadow_preview else 'off'}",
            text_color=col,
        )
        self._update_kf_list()
        self.btn_del_kf.configure(state="normal" if is_kf else "disabled")

    def _update_kf_list(self):
        for w in self.kf_list.winfo_children():
            w.destroy()
        frames_sorted = sorted(self.keyframes.keys())
        if not frames_sorted:
            ctk.CTkLabel(
                self.kf_list,
                text="Nessun keyframe",
                font=ctk.CTkFont(size=11),
                text_color=TEXT_MUTED,
            ).pack(anchor="w", padx=6, pady=4)
            return
        for f in frames_sorted:
            is_here = f == self.current_frame
            row = ctk.CTkLabel(
                self.kf_list,
                text=f"{'> ' if is_here else '  '}f{f}",
                font=ctk.CTkFont(size=11),
                text_color=ACCENT if is_here else TEXT_LIGHT,
                anchor="w",
                fg_color=CARD_BG if is_here else "transparent",
                corner_radius=4,
            )
            row.pack(fill="x", padx=4, pady=1)
            row.bind("<Button-1>", lambda e, ff=f: self._jump_to_frame(ff))

    def _jump_to_frame(self, f):
        self.current_frame = f
        self._updating_ui = True
        self.slider.set(f)
        self._updating_ui = False
        self.load_params_for_current_frame()
        self._refresh_display()

    def _default_project_name(self) -> str:
        base = pathlib.Path(self.video_path).stem if self.video_path else "progetto"
        return f"{base}{MANUAL_PROJECT_SUFFIX}"

    def save_project_dialog(self):
        if not self.frames:
            return
        initial_dir = (
            str(self.video_path.parent) if self.video_path is not None else "."
        )
        path = filedialog.asksaveasfilename(
            title="Salva progetto",
            defaultextension=".json",
            initialdir=initial_dir,
            initialfile=self._default_project_name(),
            filetypes=[("JSON project", "*.json")],
        )
        if path:
            self.save_project(path)

    def load_project_dialog(self):
        initial_dir = (
            str(self.video_path.parent) if self.video_path is not None else "."
        )
        path = filedialog.askopenfilename(
            title="Carica progetto",
            initialdir=initial_dir,
            filetypes=[("JSON project", "*.json")],
        )
        if path:
            self.load_project(path)

    def save_project(self, path):
        data = {
            "video_path": str(self.video_path) if self.video_path else None,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "keyframes": [kf.to_dict() for _, kf in sorted(self.keyframes.items())],
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        self.project_path = pathlib.Path(path)
        self._update_project_label()
        self.app.set_status(f"Progetto salvato: {path}", "success")

    def load_project(self, path, auto=False):
        path = pathlib.Path(path)
        if not path.exists():
            self.app.set_status(f"File non trovato: {path}", "error")
            return
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        self._push_undo()
        self.keyframes = {}
        for d in data.get("keyframes", []):
            kf = ManualKeyframe.from_dict(d)
            self.keyframes[kf.frame] = kf
        self.project_path = path
        self._update_project_label(auto=auto)
        prefix = "Progetto importato automaticamente" if auto else "Progetto caricato"
        self.app.set_status(
            f"{prefix}: {path.name}  ({len(self.keyframes)} keyframe)", "success"
        )
        self.load_params_for_current_frame()
        self._refresh_display()
        self._scroll_kf_to_top()
        self.after_idle(self._scroll_kf_to_top)

    def export_video(self):
        if self._exporting or not self.frames:
            return
        if not self.keyframes:
            messagebox.showwarning(
                APP_TITLE,
                "Nessun keyframe registrato. Aggiungi almeno un keyframe prima di esportare.",
            )
            return

        out_fn = SHADOW_OUTPUT_FOLDER / f"_{self.video_path.stem}.mp4"
        if out_fn.exists():
            if not messagebox.askyesno(
                APP_TITLE,
                f"'{out_fn.name}' esiste gia' in '{SHADOW_OUTPUT_FOLDER}'.\nSovrascrivere?",
            ):
                return

        self._exporting = True
        self.btn_export.configure(state="disabled")
        self.progress.set(0)
        self.lbl_export_status.configure(
            text="Avvio esportazione...", text_color=TEXT_MUTED
        )
        self.app.set_status("Esportazione video con ombra manuale in corso...", "info")

        frames = self.frames
        fps = self.fps
        width, height = self.width, self.height
        keyframes_snapshot = self._snapshot()
        easing = self.easing

        threading.Thread(
            target=self._export_worker,
            args=(frames, fps, width, height, keyframes_snapshot, easing, out_fn),
            daemon=True,
        ).start()
        self._poll_export()

    def _export_worker(
        self, frames, fps, width, height, keyframes_snapshot, easing, out_fn
    ):
        try:
            out_fn = self._generate_output_video(
                frames, fps, width, height, keyframes_snapshot, easing, out_fn
            )
            self._export_queue.put(("done", out_fn))
        except Exception as e:  # noqa: BLE001
            self._export_queue.put(("error", str(e)))

    def _poll_export(self):
        item = None
        while True:
            try:
                nxt = self._export_queue.get_nowait()
            except queue.Empty:
                break
            item = nxt
            if item[0] in ("done", "error"):
                break
        if item is None or item[0] == "progress":
            if item is not None:
                self.progress.set(item[1])
            self.after(150, self._poll_export)
            return
        kind = item[0]

        self._exporting = False
        self.btn_export.configure(state="normal")
        if kind == "error":
            self.progress.set(0)
            self.lbl_export_status.configure(
                text=f"Errore: {item[1]}", text_color=DANGER
            )
            self.app.set_status("Esportazione fallita.", "error")
            return

        out_fn = item[1]
        self.progress.set(1.0)
        self.lbl_export_status.configure(
            text=f"Video salvato: {pathlib.Path(out_fn).name}", text_color=ACCENT
        )
        self.app.set_status(
            f"Video con ombra manuale esportato: {pathlib.Path(out_fn).name}", "success"
        )
        self.refresh_last_output()

    def _generate_output_video(
        self, frames, fps, width, height, keyframes_snapshot, easing, out_fn
    ):
        SHADOW_OUTPUT_FOLDER.mkdir(exist_ok=True)
        out_fn = str(out_fn)
        writer = cv2.VideoWriter(
            out_fn, cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height)
        )

        tot = len(frames)
        fallback = MANUAL_DEFAULT_PARAMS
        for i in range(tot):
            if i % 20 == 0:
                self._export_queue.put(("progress", i / max(tot - 1, 1)))
            params = manual_get_interpolated_params(
                keyframes_snapshot, i, easing, fallback
            )
            out_frame = manual_draw_shadow(frames[i], params)
            writer.write(out_frame)
        writer.release()
        return out_fn

    def _latest_output_video(self):
        if not SHADOW_OUTPUT_FOLDER.exists():
            return None
        vids = sorted(
            SHADOW_OUTPUT_FOLDER.glob("_*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return vids[0] if vids else None

    def refresh_last_output(self):
        latest = self._latest_output_video()
        if latest is not None:
            mtime = datetime.datetime.fromtimestamp(latest.stat().st_mtime)
            self.lbl_last_output.configure(
                text=f"{latest.name}   ({mtime.strftime('%d/%m/%Y %H:%M:%S')})"
            )
            self.btn_use_in_dataset.configure(state="normal")
        else:
            self.lbl_last_output.configure(text="Nessun video generato ancora.")
            self.btn_use_in_dataset.configure(state="disabled")

    def open_output_folder(self):
        SHADOW_OUTPUT_FOLDER.mkdir(exist_ok=True)
        open_path(SHADOW_OUTPUT_FOLDER.resolve())

    def use_last_in_dataset(self):
        latest = self._latest_output_video()
        if latest is None:
            self.app.set_status("Nessun video da usare.", "warn")
            return
        dest_folder = self.app.state_.video_folder
        try:
            dest_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(
                APP_TITLE, f"Impossibile creare la cartella '{dest_folder}':\n{e}"
            )
            return

        dest_name = latest.name if latest.name.startswith("_") else f"_{latest.name}"
        dest_path = dest_folder / dest_name
        if dest_path.exists():
            if not messagebox.askyesno(
                APP_TITLE,
                f"'{dest_name}' esiste gia' in '{dest_folder}'.\nSovrascrivere?",
            ):
                return
        try:
            shutil.copy2(latest, dest_path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Copia fallita:\n{e}")
            return

        self.app.set_status(
            f"Copiato in '{dest_folder}' come video alterato: {dest_name}", "success"
        )
        messagebox.showinfo(
            APP_TITLE,
            f"'{dest_name}' e' stato aggiunto a '{dest_folder}' come video alterato.\n"
            "Vai su 'Dataset' per impostare domanda e frame target.",
        )
