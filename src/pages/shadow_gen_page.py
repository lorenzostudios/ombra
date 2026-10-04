"""
MIRA - Generazione Ombra con Calibrazione Prospettica (ShadowGenPage)
Editor semi-automatico per la calibrazione geometrica del campo di gioco (omografia),
il campionamento cromatico di erba e linee, e la proiezione dell'ombra sul terreno.
"""

import queue
import shutil
import pathlib
import datetime
import threading
import numpy as np
import cv2
from PIL import Image
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
    SHADOW_OUTPUT_FOLDER,
)
from ..utils import load_video_frames, open_path
from ..components.shadow_canvas import ShadowCanvas
from ..shadow.auto_shadow import (
    SHADOW_FIELD_W,
    SHADOW_FIELD_H,
    SHADOW_DST_PTS,
    SHADOW_PHASE_COLORS_BGR,
    SHADOW_WHITE_BGR,
    SHADOW_SMOOTH_POS,
    SHADOW_SMOOTH_VIS,
    _shadow_safe_patch,
    _bgr_to_hex,
    _shadow_put,
    shadow_compute_ellipse_params,
    shadow_draw_ellipse,
    shadow_get_visibility,
    shadow_track_calib_full,
    shadow_compute_homography_tracked,
    shadow_img_to_field,
    shadow_field_to_img,
    shadow_get_zoom_patch,
)

class ShadowGenPage(ctk.CTkFrame):
    """
    Interfaccia grafica semi-automatica per la generazione prospettica dell'ombra.
    Guida l'utente attraverso una procedura a step:
      1. Calibrazione dei 4 vertici del campo di gioco (omografia planare).
      2. Campionamento del colore dell'erba (maschera e trasparenza).
      3. Campionamento delle linee di gioco (blending corretto).
      4. Definizione degli estremi dell'azione di gioco e tracciamento.
      5. Generazione ed esportazione del video con ombra proiettata.
    """

    STEPS = [
        (
            "CALIB",
            "Calibrazione",
            "Clicca i 4 angoli del campo (senso orario dal basso-sinistra)",
        ),
        ("GRASS", "Erba", "Clicca 3 zone di erba"),
        ("LINES", "Linee", "Clicca 2 zone di linea bianca"),
        (
            "ACTION",
            "Azione",
            "Naviga al frame iniziale e finale, clicca al centro del pallone",
        ),

        ("READY", "Pronto", "Nuova azione oppure esporta il video"),
    ]

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app

        self.frames = []
        self.fps = 25.0
        self.frame_idx = 0
        self.video_path = None
        self.shadow_size_val = 12
        self.current_step = "CALIB"
        self.mouse_pos = (0.0, 0.0)
        self.show_grid = False
        self.data = {
            "calib_pts": [],
            "grass_samples": [],
            "line_samples": [],
            "actions": [],
            "temp_start": None,
        }

        self._load_queue = queue.Queue()
        self._export_queue = queue.Queue()
        self._exporting = False

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
        self.lbl_phase_title = ctk.CTkLabel(
            self.banner,
            text="Genera video con ombra",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_phase_title.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 0))
        self.lbl_phase_sub = ctk.CTkLabel(
            self.banner,
            text="Seleziona un video per iniziare.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_phase_sub.grid(row=1, column=0, sticky="w", padx=16, pady=(2, 12))
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
        self.lbl_video_name.pack(anchor="w", padx=16, pady=(6, 16))

        sep = ctk.CTkFrame(left, height=1, fg_color=CARD_BORDER)
        sep.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            left,
            text="Procedura",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(0, 8))
        self.step_rows = {}
        for key, label, _ in self.STEPS:
            row = ctk.CTkFrame(left, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)
            dot = ctk.CTkLabel(
                row,
                text="○",
                width=18,
                font=ctk.CTkFont(size=14),
                text_color=TEXT_MUTED,
            )
            dot.pack(side="left")
            lbl = ctk.CTkLabel(
                row,
                text=label,
                font=ctk.CTkFont(size=12),
                text_color=TEXT_MUTED,
                anchor="w",
            )
            lbl.pack(side="left", padx=(4, 0))
            self.step_rows[key] = (dot, lbl)

        sep2 = ctk.CTkFrame(left, height=1, fg_color=CARD_BORDER)
        sep2.pack(fill="x", padx=16, pady=(14, 14))

        ctk.CTkButton(
            left,
            text="Ricomincia calibrazione",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.reset_all,
        ).pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(
            left,
            text="Azzera calibrazione, campioni e azioni registrate per questo video.",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _build_center(self, parent):
        mid = ctk.CTkFrame(parent, fg_color=BG_DARK)
        mid.grid(row=0, column=1, sticky="nsew")
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1)

        self.canvas = ShadowCanvas(
            mid,
            on_click=self._on_canvas_click,
            on_move=self._on_canvas_move,
            on_resize=self._refresh_display,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        nav = ctk.CTkFrame(mid, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        nav.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(
            nav, text="< -1", width=48, command=lambda: self.step_frame(-1)
        ).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkButton(
            nav, text="+1 >", width=48, command=lambda: self.step_frame(1)
        ).grid(row=0, column=1, padx=4)
        self.slider = ctk.CTkSlider(
            nav, from_=0, to=1, number_of_steps=1, command=self._on_slider
        )
        self.slider.set(0)
        self.slider.grid(row=0, column=2, sticky="ew", padx=8)
        self.switch_grid = ctk.CTkSwitch(
            nav, text="Griglia", command=self._on_toggle_grid, state="disabled"
        )
        self.switch_grid.grid(row=0, column=3, padx=(4, 0))

        self.lbl_nav_info = ctk.CTkLabel(
            mid,
            text="Frame 0/0   0.00s",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_nav_info.grid(row=2, column=0, sticky="w", pady=(4, 0))

    def _build_right(self, parent):
        right = ctk.CTkScrollableFrame(
            parent, width=300, fg_color=PANEL_BG, corner_radius=12
        )
        right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))

        ctk.CTkLabel(
            right,
            text="Azioni registrate",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(16, 6))
        self.actions_list = ctk.CTkScrollableFrame(
            right, fg_color="transparent", height=110
        )
        self.actions_list.pack(fill="x", padx=12, pady=(0, 12))

        sep = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(
            right,
            text="Campioni colore",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(0, 8))
        swrow = ctk.CTkFrame(right, fg_color="transparent")
        swrow.pack(fill="x", padx=16, pady=(0, 12))
        self.swatch_grass = ctk.CTkLabel(
            swrow,
            text="Erba",
            width=110,
            height=28,
            corner_radius=6,
            fg_color=CARD_BG,
            text_color=TEXT_MUTED,
        )
        self.swatch_grass.pack(side="left", padx=(0, 6))
        self.swatch_lines = ctk.CTkLabel(
            swrow,
            text="Linee",
            width=110,
            height=28,
            corner_radius=6,
            fg_color=CARD_BG,
            text_color=TEXT_MUTED,
        )
        self.swatch_lines.pack(side="left")

        sep2 = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep2.pack(fill="x", padx=16, pady=(0, 12))

        self.lbl_shadow_size = ctk.CTkLabel(
            right,
            text="Dimensione ombra: 12px",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_LIGHT,
        )
        self.lbl_shadow_size.pack(anchor="w", padx=16, pady=(0, 4))
        self.slider_shadow = ctk.CTkSlider(
            right, from_=2, to=100, number_of_steps=98, command=self._on_shadow_size
        )
        self.slider_shadow.set(12)
        self.slider_shadow.pack(fill="x", padx=16, pady=(0, 12))

        sep3 = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep3.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(
            right,
            text="Zoom",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=16)
        self.zoom_label = ctk.CTkLabel(
            right, text="", fg_color=CARD_BG, corner_radius=8
        )
        self.zoom_label.pack(padx=16, pady=(4, 14))
        self._zoom_img = None

        sep4 = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep4.pack(fill="x", padx=16, pady=(0, 12))

        self.btn_new_action = ctk.CTkButton(
            right,
            text="Nuova azione",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.new_action,
            state="disabled",
        )
        self.btn_new_action.pack(fill="x", padx=16, pady=(0, 8))
        self.btn_export = ctk.CTkButton(
            right,
            text="Esporta video",
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
        if self._exporting:
            return
        if self.data["calib_pts"] or self.data["actions"] or self.data["temp_start"]:
            return
        if not self.frames:
            return
        self.frames = []
        self.frame_idx = 0
        self.video_path = None
        self.lbl_video_name.configure(text="Nessun video selezionato")
        self.canvas.release()

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Seleziona il video dell'azione calcistica",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
        )
        if not path:
            return
        self.reset_all(confirm=False)
        self.video_path = pathlib.Path(path)
        self.lbl_video_name.configure(text=self.video_path.name)
        self.app.set_status(f"Caricamento {self.video_path.name}...")
        self.lbl_phase_sub.configure(text="Caricamento video...")

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
        self.frames, self.fps = a, (b or 25.0)
        self.frame_idx = 0
        n_max = max(0, len(self.frames) - 1)
        self.slider.configure(from_=0, to=max(n_max, 1), number_of_steps=max(n_max, 1))
        self.slider.set(0)
        self.app.set_status(
            f"Video caricato ({len(self.frames)} frame, {self.fps:.1f} fps)", "success"
        )
        self._refresh_display()

    def step_frame(self, delta):
        if not self.frames:
            return
        self.frame_idx = max(0, min(len(self.frames) - 1, self.frame_idx + delta))
        self.slider.set(self.frame_idx)
        self._refresh_display()

    def _on_slider(self, value):
        if not self.frames:
            return
        self.frame_idx = max(0, min(len(self.frames) - 1, int(float(value))))
        self._refresh_display()

    def _on_toggle_grid(self):
        self.show_grid = bool(self.switch_grid.get())
        self._refresh_display()

    def _on_shadow_size(self, value):
        self.shadow_size_val = max(2, int(float(value)))
        self.lbl_shadow_size.configure(
            text=f"Dimensione ombra: {self.shadow_size_val}px"
        )
        self._refresh_display()

    def _on_canvas_click(self, fx, fy):
        if not self.frames:
            return
        frame = self.frames[self.frame_idx]
        step = self.current_step

        if step == "CALIB":
            self.data["calib_pts"].append((float(fx), float(fy)))
            if len(self.data["calib_pts"]) == 4:
                self.current_step = "GRASS"
                self.switch_grid.configure(state="normal")

        elif step == "GRASS":
            self.data["grass_samples"].append(
                np.mean(_shadow_safe_patch(frame, int(fx), int(fy)), axis=(0, 1))
            )
            if len(self.data["grass_samples"]) == 3:
                self.current_step = "LINES"

        elif step == "LINES":
            self.data["line_samples"].append(
                np.mean(_shadow_safe_patch(frame, int(fx), int(fy)), axis=(0, 1))
            )
            if len(self.data["line_samples"]) == 2:
                self.current_step = "START"

        elif step == "START":
            self.data["temp_start"] = {
                "frame": self.frame_idx,
                "pos": (float(fx), float(fy)),
            }
            self.current_step = "END"

        elif step == "END":
            if (
                self.data["temp_start"]
                and self.frame_idx > self.data["temp_start"]["frame"]
            ):
                self.data["actions"].append(
                    {
                        "start_frame": self.data["temp_start"]["frame"],
                        "start_pos": self.data["temp_start"]["pos"],
                        "end_frame": self.frame_idx,
                        "end_pos": (float(fx), float(fy)),
                    }
                )
                self.data["temp_start"] = None
                self.current_step = "READY"
            else:
                self.app.set_status(
                    "Il frame finale deve essere successivo al frame iniziale.", "warn"
                )

        self._refresh_display()

    def _on_canvas_move(self, fx, fy):
        self.mouse_pos = (fx, fy)
        self._update_zoom_preview()
        if self.frames and self.current_step in ("CALIB", "START", "END"):
            self._refresh_display(recompute_only_overlay=True)

    def _get_H_static(self):
        if len(self.data["calib_pts"]) < 4:
            return None, None
        src = np.float32(self.data["calib_pts"])
        H, _ = cv2.findHomography(src, SHADOW_DST_PTS, cv2.RANSAC, 3.0)
        if H is None:
            return None, None
        try:
            return H, np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None, None

    def _draw_calib_overlay(self, img):
        pts = self.data["calib_pts"]
        n = len(pts)
        col = SHADOW_PHASE_COLORS_BGR["CALIB"]
        mouse_xy = (int(self.mouse_pos[0]), int(self.mouse_pos[1]))

        for i, p in enumerate(pts):
            cv2.circle(img, (int(p[0]), int(p[1])), 5, col, -1)
            cv2.circle(img, (int(p[0]), int(p[1])), 7, SHADOW_WHITE_BGR, 1)
            _shadow_put(
                img, str(i + 1), (int(p[0]) + 9, int(p[1]) - 5), scale=0.4, color=col
            )
            if i > 0:
                cv2.line(
                    img,
                    (int(pts[i - 1][0]), int(pts[i - 1][1])),
                    (int(p[0]), int(p[1])),
                    col,
                    1,
                    cv2.LINE_AA,
                )

        if self.current_step == "CALIB" and 0 < n < 4:
            cv2.line(
                img, (int(pts[-1][0]), int(pts[-1][1])), mouse_xy, col, 1, cv2.LINE_AA
            )
            if n == 3:
                cv2.line(
                    img, mouse_xy, (int(pts[0][0]), int(pts[0][1])), col, 1, cv2.LINE_AA
                )

        if n == 4:
            cv2.line(
                img,
                (int(pts[3][0]), int(pts[3][1])),
                (int(pts[0][0]), int(pts[0][1])),
                col,
                1,
                cv2.LINE_AA,
            )
            if self.show_grid:
                H_g, _ = cv2.findHomography(SHADOW_DST_PTS, np.float32(pts))
                if H_g is not None:
                    for gx in np.linspace(0, SHADOW_FIELD_W, 11):
                        p1 = cv2.perspectiveTransform(
                            np.array([[[gx, 0.0]]], np.float32), H_g
                        )[0][0]
                        p2 = cv2.perspectiveTransform(
                            np.array([[[gx, float(SHADOW_FIELD_H)]]], np.float32), H_g
                        )[0][0]
                        cv2.line(
                            img,
                            tuple(p1.astype(int)),
                            tuple(p2.astype(int)),
                            (0, 200, 200),
                            1,
                            cv2.LINE_AA,
                        )
                    for gy in np.linspace(0, SHADOW_FIELD_H, 11):
                        p1 = cv2.perspectiveTransform(
                            np.array([[[0.0, gy]]], np.float32), H_g
                        )[0][0]
                        p2 = cv2.perspectiveTransform(
                            np.array([[[float(SHADOW_FIELD_W), gy]]], np.float32), H_g
                        )[0][0]
                        cv2.line(
                            img,
                            tuple(p1.astype(int)),
                            tuple(p2.astype(int)),
                            (0, 200, 200),
                            1,
                            cv2.LINE_AA,
                        )

    def _draw_action_overlay(self, img):
        if self.current_step in ("START", "END"):
            H, H_inv = self._get_H_static()
            if H is not None:
                try:
                    res = shadow_compute_ellipse_params(
                        (self.mouse_pos[0], self.mouse_pos[1]),
                        H,
                        H_inv,
                        self.shadow_size_val,
                    )
                    if res is not None:
                        shadow_draw_ellipse(
                            img, res[0], res[1], res[2], alpha_mult=0.75
                        )
                except Exception:
                    pass

        if self.data["temp_start"]:
            sp = self.data["temp_start"]["pos"]
            col = SHADOW_PHASE_COLORS_BGR["START"]
            cv2.circle(img, (int(sp[0]), int(sp[1])), 6, col, -1)
            cv2.circle(img, (int(sp[0]), int(sp[1])), 8, SHADOW_WHITE_BGR, 1)
            _shadow_put(
                img, "START", (int(sp[0]) + 10, int(sp[1]) - 6), scale=0.38, color=col
            )

    def _compose_frame(self, idx):
        img = self.frames[idx].copy()
        self._draw_action_overlay(img)
        self._draw_calib_overlay(img)
        return img

    def _refresh_display(self, recompute_only_overlay=False):
        if not self.frames:
            return
        img = self._compose_frame(self.frame_idx)
        self.canvas.render(img)

        t = self.frame_idx / max(self.fps, 1)
        self.lbl_nav_info.configure(
            text=f"Frame {self.frame_idx}/{len(self.frames) - 1}   {t:.2f}s"
        )
        self.lbl_frame_info.configure(
            text=f"{self.video_path.name if self.video_path else ''}   |   "
            f"{self.fps:.1f} fps   |   {len(self.frames)} frame"
        )

        self._update_phase_banner()
        self._update_stepper()
        self._update_actions_list()
        self._update_swatches()
        self._update_buttons_state()

    def _update_phase_banner(self):
        titles = {
            "CALIB": "Calibrazione campo",
            "GRASS": "Campionamento erba",
            "LINES": "Campionamento linee",
            "START": "Azione - inizio",
            "END": "Azione - fine",
            "READY": "Pronto",
        }
        subs = {
            "CALIB": f"Clicca 4 angoli del campo in senso orario  ({len(self.data['calib_pts'])}/4)",
            "GRASS": f"Clicca 3 zone di erba  ({len(self.data['grass_samples'])}/3)",
            "LINES": f"Clicca 2 zone di linea bianca  ({len(self.data['line_samples'])}/2)",
            "START": "Naviga al frame iniziale, clicca AL CENTRO del pallone",
            "END": "Naviga al frame finale, clicca AL CENTRO del pallone",

            "READY": f"{len(self.data['actions'])} azione/i registrate - 'Nuova azione' oppure 'Esporta video'",
        }
        self.lbl_phase_title.configure(
            text=titles.get(self.current_step, self.current_step)
        )
        self.lbl_phase_sub.configure(text=subs.get(self.current_step, ""))

    def _update_stepper(self):
        mapping = {
            "CALIB": "CALIB",
            "GRASS": "GRASS",
            "LINES": "LINES",
            "START": "ACTION",
            "END": "ACTION",
            "READY": "READY",
        }
        active_key = mapping.get(self.current_step)
        order = [k for k, _, _ in self.STEPS]
        active_idx = order.index(active_key) if active_key in order else -1
        for i, (key, _, _) in enumerate(self.STEPS):
            dot, lbl = self.step_rows[key]
            if i < active_idx:
                dot.configure(text="●", text_color=ACCENT)
                lbl.configure(text_color=ACCENT)
            elif i == active_idx:
                dot.configure(text="➤", text_color=WARN)
                lbl.configure(text_color=TEXT_LIGHT)
            else:
                dot.configure(text="○", text_color=TEXT_MUTED)
                lbl.configure(text_color=TEXT_MUTED)

    def _update_actions_list(self):
        for w in self.actions_list.winfo_children():
            w.destroy()
        actions = self.data["actions"]
        if not actions and not self.data["temp_start"]:
            ctk.CTkLabel(
                self.actions_list,
                text="Nessuna azione",
                font=ctk.CTkFont(size=11),
                text_color=TEXT_MUTED,
            ).pack(anchor="w", padx=6, pady=4)
        for idx, act in enumerate(actions):
            df = act["end_frame"] - act["start_frame"]
            ctk.CTkLabel(
                self.actions_list,
                text=f"#{idx + 1}   f{act['start_frame']} -> {act['end_frame']}   ({df} fr)",
                font=ctk.CTkFont(size=11),
                text_color=ACCENT,
                anchor="w",
            ).pack(fill="x", padx=6, pady=2)
        if self.data["temp_start"]:
            ctk.CTkLabel(
                self.actions_list,
                text=f"... f{self.data['temp_start']['frame']} -> in corso",
                font=ctk.CTkFont(size=11),
                text_color=WARN,
                anchor="w",
            ).pack(fill="x", padx=6, pady=2)

    def _update_swatches(self):
        if self.data["grass_samples"]:
            avg = np.mean(self.data["grass_samples"], axis=0)
            self.swatch_grass.configure(
                text="Erba", fg_color=_bgr_to_hex(avg), text_color="#06120b"
            )
        else:
            self.swatch_grass.configure(
                text="Erba -", fg_color=CARD_BG, text_color=TEXT_MUTED
            )
        if self.data["line_samples"]:
            avg = np.mean(self.data["line_samples"], axis=0)
            self.swatch_lines.configure(
                text="Linee", fg_color=_bgr_to_hex(avg), text_color="#06120b"
            )
        else:
            self.swatch_lines.configure(
                text="Linee -", fg_color=CARD_BG, text_color=TEXT_MUTED
            )

    def _update_buttons_state(self):
        ready = self.current_step == "READY"
        self.btn_new_action.configure(state="normal" if ready else "disabled")
        self.btn_export.configure(
            state="normal" if (ready and self.data["actions"]) else "disabled"
        )

    def _update_zoom_preview(self):
        if not self.frames:
            return
        x = int(
            max(0, min(self.frames[self.frame_idx].shape[1] - 1, self.mouse_pos[0]))
        )
        y = int(
            max(0, min(self.frames[self.frame_idx].shape[0] - 1, self.mouse_pos[1]))
        )
        patch = shadow_get_zoom_patch(self.frames[self.frame_idx], x, y)
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if self._zoom_img is None:
            self._zoom_img = ctk.CTkImage(
                light_image=pil_img, dark_image=pil_img, size=pil_img.size
            )
            self.zoom_label.configure(image=self._zoom_img, text="")
        else:
            self._zoom_img.configure(
                light_image=pil_img, dark_image=pil_img, size=pil_img.size
            )

    def new_action(self):
        if self.current_step != "READY":
            return
        if self.data["actions"]:
            last = self.data["actions"][-1]
            self.data["temp_start"] = {
                "frame": last["end_frame"],
                "pos": last["end_pos"],
            }
            self.current_step = "END"
        else:
            self.current_step = "START"
        self._refresh_display()

    def reset_all(self, confirm=True):
        if confirm:
            if not messagebox.askyesno(
                APP_TITLE, "Azzerare calibrazione, campioni e azioni per questo video?"
            ):
                return
        self.data = {
            "calib_pts": [],
            "grass_samples": [],
            "line_samples": [],
            "actions": [],
            "temp_start": None,
        }
        self.current_step = "CALIB"
        self.show_grid = False
        self.switch_grid.deselect()
        self.switch_grid.configure(state="disabled")
        if self.frames:
            self._refresh_display()
        else:
            self._update_phase_banner()
            self._update_stepper()
            self._update_actions_list()
            self._update_swatches()
            self._update_buttons_state()

    def export_video(self):
        if self._exporting or not self.data["actions"]:
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
        self.btn_new_action.configure(state="disabled")
        self.progress.set(0)
        self.lbl_export_status.configure(
            text="Avvio esportazione...", text_color=TEXT_MUTED
        )
        self.app.set_status("Esportazione video con ombra in corso...", "info")

        frames = self.frames
        fps = self.fps
        calib_pts = list(self.data["calib_pts"])
        grass_samples = list(self.data["grass_samples"])
        line_samples = list(self.data["line_samples"])
        actions = [dict(a) for a in self.data["actions"]]
        shadow_size = self.shadow_size_val

        threading.Thread(
            target=self._export_worker,
            args=(
                frames,
                fps,
                calib_pts,
                grass_samples,
                line_samples,
                actions,
                shadow_size,
                out_fn,
            ),
            daemon=True,
        ).start()
        self._poll_export()

    def _export_worker(
        self,
        frames,
        fps,
        calib_pts,
        grass_samples,
        line_samples,
        actions,
        shadow_size,
        out_fn,
    ):
        try:
            out_fn = self._generate_output_video(
                frames,
                fps,
                calib_pts,
                grass_samples,
                line_samples,
                actions,
                shadow_size,
                out_fn,
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
        self.btn_new_action.configure(state="normal")
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
            f"Video con ombra esportato: {pathlib.Path(out_fn).name}", "success"
        )
        self.refresh_last_output()

    def _generate_output_video(
        self,
        frames,
        fps,
        calib_pts,
        grass_samples,
        line_samples,
        actions,
        shadow_size,
        out_fn,
    ):
        SHADOW_OUTPUT_FOLDER.mkdir(exist_ok=True)

        calib_hist = [shadow_track_calib_full(frames, p) for p in calib_pts]

        grass_avg = (
            np.mean(grass_samples, axis=0)
            if grass_samples
            else np.array([0.0, 180.0, 0.0])
        )
        line_avg = (
            np.mean(line_samples, axis=0)
            if line_samples
            else np.array([210.0, 210.0, 210.0])
        )

        He_list = {}
        for act in actions:
            for key_f in (act["start_frame"], act["end_frame"]):
                if key_f not in He_list:
                    He_list[key_f] = shadow_compute_homography_tracked(
                        calib_hist, key_f
                    )

        actions_ext = []
        for act in actions:
            sf, ef = act["start_frame"], act["end_frame"]
            Hs, _ = He_list[sf]
            He, _ = He_list[ef]
            start_f = (
                shadow_img_to_field(act["start_pos"], Hs) if Hs is not None else None
            )
            end_f = shadow_img_to_field(act["end_pos"], He) if He is not None else None
            actions_ext.append(
                {
                    "meta": act,
                    "start_f": start_f,
                    "end_f": end_f,
                    "last_vis": None,
                    "last_center": None,
                }
            )

        h, w, _ = frames[0].shape
        out_fn = str(out_fn)
        writer = cv2.VideoWriter(
            out_fn, cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (w, h)
        )

        tot = len(frames)
        for i in range(tot):
            if i % 20 == 0:
                self._export_queue.put(("progress", i / max(tot - 1, 1)))

            frame = frames[i].copy()
            H, H_inv = shadow_compute_homography_tracked(
                calib_hist, i, img_shape=frame.shape
            )

            if H is not None and H_inv is not None:
                for ao in actions_ext:
                    act = ao["meta"]
                    sf, ef = act["start_frame"], act["end_frame"]
                    span = ef - sf
                    if not (sf <= i <= ef) or span <= 0:
                        continue
                    if ao["start_f"] is None or ao["end_f"] is None:
                        continue

                    t = (i - sf) / span
                    ball_f = ao["start_f"] + t * (ao["end_f"] - ao["start_f"])
                    ball_img = shadow_field_to_img(ball_f, H_inv)

                    if (
                        not np.all(np.isfinite(ball_img))
                        or ball_img[0] < -w * 2
                        or ball_img[0] > w * 3
                        or ball_img[1] < -h * 2
                        or ball_img[1] > h * 3
                    ):
                        continue

                    try:
                        res = shadow_compute_ellipse_params(
                            ball_img, H, H_inv, shadow_size
                        )
                        if res is None:
                            continue
                        center, axes, angle = res
                    except Exception:
                        continue

                    if ao["last_center"] is None:
                        ao["last_center"] = center.copy()
                    else:
                        ao["last_center"] = ao["last_center"] + SHADOW_SMOOTH_POS * (
                            center - ao["last_center"]
                        )
                    center = ao["last_center"]

                    vis = shadow_get_visibility(
                        frames[i], center, axes, grass_avg, line_avg
                    )
                    if ao["last_vis"] is None:
                        ao["last_vis"] = vis
                    else:
                        ao["last_vis"] += SHADOW_SMOOTH_VIS * (vis - ao["last_vis"])

                    frame = shadow_draw_ellipse(
                        frame, center, axes, angle, alpha_mult=ao["last_vis"]
                    )

            writer.write(frame)

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
