"""
MIRA - Componente Barra di Avanzamento con Zona Target
Widget canvas per visualizzare l'avanzamento temporale e la finestra dell'azione target durante il test.
"""

import tkinter as tk
import customtkinter as ctk
from ..constants import BG_DARK, ACCENT

class ActionProgressBar(ctk.CTkFrame):
    """
    Barra di avanzamento stile player video per la pagina di test.
    Mostra:
    - traccia di riproduzione con riempimento fluido
    - zona evidenziata indicativa a grandi linee per la posizione dell'azione.
    """

    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.total_frames = 0
        self.target_frame = None
        self.fps = 25.0
        self.current_frame = 0
        self._cached_width = 0
        self._last_rendered_frame = -1

        self._bg_id = None
        self._action_id = None
        self._fill_id = None
        self._cursor_id = None

        self.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self,
            height=20,
            bg=BG_DARK,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.canvas.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        self._cached_width = event.width
        self.redraw()

    def set_config(self, total_frames, target_frame, fps=25.0):
        self.total_frames = max(1, total_frames)
        self.target_frame = target_frame
        self.fps = max(1.0, fps)
        self.current_frame = 0
        self._last_rendered_frame = -1
        self._last_fill_x = None
        self.redraw()

    def update_progress(self, frame_idx):
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        if frame_idx == self._last_rendered_frame:
            return
        self.current_frame = frame_idx
        self._last_rendered_frame = frame_idx

        if self._fill_id is None or self._cursor_id is None:
            self.redraw()
            return

        w = self._cached_width or self.canvas.winfo_width()
        if w < 20 or self.total_frames <= 0:
            return

        pad_x = 12
        track_y1 = 5
        track_y2 = 15
        track_w = w - 2 * pad_x

        ratio = self.current_frame / max(1, self.total_frames - 1)
        fill_x = round(pad_x + ratio * track_w)
        if getattr(self, "_last_fill_x", None) == fill_x:
            return
        self._last_fill_x = fill_x

        self.canvas.coords(self._fill_id, pad_x, track_y1 + 1, max(pad_x, fill_x), track_y2 - 1)
        self.canvas.coords(self._cursor_id, fill_x - 5, track_y1 - 1, fill_x + 5, track_y2 + 1)

    def redraw(self):
        self.canvas.delete("all")
        self._bg_id = None
        self._action_id = None
        self._fill_id = None
        self._cursor_id = None
        self._last_fill_x = None

        w = self._cached_width or self.canvas.winfo_width()
        if w < 20:
            return

        pad_x = 12
        track_y1 = 5
        track_y2 = 15
        track_w = w - 2 * pad_x

        # Sfondo traccia
        self._bg_id = self.canvas.create_rectangle(
            pad_x, track_y1, pad_x + track_w, track_y2,
            fill="#1e1e26", outline="#323242", width=1
        )

        # Zona evidenziata indicativa per l'azione
        if self.target_frame is not None and 0 <= self.target_frame < self.total_frames:
            win = max(5, int(self.total_frames * 0.08))
            f_start = max(0, self.target_frame - win)
            f_end = min(self.total_frames - 1, self.target_frame + win)

            x_start = pad_x + (f_start / self.total_frames) * track_w
            x_end = pad_x + (f_end / self.total_frames) * track_w

            self._action_id = self.canvas.create_rectangle(
                x_start, track_y1, x_end, track_y2,
                fill="#ffb703", outline="#ff9f1c", width=1
            )

        # Barra di progresso riprodotta (verde accento)
        ratio = self.current_frame / max(1, self.total_frames - 1) if self.total_frames > 0 else 0.0
        fill_x = pad_x + ratio * track_w

        self._fill_id = self.canvas.create_rectangle(
            pad_x, track_y1 + 1, max(pad_x, fill_x), track_y2 - 1,
            fill=ACCENT, outline=""
        )
        # Cursore (playhead)
        self._cursor_id = self.canvas.create_oval(
            fill_x - 5, track_y1 - 1, fill_x + 5, track_y2 + 1,
            fill="#ffffff", outline=ACCENT, width=2
        )


