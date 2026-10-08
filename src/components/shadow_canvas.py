"""
OMBRA - Componente Canvas Interattivo dell'Ombra
Renderizza il frame corrente con l'ellisse dell'ombra e traduce le interazioni del mouse
(click, trascinamento, ridimensionamento) in coordinate pixel native del video.
"""

import cv2
import customtkinter as ctk
from PIL import Image
from ..constants import PANEL_BG, TEXT_MUTED
from ..utils import _ctk_clear_image


class ShadowCanvas(ctk.CTkFrame):
    """
    Mostra il frame corrente (con le annotazioni gia' disegnate sui pixel)
    e traduce i click/drag del mouse in coordinate del frame originale.
    """

    def __init__(
        self,
        master,
        on_click=None,
        on_move=None,
        on_resize=None,
        on_press=None,
        on_drag=None,
        on_release=None,
        **kwargs,
    ):
        kwargs.setdefault("fg_color", PANEL_BG)
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)
        self.on_click = on_click
        self.on_move = on_move
        self.on_resize = on_resize
        self.on_press = on_press
        self.on_drag = on_drag
        self.on_release = on_release

        self._ctk_img = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._frame_size = (1, 1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.image_label = ctk.CTkLabel(
            self,
            text="Nessun video caricato",
            text_color=TEXT_MUTED,
            fg_color="transparent",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        for _w in (self.image_label._label, self.image_label._canvas):
            _w.bind("<Button-1>", self._handle_button_down)
            _w.bind("<B1-Motion>", self._handle_button_drag)
            _w.bind("<ButtonRelease-1>", self._handle_button_up)
            _w.bind("<Motion>", self._handle_motion)
        self.image_label.bind("<Configure>", self._handle_configure)

    def get_scale(self) -> float:
        return self._scale

    def render(self, frame_bgr):
        ih, iw = frame_bgr.shape[:2]
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        if w <= 50 or h <= 50:
            w = self.winfo_width()
            h = self.winfo_height()
        if w <= 50 or h <= 50:
            try:
                top = self.winfo_toplevel()
                w = max(640, top.winfo_width() - 360)
                h = max(400, top.winfo_height() - 160)
            except Exception:
                w, h = 960, 540

        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        pil_resized = Image.fromarray(rgb)

        self._scale = scale
        self._offset = ((w - nw) // 2, (h - nh) // 2)
        self._frame_size = (iw, ih)

        if self._ctk_img is None:
            self._ctk_img = ctk.CTkImage(
                light_image=pil_resized, dark_image=pil_resized, size=(nw, nh)
            )
            self.image_label.configure(image=self._ctk_img, text="")
        else:
            try:
                self._ctk_img.configure(
                    light_image=pil_resized, dark_image=pil_resized, size=(nw, nh)
                )
            except ValueError:
                self._ctk_img = ctk.CTkImage(
                    light_image=pil_resized, dark_image=pil_resized, size=(nw, nh)
                )
                self.image_label.configure(image=self._ctk_img, text="")

    def _to_frame_coords(self, wx, wy):
        ox, oy = self._offset
        return (wx - ox) / self._scale, (wy - oy) / self._scale

    def _event_to_frame(self, event):
        wx = event.x_root - self.image_label.winfo_rootx()
        wy = event.y_root - self.image_label.winfo_rooty()
        return self._to_frame_coords(wx, wy)

    def _handle_button_down(self, event):
        if self._ctk_img is None:
            return
        fx, fy = self._event_to_frame(event)
        if self.on_press is not None:
            self.on_press(fx, fy)
            return
        if self.on_click is not None:
            iw, ih = self._frame_size
            if 0 <= fx < iw and 0 <= fy < ih:
                self.on_click(fx, fy)

    def _handle_button_drag(self, event):
        if self._ctk_img is None or self.on_drag is None:
            return
        fx, fy = self._event_to_frame(event)
        self.on_drag(fx, fy)

    def _handle_button_up(self, event):
        if self._ctk_img is None or self.on_release is None:
            return
        fx, fy = self._event_to_frame(event)
        self.on_release(fx, fy)

    def _handle_motion(self, event):
        if self._ctk_img is None or self.on_move is None:
            return
        fx, fy = self._event_to_frame(event)
        self.on_move(fx, fy)

    def _handle_configure(self, event):
        if self.on_resize:
            self.on_resize()

    def release(self):
        _ctk_clear_image(self.image_label)
        self._ctk_img = None
        try:
            self.image_label.configure(text="Nessun video caricato")
        except Exception:
            pass
