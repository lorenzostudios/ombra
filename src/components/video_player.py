"""
MIRA - Componente Player Video
Player video ad alte prestazioni (60+ FPS) basato su OpenCV e Pillow/Tkinter.
Include pre-caching asincrono dei frame, sincronizzazione clock ad alta precisione
e controlli integrati di riproduzione frame-by-frame.
"""

import time
import queue
import pathlib
import platform
import threading
import tkinter as tk
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from ..constants import PANEL_BG, TEXT_MUTED, ACCENT, ACCENT_HOVER
from ..utils import load_video_frames

class VideoPlayerFrame(ctk.CTkFrame):
    """
    Player video riusabile ad alte prestazioni basato su CustomTkinter e Tkinter nativo.
    Supporta riproduzione fluida a frame rate nativo (60+ FPS) senza frame drop né slow motion.
    Include pre-caching asincrono in background di PIL e PhotoImage e clock ad alta precisione time.perf_counter().
    """

    LOAD_TIMEOUT_SEC = 20.0  # oltre questo tempo, un caricamento e' considerato fallito

    def __init__(
        self,
        master,
        show_controls=True,
        on_frame_change=None,
        on_end=None,
        max_render_w=None,
        max_render_h=None,
        **kwargs,
    ):
        kwargs.setdefault("fg_color", PANEL_BG)
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)

        self.show_controls = show_controls
        self.on_frame_change = on_frame_change
        self.on_end = on_end

        is_mac = platform.system() == "Darwin"
        self.max_render_w = max_render_w or (1152 if is_mac else 1280)
        self.max_render_h = max_render_h or (648 if is_mac else 720)

        self.frames = []
        self.fps = 25.0
        self.frame_idx = 0
        self.playing = False
        self.start_perf = None

        self._tick_after_id = None
        self._load_after_id = None
        self._ended = False

        self._current_photo = None
        self._load_queue = queue.Queue()
        self._load_generation = 0
        self._precache_generation = 0
        self._pil_cache = {}
        self._photo_cache = {}
        self._last_w = 0
        self._last_h = 0
        self._last_nw = 0
        self._last_nh = 0


        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Label nativo Tkinter ad alte prestazioni integrato nel container scuro
        self.image_label = tk.Label(
            self,
            text="",
            font=("TkDefaultFont", 13),
            fg=TEXT_MUTED,
            bg=PANEL_BG,
            bd=0,
            highlightthickness=0,
        )

        self.image_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.image_label.bind("<Configure>", self._on_label_resize)

        if self.show_controls:
            ctrl = ctk.CTkFrame(self, fg_color="transparent")
            ctrl.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 8))
            ctrl.grid_columnconfigure(2, weight=1)

            self.btn_prev = ctk.CTkButton(
                ctrl, text="< -1", width=48, command=self.prev_frame
            )
            self.btn_prev.grid(row=0, column=0, padx=(0, 4))
            self.btn_play = ctk.CTkButton(
                ctrl,
                text="▶",
                width=48,
                font=ctk.CTkFont(size=16),
                command=self.toggle_play,
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
                text_color="#06120b",
            )
            self.btn_play.grid(row=0, column=1, padx=4)
            self.slider = ctk.CTkSlider(
                ctrl, from_=0, to=1, number_of_steps=1, command=self._on_slider
            )
            self.slider.set(0)
            self.slider.grid(row=0, column=2, sticky="ew", padx=8)
            self.btn_next = ctk.CTkButton(
                ctrl, text="+1 >", width=48, command=self.next_frame
            )
            self.btn_next.grid(row=0, column=3, padx=(4, 0))

            self.lbl_frame = ctk.CTkLabel(
                ctrl,
                text="Frame 0/0   0.00s",
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=12),
            )
            self.lbl_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

    # ── calcolo dinamico risoluzione display ottimale ───────────────────
    def _compute_display_size(self):
        """
        Calcola la dimensione ottimale di visualizzazione a piena risoluzione.
        Evita downscaling a miniature quando il player non è ancora mappato,
        e impone un limite massimo controllato per garantire 60+ FPS fluidi
        senza rallentamenti o frame drop (specialmente su macOS Retina).
        """
        if not self.frames:
            return min(1280, self.max_render_w), min(720, self.max_render_h)

        ih, iw = self.frames[0].shape[:2]

        # 1. Prova la dimensione reale del widget label se gia' mappato e visibile
        lw = self.image_label.winfo_width()
        lh = self.image_label.winfo_height()

        # 2. Se non ancora mappato (>50px), usa il contenitore VideoPlayerFrame
        if lw <= 50 or lh <= 50:
            lw = self.winfo_width()
            lh = self.winfo_height()
            if self.show_controls:
                lh -= 55

        # 3. Se ancora non mappato, ricava lo spazio disponibile dalla finestra principale (App)
        if lw <= 50 or lh <= 50:
            try:
                top = self.winfo_toplevel()
                top_w = top.winfo_width()
                top_h = top.winfo_height()
                if top_w > 100 and top_h > 100:
                    lw = top_w - 48
                    lh = top_h - (270 if not self.show_controls else 310)
            except Exception:
                pass

        # 4. Fallback ad alta definizione di sicurezza
        if lw <= 50 or lh <= 50:
            lw, lh = 1240, 680

        # Cap prestazionale massimo per garantire 30-60 FPS fluidi senza scatti
        avail_w = min(lw, self.max_render_w)
        avail_h = min(lh, self.max_render_h)

        # Adattamento proporzionale senza distorsioni
        scale = min(avail_w / iw, avail_h / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        return nw, nh

    def update_display_size(self):
        """Verifica e aggiorna la risoluzione di rendering quando la vista diventa visibile."""
        if not self.frames or self.playing:
            return
        nw, nh = self._compute_display_size()
        # Se la variazione è minima (< 25px), preserva la cache già generata per evitare ricaricamenti a vuoto
        if self._last_nw <= 0 or abs(nw - self._last_nw) > 25 or abs(nh - self._last_nh) > 25:
            self._last_nw = nw
            self._last_nh = nh
            self._pil_cache.clear()
            self._photo_cache.clear()
            self._precache_generation += 1
            self._start_precache_worker(self._precache_generation, nw, nh)
            self._refresh_current_frame()

    # ── caricamento (threaded, con coda + after()) ────────────────────
    def load_video(self, path, expected_fps=None, on_loaded=None, status_cb=None):
        self.pause()
        self._cancel_load_poll()
        self.frames = []
        self._pil_cache.clear()
        self._photo_cache.clear()
        self.frame_idx = 0
        self._ended = False
        self._load_generation += 1
        self._precache_generation += 1
        my_gen = self._load_generation
        start_perf = time.perf_counter()

        while True:
            try:
                self._load_queue.get_nowait()
            except queue.Empty:
                break

        if status_cb:
            status_cb(f"Caricamento {pathlib.Path(path).name}...")

        def worker():
            try:
                frames, fps = load_video_frames(path)
                final_fps = float(expected_fps) if (expected_fps is not None and float(expected_fps) > 0) else float(fps)
                self._load_queue.put((my_gen, "ok", frames, final_fps))
            except Exception as e:  # noqa: BLE001
                self._load_queue.put((my_gen, "error", str(e), None))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_load(on_loaded, status_cb, my_gen, start_perf)

    def _cancel_load_poll(self):
        if self._load_after_id is not None:
            try:
                self.after_cancel(self._load_after_id)
            except Exception:
                pass
            self._load_after_id = None

    def _poll_load(self, on_loaded, status_cb, my_gen, start_perf):
        self._load_after_id = None
        if my_gen != self._load_generation:
            return

        while True:
            try:
                gen, kind, a, b = self._load_queue.get_nowait()
            except queue.Empty:
                break
            if gen != my_gen:
                continue
            self._handle_load_result(kind, a, b, on_loaded, status_cb)
            return

        if time.perf_counter() - start_perf > self.LOAD_TIMEOUT_SEC:
            self._handle_load_result(
                "error",
                f"timeout dopo {self.LOAD_TIMEOUT_SEC:.0f}s",
                None,
                on_loaded,
                status_cb,
            )
            return

        self._load_after_id = self.after(
            40, lambda: self._poll_load(on_loaded, status_cb, my_gen, start_perf)
        )

    def _handle_load_result(self, kind, a, b, on_loaded, status_cb):
        if kind == "error":
            self._current_photo = None
            try:
                self.image_label.configure(image="", text="Errore caricamento video")
            except Exception:
                pass
            if status_cb:
                status_cb(f"Errore caricamento video: {a}")
            if on_loaded:
                on_loaded(False, 0, 0.0)
            return

        self.frames, self.fps = a, float(b or 25.0)
        self.frame_idx = 0
        self._ended = False
        if self.show_controls:
            n_max = max(0, len(self.frames) - 1)
            self.slider.configure(
                from_=0, to=max(n_max, 1), number_of_steps=max(n_max, 1)
            )
            self.slider.set(0)

        if self.frames:
            nw, nh = self._compute_display_size()
            self._last_nw = nw
            self._last_nh = nh
            self._start_precache_worker(self._precache_generation, nw, nh)
            self.show_frame(0)

        if status_cb:
            status_cb(f"Video caricato ({len(self.frames)} frame, {self.fps:.2f} fps)")
        if on_loaded:
            on_loaded(True, len(self.frames), self.fps)

    def _start_precache_worker(self, generation, nw, nh):
        """Pre-elabora in background tutti i frame ridimensionati e avvia il precaching PhotoImage."""
        frames_ref = self.frames
        self._precache_event = threading.Event()

        def precache_worker():
            for i, f in enumerate(frames_ref):
                if self._precache_generation != generation:
                    break
                cached = self._pil_cache.get(i)
                if cached is not None and cached.size == (nw, nh):
                    continue
                resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                self._pil_cache[i] = Image.fromarray(rgb)

            if self._precache_generation == generation:
                self._precache_event.set()

        threading.Thread(target=precache_worker, daemon=True).start()
        self._poll_photo_precache(generation)

    def _poll_photo_precache(self, generation):
        if generation != self._precache_generation or self.playing:
            return
        if getattr(self, "_precache_event", None) and self._precache_event.is_set():
            self._start_photo_precache(generation)
        else:
            self.after(25, lambda: self._poll_photo_precache(generation))

    def _start_photo_precache(self, generation, start_idx=0):
        """Pre-genera progressivamente i PhotoImage Tkinter sul thread principale per eliminare ogni overhead durante il play."""
        if generation != self._precache_generation or not self.frames or self.playing:
            return

        chunk_size = 25
        total = len(self.frames)
        end_idx = min(start_idx + chunk_size, total)

        for i in range(start_idx, end_idx):
            if generation != self._precache_generation or self.playing:
                return
            if i in self._photo_cache:
                continue
            pil_img = self._get_pil_frame(i)
            if pil_img is not None:
                try:
                    self._photo_cache[i] = ImageTk.PhotoImage(pil_img)
                except Exception:
                    pass

        if end_idx < total and generation == self._precache_generation and not self.playing:
            try:
                self.after(5, lambda: self._start_photo_precache(generation, end_idx))
            except Exception:
                pass

    def set_processed_frames(self, frames, fps=None):
        """Aggiorna i fotogrammi del player (es. output IA) sincronizzando controlli e slider."""
        self.pause()
        self.frames = frames or []
        self._pil_cache.clear()
        self._photo_cache.clear()
        self._precache_generation += 1
        if fps and float(fps) > 0:
            self.fps = float(fps)
        self.frame_idx = 0
        self._ended = False
        if self.show_controls:
            n_max = max(0, len(self.frames) - 1)
            self.slider.configure(
                from_=0, to=max(n_max, 1), number_of_steps=max(n_max, 1)
            )
            self.slider.set(0)
        if self.frames:
            nw, nh = self._compute_display_size()
            self._last_nw = nw
            self._last_nh = nh
            self._start_precache_worker(self._precache_generation, nw, nh)
            self.show_frame(0)

    # ── rendering ───────────────────────────────────────────────────
    def show_frame(self, idx):
        if not self.frames:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        self.frame_idx = idx
        self._render(idx)

        if self.show_controls:
            self.slider.set(idx)
            t = idx / max(self.fps, 1.0)
            self.lbl_frame.configure(
                text=f"Frame {idx}/{len(self.frames) - 1}   {t:.2f}s"
            )
        if self.on_frame_change:
            self.on_frame_change(idx)

    def _on_label_resize(self, event):
        if self.playing:
            return
        w, h = event.width, event.height
        if abs(w - self._last_w) < 20 and abs(h - self._last_h) < 20:
            return
        self._last_w = w
        self._last_h = h
        if not self.frames:
            return

        nw, nh = self._compute_display_size()
        if abs(nw - self._last_nw) <= 25 and abs(nh - self._last_nh) <= 25:
            return

        self._last_nw = nw
        self._last_nh = nh
        self._pil_cache.clear()
        self._photo_cache.clear()
        self._precache_generation += 1
        self._start_precache_worker(self._precache_generation, nw, nh)
        self._refresh_current_frame()


    def _refresh_current_frame(self):
        if self.frames:
            self._render(self.frame_idx)

    def _get_pil_frame(self, idx):
        if not self.frames:
            return None
        idx = max(0, min(idx, len(self.frames) - 1))
        if self._last_nw <= 0 or self._last_nh <= 0:
            self._last_nw, self._last_nh = self._compute_display_size()

        nw, nh = self._last_nw, self._last_nh
        pil_img = self._pil_cache.get(idx)
        if pil_img is None or pil_img.size != (nw, nh):
            frame_bgr = self.frames[idx]
            resized_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self._pil_cache[idx] = pil_img
        return pil_img

    def _render(self, idx):
        if not self.frames:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        photo = self._photo_cache.get(idx)
        if photo is None:
            pil_img = self._get_pil_frame(idx)
            if pil_img is None:
                return
            photo = ImageTk.PhotoImage(pil_img)
            self._photo_cache[idx] = photo

        self.image_label.configure(image=photo, text="")
        self._current_photo = photo

    # ── playback fluido a frame rate nativo con clock di precisione ──
    def play(self, restart=False):
        if not self.frames:
            return
        self.pause()
        if restart or self.frame_idx >= len(self.frames) - 1 or self._ended:
            self.frame_idx = 0
            self.show_frame(0)
        self._ended = False
        self.playing = True
        fps_val = float(self.fps) if self.fps and float(self.fps) > 0 else 25.0
        self.start_perf = time.perf_counter() - (self.frame_idx / fps_val)
        if self.show_controls:
            self.btn_play.configure(text="⏸")
        self._tick()

    def pause(self):
        self.playing = False
        if self._tick_after_id is not None:
            try:
                self.after_cancel(self._tick_after_id)
            except Exception:
                pass
            self._tick_after_id = None
        if self.show_controls:
            self.btn_play.configure(text="▶")

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def _tick(self):
        self._tick_after_id = None
        if not self.playing or not self.frames:
            return

        now = time.perf_counter()
        elapsed = now - self.start_perf
        fps_val = float(self.fps) if self.fps and float(self.fps) > 0 else 25.0
        total_frames = len(self.frames)
        duration_sec = max(0.01, (total_frames - 1) / fps_val)

        # Fine video raggiunta unicamente quando il tempo reale trascorso copre la durata
        if elapsed >= duration_sec:
            self.show_frame(total_frames - 1)
            self.playing = False
            if self.show_controls:
                self.btn_play.configure(text="▶")
            if self.on_end is not None and not self._ended:
                self._ended = True
                self.on_end()
            return


        target_idx = int(elapsed * fps_val)
        target_idx = max(0, min(target_idx, total_frames - 1))

        if target_idx != self.frame_idx:
            self.show_frame(target_idx)

        # Calcola istante esatto del prossimo fotogramma
        next_target = target_idx + 1
        if next_target >= total_frames:
            next_due = self.start_perf + ((total_frames - 1) / fps_val)
        else:
            next_due = self.start_perf + (next_target / fps_val)

        time_left = next_due - time.perf_counter()
        if time_left > 0.003:
            delay_ms = max(1, int((time_left - 0.002) * 1000.0))
        else:
            delay_ms = 1

        self._tick_after_id = self.after(delay_ms, self._tick)

    def next_frame(self):
        self.pause()
        self.show_frame(self.frame_idx + 1)

    def prev_frame(self):
        self.pause()
        self.show_frame(self.frame_idx - 1)

    def _on_slider(self, value):
        self.pause()
        self.show_frame(int(float(value)))

    def seek(self, value):
        self.pause()
        self.show_frame(int(value))

    def elapsed_ms(self):
        if self.start_perf is None:
            return None
        return int((time.perf_counter() - self.start_perf) * 1000)

    def release(self):
        self.pause()
        self._cancel_load_poll()
        self._load_generation += 1
        self._precache_generation += 1
        while True:
            try:
                self._load_queue.get_nowait()
            except queue.Empty:
                break
        self.frames = []
        self._pil_cache.clear()
        self._photo_cache.clear()
        self._current_photo = None
        self._last_w = 0
        self._last_h = 0
        self._last_nw = 0
        self._last_nh = 0
        self.frame_idx = 0
        self._ended = False
        try:
            self.image_label.configure(image="", text="")
            if self.show_controls:
                self.lbl_frame.configure(text="Frame 0/0   0.00s")
                self.slider.set(0)
        except Exception:
            pass

