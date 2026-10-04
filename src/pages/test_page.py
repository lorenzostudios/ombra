"""
MIRA - Esecuzione Esperimento di Percezione Visiva (TestPage)
Gestisce la somministrazione sequenziale dei trial, la riproduzione video a schermo,
la registrazione delle risposte tramite tastiera (Shift sinistro/destro) o voce (Whisper),
il tracciamento della latenza e il calcolo dei tempi di reazione.
"""

import re
import cv2
import time
import queue
import pathlib
import datetime
import threading
import customtkinter as ctk

from ..constants import (
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    ACCENT,
    ACCENT_HOVER,
    BLUE,
    BLUE_HOVER,
    DANGER,
    DANGER_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARN,
    DEFAULT_QUESTION,
    AUDIO_RESPONSES_FOLDER,
    VOICE_AVAILABLE,
)
from ..utils import is_augmented, trial_counter_str, _ctk_clear_image
from ..voice_engine import VoiceAnswerEngine, match_voice_answer
from ..components.video_player import VideoPlayerFrame
from ..components.action_progress import ActionProgressBar




class TestPage(ctk.CTkFrame):
    """
    Stati: intro -> question -> playing -> (finalizza trial) -> question ... -> finished
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app
        self.state = "intro"

        self.current_question = ""
        self.current_target_frame = None
        self.current_target_ms = None
        self.current_label_yes = "SI"
        self.current_label_no = "NO"
        self.current_correct_answer = None
        self.answer_given = False
        self.answer = ""
        self.answer_key = ""
        self.response_time_ms = None
        self.response_frame = None
        self._current_vpath = None

        self._advancing = False
        self._trial_finalized = False
        self._start_requested = False

        self._voice_engine = None
        self._voice_queue = queue.Queue()
        self.voice_transcript = ""
        self.audio_file = ""

        self._build()

    def _build(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.intro_view = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.intro_view.grid(row=0, column=0, sticky="nsew")
        c = ctk.CTkFrame(
            self.intro_view,
            fg_color=CARD_BG,
            corner_radius=16,
            border_width=1,
            border_color=CARD_BORDER,
        )
        c.place(relx=0.5, rely=0.45, anchor="center")
        self.lbl_intro_title = ctk.CTkLabel(
            c,
            text="Pronto per iniziare?",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_intro_title.pack(pady=(52, 32), padx=80)
        self.intro_instr_frame = ctk.CTkFrame(c, fg_color="transparent")
        self.intro_instr_frame.pack(pady=(0, 36), padx=60)
        ctk.CTkButton(
            c,
            text="Inizia test (Spazio)",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            width=220,
            height=44,
            command=self.start_next_trial,
        ).pack(pady=(0, 52))

        # ── QUESTION ───────────────────────────────────────────
        self.question_view = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.question_view.grid(row=0, column=0, sticky="nsew")

        q_wrapper = ctk.CTkFrame(self.question_view, fg_color="transparent")
        q_wrapper.place(relx=0.5, rely=0.48, anchor="center")

        qheader = ctk.CTkFrame(q_wrapper, fg_color="transparent")
        qheader.pack(fill="x", pady=(0, 16))
        self.lbl_progress = ctk.CTkLabel(
            qheader,
            text="",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_progress.pack(anchor="center")
        self.trial_progress = ctk.CTkProgressBar(
            qheader, width=220, progress_color=ACCENT
        )
        self.trial_progress.set(0)
        self.trial_progress.pack(anchor="center", pady=(6, 0))

        qcard = ctk.CTkFrame(
            q_wrapper,
            fg_color=CARD_BG,
            corner_radius=16,
            border_width=1,
            border_color=CARD_BORDER,
            width=980,
            height=410,
        )
        qcard.pack()
        qcard.pack_propagate(False)

        body_row = ctk.CTkFrame(qcard, fg_color="transparent")
        body_row.pack(fill="both", expand=True, padx=30, pady=26)

        left_col = ctk.CTkFrame(body_row, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True)

        right_col = ctk.CTkFrame(body_row, fg_color="transparent")
        right_col.pack(side="left", padx=(24, 0))

        # Pulsante ancorato in basso con la stessa distanza del titolo in alto
        self.btn_start_video = ctk.CTkButton(
            left_col,
            text="Avvia video (Spazio)",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            width=220,
            height=44,
            command=self.start_video,
            state="disabled",
        )
        self.btn_start_video.pack(side="bottom", anchor="w")

        ctk.CTkLabel(
            left_col,
            text="Domanda",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", pady=(0, 4))
        self.lbl_question = ctk.CTkLabel(
            left_col,
            text="",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_LIGHT,
            wraplength=440,
            justify="left",
        )
        self.lbl_question.pack(anchor="w", pady=(4, 10))

        ctk.CTkLabel(
            left_col,
            text="Opzioni di risposta:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(4, 4))

        self.options_container = ctk.CTkFrame(left_col, fg_color="transparent")
        self.options_container.pack(anchor="w", fill="x", pady=(0, 10))
        self.options_container.grid_columnconfigure((0, 1), weight=1)

        self.card_opt_yes = ctk.CTkFrame(
            self.options_container,
            fg_color="#0e2a22",
            corner_radius=10,
            border_width=2,
            border_color=ACCENT,
        )
        self.card_opt_yes.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.lbl_opt_yes_title = ctk.CTkLabel(
            self.card_opt_yes,
            text="SI",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_opt_yes_title.pack(padx=10, pady=(6, 1))
        self.lbl_opt_yes_sub = ctk.CTkLabel(
            self.card_opt_yes,
            text="Shift sinistro",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        )
        self.lbl_opt_yes_sub.pack(padx=10, pady=(0, 6))

        self.card_opt_no = ctk.CTkFrame(
            self.options_container,
            fg_color="#301518",
            corner_radius=10,
            border_width=2,
            border_color=DANGER,
        )
        self.card_opt_no.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.lbl_opt_no_title = ctk.CTkLabel(
            self.card_opt_no,
            text="NO",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=DANGER,
        )
        self.lbl_opt_no_title.pack(padx=10, pady=(6, 1))
        self.lbl_opt_no_sub = ctk.CTkLabel(
            self.card_opt_no,
            text="Shift destro",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        )
        self.lbl_opt_no_sub.pack(padx=10, pady=(0, 6))

        self.lbl_ready_state = ctk.CTkLabel(
            left_col,
            text="Caricamento video...",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_ready_state.pack(anchor="w")
        self.lbl_instr2 = ctk.CTkLabel(
            left_col,
            text="",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_LIGHT,
            justify="left",
        )
        self.lbl_instr2.pack(anchor="w", pady=(8, 6))

        ctk.CTkLabel(
            right_col,
            text="Anteprima",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 6))
        self.preview_label = ctk.CTkLabel(
            right_col,
            text="Caricamento anteprima...",
            text_color=TEXT_MUTED,
            fg_color=CARD_BORDER,
            corner_radius=8,
            width=340,
            height=230,
        )
        self.preview_label.pack()
        self._preview_ctk_img = None

        # ── PLAYING ────────────────────────────────────────────
        self.playing_view = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.playing_view.grid(row=0, column=0, sticky="nsew")
        self.playing_view.grid_rowconfigure(1, weight=1)
        self.playing_view.grid_columnconfigure(0, weight=1)

        top_row = ctk.CTkFrame(self.playing_view, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 6))
        self.lbl_playing_question = ctk.CTkLabel(
            top_row,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_LIGHT,
            wraplength=900,
            justify="left",
        )
        self.lbl_playing_question.pack(side="left")
        self.lbl_progress2 = ctk.CTkLabel(
            top_row, text="", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED
        )
        self.lbl_progress2.pack(side="right")

        self.player = VideoPlayerFrame(
            self.playing_view,
            show_controls=False,
            on_end=self._on_video_end,
            on_frame_change=self._on_playing_frame_change,
        )

        self.player.grid(row=1, column=0, sticky="nsew", padx=16, pady=(6, 2))

        self.action_progress = ActionProgressBar(self.playing_view)
        self.action_progress.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 6))

        bottom = ctk.CTkFrame(self.playing_view, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 4))
        bottom.grid_columnconfigure((0, 1), weight=1)
        self.btn_yes = ctk.CTkButton(
            bottom,
            text="SI   (Shift Sx)",
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            height=64,
            command=lambda: self.register_answer("y"),
        )
        self.btn_yes.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.btn_no = ctk.CTkButton(
            bottom,
            text="NO   (Shift Dx)",
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=DANGER,
            hover_color=DANGER_HOVER,
            text_color="#160604",
            height=64,
            command=lambda: self.register_answer("n"),
        )
        self.btn_no.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.lbl_feedback = ctk.CTkLabel(
            self.playing_view,
            text="Risposta non ancora registrata",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
        )
        self.lbl_feedback.grid(row=4, column=0, pady=(0, 12))

        self._raise("intro")

    def _raise(self, which):
        self.state = which
        {
            "intro": self.intro_view,
            "question": self.question_view,
            "playing": self.playing_view,
        }[which].tkraise()

    def begin_session(self):
        st = self.app.state_
        self._advancing = False
        self._trial_finalized = False
        pname = (st.participant_name or "").strip()
        if pname:
            self.lbl_intro_title.configure(text=f"Pronto per iniziare, {pname}?")
        else:
            self.lbl_intro_title.configure(text="Pronto per iniziare?")

        n_videos = len(st.session_video_list)
        if st.answer_mode == "voice":
            lines = [
                f"Vedrai {n_videos} video in sequenza.",
                "Prima di ogni video vedrai una domanda specifica per quel video.",
                "Leggila con calma poi premi la barra SPAZIATRICE (o INVIO) per avviare il video.",
                "Rispondi ad alta voce dicendo la risposta nel momento che si ritiene più opportuno.",
                "Le opzioni possibili possono variare a ogni video.",
                "Puoi anche usare i tasti Shift sinistro/destro come alternativa.",
                "Solo la prima risposta rilevata viene registrata, le successive vengono ignorate.",
            ]
            self.app.set_status(
                "Preparazione del modello vocale (la prima volta può richiedere del tempo)...",
                "info",
            )
            threading.Thread(target=VoiceAnswerEngine._get_model, daemon=True).start()
        else:
            lines = [
                f"Vedrai {n_videos} video in sequenza.",
                "Prima di ogni video vedrai una domanda specifica per quel video.",
                "Leggila con calma poi premi la barra SPAZIATRICE (o INVIO) per avviare il video.",
                "Rispondi premendo Shift sinistro per la prima opzione e Shift destro per la seconda opzione.",
                "Le opzioni possibili possono variare a ogni video.",
                "Solo la prima risposta rilevata viene registrata, le successive vengono ignorate.",
            ]

        for w in self.intro_instr_frame.winfo_children():
            w.destroy()
        for l_text in lines:
            ctk.CTkLabel(
                self.intro_instr_frame,
                text=l_text,
                font=ctk.CTkFont(size=13),
                text_color=TEXT_LIGHT,
                justify="center",
            ).pack(pady=1)

        self._raise("intro")

    def _update_option_colors(self):
        y = (self.current_label_yes or "").strip().lower()
        n = (self.current_label_no or "").strip().lower()
        is_si_no = y in ("si", "sì", "si'", "yes") and n in ("no",)

        if is_si_no:
            c1_fg, c1_border, c1_txt = "#0e2a22", ACCENT, ACCENT
            c2_fg, c2_border, c2_txt = "#301518", DANGER, DANGER
            b1_fg, b1_hover, b1_txt = ACCENT, ACCENT_HOVER, "#06120b"
            b2_fg, b2_hover, b2_txt = DANGER, DANGER_HOVER, "#160604"
        else:
            BLUE_BG = "#0f2342"
            c1_fg, c1_border, c1_txt = BLUE_BG, BLUE, BLUE
            c2_fg, c2_border, c2_txt = BLUE_BG, BLUE, BLUE
            b1_fg, b1_hover, b1_txt = BLUE, BLUE_HOVER, "#ffffff"
            b2_fg, b2_hover, b2_txt = BLUE, BLUE_HOVER, "#ffffff"

        self.card_opt_yes.configure(fg_color=c1_fg, border_color=c1_border)
        self.lbl_opt_yes_title.configure(text_color=c1_txt)
        self.card_opt_no.configure(fg_color=c2_fg, border_color=c2_border)
        self.lbl_opt_no_title.configure(text_color=c2_txt)
        self.btn_yes.configure(fg_color=b1_fg, hover_color=b1_hover, text_color=b1_txt)
        self.btn_no.configure(fg_color=b2_fg, hover_color=b2_hover, text_color=b2_txt)

    def start_next_trial(self):
        st = self.app.state_
        if st.trial_index >= len(st.session_video_list):
            self.finish_session()
            return

        vpath = st.session_video_list[st.trial_index]
        cfg = self.app.config_mgr.get_cfg(vpath)
        self.current_question = cfg.get("question", DEFAULT_QUESTION)
        self.current_target_frame = cfg.get("target_frame")
        self.current_target_ms = cfg.get("target_ms")
        self.current_label_yes = cfg.get("label_yes", "SI")
        self.current_label_no = cfg.get("label_no", "NO")
        self.current_correct_answer = cfg.get("correct_answer")
        self.answer_given = False
        self.answer = ""
        self.answer_key = ""
        self.response_time_ms = None
        self.response_frame = None
        self._current_vpath = vpath
        self.voice_transcript = ""
        self._trial_finalized = False
        self._start_requested = False
        if getattr(self, "_freeze_timer_id", None) is not None:
            try:
                self.after_cancel(self._freeze_timer_id)
            except Exception:
                pass
            self._freeze_timer_id = None

        self.lbl_progress.configure(text=trial_counter_str(st))
        self.lbl_progress2.configure(text=trial_counter_str(st))
        self.trial_progress.set(st.trial_index / max(len(st.session_video_list), 1))
        self.lbl_question.configure(text=self.current_question)
        self.lbl_playing_question.configure(text=self.current_question)
        self.lbl_opt_yes_title.configure(text=self.current_label_yes)
        self.lbl_opt_no_title.configure(text=self.current_label_no)
        self._update_option_colors()
        self.lbl_ready_state.configure(
            text="Caricamento video...", text_color=TEXT_MUTED
        )
        self.lbl_feedback.configure(
            text="Risposta non ancora registrata", text_color=TEXT_MUTED
        )


        if st.answer_mode == "voice":
            self.lbl_opt_yes_sub.configure(text="Pronuncia a voce (o Shift Sx)")
            self.lbl_opt_no_sub.configure(text="Pronuncia a voce (o Shift Dx)")
            self.lbl_instr2.configure(
                text=(
                    "Leggi la domanda, poi premi SPAZIO (o clicca 'Avvia video').\n"
                    f"Rispondi A VOCE dicendo '{self.current_label_yes}' oppure "
                    f"'{self.current_label_no}' durante la riproduzione."
                )
            )
        else:
            self.lbl_opt_yes_sub.configure(text="Tasto Shift sinistro")
            self.lbl_opt_no_sub.configure(text="Tasto Shift destro")
            self.lbl_instr2.configure(
                text=(
                    "Leggi la domanda, poi premi SPAZIO (o clicca 'Avvia video').\n"
                    "Usa i tasti Shift durante la riproduzione del video per rispondere."
                )
            )
        self.btn_yes.configure(text=f"{self.current_label_yes}   (Shift Sx)")
        self.btn_no.configure(text=f"{self.current_label_no}   (Shift Dx)")
        self.btn_start_video.configure(state="disabled")
        _ctk_clear_image(self.preview_label)
        self._preview_ctk_img = None
        self.preview_label.configure(text="Caricamento anteprima...")
        self._raise("question")

        self._video_loaded = False
        self.player.load_video(
            vpath,
            expected_fps=cfg.get("fps"),
            on_loaded=self._on_trial_video_ready,
            status_cb=lambda m: self.app.set_status(m),
        )
        watchdog_trial = st.trial_index
        self.after(
            6_000,  # 6 secondi: timeout watchdog caricamento video
            lambda: self._load_watchdog(watchdog_trial),
        )

    def _load_watchdog(self, trial_idx):
        if (
            self.state == "question"
            and self.app.state_.trial_index == trial_idx
            and not self._trial_finalized
            and not self._video_loaded
        ):
            self.app.set_status(
                "Caricamento video non riuscito: passo al successivo.", "warn"
            )
            self._finalize_trial(0)


    def _on_playing_frame_change(self, idx):
        if hasattr(self, "action_progress"):
            self.action_progress.update_progress(idx)

    def _on_trial_video_ready(self, ok, n_frames, fps):
        if not ok:
            self.app.set_status(
                "Errore caricamento video: passo al successivo.", "error"
            )
            self._finalize_trial(0)
            return
        self._video_loaded = True
        self.lbl_ready_state.configure(
            text="Video pronto. Premi SPAZIO per avviare.", text_color=ACCENT
        )
        self.btn_start_video.configure(state="normal")
        self.action_progress.set_config(n_frames, self.current_target_frame, fps)
        self._update_preview_frame()
        if getattr(self, "_start_requested", False):
            self.start_video()

    def _update_preview_frame(self):
        """Legge il primo frame direttamente con OpenCV per il thumbnail statico."""
        vpath = self._current_vpath
        if not vpath:
            _ctk_clear_image(self.preview_label)
            self._preview_ctk_img = None
            self.preview_label.configure(text="Anteprima non disponibile")
            return

        def _read_first_frame():
            try:
                cap = cv2.VideoCapture(str(vpath))
                ret, frame_bgr = cap.read()
                cap.release()
                if not ret or frame_bgr is None:
                    self.after(0, lambda: (
                        _ctk_clear_image(self.preview_label) or
                        self.preview_label.configure(text="Anteprima non disponibile")
                    ))
                    return
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(rgb)
                max_w, max_h = 320, 210
                iw, ih = pil_img.size
                scale = min(max_w / iw, max_h / ih)
                nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                pil_resized = pil_img.resize((nw, nh), _PILImage.BILINEAR)
                self.after(0, lambda: self._set_preview_image(pil_resized, nw, nh))
            except Exception:
                pass

        import threading as _threading
        _threading.Thread(target=_read_first_frame, daemon=True).start()

    def _set_preview_image(self, pil_resized, nw, nh):
        if self._preview_ctk_img is None:
            self._preview_ctk_img = ctk.CTkImage(
                light_image=pil_resized, dark_image=pil_resized, size=(nw, nh)
            )
            self.preview_label.configure(image=self._preview_ctk_img, text="")
        else:
            try:
                self._preview_ctk_img.configure(
                    light_image=pil_resized, dark_image=pil_resized, size=(nw, nh)
                )
            except ValueError:
                self._preview_ctk_img = ctk.CTkImage(
                    light_image=pil_resized, dark_image=pil_resized, size=(nw, nh)
                )
                self.preview_label.configure(image=self._preview_ctk_img, text="")

    def start_video(self):
        if self.state != "question":
            return
        if not self._video_loaded:
            self._start_requested = True
            self.lbl_ready_state.configure(
                text="Avvio in corso non appena pronto...", text_color=ACCENT
            )
            return
        self._start_requested = False
        self.btn_start_video.configure(state="disabled")
        self._raise("playing")
        self.playing_view.update_idletasks()
        self.player.update_display_size()
        self.lbl_feedback.configure(
            text="Risposta non ancora registrata", text_color=TEXT_MUTED
        )
        self.player.play(restart=True)
        if self.app.state_.answer_mode == "voice" and VOICE_AVAILABLE:
            self._voice_engine = VoiceAnswerEngine()
            try:
                self._voice_engine.start()
                self.lbl_feedback.configure(
                    text="In ascolto... rispondi a voce (o Shift/bottoni)",
                    text_color=TEXT_MUTED,
                )
            except Exception as e:  # noqa: BLE001
                self._voice_engine = None
                self.app.set_status(f"Microfono non disponibile: {e}", "error")
        else:
            self._voice_engine = None
        self.app.focus_set()

    def register_answer(self, key_char):
        if self.state != "playing" or self.answer_given or self._trial_finalized:
            return
        elapsed_ms = self.player.elapsed_ms()
        if elapsed_ms is None:
            return
        fps = self.player.fps or 25.0
        resp_frame = int(elapsed_ms / 1000.0 * fps)

        self.answer_given = True
        self.answer = "yes" if key_char == "y" else "no"
        self.answer_key = key_char
        self.response_time_ms = elapsed_ms
        self.response_frame = resp_frame

        lbl = self.current_label_yes if self.answer == "yes" else self.current_label_no
        self.lbl_feedback.configure(
            text=f"Risposta {lbl} registrata a {elapsed_ms} ms", text_color=ACCENT
        )
        self.app.set_status(f"Risposta {lbl} registrata a {elapsed_ms} ms", "success")

    def _on_video_end(self):
        # Durata calcolata da frames e fps
        n_frames = len(self.player.frames) if self.player.frames else 0
        fps_val = self.player.fps or 25.0
        vdur_ms = getattr(self.player, "_duration_ms", None) or int(n_frames / max(fps_val, 1.0) * 1000)

        if getattr(self, "_freeze_timer_id", None) is not None:
            try:
                self.after_cancel(self._freeze_timer_id)
            except Exception:
                pass

        if self.answer_given:
            # Se la risposta è già stata registrata, avanza rapidamente (400ms)
            self._freeze_timer_id = self.after(
                400, lambda: self._on_freeze_timeout(vdur_ms)
            )
        else:
            self.lbl_feedback.configure(
                text="Video terminato. Può ancora rispondere per 2 secondi...",
                text_color=WARN,
            )
            self.app.set_status("Video terminato: 2 secondi per rispondere...", "info")
            self._freeze_timer_id = self.after(
                2000, lambda: self._on_freeze_timeout(vdur_ms)
            )


    def _on_freeze_timeout(self, vdur_ms):
        self._freeze_timer_id = None
        if self._voice_engine is not None and not self.answer_given:
            self._process_voice_answer(vdur_ms)
        else:
            if self._voice_engine is not None:
                self._voice_engine.stop()
                self._voice_engine = None
            self._finalize_trial(vdur_ms)

    def _process_voice_answer(self, vdur_ms):
        engine = self._voice_engine
        audio = engine.stop()
        self._voice_engine = None
        self.audio_file = str(self._save_voice_audio(audio))
        self.lbl_feedback.configure(
            text="Elaborazione risposta vocale in corso...", text_color=TEXT_MUTED
        )
        self.app.set_status("Elaborazione risposta vocale in corso...", "info")
        threading.Thread(
            target=self._voice_worker, args=(engine, audio), daemon=True
        ).start()
        self._poll_voice_result(vdur_ms, time.perf_counter())

    def _save_voice_audio(self, audio):
        AUDIO_RESPONSES_FOLDER.mkdir(exist_ok=True)
        st = self.app.state_
        session_idx = getattr(st, "session_index", 1)
        participant_name = st.participant_name.strip() or "Partecipante"
        safe_name = re.sub(r'[\\/*?:"<>|]', "", participant_name).strip() or "Partecipante"
        subfolder_name = f"{session_idx} - {safe_name}"
        participant_folder = AUDIO_RESPONSES_FOLDER / subfolder_name
        participant_folder.mkdir(parents=True, exist_ok=True)

        vpath = st.session_video_list[st.trial_index]
        orig_video_name = vpath.name[1:] if vpath.name.startswith("_") else vpath.name
        orig_stem = pathlib.Path(orig_video_name).stem
        trial_num = st.trial_index + 1
        fn = participant_folder / f"{trial_num}_{orig_stem}.wav"
        try:
            VoiceAnswerEngine.save_wav(audio, fn)
        except Exception:
            return ""
        return fn

    def _voice_worker(self, engine, audio):
        try:
            text, words = engine.transcribe(audio, language="it")
            answer, word_onset = match_voice_answer(
                text, words, self.current_label_yes, self.current_label_no
            )
            w_end = None
            if words and word_onset is not None:
                for w in words:
                    if w.get("start") == word_onset:
                        w_end = w.get("end")
                        break
            onset_sec = engine.detect_onset_sec(audio, word_start=word_onset, word_end=w_end)
            onset = onset_sec if onset_sec is not None else word_onset
            self._voice_queue.put(("ok", answer, onset, text))
        except Exception as e:  # noqa: BLE001
            self._voice_queue.put(("error", str(e)))

    VOICE_TIMEOUT_SEC = 5.0

    def _poll_voice_result(self, vdur_ms, start_perf):
        try:
            item = self._voice_queue.get_nowait()
        except queue.Empty:
            if time.perf_counter() - start_perf > self.VOICE_TIMEOUT_SEC:
                self.app.set_status(
                    "Tempo risposta vocale scaduto: trial registrato senza risposta.",
                    "warn",
                )
                self._finalize_trial(vdur_ms)
                return
            self.after(50, lambda: self._poll_voice_result(vdur_ms, start_perf))
            return


        if item[0] == "error":
            self.app.set_status(f"Errore riconoscimento vocale: {item[1]}", "error")
            self._finalize_trial(vdur_ms)
            return

        _, answer, onset, text = item
        self.voice_transcript = text
        if answer is not None and onset is not None:
            fps = self.player.fps or 25.0
            self.answer_given = True
            self.answer = answer
            self.answer_key = "voice"
            self.response_time_ms = int(onset * 1000)
            self.response_frame = int(onset * fps)
            lbl = self.current_label_yes if answer == "yes" else self.current_label_no
            self.app.set_status(
                f"Risposta vocale riconosciuta: {lbl}  (detto: \u00ab{text}\u00bb)  "
                f"a {self.response_time_ms} ms",
                "success",
            )
        else:
            self.app.set_status(
                f"Nessuna risposta vocale riconosciuta (trascritto: \u00ab{text}\u00bb)",
                "warn",
            )
        self._finalize_trial(vdur_ms)

    def _finalize_trial(self, vdur_ms):
        if self._trial_finalized:
            return
        self._trial_finalized = True

        st = self.app.state_
        vpath = self._current_vpath
        answered = self.answer_given
        answer = self.answer if answered else "no_response"
        answer_key = self.answer_key if answered else ""
        rtms = self.response_time_ms if answered else None
        rframe = self.response_frame if answered else None

        delta_ms = ""
        delta_frames = ""
        if answered and self.current_target_ms is not None and rtms is not None:
            try:
                delta_ms = str(int(rtms) - int(self.current_target_ms))
            except (ValueError, TypeError):
                pass
        if answered and self.current_target_frame is not None and rframe is not None:
            try:
                delta_frames = str(int(rframe) - int(self.current_target_frame))
            except (ValueError, TypeError):
                pass

        is_correct = ""
        if answered and self.current_correct_answer in ("yes", "no"):
            is_correct = str(answer == self.current_correct_answer)

        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "participant_name": st.participant_name,
            "age_group": getattr(st, "participant_age_group", ""),
            "gender": getattr(st, "participant_gender", ""),
            "world_cup_2026": getattr(st, "participant_world_cup_2026", ""),
            "football_frequency": getattr(st, "participant_football_frequency", ""),
            "trial_index": st.trial_index + 1,
            "video_filename": vpath.name,
            "video_path": str(vpath),
            "is_augmented": str(is_augmented(vpath)),
            "question": self.current_question,
            "label_yes": self.current_label_yes,
            "label_no": self.current_label_no,
            "answer": answer,
            "answer_key": answer_key,
            "answered": str(answered),
            "response_time_ms": str(rtms) if rtms is not None else "",
            "response_frame": str(rframe) if rframe is not None else "",
            "video_duration_ms": str(vdur_ms),
            "session_seed": str(st.session_seed),
            "target_frame": (
                str(self.current_target_frame)
                if self.current_target_frame is not None
                else ""
            ),
            "target_ms": (
                str(self.current_target_ms)
                if self.current_target_ms is not None
                else ""
            ),
            "delta_ms": delta_ms,
            "delta_frames": delta_frames,
            "correct_answer": self.current_correct_answer or "",
            "is_correct": is_correct,
            "answer_mode": st.answer_mode,
            "audio_file": self.audio_file,
            "voice_transcript": self.voice_transcript,
        }
        st.trial_results.append(row)
        self.app.results_mgr.append_trial(row)
        st.trial_index += 1

        if self._advancing:
            return
        self._advancing = True
        self.after(0, self._advance_to_next_trial)

    def _advance_to_next_trial(self):
        self._advancing = False
        self.player.release()
        self.start_next_trial()

    def finish_session(self):
        self.app.state_.test_active = False
        self.app.show_page("survey", force=True)

    def handle_shortcut(self, keysym):
        if self.state == "intro" and keysym in ("Return", "space", "KP_Enter", " "):
            self.start_next_trial()
        elif self.state == "question" and keysym in ("Return", "space", "KP_Enter", " "):
            self.start_video()
        elif self.state == "playing":
            if keysym == "Shift_L":
                self.register_answer("y")
            elif keysym == "Shift_R":
                self.register_answer("n")
