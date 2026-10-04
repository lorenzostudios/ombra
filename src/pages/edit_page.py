"""
MIRA - Pagina di Configurazione Video (EditPage)
Consente di ispezionare ciascun video del dataset, impostare la domanda del test,
definire le opzioni di risposta e contrassegnare il frame target (momento dell'anomalia/evento).
"""

from tkinter import messagebox
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
)
from ..utils import discover_videos, video_duration_ms, counterpart_video_path
from ..components.video_player import VideoPlayerFrame

class EditPage(ctk.CTkFrame):
    """
    Interfaccia per la revisione e l'annotazione dei video.
    Permette di impostare interattivamente tramite player il frame target esatto,
    salvando i metadati associati in 'video_config.json'.
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app
        self.dirty = False
        self.current_video = None
        self.current_target_frame = None
        self.current_correct_answer = None  # None | "yes" | "no"
        self.video_list_cache = []
        self.card_widgets = {}
        self._build()

    def _build(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ── Zona A: lista video ──────────────────────────────
        left = ctk.CTkFrame(
            self,
            width=300,
            fg_color=PANEL_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        left.grid(row=0, column=0, sticky="ns", padx=(20, 10), pady=20)
        left.grid_propagate(False)
        ctk.CTkLabel(
            left,
            text="Video",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=14, pady=(14, 4))
        self.scroll_list = ctk.CTkScrollableFrame(
            left, fg_color="transparent", width=270
        )
        self.scroll_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ── Zona B: preview ───────────────────────────────────
        mid = ctk.CTkFrame(self, fg_color=BG_DARK)
        mid.grid(row=0, column=1, sticky="nsew", padx=10, pady=20)
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1)

        self.player = VideoPlayerFrame(
            mid, show_controls=True, on_frame_change=self._on_frame_change
        )
        self.player.grid(row=0, column=0, sticky="nsew")

        nav_row = ctk.CTkFrame(mid, fg_color="transparent")
        nav_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ctk.CTkButton(
            nav_row,
            text="< Video precedente",
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.prev_video,
        ).pack(side="left")
        ctk.CTkButton(
            nav_row,
            text="Video successivo >",
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.next_video,
        ).pack(side="right")

        # ── Zona C: inspector ─────────────────────────────────
        right = ctk.CTkScrollableFrame(
            self,
            width=320,
            fg_color=PANEL_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        right.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=20)

        ctk.CTkLabel(
            right,
            text="Inspector",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(14, 6))
        self.lbl_video_name = ctk.CTkLabel(
            right,
            text="Nessun video selezionato",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_LIGHT,
            wraplength=280,
            justify="left",
        )
        self.lbl_video_name.pack(anchor="w", padx=16, pady=(0, 4))
        self.badge_status = ctk.CTkLabel(
            right,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=CARD_BG,
            text_color=TEXT_MUTED,
            corner_radius=6,
        )
        self.badge_status.pack(anchor="w", padx=16, pady=(0, 4), ipadx=6, ipady=2)
        self.lbl_counterpart = ctk.CTkLabel(
            right,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            wraplength=280,
            justify="left",
        )
        self.lbl_counterpart.pack(anchor="w", padx=16, pady=(0, 12))

        ctk.CTkLabel(
            right,
            text="Domanda",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(8, 2))
        self.txt_question = ctk.CTkTextbox(right, height=90)
        self.txt_question.pack(fill="x", padx=16, pady=(2, 12))
        self.txt_question.bind("<KeyRelease>", self._on_question_change)

        ctk.CTkLabel(
            right,
            text="Frame risposta (target)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(6, 2))
        self.lbl_target = ctk.CTkLabel(
            right, text="Non impostato", font=ctk.CTkFont(size=13), text_color=WARN
        )
        self.lbl_target.pack(anchor="w", padx=16, pady=(2, 8))

        row_target = ctk.CTkFrame(right, fg_color="transparent")
        row_target.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkButton(
            row_target,
            text="Segna frame corrente come target (T)",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            command=self.mark_target,
        ).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            row_target,
            text="Rimuovi target",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.clear_target,
        ).pack(fill="x")

        sep_labels = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep_labels.pack(fill="x", padx=16, pady=10)

        ctk.CTkLabel(
            right,
            text="Etichette risposta",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(4, 2))
        label_row = ctk.CTkFrame(right, fg_color="transparent")
        label_row.pack(fill="x", padx=16, pady=(4, 4))
        label_row.grid_columnconfigure((0, 1), weight=1)
        self.entry_label_yes = ctk.CTkEntry(label_row, placeholder_text="SI")
        self.entry_label_yes.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.entry_label_yes.bind("<KeyRelease>", self._on_label_change)
        self.entry_label_no = ctk.CTkEntry(label_row, placeholder_text="NO")
        self.entry_label_no.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.entry_label_no.bind("<KeyRelease>", self._on_label_change)

        ctk.CTkLabel(
            right,
            text="Risposta corretta (opzionale)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(4, 2))
        correct_row = ctk.CTkFrame(right, fg_color="transparent")
        correct_row.pack(fill="x", padx=16, pady=(4, 4))
        correct_row.grid_columnconfigure((0, 1), weight=1)
        self.btn_correct_yes = ctk.CTkButton(
            correct_row,
            text="SI",
            fg_color=CARD_BG,
            text_color=TEXT_LIGHT,
            command=lambda: self._set_correct_answer("yes"),
        )
        self.btn_correct_yes.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_correct_no = ctk.CTkButton(
            correct_row,
            text="NO",
            fg_color=CARD_BG,
            text_color=TEXT_LIGHT,
            command=lambda: self._set_correct_answer("no"),
        )
        self.btn_correct_no.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkLabel(
            right,
            text="Clicca di nuovo per rimuovere la risposta corretta.",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            wraplength=280,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(2, 10))

        sep = ctk.CTkFrame(right, height=1, fg_color=CARD_BORDER)
        sep.pack(fill="x", padx=16, pady=10)

        ctk.CTkLabel(
            right,
            text="Parametri video",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(4, 2))
        self.lbl_params = ctk.CTkLabel(
            right,
            text="-",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_LIGHT,
            justify="left",
        )
        self.lbl_params.pack(anchor="w", padx=16, pady=(2, 14))

        self.lbl_unsaved = ctk.CTkLabel(
            right, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color=WARN
        )
        self.lbl_unsaved.pack(anchor="w", padx=16, pady=(0, 4))

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(6, 16))
        ctk.CTkButton(
            btn_row,
            text="Salva configurazione",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            command=self.save_current,
        ).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            btn_row,
            text="Ripristina",
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=self.discard_changes,
        ).pack(fill="x")

    def on_show(self):
        self.refresh_list()
        if self.current_video is not None and not self.player.frames:
            self._load_video_now(self.current_video)

    def refresh_list(self):
        for w in self.scroll_list.winfo_children():
            w.destroy()
        self.card_widgets = {}
        vids = discover_videos(self.app.state_.video_folder)
        self.video_list_cache = vids
        if not vids:
            ctk.CTkLabel(
                self.scroll_list,
                text=f"Nessun video in\n'{self.app.state_.video_folder}'",
                text_color=TEXT_MUTED,
                justify="left",
            ).pack(padx=8, pady=8)
            self.current_video = None
            return
        for vp in vids:
            self._build_card(vp)
        if self.current_video is None or self.current_video not in vids:
            self.load_video_into_editor(vids[0])
        else:
            self._highlight_selected()

    def _build_card(self, vp):
        rc = self.app.config_mgr.get_raw_resolved(vp)
        question = (rc.get("question") or "").strip()
        tf = rc.get("target_frame")
        tms = rc.get("target_ms")
        configured = bool(question) and tf is not None
        incomplete = (bool(question) or tf is not None) and not configured

        card = ctk.CTkFrame(
            self.scroll_list,
            fg_color=CARD_BG,
            corner_radius=10,
            border_width=2,
            border_color=CARD_BG,
        )
        card.pack(fill="x", pady=4, padx=2)

        name_lbl = ctk.CTkLabel(
            card,
            text=vp.name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LIGHT,
            anchor="w",
        )
        name_lbl.pack(fill="x", padx=10, pady=(8, 0))

        if configured:
            badge_text, badge_col = "CONFIGURATO", ACCENT
        elif incomplete:
            badge_text, badge_col = "INCOMPLETO", WARN
        else:
            badge_text, badge_col = "NON CONFIGURATO", TEXT_MUTED
        badge = ctk.CTkLabel(
            card,
            text=badge_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=badge_col,
            anchor="w",
        )
        badge.pack(fill="x", padx=10)

        preview = question if question else "(nessuna domanda)"
        if len(preview) > 42:
            preview = preview[:39] + "..."
        q_lbl = ctk.CTkLabel(
            card,
            text=preview,
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        q_lbl.pack(fill="x", padx=10)

        if tf is not None:
            target_txt = f"target: f{tf}" + (
                f"  ({tms / 1000:.2f}s)" if tms is not None else ""
            )
        else:
            target_txt = "target: -"
        t_lbl = ctk.CTkLabel(
            card,
            text=target_txt,
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        t_lbl.pack(fill="x", padx=10, pady=(0, 8))

        for w in (card, name_lbl, badge, q_lbl, t_lbl):
            w.bind("<Button-1>", lambda e, p=vp: self.load_video_into_editor(p))
        self.card_widgets[vp.name] = card

    def _highlight_selected(self):
        for name, card in self.card_widgets.items():
            sel = self.current_video is not None and name == self.current_video.name
            card.configure(border_color=ACCENT if sel else CARD_BG)

    def load_video_into_editor(self, vpath):
        if self.dirty:
            if not self.confirm_leave():
                return
        self._load_video_now(vpath)

    def _load_video_now(self, vpath):
        self.current_video = vpath
        self._highlight_selected()
        self.lbl_video_name.configure(text=vpath.name)
        rc = self.app.config_mgr.get_cfg(vpath)
        self.txt_question.delete("1.0", "end")
        self.txt_question.insert("1.0", rc.get("question", ""))
        self.current_target_frame = rc.get("target_frame")
        self.current_correct_answer = rc.get("correct_answer")
        self.entry_label_yes.delete(0, "end")
        self.entry_label_yes.insert(0, rc.get("label_yes", "SI"))
        self.entry_label_no.delete(0, "end")
        self.entry_label_no.insert(0, rc.get("label_no", "NO"))
        self.btn_correct_yes.configure(text=rc.get("label_yes", "SI"))
        self.btn_correct_no.configure(text=rc.get("label_no", "NO"))
        self._update_correct_buttons()
        self._update_target_label()
        self._update_counterpart_label()
        self.dirty = False
        self._update_unsaved_label()
        self.lbl_params.configure(text="Caricamento...")
        self.app.set_status(f"Caricamento {vpath.name}...")
        self.player.load_video(
            vpath,
            expected_fps=rc.get("fps"),
            on_loaded=self._on_video_loaded,
            status_cb=lambda m: self.app.set_status(m),
        )

    def _update_counterpart_label(self):
        if self.current_video is None:
            self.lbl_counterpart.configure(text="")
            return
        counterpart = counterpart_video_path(self.current_video)
        if counterpart.exists():
            kind = (
                "originale" if self.current_video.name.startswith("_") else "alterato"
            )
            self.lbl_counterpart.configure(
                text=f"È presente anche il video {kind} '{counterpart.name}'.\n"
                "Le modifiche vengono sincronizzate in entrambe le versioni.",
            )
        else:
            self.lbl_counterpart.configure(text="")

    def _on_video_loaded(self, ok, n_frames, fps):
        if not ok:
            self.app.set_status(
                f"Errore: impossibile caricare {self.current_video}", "error"
            )
            messagebox.showerror(
                APP_TITLE, f"Impossibile caricare il video:\n{self.current_video}"
            )
            self.lbl_params.configure(text="-")
            return
        dur_ms = video_duration_ms([None] * n_frames, fps)
        self.lbl_params.configure(
            text=f"FPS: {fps:.2f}\nFrame: {n_frames}\nDurata: {dur_ms / 1000:.2f}s"
        )
        self._update_target_label()
        self._update_status_badge()

    def _on_frame_change(self, idx):
        pass

    def _on_question_change(self, event=None):
        self.dirty = True
        self._update_unsaved_label()
        self._update_status_badge()

    def _on_label_change(self, event=None):
        label_yes = self.entry_label_yes.get().strip() or "SI"
        label_no = self.entry_label_no.get().strip() or "NO"
        self.btn_correct_yes.configure(text=label_yes)
        self.btn_correct_no.configure(text=label_no)
        self.dirty = True
        self._update_unsaved_label()

    def _set_correct_answer(self, which):
        self.current_correct_answer = (
            None if self.current_correct_answer == which else which
        )
        self.dirty = True
        self._update_unsaved_label()
        self._update_correct_buttons()

    def _update_correct_buttons(self):
        sel_yes = self.current_correct_answer == "yes"
        sel_no = self.current_correct_answer == "no"
        self.btn_correct_yes.configure(
            fg_color=ACCENT if sel_yes else CARD_BG,
            text_color="#06120b" if sel_yes else TEXT_LIGHT,
            hover_color=ACCENT_HOVER if sel_yes else "#2a2a34",
        )
        self.btn_correct_no.configure(
            fg_color=ACCENT if sel_no else CARD_BG,
            text_color="#06120b" if sel_no else TEXT_LIGHT,
            hover_color=ACCENT_HOVER if sel_no else "#2a2a34",
        )

    def _update_unsaved_label(self):
        self.lbl_unsaved.configure(text="Modifiche non salvate" if self.dirty else "")

    def _update_target_label(self):
        if self.current_target_frame is not None:
            fps = self.player.fps or 25.0
            tms = int(self.current_target_frame / max(fps, 1) * 1000)
            self.lbl_target.configure(
                text=f"Frame {self.current_target_frame}  ({tms} ms)", text_color=ACCENT
            )
        else:
            self.lbl_target.configure(text="Non impostato", text_color=WARN)
        self._update_status_badge()

    def _update_status_badge(self):
        q = self.txt_question.get("1.0", "end").strip()
        tf = self.current_target_frame
        if q and tf is not None:
            self.badge_status.configure(text="CONFIGURATO", text_color=ACCENT)
        else:
            missing = []
            if not q:
                missing.append("domanda")
            if tf is None:
                missing.append("target")
            self.badge_status.configure(
                text=f"INCOMPLETO: manca {', '.join(missing)}", text_color=WARN
            )

    def mark_target(self):
        if not self.player.frames:
            self.app.set_status("Carica prima un video.", "warn")
            return
        self.current_target_frame = self.player.frame_idx
        self.dirty = True
        self._update_target_label()
        self._update_unsaved_label()
        self.app.set_status(
            f"Target impostato al frame {self.current_target_frame}", "success"
        )

    def clear_target(self):
        self.current_target_frame = None
        self.dirty = True
        self._update_target_label()
        self._update_unsaved_label()
        self.app.set_status("Target rimosso", "info")

    def save_current(self):
        if self.current_video is None:
            return
        question = self.txt_question.get("1.0", "end").strip()
        if not question:
            messagebox.showwarning(
                APP_TITLE,
                "La domanda e' vuota: il video verra' salvato come incompleto.",
            )
        fps = self.player.fps or 25.0
        n_frames = len(self.player.frames)
        label_yes = self.entry_label_yes.get().strip() or "SI"
        label_no = self.entry_label_no.get().strip() or "NO"
        self.app.config_mgr.save_video_cfg(
            self.current_video,
            question,
            self.current_target_frame,
            fps,
            n_frames,
            correct_answer=self.current_correct_answer,
            label_yes=label_yes,
            label_no=label_no,
        )
        synced_name = self._sync_counterpart_video(
            question, fps, n_frames, label_yes, label_no
        )
        self.dirty = False
        self._update_unsaved_label()
        self._update_status_badge()
        if synced_name:
            self.app.set_status(
                f"Configurazione salvata: {self.current_video.name} "
                f"(sincronizzata anche su {synced_name})",
                "success",
            )
        else:
            self.app.set_status(
                f"Configurazione salvata: {self.current_video.name}", "success"
            )
        self.refresh_list()

    def _sync_counterpart_video(self, question, fps, n_frames, label_yes, label_no):
        if self.current_video is None:
            return None
        counterpart = counterpart_video_path(self.current_video)
        if not counterpart.exists():
            return None
        self.app.config_mgr.save_video_cfg(
            counterpart,
            question,
            self.current_target_frame,
            fps,
            n_frames,
            correct_answer=self.current_correct_answer,
            label_yes=label_yes,
            label_no=label_no,
        )
        return counterpart.name

    def discard_changes(self):
        if self.current_video is None:
            return
        rc = self.app.config_mgr.get_cfg(self.current_video)
        self.txt_question.delete("1.0", "end")
        self.txt_question.insert("1.0", rc.get("question", ""))
        self.current_target_frame = rc.get("target_frame")
        self.current_correct_answer = rc.get("correct_answer")
        self.entry_label_yes.delete(0, "end")
        self.entry_label_yes.insert(0, rc.get("label_yes", "SI"))
        self.entry_label_no.delete(0, "end")
        self.entry_label_no.insert(0, rc.get("label_no", "NO"))
        self.btn_correct_yes.configure(text=rc.get("label_yes", "SI"))
        self.btn_correct_no.configure(text=rc.get("label_no", "NO"))
        self._update_correct_buttons()
        self._update_target_label()
        self.dirty = False
        self._update_unsaved_label()
        self.app.set_status("Modifiche ripristinate", "info")

    def confirm_leave(self) -> bool:
        res = messagebox.askyesnocancel(
            APP_TITLE,
            "Ci sono modifiche non salvate a questo video.\n\nVuoi salvarle prima di continuare?",
        )
        if res is None:
            return False
        if res is True:
            self.save_current()
        else:
            self.dirty = False
            self._update_unsaved_label()
        return True

    def prev_video(self):
        self._navigate(-1)

    def next_video(self):
        self._navigate(1)

    def _navigate(self, delta):
        if not self.video_list_cache or self.current_video is None:
            return
        try:
            idx = self.video_list_cache.index(self.current_video)
        except ValueError:
            idx = 0
        new_idx = max(0, min(len(self.video_list_cache) - 1, idx + delta))
        if new_idx == idx:
            return
        self.load_video_into_editor(self.video_list_cache[new_idx])

    def handle_shortcut(self, keysym):
        if keysym == "space":
            self.player.toggle_play()
        elif keysym == "Left":
            self.player.prev_frame()
        elif keysym == "Right":
            self.player.next_frame()
        elif keysym.lower() == "t":
            self.mark_target()
