"""
MIRA - Analisi Audio e Tempi di Reazione (AudioAnalysisPage)
Interfaccia per la visualizzazione dell'oscillogramma/waveform delle risposte vocali,
confronto tra target temporale nel video ed onset vocale rilevato con Whisper/RMS,
e riproduzione con cursore playhead in tempo reale.
"""

import time
import re
import pathlib
import unicodedata
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import numpy as np

try:
    import soundfile as sf
    import sounddevice as sd
    _AUDIO_PLAYBACK_OK = True
except Exception:
    _AUDIO_PLAYBACK_OK = False

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
    WARN,
    TEXT_LIGHT,
    TEXT_MUTED,
    AUDIO_RESPONSES_FOLDER,
)
from ..utils import open_path

class AudioAnalysisPage(ctk.CTkFrame):
    """
    Pagina per l'analisi visiva e la riproduzione dei file audio registrati.
    Visualizza i tentativi raggruppati per sessione/partecipante, l'onda audio
    estratta dal file .wav e due marker sulla timeline:
      - Marker 1 (Rosso/Ambra): azionamento/target effettiva nel video (target_ms)
      - Marker 2 (Verde): onset/risposta vocale del partecipante (response_time_ms)
      - Cursore Playhead dinamico per il seek e la riproduzione in tempo reale.
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app

        # Stato selezione corrente
        self.selected_item = None
        self.audio_data = None
        self.sample_rate = 16000
        self.duration_sec = 0.0

        # Stato riproduzione audio
        self.is_playing = False
        self.playback_time = 0.0
        self._play_start_perf = 0.0
        self._play_start_offset = 0.0
        self._stream = None
        self._update_timer_id = None

        self._build_ui()
        self.refresh()

    # ═════════════════════════════════════════════════════════════
    #  INTERFACCIA GRAFICA
    # ═════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Header Banner ──────────────────────────────────────
        header = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", padx=16, pady=12)
        ctk.CTkLabel(
            title_box,
            text="Analisi risposte vocali",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="Visualizza le risposte vocali su forma d'onda e confronta l'istante di risposta con l'azione effettiva.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        btn_box = ctk.CTkFrame(header, fg_color="transparent")
        btn_box.grid(row=0, column=1, sticky="e", padx=16)
        ctk.CTkButton(
            btn_box,
            text="Aggiorna elenco",
            width=130,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.refresh,
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            btn_box,
            text="Apri cartella audio",
            width=140,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self._open_audio_folder,
        ).pack(side="right", padx=4)

        # Corpo principale a 2 colonne
        body = ctk.CTkFrame(self, fg_color=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0)  # Colonna sinistra: Elenco tentativi
        body.grid_columnconfigure(1, weight=1)  # Colonna destra: Player & Waveform

        self._build_left_sidebar(body)
        self._build_right_player(body)

    def _build_left_sidebar(self, parent):
        left = ctk.CTkFrame(parent, width=320, fg_color=PANEL_BG, corner_radius=12)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left.grid_propagate(False)

        ctk.CTkLabel(
            left,
            text="Tentativi e Registrazioni",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(16, 6))

        # Barra di ricerca
        self.txt_search = ctk.CTkEntry(
            left,
            placeholder_text="Cerca partecipante o video...",
            font=ctk.CTkFont(size=12),
        )
        self.txt_search.pack(fill="x", padx=16, pady=(0, 10))
        self.txt_search.bind("<KeyRelease>", lambda e: self._populate_list())

        # Scrollable Frame per la lista dei tentativi
        self.list_container = ctk.CTkScrollableFrame(
            left, fg_color="transparent", corner_radius=0
        )
        self.list_container.pack(fill="both", expand=True, padx=8, pady=(0, 12))

    def _build_right_player(self, parent):
        right = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Card Info Trial Selezionato
        self.card_info = ctk.CTkFrame(
            right,
            fg_color=CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=CARD_BORDER,
        )
        self.card_info.pack(fill="x", padx=20, pady=(20, 10))
        self.card_info.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Metriche principali
        self.lbl_info_participant = self._make_info_cell(
            self.card_info, 0, 0, "Partecipante / Sessione", "-"
        )
        self.lbl_info_video = self._make_info_cell(
            self.card_info, 0, 1, "Video / Trial", "-"
        )
        self.lbl_info_target_ms = self._make_info_cell(
            self.card_info, 0, 2, "Istante Azione Target", "-"
        )
        self.lbl_info_response_ms = self._make_info_cell(
            self.card_info, 0, 3, "Risposta Vocale (Timing)", "-"
        )

        # Domanda, opzioni e trascrizione
        sub_row = ctk.CTkFrame(self.card_info, fg_color="transparent")
        sub_row.grid(row=1, column=0, columnspan=4, sticky="ew", padx=16, pady=(4, 12))

        ctk.CTkLabel(
            sub_row,
            text="Domanda:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")
        self.lbl_info_question = ctk.CTkLabel(
            sub_row,
            text="-",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LIGHT,
        )
        self.lbl_info_question.pack(side="left", padx=(6, 16))

        ctk.CTkLabel(
            sub_row,
            text="Trascritto:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")
        self.lbl_info_transcript = ctk.CTkLabel(
            sub_row,
            text="-",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_info_transcript.pack(side="left", padx=(6, 16))

        ctk.CTkLabel(
            sub_row,
            text="Risposta originale:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")
        self.lbl_info_dataset_answer = ctk.CTkLabel(
            sub_row,
            text="-",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=BLUE,
        )
        self.lbl_info_dataset_answer.pack(side="left", padx=(6, 0))

        # Card per la correzione manuale della risposta vocale
        card_manual = ctk.CTkFrame(
            right,
            fg_color=CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=CARD_BORDER,
        )
        card_manual.pack(fill="x", padx=20, pady=(0, 12))

        m_head = ctk.CTkFrame(card_manual, fg_color="transparent")
        m_head.pack(fill="x", padx=16, pady=(8, 4))
        ctk.CTkLabel(
            m_head,
            text="Correzione manuale della risposta",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left")
        self.lbl_manual_badge = ctk.CTkLabel(
            m_head,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=BLUE,
        )
        self.lbl_manual_badge.pack(side="right")

        m_row = ctk.CTkFrame(card_manual, fg_color="transparent")
        m_row.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            m_row,
            text="Timing (ms):",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 4))
        self.entry_manual_onset_ms = ctk.CTkEntry(
            m_row, width=80, font=ctk.CTkFont(size=12)
        )
        self.entry_manual_onset_ms.pack(side="left", padx=(0, 6))
        self.entry_manual_onset_ms.bind("<KeyRelease>", self._on_manual_onset_typing)

        ctk.CTkButton(
            m_row,
            text="Usa cursore",
            width=90,
            height=28,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self._set_manual_onset_from_playhead,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            m_row,
            text="Trascrizione:",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 4))
        self.entry_manual_transcript = ctk.CTkEntry(
            m_row, width=250, font=ctk.CTkFont(size=12)
        )
        self.entry_manual_transcript.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            m_row,
            text="Risposta:",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 4))
        self.seg_manual_answer = ctk.CTkSegmentedButton(
            m_row,
            values=["SI", "NO", "Non risposto"],
            height=28,
            command=self._on_manual_answer_segment_change,
        )
        self.seg_manual_answer.pack(side="left", padx=(0, 12))


        ctk.CTkButton(
            m_row,
            text="Salva",
            width=90,
            height=28,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self._save_manual_fix,
        ).pack(side="right")

        # Area Forma d'Onda & Timeline Canvas
        wf_box = ctk.CTkFrame(
            right,
            fg_color=CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=CARD_BORDER,
        )
        wf_box.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        wf_box.grid_rowconfigure(1, weight=1)
        wf_box.grid_columnconfigure(0, weight=1)

        # Legend header
        leg_frame = ctk.CTkFrame(wf_box, fg_color="transparent")
        leg_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            leg_frame,
            text="Timeline Forma d'Onda (Waveform)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(side="left")

        # Legenda marker
        leg_right = ctk.CTkFrame(leg_frame, fg_color="transparent")
        leg_right.pack(side="right")

        self._make_legend_badge(leg_right, "Azione Video (Target)", WARN)
        self._make_legend_badge(leg_right, "Risposta Vocale", ACCENT)
        self._make_legend_badge(leg_right, "Riproduzione attuale", "#ffffff")

        # Canvas personalizzato per la forma d'onda
        self.canvas = tk.Canvas(
            wf_box,
            bg="#111118",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_click)

        # ── Controlli di Riproduzione ──────────────────────────
        ctrl_bar = ctk.CTkFrame(right, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=20, pady=(0, 20))

        self.btn_play = ctk.CTkButton(
            ctrl_bar,
            text="▶ Riproduci",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            width=130,
            height=40,
            command=self.toggle_play,
        )
        self.btn_play.pack(side="left", padx=(0, 16))

        ctk.CTkButton(
            ctrl_bar,
            text="-0.1s",
            width=54,
            height=36,
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=lambda: self.seek_offset(-0.1),
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            ctrl_bar,
            text="+0.1s",
            width=54,
            height=36,
            fg_color=CARD_BG,
            hover_color="#2a2a34",
            command=lambda: self.seek_offset(0.1),
        ).pack(side="left", padx=2)

        self.lbl_time = ctk.CTkLabel(
            ctrl_bar,
            text="0.00s / 0.00s",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_LIGHT,
        )
        self.lbl_time.pack(side="right", padx=10)

    @staticmethod
    def _make_info_cell(parent, r, c, label, value):
        cell = ctk.CTkFrame(parent, fg_color="transparent")
        cell.grid(row=r, column=c, sticky="w", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            cell, text=label, font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        ).pack(anchor="w")
        val_lbl = ctk.CTkLabel(
            cell,
            text=value,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_LIGHT,
        )
        val_lbl.pack(anchor="w")
        return val_lbl

    @staticmethod
    def _make_legend_badge(parent, text, color):
        b = ctk.CTkFrame(parent, fg_color="transparent")
        b.pack(side="left", padx=8)
        dot = ctk.CTkFrame(b, width=10, height=10, corner_radius=5, fg_color=color)
        dot.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(
            b, text=text, font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        ).pack(side="left")

    # ═════════════════════════════════════════════════════════════
    #  SCANSIONE REPOSITORIO AUDIO E DATI
    # ═════════════════════════════════════════════════════════════
    def handle_shortcut(self, key):
        if key in ("space", "Space"):
            # Non attivare se l'utente sta scrivendo nella casella di ricerca
            try:
                focused = self.focus_get()
                if hasattr(self, "txt_search") and focused == self.txt_search:
                    return
            except Exception:
                pass
            self.toggle_play()

    def on_show(self):
        self.refresh()

    def refresh(self):
        self.stop()
        sel_item = self.selected_item
        sel_path = sel_item["path"] if sel_item else None
        sel_session = sel_item.get("session_index") if sel_item else None
        sel_trial = sel_item.get("trial_num") if sel_item else None

        self.items = self._discover_audio_files()
        self._populate_list()

        if self.items:
            # Mantieni selezionato lo stesso elemento dopo il refresh se presente
            matched = None
            if sel_path:
                for it in self.items:
                    if it["path"] == sel_path:
                        matched = it
                        break
            if not matched and sel_session is not None and sel_trial is not None:
                for it in self.items:
                    if it.get("session_index") == sel_session and it.get("trial_num") == sel_trial:
                        matched = it
                        break

            if matched:
                self.selected_item = matched
                self._populate_list()
                self._load_manual_controls(matched)
                self._update_info_card()
                self._render_waveform()
            elif not self.selected_item:
                self._select_item(self.items[0])

    def _discover_audio_files(self):
        """Scansiona ricorsivamente la cartella audio_responses/ e le sottocartelle
        ed incrocia le informazioni con test_results.csv e video_config.json."""
        items = []

        found_paths = set()

        candidates = [
            AUDIO_RESPONSES_FOLDER,
            pathlib.Path("audio_responses"),
            pathlib.Path.cwd() / "audio_responses",
        ]
        for cdir in candidates:
            try:
                resolved = cdir.resolve()
                if resolved.exists():
                    for ext in ("*.wav", "*.WAV", "*.mp3", "*.MP3", "*.m4a", "*.ogg", "*.flac"):
                        for p in resolved.rglob(ext):
                            if p.is_file():
                                found_paths.add(p)
            except Exception:
                pass

        csv_results = self.app.results_mgr.load_results()
        for r in csv_results:
            af = r.get("audio_file", "").strip()
            if af:
                for cdir in candidates:
                    p = (cdir / af).resolve() if not pathlib.Path(af).is_absolute() else pathlib.Path(af)
                    if p.exists() and p.is_file():
                        found_paths.add(p)

        cfg_all = self.app.config_mgr.load()

        for audio_path in sorted(found_paths, key=lambda p: p.name):
            folder_name = audio_path.parent.name
            m_folder = re.match(r"^(\d+)\s*-\s*(.+)$", folder_name)

            if m_folder:
                session_idx = int(m_folder.group(1))
                participant_name = m_folder.group(2).strip()
            else:
                session_idx = 0
                participant_name = folder_name if folder_name != "audio_responses" else "Partecipante"
                for parent in audio_path.parents:
                    m_p = re.match(r"^(\d+)\s*-\s*(.+)$", parent.name)
                    if m_p:
                        session_idx = int(m_p.group(1))
                        participant_name = m_p.group(2).strip()
                        break

            m_file = re.match(r"^(\d+)_(.+)\.\w+$", audio_path.name, re.IGNORECASE)
            if m_file:
                trial_num = int(m_file.group(1))
                video_stem = m_file.group(2)
            else:
                trial_num = 0
                video_stem = audio_path.stem

            norm_audio = unicodedata.normalize("NFC", str(audio_path)).replace("\\", "/")
            rel_audio = (
                norm_audio.split("audio_responses/", 1)[1]
                if "audio_responses/" in norm_audio
                else audio_path.name
            )

            # Match matching row in CSV
            csv_row = None

            # 1. Match su path relativo sotto audio_responses/
            for r in csv_results:
                af = r.get("audio_file", "").strip()
                if af:
                    norm_af = unicodedata.normalize("NFC", str(af)).replace("\\", "/")
                    r_rel = (
                        norm_af.split("audio_responses/", 1)[1]
                        if "audio_responses/" in norm_af
                        else pathlib.Path(norm_af).name
                    )
                    if r_rel == rel_audio or norm_af.endswith(rel_audio):
                        csv_row = r
                        break

            # 2. Match su participant_name + trial_num + session_index
            if not csv_row and participant_name and trial_num:
                for r in csv_results:
                    pname = r.get("participant_name", "").strip().lower()
                    tidx = str(r.get("trial_index", "")).strip()
                    if pname == participant_name.lower() and tidx == str(trial_num):
                        af = r.get("audio_file", "")
                        if session_idx > 0 and af and f"{session_idx} -" not in af:
                            continue
                        csv_row = r
                        break

            # 3. Fallback: match su participant_name + trial_num
            if not csv_row and participant_name and trial_num:
                for r in csv_results:
                    pname = r.get("participant_name", "").strip().lower()
                    tidx = str(r.get("trial_index", "")).strip()
                    if pname == participant_name.lower() and tidx == str(trial_num):
                        csv_row = r
                        break

            # 4. Fallback: match per nome file univoco
            if not csv_row:
                cands = [
                    r for r in csv_results
                    if r.get("audio_file") and pathlib.Path(r["audio_file"].strip()).name == audio_path.name
                ]
                if len(cands) == 1:
                    csv_row = cands[0]

            matched_vcfg = None
            for vname, vcfg in cfg_all.items():
                if pathlib.Path(vname).stem == video_stem or vname == video_stem:
                    matched_vcfg = vcfg
                    break
            if not matched_vcfg and csv_row and csv_row.get("video_filename"):
                v_stem2 = pathlib.Path(csv_row["video_filename"]).stem
                for vname, vcfg in cfg_all.items():
                    if pathlib.Path(vname).stem == v_stem2 or vname == csv_row["video_filename"]:
                        matched_vcfg = vcfg
                        break

            question = ""
            label_yes = "SI"
            label_no = "NO"
            target_ms = None
            response_time_ms = None
            delta_ms = None
            answer = "no_response"
            transcript = ""
            video_filename = f"{video_stem}.mp4"

            answer_key = ""
            correct_answer = ""
            timestamp = ""
            session_seed = ""

            if csv_row:
                timestamp = csv_row.get("timestamp", "")
                session_seed = csv_row.get("session_seed", "")
                video_filename = csv_row.get("video_filename", video_filename)
                question = csv_row.get("question", "")
                label_yes = (csv_row.get("label_yes") or "").strip() or "SI"
                label_no = (csv_row.get("label_no") or "").strip() or "NO"
                answer = csv_row.get("answer", "no_response")
                transcript = csv_row.get("voice_transcript", "")
                answer_key = csv_row.get("answer_key", "")
                correct_answer = csv_row.get("correct_answer", "")
                try:
                    target_ms = int(csv_row.get("target_ms"))
                except (ValueError, TypeError):
                    pass
                try:
                    response_time_ms = int(csv_row.get("response_time_ms"))
                except (ValueError, TypeError):
                    pass
                try:
                    delta_ms = int(csv_row.get("delta_ms"))
                except (ValueError, TypeError):
                    pass

                answered_flag = str(csv_row.get("answered", "")).lower() in ("true", "1", "yes")
                if answer == "no_response" and response_time_ms is None:
                    delta_ms = None


            if matched_vcfg:
                if target_ms is None:
                    target_ms = matched_vcfg.get("target_ms")
                if not question:
                    question = matched_vcfg.get("question", "")
                if matched_vcfg.get("correct_answer"):
                    correct_answer = matched_vcfg.get("correct_answer", "")
                elif not correct_answer:
                    correct_answer = matched_vcfg.get("correct_answer", "")
                if matched_vcfg.get("label_yes"):
                    label_yes = (matched_vcfg.get("label_yes") or "").strip() or label_yes
                elif not label_yes or label_yes == "SI":
                    label_yes = (matched_vcfg.get("label_yes") or "").strip() or label_yes or "SI"
                if matched_vcfg.get("label_no"):
                    label_no = (matched_vcfg.get("label_no") or "").strip() or label_no
                elif not label_no or label_no == "NO":
                    label_no = (matched_vcfg.get("label_no") or "").strip() or label_no or "NO"

            items.append(
                {
                    "path": audio_path,
                    "session_index": session_idx,
                    "participant_name": participant_name,
                    "trial_num": trial_num,
                    "video_stem": video_stem,
                    "video_filename": video_filename,
                    "question": question,
                    "label_yes": label_yes,
                    "label_no": label_no,
                    "answer": answer,
                    "answer_key": answer_key,
                    "correct_answer": correct_answer,
                    "target_ms": target_ms,
                    "response_time_ms": response_time_ms,
                    "delta_ms": delta_ms,
                    "transcript": transcript,
                    "timestamp": timestamp,
                    "session_seed": session_seed,
                }
            )

        items.sort(key=lambda x: (x["session_index"], x["trial_num"]))
        return items

    def _populate_list(self):
        for w in self.list_container.winfo_children():
            w.destroy()

        query = (self.txt_search.get() or "").strip().lower()

        filtered = [
            it
            for it in getattr(self, "items", [])
            if not query
            or query in it["participant_name"].lower()
            or query in it["video_filename"].lower()
            or query in it["transcript"].lower()
        ]

        if not filtered:
            ctk.CTkLabel(
                self.list_container,
                text="Nessun audio trovato.",
                font=ctk.CTkFont(size=12),
                text_color=TEXT_MUTED,
            ).pack(pady=20)
            return

        current_session = None
        for item in filtered:
            sess_title = f"Sessione {item['session_index']} - {item['participant_name']}"
            if sess_title != current_session:
                current_session = sess_title
                grp = ctk.CTkFrame(self.list_container, fg_color="transparent")
                grp.pack(fill="x", pady=(10, 4))
                ctk.CTkLabel(
                    grp,
                    text=sess_title,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=ACCENT,
                ).pack(anchor="w", padx=6)

            # Bottone card per il singolo trial audio
            is_sel = self.selected_item and self.selected_item["path"] == item["path"]
            btn = ctk.CTkFrame(
                self.list_container,
                fg_color=CARD_BG if is_sel else "transparent",
                border_width=1,
                border_color=ACCENT if is_sel else CARD_BORDER,
                corner_radius=8,
                cursor="hand2",
            )
            btn.pack(fill="x", pady=2, padx=2)

            inner = ctk.CTkFrame(btn, fg_color="transparent")
            inner.pack(fill="x", padx=10, pady=8)

            t_title = f"Trial #{item['trial_num']}  -  {item['video_filename']}"
            ctk.CTkLabel(
                inner,
                text=t_title,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_LIGHT,
                anchor="w",
            ).pack(anchor="w")

            ly = (item.get("label_yes") or "").strip() or "SI"
            lno = (item.get("label_no") or "").strip() or "NO"
            if item["answer"] == "yes":
                ans_txt = f"Risposta: {ly}"
            elif item["answer"] == "no":
                ans_txt = f"Risposta: {lno}"
            else:
                ans_txt = "Non risposto"

            sub_info = f"{ans_txt}"
            if item["response_time_ms"] is not None:
                sub_info += f"  ({item['response_time_ms']} ms)"

            ctk.CTkLabel(
                inner,
                text=sub_info,
                font=ctk.CTkFont(size=10),
                text_color=BLUE if item["answer"] != "no_response" else TEXT_MUTED,
                anchor="w",
            ).pack(anchor="w", pady=(2, 0))

            if item["transcript"]:
                ctk.CTkLabel(
                    inner,
                    text=f"\u00ab{item['transcript']}\u00bb",
                    font=ctk.CTkFont(size=10, slant="italic"),
                    text_color=TEXT_MUTED,
                    anchor="w",
                ).pack(anchor="w", pady=(1, 0))

            def bind_click(w, it=item):
                w.bind("<Button-1>", lambda e: self._select_item(it))
                for child in w.winfo_children():
                    bind_click(child, it)

            bind_click(btn, item)

    def _select_item(self, item):
        self.stop()
        self.selected_item = item
        self._populate_list()
        self._load_audio_file(item["path"])
        self._load_manual_controls(item)
        self._update_info_card()
        self.playback_time = 0.0
        self._render_waveform()

    def _load_manual_controls(self, it):
        if not it:
            return
        self.lbl_manual_badge.configure(text="")

        self.entry_manual_onset_ms.delete(0, "end")
        is_answered = (
            str(it.get("answered", "")).lower() in ("true", "1", "yes")
            or it.get("answer") in ("yes", "no")
        ) and it.get("response_time_ms") is not None
        if is_answered and it.get("response_time_ms") is not None:
            self.entry_manual_onset_ms.insert(0, str(it["response_time_ms"]))

        self.entry_manual_transcript.delete(0, "end")
        self.entry_manual_transcript.insert(0, it.get("transcript") or "")

        ly = (it.get("label_yes") or "").strip() or "SI"
        lno = (it.get("label_no") or "").strip() or "NO"
        opt1_val = ly
        opt2_val = lno
        opt3_val = "Non risposto"
        self._current_opt_values = (opt1_val, opt2_val, opt3_val)
        self.seg_manual_answer.configure(values=[opt1_val, opt2_val, opt3_val])

        if it.get("answer") == "yes":
            self.seg_manual_answer.set(opt1_val)
        elif it.get("answer") == "no":
            self.seg_manual_answer.set(opt2_val)
        else:
            self.seg_manual_answer.set(opt3_val)

    def _update_info_card(self):
        it = self.selected_item
        if not it:
            return
        self.lbl_info_participant.configure(
            text=f"{it['participant_name']} (Sess. #{it['session_index']})"
        )
        self.lbl_info_video.configure(
            text=f"Trial #{it['trial_num']} - {it['video_filename']}"
        )
        self.lbl_info_target_ms.configure(
            text=f"{it['target_ms']} ms" if it["target_ms"] is not None else "-"
        )
        is_answered = (
            (str(it.get("answered", "")).lower() in ("true", "1", "yes") or it.get("answer") in ("yes", "no"))
            and it.get("response_time_ms") is not None
        )
        if is_answered:
            resp_txt = f"{it['response_time_ms']} ms"
            if it.get("delta_ms") is not None:
                resp_txt += f"  (Δ {it['delta_ms']:+} ms)"
        else:
            resp_txt = "Non risposto (-)"
        self.lbl_info_response_ms.configure(text=resp_txt)
        self.lbl_info_question.configure(text=it.get("question") or "-")
        self.lbl_info_transcript.configure(
            text=f"\u00ab{it['transcript']}\u00bb" if it.get("transcript") else "-"
        )

        # Risposta originale del dataset
        ca = (it.get("correct_answer") or "").strip().lower()
        ly = (it.get("label_yes") or "").strip() or "SI"
        lno = (it.get("label_no") or "").strip() or "NO"
        if ca in ("yes", "true", "1", "sì", "si"):
            dataset_ans = ly
        elif ca in ("no", "false", "0"):
            dataset_ans = lno
        elif ca:
            dataset_ans = ca.upper()
        else:
            dataset_ans = "-"
        self.lbl_info_dataset_answer.configure(text=dataset_ans)

    def _on_manual_answer_segment_change(self, value):
        if not self.selected_item:
            return
        it = self.selected_item
        opt1_val, opt2_val, opt3_val = getattr(
            self, "_current_opt_values", ("SI", "NO", "Non risposto")
        )
        ly = (it.get("label_yes") or "").strip() or "SI"
        lno = (it.get("label_no") or "").strip() or "NO"
        v_clean = (value or "").strip().lower()

        if value == opt3_val or v_clean in ("non risposto", "no_response", "nessuna risposta", "none", ""):
            it["answer"] = "no_response"
            it["answered"] = "False"
            it["response_time_ms"] = None
            it["delta_ms"] = None
            self.entry_manual_onset_ms.delete(0, "end")
        elif value == opt1_val or v_clean == ly.lower() or v_clean in ("yes", "sì", "si", "true", "y", "opzione 1"):
            it["answer"] = "yes"
            it["answered"] = "True"
            curr_txt = (self.entry_manual_onset_ms.get() or "").strip().replace(",", ".")
            if not curr_txt and self.playback_time > 0:
                val = int(round(self.playback_time * 1000.0))
                self.entry_manual_onset_ms.delete(0, "end")
                self.entry_manual_onset_ms.insert(0, str(val))
                it["response_time_ms"] = val
                if it.get("target_ms") is not None:
                    it["delta_ms"] = val - it["target_ms"]
            elif curr_txt:
                try:
                    val = int(round(float(curr_txt)))
                    it["response_time_ms"] = val
                    if it.get("target_ms") is not None:
                        it["delta_ms"] = val - it["target_ms"]
                except Exception:
                    pass
        elif value == opt2_val or v_clean == lno.lower() or v_clean in ("no", "false", "n", "opzione 2"):
            it["answer"] = "no"
            it["answered"] = "True"
            curr_txt = (self.entry_manual_onset_ms.get() or "").strip().replace(",", ".")
            if not curr_txt and self.playback_time > 0:
                val = int(round(self.playback_time * 1000.0))
                self.entry_manual_onset_ms.delete(0, "end")
                self.entry_manual_onset_ms.insert(0, str(val))
                it["response_time_ms"] = val
                if it.get("target_ms") is not None:
                    it["delta_ms"] = val - it["target_ms"]
            elif curr_txt:
                try:
                    val = int(round(float(curr_txt)))
                    it["response_time_ms"] = val
                    if it.get("target_ms") is not None:
                        it["delta_ms"] = val - it["target_ms"]
                except Exception:
                    pass

        self.lbl_manual_badge.configure(text="")
        self._update_info_card()
        self._render_waveform()

    def _set_manual_onset_from_playhead(self):
        val = int(round(self.playback_time * 1000.0))
        self.entry_manual_onset_ms.delete(0, "end")
        self.entry_manual_onset_ms.insert(0, str(val))
        if self.selected_item:
            self.selected_item["response_time_ms"] = val
            if self.selected_item.get("target_ms") is not None:
                self.selected_item["delta_ms"] = val - self.selected_item["target_ms"]
            self._update_info_card()
            self._render_waveform()

    def _on_manual_onset_typing(self, event=None):
        if not self.selected_item:
            return
        txt = (self.entry_manual_onset_ms.get() or "").strip().replace(",", ".")
        try:
            val = int(round(float(txt)))
            self.selected_item["response_time_ms"] = val
            if self.selected_item.get("target_ms") is not None:
                self.selected_item["delta_ms"] = val - self.selected_item["target_ms"]
        except Exception:
            if not txt:
                self.selected_item["response_time_ms"] = None
                self.selected_item["delta_ms"] = None
        self._update_info_card()
        self._render_waveform()

    def _save_manual_fix(self):
        if not self.selected_item:
            return

        it = self.selected_item
        transcript = (self.entry_manual_transcript.get() or "").strip()
        ans_seg = self.seg_manual_answer.get()
        opt1_val, opt2_val, opt3_val = getattr(
            self, "_current_opt_values", ("SI", "NO", "Non risposto")
        )
        ly = (it.get("label_yes") or "").strip() or "SI"
        lno = (it.get("label_no") or "").strip() or "NO"
        ans_clean = (ans_seg or "").strip().lower()

        # Priorità a 'Non risposto' per evitare conflitti con 'no'
        if ans_seg == opt3_val or ans_clean in ("non risposto", "no_response", "nessuna risposta", "none", ""):
            answer = "no_response"
        elif ans_seg == opt1_val or ans_clean == ly.lower() or ans_clean in ("yes", "sì", "si", "true", "y", "opzione 1"):
            answer = "yes"
        elif ans_seg == opt2_val or ans_clean == lno.lower() or ans_clean in ("no", "false", "n", "opzione 2"):
            answer = "no"
        else:
            answer = "no_response"

        answered = "True" if answer in ("yes", "no") else "False"

        if answer == "no_response":
            onset_ms = None
            response_frame = None

            delta_ms = None
            delta_frames = None
            is_correct = ""
            self.entry_manual_onset_ms.delete(0, "end")
        else:
            raw_onset = (self.entry_manual_onset_ms.get() or "").strip().replace(",", ".")
            try:
                onset_ms = int(round(float(raw_onset)))
            except (ValueError, TypeError):
                messagebox.showerror(
                    APP_TITLE, "Inserisci un valore numerico valido per il timing in ms."
                )
                return

            fps = 25.0
            response_frame = round((onset_ms / 1000.0) * fps)
            target_ms = it.get("target_ms")
            delta_ms = (onset_ms - target_ms) if target_ms is not None else None
            delta_frames = (
                round(delta_ms / (1000.0 / fps)) if delta_ms is not None else None
            )

            correct_ans = (it.get("correct_answer") or "").strip().lower()
            if correct_ans in ("yes", "no"):
                is_correct = "True" if answer == correct_ans else "False"
            else:
                is_correct = ""

        updated_dict = {
            "participant_name": it["participant_name"],
            "trial_index": it["trial_num"],
            "label_yes": it.get("label_yes") or "SI",
            "label_no": it.get("label_no") or "NO",
            "answer": answer,
            "answer_key": "voice_manual",
            "answered": answered,
            "response_time_ms": str(onset_ms) if onset_ms is not None else "",
            "response_frame": str(response_frame) if response_frame is not None else "",
            "delta_ms": str(delta_ms) if delta_ms is not None else "",
            "delta_frames": str(delta_frames) if delta_frames is not None else "",
            "is_correct": is_correct,
            "voice_transcript": transcript,
        }

        match_criteria = {
            "timestamp": it.get("timestamp"),
            "session_seed": it.get("session_seed"),
            "audio_path": str(it["path"]),
            "participant_name": it["participant_name"],
            "trial_index": it["trial_num"],
            "video_filename": it["video_filename"],
            "session_index": it.get("session_index"),
        }

        ok = self.app.results_mgr.update_trial(
            it["path"], updated_dict, match_criteria=match_criteria
        )
        if ok:
            # Aggiorna i dati in memoria dell'elemento selezionato
            it["response_time_ms"] = onset_ms
            it["response_frame"] = response_frame
            it["answer"] = answer
            it["transcript"] = transcript
            it["delta_ms"] = delta_ms
            it["delta_frames"] = delta_frames
            it["answer_key"] = "voice_manual"
            it["answered"] = answered
            it["is_correct"] = is_correct

            self.app.set_status(
                f"Correzione salvata per il trial #{it['trial_num']}.", "success"
            )
            self.lbl_manual_badge.configure(
                text="MODIFICA SALVATA", text_color=ACCENT
            )

            # Aggiorna immediatamente l'interfaccia utente:
            # 1. Scheda informativa in alto con i nuovi ms e delta
            self._update_info_card()
            # 2. Timeline forma d'onda: linea verde del cursore vocale rimossa o aggiornata
            self._render_waveform()
            # 3. Lista dei trial a sinistra (testo risposta e timing aggiornati)
            self._populate_list()
        else:
            self.app.set_status("Impossibile aggiornare il file dei risultati.", "error")
            messagebox.showerror(
                APP_TITLE,
                "Impossibile trovare il record corrispondente nel file dei risultati test_results.csv."
            )


    def _load_audio_file(self, path):
        if not path.exists():
            self.audio_data = None
            self.duration_sec = 0.0
            return
        try:
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]  # Converti in mono se stereo
            self.audio_data = data
            self.sample_rate = sr
            self.duration_sec = len(data) / float(sr)
        except Exception as e:  # noqa: BLE001
            self.app.set_status(f"Errore lettura file audio: {e}", "error")
            self.audio_data = None
            self.duration_sec = 0.0

    def _on_canvas_resize(self, event):
        self._render_waveform()

    def _render_waveform(self):
        self.canvas.delete("all")
        self._playhead_line_id = None
        self._playhead_head_id = None

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        cy = ch / 2.0
        self.canvas.create_line(0, cy, cw, cy, fill="#232330", width=1)

        if self.audio_data is None or len(self.audio_data) == 0:
            self.canvas.create_text(
                cw / 2,
                cy,
                text="Audio non disponibile o file danneggiato",
                fill=TEXT_MUTED,
                font=("sans-serif", 12),
            )
            return

        # Forma d'onda sottocampionata
        n_samples = len(self.audio_data)
        n_bins = max(10, int(cw))
        samples_per_bin = n_samples / float(n_bins)

        amps = []
        for i in range(n_bins):
            s_idx = int(i * samples_per_bin)
            e_idx = int((i + 1) * samples_per_bin)
            chunk = self.audio_data[s_idx:e_idx]
            max_amp = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
            amps.append(max_amp)

        max_all = max(np.max(amps), 1e-5)
        bar_w = cw / float(n_bins)
        margin_y = 20
        max_h = (ch / 2.0) - margin_y

        for i, a in enumerate(amps):
            h = (a / max_all) * max_h
            x = i * bar_w
            self.canvas.create_line(
                x, cy - h, x, cy + h, fill=BLUE, width=max(1, int(bar_w))
            )

        # Time Ruler Ticks
        dur = max(self.duration_sec, 0.1)
        step_sec = 0.5 if dur <= 5.0 else (1.0 if dur <= 12.0 else 2.0)
        t = 0.0
        while t <= dur:
            x_t = (t / dur) * cw
            self.canvas.create_line(x_t, ch - 12, x_t, ch, fill="#3b3b4d", width=1)
            self.canvas.create_text(
                x_t,
                ch - 18,
                text=f"{t:.1f}s",
                fill=TEXT_MUTED,
                font=("sans-serif", 9),
            )
            t += step_sec

        # Marker 1: Azione Target Video (Rosso/Ambra)
        it = self.selected_item
        if it and it["target_ms"] is not None:
            target_sec = it["target_ms"] / 1000.0
            x_target = (target_sec / dur) * cw
            if 0 <= x_target <= cw:
                self.canvas.create_line(
                    x_target, 0, x_target, ch, fill=WARN, width=2, dash=(4, 2)
                )
                self.canvas.create_rectangle(
                    x_target - 55, 4, x_target + 55, 22, fill="#38240b", outline=WARN
                )
                self.canvas.create_text(
                    x_target,
                    13,
                    text=f"Azione: {it['target_ms']}ms",
                    fill=WARN,
                    font=("sans-serif", 9, "bold"),
                )

        # Marker 2: Cursore Risposta Vocale (Verde Accent)
        resp_ms = it.get("response_time_ms") if it else None
        if resp_ms is not None:
            resp_sec = resp_ms / 1000.0
            x_resp = (resp_sec / dur) * cw
            if 0 <= x_resp <= cw:
                # Linea verticale cursore verde
                self.canvas.create_line(
                    x_resp, 0, x_resp, ch, fill=ACCENT, width=2
                )
                # Cursore triangolare in alto
                self.canvas.create_polygon(
                    x_resp - 7, 0, x_resp + 7, 0, x_resp, 9, fill=ACCENT
                )
                # Cursore triangolare in basso
                self.canvas.create_polygon(
                    x_resp - 7, ch, x_resp + 7, ch, x_resp, ch - 9, fill=ACCENT
                )
                # Badge informativo sul timing della risposta
                self.canvas.create_rectangle(
                    x_resp - 55, 26, x_resp + 55, 44, fill="#082b1c", outline=ACCENT
                )
                self.canvas.create_text(
                    x_resp,
                    35,
                    text=f"Risposta: {resp_ms}ms",
                    fill=ACCENT,
                    font=("sans-serif", 9, "bold"),
                )


        # Cursore Playhead Timeline (oggetti canvas persistenti aggiornati in tempo reale)
        x_play = (self.playback_time / dur) * cw
        x_play = max(0, min(cw, x_play))
        self._playhead_line_id = self.canvas.create_line(
            x_play, 0, x_play, ch, fill="#ffffff", width=2
        )
        self._playhead_head_id = self.canvas.create_polygon(
            x_play - 6, 0, x_play + 6, 0, x_play, 8, fill="#ffffff"
        )

        self.lbl_time.configure(
            text=f"{self.playback_time:.2f}s / {self.duration_sec:.2f}s"
        )

    def _update_playhead_only(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        dur = max(self.duration_sec, 0.1)

        if cw > 0 and getattr(self, "_playhead_line_id", None) is not None:
            x_play = (self.playback_time / dur) * cw
            x_play = max(0, min(cw, x_play))
            self.canvas.coords(self._playhead_line_id, x_play, 0, x_play, ch)
            self.canvas.coords(
                self._playhead_head_id, x_play - 6, 0, x_play + 6, 0, x_play, 8
            )

        self.lbl_time.configure(
            text=f"{self.playback_time:.2f}s / {self.duration_sec:.2f}s"
        )

    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if not _AUDIO_PLAYBACK_OK or self.audio_data is None:
            self.app.set_status("Librerie audio non disponibili o nessun file.", "error")
            return
        if self.playback_time >= self.duration_sec:
            self.playback_time = 0.0

        self.is_playing = True
        self.btn_play.configure(
            text="⏸ Pausa", fg_color=WARN, hover_color="#d08020"
        )

        start_sample = int(self.playback_time * self.sample_rate)
        chunk = self.audio_data[start_sample:]

        try:
            sd.stop()
            sd.play(chunk, self.sample_rate)
            self._play_start_perf = time.perf_counter()
            self._play_start_offset = self.playback_time
            self._schedule_playhead_update()
        except Exception as e:  # noqa: BLE001
            self.app.set_status(f"Errore riproduzione audio: {e}", "error")
            self.is_playing = False
            self.btn_play.configure(text="▶ Riproduci", fg_color=ACCENT)

    def pause(self):
        self.is_playing = False
        if _AUDIO_PLAYBACK_OK:
            try:
                sd.stop()
            except Exception:
                pass
        self.btn_play.configure(
            text="▶ Riproduci", fg_color=ACCENT, hover_color=ACCENT_HOVER
        )
        if self._update_timer_id is not None:
            try:
                self.after_cancel(self._update_timer_id)
            except Exception:
                pass
            self._update_timer_id = None
        self._update_playhead_only()

    def stop(self):
        self.playback_time = 0.0
        self.pause()

    def seek_offset(self, delta_sec):
        new_time = max(0.0, min(self.duration_sec, self.playback_time + delta_sec))
        was_playing = self.is_playing
        self.stop()
        self.playback_time = new_time
        self._update_playhead_only()
        if was_playing:
            self.play()

    def _on_canvas_click(self, event):
        if self.duration_sec <= 0:
            return
        cw = self.canvas.winfo_width()
        if cw <= 0:
            return
        click_x = max(0, min(cw, event.x))
        ratio = click_x / float(cw)
        was_playing = self.is_playing
        self.stop()
        self.playback_time = ratio * self.duration_sec
        self._update_playhead_only()
        if was_playing:
            self.play()

    def _schedule_playhead_update(self):
        if not self.is_playing:
            return
        elapsed = (
            time.perf_counter() - self._play_start_perf
        ) + self._play_start_offset
        if elapsed >= self.duration_sec:
            self.playback_time = self.duration_sec
            self.pause()
            return
        self.playback_time = elapsed
        self._update_playhead_only()
        self._update_timer_id = self.after(33, self._schedule_playhead_update)

    def _open_audio_folder(self):
        AUDIO_RESPONSES_FOLDER.mkdir(exist_ok=True)
        open_path(AUDIO_RESPONSES_FOLDER.resolve())
