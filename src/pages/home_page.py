"""
MIRA - Pagina Iniziale (HomePage)
Gestisce la registrazione anagrafica del partecipante (nome, età, genere, abitudini calcistiche),
la selezione della modalità di risposta (tastiera/microfono), la selezione della cartella video
e l'avvio della sessione di test bilanciata.
"""

import pathlib
from tkinter import filedialog, messagebox
import customtkinter as ctk
from ..constants import (
    APP_TITLE,
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    ACCENT,
    ACCENT_HOVER,
    BLUE,
    BLUE_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARN,
    VOICE_AVAILABLE,
    _SOUNDDEVICE_OK,
    _WHISPER_OK,
)
from ..utils import discover_videos

class HomePage(ctk.CTkFrame):
    """
    Schermata principale di onboarding del partecipante.
    Presenta i campi anagrafici, la verifica dei requisiti audio e il conteggio
    dei video configurati pronti per la somministrazione del test.
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app
        self.grid_columnconfigure(0, weight=2, uniform="cards")
        self.grid_columnconfigure(1, weight=1, uniform="cards")
        self._build()

    def _card(self, row, col, colspan=1):
        c = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        c.grid(row=row, column=col, columnspan=colspan, sticky="new", padx=30, pady=12)
        return c

    def _build(self):
        ctk.CTkLabel(
            self,
            text="Benvenuti in un esperimento percettivo basato su azioni calcistiche!",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#ffffff",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=30, pady=(26, 20))

        # Card 1: partecipante
        card1 = self._card(2, 0)
        ctk.CTkLabel(
            card1,
            text="Partecipante",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            card1, text="Nome", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED
        ).pack(anchor="w", padx=18)
        self.entry_name = ctk.CTkEntry(card1, placeholder_text="Inserisci il tuo nome")
        self.entry_name.pack(fill="x", padx=18, pady=(2, 10))
        self.entry_name.bind("<KeyRelease>", self._on_name_change)

        # Fascia d'età
        ctk.CTkLabel(
            card1,
            text="Fascia d'età",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=18)
        self.var_age = ctk.StringVar(value="")
        self.age_options = [
            "Tra 18 e 30 anni",
            "Tra 31 e 40 anni",
            "Tra 41 e 50 anni",
            "Tra 51 e 60 anni",
            "Tra 61 e 70 anni",
            "Più di 71 anni",
        ]
        age_frame = ctk.CTkFrame(card1, fg_color="transparent")
        age_frame.pack(fill="x", padx=18, pady=(4, 12))
        for c in range(3):
            age_frame.grid_columnconfigure(c, weight=1, uniform="age_cols")
        self.age_buttons = {}
        for idx, opt in enumerate(self.age_options):
            r, c = divmod(idx, 3)
            btn = ctk.CTkButton(
                age_frame,
                text=opt,
                height=34,
                corner_radius=8,
                border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda o=opt: self._select_age(o),
            )
            btn.grid(
                row=r,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 4, 0 if c == 2 else 4),
                pady=3,
            )
            self.age_buttons[opt] = btn
        self._update_age_buttons()

        # Sesso
        ctk.CTkLabel(
            card1,
            text="Sesso",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=18)
        self.var_gender = ctk.StringVar(value="")
        self.gender_options = ["Maschio", "Femmina", "Altro"]
        gender_frame = ctk.CTkFrame(card1, fg_color="transparent")
        gender_frame.pack(fill="x", padx=18, pady=(4, 14))
        for c in range(3):
            gender_frame.grid_columnconfigure(c, weight=1, uniform="gender_cols")
        self.gender_buttons = {}
        for c, opt in enumerate(self.gender_options):
            btn = ctk.CTkButton(
                gender_frame,
                text=opt,
                height=34,
                corner_radius=8,
                border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda o=opt: self._select_gender(o),
            )
            btn.grid(
                row=0,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 4, 0 if c == 2 else 4),
                pady=2,
            )
            self.gender_buttons[opt] = btn
        self._update_gender_buttons()

        # Hai seguito la FIFA World Cup 2026?
        ctk.CTkLabel(
            card1,
            text="Hai seguito la FIFA World Cup 2026?",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=18)
        self.var_world_cup = ctk.StringVar(value="")
        self.wc_options = ["Sì", "No"]
        wc_frame = ctk.CTkFrame(card1, fg_color="transparent")
        wc_frame.pack(fill="x", padx=18, pady=(4, 12))
        for c in range(2):
            wc_frame.grid_columnconfigure(c, weight=1, uniform="wc_cols")
        self.wc_buttons = {}
        for c, opt in enumerate(self.wc_options):
            btn = ctk.CTkButton(
                wc_frame,
                text=opt,
                height=34,
                corner_radius=8,
                border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda o=opt: self._select_world_cup(o),
            )
            btn.grid(
                row=0,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 4, 0 if c == 1 else 4),
                pady=2,
            )
            self.wc_buttons[opt] = btn
        self._update_world_cup_buttons()

        # Quanto regolarmente segui una partita di calcio?
        ctk.CTkLabel(
            card1,
            text="Quanto regolarmente segui una partita di calcio?",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=18)
        self.var_football_freq = ctk.StringVar(value="")
        self.freq_options = ["Mai", "2/3 all'anno", "1 al mese", "Spesso"]
        freq_frame = ctk.CTkFrame(card1, fg_color="transparent")
        freq_frame.pack(fill="x", padx=18, pady=(4, 14))
        for c in range(4):
            freq_frame.grid_columnconfigure(c, weight=1, uniform="freq_cols")
        self.freq_buttons = {}
        for c, opt in enumerate(self.freq_options):
            btn = ctk.CTkButton(
                freq_frame,
                text=opt,
                height=34,
                corner_radius=8,
                border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda o=opt: self._select_football_freq(o),
            )
            btn.grid(
                row=0,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 3, 0 if c == 3 else 3),
                pady=2,
            )
            self.freq_buttons[opt] = btn
        self._update_football_freq_buttons()

        ctk.CTkLabel(
            card1,
            text="Modalità di risposta",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=18)
        self.mode_options = ["Input (tastiera)", "Voce (microfono)"]
        mode_frame = ctk.CTkFrame(card1, fg_color="transparent")
        mode_frame.pack(fill="x", padx=18, pady=(4, 6))
        for c in range(2):
            mode_frame.grid_columnconfigure(c, weight=1, uniform="mode_cols")
        self.mode_buttons = {}
        for c, opt in enumerate(self.mode_options):
            btn = ctk.CTkButton(
                mode_frame,
                text=opt,
                height=34,
                corner_radius=8,
                border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda o=opt: self._select_mode(o),
            )
            btn.grid(
                row=0,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 4, 0 if c == 1 else 4),
                pady=2,
            )
            self.mode_buttons[opt] = btn
        self._update_mode_buttons()
        self.lbl_mode_hint = ctk.CTkLabel(
            card1,
            text="Utilizzare i tasti shift sinistro e destro per dare la risposta durante la riproduzione del video quando si ritenga sia il momento più adatto.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
        )
        self.lbl_mode_hint.pack(fill="x", padx=18, pady=(10, 12))
        card1.bind(
            "<Configure>",
            lambda e: self.lbl_mode_hint.configure(wraplength=max(50, e.width - 36)),
        )

        self.lbl_warn = ctk.CTkLabel(
            card1, text="", font=ctk.CTkFont(size=11), text_color=WARN
        )
        self.lbl_warn.pack(anchor="w", padx=18)

        self.btn_start = ctk.CTkButton(
            card1,
            text="Avvia test",
            height=40,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            command=self._on_start,
        )
        self.btn_start.pack(fill="x", padx=18, pady=(6, 18))

        # Card 2: dataset video
        card2 = self._card(2, 1)
        ctk.CTkLabel(
            card2,
            text="Dataset video",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=18, pady=(16, 4))
        self.lbl_folder = ctk.CTkLabel(
            card2,
            text=str(self.app.state_.video_folder),
            font=ctk.CTkFont(size=12),
            text_color=TEXT_LIGHT,
            anchor="w",
            justify="left",
        )
        self.lbl_folder.pack(fill="x", padx=18, pady=(6, 4))
        card2.bind(
            "<Configure>",
            lambda e: self.lbl_folder.configure(wraplength=max(50, e.width - 36)),
        )
        ctk.CTkButton(
            card2,
            text="Seleziona cartella",
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self._on_select_folder,
        ).pack(fill="x", padx=18, pady=(4, 10))
        self.lbl_counts = ctk.CTkLabel(
            card2,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            justify="left",
        )
        self.lbl_counts.pack(anchor="w", padx=18, pady=(4, 18))

    def _on_name_change(self, event=None):
        self.app.state_.participant_name = self.entry_name.get()
        self.lbl_warn.configure(text="")

    def _select_mode(self, value):
        if value.startswith("Voce") and not VOICE_AVAILABLE:
            missing = []
            if not _SOUNDDEVICE_OK:
                missing.append("sounddevice/soundfile")
            if not _WHISPER_OK:
                missing.append("openai-whisper")
            messagebox.showwarning(
                APP_TITLE,
                "Librerie per la risposta vocale non installate ("
                + ", ".join(missing)
                + "). Uso i tasti Shift.",
            )
            self.app.state_.answer_mode = "keys"
            self._update_mode_buttons()
            self._update_mode_hint()
            return
        self.app.state_.answer_mode = "voice" if value.startswith("Voce") else "keys"
        self._update_mode_buttons()
        self._update_mode_hint()

    def _update_mode_hint(self):
        if self.app.state_.answer_mode == "voice":
            self.lbl_mode_hint.configure(
                text=(
                    "Dire ad alta voce la risposta durante la riproduzione del video quando si ritenga sia il momento più adatto. L'audio viene registrato e analizzato al termine della riproduzione. La prima trascrizione può richiedere più tempo in modo da scaricare il modello vocale."
                )
            )
        else:
            self.lbl_mode_hint.configure(
                text="Utilizzare i tasti shift sinistro e destro per dare la risposta durante la riproduzione del video quando si ritenga sia il momento più adatto."
            )

    def _on_select_folder(self):
        folder = filedialog.askdirectory(title="Seleziona cartella video")
        if folder:
            self.app.state_.video_folder = pathlib.Path(folder)
            self.lbl_folder.configure(text=str(folder))
            self.app.set_status(f"Cartella video: {folder}", "success")
        self.refresh_counts()

    def _on_start(self):
        name = self.entry_name.get().strip()
        age = self.var_age.get().strip()
        gender = self.var_gender.get().strip()
        wc = self.var_world_cup.get().strip()
        freq = self.var_football_freq.get().strip()

        self.app.state_.participant_name = name
        self.app.state_.participant_age_group = age
        self.app.state_.participant_gender = gender
        self.app.state_.participant_world_cup_2026 = wc
        self.app.state_.participant_football_frequency = freq

        if not name:
            self.lbl_warn.configure(text="Inserisci il tuo nome per continuare.")
            self.entry_name.focus_set()
            return

        if not age:
            self.lbl_warn.configure(text="Seleziona la tua fascia d'età per continuare.")
            return

        if not gender:
            self.lbl_warn.configure(text="Seleziona il sesso per continuare.")
            return

        if not wc:
            self.lbl_warn.configure(text="Rispondi se hai seguito la FIFA World Cup 2026.")
            return

        if not freq:
            self.lbl_warn.configure(text="Indica quanto regolarmente segui una partita di calcio.")
            return

        self.lbl_warn.configure(text="")
        self.app.start_test_session()

    def refresh_counts(self):
        vids = discover_videos(self.app.state_.video_folder)
        n_conf = 0
        n_incomplete = 0
        for v in vids:
            rc = self.app.config_mgr.get_raw_resolved(v)
            q = (rc.get("question") or "").strip()
            tf = rc.get("target_frame")
            if q and tf is not None:
                n_conf += 1
            elif q or tf is not None:
                n_incomplete += 1
        if vids:
            self.lbl_counts.configure(
                text=f"{len(vids)} video trovati\n{n_conf} configurati   |   {n_incomplete} incompleti"
            )
        else:
            self.lbl_counts.configure(
                text="ATTENZIONE: nessun video trovato nella cartella selezionata."
            )

    def _update_tile_group(self, buttons_dict, current_val):
        for opt, btn in buttons_dict.items():
            if opt == current_val:
                btn.configure(
                    fg_color=ACCENT,
                    hover_color=ACCENT_HOVER,
                    border_color=ACCENT,
                    border_width=1,
                    text_color="#06120b",
                    font=ctk.CTkFont(size=12, weight="bold"),
                )
            else:
                btn.configure(
                    fg_color="#1e1e27",
                    hover_color="#282834",
                    border_color="#363647",
                    border_width=1,
                    text_color=TEXT_LIGHT,
                    font=ctk.CTkFont(size=12, weight="normal"),
                )

    def _update_age_buttons(self):
        self._update_tile_group(self.age_buttons, self.var_age.get())

    def _select_age(self, opt):
        self.var_age.set(opt)
        self.app.state_.participant_age_group = opt
        self.lbl_warn.configure(text="")
        self._update_age_buttons()

    def _update_gender_buttons(self):
        self._update_tile_group(self.gender_buttons, self.var_gender.get())

    def _select_gender(self, opt):
        self.var_gender.set(opt)
        self.app.state_.participant_gender = opt
        self.lbl_warn.configure(text="")
        self._update_gender_buttons()

    def _update_world_cup_buttons(self):
        self._update_tile_group(self.wc_buttons, self.var_world_cup.get())

    def _select_world_cup(self, opt):
        self.var_world_cup.set(opt)
        self.app.state_.participant_world_cup_2026 = opt
        self.lbl_warn.configure(text="")
        self._update_world_cup_buttons()

    def _update_football_freq_buttons(self):
        self._update_tile_group(self.freq_buttons, self.var_football_freq.get())

    def _select_football_freq(self, opt):
        self.var_football_freq.set(opt)
        self.app.state_.participant_football_frequency = opt
        self.lbl_warn.configure(text="")
        self._update_football_freq_buttons()

    def _update_mode_buttons(self):
        curr_opt = (
            "Voce (microfono)"
            if getattr(self.app.state_, "answer_mode", "keys") == "voice"
            else "Input (tastiera)"
        )
        self._update_tile_group(self.mode_buttons, curr_opt)

    def on_show(self):
        self.entry_name.delete(0, "end")
        self.entry_name.insert(0, self.app.state_.participant_name)
        self.var_age.set(self.app.state_.participant_age_group or "")
        self._update_age_buttons()
        self.var_gender.set(self.app.state_.participant_gender or "")
        self._update_gender_buttons()
        self.var_world_cup.set(self.app.state_.participant_world_cup_2026 or "")
        self._update_world_cup_buttons()
        self.var_football_freq.set(self.app.state_.participant_football_frequency or "")
        self._update_football_freq_buttons()
        self._update_mode_buttons()
        self.lbl_folder.configure(text=str(self.app.state_.video_folder))
        self.lbl_warn.configure(text="")
        self._update_mode_hint()
        self.refresh_counts()
