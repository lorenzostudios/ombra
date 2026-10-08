"""
OMBRA - Video Perception Test Application
Modulo principale dell'applicazione desktop: gestisce la finestra principale,
la barra di navigazione superiore, il ciclo di vita delle pagine e la barra di stato.
"""

import re
import random
import platform
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from .constants import (
    APP_TITLE,
    BG_DARK,
    PANEL_BG,
    CARD_BG,
    ACCENT,
    ACCENT_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARN,
    DANGER,
    VIDEO_CONFIG_FILE,
    RESULTS_FILE,
    SURVEY_FILE,
    AUDIO_RESPONSES_FOLDER,
    N_TRIALS,
)
from .state import AppState
from .utils import discover_videos, build_balanced_test_session
from .config_manager import ConfigManager
from .results_manager import ResultsManager
from .pages.home_page import HomePage
from .pages.edit_page import EditPage
from .pages.test_page import TestPage
from .pages.survey_page import SurveyPage
from .pages.report_page import ReportPage
from .pages.shadow_gen_page import ShadowGenPage
from .pages.manual_shadow_page import ManualShadowPage
from .pages.ai_shadow_page import AIShadowPage
from .pages.audio_analysis_page import AudioAnalysisPage


class App(ctk.CTk):
    """
    Finestra principale e controller dell'applicazione OMBRA.
    Gestisce lo switch dinamico delle viste (HomePage, TestPage, EditPage, ReportPage, ecc.),
    lo stato globale dell'esperimento (AppState) e il routing dei comandi.
    """

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")

        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        self.state_ = AppState()
        self.config_mgr = ConfigManager(VIDEO_CONFIG_FILE)
        self.results_mgr = ResultsManager(RESULTS_FILE, SURVEY_FILE)

        self._status_after_id = None
        self.current_page_name = "home"

        self._build_layout()
        self._build_pages()
        self.show_page("home", force=True)

        self.bind_all("<KeyPress>", self._on_global_key)
        self.bind_all("<KeyPress-Shift_L>", self._on_global_key)
        self.bind_all("<KeyPress-Shift_R>", self._on_global_key)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(10, self._maximize_window)

    def _build_layout(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.topbar = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color=PANEL_BG)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_propagate(False)

        ctk.CTkLabel(
            self.topbar,
            text=APP_TITLE,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left", padx=(18, 16))

        nav_container = ctk.CTkFrame(self.topbar, fg_color="transparent")
        nav_container.pack(side="right", padx=(0, 16))

        self.nav_buttons = {}
        for key, label in [
            ("home", "Home"),
            ("test", "Test"),
            ("edit", "Dataset"),
            ("shadowgen", "Ombra (calibrazione)"),
            ("manualshadow", "Ombra (manuale)"),
            ("aishadow", "Ombra (AI)"),
            ("audioanalysis", "Analisi risposte vocali"),
            ("report", "Risultati"),
        ]:
            b = ctk.CTkButton(
                nav_container,
                text=label,
                corner_radius=8,
                fg_color="transparent",
                hover_color=CARD_BG,
                text_color=TEXT_LIGHT,
                text_color_disabled=TEXT_MUTED,
                width=120,
                height=32,
                font=ctk.CTkFont(size=12),
                command=lambda k=key: self.show_page(k),
            )
            b.pack(side="left", padx=3, pady=10)
            b.bind("<Enter>", lambda e, k=key: self._on_nav_hover(k, True))
            b.bind("<Leave>", lambda e, k=key: self._on_nav_hover(k, False))
            self.nav_buttons[key] = b

        self.content = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.statusbar = ctk.CTkFrame(
            self, height=34, corner_radius=0, fg_color=PANEL_BG
        )
        self.statusbar.grid(row=2, column=0, sticky="ew")
        self.statusbar.grid_propagate(False)
        self.lbl_status = ctk.CTkLabel(
            self.statusbar,
            text="Pronto.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_status.pack(side="left", padx=14)

    def _on_nav_hover(self, key, entering):
        b = self.nav_buttons.get(key)
        if b is None:
            return
        if key == self.current_page_name or str(b.cget("state")) == "disabled":
            return
        b.configure(text_color="#ffffff" if entering else TEXT_LIGHT)

    def _build_pages(self):
        self.home_page = HomePage(self.content, self)
        self.edit_page = EditPage(self.content, self)
        self.shadowgen_page = ShadowGenPage(self.content, self)
        self.manualshadow_page = ManualShadowPage(self.content, self)
        self.aishadow_page = AIShadowPage(self.content, self)
        self.audioanalysis_page = AudioAnalysisPage(self.content, self)
        self.test_page = TestPage(self.content, self)
        self.survey_page = SurveyPage(self.content, self)
        self.report_page = ReportPage(self.content, self)
        for p in (
            self.home_page,
            self.edit_page,
            self.shadowgen_page,
            self.manualshadow_page,
            self.aishadow_page,
            self.audioanalysis_page,
            self.test_page,
            self.survey_page,
            self.report_page,
        ):
            p.grid(row=0, column=0, sticky="nsew")
        self.pages = {
            "home": self.home_page,
            "edit": self.edit_page,
            "shadowgen": self.shadowgen_page,
            "manualshadow": self.manualshadow_page,
            "aishadow": self.aishadow_page,
            "audioanalysis": self.audioanalysis_page,
            "test": self.test_page,
            "survey": self.survey_page,
            "report": self.report_page,
        }

    def show_page(self, name, force=False):
        if name == "test" and not self.state_.test_active:
            self.set_status("Il test e' accessibile solo durante una sessione.", "warn")
            name, force = "home", True
        if name == self.current_page_name and not force:
            return
        if self.current_page_name == "edit" and self.edit_page.dirty and name != "edit":
            if not self.edit_page.confirm_leave():
                return
        self.current_page_name = name
        page = self.pages[name]
        page.tkraise()
        if hasattr(page, "on_show"):
            page.on_show()
        for k, b in self.nav_buttons.items():
            if k == name:
                b.configure(
                    fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#06120b"
                )
            else:
                b.configure(
                    fg_color="transparent", hover_color=CARD_BG, text_color=TEXT_LIGHT
                )
        self.nav_buttons["test"].configure(
            state="normal" if self.state_.test_active else "disabled"
        )
        self.focus_set()

    def set_status(self, msg, kind="info"):
        colors = {"info": TEXT_MUTED, "success": ACCENT, "error": DANGER, "warn": WARN}
        self.lbl_status.configure(text=msg, text_color=colors.get(kind, TEXT_MUTED))
        if self._status_after_id:
            try:
                self.after_cancel(self._status_after_id)
            except Exception:
                pass
        self._status_after_id = self.after(
            6000,
            lambda: self.lbl_status.configure(text="Pronto.", text_color=TEXT_MUTED),
        )

    def start_test_session(self):
        name = self.state_.participant_name.strip()
        if not name:
            messagebox.showwarning(
                APP_TITLE, "Inserisci il nome del partecipante prima di iniziare."
            )
            self.set_status("Nome partecipante mancante.", "error")
            return
        vids = discover_videos(self.state_.video_folder)
        if not vids:
            messagebox.showwarning(
                APP_TITLE, f"Nessun video trovato in '{self.state_.video_folder}'."
            )
            self.set_status("Nessun video disponibile.", "error")
            return

        seed = random.randint(0, 2**31)
        rng = random.Random(seed)
        chosen = build_balanced_test_session(vids, rng, n_orig=5, n_alt=5)
        if len(chosen) < N_TRIALS:
            self.set_status(
                f"Trovati solo {len(chosen)} video (richiesti {N_TRIALS}).", "warn"
            )

        existing_seeds = set()
        for r in self.results_mgr.load_results():
            s = r.get("session_seed")
            if s:
                existing_seeds.add(s)
        for r in self.results_mgr.load_survey():
            s = r.get("session_seed")
            if s:
                existing_seeds.add(s)
        folder_indices = []
        if AUDIO_RESPONSES_FOLDER.exists():
            for p in AUDIO_RESPONSES_FOLDER.iterdir():
                if p.is_dir():
                    m = re.match(r"^(\d+)\s*-", p.name)
                    if m:
                        folder_indices.append(int(m.group(1)))
        next_idx = max(len(existing_seeds), max(folder_indices, default=0)) + 1
        self.state_.session_index = next_idx

        self.state_.session_video_list = chosen
        self.state_.session_seed = seed
        self.state_.trial_index = 0
        self.state_.trial_results = []
        self.state_.test_active = True
        self._free_editor_memory()
        self.show_page("test", force=True)
        self.test_page.begin_session()

    def _free_editor_memory(self):
        try:
            self.edit_page.player.release()
        except Exception:
            pass
        try:
            self.shadowgen_page.free_memory_if_idle()
        except Exception:
            pass
        try:
            self.manualshadow_page.free_memory_if_idle()
        except Exception:
            pass

    def _is_typing_context(self):
        try:
            w = self.focus_get()
        except Exception:
            return False
        return isinstance(w, (tk.Entry, tk.Text))

    _LEFT_SHIFT_KEYCODES = {50, 160, 0xA0}
    _RIGHT_SHIFT_KEYCODES = {62, 161, 0xA1}

    def _normalize_shift_keysym(self, event) -> str:
        ks = event.keysym
        if ks not in ("Shift_L", "Shift_R", "Shift"):
            return ks
        kc = getattr(event, "keycode", None)
        if kc in self._LEFT_SHIFT_KEYCODES:
            return "Shift_L"
        if kc in self._RIGHT_SHIFT_KEYCODES:
            return "Shift_R"
        return ks

    def _on_global_key(self, event):
        if self._is_typing_context():
            return
        ks = self._normalize_shift_keysym(event)
        ctrl = bool(event.state & 0x4)

        if ks == "Escape":
            self._handle_escape()
            return
        if ctrl and ks.lower() == "s" and self.current_page_name == "edit":
            self.edit_page.save_current()
            return

        page = self.pages.get(self.current_page_name)
        if page is not None and hasattr(page, "handle_shortcut"):
            page.handle_shortcut(ks)

    def _handle_escape(self):
        if self.current_page_name == "edit":
            if self.edit_page.player.playing:
                self.edit_page.player.pause()
            else:
                self.show_page("home")
        elif self.current_page_name == "test":
            pass
        else:
            self.show_page("home")

    def _maximize_window(self):
        system = platform.system()
        try:
            if system == "Windows":
                self.state("zoomed")
            elif system == "Darwin":
                self.update_idletasks()
                self.geometry(
                    f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0"
                )
            else:
                try:
                    self.attributes("-zoomed", True)
                except tk.TclError:
                    self.update_idletasks()
                    self.geometry(
                        f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0"
                    )
        except Exception:
            pass

    def _on_close(self):
        try:
            self.edit_page.player.release()
            self.test_page.player.release()
            if self.test_page._voice_engine is not None:
                self.test_page._voice_engine.stop()
        except Exception:
            pass
        self.destroy()
