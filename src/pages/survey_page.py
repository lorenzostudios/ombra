"""
OMBRA - Questionario Post-Esperimento (SurveyPage)
Raccoglie i feedback qualitativi del partecipante al termine dei trial visivi,
salvandoli in 'survey_results.csv' e reindirizzando alla schermata dei report.
"""

import datetime
import customtkinter as ctk
from ..constants import (
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    ACCENT,
    ACCENT_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
)


class SurveyPage(ctk.CTkFrame):
    """
    Modulo di raccolta delle impressioni e note finali del partecipante.
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self,
            text="Sondaggio finale",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=30, pady=(26, 6))
        ctk.CTkLabel(
            self,
            text="C'e' stato qualcosa di strano durante il test?",
            font=ctk.CTkFont(size=15),
            text_color=TEXT_LIGHT,
        ).pack(anchor="w", padx=30, pady=(0, 14))

        card = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        card.pack(fill="both", expand=True, padx=30, pady=(0, 10))
        ctk.CTkLabel(
            card,
            text="Appunti finali (facoltativo)",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=18, pady=(16, 4))
        self.txt_survey = ctk.CTkTextbox(card, height=220)
        self.txt_survey.pack(fill="both", expand=True, padx=18, pady=(2, 16))

        ctk.CTkButton(
            self,
            text="Salva e genera report",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            height=44,
            command=self.submit,
        ).pack(padx=30, pady=(0, 20), fill="x")

    def on_show(self):
        self.txt_survey.delete("1.0", "end")

    def submit(self):
        text = self.txt_survey.get("1.0", "end").strip()
        st = self.app.state_
        n_seen = len(st.trial_results)
        n_augm = sum(1 for r in st.trial_results if r.get("is_augmented") == "True")
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "participant_name": st.participant_name,
            "survey_text": text,
            "n_videos_seen": str(n_seen),
            "n_augmented": str(n_augm),
            "n_original": str(n_seen - n_augm),
            "session_seed": str(st.session_seed),
        }
        self.app.results_mgr.append_survey(row)
        self.app.set_status("Sondaggio salvato. Report generato.", "success")
        self.app.show_page("report", force=True)
