"""
MIRA - Dashboard Statistica e Risultati (ReportPage)
Visualizza l'aggregazione statistica delle risposte: confronto tra video originali e alterati,
tempi di reazione medi, delta temporale dal frame target, accuratezza e filtri demografici.
"""

import re
import pathlib
from tkinter import messagebox
import customtkinter as ctk
from ..constants import (
    APP_TITLE,
    BG_DARK,
    PANEL_BG,
    CARD_BG,
    CARD_BORDER,
    ACCENT,
    BLUE,
    BLUE_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARN,
    RESULTS_FILE,
)
from ..utils import open_path

class ReportPage(ctk.CTkFrame):
    """
    Dashboard interattiva per l'analisi dei risultati sperimentali.
    Elabora in tempo reale le statistiche caricate da 'test_results.csv'
    e presenta tabelle riassuntive per età, genere e interesse sportivo.
    """
    SHADOW_WORD_RE = re.compile(r"ombr\w*", re.IGNORECASE)

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app
        self._build()

    def _build(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=100, pady=(24, 20))
        ctk.CTkLabel(
            header,
            text="Report risultati",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="Aggiorna",
            width=100,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.refresh,
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            header,
            text="Apri CSV",
            width=100,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.open_csv,
        ).pack(side="right", padx=4)

        body = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        body.pack(fill="both", expand=True)

        stats_card = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        stats_card.pack(fill="x", padx=100, pady=(12, 14))
        grid = ctk.CTkFrame(stats_card, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=16)
        for i in range(4):
            grid.grid_columnconfigure(i, weight=1)

        specs = [
            ("n_trials", "Video totali"),
            ("yes", "Risposte 1° opz."),
            ("no", "Risposte 2° opz."),
            ("no_response", "Nessuna risposta"),
            ("mean_response_ms", "Tempo medio risposta (ms)"),
            ("mean_delta_ms", "Delta medio (ms)"),
            ("std_delta_ms", "Deviazione std delta (ms)"),
            ("cohens_d", "d di Cohen (Delta)"),
            ("n_correct", "Risposte corrette"),
            ("accuracy_pct", "Accuratezza"),
            ("voice_recognized", "Riconosciute da microfono"),
            ("voice_manual", "Trascritte ma corrette manualmente"),
        ]
        self.stat_labels = {}
        for i, (key, label) in enumerate(specs):
            r, c = divmod(i, 4)
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=r, column=c, sticky="w", padx=10, pady=8)
            ctk.CTkLabel(
                cell, text=label, font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
            ).pack(anchor="w")
            v = ctk.CTkLabel(
                cell,
                text="-",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=TEXT_LIGHT,
            )
            v.pack(anchor="w")
            self.stat_labels[key] = v

        compare_card = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        compare_card.pack(fill="x", padx=100, pady=(0, 14))
        ctk.CTkLabel(
            compare_card,
            text="Originali vs Alterati",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=18, pady=(14, 4))
        compare_grid = ctk.CTkFrame(compare_card, fg_color="transparent")
        compare_grid.pack(fill="x", padx=18, pady=(2, 10))
        compare_grid.grid_columnconfigure(0, weight=0)
        compare_grid.grid_columnconfigure((1, 2), weight=1)

        compare_headers = ["", "Originali", "Alterati"]
        for c, h in enumerate(compare_headers):
            ctk.CTkLabel(
                compare_grid,
                text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_MUTED,
            ).grid(row=0, column=c, sticky="w", padx=6, pady=(0, 6))

        compare_rows = [
            ("n_trials", "Video visualizzati"),
            ("accuracy_pct", "Accuratezza"),
            ("mean_delta_ms", "Delta medio (ms)"),
            ("std_delta_ms", "Deviazione standard (ms)"),
        ]
        self.compare_labels = {"original": {}, "altered": {}}
        for i, (key, label) in enumerate(compare_rows, start=1):
            ctk.CTkLabel(
                compare_grid,
                text=label,
                font=ctk.CTkFont(size=12),
                text_color=TEXT_LIGHT,
            ).grid(row=i, column=0, sticky="w", padx=6, pady=3)
            for c, group in ((1, "original"), (2, "altered")):
                v = ctk.CTkLabel(
                    compare_grid,
                    text="-",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=TEXT_LIGHT,
                )
                v.grid(row=i, column=c, sticky="w", padx=6, pady=3)
                self.compare_labels[group][key] = v

        compare_stats_bar = ctk.CTkFrame(
            compare_card, fg_color=PANEL_BG, corner_radius=10
        )
        compare_stats_bar.pack(fill="x", padx=18, pady=(6, 16))

        self.lbl_welch = ctk.CTkLabel(
            compare_stats_bar,
            text="Test t di Welch: -",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LIGHT,
        )
        self.lbl_welch.pack(side="left", padx=16, pady=8)

        self.lbl_cohen = ctk.CTkLabel(
            compare_stats_bar,
            text="d di Cohen: -",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT,
        )
        self.lbl_cohen.pack(side="right", padx=16, pady=8)

        # Cards: Statistiche demografiche (Originali vs Alterati)
        self.age_stats_table = self._create_demographic_card(
            body,
            title="Originali vs Alterati per Fascia d'Età",
            subtitle="Confronto delle risposte tra video originali e alterati suddivise per le fasce d'età dei partecipanti.",
            height=170,
        )
        self.gender_stats_table = self._create_demographic_card(
            body,
            title="Originali vs Alterati per Sesso",
            subtitle="Confronto delle risposte tra video originali e alterati suddivise per sesso dei partecipanti.",
            height=140,
        )
        self.wc_stats_table = self._create_demographic_card(
            body,
            title="Originali vs Alterati per Seguito World Cup 2026",
            subtitle="Confronto delle risposte tra chi ha seguito la FIFA World Cup 2026 e chi no.",
            height=125,
        )
        self.freq_stats_table = self._create_demographic_card(
            body,
            title="Originali vs Alterati per Frequenza Partite",
            subtitle="Confronto delle risposte tra i partecipanti in base alla regolarità con cui seguono partite di calcio.",
            height=150,
        )

        pervideo_card = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        pervideo_card.pack(fill="x", padx=100, pady=(0, 14))
        ctk.CTkLabel(
            pervideo_card,
            text="Delta medio",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=18, pady=(14, 4))
        ctk.CTkLabel(
            pervideo_card,
            text="Viene calcolata la differenza tra video alterato e la relativa versione originale.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 6))
        self.pervideo_table = ctk.CTkScrollableFrame(
            pervideo_card, fg_color="transparent", height=190
        )
        self.pervideo_table.pack(fill="x", padx=8, pady=(0, 14))
        self.pervideo_table.grid_columnconfigure(0, weight=1)

        comments_card = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        comments_card.pack(fill="x", padx=100, pady=(0, 14))
        chead = ctk.CTkFrame(comments_card, fg_color="transparent")
        chead.pack(fill="x", padx=18, pady=(14, 4))
        ctk.CTkLabel(
            chead,
            text="Commenti sul test",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left")
        self.lbl_shadow_mentions = ctk.CTkLabel(
            chead, text="", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        )
        self.lbl_shadow_mentions.pack(side="right")
        self.txt_comments = ctk.CTkTextbox(
            comments_card, height=170, fg_color=PANEL_BG, wrap="word"
        )
        self.txt_comments.pack(fill="x", padx=18, pady=(2, 16))
        self.txt_comments.configure(state="disabled")

        recent_card = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        recent_card.pack(fill="x", padx=100, pady=(0, 24))
        rhead = ctk.CTkFrame(recent_card, fg_color="transparent")
        rhead.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(
            rhead,
            text="Tentativi recenti",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left")

        self.table = ctk.CTkScrollableFrame(
            recent_card, fg_color="transparent", height=220
        )
        self.table.pack(fill="x", padx=12, pady=(0, 14))
        self.table.grid_columnconfigure(3, weight=1)

    def on_show(self):
        self.refresh()

    @staticmethod
    def _create_demographic_card(parent, title: str, subtitle: str, height: int = 150):
        card = ctk.CTkFrame(
            parent,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
        )
        card.pack(fill="x", padx=100, pady=(0, 14))
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=18, pady=(14, 4))
        ctk.CTkLabel(
            card,
            text=subtitle,
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 6))
        tbl = ctk.CTkScrollableFrame(card, fg_color="transparent", height=height)
        tbl.pack(fill="x", padx=8, pady=(0, 14))
        tbl.grid_columnconfigure(0, weight=1)
        return tbl

    @staticmethod
    def _cell_padx(col: int, n_cols: int):
        left = 14 if col == 0 else 8
        right = 14 if col == n_cols - 1 else 8
        return (left, right)

    def _fmt_group_value(self, grp: dict, key: str) -> str:
        if key == "n_trials":
            return str(grp["n_trials"])
        if key == "accuracy_pct":
            return (
                f"{grp['accuracy_pct']:.0f}%"
                if grp["accuracy_pct"] is not None
                else "-"
            )
        if key == "mean_response_ms":
            return (
                f"{grp['mean_response_ms']:.0f}"
                if grp["mean_response_ms"] is not None
                else "-"
            )
        if key == "mean_delta_ms":
            return (
                f"{grp['mean_delta_ms']:.0f}"
                if grp["mean_delta_ms"] is not None
                else "-"
            )
        if key == "std_delta_ms":
            return (
                f"{grp['std_delta_ms']:.0f}"
                if grp.get("std_delta_ms") is not None
                else "-"
            )
        return "-"

    def _populate_comments(self):
        rows = self.app.results_mgr.load_survey()
        tb = getattr(self.txt_comments, "_textbox", None)
        self.txt_comments.configure(state="normal")
        self.txt_comments.delete("1.0", "end")
        if tb is not None:
            try:
                tb.tag_config("shadow", background=WARN, foreground="#16100a")
            except Exception:
                tb = None

        n_mentions = 0
        if not rows:
            self.txt_comments.insert("1.0", "Nessun commento registrato.")
        else:
            offset = 0
            for r in reversed(rows):
                name = (r.get("participant_name") or "").strip() or "(anonimo)"
                ts = r.get("timestamp", "")
                text = (r.get("survey_text") or "").strip() or "(nessun commento)"
                header = f"{name}   -   {ts}\n"
                self.txt_comments.insert("end", header + text + "\n\n")
                for m in self.SHADOW_WORD_RE.finditer(text):
                    n_mentions += 1
                    if tb is None:
                        continue
                    a = f"1.0+{offset + len(header) + m.start()}c"
                    b = f"1.0+{offset + len(header) + m.end()}c"
                    try:
                        tb.tag_add("shadow", a, b)
                    except Exception:
                        pass
                offset += len(header) + len(text) + 2

        self.txt_comments.configure(state="disabled")
        self.lbl_shadow_mentions.configure(
            text=(
                f"{n_mentions} riferimenti alle ombre"
                if n_mentions
                else "nessun riferimento alle ombre"
            ),
            text_color=WARN if n_mentions else TEXT_MUTED,
        )

    def refresh(self):
        rm = self.app.results_mgr
        stats = rm.stats()
        self.stat_labels["n_trials"].configure(text=str(stats["n_trials"]))
        self.stat_labels["yes"].configure(text=str(stats["yes"]))
        self.stat_labels["no"].configure(text=str(stats["no"]))
        self.stat_labels["no_response"].configure(text=str(stats["no_response"]))
        self.stat_labels["mean_response_ms"].configure(
            text=(
                f"{stats['mean_response_ms']:.0f}"
                if stats["mean_response_ms"] is not None
                else "-"
            )
        )
        self.stat_labels["mean_delta_ms"].configure(
            text=(
                f"{stats['mean_delta_ms']:.0f}"
                if stats["mean_delta_ms"] is not None
                else "-"
            )
        )
        self.stat_labels["std_delta_ms"].configure(
            text=(
                f"{stats['std_delta_ms']:.0f}"
                if stats.get("std_delta_ms") is not None
                else "-"
            )
        )
        cd = stats.get("cohens_d")
        self.stat_labels["cohens_d"].configure(
            text=f"{cd:.2f}" if cd is not None else "-"
        )
        self.stat_labels["n_correct"].configure(
            text=(
                f"{stats['n_correct']}/{stats['n_scored']}"
                if stats["n_scored"]
                else "-"
            )
        )
        self.stat_labels["accuracy_pct"].configure(
            text=(
                f"{stats['accuracy_pct']:.0f}%"
                if stats["accuracy_pct"] is not None
                else "-"
            )
        )

        wt = stats.get("welch_t")
        wdf = stats.get("welch_df")
        wp = stats.get("welch_p")
        if wt is not None and wdf is not None and wp is not None:
            self.lbl_welch.configure(
                text=f"Test t di Welch:  t({wdf:.1f}) = {wt:.2f},  p = {wp:.3f}"
            )
        else:
            self.lbl_welch.configure(text="Test t di Welch: -")

        if cd is not None:
            self.lbl_cohen.configure(
                text=f"d di Cohen:  d = {cd:.2f} (Effect Size)"
            )
        else:
            self.lbl_cohen.configure(text="d di Cohen: -")

        for group in ("original", "altered"):
            grp = stats[group]
            for key, lbl in self.compare_labels[group].items():
                lbl.configure(text=self._fmt_group_value(grp, key))

        if stats["n_voice_trials"]:
            self.stat_labels["voice_recognized"].configure(
                text=f"{stats['n_voice_recognized']}/{stats['n_voice_trials']}"
                f"  ({stats['voice_recognition_pct']:.0f}%)"
            )
            self.stat_labels["voice_manual"].configure(
                text=f"{stats.get('n_voice_manual', 0)}/{stats['n_voice_trials']}"
            )
        else:
            self.stat_labels["voice_recognized"].configure(text="-")
            self.stat_labels["voice_manual"].configure(text="-")

        rows = rm.load_results()

        for w in self.table.winfo_children():
            w.destroy()
        headers = [
            "#",
            "Video",
            "Tipo",
            "Domanda",
            "Risposta",
            "Corretta",
            "Tempo(ms)",
            "Delta(ms)",
        ]
        n_cols = len(headers)
        for c, h in enumerate(headers):
            ctk.CTkLabel(
                self.table,
                text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=ACCENT,
            ).grid(
                row=0,
                column=c,
                sticky="w",
                padx=self._cell_padx(c, n_cols),
                pady=(10, 4),
            )
        last_rows = rows[-30:]
        if not last_rows:
            ctk.CTkLabel(
                self.table,
                text="Nessun risultato disponibile ancora.",
                text_color=TEXT_MUTED,
            ).grid(row=1, column=0, columnspan=8, padx=14, pady=10)
        for i, r in enumerate(reversed(last_rows), start=1):
            q = r.get("question", "")
            q_short = (q[:24] + "...") if len(q) > 24 else q
            is_correct = r.get("is_correct", "")
            correct_txt = {"True": "SI", "False": "NO"}.get(is_correct, "-")
            tipo_txt = "Alterato" if r.get("is_augmented") == "True" else "Originale"
            vals = [
                r.get("trial_index", ""),
                r.get("video_filename", ""),
                tipo_txt,
                q_short,
                r.get("answer", ""),
                correct_txt,
                r.get("response_time_ms", ""),
                r.get("delta_ms", ""),
            ]
            tipo_col = BLUE if tipo_txt == "Alterato" else TEXT_LIGHT
            last_row = i == len(last_rows)
            for c, v in enumerate(vals):
                ctk.CTkLabel(
                    self.table,
                    text=str(v),
                    font=ctk.CTkFont(size=11),
                    text_color=(tipo_col if c == 2 else TEXT_LIGHT),
                ).grid(
                    row=i,
                    column=c,
                    sticky="w",
                    padx=self._cell_padx(c, n_cols),
                    pady=(2, 10 if last_row else 2),
                )

        self._populate_age_stats()
        self._populate_gender_stats()
        self._populate_world_cup_stats()
        self._populate_football_freq_stats()
        self._populate_per_video()
        self._populate_comments()

    def _populate_demographic_table(self, table_widget, first_col_name: str, data: list):
        for w in table_widget.winfo_children():
            w.destroy()

        headers = [
            first_col_name,
            "Partecipanti",
            "Video (Orig / Alt)",
            "Accuratezza Orig",
            "Accuratezza Alt",
            "Delta Orig (ms)",
            "Delta Alt (ms)",
            "Diff Delta (ms)",
            "d di Cohen",
        ]
        n_cols = len(headers)
        for c, h in enumerate(headers):
            table_widget.grid_columnconfigure(c, weight=1)
            ctk.CTkLabel(
                table_widget,
                text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=ACCENT,
            ).grid(
                row=0,
                column=c,
                sticky="w",
                padx=self._cell_padx(c, n_cols),
                pady=(8, 4),
            )

        if not data:
            ctk.CTkLabel(
                table_widget,
                text="Nessun dato disponibile ancora.",
                font=ctk.CTkFont(size=11),
                text_color=TEXT_MUTED,
            ).grid(row=1, column=0, columnspan=n_cols, padx=14, pady=10)
            return

        for i, d in enumerate(data, start=1):
            last_row = i == len(data)
            orig = d["original"]
            alt = d["altered"]

            acc_orig = (
                f"{orig['accuracy_pct']:.0f}%"
                if orig.get("accuracy_pct") is not None
                else "-"
            )
            acc_alt = (
                f"{alt['accuracy_pct']:.0f}%"
                if alt.get("accuracy_pct") is not None
                else "-"
            )

            dm_orig = (
                f"{orig['mean_delta_ms']:+.0f}"
                if orig.get("mean_delta_ms") is not None
                else "-"
            )
            dm_alt = (
                f"{alt['mean_delta_ms']:+.0f}"
                if alt.get("mean_delta_ms") is not None
                else "-"
            )

            if d["diff_delta"] is not None:
                diff_txt = f"{d['diff_delta']:+.0f}"
                diff_col = WARN if abs(d["diff_delta"]) >= 100 else TEXT_LIGHT
            else:
                diff_txt, diff_col = "-", TEXT_MUTED

            cd_txt = f"{d['cohens_d']:.2f}" if d.get("cohens_d") is not None else "-"
            grp_name = d.get("group_name") or d.get("age_group") or d.get("gender") or "-"

            cells = [
                (grp_name, TEXT_LIGHT, "bold"),
                (str(d["n_participants"]), TEXT_LIGHT, "normal"),
                (
                    f"{d['total_trials']} ({orig['n_trials']} / {alt['n_trials']})",
                    TEXT_MUTED,
                    "normal",
                ),
                (acc_orig, TEXT_LIGHT, "normal"),
                (acc_alt, BLUE, "normal"),
                (dm_orig, TEXT_LIGHT, "normal"),
                (dm_alt, BLUE, "normal"),
                (diff_txt, diff_col, "bold"),
                (cd_txt, ACCENT if cd_txt != "-" else TEXT_MUTED, "normal"),
            ]

            for c, (txt, col, weight) in enumerate(cells):
                ctk.CTkLabel(
                    table_widget,
                    text=txt,
                    font=ctk.CTkFont(size=11, weight=weight),
                    text_color=col,
                    anchor="w",
                ).grid(
                    row=i,
                    column=c,
                    sticky="w",
                    padx=self._cell_padx(c, n_cols),
                    pady=(2, 8 if last_row else 2),
                )

    def _populate_age_stats(self):
        data = self.app.results_mgr.stats_by_age_group()
        self._populate_demographic_table(self.age_stats_table, "Fascia d'età", data)

    def _populate_gender_stats(self):
        data = self.app.results_mgr.stats_by_gender()
        self._populate_demographic_table(self.gender_stats_table, "Sesso", data)

    def _populate_world_cup_stats(self):
        data = self.app.results_mgr.stats_by_world_cup()
        self._populate_demographic_table(self.wc_stats_table, "World Cup 2026", data)

    def _populate_football_freq_stats(self):
        data = self.app.results_mgr.stats_by_football_frequency()
        self._populate_demographic_table(self.freq_stats_table, "Frequenza Calcio", data)

    def _populate_per_video(self):
        for w in self.pervideo_table.winfo_children():
            w.destroy()

        headers = ["Video", "Originale (ms)", "Alterato (ms)", "Differenza (ms)"]
        n_cols = len(headers)
        for c, h in enumerate(headers):
            ctk.CTkLabel(
                self.pervideo_table,
                text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=ACCENT,
                anchor="w",
            ).grid(
                row=0,
                column=c,
                sticky="w",
                padx=self._cell_padx(c, n_cols),
                pady=(8, 4),
            )

        data = self.app.results_mgr.per_video_delta()
        if not data:
            ctk.CTkLabel(
                self.pervideo_table,
                text="Nessun dato disponibile ancora.",
                font=ctk.CTkFont(size=11),
                text_color=TEXT_MUTED,
            ).grid(row=1, column=0, columnspan=n_cols, padx=14, pady=10)
            return

        def fmt(mean, n):
            if mean is None:
                return "-"
            return f"{mean:+.0f}   (n={n})"

        for i, d in enumerate(data, start=1):
            last_row = i == len(data)
            if d["diff"] is None:
                diff_txt, diff_col = "-", TEXT_MUTED
            else:
                diff_txt = f"{d['diff']:+.0f}"
                diff_col = WARN if abs(d["diff"]) >= 100 else TEXT_LIGHT
            cells = [
                (d["video"], TEXT_LIGHT),
                (fmt(d["original_mean"], d["original_n"]), TEXT_LIGHT),
                (fmt(d["altered_mean"], d["altered_n"]), BLUE),
                (diff_txt, diff_col),
            ]
            for c, (txt, col) in enumerate(cells):
                ctk.CTkLabel(
                    self.pervideo_table,
                    text=txt,
                    font=ctk.CTkFont(size=11, weight="bold" if c == 3 else "normal"),
                    text_color=col,
                    anchor="w",
                ).grid(
                    row=i,
                    column=c,
                    sticky="w",
                    padx=self._cell_padx(c, n_cols),
                    pady=(2, 8 if last_row else 2),
                )

    def open_csv(self):
        p = pathlib.Path(RESULTS_FILE)
        if not p.exists():
            messagebox.showinfo(
                APP_TITLE, "Nessun file CSV di risultati ancora presente."
            )
            return
        open_path(p.resolve())
