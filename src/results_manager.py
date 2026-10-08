"""
OMBRA - Gestore dei Risultati Sperimentali e Statistiche
Modulo dedicato alla persistenza su CSV dei trial e dei questionari, al calcolo
delle metriche di percezione visiva (tempi di reazione, accuratezza, delta temporali)
e all'analisi statistica per fasce demografiche e tipologie video.
"""

import csv
import math
import pathlib
import unicodedata


class ResultsManager:
    """
    Gestisce l'archiviazione e l'elaborazione statistica dei dati sperimentali.
    Scrive e legge 'test_results.csv' e 'survey_results.csv', calcola medie e tassi
    di riconoscimento per video originali e alterati, e produce i riepiloghi per la pagina Report.
    """

    FIELDS = [
        "timestamp",
        "participant_name",
        "age_group",
        "gender",
        "world_cup_2026",
        "football_frequency",
        "trial_index",
        "video_filename",
        "video_path",
        "is_augmented",
        "question",
        "label_yes",
        "label_no",
        "answer",
        "answer_key",
        "answered",
        "response_time_ms",
        "response_frame",
        "video_duration_ms",
        "session_seed",
        "target_frame",
        "target_ms",
        "delta_ms",
        "delta_frames",
        "correct_answer",
        "is_correct",
        "answer_mode",
        "audio_file",
        "voice_transcript",
    ]

    SURVEY_FIELDS = [
        "timestamp",
        "participant_name",
        "survey_text",
        "n_videos_seen",
        "n_augmented",
        "n_original",
        "session_seed",
    ]

    def __init__(self, results_file, survey_file):
        self.results_file = pathlib.Path(results_file)
        self.survey_file = pathlib.Path(survey_file)
        self._ensure_headers()

    def _ensure_headers(self):
        """Assicura che il CSV contenga tutti i campi previsti nell'intestazione."""
        if not self.results_file.exists():
            return
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header and any(
                col not in header
                for col in (
                    "age_group",
                    "gender",
                    "world_cup_2026",
                    "football_frequency",
                )
            ):
                rows = self.load_results()
                with open(self.results_file, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rows)
        except Exception:
            pass

    def append_trial(self, row: dict):
        """Aggiunge la riga dei risultati di un singolo trial in fondo al file CSV."""
        self._ensure_headers()
        is_new = not self.results_file.exists()
        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            if is_new:
                w.writeheader()
            w.writerow(row)

    def append_survey(self, row: dict):
        """Aggiunge le risposte del sondaggio finale in fondo al CSV dedicato."""
        is_new = not self.survey_file.exists()
        with open(self.survey_file, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.SURVEY_FIELDS, extrasaction="ignore")
            if is_new:
                w.writeheader()
            w.writerow(row)

    def load_results(self) -> list:
        """Legge e restituisce l'elenco completo dei risultati registrati."""
        if not self.results_file.exists():
            return []
        with open(self.results_file, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def load_survey(self) -> list:
        """Legge le risposte del sondaggio finale memorizzate nel CSV."""
        if not self.survey_file.exists():
            return []
        with open(self.survey_file, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _normalize_path(p) -> str:
        if not p:
            return ""
        return unicodedata.normalize("NFC", str(p)).replace("\\", "/")

    @classmethod
    def _extract_audio_rel(cls, p) -> str:
        norm = cls._normalize_path(p)
        if not norm:
            return ""
        if "audio_responses/" in norm:
            return norm.split("audio_responses/", 1)[1]
        return pathlib.Path(norm).name

    def update_trial(
        self,
        audio_file_name: str | pathlib.Path,
        updated_dict: dict,
        match_criteria: dict | None = None,
    ) -> bool:
        """Aggiorna i dati di un trial specifico nel file CSV dopo una correzione manuale.
        Usa criteri multipli ad alta precisione (timestamp, path relativo sotto audio_responses,
        path canonico, partecipante + trial_index + sessione) per evitare collisioni di nomi.
        """
        rows = self.load_results()
        if not rows:
            return False

        criteria = match_criteria or {}
        target_timestamp = str(criteria.get("timestamp", "")).strip()
        target_path_norm = self._normalize_path(
            audio_file_name or criteria.get("audio_path", "")
        )
        target_rel = self._extract_audio_rel(target_path_norm)
        target_name = pathlib.Path(target_path_norm).name if target_path_norm else ""

        target_pname = (
            str(
                criteria.get("participant_name")
                or updated_dict.get("participant_name")
                or ""
            )
            .strip()
            .lower()
        )
        target_trial = str(
            criteria.get("trial_index") or updated_dict.get("trial_index") or ""
        ).strip()
        target_seed = str(criteria.get("session_seed") or "").strip()
        target_vname = str(criteria.get("video_filename") or "").strip().lower()
        target_sess_idx = criteria.get("session_index")

        matched_row = None

        # Priorità 1: Match esatto su timestamp univoco
        if target_timestamp:
            for r in rows:
                if str(r.get("timestamp", "")).strip() == target_timestamp:
                    matched_row = r
                    break

        # Priorità 2: Match su path relativo sotto audio_responses/ (es. "15 - Lorenzo/7_WorldCup...wav")
        if matched_row is None and target_rel:
            for r in rows:
                af = r.get("audio_file", "").strip()
                if af:
                    r_rel = self._extract_audio_rel(af)
                    if r_rel == target_rel:
                        matched_row = r
                        break

        # Priorità 3: Match su path normalizzato completo / ends_with
        if matched_row is None and target_path_norm:
            for r in rows:
                af = r.get("audio_file", "").strip()
                if af:
                    r_norm = self._normalize_path(af)
                    if r_norm == target_path_norm or r_norm.endswith(target_rel):
                        matched_row = r
                        break

        # Priorità 4: Match su participant_name + trial_index + session_seed
        if matched_row is None and target_pname and target_trial and target_seed:
            for r in rows:
                if (
                    r.get("participant_name", "").strip().lower() == target_pname
                    and str(r.get("trial_index", "")).strip() == target_trial
                    and str(r.get("session_seed", "")).strip() == target_seed
                ):
                    matched_row = r
                    break

        # Priorità 5: Match su participant_name + trial_index + video_filename
        if matched_row is None and target_pname and target_trial and target_vname:
            for r in rows:
                if (
                    r.get("participant_name", "").strip().lower() == target_pname
                    and str(r.get("trial_index", "")).strip() == target_trial
                    and str(r.get("video_filename", "")).strip().lower() == target_vname
                ):
                    matched_row = r
                    break

        # Priorità 6: Match su participant_name + trial_index (e controllo session_index se presente)
        if matched_row is None and target_pname and target_trial:
            for r in rows:
                if (
                    r.get("participant_name", "").strip().lower() == target_pname
                    and str(r.get("trial_index", "")).strip() == target_trial
                ):
                    af = r.get("audio_file", "")
                    if target_sess_idx is not None and af:
                        if f"{target_sess_idx} -" in af:
                            matched_row = r
                            break
                    else:
                        matched_row = r
                        break

        # Priorità 7: Fallback finale solo se c'è un match univoco sul nome file audio
        if matched_row is None and target_name:
            candidates = []
            for r in rows:
                af = r.get("audio_file", "").strip()
                if af and pathlib.Path(self._normalize_path(af)).name == target_name:
                    candidates.append(r)
            if len(candidates) == 1:
                matched_row = candidates[0]

        if matched_row is not None:
            for k, v in updated_dict.items():
                if k in self.FIELDS:
                    matched_row[k] = str(v) if v is not None else ""

            # Se audio_file non era impostato nella riga CSV, aggiornalo con il path corrente
            if not matched_row.get("audio_file") and target_path_norm:
                matched_row["audio_file"] = str(audio_file_name)

            with open(self.results_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            return True

        return False

    @staticmethod
    def _compute_group_stats(rows: list) -> dict:
        """Calcola le metriche statistiche generali per un gruppo di trial."""
        n = len(rows)
        yes = sum(1 for r in rows if r.get("answer") == "yes")
        no = sum(1 for r in rows if r.get("answer") == "no")
        nr = sum(1 for r in rows if r.get("answer") == "no_response")
        rt = [
            int(r["response_time_ms"])
            for r in rows
            if r.get("answered") == "True"
            and r.get("response_time_ms", "").lstrip("-").isdigit()
        ]
        dm = [
            int(r["delta_ms"])
            for r in rows
            if r.get("delta_ms") not in ("", None)
            and r.get("delta_ms", "").lstrip("-").isdigit()
        ]
        scored = [r for r in rows if r.get("is_correct") in ("True", "False")]
        n_correct = sum(1 for r in scored if r.get("is_correct") == "True")

        mean_rt = (sum(rt) / len(rt)) if rt else None
        var_rt = (
            (sum((x - mean_rt) ** 2 for x in rt) / (len(rt) - 1))
            if len(rt) > 1
            else (0.0 if len(rt) == 1 else None)
        )
        std_rt = (var_rt**0.5) if var_rt is not None else None

        mean_dm = (sum(dm) / len(dm)) if dm else None
        var_dm = (
            (sum((x - mean_dm) ** 2 for x in dm) / (len(dm) - 1))
            if len(dm) > 1
            else (0.0 if len(dm) == 1 else None)
        )
        std_dm = (var_dm**0.5) if var_dm is not None else None

        return {
            "n_trials": n,
            "yes": yes,
            "no": no,
            "no_response": nr,
            "mean_response_ms": mean_rt,
            "var_response_ms": var_rt,
            "std_response_ms": std_rt,
            "mean_delta_ms": mean_dm,
            "var_delta_ms": var_dm,
            "std_delta_ms": std_dm,
            "n_valid_delta": len(dm),
            "n_scored": len(scored),
            "n_correct": n_correct,
            "accuracy_pct": (n_correct / len(scored) * 100) if scored else None,
        }

    @staticmethod
    def _base_video_name(filename: str) -> str:
        """Estrae il nome base del video rimuovendo l'eventuale prefisso '_' dei video alterati."""
        name = (filename or "").strip()
        return name[1:] if name.startswith("_") else name

    @staticmethod
    def _mean_delta(rows: list):
        vals = [
            int(r["delta_ms"])
            for r in rows
            if r.get("delta_ms") not in ("", None)
            and str(r.get("delta_ms", "")).lstrip("-").isdigit()
        ]
        return (sum(vals) / len(vals)) if vals else None, len(vals)

    def per_video_delta(self) -> list:
        """Confronta il delta medio riscontrato tra video originali e rispettive versioni alterate."""
        groups = {}
        for r in self.load_results():
            key = self._base_video_name(r.get("video_filename", ""))
            if not key:
                continue
            side = "altered" if r.get("is_augmented") == "True" else "original"
            groups.setdefault(key, {"original": [], "altered": []})[side].append(r)

        out = []
        for key in sorted(groups):
            orig_mean, orig_n = self._mean_delta(groups[key]["original"])
            alt_mean, alt_n = self._mean_delta(groups[key]["altered"])
            diff = (
                (alt_mean - orig_mean)
                if (orig_mean is not None and alt_mean is not None)
                else None
            )
            out.append(
                {
                    "video": key,
                    "original_mean": orig_mean,
                    "original_n": orig_n,
                    "altered_mean": alt_mean,
                    "altered_n": alt_n,
                    "diff": diff,
                }
            )
        return out

    @staticmethod
    def _student_t_cdf(t: float, df: float) -> float:
        """Calcola il p-value a due code per la distribuzione t di Student (frazione continua di Betai)."""
        if df <= 0:
            return 1.0

        def betacf(a, b, x):
            maxit, eps, fpmin = 200, 3.0e-12, 1.0e-30
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c = 1.0
            d = 1.0 - qab * x / qap
            if abs(d) < fpmin:
                d = fpmin
            d = 1.0 / d
            h = d
            for m in range(1, maxit + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < fpmin:
                    d = fpmin
                c = 1.0 + aa / c
                if abs(c) < fpmin:
                    c = fpmin
                d = 1.0 / d
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < fpmin:
                    d = fpmin
                c = 1.0 + aa / c
                if abs(c) < fpmin:
                    c = fpmin
                d = 1.0 / d
                del_val = d * c
                h *= del_val
                if abs(del_val - 1.0) < eps:
                    break
            return h

        def betai(a, b, x):
            if x <= 0.0:
                return 0.0
            if x >= 1.0:
                return 1.0
            bt = math.exp(
                math.lgamma(a + b)
                - math.lgamma(a)
                - math.lgamma(b)
                + a * math.log(x)
                + b * math.log(1.0 - x)
            )
            if x < (a + 1.0) / (a + b + 2.0):
                return bt * betacf(a, b, x) / a
            return 1.0 - bt * betacf(b, a, 1.0 - x) / b

        x_val = df / (df + t * t)
        return betai(0.5 * df, 0.5, x_val)

    @classmethod
    def _compute_inferential_stats(cls, orig_rows: list, alt_rows: list) -> dict:
        """Calcola il Test t di Welch e il d di Cohen per il delta (Originali vs Alterati)."""
        dm_orig = [
            int(r["delta_ms"])
            for r in orig_rows
            if r.get("delta_ms") not in ("", None)
            and str(r.get("delta_ms", "")).lstrip("-").isdigit()
        ]
        dm_alt = [
            int(r["delta_ms"])
            for r in alt_rows
            if r.get("delta_ms") not in ("", None)
            and str(r.get("delta_ms", "")).lstrip("-").isdigit()
        ]
        n1, n2 = len(dm_orig), len(dm_alt)
        if n1 < 2 or n2 < 2:
            return {
                "cohens_d": None,
                "welch_t": None,
                "welch_df": None,
                "welch_p": None,
            }

        m1 = sum(dm_orig) / n1
        m2 = sum(dm_alt) / n2
        v1 = sum((x - m1) ** 2 for x in dm_orig) / (n1 - 1)
        v2 = sum((x - m2) ** 2 for x in dm_alt) / (n2 - 1)

        # Cohen's d (pooled standard deviation)
        sp = (
            math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            if (n1 + n2 - 2) > 0
            else 0
        )
        cohens_d = ((m1 - m2) / sp) if sp > 0 else 0.0

        # Welch's t-test
        se_welch = math.sqrt(v1 / n1 + v2 / n2)
        welch_t = ((m1 - m2) / se_welch) if se_welch > 0 else 0.0
        denom = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
        welch_df = (((v1 / n1 + v2 / n2) ** 2) / denom) if denom > 0 else (n1 + n2 - 2)
        welch_p = cls._student_t_cdf(abs(welch_t), welch_df)

        return {
            "cohens_d": cohens_d,
            "welch_t": welch_t,
            "welch_df": welch_df,
            "welch_p": welch_p,
        }

    def stats(self) -> dict:
        """Raccoglie e organizza le statistiche aggregate sui risultati registrati."""
        rows = self.load_results()
        out = self._compute_group_stats(rows)
        original_rows = [r for r in rows if r.get("is_augmented") == "False"]
        altered_rows = [r for r in rows if r.get("is_augmented") == "True"]
        out["original"] = self._compute_group_stats(original_rows)
        out["altered"] = self._compute_group_stats(altered_rows)

        # Statistiche inferenziali Originali vs Alterati
        inf = self._compute_inferential_stats(original_rows, altered_rows)
        out.update(inf)

        voice_rows = [r for r in rows if r.get("answer_mode") == "voice"]
        out["n_voice_trials"] = len(voice_rows)
        out["n_voice_recognized"] = sum(
            1 for r in voice_rows if r.get("answer_key") == "voice"
        )
        out["n_voice_manual"] = sum(
            1 for r in voice_rows if r.get("answer_key") == "voice_manual"
        )
        out["voice_recognition_pct"] = (
            (out["n_voice_recognized"] / len(voice_rows) * 100) if voice_rows else None
        )
        return out

    def stats_by_demographic(
        self,
        field_name: str,
        field_order: list | None = None,
        label_key: str | None = None,
    ) -> list:
        """Calcola le metriche suddivise per un campo demografico (Originali vs Alterati)."""
        rows = self.load_results()
        if not rows:
            return []

        fallback_label = (
            "Non specificata" if field_name == "age_group" else "Non specificato"
        )

        groups = {}
        for r in rows:
            raw_val = (r.get(field_name) or "").strip()
            if field_name == "world_cup_2026" and raw_val.lower() in ("si", "sì"):
                val = "Sì"
            elif field_name == "world_cup_2026" and raw_val.lower() == "no":
                val = "No"
            else:
                val = raw_val or fallback_label
            groups.setdefault(val, []).append(r)

        sorted_keys = []
        if field_order:
            for opt in field_order:
                if opt in groups:
                    sorted_keys.append(opt)
        for opt in groups:
            if opt not in sorted_keys and opt not in (
                "Non specificato",
                "Non specificata",
            ):
                sorted_keys.append(opt)
        for fb in ("Non specificato", "Non specificata"):
            if fb in groups and fb not in sorted_keys:
                sorted_keys.append(fb)

        result = []
        for g_label in sorted_keys:
            g_rows = groups[g_label]
            participants = set(
                (r.get("participant_name") or "").strip().lower()
                for r in g_rows
                if (r.get("participant_name") or "").strip()
            )
            orig_rows = [r for r in g_rows if r.get("is_augmented") == "False"]
            alt_rows = [r for r in g_rows if r.get("is_augmented") == "True"]
            orig_stats = self._compute_group_stats(orig_rows)
            alt_stats = self._compute_group_stats(alt_rows)
            tot_stats = self._compute_group_stats(g_rows)

            diff_delta = None
            if (
                alt_stats["mean_delta_ms"] is not None
                and orig_stats["mean_delta_ms"] is not None
            ):
                diff_delta = alt_stats["mean_delta_ms"] - orig_stats["mean_delta_ms"]

            inf = self._compute_inferential_stats(orig_rows, alt_rows)

            item = {
                "group_name": g_label,
                field_name: g_label,
                "n_participants": len(participants),
                "total_trials": len(g_rows),
                "original": orig_stats,
                "altered": alt_stats,
                "total": tot_stats,
                "diff_delta": diff_delta,
                "cohens_d": inf.get("cohens_d"),
                "welch_p": inf.get("welch_p"),
            }
            if label_key and label_key != field_name:
                item[label_key] = g_label
            result.append(item)
        return result

    def stats_by_age_group(self) -> list:
        """Calcola le metriche suddivise per fascia d'età (Originali vs Alterati)."""
        age_order = [
            "Tra 18 e 30 anni",
            "Tra 31 e 40 anni",
            "Tra 41 e 50 anni",
            "Tra 51 e 60 anni",
            "Tra 61 e 70 anni",
            "Più di 71 anni",
        ]
        return self.stats_by_demographic("age_group", age_order, label_key="age_group")

    def stats_by_gender(self) -> list:
        """Calcola le metriche suddivise per sesso (Originali vs Alterati)."""
        gender_order = ["Maschio", "Femmina", "Altro"]
        return self.stats_by_demographic("gender", gender_order, label_key="gender")

    def stats_by_world_cup(self) -> list:
        """Calcola le metriche suddivise per interesse World Cup 2026 (Originali vs Alterati)."""
        wc_order = ["Sì", "No"]
        return self.stats_by_demographic(
            "world_cup_2026", wc_order, label_key="world_cup_2026"
        )

    def stats_by_football_frequency(self) -> list:
        """Calcola le metriche suddivise per frequenza partite di calcio (Originali vs Alterati)."""
        freq_order = ["Mai", "2/3 all'anno", "1 al mese", "Spesso"]
        return self.stats_by_demographic(
            "football_frequency", freq_order, label_key="football_frequency"
        )
