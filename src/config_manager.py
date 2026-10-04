"""
MIRA - Gestore della Configurazione Video
Salva e carica da JSON le annotazioni dei filmati (domande, opzioni e frame target).
Include l'auto-inizializzazione da template interni se il file utente non è presente.
"""

import json
import shutil
import pathlib
from .constants import DEFAULT_QUESTION, BUNDLE_DIR

class ConfigManager:
    """
    Interfaccia per la persistenza delle configurazioni video in 'video_config.json'.
    Normalizza le definizioni dei target (frame e millisecondi) e assicura
    valori di default per le domande e le opzioni di risposta (SI/NO o personalizzate).
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        # Se il file di configurazione non esiste nella cartella di esecuzione,
        # prova a copiarlo dai template interni inclusi da PyInstaller (BUNDLE_DIR)
        if not self.path.exists():
            bundled_cfg = BUNDLE_DIR / self.path.name
            if bundled_cfg.exists() and bundled_cfg != self.path:
                try:
                    shutil.copy2(bundled_cfg, self.path)
                except Exception:
                    pass

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_all(self, cfg: dict):
        self.path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def resolve(raw: dict) -> dict:
        """Mappa i vecchi campi (goal_frame/goal_ms) sui nuovi, senza
        forzare una domanda di default: usato per lo stato di configurazione."""
        out = dict(raw)
        if out.get("target_frame") is None and raw.get("goal_frame") is not None:
            out["target_frame"] = raw["goal_frame"]
        if out.get("target_ms") is None and raw.get("goal_ms") is not None:
            out["target_ms"] = raw["goal_ms"]
        return out

    def get_raw_resolved(self, video_path) -> dict:
        cfg = self.load()
        raw = cfg.get(pathlib.Path(video_path).name, {})
        return self.resolve(raw)

    def get_cfg(self, video_path) -> dict:
        """Config normalizzata (con domanda di default) usata in edit/test."""
        out = self.get_raw_resolved(video_path)
        if not out.get("question"):
            out["question"] = DEFAULT_QUESTION
        if not out.get("label_yes"):
            out["label_yes"] = "SI"
        if not out.get("label_no"):
            out["label_no"] = "NO"
        # correct_answer resta None se non impostata (non e' obbligatoria)
        return out

    def save_video_cfg(
        self,
        video_path,
        question,
        target_frame,
        fps,
        n_frames,
        correct_answer=None,
        label_yes="SI",
        label_no="NO",
    ):
        cfg = self.load()
        name = pathlib.Path(video_path).name
        existing = cfg.get(name, {})
        tms = (
            int(target_frame / max(fps, 1) * 1000) if target_frame is not None else None
        )
        existing.update(
            {
                "question": (question or "").strip(),
                "target_frame": target_frame,
                "target_ms": tms,
                "fps": fps,
                "n_frames": n_frames,
                "correct_answer": (
                    correct_answer if correct_answer in ("yes", "no") else None
                ),
                "label_yes": (label_yes or "").strip() or "SI",
                "label_no": (label_no or "").strip() or "NO",
            }
        )
        cfg[name] = existing
        self.save_all(cfg)
        return existing
