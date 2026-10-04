"""
Modulo Machine Learning per la predizione della traiettoria e dei parametri dell'ombra virtuale.
Supporta l'estrazione delle feature cinematiche, l'Action Recognition balistica (lancio, tiro, rasoterra, rimbalzo)
e l'apprendimento per differenza pixel supervisionata dai video originali e modificati.
"""

import json
import pathlib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

WEIGHTS_FILE = pathlib.Path(__file__).resolve().parent / "shadow_model_weights.json"

# Tipologie di azioni calcistiche riconosciute
ACTION_LANCIO = "LANCIO_PARABOLICO"    # Lancio lungo / Cross / Pallonetto in quota
ACTION_TIRO = "TIRO_TESO"              # Tiro potente / Passaggio filtrante teso
ACTION_RASOTERRA = "RASOTERRA"          # Passaggio rasoterra / Pallone che rotola sul prato
ACTION_RIMBALZO = "RIMBALZO"            # Rimbalzo / Impatto al suolo


def classify_trajectory_action(
    ball_xs: np.ndarray,
    ball_ys: np.ndarray,
    heights: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    fps: float = 25.0,
) -> Dict:
    """
    Analizza la cinematica fisica della traiettoria (curvatura parabolica R2, escursione Y, velocità, rimbalzi)
    per classificare con precisione il tipo di azione calcistica reale:
    - LANCIO_PARABOLICO: arco aereo balistico in quota (curvatura a > 0.04, R2 > 0.60 o escursione Y > 85 px).
    - TIRO_TESO: traiettoria tesa e diretta ad alta velocità.
    - RASOTERRA: pallone aderente al manto erboso (escursione Y ridotta < 26 px).
    - RIMBALZO: impatto al suolo con inversione repentina della velocità verticale vy.
    """

    if valid_mask is None:
        valid_mask = np.ones(len(ball_xs), dtype=bool)

    valid_idxs = np.where(valid_mask)[0]
    if len(valid_idxs) < 5:
        return {
            "action_type": ACTION_TIRO,
            "label_it": "Traiettoria Diretta",
            "confidence": 0.65,
            "delta_y": 10.0,
            "mean_speed": 15.0,
            "description": "Traiettoria breve",
        }

    xs = ball_xs[valid_idxs]
    ys = ball_ys[valid_idxs]
    n = len(xs)
    t = np.arange(n, dtype=float)

    # 1. Cinematica e velocità
    vx = np.gradient(xs)
    vy = np.gradient(ys)
    speeds = np.hypot(vx, vy)
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))

    delta_y = float(np.max(ys) - np.min(ys))
    span_x = float(np.max(xs) - np.min(xs))

    # 2. Fit parabolico: y(t) = a*t^2 + b*t + c
    poly = np.polyfit(t, ys, 2)
    fit_y = np.polyval(poly, t)
    ss_res = np.sum((ys - fit_y) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2) + 1e-5
    r2 = float(1.0 - (ss_res / ss_tot))
    curv_a = float(poly[0])

    # 3. Rilevamento rimbalzo al suolo (inversione rapida di vy nella zona bassa del campo)
    has_bounce = False
    if n >= 8:
        for i in range(2, n - 2):
            if vy[i - 1] > 2.5 and vy[i + 1] < -2.5 and ys[i] >= np.percentile(ys, 70):
                has_bounce = True
                break

    # 4. Classificazione multi-criterio ad alta fedeltà
    if has_bounce:
        act = ACTION_RIMBALZO
        lbl = "Rimbalzo / Impatto al Suolo"
        conf = 0.88 + 0.10 * min(1.0, abs(vy[1] - vy[-2]) / 10.0)
        desc = f"Impatto e rimbalzo sul campo (vel: {mean_speed:.1f} px/f)"
    elif delta_y < 18.0 or (delta_y < 26.0 and r2 < 0.35):
        act = ACTION_RASOTERRA
        lbl = "Passaggio Rasoterra"
        conf = 0.82 + 0.14 * (1.0 - min(1.0, delta_y / 26.0))
        desc = f"Pallone aderente al manto erboso (escursione Y: {delta_y:.1f} px)"

    elif (curv_a > 0.04 and r2 > 0.60 and delta_y > 40.0) or (delta_y > 85.0 and r2 > 0.40):
        act = ACTION_LANCIO
        lbl = "Lancio Parabolico in Quota"
        conf = 0.76 + 0.20 * min(1.0, (delta_y / 150.0) * max(0.0, r2))
        desc = f"Arco balistico aereo (escursione: {delta_y:.1f} px, R²={r2:.2f})"
    else:
        act = ACTION_TIRO
        lbl = "Tiro Teso / Passaggio Diretto"
        conf = 0.74 + 0.22 * min(1.0, mean_speed / 25.0)
        desc = f"Traiettoria tesa e diretta (vel media: {mean_speed:.1f} px/f, delta Y: {delta_y:.1f} px)"

    return {
        "action_type": act,
        "label_it": lbl,
        "confidence": float(np.clip(conf, 0.60, 0.98)),
        "delta_y": delta_y,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "r2": r2,
        "curv_a": curv_a,
        "description": desc,
    }



class ShadowMLModel:
    """
    Modello di Machine Learning Physics-Informed per la proiezione dell'ombra al suolo
    e la determinazione dei parametri ottico-geometrici dell'ellisse condizionati dall'azione.
    """

    def __init__(self, weights_path: Optional[Union[str, pathlib.Path]] = None):
        self.weights_path = pathlib.Path(weights_path) if weights_path else WEIGHTS_FILE
        self.is_trained = False

        # Parametri e pesi di default
        self.params: Dict = {
            "version": "2.0_pixel_diff",
            "n_training_samples": 0,
            "n_training_videos": 0,
            "aspect_ratio_mean": 0.313,
            "base_width_scale": 1.85,
            "width_height_ratio": 0.313,
            "base_opacity": 0.65,
            "base_blur": 27,
            "ground_contact_percentile": 90.0,
            # Profili ottico-geometrici specifici per tipo di azione calcistica
            "action_profiles": {
                ACTION_LANCIO: {
                    "aspect_ratio": 0.313,
                    "rx_scale": 2.10,
                    "opacity_factor": 0.85,
                    "blur_offset": 6,
                },
                ACTION_TIRO: {
                    "aspect_ratio": 0.295,
                    "rx_scale": 1.85,
                    "opacity_factor": 1.00,
                    "blur_offset": 0,
                },
                ACTION_RASOTERRA: {
                    "aspect_ratio": 0.350,
                    "rx_scale": 1.50,
                    "opacity_factor": 1.05,
                    "blur_offset": -6,
                },
                ACTION_RIMBALZO: {
                    "aspect_ratio": 0.320,
                    "rx_scale": 1.90,
                    "opacity_factor": 0.95,
                    "blur_offset": 2,
                },
            },
        }

        self.load_weights()

    def load_weights(self) -> bool:
        """Carica i pesi del modello da file JSON se presente."""
        if self.weights_path.exists():
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.params.update(loaded)
                    self.is_trained = True
                    return True
            except Exception:
                self.is_trained = False
        return False

    def save_weights(self, out_path: Optional[Union[str, pathlib.Path]] = None) -> bool:
        """Salva i pesi del modello in un file JSON."""
        target = pathlib.Path(out_path) if out_path else self.weights_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=2)
            return True
        except Exception:
            return False

    def train_on_pixel_diff_dataset(
        self,
        dataset_samples: List[Dict],
        n_videos: int = 0,
    ) -> Dict:
        """
        Addestra il modello sulle coppie estratte per differenza pixel (Pixel Difference Ground Truth).
        Calcola i profili geometrici e ottici condizionati da ciascuna azione.
        """
        if not dataset_samples:
            return {"status": "error", "message": "Nessun campione per l'addestramento"}

        n_samples = len(dataset_samples)

        widths = np.array([s["w"] for s in dataset_samples], dtype=float)
        heights = np.array([s["h"] for s in dataset_samples], dtype=float)
        opacities = np.array([s["opacity"] for s in dataset_samples], dtype=float)
        actions = [s.get("action_type", ACTION_TIRO) for s in dataset_samples]

        valid_wh = (widths > 2.0) & (heights > 1.0)
        aspect_ratios = heights[valid_wh] / widths[valid_wh]
        mean_aspect = float(np.median(aspect_ratios)) if len(aspect_ratios) > 0 else 0.313
        mean_aspect = float(np.clip(mean_aspect, 0.25, 0.45))

        mean_opacity = float(np.median(opacities[opacities > 0.02])) if np.any(opacities > 0.02) else 0.65
        mean_width = float(np.median(widths[valid_wh])) if np.any(valid_wh) else 18.0

        # Calcola profili specifici per azione
        action_profiles = dict(self.params.get("action_profiles", {}))
        for act in [ACTION_LANCIO, ACTION_TIRO, ACTION_RASOTERRA, ACTION_RIMBALZO]:
            act_mask = np.array([a == act for a in actions]) & valid_wh
            if np.sum(act_mask) >= 5:
                act_aspect = float(np.median(heights[act_mask] / widths[act_mask]))
                act_aspect = float(np.clip(act_aspect, 0.25, 0.45))
                act_w = float(np.median(widths[act_mask]))
                scale = float(np.clip(act_w / 14.0, 1.30, 2.50))
                if act not in action_profiles:
                    action_profiles[act] = {}
                action_profiles[act]["aspect_ratio"] = act_aspect
                action_profiles[act]["rx_scale"] = scale

        self.params.update({
            "version": "2.0_pixel_diff",
            "n_training_samples": n_samples,
            "n_training_videos": n_videos,
            "aspect_ratio_mean": mean_aspect,
            "base_width_scale": 1.85,
            "width_height_ratio": mean_aspect,
            "base_opacity": float(np.clip(mean_opacity, 0.45, 0.85)),
            "base_blur": 27,
            "ground_contact_percentile": 90.0,
            "action_profiles": action_profiles,
        })

        self.is_trained = True
        self.save_weights()

        return {
            "status": "ok",
            "n_samples": n_samples,
            "n_videos": n_videos,
            "mean_aspect": mean_aspect,
            "mean_opacity": mean_opacity,
            "mean_width": mean_width,
            "action_profiles": action_profiles,
            "weights_saved_to": str(self.weights_path),
        }

    def predict_ground_trajectory(
        self,
        ball_xs: np.ndarray,
        ball_ys: np.ndarray,
        valid_mask: np.ndarray,
        field_tilt: float = 0.0,
        field_aspect: float = 0.32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predice la traiettoria dell'ombra al suolo (gx(t), gy(t)) e la quota di volo h(t).
        L'ombra è posizionata direttamente sotto il pallone (gx(t) = ball_xs[t]) sul piano del campo da gioco.
        """
        n = len(ball_xs)
        gx = np.copy(ball_xs)
        gy = np.copy(ball_ys)
        heights = np.zeros(n, dtype=float)

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return gx, gy, heights

        first_idx, last_idx = valid_indices[0], valid_indices[-1]
        active_ys = ball_ys[first_idx : last_idx + 1]

        # Quota di base del terreno (piano del campo da gioco)
        # Usiamo il percentile alto della Y per trovare il punto di contatto al suolo,
        # senza forzare un offset minimo che penalizzerebbe le palle a terra.
        p_contact = self.params.get("ground_contact_percentile", 90.0)
        y_ground_baseline = float(np.percentile(active_ys, p_contact))

        # Evita un baseline troppo basso rispetto alla mediana: al massimo 15px sopra la media,
        # mai più del percentile calcolato (al contrario del vecchio +30px fisso)
        y_ground_baseline = max(y_ground_baseline, float(np.median(active_ys)) + 8.0)

        # Adattamento alla pendenza prospettica del campo
        tilt_rad = np.deg2rad(field_tilt)
        mid_x = float(np.mean(ball_xs[valid_indices])) if len(valid_indices) > 0 else 960.0

        # Calcolo del piano del suolo per ciascun fotogramma
        plane_gy = y_ground_baseline + (ball_xs - mid_x) * np.sin(tilt_rad) * 0.30
        # Offset minimo sotto il pallone: almeno 6px (non 15px come prima)
        target_gy = np.maximum(ball_ys + 6.0, plane_gy)


        kernel = np.array([0.15, 0.70, 0.15])
        smooth_gy = np.convolve(target_gy, kernel, mode="same")

        for i in range(n):
            if valid_mask[i]:
                gx[i] = ball_xs[i]
                gy[i] = max(ball_ys[i] + 6.0, smooth_gy[i])
                heights[i] = max(0.0, gy[i] - ball_ys[i])
            else:
                gx[i] = ball_xs[i]
                gy[i] = ball_ys[i]
                heights[i] = 0.0

        return gx, gy, heights

    def predict_shadow_appearance(
        self,
        height: float,
        max_height: float,
        user_shadow_size: int,
        user_shadow_opacity: float,
        field_tilt: float = 0.0,
        field_aspect: float = 0.32,
        action_type: Optional[str] = None,
    ) -> Dict:
        """
        Calcola i parametri ottici ed ellittici per il rendering dell'ombra,
        adattandoli dinamicamente al tipo di azione calcistica riconosciuta.
        """
        h_ratio = float(np.clip(height / max(1.0, max_height), 0.0, 1.0))

        # Recupera il profilo specifico per l'azione
        profiles = self.params.get("action_profiles", {})
        profile = profiles.get(action_type or ACTION_TIRO, {})

        aspect = profile.get("aspect_ratio", self.params.get("aspect_ratio_mean", field_aspect))
        aspect = float(np.clip(aspect, 0.25, 0.45))
        rx_scale = float(profile.get("rx_scale", 1.85))
        opac_factor = float(profile.get("opacity_factor", 1.00))
        blur_offset = int(profile.get("blur_offset", 0))

        # Raggio orizzontale e verticale con penombra
        base_rx = max(1.5, user_shadow_size * rx_scale)
        penumbra_expand = 1.0 + 0.25 * np.sqrt(h_ratio)
        rx = max(1, int(round(base_rx * penumbra_expand)))
        ry = max(1, int(round(rx * aspect)))

        # Opacità (supporta da 0.01 a 1.00)
        opac_base = float(np.clip(user_shadow_opacity * opac_factor, 0.01, 1.0))
        effective_opacity = opac_base * (0.85 + 0.15 * (1.0 - 0.3 * h_ratio))
        effective_opacity = float(np.clip(effective_opacity, 0.01, 1.0))

        # Sfocatura proporzionata alle dimensioni
        blur_val = max(3, int(round(ry * (1.6 + 0.8 * h_ratio))) + blur_offset)
        if blur_val % 2 == 0:
            blur_val += 1

        return {
            "rx": rx,
            "ry": ry,
            "angle": float(field_tilt),
            "opacity": effective_opacity,
            "blur": blur_val,
        }

