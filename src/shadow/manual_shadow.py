"""
MIRA - Strutture Dati ed Interpolazione Ombra Manuale (Manual Shadow Module)
Gestisce i keyframe manuali, l'interpolazione temporale fluida (posizione, dimensioni, angolo,
opacità, sfocatura gaussiana, colore BGR) e il rendering raster dell'ellisse.
"""

import cv2
import math
import bisect
import pathlib
import numpy as np
from dataclasses import dataclass, asdict
from ..constants import MANUAL_PROJECT_SUFFIX, SHADOW_OUTPUT_FOLDER

MANUAL_DEFAULT_COLOR = (8, 8, 8)
MANUAL_DEFAULT_PARAMS = {
    "x": 0.0,
    "y": 0.0,
    "width": 80,
    "height": 28,
    "angle": 0,
    "opacity": 0.45,
    "blur": 31,
    "color": list(MANUAL_DEFAULT_COLOR),
}


@dataclass
class ManualKeyframe:
    """Rappresenta un singolo fotogramma chiave (keyframe) con i parametri geometrici e visivi dell'ombra."""
    frame: int
    x: float
    y: float
    width: float
    height: float
    angle: float
    opacity: float
    blur: int
    color: list

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ManualKeyframe":
        return ManualKeyframe(
            frame=int(d["frame"]),
            x=float(d["x"]),
            y=float(d["y"]),
            width=float(d["width"]),
            height=float(d["height"]),
            angle=float(d["angle"]),
            opacity=float(d["opacity"]),
            blur=int(d["blur"]),
            color=list(d.get("color", MANUAL_DEFAULT_COLOR)),
        )


def _manual_kf_to_params(kf: "ManualKeyframe") -> dict:
    return {
        "x": kf.x,
        "y": kf.y,
        "width": kf.width,
        "height": kf.height,
        "angle": kf.angle,
        "opacity": kf.opacity,
        "blur": kf.blur,
        "color": list(kf.color),
    }


def _manual_lerp_angle(a: float, b: float, t: float) -> float:
    diff = (b - a) % 180
    if diff > 90:
        diff -= 180
    return (a + diff * t) % 180


def manual_get_interpolated_params(
    keyframes: dict, frame_idx: int, easing: bool, fallback_params: dict
) -> dict:
    """
    Calcola i parametri dell'ombra al frame_idx interpolando linearmente o con easing cubico
    tra i due keyframe adiacenti più vicini (ricerca binaria bisect).
    """
    if not keyframes:
        return dict(fallback_params)

    frames_sorted = sorted(keyframes.keys())
    if frame_idx <= frames_sorted[0]:
        return _manual_kf_to_params(keyframes[frames_sorted[0]])
    if frame_idx >= frames_sorted[-1]:
        return _manual_kf_to_params(keyframes[frames_sorted[-1]])

    pos = bisect.bisect_right(frames_sorted, frame_idx) - 1
    fa = frames_sorted[pos]
    fb = frames_sorted[pos + 1]
    if fa == frame_idx:
        return _manual_kf_to_params(keyframes[fa])

    a = keyframes[fa]
    b = keyframes[fb]
    t = (frame_idx - fa) / (fb - fa)
    if easing:
        t = t * t * (3 - 2 * t)

    return {
        "x": a.x + t * (b.x - a.x),
        "y": a.y + t * (b.y - a.y),
        "width": a.width + t * (b.width - a.width),
        "height": a.height + t * (b.height - a.height),
        "angle": _manual_lerp_angle(a.angle, b.angle, t),
        "opacity": a.opacity + t * (b.opacity - a.opacity),
        # Usa round() per un'interpolazione del blur corretta (non troncamento)
        "blur": round(a.blur + t * (b.blur - a.blur)),
        # Interpola i canali colore (BGR) tra i due keyframe
        "color": [
            int(round(a.color[i] + t * (b.color[i] - a.color[i])))
            for i in range(len(a.color))
        ],
    }


def manual_draw_shadow(frame: np.ndarray, params: dict) -> np.ndarray:
    """Disegna l'ellisse dell'ombra su una copia del frame applicando sfocatura e alpha blending."""
    img = frame.copy()
    h, w = img.shape[:2]

    opacity = float(np.clip(params["opacity"], 0.0, 1.0))
    if opacity <= 0:
        return img

    cx, cy = params["x"], params["y"]
    axes = (
        max(1, int(round(params["width"] / 2))),
        max(1, int(round(params["height"] / 2))),
    )
    angle = params["angle"]
    blur = int(params["blur"])
    if blur % 2 == 0:
        blur += 1
    blur = max(1, blur)
    color = params.get("color", MANUAL_DEFAULT_COLOR)

    margin = max(axes) + blur
    if cx < -margin or cx > w + margin or cy < -margin or cy > h + margin:
        return img

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), axes, angle, 0, 360, 255, -1)
    if blur > 1:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    alpha = (mask.astype(np.float32) / 255.0) * opacity
    shadow_layer = np.full_like(img, color, dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c] * (1 - alpha) + shadow_layer[:, :, c] * alpha, 0, 255
        ).astype(np.uint8)
    return img


def _manual_rotate_point(local_x: float, local_y: float, angle_deg: float):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return local_x * c - local_y * s, local_x * s + local_y * c


def _manual_signed_angle(angle_deg: float) -> float:
    a = angle_deg % 180
    return a if a <= 90 else a - 180


def manual_handle_positions(params: dict):
    cx, cy = params["x"], params["y"]
    aw, ah = params["width"] / 2.0, params["height"] / 2.0
    angle = _manual_signed_angle(params["angle"])
    max_r = max(aw, ah)

    rx, ry = _manual_rotate_point(aw, 0, angle)
    resize_w = (cx + rx, cy + ry)

    rx, ry = _manual_rotate_point(0, ah, angle)
    resize_h = (cx + rx, cy + ry)

    rx, ry = _manual_rotate_point(0, -(max_r + 28), angle)
    rotate = (cx + rx, cy + ry)

    return {"resize_w": resize_w, "resize_h": resize_h, "rotate": rotate}


def manual_project_candidates(video_path) -> list:
    if not video_path:
        return []
    video_path = pathlib.Path(video_path)
    stem = video_path.stem
    names = [f"{stem}{MANUAL_PROJECT_SUFFIX}", f"{stem}.json"]
    folders = [video_path.parent, pathlib.Path("."), SHADOW_OUTPUT_FOLDER]
    out, seen = [], set()
    for folder in folders:
        for n in names:
            p = folder / n
            key = str(p.resolve()) if p.parent.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def manual_find_project(video_path):
    for p in manual_project_candidates(video_path):
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None
