"""
MIRA - Addestramento avanzato del modello di ombra virtuale basato sulla differenza pixel
tra i 38 video originali ('WorldCupXXXX.mp4') e i video modificati con ombra ('_WorldCupXXXX.mp4').
Integra l'Action Recognition per classificare le giocate balistiche e calcola i pesi ottici ed ellittici.
"""

from __future__ import annotations

import pathlib
import time
import cv2
import numpy as np

from .ai_shadow import AIShadowEngine
from .shadow_ml import (
    ShadowMLModel,
    WEIGHTS_FILE,
    classify_trajectory_action,
    ACTION_TIRO,
)
from ..constants import BASE_DIR


def extract_pixel_diff_dataset_and_train(
    videos_dir: pathlib.Path = BASE_DIR / "videos",
    output_weights: pathlib.Path = WEIGHTS_FILE,
    verbose: bool = True,
) -> dict:
    """
    Estrae il dataset confrontando pixel per pixel i video originali e modificati.
    Addestra il modello ShadowMLModel con condizionamento per tipo di azione.
    """
    videos_dir = pathlib.Path(videos_dir)
    output_weights = pathlib.Path(output_weights)

    if verbose:
        print("=" * 70)
        print(" ADDESTRAMENTO MODELLO IA: DIFFERENZA PIXEL & ACTION RECOGNITION")
        print(" (Supervisione densa tra coppie di video originali e modificati)")
        print("=" * 70)

    clean_files = sorted([f for f in videos_dir.glob("WorldCup*.mp4") if not f.name.startswith("_")])
    video_pairs = []
    for c in clean_files:
        alt = videos_dir / f"_{c.name}"
        if alt.exists():
            video_pairs.append((c, alt))

    if not video_pairs:
        print(f"Nessuna coppia di video trovata in: {videos_dir}")
        return {"status": "error", "message": "Nessuna coppia trovata"}

    if verbose:
        print(f"Trovate {len(video_pairs)} coppie complete di video (Originale <-> Modificato).")
        print("Estrazione Ground Truth per differenza pixel ed Action Recognition...")

    engine = AIShadowEngine(confidence_thresh=0.10)
    dataset_samples = []
    video_stats = []
    t_start = time.time()

    for idx, (clean_path, alt_path) in enumerate(video_pairs, 1):
        cap_clean = cv2.VideoCapture(str(clean_path))
        frames_clean = []
        while True:
            ret, frame = cap_clean.read()
            if not ret:
                break
            frames_clean.append(frame)
        cap_clean.release()

        cap_alt = cv2.VideoCapture(str(alt_path))
        frames_alt = []
        while True:
            ret, frame = cap_alt.read()
            if not ret:
                break
            frames_alt.append(frame)
        cap_alt.release()

        n_frames = min(len(frames_clean), len(frames_alt))
        if n_frames == 0:
            continue

        # 1. Delimitazione campo e prospettiva
        field_tilt, field_aspect = AIShadowEngine.estimate_field_perspective(frames_clean)
        min_pitch_y = AIShadowEngine.detect_pitch_boundary(frames_clean)
        static_landmarks = AIShadowEngine.find_static_pitch_landmarks(frames_clean, min_pitch_y=min_pitch_y)

        # 2. Tracciamento traiettoria del pallone
        ball_xs, ball_ys, valid_mask, _ = engine.find_dominant_ball_trajectory(
            frames_clean, min_pitch_y=min_pitch_y, static_landmarks=static_landmarks
        )

        gx, gy, heights = engine.ml_model.predict_ground_trajectory(
            ball_xs, ball_ys, valid_mask, field_tilt=field_tilt, field_aspect=field_aspect
        )

        # 3. Classificazione dell'Azione Balistica
        action_info = classify_trajectory_action(ball_xs, ball_ys, heights, valid_mask)
        action_type = action_info["action_type"]

        # 4. Estrazione delle maschere ombra per differenza pixel
        samples_in_video = 0
        for f_idx in range(n_frames):
            f_clean = frames_clean[f_idx]
            f_alt = frames_alt[f_idx]

            diff = cv2.absdiff(f_clean, f_alt)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(diff_gray, 8, 255, cv2.THRESH_BINARY)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 8:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]

                        if len(c) >= 5:
                            (ex, ey), (ew, eh), angle = cv2.fitEllipse(c)
                        else:
                            ex, ey, ew, eh, angle = cx, cy, 20.0, 7.0, 0.0

                        # Misurazione opacità effettiva
                        clean_pixels = f_clean[mask > 0].astype(float)
                        alt_pixels = f_alt[mask > 0].astype(float)
                        mean_clean = np.mean(clean_pixels) if len(clean_pixels) > 0 else 100.0
                        mean_alt = np.mean(alt_pixels) if len(alt_pixels) > 0 else 50.0
                        opacity = float(np.clip(1.0 - (mean_alt / max(1.0, mean_clean)), 0.10, 0.95))

                        sample = {
                            "video": clean_path.name,
                            "frame": f_idx,
                            "ball_x": float(ball_xs[f_idx]),
                            "ball_y": float(ball_ys[f_idx]),
                            "shadow_x": float(cx),
                            "shadow_y": float(cy),
                            "w": float(max(ew, eh)),
                            "h": float(min(ew, eh)),
                            "angle": float(angle),
                            "opacity": float(opacity),
                            "area": float(area),
                            "action_type": action_type,
                            "field_tilt": float(field_tilt),
                            "field_aspect": float(field_aspect),
                        }
                        dataset_samples.append(sample)
                        samples_in_video += 1

        video_stats.append({
            "video": clean_path.name,
            "n_frames": n_frames,
            "samples": samples_in_video,
            "action": action_info["label_it"],
            "action_type": action_type,
        })

        if verbose:
            print(
                f"[{idx:02d}/{len(video_pairs):02d}] {clean_path.name:<22} -> "
                f"{samples_in_video:3d} frame estratti | Azione: {action_info['label_it']:<28}"
            )

    elapsed = time.time() - t_start
    if verbose:
        print("-" * 70)
        print(f"Estrazione completata in {elapsed:.1f}s. Totale campioni pixel: {len(dataset_samples)}")
        print("Calcolo pesi ottici condizionati dall'azione (ShadowMLModel)...")

    # Addestramento modello ML
    ml_model = ShadowMLModel(weights_path=output_weights)
    train_results = ml_model.train_on_pixel_diff_dataset(dataset_samples, n_videos=len(video_stats))

    if verbose:
        print("=" * 70)
        print(" RISULTATI DELL'ADDESTRAMENTO PIXEL-DIFFERENCE")
        print("=" * 70)
        print(f"• Video analizzati:             {train_results.get('n_videos')}")
        print(f"• Campioni temporali pixel:     {train_results.get('n_samples')}")
        print(f"• Aspect Ratio medio appreso:   {train_results.get('mean_aspect'):.3f}")
        print(f"• Opacità media appresa:        {train_results.get('mean_opacity') * 100:.1f}%")
        print("• Profili per Azione appresi:")
        for act, prof in train_results.get("action_profiles", {}).items():
            print(f"   - {act:<20}: aspect={prof.get('aspect_ratio', 0.31):.3f}, rx_scale={prof.get('rx_scale', 1.8):.2f}x")
        print(f"• Pesi serializzati in:         {output_weights.name}")
        print("=" * 70)

    return train_results


if __name__ == "__main__":
    extract_pixel_diff_dataset_and_train()
