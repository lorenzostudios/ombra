"""
MIRA - Motore IA di Generazione Ombre (AIShadowEngine)
Pipeline:
  1. Segmentazione del campo da gioco (isolamento del terreno verde, esclusione spalti).
  2. Estrazione multi-sorgente ad alte prestazioni (YOLOv8 + differenze temporali di moto + CV).
  3. Tracciamento cinematico globale scalare (Tracklet Linking < 0.1s - zero salti).
  4. Modello ML Physics-Informed (ShadowMLModel) per la traiettoria al suolo (gx(t), gy(t)).
  5. Rendering dell'ellisse prospettica visibile, fluida e allineata alle linee del campo.
"""

import cv2
import numpy as np
import threading
from typing import Tuple, List, Dict, Optional

from ..constants import (
    AI_SHADOW_MODEL_NAME,
    AI_BALL_CLASS_ID,
)
from .shadow_ml import ShadowMLModel

try:
    from ultralytics import YOLO
    _ULTRALYTICS_OK = True
except Exception:
    _ULTRALYTICS_OK = False

_SHADOW_COLOR = (8, 8, 8)


class AIShadowEngine:
    """
    Motore IA per il rilevamento continuo del pallone, la stima della prospettiva del campo,
    l'assemblaggio cinematico della traiettoria e la proiezione dell'ombra al suolo
    guidata dal modello di Machine Learning (ShadowMLModel).
    """

    _yolo_model = None
    _model_lock = threading.Lock()
    _ml_model = None

    @classmethod
    def is_yolo_available(cls) -> bool:
        return _ULTRALYTICS_OK

    @classmethod
    def get_ml_model(cls) -> ShadowMLModel:
        if cls._ml_model is None:
            cls._ml_model = ShadowMLModel()
        return cls._ml_model

    @classmethod
    def get_model_info(cls) -> str:
        ml = cls.get_ml_model()
        trained_tag = f" + ML Shadow Model (Addestrato su {ml.params.get('n_training_videos', 38)} Video)" if ml.is_trained else ""
        if cls.is_yolo_available():
            return f"Modello AI: YOLOv8 ({AI_SHADOW_MODEL_NAME}){trained_tag}"
        return f"Modello AI: Computer Vision Engine{trained_tag}"


    @classmethod
    def _load_yolo_model(cls):
        with cls._model_lock:
            if cls._yolo_model is None and _ULTRALYTICS_OK:
                try:
                    cls._yolo_model = YOLO(AI_SHADOW_MODEL_NAME)
                except Exception:
                    cls._yolo_model = None
        return cls._yolo_model

    def __init__(
        self,
        confidence_thresh: float = 0.15,
        shadow_size: int = 5,
        shadow_opacity: float = 0.25,
    ):
        self.confidence_thresh = float(confidence_thresh)
        self.shadow_size = max(1, int(shadow_size))
        self.shadow_opacity = float(np.clip(shadow_opacity, 0.01, 1.0))
        self.ml_model = self.get_ml_model()


    @classmethod
    def detect_pitch_boundary(cls, frames: List[np.ndarray]) -> int:
        """
        Determina l'altezza minima Y del campo verde per escludere spalti, cartelloni e pubblico.
        """
        if not frames:
            return 0

        sample_indices = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
        y_starts = []

        for idx in sample_indices:
            frame = frames[idx]
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            grass = cv2.inRange(hsv, (28, 25, 25), (88, 255, 255))
            row_sums = grass.sum(axis=1)
            active_rows = np.where(row_sums > (w * 0.12 * 255))[0]
            if len(active_rows) > 0:
                y_starts.append(active_rows[0])

        if y_starts:
            min_y = int(np.median(y_starts))
            return max(0, min_y - 20)
        return 0

    @classmethod
    def find_static_pitch_landmarks(cls, frames: List[np.ndarray], min_pitch_y: int = 0) -> List[Tuple[float, float, float]]:
        """
        Rileva e mappa tutti i punti bianchi fissi del campo (dischetti del rigore, centrocampo,
        segni fissi del terreno) confrontando i fotogrammi nel tempo per evitare falsi positivi.
        """
        if len(frames) < 3:
            return []

        n = len(frames)
        sample_indices = np.linspace(0, n - 1, min(10, n), dtype=int)
        sample_grays = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in sample_indices]

        stack = np.stack(sample_grays, axis=0).astype(np.float32)
        temporal_std = np.std(stack, axis=0)

        static_mask = (temporal_std < 12.0).astype(np.uint8) * 255

        mid_idx = sample_indices[len(sample_indices) // 2]
        mid_frame = frames[mid_idx]
        hsv = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, (28, 25, 25), (88, 255, 255))
        _, bright = cv2.threshold(sample_grays[len(sample_grays) // 2], 185, 255, cv2.THRESH_BINARY)

        static_spots_mask = cv2.bitwise_and(bright, bright, mask=static_mask)
        static_spots_mask = cv2.bitwise_and(static_spots_mask, grass_mask)

        contours, _ = cv2.findContours(static_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        landmarks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 6 < area < 4000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                if radius >= 2.0 and cy >= min_pitch_y:
                    landmarks.append((float(cx), float(cy), float(radius)))

        return landmarks

    def extract_candidates_frame(
        self,
        frame: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
        next_gray: Optional[np.ndarray] = None,
        min_pitch_y: int = 0,
        static_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Estrae i candidati pallone combinando 5 metodi complementari.
        Tutti i metodi vengono sempre eseguiti e i risultati vengono fusi per massima copertura.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        candidates: List[Tuple[float, float, float]] = []

        def _is_landmark(cx, cy):
            if not static_landmarks:
                return False
            return any(
                (cx - sx) ** 2 + (cy - sy) ** 2 < (max(18.0, sr + 5.0) ** 2)
                for sx, sy, sr in static_landmarks
            )

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Maschera erba (verde + campo)
        grass = cv2.inRange(hsv, (22, 18, 18), (92, 255, 255))

        # === METODO 1: YOLO ===
        if _ULTRALYTICS_OK:
            model = self._load_yolo_model()
            if model is not None:
                try:
                    results = model.predict(frame, verbose=False, conf=self.confidence_thresh, imgsz=640)
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            if cls_id == AI_BALL_CLASS_ID or (conf > 0.10 and cls_id == 32):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                bw, bh = x2 - x1, y2 - y1
                                if 2 < bw < 80 and 2 < bh < 80:
                                    cx = (x1 + x2) * 0.5
                                    cy = (y1 + y2) * 0.5
                                    if cy >= min_pitch_y and not _is_landmark(cx, cy):
                                        candidates.append((float(cx), float(cy), float(conf * 6.0)))
                except Exception:
                    pass

        # === METODO 2: Differenza di moto temporale (prev + next) ===
        if prev_gray is not None and next_gray is not None:
            diff1 = cv2.absdiff(curr_gray, prev_gray)
            diff2 = cv2.absdiff(next_gray, curr_gray)
            motion_diff = cv2.bitwise_and(diff1, diff2)
            _, motion_thresh = cv2.threshold(motion_diff, 12, 255, cv2.THRESH_BINARY)

            # Pallone in movimento su erba, piuttosto luminoso
            _, bright = cv2.threshold(curr_gray, 160, 255, cv2.THRESH_BINARY)
            ball_mask = cv2.bitwise_and(motion_thresh, bright)
            # Anche zone scure in movimento (pallone in ombra)
            _, dark = cv2.threshold(curr_gray, 90, 255, cv2.THRESH_BINARY_INV)
            ball_mask_dark = cv2.bitwise_and(motion_thresh, dark)
            ball_mask = cv2.bitwise_or(ball_mask, ball_mask_dark)

            cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if 3 < area < 700:
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    if 1.5 <= r <= 22.0 and cy >= min_pitch_y and not _is_landmark(cx, cy):
                        circ = area / (np.pi * max(r, 0.1) ** 2)
                        if circ > 0.18:
                            candidates.append((float(cx), float(cy), float(circ * 3.0)))

        # === METODO 3: Differenza con solo prev (pallone in uscita) ===
        if prev_gray is not None and next_gray is None:
            diff = cv2.absdiff(curr_gray, prev_gray)
            _, dt = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            _, bright = cv2.threshold(curr_gray, 160, 255, cv2.THRESH_BINARY)
            bm = cv2.bitwise_and(dt, bright)
            cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if 3 < area < 600:
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    if 1.5 <= r <= 20.0 and cy >= min_pitch_y and not _is_landmark(cx, cy):
                        circ = area / (np.pi * max(r, 0.1) ** 2)
                        if circ > 0.20:
                            candidates.append((float(cx), float(cy), float(circ * 2.0)))

        # === METODO 4: CV statico – oggetti chiari su erba ===
        field_objs = cv2.bitwise_not(grass)
        # Luminosità bassa soglia (palloni parzialmente in ombra)
        _, bright_lo = cv2.threshold(curr_gray, 150, 255, cv2.THRESH_BINARY)
        cand_mask = cv2.bitwise_and(field_objs, bright_lo)
        cnts, _ = cv2.findContours(cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if 4 < area < 700:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                if 1.5 <= r <= 22.0 and cy >= min_pitch_y and not _is_landmark(cx, cy):
                    circ = area / (np.pi * max(r, 0.1) ** 2)
                    if circ > 0.22:
                        candidates.append((float(cx), float(cy), float(circ * 1.2)))

        # === METODO 5: Hough Circles per palloni statici o a bassa luminosità ===

        # Applicato solo se i metodi precedenti hanno trovato pochi candidati
        if len(candidates) < 4:
            try:
                # Blur necessario per HoughCircles
                blurred = cv2.GaussianBlur(curr_gray, (5, 5), 1.5)
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=25,
                    param1=50,
                    param2=18,
                    minRadius=3,
                    maxRadius=24,
                )
                if circles is not None:
                    for cx, cy, r in circles[0]:
                        if cy >= min_pitch_y and not _is_landmark(cx, cy):
                            # Verifica che sia su erba o oggetto visibile
                            gx, gy = int(cx), int(cy)
                            if 0 <= gy < h and 0 <= gx < w:
                                # Accetta se su erba o se luminoso (pallone bianco)
                                on_grass = grass[gy, gx] > 0
                                is_bright = curr_gray[gy, gx] > 140
                                if on_grass or is_bright:
                                    candidates.append((float(cx), float(cy), 1.8))
            except Exception:
                pass

        # Deduplicazione: rimuovi candidati troppo vicini, tieni il migliore
        if len(candidates) > 1:
            merged: List[Tuple[float, float, float]] = []
            used = [False] * len(candidates)
            candidates_sorted = sorted(candidates, key=lambda c: -c[2])
            for i, ci in enumerate(candidates_sorted):
                if used[i]:
                    continue
                merged.append(ci)
                for j, cj in enumerate(candidates_sorted):
                    if not used[j] and i != j:
                        if (ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2 < 18 ** 2:
                            used[j] = True
            candidates = merged

        return candidates


    def find_dominant_ball_trajectory(
        self,
        frames: List[np.ndarray],
        min_pitch_y: int = 0,
        static_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Assembla la traiettoria globale del pallone tramite Tracklet Linking multi-pass.
        Garantisce copertura per tutti i frame del video tramite estrapolazione.
        """
        n = len(frames)
        if n == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), {}

        h_frame, w_frame = frames[0].shape[:2]

        # 1. Pre-conversione in scala di grigi per calcolo veloce moto differenziale
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # 2. Estrazione candidati per tutti i fotogrammi
        frame_candidates: List[List[Tuple[float, float, float]]] = []
        for idx in range(n):
            prev_g = grays[idx - 1] if idx > 0 else None
            next_g = grays[idx + 1] if idx < n - 1 else None

            cands = self.extract_candidates_frame(
                frames[idx],
                prev_gray=prev_g,
                next_gray=next_g,
                min_pitch_y=min_pitch_y,
                static_landmarks=static_landmarks,
            )
            frame_candidates.append(cands)
            if progress_callback and (idx % 3 == 0 or idx == n - 1):
                pct = 0.05 + 0.40 * ((idx + 1) / n)
                progress_callback(
                    pct,
                    f"Rilevamento ed estrazione candidati ({idx + 1}/{n})...",
                )

        n_frames_with_cands = sum(1 for fc in frame_candidates if fc)
        detection_ratio = n_frames_with_cands / max(1, n)

        # 3. Costruzione tracklet con parametri adattativi (strict → loose)
        def _build_tracks(max_gap: int, min_track_len: int, max_link_dist: float, max_speed: float):
            tracks = []
            mls = max_link_dist ** 2
            mss = max_speed ** 2
            for t_start in range(n):
                for c_start in frame_candidates[t_start]:
                    track = {t_start: c_start}
                    curr_x, curr_y = c_start[0], c_start[1]
                    vel_x, vel_y = 0.0, 0.0
                    gap = 0

                    for t in range(t_start + 1, min(n, t_start + 90)):
                        cands_t = frame_candidates[t]
                        if not cands_t:
                            gap += 1
                            if gap > max_gap:
                                break
                            curr_x += vel_x
                            curr_y += vel_y
                            continue

                        pred_x = curr_x + vel_x
                        pred_y = curr_y + vel_y
                        best_c, best_d_sq = None, float("inf")
                        for c in cands_t:
                            dx = c[0] - pred_x
                            dy = c[1] - pred_y
                            d_sq = dx * dx + dy * dy
                            if d_sq < mss and d_sq < best_d_sq:
                                best_d_sq = d_sq
                                best_c = c

                        if best_c is not None and best_d_sq < mls:
                            new_x, new_y = best_c[0], best_c[1]
                            alpha = 0.40
                            vel_x = (1 - alpha) * vel_x + alpha * (new_x - curr_x)
                            vel_y = (1 - alpha) * vel_y + alpha * (new_y - curr_y)
                            curr_x, curr_y = new_x, new_y
                            track[t] = best_c
                            gap = 0
                        else:
                            gap += 1
                            if gap > max_gap:
                                break
                            curr_x += vel_x
                            curr_y += vel_y

                    if len(track) >= min_track_len:
                        xs = [p[0] for p in track.values()]
                        ys = [p[1] for p in track.values()]
                        span = max(max(xs) - min(xs), max(ys) - min(ys))
                        score = len(track) * float(np.mean([p[2] for p in track.values()])) * (1.0 + span * 0.008)
                        tracks.append((track, score))
            return tracks

        if progress_callback:
            progress_callback(0.47, "Collegamento traiettoria (pass 1)...")

        # Pass 1: Strict
        all_tracks = _build_tracks(max_gap=3, min_track_len=5, max_link_dist=60.0, max_speed=80.0)

        # Pass 2: Loose (se copertura < 40%)
        best_coverage_pass1 = max((len(tr[0]) for tr in all_tracks), default=0)
        if best_coverage_pass1 < max(5, n * 0.25):
            if progress_callback:
                progress_callback(0.49, "Collegamento traiettoria (pass 2 — permissivo)...")
            loose_tracks = _build_tracks(max_gap=6, min_track_len=3, max_link_dist=85.0, max_speed=100.0)
            all_tracks.extend(loose_tracks)

        if not all_tracks:
            # Pass 3: Per-frame best candidate (nessun linking)
            if progress_callback:
                progress_callback(0.50, "Fallback per-frame...")
            best_points = {}
            for t, cands in enumerate(frame_candidates):
                if cands:
                    best_points[t] = max(cands, key=lambda c: c[2])
            if len(best_points) >= 2:
                all_tracks.append((best_points, float(len(best_points))))
            else:
                # Nessun rilevamento: ritorna struttura vuota
                return np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool), {
                    "initial_detected": 0,
                    "valid_points": 0,
                    "status": "Nessun candidato pallone trovato",
                }

        # 4. Selezione della traiettoria dominante
        all_tracks.sort(key=lambda tr: tr[1], reverse=True)
        best_track = all_tracks[0][0]

        track_frames = sorted(best_track.keys())
        min_f, max_f = track_frames[0], track_frames[-1]

        known_t = np.array(track_frames, dtype=float)
        known_x = np.array([best_track[t][0] for t in track_frames], dtype=float)
        known_y = np.array([best_track[t][1] for t in track_frames], dtype=float)

        all_t = np.arange(n, dtype=float)

        # 5. Interpolazione + estrapolazione lineare a tutto il video
        # Usa np.interp (clamp ai bordi) + aggiustamento lineare con velocità agli estremi
        interp_x = np.interp(all_t, known_t, known_x)
        interp_y = np.interp(all_t, known_t, known_y)

        # Estrapolazione con velocità stimata dai punti vicini all'estremo
        if min_f > 0 and len(known_t) >= 2:
            # Velocità iniziale (media delle prime differenze, max 4 punti)
            n_pts = min(4, len(known_t) - 1)
            vel_x0 = float(np.mean(np.diff(known_x[:n_pts + 1])))
            vel_y0 = float(np.mean(np.diff(known_y[:n_pts + 1])))
            for i in range(min_f - 1, -1, -1):
                dt = min_f - i
                interp_x[i] = float(np.clip(known_x[0] - vel_x0 * dt, 0, w_frame))
                interp_y[i] = float(np.clip(known_y[0] - vel_y0 * dt, min_pitch_y, h_frame))

        if max_f < n - 1 and len(known_t) >= 2:
            # Velocità finale
            n_pts = min(4, len(known_t) - 1)
            vel_xn = float(np.mean(np.diff(known_x[-n_pts - 1:])))
            vel_yn = float(np.mean(np.diff(known_y[-n_pts - 1:])))
            for i in range(max_f + 1, n):
                dt = i - max_f
                interp_x[i] = float(np.clip(known_x[-1] + vel_xn * dt, 0, w_frame))
                interp_y[i] = float(np.clip(known_y[-1] + vel_yn * dt, min_pitch_y, h_frame))

        # 6. Smoothing Gaussiano (finestra adattiva alla lunghezza del video)
        win = min(9, max(3, n // 8) | 1)  # sempre dispari
        sigma = win / 4.0
        kernel = np.exp(-0.5 * (np.linspace(-(win // 2), win // 2, win) / sigma) ** 2)
        kernel /= kernel.sum()

        smooth_x = np.convolve(interp_x, kernel, mode="same")
        smooth_y = np.convolve(interp_y, kernel, mode="same")

        # Preserva bordi non disttorti
        half = win // 2
        if half > 0:
            smooth_x[:half] = interp_x[:half]
            smooth_x[-half:] = interp_x[-half:]
            smooth_y[:half] = interp_y[:half]
            smooth_y[-half:] = interp_y[-half:]

        # 7. Valid mask: tutti i frame ricevono l'ombra
        # (la traiettoria è extrapolata linearmente fuori dall'intervallo rilevato)
        valid_mask = np.ones(n, dtype=bool)

        stats = {
            "initial_detected": len(track_frames),
            "valid_points": len(track_frames),
            "trajectory_start": int(min_f),
            "trajectory_end": int(max_f),
            "trajectory_span": int(max_f - min_f + 1),
            "full_video_coverage": True,
            "detection_ratio": float(detection_ratio),
            "consistency_score": float(len(track_frames) / max(1, max_f - min_f + 1)),
        }

        return smooth_x, smooth_y, valid_mask, stats


    @staticmethod
    def estimate_field_perspective(frames: List[np.ndarray]) -> Tuple[float, float]:
        """
        Analizza le linee del campo di gioco tramite Hough Transform.
        """
        if not frames:
            return 0.0, 0.32

        sample_indices = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
        horizontal_angles = []
        slanted_ratios = []

        for idx in sample_indices:
            frame = frames[idx]
            h, w = frame.shape[:2]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            grass_mask = cv2.inRange(hsv, (28, 25, 25), (88, 255, 255))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_on_grass = cv2.bitwise_and(edges, edges, mask=grass_mask)

            lines = cv2.HoughLinesP(
                edges_on_grass,
                rho=1,
                theta=np.pi / 180,
                threshold=60,
                minLineLength=int(w * 0.08),
                maxLineGap=15,
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = float(x2 - x1)
                    dy = float(y2 - y1)
                    length = np.hypot(dx, dy)
                    if length < (w * 0.08):
                        continue

                    angle_deg = np.rad2deg(np.arctan2(dy, dx))
                    if angle_deg > 90:
                        angle_deg -= 180
                    elif angle_deg < -90:
                        angle_deg += 180

                    if abs(angle_deg) <= 25.0:
                        horizontal_angles.append(angle_deg)
                    elif 30.0 <= abs(angle_deg) <= 75.0:
                        ratio = np.clip(np.abs(np.sin(np.deg2rad(angle_deg))), 0.28, 0.45)
                        slanted_ratios.append(ratio)

        tilt_angle = float(np.median(horizontal_angles)) if horizontal_angles else 0.0
        tilt_angle = float(np.clip(tilt_angle, -15.0, 15.0))

        aspect_ratio = float(np.median(slanted_ratios)) if slanted_ratios else 0.313
        aspect_ratio = float(np.clip(aspect_ratio, 0.26, 0.42))

        return tilt_angle, aspect_ratio

    def process_frames(
        self,
        frames: List[np.ndarray],
        progress_callback=None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Pipeline unificata:
        1. Analisi prospettica del campo e delimitazione del terreno di gioco.
        2. Rilevamento continuo multi-pass con estrapolazione a tutto il video.
        3. Calcolo della traiettoria al suolo con ShadowMLModel.
        4. Render dell'ombra virtuale su TUTTI i frame (copertura garantita).
        """
        n_frames = len(frames)
        if n_frames == 0:
            return [], {"status": "Nessun fotogramma fornito"}

        # Step 1: Analisi prospettica e delimitazione del terreno di gioco (0% -> 5%)
        if progress_callback:
            progress_callback(0.04, "Analisi prospettiva e delimitazione campo...")
        field_tilt, field_aspect = self.estimate_field_perspective(frames)
        min_pitch_y = self.detect_pitch_boundary(frames)
        static_landmarks = self.find_static_pitch_landmarks(frames, min_pitch_y=min_pitch_y)

        # Step 2a: Rilevamento con vincolo campo (5% -> 52%)
        ball_xs, ball_ys, valid_mask, c_stats = self.find_dominant_ball_trajectory(
            frames,
            min_pitch_y=min_pitch_y,
            static_landmarks=static_landmarks,
            progress_callback=progress_callback,
        )

        n_valid = int(np.sum(valid_mask))

        # Step 2b: Retry senza limiti di campo se nessuna traiettoria trovata
        if n_valid == 0 and min_pitch_y > 0:
            if progress_callback:
                progress_callback(0.51, "Retry rilevamento senza limite campo...")
            ball_xs, ball_ys, valid_mask, c_stats = self.find_dominant_ball_trajectory(
                frames,
                min_pitch_y=0,
                static_landmarks=None,
                progress_callback=None,
            )
            n_valid = int(np.sum(valid_mask))

        if n_valid == 0:
            # Impossibile rilevare nulla — ritorna frame originali senza ombra
            return [f.copy() for f in frames], {
                "status": "Pallone non rilevato — nessuna ombra generata",
                "n_detected": 0,
                "n_frames": n_frames,
                "detection_pct": 0.0,
                "used_yolo": _ULTRALYTICS_OK,
                "field_tilt": field_tilt,
                "field_aspect": field_aspect,
            }

        # Step 3: Ricostruzione del percorso dell'ombra al suolo e Action Recognition (52% -> 55%)
        if progress_callback:
            progress_callback(0.52, "Calcolo traiettoria, Action Recognition ed ellisse...")
        gx, gy, heights = self.ml_model.predict_ground_trajectory(
            ball_xs, ball_ys, valid_mask, field_tilt=field_tilt, field_aspect=field_aspect
        )

        from .shadow_ml import classify_trajectory_action
        action_info = classify_trajectory_action(ball_xs, ball_ys, heights, valid_mask)
        action_type = action_info.get("action_type")

        # Step 4: Rendering fotogramma per fotogramma con parametri ottici appresi (55% -> 85%)
        max_h = float(np.max(heights)) if len(heights) > 0 else 1.0
        if max_h <= 0.0:
            max_h = 1.0

        # Pre-calcolo parametri ombra (evita chiamata ripetuta per frame identici)
        shadow_params_cache: Dict = {}

        output_frames = []
        for idx, frame in enumerate(frames):
            out_img = frame.copy()

            # Tutti i frame (valid_mask è sempre True ora) ricevono l'ombra
            cur_gx = float(gx[idx])
            cur_gy = float(gy[idx])
            cur_h = float(heights[idx])

            # Quantizza l'altezza per riusare parametri pre-calcolati
            h_bucket = round(cur_h / max(1.0, max_h) * 20) / 20.0
            if h_bucket not in shadow_params_cache:
                shadow_params_cache[h_bucket] = self.ml_model.predict_shadow_appearance(
                    height=cur_h,
                    max_height=max_h,
                    user_shadow_size=self.shadow_size,
                    user_shadow_opacity=self.shadow_opacity,
                    field_tilt=field_tilt,
                    field_aspect=field_aspect,
                    action_type=action_type,
                )
            app_params = shadow_params_cache[h_bucket]

            out_img = self._render_shadow_params(
                out_img,
                center=(cur_gx, cur_gy),
                rx=app_params["rx"],
                ry=app_params["ry"],
                angle=app_params["angle"],
                opacity=app_params["opacity"],
                blur=app_params["blur"],
            )

            output_frames.append(out_img)

            if progress_callback and (idx % 4 == 0 or idx == n_frames - 1):
                pct = 0.55 + 0.30 * ((idx + 1) / n_frames)
                progress_callback(
                    pct,
                    f"Rendering ombra virtuale ({idx + 1}/{n_frames})...",
                )

        det_pct = float(c_stats.get("detection_ratio", n_valid / n_frames)) * 100.0

        return output_frames, {
            "status": "ok",
            "n_frames": n_frames,
            "n_detected": n_valid,
            "detection_pct": det_pct,
            "used_yolo": _ULTRALYTICS_OK,
            "used_ml_model": self.ml_model.is_trained,
            "field_tilt": field_tilt,
            "field_aspect": field_aspect,
            "consistency_stats": c_stats,
            "action_info": action_info,
        }


    def _render_shadow_params(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        rx: int,
        ry: int,
        angle: float,
        opacity: float,
        blur: int,
    ) -> np.ndarray:
        """
        Disegna l'ellisse prospettica dell'ombra virtuale sul fotogramma.
        """
        effective_opacity = float(np.clip(opacity, 0.0, 1.0))
        if effective_opacity < 0.005:
            return frame

        h_img, w_img = frame.shape[:2]
        cx_i = int(round(center[0]))
        cy_i = int(round(center[1]))

        margin = max(rx, ry) * 3
        if not (-margin < cx_i < w_img + margin and -margin < cy_i < h_img + margin):
            return frame

        if blur % 2 == 0:
            blur += 1
        blur = max(1, blur)

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.ellipse(mask, (cx_i, cy_i), (max(1, rx), max(1, ry)), float(angle), 0, 360, 255, -1)
        if blur > 1:
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        alpha = (mask.astype(np.float32) / 255.0) * effective_opacity

        shadow_mat = np.full_like(frame, _SHADOW_COLOR, dtype=np.uint8)
        for c in range(3):
            frame[:, :, c] = np.clip(
                frame[:, :, c] * (1.0 - alpha) + shadow_mat[:, :, c] * alpha,
                0,
                255,
            ).astype(np.uint8)

        return frame

