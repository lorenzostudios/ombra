"""
MIRA - Geometria Proiettiva e Omografia Campo (Auto Shadow Module)
Fornisce funzioni matematiche e di computer vision per:
  - Calcolo e validazione dell'omografia planare campo <-> immagine (H, H_inv).
  - Proiezione prospettica di una conica circolare dal piano campo al piano camera (H^T C H).
  - Tracciamento ottico Lucas-Kanade forward-backward con estrapolazione della velocità.
  - Rendering dell'ombra ellittica sfumata (alpha blending e maschera di visibilità).
"""

import cv2
import numpy as np

# Colori BGR per il disegno diretto sui pixel del frame (annotazioni video)
SHADOW_UI_ACCENT_BGR = (0, 200, 120)
SHADOW_WARN_BGR = (40, 140, 255)
SHADOW_DONE_BGR = (80, 210, 80)
SHADOW_WHITE_BGR = (240, 240, 240)
SHADOW_MUTED_BGR = (140, 140, 150)
SHADOW_ELLIPSE_COLOR = (8, 8, 8)
SHADOW_ELLIPSE_ALPHA = 0.42
SHADOW_SMOOTH_VIS = 0.20
SHADOW_SMOOTH_POS = 0.7

SHADOW_PHASE_COLORS_BGR = {
    "CALIB": (200, 80, 200),
    "GRASS": (60, 200, 60),
    "LINES": (60, 200, 200),
    "START": (60, 180, 255),
    "END": (40, 140, 255),
    "READY": (80, 210, 80),
}

SHADOW_ZOOM_SIZE = 160
SHADOW_ZOOM_FACTOR = 4
SHADOW_FIELD_W = 400
SHADOW_FIELD_H = 400

SHADOW_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
SHADOW_FB_ERR_MAX = 5.0

SHADOW_DST_PTS = np.float32(
    [[0, 0], [SHADOW_FIELD_W, 0], [SHADOW_FIELD_W, SHADOW_FIELD_H], [0, SHADOW_FIELD_H]]
)


def _shadow_safe_patch(frame, x, y, r=2):
    h, w = frame.shape[:2]
    return frame[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)]


def _shadow_is_H_valid(H, src_pts, img_w, img_h):
    if H is None or not np.all(np.isfinite(H)):
        return False
    margin = 0.40
    mx, my = img_w * margin, img_h * margin
    for px, py in src_pts:
        if not (-mx < px < img_w + mx and -my < py < img_h + my):
            return False
    hull = cv2.convexHull(src_pts.reshape(-1, 1, 2))
    if len(hull) < 4:
        return False
    area = cv2.contourArea(src_pts.reshape(-1, 1, 2))
    if area < img_w * img_h * 0.01:
        return False
    return True


def shadow_compute_homography_tracked(calib_histories, frame_idx, img_shape=None):
    """
    Calcola l'omografia H (pixel -> coordinate campo) e la sua inversa H_inv
    al fotogramma frame_idx partendo dalle traiettorie tracciate dei 4 punti di calibrazione.
    """
    src = np.float32([ch[frame_idx] for ch in calib_histories])
    H, _ = cv2.findHomography(src, SHADOW_DST_PTS, cv2.RANSAC, 3.0)
    if H is None:
        return None, None
    if img_shape is not None:
        img_h, img_w = img_shape[:2]
        if not _shadow_is_H_valid(H, src, img_w, img_h):
            return None, None
    try:
        H_inv = np.linalg.inv(H)
        if not np.all(np.isfinite(H_inv)):
            return None, None
        return H, H_inv
    except np.linalg.LinAlgError:
        return None, None


def shadow_img_to_field(pt, H):
    """Trasforma un punto dalle coordinate immagine (pixel) alle coordinate campo (rettangolo piano)."""
    r = cv2.perspectiveTransform(
        np.array([[[float(pt[0]), float(pt[1])]]], np.float32), H
    )
    return r[0][0]


def shadow_field_to_img(pt, H_inv):
    """Trasforma un punto dalle coordinate campo (piano) alle coordinate immagine (pixel prospettici)."""
    r = cv2.perspectiveTransform(
        np.array([[[float(pt[0]), float(pt[1])]]], np.float32), H_inv
    )
    return r[0][0]


def _shadow_circle_to_conic(cx, cy, r):
    return np.array(
        [[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx * cx + cy * cy - r * r]],
        dtype=np.float64,
    )


def shadow_compute_ellipse_params(ball_img, H, H_inv, shadow_size):
    """
    Calcola la conica proiettata (ellisse) sul piano immagine:
    un cerchio perfetto definito sul campo viene retroproiettato mediante C_img = H^T * C_field * H.
    Estrae centro, semiassi (a, b) e angolo di rotazione in gradi.
    """
    ball_f = shadow_img_to_field(ball_img, H)
    cx_f, cy_f = float(ball_f[0]), float(ball_f[1])

    C_field = _shadow_circle_to_conic(cx_f, cy_f, float(shadow_size))
    Hd = H.astype(np.float64)
    C_img = Hd.T @ C_field @ Hd

    A = C_img[0, 0]
    B = 2 * C_img[0, 1]
    C_ = C_img[1, 1]
    D = 2 * C_img[0, 2]
    E = 2 * C_img[1, 2]

    M = np.array([[2 * A, B], [B, 2 * C_]])
    if abs(np.linalg.det(M)) < 1e-10:
        return None
    try:
        center = np.linalg.solve(M, np.array([-D, -E]))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(center)):
        return None

    det_quad = 4 * A * C_ - B * B
    if det_quad <= 0:
        return None
    det_C = np.linalg.det(C_img)
    if det_C == 0:
        return None

    Q = np.array([[A, B / 2], [B / 2, C_]])
    eigenvalues, eigvecs = np.linalg.eigh(Q)
    if np.any(eigenvalues <= 0):
        return None

    # det_quad = 4*det(Q) poiché B = 2*C_img[0,1], quindi:
    # scale = -det_C / det_quad = -det_C / (4*det(Q))
    # semiaxes_correct = sqrt(-det_C / (det(Q) * λ)) = sqrt(scale * 4 / λ)
    # Correggiamo moltiplicando per 4 per ottenere la dimensione prospettica corretta.
    scale = -det_C / det_quad * 4.0
    if scale <= 0:
        return None
    semiaxes = np.sqrt(scale / eigenvalues)
    if not np.all(np.isfinite(semiaxes)) or np.any(semiaxes <= 0):
        return None

    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = float(np.rad2deg(angle_rad))

    ax_a = max(1, int(round(semiaxes[0])))
    ax_b = max(1, int(round(semiaxes[1])))
    return center, (ax_a, ax_b), angle_deg


def shadow_draw_ellipse(img, center, axes, angle, alpha_mult=1.0):
    h, w = img.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    ax_a, ax_b = int(round(axes[0])), int(round(axes[1]))
    ax_a, ax_b = max(1, ax_a), max(1, ax_b)
    if not (-ax_a * 4 < cx < w + ax_a * 4 and -ax_b * 4 < cy < h + ax_b * 4):
        return img
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax_a, ax_b), float(angle), 0, 360, 255, -1)
    blur_k = max(3, (max(ax_a, ax_b) // 2) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    alpha = (mask / 255.0) * SHADOW_ELLIPSE_ALPHA * float(np.clip(alpha_mult, 0.0, 1.0))
    shadow = np.full_like(img, SHADOW_ELLIPSE_COLOR, dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c] * (1.0 - alpha) + shadow[:, :, c] * alpha, 0, 255
        ).astype(np.uint8)
    return img


def shadow_get_visibility(frame, center, axes, grass_avg, line_avg):
    h, w = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(max(axes)))
    roi = frame[max(0, cy - r) : min(h, cy + r), max(0, cx - r) : min(w, cx + r)]
    if roi.size == 0:
        return 1.0
    dg = np.linalg.norm(roi.astype(float) - grass_avg, axis=2)
    dl = np.linalg.norm(roi.astype(float) - line_avg, axis=2)
    return float(np.clip(np.mean((dg < 50) | (dl < 60)) * 1.15, 0.0, 1.0))


def shadow_track_ball_forward(frames, start_idx, start_pos, end_idx):
    positions = {start_idx: tuple(start_pos)}
    curr = np.array([[list(start_pos)]], dtype=np.float32)
    last_good = curr.copy()
    # Stima della velocità corrente basata sulle ultime 2 posizioni buone
    last_pos = np.array(start_pos, dtype=np.float32)
    prev_pos = np.array(start_pos, dtype=np.float32)

    for i in range(start_idx + 1, end_idx + 1):
        pg = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        ng = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(pg, ng, curr, None, **SHADOW_LK_PARAMS)
        if st_f is None or not st_f[0][0]:
            # Fallback: estrapolazione lineare dalla velocità stimata
            velocity = last_pos - prev_pos
            extrapolated = last_pos + velocity
            curr = np.array([[[extrapolated[0], extrapolated[1]]]], dtype=np.float32)
            positions[i] = (float(extrapolated[0]), float(extrapolated[1]))
            continue
        bwd, st_b, _ = cv2.calcOpticalFlowPyrLK(ng, pg, fwd, None, **SHADOW_LK_PARAMS)
        # Soglia FB più stretta per tracking più preciso (era 5.0)
        if st_b is not None and st_b[0][0]:
            if float(np.linalg.norm(curr[0][0] - bwd[0][0])) <= SHADOW_FB_ERR_MAX:
                new_pos = fwd[0][0]
                # Aggiorna stima velocità solo su tracking confermato
                prev_pos = last_pos.copy()
                last_pos = new_pos.copy()
                curr = fwd
                last_good = fwd.copy()
            else:
                # Flow inaffidabile: estrapolazione dalla velocità
                velocity = last_pos - prev_pos
                extrapolated = last_pos + velocity
                curr = np.array([[[extrapolated[0], extrapolated[1]]]], dtype=np.float32)
                new_pos = extrapolated
        else:
            new_pos = fwd[0][0]
            prev_pos = last_pos.copy()
            last_pos = new_pos.copy()
            curr = fwd
            last_good = fwd.copy()
        positions[i] = (float(curr[0][0][0]), float(curr[0][0][1]))
    return positions


def shadow_track_calib_full(frames, start_pos):
    return shadow_track_ball_forward(frames, 0, start_pos, len(frames) - 1)


def shadow_get_zoom_patch(frame, x, y):
    h, w = frame.shape[:2]
    ps = SHADOW_ZOOM_SIZE // SHADOW_ZOOM_FACTOR
    half = ps // 2
    x1 = max(0, min(x - half, w - ps))
    y1 = max(0, min(y - half, h - ps))
    zoom = cv2.resize(
        frame[y1 : y1 + ps, x1 : x1 + ps].copy(),
        (SHADOW_ZOOM_SIZE, SHADOW_ZOOM_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    mid = SHADOW_ZOOM_SIZE // 2
    cv2.line(zoom, (mid, mid - 9), (mid, mid + 9), (0, 0, 220), 1, cv2.LINE_AA)
    cv2.line(zoom, (mid - 9, mid), (mid + 9, mid), (0, 0, 220), 1, cv2.LINE_AA)
    cv2.circle(zoom, (mid, mid), 3, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.rectangle(
        zoom,
        (0, 0),
        (SHADOW_ZOOM_SIZE - 1, SHADOW_ZOOM_SIZE - 1),
        SHADOW_UI_ACCENT_BGR,
        1,
    )
    return zoom


def _shadow_put(img, text, pos, scale=0.5, color=SHADOW_WHITE_BGR, thickness=1):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, f, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, f, scale, color, thickness, cv2.LINE_AA)


def _bgr_to_hex(bgr) -> str:
    b, g, r = [int(round(float(c))) for c in bgr]
    b, g, r = (max(0, min(255, v)) for v in (b, g, r))
    return f"#{r:02x}{g:02x}{b:02x}"
