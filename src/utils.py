"""
MIRA - Funzioni di Utilità Generali
Fornisce funzioni helper per la scansione dei video sul filesystem,
il caricamento asincrono dei frame, il bilanciamento casuale delle sessioni di test
e l'apertura di cartelle/file con il file manager nativo del sistema operativo.
"""

import os
import cv2
import pathlib
import platform
import subprocess

def is_augmented(path: pathlib.Path) -> bool:
    return pathlib.Path(path).stem.startswith("_")


def discover_videos(folder) -> list:
    p = pathlib.Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    return sorted(p.glob("*.mp4"))


def load_video_frames(path):
    """Lettura bloccante: va sempre chiamata da un thread di background."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def video_duration_ms(frames: list, fps: float) -> int:
    return int(len(frames) / max(fps, 1) * 1000)


def _ctk_clear_image(ctk_label):
    """Rimuove DAVVERO l'immagine dal tk.Label interno di un CTkLabel."""
    try:
        ctk_label._label.configure(image="")
    except Exception:
        pass


def open_path(path):
    """Apre un file o una cartella con l'applicazione di sistema predefinita."""
    path = str(path)
    try:
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass


def trial_counter_str(state) -> str:
    n = len(state.session_video_list)
    idx = state.trial_index
    return f"Video {min(idx + 1, n)} di {n}"


def counterpart_video_path(video_path: pathlib.Path) -> pathlib.Path:
    """Percorso della versione 'gemella' di un video (originale<->alterato):
    stesso nome, con o senza il prefisso '_' iniziale. Non verifica che il
    file esista davvero: e' compito del chiamante controllarlo."""
    video_path = pathlib.Path(video_path)
    name = video_path.name
    counterpart_name = name[1:] if name.startswith("_") else f"_{name}"
    return video_path.parent / counterpart_name


def build_balanced_test_session(videos: list, rng, n_orig: int = 5, n_alt: int = 5) -> list:
    """
    Seleziona per il test un insieme bilanciato e casuale di video:
    n_orig video originali e n_alt video alterati, garantendo che non compaiano
    mai contemporaneamente la versione originale e quella alterata dello stesso video clip.
    L'ordine finale di presentazione viene poi mescolato casualmente.
    """
    pairs = {}
    for v in videos:
        v = pathlib.Path(v)
        is_alt = is_augmented(v)
        base = v.name[1:] if is_alt else v.name
        side = "altered" if is_alt else "original"
        pairs.setdefault(base, {})[side] = v

    complete_bases = [b for b, d in pairs.items() if "original" in d and "altered" in d]
    rng.shuffle(complete_bases)

    target_total = n_orig + n_alt
    chosen = []

    if len(complete_bases) >= target_total:
        selected_bases = complete_bases[:target_total]
        rng.shuffle(selected_bases)
        orig_bases = selected_bases[:n_orig]
        alt_bases = selected_bases[n_orig:target_total]
        chosen = [pairs[b]["original"] for b in orig_bases] + [pairs[b]["altered"] for b in alt_bases]
    else:
        chosen_bases = set()
        available_orig = [b for b, d in pairs.items() if "original" in d]
        rng.shuffle(available_orig)
        for b in available_orig:
            if len(chosen) < n_orig:
                chosen.append(pairs[b]["original"])
                chosen_bases.add(b)

        available_alt = [b for b, d in pairs.items() if "altered" in d and b not in chosen_bases]
        rng.shuffle(available_alt)
        alt_chosen = []
        for b in available_alt:
            if len(alt_chosen) < n_alt:
                alt_chosen.append(pairs[b]["altered"])
                chosen_bases.add(b)
        chosen.extend(alt_chosen)

        if len(chosen) < target_total:
            remaining = [v for v in videos if v not in chosen]
            rng.shuffle(remaining)
            chosen.extend(remaining[:target_total - len(chosen)])

    rng.shuffle(chosen)
    return chosen
