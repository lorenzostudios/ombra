"""
OMBRA - Motore di Acquisizione Vocale e Riconoscimento (Whisper)
Gestisce la registrazione audio asincrona dal microfono (sounddevice/soundfile),
il preprocessing del segnale (rimozione DC offset, peak normalization),
il calcolo preciso dell'istante di onset vocale (RMS energy peak onset in NumPy)
e la trascrizione rapida deterministica delle risposte in italiano tramite OpenAI Whisper.
"""

import threading
import unicodedata
import numpy as np
from .constants import (
    VOICE_AVAILABLE,
    VOICE_SAMPLE_RATE,
    WHISPER_MODEL_NAME,
    _SOUNDDEVICE_OK,
    _WHISPER_OK,
)

if _SOUNDDEVICE_OK:
    import sounddevice as sd
    import soundfile as sf

if _WHISPER_OK:
    import whisper as _whisper_lib


# ─────────────────────────────────────────────────────────────────────────────
# Parametri trascrizione ottimizzati per italiano breve (risposte vocali)
# ─────────────────────────────────────────────────────────────────────────────
_WHISPER_DECODE_OPTIONS = dict(
    # Whisper deve fare meno supposizioni sul testo precedente:
    # ogni risposta è autonoma e breve (1-3 parole).
    condition_on_previous_text=False,
    # Decodifica deterministica ultra-rapida (greedy search): risposta istantanea (<0.3s)
    beam_size=1,
    best_of=1,
    # Soglia bassa per silence: meglio trascrivere il silenzio che ignorare
    # parole brevi dette piano (tipo "sì").
    no_speech_threshold=0.4,
    # Compressione: soglia alta per non scartare audio breve e rapido.
    compression_ratio_threshold=2.8,
    # Temperatura: solo 0 (greedy deterministic) — le risposte sono cortissime
    # e non beneficiano di campionamento stocastico.
    temperature=0.0,
    # Prompt iniziale che orienta Whisper verso le parole attese in italiano.
    # Include varianti comuni per "sì" e "no" in italiano sportivo.
    initial_prompt=(
        "L'utente risponde brevemente con una parola o due in italiano. "
        "Possibili risposte: sì, no, certo, assolutamente, niente, "
        "sicuro, esatto, sbagliato, destra, sinistra, dentro, fuori, "
        "gol, fallo, rigore, tocco, contatto, normale."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern allucinazioni note di Whisper su silenzio / audio a basso volume
# ─────────────────────────────────────────────────────────────────────────────
_HALLUCINATION_PATTERNS = [
    "sottotitoli",
    "sottotitolo",
    "qtss",
    "amara.org",
    "opensubtitles",
    "subfactory",
    "traduzione a cura",
    "sincronizzazione a cura",
    "revisione a cura",
    "trascrizione a cura",
    "grazie per la visione",
    "grazie per aver guardato",
    "iscriviti al canale",
    "iscriviti",
    "lascia un like",
    "tutti i diritti riservati",
    "buona visione",
    "alla prossima",
    "watching",
    "subscribed",
]


def _is_hallucination(text: str) -> bool:
    """Verifica se il testo trascritto è una tipica allucinazione di sottotitoli di Whisper."""
    if not text:
        return False
    t_clean = text.lower().strip()
    # Se contiene uno dei pattern di allucinazione noti
    if any(pat in t_clean for pat in _HALLUCINATION_PATTERNS):
        # Verifica se l'utente ha comunque pronunciato una parola valida per il task
        valid_words = {
            "si",
            "sì",
            "no",
            "yes",
            "destra",
            "sinistra",
            "dentro",
            "fuori",
            "propria",
            "avversaria",
            "portiere",
            "giocatore",
            "bianco",
            "bianca",
            "rosso",
            "rossa",
            "blu",
            "azzurra",
            "gialla",
            "gol",
            "testa",
        }
        tokens = {_normalize_word(w) for w in t_clean.split() if _normalize_word(w)}
        if not (tokens & valid_words):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Normalizzazione e matching
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_word(w: str) -> str:
    """Minuscolo, senza accenti diacritici, senza punteggiatura ai bordi."""
    w = w.strip().lower()
    # Rimuovi diacritici (à→a, è→e, ì→i, ecc.)
    w = "".join(
        c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn"
    )
    # Rimuovi apostrofi, punteggiatura e spazi residui
    w = w.strip(" .,!?;:\"'«»()[]")
    # Collassa l'apostrofo italiano: "si'" → "si", "po'" → "po"
    w = w.replace("'", "").replace("`", "").replace("'", "")
    return w.strip()


def _levenshtein_distance(a: str, b: str) -> int:
    """Distanza di Levenshtein tra due stringhe brevi."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        new_row = [i + 1]
        for j, cb in enumerate(b):
            new_row.append(
                min(
                    row[j + 1] + 1,
                    new_row[j] + 1,
                    row[j] + (0 if ca == cb else 1),
                )
            )
        row = new_row
    return row[-1]


def _fuzzy_match(word: str, target: str, max_dist: int = 1) -> bool:
    """True se `word` corrisponde a `target` con al massimo `max_dist` errori."""
    if not word or not target:
        return False
    if word == target:
        return True
    # Non usare fuzzy su parole molto brevi (troppi falsi positivi)
    if len(target) < 3 or len(word) < 2:
        return False
    return _levenshtein_distance(word, target) <= max_dist


def match_voice_answer(transcript_text, words, label_yes, label_no):
    """Cerca nelle parole trascritte (con timestamp) la prima corrispondenza
    con l'etichetta SI o NO configurata per il video corrente.

    Strategia multi-livello:
      1. Match esatto sulla parola con timestamp
      2. Match fuzzy (Levenshtein ≤ 1) sulla parola con timestamp
      3. Match esatto sul testo completo (senza timestamp)
      4. Match tramite alias italiani comuni
    """
    ly = _normalize_word(label_yes or "")
    lno = _normalize_word(label_no or "")
    if not ly and not lno:
        return None, None

    # Alias per il caso classico SI/NO italiano
    yes_aliases = {
        "si",
        "si'",
        "sì",
        "yes",
        "gia",
        "già",
        "esatto",
        "esattamente",
        "affermativo",
        "certo",
        "certamente",
        "sicuro",
        "sicuramente",
        "assolutamente",
        "confermo",
        "giusto",
        "corretto",
        "ok",
        "dai",
    }
    no_aliases = {
        "no",
        "nope",
        "non",
        "negativo",
        "niente",
        "mai",
        "sbagliato",
        "errato",
        "falso",
        "negato",
    }

    # Determina se stiamo usando le etichette classiche SI/NO
    is_si_no = (ly in ("si", "yes") or not ly) and (lno in ("no",) or not lno)

    # ── Pass 1 & 2: parole con timestamp ────────────────────────────────────
    for w in words:
        nw = _normalize_word(w.get("word", ""))
        if not nw:
            continue
        t_start = w.get("start")

        # Match esatto
        if ly and nw == ly:
            return "yes", t_start
        if lno and nw == lno:
            return "no", t_start

        # Alias SI/NO
        if is_si_no:
            if nw in yes_aliases:
                return "yes", t_start
            if nw in no_aliases:
                return "no", t_start

        # Sottostringa bidirezionale (etichette ≥ 3 char)
        if ly and len(ly) >= 3 and len(nw) >= 3:
            if nw in ly or ly in nw:
                return "yes", t_start
        if lno and len(lno) >= 3 and len(nw) >= 3:
            if nw in lno or lno in nw:
                return "no", t_start

        # Match fuzzy per errori di trascrizione (es. "sì" → "si", "si'" → "si")
        if _fuzzy_match(nw, ly):
            return "yes", t_start
        if _fuzzy_match(nw, lno):
            return "no", t_start

    # ── Pass 3: testo completo senza timestamp ────────────────────────────────
    nt_words = [
        _normalize_word(w)
        for w in (transcript_text or "").split()
        if _normalize_word(w)
    ]
    for w in nt_words:
        if ly and w == ly:
            return "yes", None
        if lno and w == lno:
            return "no", None
        if is_si_no:
            if w in yes_aliases:
                return "yes", None
            if w in no_aliases:
                return "no", None
        if ly and len(ly) >= 3 and len(w) >= 3:
            if w in ly or ly in w:
                return "yes", None
        if lno and len(lno) >= 3 and len(w) >= 3:
            if w in lno or lno in w:
                return "no", None
        if _fuzzy_match(w, ly):
            return "yes", None
        if _fuzzy_match(w, lno):
            return "no", None

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing audio
# ─────────────────────────────────────────────────────────────────────────────


def _preprocess_audio(audio: np.ndarray, sr: int = VOICE_SAMPLE_RATE) -> np.ndarray:
    """Preprocessing leggero per migliorare la trascrizione:
    1. Rimuovi DC offset
    2. Normalizzazione ampiezza (peak normalization)
    3. Trim del silenzio iniziale/finale (usando soglia energia RMS)
    4. Padding minimo (Whisper si aspetta almeno ~1s di audio)
    """
    if audio is None or audio.size == 0:
        return audio

    # 1. Rimuovi DC offset
    audio = audio - np.mean(audio)

    # 2. Normalizzazione picco (evita clipping, massimizza SNR)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    # 3. Trim silenzio: rimuovi sezioni con energia RMS < soglia
    win = max(1, int(sr * 0.02))  # finestra 20ms
    hop = max(1, win // 2)
    n_wins = (len(audio) - win) // hop + 1
    if n_wins > 1:
        rms = np.array(
            [
                np.sqrt(np.mean(audio[i * hop : i * hop + win] ** 2))
                for i in range(n_wins)
            ]
        )
        threshold = max(0.01, np.max(rms) * 0.05)
        active = np.where(rms >= threshold)[0]
        if len(active) > 0:
            # Mantieni un margine di 50ms prima e dopo
            margin = max(1, int(sr * 0.05) // hop)
            start_win = max(0, active[0] - margin)
            end_win = min(n_wins - 1, active[-1] + margin)
            start_sample = start_win * hop
            end_sample = min(len(audio), end_win * hop + win)
            audio = audio[start_sample:end_sample]

    # 4. Padding minimo a 1 secondo (Whisper lavora meglio con almeno 1s)
    min_len = int(sr * 1.0)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)), mode="constant")

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Engine principale
# ─────────────────────────────────────────────────────────────────────────────


class VoiceAnswerEngine:
    """Incapsula registrazione microfono (sounddevice), rilevazione
    dell'istante di inizio del parlato (librosa) e trascrizione (whisper)."""

    _model = None
    _model_lock = threading.Lock()

    @classmethod
    def _get_model(cls):
        with cls._model_lock:
            if cls._model is None and _WHISPER_OK:
                cls._model = _whisper_lib.load_model(WHISPER_MODEL_NAME)
        return cls._model

    def __init__(self):
        self.recording = False
        self._frames = []
        self._stream = None

    def start(self):
        if not VOICE_AVAILABLE:
            raise RuntimeError("Librerie per la risposta vocale non disponibili")
        self._frames = []
        self.recording = True

        def callback(indata, frames_count, time_info, status):
            if self.recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=VOICE_SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        self.recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if not self._frames:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self._frames, axis=0).reshape(-1)

    @staticmethod
    def save_wav(audio: np.ndarray, path):
        if _SOUNDDEVICE_OK:
            sf.write(str(path), audio, VOICE_SAMPLE_RATE)

    @classmethod
    def transcribe(cls, audio: np.ndarray, language: str = "it"):
        """Trascrive l'audio in italiano usando Whisper con parametri ottimizzati.
        Applica pre-processing audio e usa opzioni di decodifica calibrate
        per risposte vocali brevi in italiano.
        """
        model = cls._get_model()
        if model is None:
            return "", []

        # Pre-processa l'audio per migliorare SNR e qualità di input
        audio_proc = _preprocess_audio(audio, sr=VOICE_SAMPLE_RATE)

        # Se l'audio originale è silenzio o rumore di fondo puro, evita allucinazioni
        if audio is not None and len(audio) > 0:
            rms_raw = float(np.sqrt(np.mean(audio**2)))
            max_raw = float(np.max(np.abs(audio)))
            if rms_raw < 0.0015 and max_raw < 0.02:
                return "", []

        result = model.transcribe(
            audio_proc,
            language=language,
            word_timestamps=True,
            fp16=False,
            **_WHISPER_DECODE_OPTIONS,
        )

        raw_text = (result.get("text") or "").strip()
        if _is_hallucination(raw_text):
            return "", []

        words = []
        for seg in result.get("segments", []):
            seg_text = (seg.get("text") or "").strip()
            if _is_hallucination(seg_text):
                continue
            for w in seg.get("words", []) or []:
                word_text = (w.get("word") or "").strip()
                if word_text and not _is_hallucination(word_text):
                    words.append(
                        {
                            "word": word_text,
                            "start": w.get("start"),
                            "end": w.get("end"),
                        }
                    )
        return raw_text, words

    @staticmethod
    def detect_onset_sec(
        audio: np.ndarray, word_start: float = None, word_end: float = None
    ):
        """Calcola l'onset fisico dell'inviluppo audio (RMS energy peak onset).
        Se vengono forniti word_start/word_end da Whisper, analizza l'inviluppo RMS
        attorno alla parola per individuare l'istante esatto di inizio dell'emissione vocale.
        Non dipende da librerie esterne oltre a numpy."""
        if audio is None or audio.size == 0:
            return None

        sr = VOICE_SAMPLE_RATE

        if word_start is not None:
            s_idx = max(0, int((word_start - 0.2) * sr))
            e_end = word_end if word_end is not None else (word_start + 0.6)
            e_idx = min(len(audio), int((e_end + 0.3) * sr))
        else:
            s_idx = 0
            e_idx = len(audio)

        segment = audio[s_idx:e_idx]
        if len(segment) == 0:
            return word_start

        win_size = int(sr * 0.02)  # 20ms window
        hop_size = int(sr * 0.005)  # 5ms hop
        n_wins = (len(segment) - win_size) // hop_size + 1
        if n_wins <= 0:
            return word_start

        rms_vals = np.array(
            [
                np.sqrt(np.mean(segment[i * hop_size : i * hop_size + win_size] ** 2))
                for i in range(n_wins)
            ]
        )

        max_rms = np.max(rms_vals)
        if (
            max_rms < 0.015
        ):
            return word_start

        # Soglia onset: 18% del picco RMS
        thresh = max(0.015, max_rms * 0.18)
        above = np.where(rms_vals >= thresh)[0]
        if len(above) == 0:
            return word_start

        first_win = above[0]
        precise_onset = (s_idx + first_win * hop_size) / float(sr)
        return float(precise_onset)
