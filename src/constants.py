"""
OMBRA - Costanti e Configurazioni Globali
Definisce la palette dei colori (UI dark modern), le impostazioni predefinite dei test,
i percorsi dei file (compatibili con PyInstaller e sorgente) e i controlli di disponibilità hardware.
"""

import sys
import pathlib

# Verifiche di disponibilità dei moduli per la risposta vocale e trascrizione (Whisper/sounddevice).
try:
    import sounddevice as sd  # noqa: F401
    import soundfile as sf  # noqa: F401

    _SOUNDDEVICE_OK = True
except Exception:
    _SOUNDDEVICE_OK = False

try:
    import whisper as _whisper_lib  # noqa: F401

    _WHISPER_OK = True
except Exception:
    _WHISPER_OK = False

VOICE_AVAILABLE = _SOUNDDEVICE_OK and _WHISPER_OK

# Percorso base del progetto: gestisce sia l'esecuzione da codice sorgente che da applicazione compilata (PyInstaller)
if getattr(sys, "frozen", False):
    if sys.platform == "darwin" and ".app/Contents/MacOS" in sys.executable:
        # Bundle .app macOS: punta alla cartella che contiene la .app (accanto ad essa)
        BASE_DIR = pathlib.Path(sys.executable).resolve().parents[2].parent
    else:
        # Windows / Linux / binario diretto: punta alla cartella dove risiede l'eseguibile
        BASE_DIR = pathlib.Path(sys.executable).resolve().parent
else:
    # Esecuzione standard da sorgente Python
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Cartella interna degli asset estratti/inclusi da PyInstaller (sys._MEIPASS) o cartella del progetto
BUNDLE_DIR = pathlib.Path(getattr(sys, "_MEIPASS", BASE_DIR))

# Impostazioni generali dell'applicazione, percorsi di salvataggio e file di configurazione.
APP_TITLE = "OMBRA"
DEFAULT_VIDEO_FOLDER = BASE_DIR / "videos"
N_TRIALS = 10
RESULTS_FILE = BASE_DIR / "test_results.csv"
SURVEY_FILE = BASE_DIR / "survey_results.csv"
VIDEO_CONFIG_FILE = BASE_DIR / "video_config.json"
DEFAULT_QUESTION = "Secondo te succede qualcosa di anomalo in questo video?"

# Cartella di output dei video modificati con l'ombra virtuale.
SHADOW_OUTPUT_FOLDER = BASE_DIR / "output"

# File di salvataggio del progetto keyframe per l'editor ombra manuale.
MANUAL_PROJECT_SUFFIX = "_shadow_project.json"

# Parametri audio per l'acquisizione dal microfono e la trascrizione con Whisper.
AUDIO_RESPONSES_FOLDER = BASE_DIR / "audio_responses"
VOICE_SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = "small"  # "small" = ~3× più preciso di "base" per l'italiano; "medium" se si preferisce più precisione a scapito della velocità

# Parametri del modello AI per il rilevamento del pallone e del campo.
_LOCAL_YOLO = BASE_DIR / "yolov8n.pt"
_BUNDLED_YOLO = BUNDLE_DIR / "yolov8n.pt"
if _LOCAL_YOLO.exists():
    AI_SHADOW_MODEL_NAME = str(_LOCAL_YOLO)
elif _BUNDLED_YOLO.exists():
    AI_SHADOW_MODEL_NAME = str(_BUNDLED_YOLO)
else:
    AI_SHADOW_MODEL_NAME = "yolov8n.pt"

AI_BALL_CLASS_ID = 32

# Tavolozza di colori coerente per la modalità scura dell'interfaccia.
BG_DARK = "#0c0c10"
PANEL_BG = "#111116"
CARD_BG = "#1b1b22"
CARD_BORDER = "#2b2b36"
ACCENT = "#00d47e"
ACCENT_HOVER = "#00b56b"
BLUE = "#3c9fff"
BLUE_HOVER = "#2b82d9"
WARN = "#f0a030"
DANGER = "#e0503c"
DANGER_HOVER = "#c23f30"
TEXT_LIGHT = "#e8e8ee"
TEXT_MUTED = "#9a9aa8"

# Cursore di puntamento per elementi interattivi (mano con indice teso: pointinghand su Mac, hand2 su Windows/Linux)
POINTER_CURSOR = "pointinghand" if sys.platform == "darwin" else "hand2"
