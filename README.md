# OMBRA - Video Perception Test Application & Shadow Augmentation

Applicazione desktop per la somministrazione e l'analisi di test di percezione visiva su filmati sportivi con integrazione di computer vision (YOLOv8), risposte vocali (OpenAI Whisper) e generazione di ombre sintetiche per l'aumento della percezione di profondità.

---

## 📖 Panoramica del Progetto (Tesi)

Il progetto nasce nell'ambito di una tesi sperimentale finalizzata a valutare l'impatto di stimoli visivi sintetici (nello specifico **ombre a terra**) sulla percezione umana della profondità e delle traiettorie tridimensionali in video sportivi (calcio/World Cup). 

L'ipotesi di ricerca indaga se l'aggiunta di un'ombra artificiale proiettata al suolo:
1. **Migliori l'accuratezza** di giudizio degli utenti rispetto a eventi critici (es. se la palla entra in porta, se rimbalza, se supera una linea, traiettoria verso destra/sinistra).
2. **Riduca il tempo di risposta** (Response Time, RT) e favorisca una **maggiore capacità di anticipazione predittiva** rispetto al frame/millisecondo chiave dell'evento.

### Componenti e Moduli Software
L'applicativo è strutturato come una suite integrata suddivisa in quattro aree funzionali:

- **Somministrazione Test (`TestPage`, `SurveyPage`)**:
  - Esecuzione di batterie di test con riproduzione sincronizzata frame-by-frame di video originali e aumentati (con ombra).
  - Raccolta dati demografici/abitudini calcistiche ed esportazione in tempo reale di metriche temporali dettagliate (`timestamp`, `response_time_ms`, `delta_ms` dall'evento target).
  - Supporto per risposte tramite **tastiera** o **voce** (riconoscimento vocale offline con OpenAI Whisper).
- **Generazione e Aumento Ombre (`src/shadow/`)**:
  - **AI / Computer Vision**: tracciamento del pallone e dei giocatori mediante **YOLOv8** (`yolov8n.pt`).
  - **Machine Learning & Trajectory**: stima del punto di proiezione a terra della sfera con modelli dedicati (`shadow_ml.py`, pesi in `shadow_model_weights.json`).
  - **Strumenti Manuali**: interfaccia canvas interattiva per l'editing, correzione e interpolazione fine delle ombre frame per frame (`manual_shadow_page.py`).
- **Analisi Audio & Vocale (`AudioAnalysisPage`, `voice_engine.py`)**:
  - Registrazione audio delle risposte, trascrizione con Whisper e analisi della latenza vocale per sincronizzare l'istante effettivo della risposta verbale.
- **Reportistica & Statistiche (`ReportPage`, `results_manager.py`)**:
  - Dashboard per visualizzare accuratezza globale, tempi medi di reazione, confronto diretto fra stimoli aumentati e originali e analisi dei delta temporali.

---

## 📂 Struttura del Progetto

```
ombra/
├── main.py                     # Entry point dell'applicazione Tkinter/CustomTkinter
├── ombra.spec                  # Configurazione PyInstaller per il packaging multi-piattaforma
├── requirements.txt            # Dipendenze Python del progetto
├── test_results.csv            # Dataset dei risultati sperimentali registrati
├── survey_results.csv          # Risposte ai questionari pre-test
├── video_config.json           # Metadati dei video (domande, frame target, risposte corrette)
├── yolov8n.pt                  # Modello compatto YOLOv8 per il tracciamento del pallone
├── videos/                     # Cartella contenente i video (.mp4) del test
├── audio_responses/            # Registrazioni audio acquisite durante i test vocali
├── .github/workflows/          # Workflow CI/CD per compilazione e release automatica
│   └── build-and-release.yml
└── src/                        # Codice sorgente modulare
    ├── app.py                  # Finestra principale e routing delle schermate
    ├── constants.py            # Costanti grafiche, percorsi e configurazioni globali
    ├── config_manager.py       # Gestione del file video_config.json
    ├── results_manager.py      # Gestione, validazione ed esportazione dei file CSV
    ├── voice_engine.py         # Engine audio e trascrizione con Whisper
    ├── utils.py                # Funzioni di utilità per percorsi (PyInstaller) e frame video
    ├── state.py                # Stato globale di sessione
    ├── components/             # Componenti GUI riutilizzabili
    │   ├── video_player.py     # Player video ad alte prestazioni integrato in Tkinter
    │   ├── shadow_canvas.py    # Canvas interattivo per manipolazione grafica dell'ombra
    │   └── action_progress.py  # Barre di avanzamento per task asincroni
    ├── pages/                  # Viste / Schermate dell'interfaccia utente
    │   ├── home_page.py        # Menu principale e selezione modalità
    │   ├── test_page.py        # Schermata di somministrazione del test percettivo
    │   ├── survey_page.py      # Form anagrafico del partecipante
    │   ├── report_page.py      # Visualizzazione statistiche e grafici
    │   ├── manual_shadow_page.py # Tool di annotazione manuale delle ombre
    │   ├── ai_shadow_page.py   # Pipeline semi-automatica YOLO per l'ombra
    │   ├── shadow_gen_page.py  # Generazione batch di video aumentati
    │   ├── edit_page.py        # Editor di configurazione video e frame target
    │   └── audio_analysis_page.py # Revisione delle registrazioni vocali
    └── shadow/                 # Algoritmi di stima e rendering dell'ombra
        ├── ai_shadow.py        # Pipeline di computer vision con YOLOv8
        ├── auto_shadow.py      # Tracciamento ed estrapolazione traiettorie
        ├── manual_shadow.py    # Logica di interpolazione fotogrammi
        ├── shadow_ml.py        # Modello predittivo ML per la coordinata Y dell'ombra
        └── train_shadow_model.py # Script di addestramento modello
```

---

## 🚀 Istruzioni di Download (GitHub Releases)

Se desideri eseguire OMBRA **senza installare Python o dipendenze**:

1. Vai alla sezione [Releases](../../releases) della repository GitHub.
2. Scarica il pacchetto per il tuo sistema operativo:
   - **Windows (x64)**: `OMBRA-Windows.zip`
   - **macOS (Apple Silicon / Intel)**: `OMBRA-macOS.zip`
   - **Linux (x64)**: `OMBRA-Linux.tar.gz`
3. Scarica anche il pacchetto dei video: `videos.zip`.
4. **Organizzazione cartelle**:
   Estrai `videos.zip` in modo che la cartella `videos` risieda accanto all'eseguibile:
   - **Windows**:
     ```
     OMBRA/
     ├── OMBRA.exe
     ├── _internal/
     ├── video_config.json
     ├── yolov8n.pt
     └── videos/          <-- cartella con i video .mp4
     ```
   - **macOS**:
     ```
     OMBRA-macOS/
     ├── OMBRA.app
     └── videos/          <-- cartella con i video .mp4
     ```
   - **Linux**:
     ```
     OMBRA/
     ├── OMBRA             <-- binario eseguibile
     ├── _internal/
     ├── video_config.json
     ├── yolov8n.pt
     └── videos/          <-- cartella con i video .mp4
     ```

> [!TIP]
> Se la cartella `videos` si trova altrove sul disco, è possibile selezionarla in qualsiasi momento dalla schermata principale dell'app tramite il pulsante **"Sfoglia..."**.

---

## 💻 Istruzioni di Esecuzione

### Esecuzione della Release Pre-compilata
- **Windows**: Doppio click su `OMBRA.exe`.
- **macOS**: Doppio click su `OMBRA.app`.  
  *(Se macOS blocca l'avvio con avviso sviluppatore non identificato: click destro su `OMBRA.app` $\rightarrow$ **Apri** $\rightarrow$ confermare su **Apri**).*
- **Linux**: Aprire un terminale nella cartella ed eseguire:
  ```bash
  chmod +x OMBRA
  ./OMBRA
  ```

### Esecuzione da Codice Sorgente (Sviluppo)
1. **Requisiti**: Python 3.10 o versioni successive (fino a 3.13).
   - Su distribuzioni **Linux (Ubuntu/Debian)** installare prima i pacchetti di sistema per Tkinter, audio e OpenCV:
     ```bash
     sudo apt update && sudo apt install -y python3-tk libportaudio2 libasound2-dev libgl1 libglib2.0-0
     ```
2. **Ambiente virtuale (consigliato)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Su Linux/macOS
   # venv\Scripts\activate         # Su Windows
   ```
3. **Installazione dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Avvio dell'applicazione**:
   ```bash
   python main.py
   ```

---

## 🛠️ Istruzioni di Compilazione Eseguibile Locale (PyInstaller)

Per compilare autonomamente l'eseguibile standalone sulla propria macchina utilizzando il file di specifica [`ombra.spec`](ombra.spec):

1. Assicurarsi di aver attivato il virtual environment e installato sia le dipendenze che **PyInstaller**:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```
2. Avviare la compilazione:
   ```bash
   pyinstaller --clean ombra.spec
   ```
3. Gli output verranno generati all'interno della directory `dist/`:
   - **macOS**: `dist/OMBRA.app`
   - **Windows**: `dist/OMBRA/` (contenente `OMBRA.exe` e la cartella `_internal`)
   - **Linux**: `dist/OMBRA/` (contenente il binario eseguibile `OMBRA`)