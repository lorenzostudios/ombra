# MIRA - Video Perception Test Application

Applicazione desktop per la somministrazione e l'analisi di test di percezione visiva su filmati sportivi con integrazione di risposte manuali e vocali (OpenAI Whisper) e computer vision (YOLOv8).

---

## 🚀 Istruzioni per gli Utenti Finali (Download da GitHub Releases)

Per utilizzare l'applicazione senza installare Python:

1. **Scarica l'applicazione**:
   - Dalla sezione [Releases](../../releases) su GitHub, scarica l'archivio relativo al tuo sistema operativo:
     - `MIRA-Windows.zip` (per PC Windows a 64 bit)
     - `MIRA-macOS.zip` (per Mac con Apple Silicon / Intel)
     - `MIRA-Linux.tar.gz` (per distribuzioni Linux a 64 bit, es. Ubuntu/Debian/Fedora)
   - Scarica inoltre l'archivio dei video: `videos.zip`.

2. **Estrazione**:
   - Estrai l'archivio dell'applicazione in una cartella a tuo piacimento.
   - Estrai `videos.zip` in modo da avere la cartella `videos` nella stessa posizione dell'applicazione:
     - **Windows**:
       ```
       MIRA/
       ├── MIRA.exe
       ├── _internal/
       ├── video_config.json
       ├── yolov8n.pt
       └── videos/          <-- cartella con i file .mp4
       ```
     - **macOS**:
       ```
       MIRA-macOS/
       ├── MIRA.app
       └── videos/          <-- cartella con i file .mp4
       ```
     - **Linux**:
       ```
       MIRA/
       ├── MIRA             <-- binario eseguibile
       ├── _internal/
       ├── video_config.json
       ├── yolov8n.pt
       └── videos/          <-- cartella con i file .mp4
       ```

3. **Avvio**:
   - **Windows**: Doppio click su `MIRA.exe`.
   - **macOS**: Doppio click su `MIRA.app`. *(Se macOS blocca l'avvio perché l'app non è firmata da uno sviluppatore identificato: fai click destro su `MIRA.app` -> Apri -> Conferma "Apri").*
   - **Linux**: Apri il terminale nella cartella ed esegui `./MIRA` (oppure doppio click sull'eseguibile `MIRA`).

> [!TIP]
> Se preferisci conservare la cartella `videos` in una posizione differente sul tuo disco, puoi selezionarla in qualunque momento dalla schermata principale dell'applicazione tramite il pulsante **"Sfoglia..."**.

---

## 🛠️ Esecuzione da Codice Sorgente (Sviluppo)

1. **Requisiti**: Python 3.10, 3.11, 3.12 o 3.13.
   - *Su Linux (Ubuntu/Debian)*, installa i pacchetti di sistema per Tkinter, audio e OpenCV:
     ```bash
     sudo apt update && sudo apt install -y python3-tk libportaudio2 libasound2-dev libgl1 libglib2.0-0
     ```
2. **Installazione dipendenze Python**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Avvio dell'applicazione**:
   ```bash
   python main.py
   ```

---

## 📦 Compilazione Eseguibile Locale (PyInstaller)

Per compilare manualmente l'eseguibile sul proprio computer:

```bash
# 1. Installa PyInstaller
pip install pyinstaller

# 2. Compila usando il file di specifica del progetto
pyinstaller --clean mira.spec
```

I file compilati saranno generati nella cartella `dist/`:
- **macOS**: `dist/MIRA.app`
- **Windows**: `dist/MIRA/` (contenente `MIRA.exe`)
- **Linux**: `dist/MIRA/` (contenente il binario `MIRA`)

---

## 🤖 Build e Release Automatica (GitHub Actions)

Il progetto include un workflow automatizzato in [`.github/workflows/build-and-release.yml`](.github/workflows/build-and-release.yml).

- Quando crei un tag (es. `v1.0.0`) e lo invii a GitHub:
  ```bash
  git tag v1.0.0
  git push origin v1.0.0
  ```
- GitHub Actions compilerà in automatico per **Windows**, **macOS** e **Linux** e creerà la bozza/release con i file `.zip` e `.tar.gz` già allegati.
