"""
MIRA - Generazione Ombra IA Automatica (AIShadowPage)
Interfaccia grafica per l'elaborazione end-to-end con modelli di intelligenza artificiale
(rilevamento YOLOv8, modello ML di traiettoria e rendering al suolo).
"""

import cv2
import queue
import pathlib
import threading
from tkinter import filedialog, messagebox
import customtkinter as ctk

from ..constants import (
    APP_TITLE,
    BG_DARK,
    PANEL_BG,
    CARD_BG,
    CARD_BORDER,
    ACCENT,
    ACCENT_HOVER,
    BLUE,
    BLUE_HOVER,
    TEXT_LIGHT,
    TEXT_MUTED,
    SHADOW_OUTPUT_FOLDER,
)
from ..utils import open_path
from ..components.video_player import VideoPlayerFrame
from ..shadow.ai_shadow import AIShadowEngine


class AIShadowPage(ctk.CTkFrame):
    """
    Pagina dedicata alla generazione 100% automatica dell'ombra virtuale con modelli IA.
    Include analisi prospettica del campo, check di consistenza e rendering/esportazione in un unico passaggio.
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color=BG_DARK)
        self.app = app

        self.video_path = None
        self.raw_frames = []
        self.processed_frames = []
        self.exported_video_path = None
        self.fps = 25.0
        self.is_processing = False
        self.is_training = False

        self._task_queue = queue.Queue()

        self._build_ui()

    def on_show(self):
        """Callback all'apertura della schermata."""
        pass

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header Banner
        header = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", padx=16, pady=12)
        ctk.CTkLabel(
            title_box,
            text="Generazione Ombra con Intelligenza Artificiale",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="Rilevamento di azioni e generazione di ombra virtuale automatiche attraverso un modello allenato di intelligenza artificiale.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            wraplength=850,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        # Corpo principale a due colonne
        body = ctk.CTkFrame(self, fg_color=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=320)  # Colonna sinistra a larghezza fissa
        body.grid_columnconfigure(1, weight=1)  # Colonna destra player preview

        self._build_left_sidebar(body)
        self._build_right_player(body)

    def _build_left_sidebar(self, parent):
        left = ctk.CTkFrame(parent, width=320, fg_color=PANEL_BG, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.pack_propagate(False)  # Impedisce ai widget interni di ridimensionare la colonna

        ctk.CTkLabel(
            left,
            text="Video",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(16, 6))

        ctk.CTkButton(
            left,
            text="Seleziona video",
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            text_color="#ffffff",
            command=self.select_video,
        ).pack(fill="x", padx=16)

        self.lbl_video_name = ctk.CTkLabel(
            left,
            text="Nessun video selezionato",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=280,
            justify="left",
            anchor="w",
        )
        self.lbl_video_name.pack(fill="x", padx=16, pady=(6, 12))

        sep1 = ctk.CTkFrame(left, height=1, fg_color=CARD_BORDER)
        sep1.pack(fill="x", padx=16, pady=(0, 12))

        # Parametri del modello AI
        ctk.CTkLabel(
            left,
            text="Parametri Rilevamento AI",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT,
        ).pack(anchor="w", padx=16, pady=(0, 6))

        ctk.CTkLabel(
            left,
            text="Dimensione dell'ombra:",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=16, pady=(4, 2))

        self.slider_shadow_size = ctk.CTkSlider(
            left, from_=1, to=30, number_of_steps=29, command=self._on_param_change
        )
        self.slider_shadow_size.set(5)
        self.slider_shadow_size.pack(fill="x", padx=16, pady=(0, 2))

        self.lbl_shadow_size_val = ctk.CTkLabel(
            left, text="5 px", font=ctk.CTkFont(size=11), text_color=TEXT_LIGHT
        )
        self.lbl_shadow_size_val.pack(anchor="w", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            left,
            text="Sensibilità rilevamento AI:",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=16, pady=(4, 2))

        self.slider_conf = ctk.CTkSlider(
            left, from_=0.05, to=0.50, number_of_steps=45, command=self._on_param_change
        )
        self.slider_conf.set(0.15)
        self.slider_conf.pack(fill="x", padx=16, pady=(0, 2))

        self.lbl_conf_val = ctk.CTkLabel(
            left, text="0.15", font=ctk.CTkFont(size=11), text_color=TEXT_LIGHT
        )
        self.lbl_conf_val.pack(anchor="w", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            left,
            text="Opacità dell'ombra:",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=16, pady=(4, 2))

        self.slider_opacity = ctk.CTkSlider(
            left, from_=0.01, to=1.00, number_of_steps=99, command=self._on_param_change
        )
        self.slider_opacity.set(0.25)
        self.slider_opacity.pack(fill="x", padx=16, pady=(0, 2))

        self.lbl_opacity_val = ctk.CTkLabel(
            left, text="25%", font=ctk.CTkFont(size=11), text_color=TEXT_LIGHT
        )
        self.lbl_opacity_val.pack(anchor="w", padx=16, pady=(0, 14))

        sep2 = ctk.CTkFrame(left, height=1, fg_color=CARD_BORDER)
        sep2.pack(fill="x", padx=16, pady=(0, 12))

        # UNICO PULSANTE: Analisi IA + Rendering + Esportazione Video
        self.btn_generate = ctk.CTkButton(
            left,
            text="Genera video",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#06120b",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.run_ai_processing,
        )
        self.btn_generate.pack(fill="x", padx=16, pady=(0, 8))

        # Pulsante riaddestramento modello ML sui 38 progetti
        self.btn_retrain = ctk.CTkButton(
            left,
            text="Riallena modello sul dataset",
            fg_color="transparent",
            border_width=1,
            border_color=CARD_BORDER,
            hover_color=CARD_BG,
            text_color=TEXT_LIGHT,
            height=30,
            font=ctk.CTkFont(size=11),
            command=self.run_dataset_training,
        )
        self.btn_retrain.pack(fill="x", padx=16, pady=(0, 10))

        # Badge Azione Calcistica Riconosciuta
        self.lbl_action_badge = ctk.CTkLabel(
            left,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT,
            wraplength=280,
            justify="left",
            anchor="w",
        )
        self.lbl_action_badge.pack(fill="x", padx=16, pady=(0, 4))

        # Barra di avanzamento e stato
        self.progress_bar = ctk.CTkProgressBar(left, progress_color=ACCENT)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 6))

        self.lbl_status = ctk.CTkLabel(
            left,
            text="Pronto.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=280,
            justify="left",
            anchor="w",
        )
        self.lbl_status.pack(fill="x", padx=16, pady=(0, 10))


        # Tasto rapido per aprire la cartella del video esportato (visibile al termine)

        self.btn_open_folder = ctk.CTkButton(
            left,
            text="Apri cartella video",
            fg_color="transparent",
            border_width=1,
            border_color=CARD_BORDER,
            hover_color=CARD_BG,
            text_color=TEXT_LIGHT,
            height=30,
            command=self._open_output_folder,
        )

    def _build_right_player(self, parent):
        right = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.player = VideoPlayerFrame(right, show_controls=True)
        self.player.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    def select_video(self):
        """Apre il file picker per selezionare un video sorgente."""
        path = filedialog.askopenfilename(
            title="Seleziona video per elaborazione AI",
            filetypes=[("Video MP4/AVI", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not path:
            return

        chosen_path = pathlib.Path(path)
        # Se l'utente ha selezionato un file già modificato con ombra (es. _WorldCup....mp4),
        # cerca automaticamente il video originale pulito per evitare sovrapposizione di ombre.
        if chosen_path.name.startswith("_"):
            clean_name = chosen_path.name.lstrip("_")
            orig_candidate = chosen_path.parent / clean_name
            if not orig_candidate.exists():
                # Controlla anche nella cartella videos
                orig_candidate = chosen_path.parent.parent / "videos" / clean_name

            if orig_candidate.exists():
                chosen_path = orig_candidate
                self.app.set_status(
                    f"Selezionato automaticamente il video originale {chosen_path.name} (per evitare doppie ombre).", "warn"
                )

        self.video_path = chosen_path
        self.lbl_video_name.configure(text=self.video_path.name)
        self.btn_open_folder.pack_forget()
        self.app.set_status(f"Caricamento video: {self.video_path.name}", "info")

        def on_loaded(ok, n_frames, fps):
            if ok:
                self.fps = fps
                # Salva una copia immutabile e pulita dei frame originali
                self.raw_frames = [f.copy() for f in self.player.frames]
                self.processed_frames = []
                self.lbl_status.configure(text=f"Caricati {n_frames} fotogrammi ({fps:.1f} FPS).")
                self.app.set_status(f"Video {self.video_path.name} pronto per elaborazione AI.", "success")
            else:
                self.lbl_status.configure(text="Errore durante il caricamento del video.")


        self.player.load_video(self.video_path, on_loaded=on_loaded)

    def _on_param_change(self, value):
        shadow_sz = int(self.slider_shadow_size.get())
        conf_val = round(float(self.slider_conf.get()), 2)
        opac_val = int(round(float(self.slider_opacity.get()) * 100))
        self.lbl_shadow_size_val.configure(text=f"{shadow_sz} px")
        self.lbl_conf_val.configure(text=f"{conf_val:.2f}")
        self.lbl_opacity_val.configure(text=f"{opac_val}%")

    def run_dataset_training(self):
        """Avvia l'addestramento del modello ML sui 38 progetti in background."""
        if self.is_processing or self.is_training:
            return

        self.is_training = True
        self.btn_retrain.configure(state="disabled", text="Addestramento in corso...")
        self.btn_generate.configure(state="disabled")
        self.progress_bar.set(0.1)
        self.lbl_status.configure(text="Addestramento modello su 38 progetti in corso...")

        def train_worker():
            try:
                from ..shadow.train_shadow_model import extract_pixel_diff_dataset_and_train
                results = extract_pixel_diff_dataset_and_train(verbose=False)
                self._task_queue.put(("train_done", results))
            except Exception as e:  # noqa: BLE001
                self._task_queue.put(("train_error", str(e)))

        threading.Thread(target=train_worker, daemon=True).start()
        self.after(50, self._poll_ai_task)

    def run_ai_processing(self):
        """Avvia in un solo click: rilevamento IA + check consistenza + render + esportazione MP4."""
        if not self.video_path:
            messagebox.showwarning(APP_TITLE, "Seleziona prima un video valido.")
            return

        if self.is_processing or self.is_training:
            return

        # Assicura di avere i frame originali puliti (senza ombra)
        if not self.raw_frames:
            if self.player.frames:
                # Usa i frame già caricati nel player come sorgente primaria
                self.raw_frames = [f.copy() for f in self.player.frames]
                self.fps = self.player.fps or self.fps
            elif self.video_path and self.video_path.exists():
                from ..utils import load_video_frames as _load_vf
                loaded_f, loaded_fps = _load_vf(self.video_path)
                self.raw_frames = [f.copy() for f in loaded_f]
                self.fps = loaded_fps

        if not self.raw_frames:
            messagebox.showwarning(APP_TITLE, "Impossibile caricare i fotogrammi originali del video.")
            return

        self.is_processing = True
        self.btn_generate.configure(state="disabled", text="Elaborazione in corso")
        self.btn_retrain.configure(state="disabled")
        self.btn_open_folder.pack_forget()
        self.lbl_action_badge.configure(text="")
        self.progress_bar.set(0.0)

        shadow_sz = int(self.slider_shadow_size.get())
        conf_val = round(float(self.slider_conf.get()), 2)
        opac_val = round(float(self.slider_opacity.get()), 2)

        engine = AIShadowEngine(
            confidence_thresh=conf_val,
            shadow_size=shadow_sz,
            shadow_opacity=opac_val,
        )

        video_in_path = self.video_path
        fps_val = self.fps
        # CLONA SEMPRE i frame puliti originali: mai riutilizzare i frame già modificati con ombra!
        input_frames = [f.copy() for f in self.raw_frames]

        def worker():
            def progress_cb(pct, msg):
                self._task_queue.put(("progress", pct, msg))

            try:
                # 1. Pipeline di analisi, consistenza e rendering con modello ML (0% -> 85%)
                out_frames, stats = engine.process_frames(input_frames, progress_callback=progress_cb)

                # 2. Esportazione automatica video MP4 (85% -> 100%)
                if out_frames:
                    progress_cb(0.85, "Inizializzazione esportazione video MP4...")
                    SHADOW_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
                    out_name = f"_{video_in_path.stem}.mp4"
                    out_path = SHADOW_OUTPUT_FOLDER / out_name

                    h_f, w_f = out_frames[0].shape[:2]
                    writer = None
                    for codec in ["mp4v", "avc1", "H264", "MJPG"]:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps_val, (w_f, h_f))
                        if writer.isOpened():
                            break

                    if writer is not None and writer.isOpened():
                        n_out = len(out_frames)
                        for f_i, f in enumerate(out_frames):
                            writer.write(f)
                            if f_i % 12 == 0 or f_i == n_out - 1:
                                pct = 0.85 + 0.14 * ((f_i + 1) / n_out)
                                progress_cb(pct, f"Esportazione video MP4 ({f_i + 1}/{n_out})...")
                        writer.release()
                        stats["exported_path"] = out_path

                self._task_queue.put(("done", out_frames, stats))
            except Exception as e:  # noqa: BLE001
                self._task_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(50, self._poll_ai_task)


    def _poll_ai_task(self):
        """Verifica lo stato di avanzamento del thread AI tramite coda messaggi."""
        while not self._task_queue.empty():
            msg = self._task_queue.get_nowait()
            kind = msg[0]

            if kind == "progress":
                pct, status_msg = msg[1], msg[2]
                self.progress_bar.set(pct)
                self.lbl_status.configure(text=status_msg)

            elif kind == "train_done":
                res = msg[1]
                self.is_training = False
                self.btn_retrain.configure(state="normal", text="Riallena modello sul dataset")
                self.btn_generate.configure(state="normal", text="Genera video")
                self.progress_bar.set(1.0)
                self.lbl_status.configure(
                    text=f"Modello riaddestrato su differenza pixel! ({res.get('n_videos', 38)} video, {res.get('n_samples', 0)} campioni pixel)"
                )
                self.app.set_status("Modello AI riaddestrato su dataset differenziale.", "success")
                return

            elif kind == "train_error":
                err_msg = msg[1]
                self.is_training = False
                self.btn_retrain.configure(state="normal", text="Riallena modello sul dataset")
                self.btn_generate.configure(state="normal", text="Genera video")
                self.lbl_status.configure(text=f"Errore addestramento: {err_msg}")
                self.app.set_status(f"Errore riaddestramento: {err_msg}", "error")
                return

            elif kind == "done":
                self.processed_frames = msg[1]
                stats = msg[2]
                self.is_processing = False
                self.btn_generate.configure(state="normal", text="Genera video")
                self.btn_retrain.configure(state="normal", text="Riallena modello sul dataset")
                self.progress_bar.set(1.0)

                pct_det = stats.get("detection_pct", 0.0)
                used_yolo = stats.get("used_yolo", False)
                engine_type = "YOLOv8" if used_yolo else "Computer Vision"
                c_stats = stats.get("consistency_stats", {})
                pruned = c_stats.get("pruned_outliers", 0)

                # Mostra badge dell'azione calcistica riconosciuta
                act_info = stats.get("action_info", {})
                if act_info:
                    self.lbl_action_badge.configure(
                        text=f"Azione: {act_info.get('label_it', 'Dinamica Calcio')} ({act_info.get('confidence', 0.8)*100:.0f}%)"
                    )

                out_path = stats.get("exported_path")
                self.exported_video_path = out_path

                # Rimossa l'azione duplicata da status_text (già mostrata nel badge sopra)
                status_text = (
                    f"Completato! Pallone rilevato: {pct_det:.0f}% ({engine_type} + ML)\n"
                    f"Esportato in: {out_path.name if out_path else 'OK'}"
                )

                self.lbl_status.configure(text=status_text)



                # Mostra la preview dei frame elaborati nel player sincronizzando slider e controlli
                if self.processed_frames:
                    self.player.set_processed_frames(self.processed_frames, self.fps)

                if self.exported_video_path and self.exported_video_path.exists():
                    self.btn_open_folder.pack(fill="x", padx=16, pady=(4, 12))

                self.app.set_status("Generazione ed esportazione ombra AI completata con successo.", "success")
                return

            elif kind == "error":
                err_msg = msg[1]
                self.is_processing = False
                self.btn_generate.configure(state="normal", text="Genera video")
                self.btn_retrain.configure(state="normal", text="Riallena modello sul dataset")
                self.lbl_status.configure(text=f"Errore durante l'elaborazione: {err_msg}")
                self.app.set_status(f"Errore elaborazione: {err_msg}", "error")
                return


        if self.is_processing or self.is_training:
            self.after(50, self._poll_ai_task)

    def _open_output_folder(self):

        """Apre la cartella di esportazione con il file generato."""
        if self.exported_video_path and self.exported_video_path.exists():
            open_path(self.exported_video_path.parent)
        elif SHADOW_OUTPUT_FOLDER.exists():
            open_path(SHADOW_OUTPUT_FOLDER)
