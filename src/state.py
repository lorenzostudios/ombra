"""
OMBRA - Stato dell'Applicazione
Mantiene in memoria i dati di sessione del partecipante corrente e l'avanzamento dei trial.
"""

import pathlib
from .constants import DEFAULT_VIDEO_FOLDER


class AppState:
    """
    Rappresenta lo stato dinamico di una sessione di test.
    Contiene le informazioni anagrafiche del partecipante, la modalità di risposta selezionata,
    l'elenco bilanciato dei video estratti per la sessione e l'indice del trial in corso.
    """

    def __init__(self):
        self.participant_name = ""
        self.participant_notes = ""
        self.participant_age_group = ""
        self.participant_gender = ""
        self.participant_world_cup_2026 = ""
        self.participant_football_frequency = ""
        self.video_folder = pathlib.Path(DEFAULT_VIDEO_FOLDER)
        self.session_video_list = []
        self.trial_index = 0
        self.session_seed = 0
        self.session_index = 1
        self.trial_results = []
        self.answer_mode = "keys"  # "keys" (Shift sx/dx) | "voice" (microfono)
        # True solo tra l'avvio della sessione e l'apertura del sondaggio:
        # fuori da questa finestra la pagina Test non e' navigabile.
        self.test_active = False
