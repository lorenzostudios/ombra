"""
MIRA - Core Package
"""

import customtkinter as ctk
from .constants import BLUE, BLUE_HOVER

# Configura globalmente il testo in grassetto (bold) su tutti i pulsanti e segmented button,
# e unifica il colore azzurro e il colore hover in tutta l'applicazione.
_orig_btn_init = ctk.CTkButton.__init__


def _bold_btn_init(self, *args, **kwargs):
    font = kwargs.get("font", None)
    if font is None:
        kwargs["font"] = ctk.CTkFont(size=13, weight="bold")
    elif isinstance(font, ctk.CTkFont):
        font.configure(weight="bold")
    elif isinstance(font, tuple) and len(font) >= 2:
        kwargs["font"] = ctk.CTkFont(family=font[0], size=font[1], weight="bold")

    # Unifica il colore azzurro predefinito di CustomTkinter sul colore BLUE canonico (#3c9fff)
    if "fg_color" not in kwargs:
        kwargs["fg_color"] = BLUE
    if "hover_color" not in kwargs and kwargs.get("fg_color") == BLUE:
        kwargs["hover_color"] = BLUE_HOVER
    if kwargs.get("fg_color") == BLUE and "text_color" not in kwargs:
        kwargs["text_color"] = "#ffffff"

    _orig_btn_init(self, *args, **kwargs)


ctk.CTkButton.__init__ = _bold_btn_init

_orig_sb_init = ctk.CTkSegmentedButton.__init__


def _bold_sb_init(self, *args, **kwargs):
    font = kwargs.get("font", None)
    if font is None:
        kwargs["font"] = ctk.CTkFont(size=12, weight="bold")
    elif isinstance(font, ctk.CTkFont):
        font.configure(weight="bold")
    elif isinstance(font, tuple) and len(font) >= 2:
        kwargs["font"] = ctk.CTkFont(family=font[0], size=font[1], weight="bold")

    if "selected_color" not in kwargs:
        kwargs["selected_color"] = BLUE
    if "selected_hover_color" not in kwargs:
        kwargs["selected_hover_color"] = BLUE_HOVER

    _orig_sb_init(self, *args, **kwargs)


ctk.CTkSegmentedButton.__init__ = _bold_sb_init
