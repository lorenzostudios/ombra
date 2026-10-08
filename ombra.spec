# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Raccolta automatica degli asset e moduli per CustomTkinter, Whisper e Sound
ctk_datas, ctk_binaries, ctk_hiddenimports = collect_all('customtkinter')

try:
    whisper_datas, whisper_binaries, whisper_hiddenimports = collect_all('whisper')
except Exception:
    whisper_datas, whisper_binaries, whisper_hiddenimports = [], [], []

try:
    sd_datas, sd_binaries, sd_hiddenimports = collect_all('sounddevice')
except Exception:
    sd_datas, sd_binaries, sd_hiddenimports = [], [], []

try:
    sf_datas, sf_binaries, sf_hiddenimports = collect_all('soundfile')
except Exception:
    sf_datas, sf_binaries, sf_hiddenimports = [], [], []

datas = [
    ('video_config.json', '.'),
    ('yolov8n.pt', '.'),
] + ctk_datas + whisper_datas + sd_datas + sf_datas

binaries = ctk_binaries + whisper_binaries + sd_binaries + sf_binaries

hiddenimports = [
    'customtkinter',
    'PIL._tkinter_finder',
    'cv2',
    'torch',
    'torchvision',
    'ultralytics',
    'sounddevice',
    'soundfile',
    'whisper',
    'pandas',
    'numpy',
] + ctk_hiddenimports + whisper_hiddenimports + sd_hiddenimports + sf_hiddenimports

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['videos', 'audio_responses', 'output', '.git', '.vscode'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OMBRA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OMBRA',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='OMBRA.app',
        icon=None,
        bundle_identifier='com.ombra.videoperception',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'NSMicrophoneUsageDescription': 'OMBRA richiede l\'accesso al microfono per le risposte vocali.',
        },
    )
