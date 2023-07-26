# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('VSE.ico', '.'), ('custom_paddleocr/ppocr/utils', 'custom_paddleocr/ppocr/utils'), ('models', 'models')]
bin_path = "C:\\Users\\nwaez\\miniconda3\\envs\\VSE\\Library\\bin\\"
binaries = [
(bin_path + "zlibwapi.dll", "."),
(bin_path + "cudnn_ops_infer64_8.dll", "."),
(bin_path + "cudnn_cnn_infer64_8.dll", "."),
(bin_path + "cudnn64_8.dll", "."),
(bin_path + "cublasLt64_11.dll", "."),
(bin_path + "cublas64_11.dll", ".")
]
hiddenimports = []
tmp_ret = collect_all('paddle')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


block_cipher = None


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='gui',
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
    icon=['VSE.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui',
)
