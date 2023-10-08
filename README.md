# Video Sub Extractor

![python version](https://img.shields.io/badge/Python-3.11-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

Program that extracts hard coded subtitles from video and creates external subtitles.

## Nuitka compile command:

```
nuitka --standalone --enable-plugin=tk-inter --include-data-dir=models=models --include-data-files=VSE.ico=VSE.ico --windows-icon-from-ico=VSE.ico --user-package-configuration-file=gui_nuitka_config.yml gui.py
```

### Run compiled program:

```
.\gui.dist\gui.exe
```