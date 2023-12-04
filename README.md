# Video Sub Extractor

![python version](https://img.shields.io/badge/Python-3.11-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

Program that extracts hard coded subtitles from video and creates external subtitles.

## Setup Instructions:

### Download and Install:

[Latest Version of Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170)

### Install packages:

```commandline
pip install -r requirements.txt
```

## Nuitka compile command:

```
nuitka --standalone --enable-plugin=tk-inter --disable-console --include-data-dir=models=models --include-data-files=VSE.ico=VSE.ico --windows-icon-from-ico=VSE.ico --user-package-configuration-file=gui_nuitka_config.yml gui.py
```

### Run compiled program:

```
.\gui.dist\gui.exe
```

#### NOTE: If flashing terminal occurs after build. Fix with the following:

Locate this module in paddle package.

```
venv/Lib/site-packages/paddle/utils/cpp_extension/extension_utils.py
```

Find and comment out this block of code.

```
nvcc_path = subprocess.check_output(
    [which_cmd, 'nvcc'], stderr=devnull
)
```
