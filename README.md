# Video Sub Extractor

![python version](https://img.shields.io/badge/Python-3.11-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

Program that extracts hard coded subtitles from video and creates external subtitles.

## Setup Instructions

### Download and Install:

[Latest Version of Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist)

Install packages

For GPU

```
conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

For CPU

```
pip install paddlepaddle==2.6.2
```

Other packages

```commandline
pip install -r requirements.txt
```

### Compile Instructions

Install package

```
pip install Nuitka==2.4.11
```

Compile command:

```
nuitka --standalone --enable-plugin=tk-inter --windows-console-mode=disable --include-package-data=paddleocr --include-data-files=VSE.ico=VSE.ico --windows-icon-from-ico=VSE.ico gui.py
```

#### Run compiled program:

```
.\gui.dist\gui.exe
```
