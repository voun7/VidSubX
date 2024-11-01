# Video Sub Extractor

![python version](https://img.shields.io/badge/Python-3.11-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

A free program that extracts hard coded subtitles from a video and generates an external subtitle file.

<img src="docs/images/gui%20screenshot.png" width="500">


**Features**

- Detect subtitle area by searching common area
- Manual resize or change of subtitle area (click and drag mouse to perform)
- Single and Batch subtitle detection and extraction
- Start and Stop subtitle extraction positions can be selected (use arrow keys for precise selection)
- Resize video display (Zoom In (Ctrl+Plus), Zoom Out (Ctrl+Minus))
- Non subtitle area of the video can be hidden to limit spoilers
- Toast Notification available on Windows upon completion of subtitle detection and extraction
- Preferences available for modification of options when extraction subtitles
- Multiple languages supported. They will be automatically downloaded as needed.

**Download**

[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist) must be
installed. The program will not start without it.

- [Windows CPU Version](https://github.com/voun7/Video_Sub_Extractor/releases/download/v1.0/VSE-windows-cpu.zip)

## Demo

[![Demo Video](docs/images/demo%20screenshot.png)](https://youtu.be/nnm_waobgnI "Demo Video")

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

Run `gui.py` to use Graphical interface and `main.py` to use Terminal.

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

`gui.exe` can be manually renamed to `VSE.exe`
