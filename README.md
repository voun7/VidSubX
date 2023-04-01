# Video Sub Extractor

![python version](https://img.shields.io/badge/Python-3.10.9-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

Program that extracts hard coded subtitles from video and creates external subtitles.

Installation steps:

1st Miniconda must be installed and a virtual environment created and activated.

```
https://conda.io/projects/conda/en/stable/user-guide/install/download.html
```

2nd Install paddlepaddle gpu in the conda virtual environment

```
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

Test if paddlepaddle installation its working:

```
import paddle
```

```
paddle.utils.run_check()
```

3rd Install the following in the conda virtual environment:

```
pip install opencv-python
```

```
pip install tqdm
```

```
pip install Shapely
```

```
pip install pyclipper
```

```
pip install lmdb
```

```
pip install --no-deps paddleocr
```

Not all of paddleocr dependencies are needed, comment out the imports for the following packages in paddleocr:
skimage, scipy, imgaug.
