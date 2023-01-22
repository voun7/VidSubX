# Video Sub Extractor
Program that extracts hardsubs from video and creates external subtitle.

Installation steps:

1st Miniconda must be installed and a virtual environment created and activated.
```
https://conda.io/projects/conda/en/stable/user-guide/install/download.html
```

2nd Install paddlepaddle gpu in the conda virtual environment

```
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```
Test if paddlepaddle installation its working:
```
import paddle
paddle.utils.run_check()
```

3rd Install the following in the conda virtual environment:
```
pip install opencv-python == 4.7.0.68
```
```
pip install natsort == 8.2.0
```
```
pip install fastnumbers == 3.2.1
```
```
pip install tqdm == 4.64.1
```
```
pip install Shapely == 2.0.0
```
```
pip install pyclipper == 1.3.0.post4
```
