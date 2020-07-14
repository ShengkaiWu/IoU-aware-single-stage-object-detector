## Installation

### Requirements

- Linux
- Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
  (The version of Python we use: 3.7)
- PyTorch 1.0+ or PyTorch-nightly
  (The version of PyTorch we use: 1.1.0)
- CUDA 9.0+
  (The version of CUDA we use: 10.0)
- NCCL 2+
  (The version of NCCL we use: 2.4.2)
- GCC 4.9+
  (The version of GCC we use: 5.4.0)
- [mmcv](https://github.com/open-mmlab/mmcv)
  (The version of mmcv we use: 0.2.8)
  ```shell
  pip install mmcv==0.2.8
  ```

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC: 4.9/5.3/5.4/7.3

### Install mmdetection

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n open-mmlab python=3.7 -y
source activate open-mmlab

conda install cython
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the mmdetection repository.

```shell
git clone https://github.com/ShengkaiWu/IoU-aware-single-stage-object-detector.git
cd IoU-aware-single-stage-object-detector
```

d. Compile cuda extensions.

```shell
./compile.sh
```

e. Install mmdetection (other dependencies will be installed automatically).

```shell
python setup.py develop
# or "pip install -e ."
```

Note:

1. It is recommended that you run the step e each time you pull some updates from github. If there are some updates of the C/CUDA codes, you also need to run step d.
The git commit id will be written to the version number with step e, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, mmdetection is installed on `dev` mode, any modifications to the code will take effect without installing it again.

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```

### Scripts
[Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.

### Notice
You can run `python(3) setup.py develop` or `pip install -e .` to install mmdetection if you want to make modifications to it frequently.

If there are more than one mmdetection on your machine, and you want to use them alternatively.
Please insert the following code to the main file
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
