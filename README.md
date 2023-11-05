# CLRmatchNet


<div align="center">

# CLRmatchNet: Enhancing Curved Lane Detection with Deep Matching Process

</div>


Pytorch implementation of the paper "[CLRmatchNet: Enhancing Curved Lane Detection with Deep Matching Process. Sapir Kontente, Roy Orfaig and Ben-Zion Bobrovsky, Tel-Aviv University](https://arxiv.org/pdf/2309.15204v1)"

## Introduction
![Arch](.github/clrmatchnet.jpg)
![Arch](.github/matchnet.jpg)

- CLRmatchNet introduces an approach that aims to enhance models’ performance by utilizing a deep-learning-based label
assignment model to overcome the limitations of classical cost functions.
- CLRmatchNet focused on enhancing the detection capability of curved lanes. 

## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.7 (tested with Pytorch1.7)
- CUDA (tested with cuda11)
- Other dependencies described in `requirements.txt` 

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$CLRNET_ROOT`
```Shell
git clone https://github.com/sapirkontente/CLRmatchNet.git
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n clrnet python=3.8 -y 
conda activate clrnet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c nvidia

# Or you can install via pip
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install python packages
python setup.py build develop
```



### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

## Getting Started
We use a pretrained CLRNet model as our baseline for training, for 
### Training MatchNet
For training matchnet, use a pretrained CLRNet model and run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num] --finetune_from [releases/path_to_CLRNet_ckp] --train_matchnet [True]
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet101_culane.py --gpus=2 --finetune_from=releases/culane_resnet101.pth --train_matchnet=True
```

### Training CLRmatchNet
For training CLRmatchNet, use your pretrained matchnet ckp and a pretrained CLRNet model and run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num] --finetune_from [releases/path_to_CLRNet_ckp] --train_matchnet [False] --matchnet_ckp [matchnet/ckp/path_to_your_matchnet_ckp]
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet101_culane.py --gpus=1 --finetune_from=releases/culane_resnet101.pth --train_matchnet=False --matchnet_ckp=matchnet/ckp/matchnet.pth
```


### Validation
For testing CLRmatchNet, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate|demo] --load_from [releases/path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --test --load_from=releases/culane_dla34.pth --gpus=1
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/xxx/visualization`.

### Demo
For demonstration of CLRmatchNet results, run
```Shell
python main.py [configs/path_to_your_config] --[demo] --load_from [releases/path_to_your_model] --gpus [gpu_num] --view
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet101_culane.py --demo --load_from=releases/culane_resnet101.pth --gpus=1 --view
```

[assets]: https://github.com/sapirkontetne/CLRmatchNet/releases

### CULane

|   Backbone  |  mF1 | F1@50  | F1@75 | curve |
| :---  |  :---:   |   :---:    | :---:|  :---:|
| [ResNet-34][assets]     | 55.22  |  79.60   | 62.10 | 75.57 |
| [ResNet-101][assets]     | 55.69| 80.00   | 63.07 | 77.87 |
| [DLA-34][assets]     | 55.14|  79.97   | 62.10 | 77.09  |

“F1@50” refers to the official metric, i.e., F1 score when IoU threshold is 0.5 between the gt and prediction. "F1@75" is the F1 score when IoU threshold is 0.75.

## Citation

If our paper and code are beneficial to your work, please consider citing:
```
@article{CLRmatchNet,
  title={CLRmatchNet: Enhancing Curved Lane Detection with Deep Matching Process},
  author={S. Kontente, R. Orfaig and B. Bobrovsky},
  journal={arXiv preprint arXiv:2309.15204},
  year={2023}
}
```

## Acknowledgement
<!--ts-->
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->