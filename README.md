# Content-decoupled Contrastive Learning-based Implicit Degradation Modeling for Blind Image Super-Resolution

Created by [Jiang Yuan](https://github.com/Fieldhunter), [Ji Ma](https://github.com/MJ-NCEPU), [Bo Wang](https://github.com/wangbo2016), Weiming Hu

This repository contains PyTorch implementation for Content-decoupled Contrastive Learning-based Implicit Degradation Modeling for Blind Image Super-Resolution (Accepted by IEEE TIP 2025).
## Train
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `./datasets/DF2K/HR` to build the DF2K dataset. 

### 2. change degradation config

Change the TODO section in `main.sh`, `option.py` and `trainer.py` to select the corresponding degradation settings.

### 3. Begin to train
Run `main.sh` to train on the DF2K dataset.

## Test
### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `./datasets/benchmark`.

### 2. change degradation config

Change the TODO section in `test.sh`, `option.py` and `trainer.py` to select the corresponding degradation settings.

### 3. Begin to test
Run `test.sh` to test on benchmark datasets.

## Citation
```

```
