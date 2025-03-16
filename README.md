# Content-decoupled Contrastive Learning-based Implicit Degradation Modeling for Blind Image Super-Resolution

Created by [Jiang Yuan](), [Ji Ma](), [Bo Wang](), [Weiming Hu]()

[[arXiv]](https://arxiv.org/abs/2408.05440) [[supp]]()

This repository contains PyTorch implementation for __Content-decoupled Contrastive Learning-based Implicit Degradation Modeling for Blind Image Super-Resolution__ (Accepted by TIP).

## 🔥Abstract
Implicit degradation modeling-based blind super-resolution (SR) has attracted more increasing attention in the community due to its excellent generalization to complex degradation scenarios and wide application range. How to extract more discriminative degradation representations and fully adapt them to specific image features is the key to this task. In this paper, we propose a new Content-decoupled Contrastive Learning-based blind image super-resolution (CdCL) framework following the typical blind SR pipeline.This framework introduces negative-free contrastive learning technique for the first time to model the implicit degradation representation, in which a new cyclic shift sampling strategy is designed to ensure decoupling between content features and degradation features from the data perspective, thereby improving the purity and discriminability of the learned implicit degradation space. In addition, we propose a detail-aware implicit degradation adapting module that can better adapt degradation representations to specific LR features by enhancing the basic adaptation unit's perception of image details, significantly reducing the overall SR model complexity. Extensive experiments on synthetic and real data show that our method achieves highly competitive quantitative and qualitative results in various degradation settings while obviously reducing parameters and computational costs, validating the feasibility of designing practical and lightweight blind SR tools.

## 🔥Overview


## 🔥Requirements


## 🔥Train
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `your_data_path/DF2K/HR` to build the DF2K dataset. 

### 2. Begin to train
Run `./main.sh` to train on the DF2K dataset. Please update `dir_data` in the bash file as `your_data_path`.


## 🏰 Model Zoo

## 🔥Test
### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `your_data_path/benchmark`.


### 2. Begin to test
Run `./test.sh` to test on benchmark datasets. Please update `dir_data` in the bash file as `your_data_path`.



## 🔥Visualization of Degradation Representations



## 🔥Comparative Results
### Noise-Free Degradations with Isotropic Gaussian Kernels

### General Degradations with Anisotropic Gaussian Kernels and Noises

## 🔥Citation
```
@article{yuan2024content,
  title={Content-decoupled Contrastive Learning-based Implicit Degradation Modeling for Blind Image Super-Resolution},
  author={Yuan, Jiang and Ma, Ji and Wang, Bo and Hu, Weiming},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```
