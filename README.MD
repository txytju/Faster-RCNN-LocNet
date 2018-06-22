# Improved Localization Accuracy by LocNet for Faster R-CNN 

## 1. Introduction

![](http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-LocNet-FasterRCNN.001.jpeg)

This project is a Simplified Faster R-CNN **improved by LocNet** implementation based on [Faster R-CNN by chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). It aims to:

- Improve the localization accuracy of Faster R-CNN by using LocNet in the Fast R-CNN part.
- Complete the first public implementation of the [origin paper](https://ieeexplore.ieee.org/abstract/document/8270086/). The author of the paper didn't release their version.
- Match the performance reported in origin paper.

And it has the following features:

- It can be run as pure Python code, no more build affair. (cuda code moves to cupy, Cython acceleration are optional)

This implementation is slightlly different from the original paper:

- Skip pooling is not used here. Informations from conv5_3 layer(the feature map of original Faster R-CNN) is enough for my task, so skip pooling is droped in this repo.
- The RPN net is exactly the same as Faster R-CNN, which means onely 3X3 conv is applied, rather than 3X3 and 5X5 conv nets in the original paper.
- Training strategy. The original paper train the RPN and LocNet alternately, but losses of RPN and LocNet are backproped at the same time.

## 2. Performance

### 2.1 Pascal VOC

VGG16 train on `trainval` and test on `test` split.

### 2.2 Text detection dataset 

ICDAR-2011 and ICDAR-2013 are used in training and eveluating.

## 3. Install dependencies

This repo is built basically on [Faster R-CNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). You can check this repo to see dependencies.

## 4. Train



## Troubleshooting



## More

- [x] model structure
- [ ] maybe : skip pooling
- [ ] Maybe : conv 3X3 and conv 5X5 in RPN

## Acknowledgement

This work builds on many excellent works, which include:

- [Faster R-CNN by chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch), on which this repo is built on. The best implementation of Faster R-CNN in Pytorch I've ever seen.

- [LocNet by the paper author](https://github.com/gidariss/LocNet).

  

***

Licensed under MIT, see the LICENSE for more detail.

Contribution Welcome.

If you encounter any problem, feel free to open an issue.

Correct me if anything is wrong or unclear.