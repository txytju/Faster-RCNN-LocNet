# Improved Localization Accuracy by LocNet for Faster R-CNN 

## 1. Introduction

![](http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-LocNet-FasterRCNN.001.jpeg)

This project is a Simplified Faster R-CNN **improved by LocNet** (**Loc-Faster-RCNN** for short) implementation based on [Faster R-CNN by chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). It aims in:

- Improve the localization accuracy of Faster R-CNN by using LocNet in the Fast R-CNN part.
- The first public implementation of the [origin paper](https://ieeexplore.ieee.org/abstract/document/8270086/). The author of the paper didn't release their version.
- Match the performance reported in original paper.

And it has the following features:

- It can be run as pure Python code, no more build affair. (cuda code moves to cupy, Cython acceleration are optional)

This implementation is slightlly different from the original paper:

- Skip pooling is not used here. Informations from conv5_3 layer(the feature map of original Faster R-CNN) is enough for my task, so skip pooling is droped in this repo.
- The RPN net is exactly same as Faster R-CNN, which means onely 3X3 conv is applied, rather than 3X3 and 5X5 conv nets in the original paper.
- Training strategy. The original paper train the RPN and LocNet alternately, but losses of RPN and LocNet are backproped at the same time.

**prob_thre** :

- Hyperparameters in Loc-Faster-RCNN are mostly like Faster R-CNN except for **prob_thre**. 
- prob_thre is the threshold of probability used when predicting the bounding box, if px or py is greater than prob_thre, this row or column is considered to be part of some object.
- Different detection tasks may have different appropriate prob_thre to achive best performance. If most objects in the detection task are dense blocks, a higher prob_thre may achive better performance. 
- You can choose your own prob_thre according to your task characteristics.

## 2. Performance

### 2.1 Pascal VOC

#### 2.1.1 mAP

The best prob_thre for Pascal VOC is 0.5. When using prob_thre=0.5, the performance of Loc-Faster-RCNN is listed as follows. So with dataset like Pascal VOC, Loc-Faster-RCNN can not achieve better result than Faster R-CNN.

| Implementation  |  mAP   |
| :-------------: | :----: |
| Loc-Faster-RCNN | 0.6527 |
|  Faster R-CNN   | 0.7097 |

#### 2.1.2 Differences between models predictions in Pascal VOC

- LocNet part improves localization accuracy of  Loc-Faster-RCNN by predicting the probability rather than locations. This helps when models is used to detection small objects or objects that are not so obvious. Like shown in the first 2 rows below, Loc-Faster-RCNN detected a person(row 1) and plant(row 2) even the objects are too small and not obvious.
- However, LocNet part also hinders the model from identifying small parts of objects,which are more densely connected with the background rather than the main part of that object, like tail of a cat or wings of a bird ,as shown in the 3th~5th rows bellow.
- What's more, if objects are overlaping or densely connected with each other in the same image, Loc-Faster-RCNN also have difficulty in drawing accurate bounding boxes around objects, as shown in the last row bellow.

|                         Ground Truth                         |                       Loc-Faster-RCNN                        |                         Faster R-CNN                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-1.0.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-1.1.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-1.2.png" width="300" />  </p> |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-2.0.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-2.1.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-2.2.png" width="300" />  </p> |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-3.0.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-3.1.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-3.2.png" width="300" />  </p> |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-4.0.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-4.1.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-4.2.png" width="300" />  </p> |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-5.0.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-5.1.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-5.2.png" width="300" />  </p> |
| <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-6.0.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-6.1.png" width="300" />  </p> | <p float="left"> <img src="http://p3rz3gu1u.bkt.clouddn.com/2018-06-22-6.2.png" width="300" />  </p> |

### 2.2 Text detection dataset 

ICDAR-2011 and ICDAR-2013 are used in training and eveluating.

## 3. Install dependencies

This repo is built basically on [Faster R-CNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). You can check this repo to see dependencies.

## 4. Train

VGG16 train on `trainval` and test on `test` split.

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