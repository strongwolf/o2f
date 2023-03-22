# One-to-Few Label Assignment for End-to-End Dense Detection
This repo hosts the code for implementing the [one-to-few label assignemnt](https://arxiv.org/abs/2303.11567), as presented in our CVPR 2023 paper.

## Introduction
One-to-one (o2o) label assignment plays a key role for transformer based end-to-end detection, and it has been recently introduced in fully convolutional detectors for end-to-end dense detection. However, o2o can degrade the feature learning efficiency due to the limited number of positive samples. Though extra positive samples are introduced to mitigate this issue in recent DETRs, the computation of self- and cross- attentions in the decoder limits its practical application to dense and fully convolutional detectors. In this work, we propose a simple yet effective one-to-few (o2f) label assignment strategy for end-to-end dense detection. Apart from defining one positive and many negative anchors for each object, we define several soft anchors, which serve as positive and negative samples simultaneously. The positive and negative weights of these soft anchors are dynamically adjusted during training so that they can contribute more to "representation learning" in the early training stage, and contribute more to "duplicated prediction removal" in the later stage. The detector trained in this way can not only learn a strong feature representation but also perform end-to-end dense detection. Experiments on COCO and CrowdHuman datasets demonstrate the effectiveness of the o2f scheme. 

## Installation
This implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Therefore the installation is the same as original MMDetection.

## Results and Models on COCO Detection
| Backbone     | MS <br> train | Lr <br> schd | box AP <br> (val)  | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:------------:|:-------------:|:------------:|:------------------:|:--------------------------------------:|
| R-50         | N             | 1x           | 39.0               |  [model](https://drive.google.com/file/d/1sKI4V_3h_kY8A6TPD2_iJjCvzD2zoRCz/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1CirouBUz8wdHEwfjwJNEWrdOAlCS-4bI/view?usp=share_link)|
| R-50         | Y             | 3x           | 42.2               |  [model](https://drive.google.com/file/d/1OvFsRpHdEVKOWrBM1y21_y7YAxjkSC5l/view?usp=share_link) &#124; [log](https://drive.google.com/file/d/1zDRgOGZ2cb_TqD0axifgODHYTPcbPpei/view?usp=share_link)|
| R-101        | N             | 1x           | 41.0               |  [model](https://drive.google.com/file/d/1kdI_QoXWCwWDDtJhXLehvh7b7aH4AF_n/view?usp=share_link) &#124; [log](https://drive.google.com/file/d/1FWSGc6ZqSPLZJRkThIELKiuv0x0SHIO9/view?usp=share_link)|
| R-101        | Y             | 3x           | 43.8               |  [model](https://drive.google.com/file/d/1T-ClVrmDCpps4QlivgLJlGJAmv9WhDSo/view?usp=share_link) &#124; [log](https://drive.google.com/file/d/1AM-vT7CzcGMEpEs2A7-1xFAulHdCWJew/view?usp=share_link)|

## Results and Models on COCOSegmentation
| Backbone     | MS <br> train | Lr <br> schd | box AP <br> (val)  | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:------------:|:-------------:|:------------:|:------------------:|:--------------------------------------:|
| R-50         | N             | 1x           | 35.9               |  [model](https://drive.google.com/file/d/1n1MS-0MaURbqnToxweejj1Gb8EF1c3rP/view?usp=share_link) &#124; [log](https://drive.google.com/file/d/1j8mOuMJ3unl6JHLjLBtSx7t_RBq0Y7Yb/view?usp=share_link)|
| R-50         | Y             | 3x           | 38.0               |  [model](https://drive.google.com/file/d/1vn4GFBSwznNCijU3iDA7Fu39BTTW2VR-/view?usp=share_link) &#124; [log](https://drive.google.com/file/d/1Jm0TyiJqwUIjQNUQMZp0In3Li3FRxXuL/view?usp=share_link)|

## Results and Models on CrowdHuman Detection
| Backbone     |  AP  |  MAR  |  Recall  | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:------------:|:-------------:|:------------:|:------------------:|:--------------------------------------:|
| R-50         | 91.0 | 45.3 | 98.0 | [model](https://drive.google.com/file/d/10qe5s46rY-2zAr6zHF20_k2V2iBpeo_E/view?usp=share_link) &#124; [log](https://drive.google.com/file/d/1dd5JfbcoqC6EaoIw6ZtxZ1ccfQ_90a5E/view?usp=share_link)|

## Inference

Assuming you have put the COCO dataset into `data/coco/` and have downloaded the models into the `weights/`, you can now evaluate the models on the COCO val2017 split:

```
bash dist_test.sh configs/r50_1x.py weights/r50_1x.pth 8 --eval bbox
```

## Training

The following command line will train `r50_fpn_1x_coco` on 8 GPUs:

```
bash dist_train.sh configs/r50_1x.py 8 --work-dir weights/r50_1x
```

## Citation
```
@inproceedings{shuai2023o2f,
  title={One-to-Few Label Assignment for End-to-End Dense Detection},
  author={Li, Shuai and Li, Minghan and Li, Ruihuang and He, Chenhang and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```