# Unified 2D Human Pose Estimation

There exist multiple datasets for 2D human pose estimation, each with its unique set of annotated body joints.Generally, each neural network is trained on a particular dataset.This approach often results in networks that excel on their specific dataset but struggle when tested on others. Thus, there rarely exists a network which can perform well on most of the datasets. Using the concept of knowledge distillation, we have used a group of pre-trained teachers networks (COCO teacher model & MPII teacher model) to help create a single, adaptable student network capable of estimating poses across dataset (COCO as well as MPII).
## Installation

Entire project is built using python 3.11

```bash
  python3.11 -m venv venv

  source venv/bin/activate

  pip install -r requirements.txt
  
  mkdir data
```
Download COCO and MPII pose estimation datasets in 'data' directory.
    
## About

In this work, we used MMPose - an open-source toolbox for pose estimation based on PyTorch. It includes scripts for model and dataset configurations which makes it easier for experimentation.

Moreover, we have used MMPose based Real-Time Models for Pose Estimation (RTMPose), a model architecture with SimCC based algorithm that treats keypoint localization as a classification task. It uses a top down approach for pose estimation with CSPNeXt as backbone for object detection and SimCC as prediction head of architecture.

Directory description:
```bash
demo : includes sample images, model inference scipt etc
mmpose : cloned MMPose repo on top of which we have added our model and dataset configuration
results : visualization results
teachers : contains COCO and MPII teacher models for RTMPose-m
utils : utility scripts
```




## Usage

Training a model-
```bash
python3.11 mmpose/tools/train.py <LOCATION OF MODEL CONFIG>
```
Visualization using trained model on an image
```bash
python demo/inferencer_demo.py <IMAGE_PATH> --pose2d <MODEL CONFIG_PATH> --pose2d-weights <MODEL CHECKPOINT> --vis-out-dir <O/P DIR> --radius 4 --thickness 2
```
Major contributation of this project for performing knowledge distillation:
```
Distilling RTMPose-s using RTMPose-m teacher models trained on COCO and MPII : mmpose/configs/body_2d_keypoint/multiteacher/coco/distill_config_small.py

Distilling RTMPose-t using RTMPose-m teacher models trained on COCO and MPII : mmpose/configs/body_2d_keypoint/multiteacher/coco/distill_config_tiny.py

Combined dataset config for COCO and MPII : mmpose/configs/_base_/datasets/coco_mpii.py

RTMPose-t config for training on MPII and COCO without knowledge Distillation : mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_mpii-coco-256x192.py

RTMPose-s config for training on MPII and COCO without knowledge Distillation : mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_mpii-coco-256x192.py
```





