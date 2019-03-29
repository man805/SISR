# Single Image Super-Resolution Network (SISR)
Freshmen assignment1

Implementation and Verification of deep learning based Single Image Super-Resolution Network (SISR)

## Abstract
Recently, several models based on deep neural networks have achieved great success in terms of both reconstruction accuracy and computational performance for single image super-resolution. In these methods, the low resolution (LR) input image is upscaled to the high resolution (HR) space using a single filter, commonly bicubic interpolation, before reconstruction. This means that the super-resolution (SR) operation is performed in HR space. We demonstrate that this is sub-optimal and adds computational complexity. In this paper, we present the first convolutional neural network (CNN) capable of real-time SR of 1080p videos on a single K2 GPU.

## Original Paper
Wenzhe Shi et al., Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, CoRR, 2016

## Implementation Details
Convolutional layers
Adam Optimizer
L2 Loss
Pixel shuffle layer
Training patch size: 64x64, batch size: 16
The following SISR structure

## MSCOCO2017 Toy Dataset Download links:
Training Samples: 4,900 Color images (About 500x500)
https://drive.google.com/file/d/1wlg5hUYrLixZ9TyWbzaZbFNHuJeQCWqu/view?usp=sharing
Validation Samples: 100 Color images (About 500x500)
https://drive.google.com/file/d/1Grqrz9aZ-c-LlTAXbi3lbKt1IAtz551l/view?usp=sharing

## Requirement     
matplotlib           3.0.3         
tensorboard          1.13.1     
tensorboardX         1.6          
tensorflow-gpu       1.13.1     
torch                1.0.1.post2
torchvision          0.2.2.post3

## Steps
1. put this folders in data folder
2. run resize.py with 'python resize.py'
3. make 'models' and 'logs' folder
4. run train.py with 'python train.py --learning_rate {0.001} --model_name {0.001}'  (value 0.001 can change)
5. run tensorboard with 'tensorboard --logdir=./logs'
6. run sample.py with 'python sample.py --model_path ./models/0.001/SISR-{epoch}.ckpt'

