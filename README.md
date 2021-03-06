# Unsupervised Deep Shape Descriptor with Point Distribution Learning 
&nbsp; &nbsp;This repository contains sampling code for the 'Unsupervised Deep Shape Descriptor with Point Distribution Learning'. This README file contains data details and important infomation for training and testing. The code has just been reoredered and separated from a larger project. Please make a post in Issues if you have encountered any question.

Useful Links
----------------
### [Paper](https://drive.google.com/open?id=1OjtTes9h4y1X0_yZNrWCmUTsuXgI4Ev1)&nbsp;
### [Slides](https://drive.google.com/open?id=14K5LHh_mtf7znlY2Re83OcQicpG-eF1C)&nbsp;
### [Teaser](https://drive.google.com/open?id=1cSuHC03yJhB2QISNpjMzAtYVPydVXkqi)&nbsp;
### [Lab](https://wp.nyu.edu/mmvc/publications/)

## Overview
&nbsp; &nbsp;This work focuses on unsupervised 3D point cloud descriptor/feature computation. The proposed learning based approach treats each point as a Gaussian and introduces an operation called 'Gaussian Sampling' which applies multiple 'disruptions' to each point. Then an auto-decoder model is leveraged to model a Maximum Likelihood Estimation process where the parameters of each point Gaussian. (i.e guess the original location of each point), through which the geometric information of the shape is learned. 
    
<p align="center">
<img src="imgs/Slide6.jpg" width="500">
</p>
   
## Data and Experiment
&nbsp; &nbsp; On contrary to using the entire [ShapeNet55](https://www.shapenet.org/) subset which contains 55 categories and 57,000 data, we follow the setting as in [3DGAN](http://3dgan.csail.mit.edu/) where only seven categories from the [ShapeNet](https://www.shapenet.org/) are used in training and evaluation is on the entire ModelNet40. We provide processed partial data in the following link:

Training: [A subset](https://drive.google.com/open?id=1Pmu9e70uKBvxgBYbjU8GxuzLY3xWPc0x) consist of 7 categories from ShapeNet.    
Evaluation: [ModelNet40 Aligned](https://modelnet.cs.princeton.edu/)  

<p align="center">
<img src="imgs/Slide9.jpg" width="680">
</p>

In the ablation study (reconstruction, multiscale, roatation & noise invariance), the experiment is conducted on our evaluation set containing 16 categories of [ShapeNet](https://drive.google.com/open?id=1Pmu9e70uKBvxgBYbjU8GxuzLY3xWPc0x). The evaluation for classification uses a random train/test split of 8:2. The classifier used in the final evaluation of computed descriptors is an MLP.

<p float="left">
<img src="imgs/Slide10.jpg" width="430"/>
<img src="imgs/Slide12.jpg" width="430"/>
</p>

Training and testing Details
----------------------
The entire pipline involves two phases: decoder model training and descriptor computation. 

<p align="center">
<img src="imgs/Slide15.jpg" width="500">
</p>

During model training, we use 7 major categories from ShapeNet. Beware that the learning rate for the descriptors should be higher than that with the decoder so that descriptors are forced to capture information rather than overfitting the decoder model itself.    
   
During descriptor computation, the descriptors obtained in the previous model training will be discarded. The learning rate of the descriptor will be set higher than previous stage for fast convergence while the model parameters remain fixed. 

For each dataset involved, the hyper parameters should be tuned for the optimal performance. The evaluation over generated descriptors is performed with the default function provided by sklearn. 

Dependencies
-----------------
We use Pytorch 1.3 for our model implementation.  

-matplotlib 

-numpy 

-sklearn 

-open3d  

Reference
---------------
@InProceedings{Xu_Shi_2020_CVPR,  
&nbsp; &nbsp;author = {Shi, Yi and Xu, Mengchen and Yuan, Shuaihang and Fang, Yi},  
&nbsp; &nbsp;title = {Unsupervised Deep Shape Descriptor With Point Distribution Learning},  
&nbsp; &nbsp;booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
&nbsp; &nbsp;month = {June},  
&nbsp; &nbsp;year = {2020}  
}


