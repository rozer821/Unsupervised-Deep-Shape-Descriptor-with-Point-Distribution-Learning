# Unsupervised Deep Shape Descriptor with Point Distribution Learning 
This repository contains sampling code for the 'Unsupervised Deep Shape Descriptor with Point Distribution Learning'. 
It contains implemetation details and important infomation for training and testing. The code has just been cleaned, reoredered and separated from a larger project, please make a post in Issues if you have any question about using it.

## Data
On contrary to using the entire ShapeNet subset which contains 55 categories and 57,000 data in total, we follow the same setting as in [3DGAN](http://3dgan.csail.mit.edu/) where only seven categories from the [ShapeNet](https://www.shapenet.org/) are used in training and evaluation is on ModelNet40.

Training: [A subset](https://drive.google.com/open?id=1Pmu9e70uKBvxgBYbjU8GxuzLY3xWPc0x) consist of 7 categories from ShapeNet.    
Testing: [ModelNet40 Aligned](https://modelnet.cs.princeton.edu/)  

In our ablation study, we conduct the evalution on the [ShapeNet set](https://drive.google.com/open?id=1Pmu9e70uKBvxgBYbjU8GxuzLY3xWPc0x) includes 16 categories and 15000 data in total.

Training and testing Details
----------------------
During model training, we follow the idea of proving out-of-category capability, so that we choose 7 major categories from ShapeNet. Beware that the learning rate of the descriptors should be higher so that it can capture information rather than overfitting the decoder model itself.    
   
During testing (descriptor calculation), the descriptor obtained in the model training will be discarded. The learning rate of the descriptor will be set higher than previous stage for fast convergence while the model parameters remain fixed. The batch size should be one.

Dependencies
-----------------
We use Pytorch 1.3 for our model implementation.  The evalution is performed with default function provided by sklearn.

-matplotlib 

-numpy 

-sklearn 

-open3d  

Website
----------------
#tolink

Paper
----------------
[link](https://drive.google.com/open?id=1OjtTes9h4y1X0_yZNrWCmUTsuXgI4Ev1)

Reference
---------------
@article{unded2020, 
  title={Unsupervised Deep Shape Descriptor with Point Distribution Learning},  
  author={Shi, Yi and Xu, Mengchen and Yuan, Shuaihang and Fang, Yi},  
  journal={},   
  volume={},  
  number={},  
  pages={},   
  year={2020}, 
}


