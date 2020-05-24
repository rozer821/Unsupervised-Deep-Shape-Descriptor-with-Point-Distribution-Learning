# Unsupervised Deep Shape Descriptor with Point Distribution Learning 
This repository contains sampling code for the 'Unsupervised Deep Shape Descriptor with Point Distribution Learning'. 
It contains implemetation details and important infomation for training and testing. You are welcome to use it in different tasks, such as segmentation. 

## Data
On contrary to using the entire ShapeNet(55) subset contains 55 categories and 57,000 data in total, we follow the same setting as in [3DGAN](http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf) where only seven categories are used in training and evaluation is on ModelNet40.

Training: [A subset]() consist of 7 categories from ShapeNet.    
Testing: ModelNet40 Aligned  

In our ablation study evalution, we use the [ShapeNet subset]() with 16 categories and 15000 data in total.

Training and testing Details
----------------------
During model training, we follow the idea of proving out-of-category capability, so that we choose 7 majorcategories from ShapeNet. Beware that the learning rate of the descriptors should be higher so that it can capture information rather than overfitting the decoder model itself.    
   
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
#tolink

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


