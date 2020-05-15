# LatentDisc
This repository contains sampling code for the 'Unsupervised Deep Shape Descriptor with Point Distribution Learning' accepted by CVPR2020. It contains implemetation details and important infomation for training and testing. 

## Data
For convenience, in our context, ShapeNet refers to the subset of the whole ShapeNet database which contains 15,011 3D point clouds belongings to 16 categories. In official data split, Dataset is split to 12,137 samples for training set and 2,874 samples for testing set. In comparison, ShapeNet55 refers to ShapeNetCore55 which is a subset contains 55 categories and 57,000 data in total.  

### Training data download:

'wget '

Training and testing setting
----------------------
During model training, we follow the idea of proving out-of-category capability, so that we choose 7 major (most populous data) categories from ShapeNet. Beware that the learning rate of the descriptors should be higher so that it can capture information rather than overfitting the model itself.       
During testing (descriptor calculation), the descriptor obtained in the model training will be discarded. The learning rate of the descriptor will be set higher than previous stage for fast convergence while the model parameters remain fixed. The batch size should be one.

The ModelNet40 evaluation
----------------
It is performed on the calculated descriptors that stand for shapes in ModelNet40 during testing (descriptor calculation). As for the training and testing data setting during evaluation, we follow the exact same train/test split included in the dataset as 3DGAN.

Dependencies
-----------------
We use Pytorch 1.3 for our model implementation. The evaluation is performed using sklearn and numpy. 

Paper
----------------
#todo

Reference
---------------
#todo


