import os
import glob
import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import open3d as o3d

import utils
eps = 1e-6

def find_valid_points(point_cloud):
    valid_points = point_cloud[:,:,:,-1] > eps
    return valid_points

#method->simple,poisson
def generate_random_pc(point_cloud,method='simple'):
    num_pc = point_cloud.size()[0]*2
    limit_size = [-1,1]
    
    if method=='simple':
        x = torch.randn(num_pc, 3)*2-torch.ones(num_pc, 3)
    elif method=='poisson':
        limit_dis = 0.00001
        pd = PoissonDisk3D(num_pc,limit_dis,limit_size=[-1,1])
        pass

class pcl_loader(Dataset):
    def __init__(self,root,mode='train',subsample_rate=20,depth_scale=2000):
        self.root = root
        self.offs = glob.glob(root+'*/' + mode +
        
            point_clouds.append(current_point_cloud)
           
    def __getitem__(self,index):
        pcd = self.point_clouds[index,:,:,:]  # <HxWx3>
        valid_points = self.valid_points[index,:]
        return pcd,valid_points,self.lat_indexes[index]

    def __len__(self):
        return self.n_pc
   