import os
import glob
import random
import numpy as np
import open3d as o3d
import torch
#from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
#import utils
    
def open_mesh(filename):
    pc = o3d.read_point_cloud(filename)
    return pc

def get_mean_dis(pc):
    center = np.sum(pc)
    mean_dis = np.mean(pc-center)
    print(mean_dis)
    return mean_dis

def show_mesh(pc):
    pass


def generate_random(pc,sigma):
    # pc [N*3] 
    # sigma [1],for controlling the random distance
    std = sigma
    pc_new = []
    for pt in pc:  
       #pt [1*3] x y z
       pt_new = torch.normal(means, std, out=None)
       pc_new.append(pt_new)
       #z=multivaria.pdf(xy, mean=mu, cov=covariance) 
    return pc_new
               
class ModelNet_aligned(Dataset):
    def __init__(self,root,mode='train',subsample_rate=None):
        self.root = root
        self.offs = glob.glob(root+'/*/' + mode + '/*')
        self.meshes = []
        self.annots = []
        self.annots_cls=[]
        for file_name in self.offs:
            annot_line = open(file_name+'.annot','r').readlines()
            annot_angle = [eval(x.strip('\n')) for x in annot_line]
            self.annots.append(annot_angle) #useless annots
            self.meshes.append(open_mesh(f,subsample_rate)) 
            self.annots_cls.append(os.path.split(file_name))
        self.indices = range(len(self.offs))
        
    def __getitem__(self,index):
        return self.meshes[index],self.annots_cls[index],
               self.annots[index],self.indices[index]

    def __len__(self):
        return len(self.offs)
   
