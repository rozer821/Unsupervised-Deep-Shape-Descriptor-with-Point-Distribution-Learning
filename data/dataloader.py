import os
import glob
import random
import numpy as np
import open3d as o3d
import trimesh
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

def read_off_trimesh(cur_off):
    mesh = trimesh.load(cur_off,process=False)
    print(cur_off)
    #xs = mesh.vertices[:,0].ravel()
    #ys = mesh.vertices[:,1].ravel()
   # zs = mesh.vertices[:,2].ravel()
    return np.array(mesh.vertices)

def read_off(cur_off):
    file_off = open(cur_off,'r')
    if 'OFF' != file_off.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file_off.readline().strip().split(' ')])
    verts = [[float(s) for s in file_off.readline().strip().split(' ')] for i_vert in range(n_verts)]
    #faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    #return verts, faces
    print(len(verts))
    return torch.FloatTensor(verts)

def show_mesh(pc):
    #print(len(verts),len(verts[0]))
    pass

def generate_random(pc,sigma=0.5):
    # pc [N*3] 
    # sigma [1],for controlling the random distance
    std = sigma
    pc_new = []
    for pt in pc:  
        #pt [1*3] x y z
        pt_new = torch.normal(pt, std, out=None)
        pc_new.append(pt_new)
        #z=multivaria.pdf(xy, mean=mu, cov=covariance) 
    return pc_new
               
class ModelNet_aligned(Dataset):
    def __init__(self,root,mode='train',subsample_rate=None):
        self.root = root
        self.offs = glob.glob(root+'/*/' + mode + '/*.off')
        
        self.meshes_gt = []
        self.meshes_gen = []
        
        self.annots = []
        self.annots_cls=[]
        
        for file_name in self.offs:
            annot_line = open(file_name+'.annot','r').readlines()
            annot_angle = [eval(x.strip('\n')) for x in annot_line]
            self.annots.append(annot_angle) #useless annots
            
            pc = read_off(file_name)
            self.meshes_gt.append(pc) 
            
            pc_gen = generate_random(pc)
            self.meshes_gen.append(pc_gen)
            self.annots_cls.append(os.path.split(file_name))
        self.indices = range(len(self.offs))
        
    def __getitem__(self,index):
        return self.meshes_gt[index],self.meshes_gen[index],self.annots_cls[index],self.indices[index]

    def __len__(self):
        return len(self.offs)
   
