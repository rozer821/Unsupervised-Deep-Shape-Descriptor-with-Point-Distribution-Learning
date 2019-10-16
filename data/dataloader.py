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
    #print(cur_off)
    #xs = mesh.vertices[:,0].ravel()
    #ys = mesh.vertices[:,1].ravel()
   # zs = mesh.vertices[:,2].ravel()
    return np.array(mesh.vertices)

def read_off(cur_off,sample_num):
    file_off = open(cur_off,'r')
    if 'OFF' != file_off.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file_off.readline().strip().split(' ')])
    verts = [[float(s) for s in file_off.readline().strip().split(' ')] for i_vert in range(n_verts)]
    #faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    print(cur_off,verts.len())
    try: 
        indeces = random.sample(range(len(verts)),sample_num);
        verts = [verts[index] for index in indeces]
        return torch.FloatTensor(verts)
    except ValueError:
        return -1

def show_mesh(pc):
    #print(len(verts),len(verts[0]))
    pass

def generate_random(pc,device,sigma=0.5):
    # pc [N*3] 
    # sigma [1],for controlling the random distance
    # pc_gen [N*3]
    std = sigma
    pc_gen = torch.zeros(0).to(device)
    for pt in pc:  
        #pt [1*3] x y z
        pt_new = torch.normal(pt, std, out=None).unsqueeze(-1).to(device).transpose(0,1)
        pc_gen = torch.cat([pc_gen, pt_new], 0)
        
        #z=multivaria.pdf(xy, mean=mu, cov=covariance) 
    return pc_gen
               
class ModelNet_aligned(Dataset):
    def __init__(self,root,device,mode='train',downsample_num=1024):
        self.root = root
        self.offs = glob.glob(root+'/*/' + mode + '/*.off')
        self.meshes_gt = torch.zeros(0).to(device)
        self.meshes_gen = torch.zeros(0).to(device)
        
        #self.annots = []
        #self.annots_cls=[]
        total_num = 0
        for file_name in self.offs:
            
            pc = read_off(file_name,downsample_num).to(device)
            if pc == -1:
                continue
            self.meshes_gt = torch.cat([self.meshes_gt,pc.unsqueeze(0)]) 
            print('gt:',pc.size(),self.meshes_gt.size())
            
            pc_gen = generate_random(pc,device)
            self.meshes_gen = torch.cat([self.meshes_gen,pc_gen.unsqueeze(0)])
            print('gen:',pc_gen.size(),self.meshes_gen.size())
            total_num += 1                                                                      
            #annot_line = open(file_name+'.annot','r').readlines()
            #annot_angle = [eval(x.strip('\n')) for x in annot_line]
            
            #self.annots.append(annot_angle) #useless annots
            #self.annots_cls.append(os.path.split(file_name))
        self.indices = range(total_num)
        #print(self.meshes_gen.size(),self.meshes_gt.size())
        
    def __getitem__(self,index):
        return self.meshes_gen[index],self.meshes_gt[index],self.indices[index]

    def __len__(self):
        return len(self.offs)
   
