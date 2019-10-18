import os
import glob
import random
import numpy as np
import open3d as o3d
import tqdm
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


def read_off(cur_off,sample_num=None):
    file_off = open(cur_off,'r')
    if 'OFF' != file_off.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file_off.readline().strip().split(' ')])
    verts = [[float(s) for s in file_off.readline().strip().split(' ')] for i_vert in range(n_verts)]
    #faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    #print(cur_off,len(verts))
    if sample_num==None:
        return verts
    try: 
        indeces = random.sample(range(len(verts)),sample_num);
        verts = [verts[index] for index in indeces]
        return verts
    except ValueError:
        return None

def show_mesh(pc):
    #print(len(verts),len(verts[0]))
    pass

def generate_random(pc,sigma=0.5):
    # pc [N*3] 
    # sigma [1],for controlling the random distance
    # pc_gen [N*3]
    pc_gen = torch.zeros(0)
    for pt in pc:  
        #pt [1*3] x y z
        pt = torch.FloatTensor(pt)
        pt_new = torch.normal(pt, sigma, out=None).unsqueeze(-1).transpose(0,1)
        pc_gen = torch.cat([pc_gen, pt_new], 0)
        #z=multivaria.pdf(xy, mean=mu, cov=covariance) 
    return pc_gen


def gen_mesh(root,mode,downsample_num,sample_gen_per_point, sigmas=[0.5]):
    offs = glob.glob(root+'/*/' + mode + '/*.off')
    parent = os.path.join(root,'random_sample_gt_rate_%d'%downsample_num,mode)
    if not os.path.isdir(parent):
        os.makedirs(parent)
        
    for file_path in tqdm.tqdm(offs):
        category = file_path.split('/')[-3]
        mode =  file_path.split('/')[-2]
        off_name = os.path.split(file_path)[-1].split('.')[0]
       
        target_file = os.path.join(parent,off_name+'.npy')
        pc = read_off(file_path,sample_num = downsample_num)
        if not pc:
            continue
        np.save(target_file,pc)
        #print("saving %s"%target_file)
        
        for s in sigmas:
            parent_s = os.path.join(root,'gen_sigma_%.1f'%s,mode)
            if not os.path.isdir(parent_s):
                os.makedirs(parent_s)
            target_file_gen = os.path.join(parent_s,off_name+'.npy')
            pc_gen = generate_random(pc,sigma=s).tolist()
            np.save(target_file_gen,pc_gen)
            #print(":-- saving %s"%target_file_gen)    
    
    
class ModelNet_aligned(Dataset):
    def __init__(self,root,device,mode='train',sigmas=[0.5],sample_gen_per_point=1,downsample_num=1024):
        print(os.path.join(root,'random_sample_gt_rate_*/' + mode + '/*.npy'))
        self.npys_gen = glob.glob(os.path.join(root,'random_sample_gt_rate_*' , mode ,'*.npy'))
        self.npys_drift = glob.glob(os.path.join(root,'gen_sigma_*' , mode ,'*.npy'))
        
        if len(self.npys_gen)<=0 or len(self.npys_drift)<=0:
            gen_mesh(root,mode,downsample_num,sample_gen_per_point,sigmas=sigmas)
            
        self.npys_gen = glob.glob(os.path.join(root,'random_sample_gt_rate_*' , mode ,'*.npy'))    
        self.sigmas = sigmas
        self.root = root
        self.device = device
        self.mode = mode
        self.meshes_gt = torch.zeros(0)
        
        for npy_name in self.npys_gen:   
            pc = np.load(npy_name)
            pc = torch.FloatTensor(pc)
            self.meshes_gt = torch.cat([self.meshes_gt,pc.transpose(0,1).unsqueeze(0)],0)
           
        self.indices = range(len(self.npys_gen))
        print(len(self.indices),self.meshes_gt.size())
        
    def __getitem__(self,index):
        path_gt = self.npys_gen[index]
        categ = path_gt.split('/')[-3]
        name_gt = path_gt.split('/')[-1]
        template_sigma_file = os.path.join(self.root,'gen_sigma_%.1f',self.mode,name_gt)
        meshes_sigmas = torch.zeros(0)
        for sig in self.sigmas:
            npy_sig = torch.FloatTensor(np.load(template_sigma_file%sig))
            meshes_sigmas = torch.cat([meshes_sigmas,npy_sig.transpose(0,1).unsqueeze(0)],0)
        meshes_gt_repeat = self.meshes_gt[index].unsqueeze(0).repeat(meshes_sigmas.size()[0],1,1)
        #print(meshes_gt_repeat.size(),meshes_sigmas.size(),self.indices[index])
        return meshes_gt_repeat,meshes_sigmas,self.indices[index]

    def __len__(self):
        return len(self.npys_gen)
   
