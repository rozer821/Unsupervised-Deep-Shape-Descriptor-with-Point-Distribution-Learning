import os
import glob
import random
import numpy as np
import open3d as o3d
import tqdm
import torch
import math
from copy import deepcopy


#from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
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


def gen_mesh(root,mode,downsample_num,sample_gen_per_point, sigmas =[]):
    #print(root+'/*/' + mode + '/*.off')
    offs = glob.glob(root+'/*/' + mode + '/*.off')
    gen_path = os.path.split(root)[0]+'/gen'
    parent = os.path.join(gen_path,'random_sample_gt_rate_%d'%downsample_num,mode)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    #print(len(offs),'dsss')
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

def generate_random(pc,sigma=0.02):
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

def get_margin(pc):
    x_limit = [min(pt_file[:,0]),max(pt_file[:,0])]
    y_limit = [min(pt_file[:,1]),max(pt_file[:,1])]
    z_limit = [min(pt_file[:,2]),max(pt_file[:,2])]
    return x_limit,y_limit,z_limit
    
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNet_aligned(Dataset):
    def __init__(self,root,device,mode='train',sigmas =[0.25],sample_gen_per_point=1,downsample_num=1024):
        #print(os.path.join(root,'random_sample_gt_rate_*/' + mode + '/*.npy'))
        self.downsample_num = downsample_num
        self.root = os.path.split(root)[0]+'/gen'
        sample_conf = 'random_sample_gt_rate_%d'%self.downsample_num
        genned_file_path = os.path.join(self.root, sample_conf, mode)
        print(genned_file_path)
        self.npys_gen = glob.glob(os.path.join(genned_file_path,'*.npy'))
            
        self.sigmas = sigmas
        self.device = device
        self.mode = mode
        self.categ = [os.path.split(i)[-1] for i in glob.glob('/home/mmvc/mmvc-ny-nas/Yi_Shi/data/modelnet40_off_aligned/*')]
        
        self.names_instance = [os.path.split(i)[-1].split('.')[0] for i in self.npys_gen]
        self.categs_instance = ['_'.join(i.split('_')[:-1]) for i in self.names_instance]
        self.indices = range(len(self.npys_gen))
        
        
        
    def __getitem__(self,index):
        meshes_gt = np.load(self.npys_gen[index])
        pc_gt_normalized = pc_normalize(meshes_gt)
        #template_sigma_file = os.path.join(self.gen_path,'%d_gen_sigma_%.1f',self.mode,name_gt)
        sig = self.sigmas[random.randint(0,len(self.sigmas)-1)]
        #if not is_online:
        #    pc = os.path.join(self.gen_path,'%d_gen_sigma_%.1f',self.mode, name_gt)
        
        pc = generate_random(pc_gt_normalized,sigma=sig).tolist()
        pc_gt_normalized = torch.FloatTensor(pc_gt_normalized).transpose(0,1)
        pc_gen = torch.FloatTensor(pc).transpose(0,1) 
            
        #print(meshes_gt_repeat.size(),meshes_sigmas.size(),self.indices[index])
        return pc_gen,pc_gt_normalized,self.indices[index]
    
    def __len__(self):
        return len(self.npys_gen)
    
    
class ModelNet_aligned_rotate(Dataset):
    def __init__(self,root,device,mode='train',sigmas =[0.25],sample_gen_per_point=1,downsample_num=1024):
        #print(os.path.join(root,'random_sample_gt_rate_*/' + mode + '/*.npy'))
        self.downsample_num = downsample_num
        self.root = os.path.split(root)[0]+'/gen'
        sample_conf = 'random_sample_gt_rate_%d'%self.downsample_num
        genned_file_path = os.path.join(self.root, sample_conf, mode)
        print(genned_file_path)
        self.npys_gen = glob.glob(os.path.join(genned_file_path,'*.npy'))
            
        self.sigmas = sigmas
        self.device = device
        self.mode = mode
        self.categ = [os.path.split(i)[-1] for i in glob.glob('/home/mmvc/mmvc-ny-nas/Yi_Shi/data/modelnet40_off_aligned/*')]
        
        self.names_instance = [os.path.split(i)[-1].split('.')[0] for i in self.npys_gen]
        self.categs_instance = ['_'.join(i.split('_')[:-1]) for i in self.names_instance]
        self.indices = range(len(self.npys_gen))
        
    def __getitem__(self,index,is_online=True):
        meshes_gt = np.load(self.npys_gen[index])
        x = deepcopy(meshes_gt[:, 0])
        y = deepcopy(meshes_gt[:, 1])
        angle = math.pi/4
        meshes_gt[:, 0] = math.cos(angle)*x -math.sin(angle)*y
        meshes_gt[:, 1] = math.cos(angle)*y + math.sin(angle)*x
        pc_gt_normalized = pc_normalize(meshes_gt)
        #template_sigma_file = os.path.join(self.gen_path,'%d_gen_sigma_%.1f',self.mode,name_gt)
        sig = self.sigmas[random.randint(0,len(self.sigmas)-1)]
        if not is_online:
            pc = os.path.join(self.gen_path,'%d_gen_sigma_%.1f',self.mode, name_gt)
        else:
            pc = generate_random(pc_gt_normalized,sigma=sig).tolist()
            
          
        pc_gt_normalized = torch.FloatTensor(pc_gt_normalized).transpose(0,1)
        pc_gen = torch.FloatTensor(pc).transpose(0,1) 
            
        #print(meshes_gt_repeat.size(),meshes_sigmas.size(),self.indices[index])
        return pc_gen,pc_gt_normalized,self.indices[index]
    
    def __len__(self):
        return len(self.npys_gen)
    

   
