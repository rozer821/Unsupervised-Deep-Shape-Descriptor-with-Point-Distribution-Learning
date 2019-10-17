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
    std = sigma
    pc_gen = torch.zeros(0)
    for pt in pc:  
        #pt [1*3] x y z
        pt_new = torch.normal(pt, std, out=None).unsqueeze(-1).transpose(0,1)
        pc_gen = torch.cat([pc_gen, pt_new], 0)
        #z=multivaria.pdf(xy, mean=mu, cov=covariance) 
    return pc_gen


def gen_mesh(root,mode,downsample_num,sample_gen_per_point, sigmas=[0.5]):
    offs = glob.glob(root+'/*/' + mode + '/*.off')
    for file_path in offs:
        category = file_path.split('/')[-3]
        mode =  file_path.split('/')[-2]
        off_name = os.path.split(file_path)[-1]
        target_file = os.path.join(root,'random_sample_gt_rate_%d'%downsample_num,mode,off_name.split('.')[0]+'.npy')
        pc = read_off(file_path,sample_num = downsample_num)
        if not pc:
            continue
        np.save(target_file,meshes_gen)
        print("saving %s"%target_file)
        
        for s in sigma:
            target_file_gen = os.path.join(root,'gen_sigma_%d'%sigma,mode,off_name.split('.')[0]+'.npy')
            pc_gen = generate_random(pc,s).tolist()
            np.save(target_file_gen,pc_gen)
            print(":-- saving %s"%target_file_gen)    
    
    
class ModelNet_aligned(Dataset):
    def __init__(self,root,device,mode='train',sigmas=[0.5],sample_gen_per_point,downsample_num=1024):
        npys_gen = glob.glob(root+'random_sample_gt_rate_*/*/' + mode + '/*.npy')
        if len(npys_gen)<=0:
            gen_mesh(root,mode,downsample_num,sample_gen_per_point,sigmas=sigmas)
        self.sigmas = sigmas
        self.root = root
        self.device = device
        self.mode = mode
        self.meshes_gt = torch.zeros(0).to(device)
        
        self.meshes_gt_name = npys_gen
        for npy_name in npys_gen:   
            pc = np.load(npy_name)
            pc = torch.FloatTensor(pc).to(device)
            self.meshes_gt = torch.cat([self.meshes_gt,pc.unsqueeze(0)],1)
           
        self.indices = range(len(npys_gen))
        #print(self.meshes_gen.size(),self.meshes_gt.size())
        
    def __getitem__(self,index):
        path_gt = self.meshes_gt_name[index]
        categ = path_gt.split('/')[-3]
        name_gt = path_gt.split('/')[-1]
        template_sigma_file = root+'gen_sigma_%d/'+self.mode+'/'+categ+'/'+name_gt
        meshes_sigmas = torch.zeros(0).to(self.device)
        for sig in self.sigmas:
            npy_sig = np.load(template_sigma_file%sig)
            meshes_sigmas = torch.cat([meshes_sigmas,npy_sig],1)
        
        return self.meshes_gt[index],meshes_sigmas[index],self.indices[index]

    def __len__(self):
        return self.meshes_gt.size()[0]
   
