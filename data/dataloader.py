import os
import glob
import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import open3d as o3d

import utils

def find_valid_points(local_point_cloud):
    eps = 1e-6
    valid_points = local_point_cloud[:,:,:,-1] > eps
    return valid_points

class pcl_loader(Dataset):

    def __init__(self,root,mode='train',subsample_rate=20,depth_scale=2000):
        self.root = root
        self.offs = glob.glob(root+'*/' + mode +
        #
        point_clouds = []
        self.gt = np.zeros((self.n_pc,6))
        for index,depth_file in enumerate(depth_files):
            depth_file_full = os.path.join(root,'high_res_depth',depth_file)
            depth_map = np.asarray(o3d.read_image(depth_file_full))
            current_point_cloud = utils.convert_depth_map_to_pc(depth_map,self.focal_len,self.principal_pt,depth_scale=self.depth_scale)
            
            #print("pcsz",len(current_point_cloud),len(current_point_cloud[0]),len(current_point_cloud[0][0]))
            current_point_cloud = current_point_cloud[::subsample_rate,::subsample_rate,:]
            #print("pcsz",len(current_point_cloud),len(current_point_cloud[0]),len(current_point_cloud[0][0]))
            #print("pc",current_point_cloud)
            point_clouds.append(current_point_cloud)
            
            image_file = depth_file[:14] + '1.jpg'
            idx = image_files.index(image_file)
            current_image_struct = image_structs[idx]
            current_pos = current_image_struct[6]
            current_direction = current_image_struct[4]
            current_gt = np.concatenate((current_pos,current_direction)).T
            #print("agt",np.shape(current_gt))
            self.gt[index,:] = current_gt
            #print("gt",np.shape(self.gt))
        point_clouds = np.asarray(point_clouds)
        self.point_clouds = torch.from_numpy(point_clouds) # <NxHxWx3>
        self.valid_points = find_valid_points(self.point_clouds) 
        self.lat_indexes = range(latent_size)
        
    def __getitem__(self,index):
        pcd = self.point_clouds[index,:,:,:]  # <HxWx3>
        valid_points = self.valid_points[index,:]
        return pcd,valid_points,self.lat_indexes[index]

    def __len__(self):
        return self.n_pc
                              
class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        for line in open(triplets_file_name):
            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path1)]))
        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path2)]))
        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path3)]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
    
