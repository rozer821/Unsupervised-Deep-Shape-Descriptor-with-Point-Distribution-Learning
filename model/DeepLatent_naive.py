from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import sys
 
sys.path.append('../')

from loss.chamfer import ChamferDistance
from model.networks import ShapeFeature

class DeepLatent(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, latent_length, n_samples = 1024, chamfer_weight=0.12):
        super(DeepLatent, self).__init__()
        self.latent_length = latent_length
        
        self.shape_net = ShapeFeature(latent_length,n_samples)
        #dim=[3+self.latent_length, 64, 512, 512, 256, 128, 1]
        #self.cls_net = MLP(dim)
        self.chamfer_dist = ChamferDistance()
        self.L2_dist = nn.MSELoss()
        self.chamfer_weight = chamfer_weight
    def forward(self, obs, obs_gt, latent):
        
        self.obs = deepcopy(obs)
        self.obs_gt = deepcopy(obs_gt)
        print(obs.size(),self.obs_gt.size(),latent.size())
        obs_with_lat = torch.cat([self.obs,latent], 1)
        #batch_size = self.obs.size()[0]
        #sigma_num = self.obs.size()[1]
        #self.obs = self.obs.view(num_instance,self.obs.size()[2],-1)
        #self.obs_gt = self.obs_gt.view(num_instance,self.obs_gt.size()[2],-1)
        #latent_repeat = latent_repeat.view(num_instance,latent_repeat.size()[2],-1)
        #print(self.obs.size(),self.obs_gt.size(),latent.size(),latent_repeat.size())
        #print(obs_with_lat.size())
        
        self.obs_est = self.shape_net(obs_with_lat, self.obs, latent)
        loss = self.compute_loss()
        return loss

    def compute_loss(self):
        loss_chamfer = self.chamfer_dist(self.obs_gt,self.obs_est)
        loss_L2 = self.L2_dist(self.obs_gt,self.obs_est)
        loss = self.chamfer_weight * loss_chamfer + (1 - self.chamfer_weight)*loss_L2
        return loss,loss_chamfer,loss_L2    
    

def bce(pred, targets, weight=None):
    criterion = nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, targets)
    return loss
