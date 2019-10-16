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
    def __init__(self, latent_length, n_samples = 1024, chamfer_weight=0.2):
        super(DeepLatent, self).__init__()
        self.latent_length = latent_length
        
        self.shape_net = ShapeFeature(latent_length,n_samples)
        #dim=[3+self.latent_length, 64, 512, 512, 256, 128, 1]
        #self.cls_net = MLP(dim)
        self.chamfer_dist = ChamferDistance()
        self.L2_dist = nn.MSELoss()
        self.chamfer_weight = chamfer_weight
    def forward(self, obs, obs_gt, latent):
        self.obs = deepcopy(obs).transpose(1,2)
        self.obs_gt = deepcopy(obs_gt).transpose(1,2)
        
        #print(latent.size())
        latent_repeat = latent.unsqueeze(1).repeat(1,obs.size()[1],1).transpose(1,2)
 
        #print(self.obs.size(),latent_repeat.size())
        obs_with_lat = torch.cat([self.obs,latent_repeat], 1)
        
        #print(obs_with_lat.size())
        self.obs_est = self.shape_net(obs_with_lat, self.obs, latent_repeat)
        self.batch_size = obs.size()[0] #
        loss = self.compute_loss()
        return loss

    def compute_loss(self):
        loss_chamfer = self.chamfer_dist(self.obs_gt,self.obs_est)
        loss_L2 = self.L2_dist(self.obs_gt,self.obs_est)
        print('l2:',loss_L2,'chm:',loss_chamfer)
        loss = self.chamfer_weight * loss_chamfer + (1 - self.chamfer_weight)*loss_L2
        return loss    
    

def bce(pred, targets, weight=None):
    criterion = nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, targets)
    return loss
