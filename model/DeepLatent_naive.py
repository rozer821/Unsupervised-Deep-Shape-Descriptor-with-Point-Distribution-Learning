from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance
class DeepLatent(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, latent_length, n_samples, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepLatent, self).__init__()
        self.n_samples = n_sampless
        self.loc_net = PointReg(latent_length)
        self.cls_net = MLP(dim)
        self.chamfer_dist = ChamferDistance()

    def forward(self, obs, obs_gt, labels):
        self.obs = deepcopy(obs)
        self.obs_est = self.loc_net(self.obs)
        self.ob_gt = obs_gt
        self.batch_size = obs_local.shape[0] #B
        self.labels = labels
        if self.training:
            self.labels_est = self.cls_net(self.obs)
            loss = self.compute_loss()
            return loss

    def compute_loss(self):
        loss = 0.0
        for i in range(self.batch_size):

            dist1,dist2 = chamfer_dist(self.obs_est[i],self.obs_gt[i])
            loss_chamfer = dist1+dist2
            loss_bce = bce(self.labels[i],self.labels_est[i])
            loss += loss_chamfer+loss_bce
        return loss/self.batch_size    
    

def bce(pred, targets, weight=None):
    criterion = CrossEntropyLoss(weight=weight)
    loss = criterion(pred, targets)
    return loss
