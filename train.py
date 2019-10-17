import argparse
import os
import numpy as np
import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from data.dataloader import *
from model.DeepLatent_naive import *
from model.DeepLatent_couple import *
from model.networks import *
from utils import *

parser = argparse.ArgumentParser(description='3D auto decoder for tracking')
parser.add_argument('-r','--root', type=str, default='dataset/modelnet40_off_aligned', help='data_root')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--debug', default=True, type=lambda x: (str(x).lower() == 'true'),help='load part of the dataset to run quickly')
parser.add_argument('-y','--latent_size', type=int, default=512, help='length_latent')
parser.add_argument('--weight_file', default='', help='weights to load')
parser.add_argument('--name', type=str, default='default', help='name of experiment (continue training if existing)')
parser.add_argument('-s','--seed',type=str,default=8842,help="seed string")
parser.add_argument('--log_interval',type=str, default=20,help="log_interval")
parser.add_argument('--sample_num',type=int, default=1024,help="num_point")
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
root = opt.root
dataset = ModelNet_aligned(root,device,mode='train',downsample_num=opt.sample_num)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)

checkpoint_dir = 'result/'
latent_vecs = []
latent_size = opt.latent_size

for i in range(len(dataset)):
    vec = (torch.ones(latent_size).normal_(0, 0.8).to(device))
    vec.requires_grad = True
    latent_vecs.append(vec)
    
model = DeepLatent(latent_length = latent_size, n_samples = opt.sample_num).to(device)
optimizer = optim.Adam(
            [
                {
                     "params":model.parameters(), "lr":opt.lr,
                },
                {
                     "params": latent_vecs, "lr":opt.lr,  
                }
            ]
            )
                    
min_loss = 100000
for epoch in tqdm.tqdm(range(opt.epochs)):
    training_loss= 0.0
    model.train()
    for index,(shape_batch,shape_gt_batch,latent_indices) in enumerate(loader):
        latent_inputs = torch.zeros(0).to(device)
        for i_lat in latent_indices.cpu().detach().numpy():
            latent = latent_vecs[i_lat] 
            latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(1)], 1)
        latent_inputs = latent_inputs.transpose(0,1)
        shape_batch = shape_batch.to(device)
        loss = model(shape_batch,shape_gt_batch,latent_inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
                    
    training_loss_epoch = training_loss/len(loader)
    
    if (epoch+1) % opt.log_interval == 0:
        print("Epoch:[%d|%d], training loss:%f"%(epoch,opt.epochs,training_loss_epoch))
        
    if training_loss_epoch < min_loss:
        print('New best performance! saving')
        save_name = os.path.join(checkpoint_dir,'model_best')
        utils.save_checkpoint(save_name,model,optimizer)
                    
    if (epoch+1) % (opt.log_interval*10) == 0:
        min_loss = training_loss_epoch           
        save_name = os.path.join(checkpoint_dir,'model_routine')
        utils.save_checkpoint(save_name,model,z,optimizer)
        
        

    
       
    






