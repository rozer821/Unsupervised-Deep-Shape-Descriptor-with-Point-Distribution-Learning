import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from loss.chamfer import ChamfersDistance
from data.dataloader import ModelNet_aligned,generate_random_pc
from network import *
from utils import *

parser = argparse.ArgumentParser(description='3D auto decoder for tracking')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--epochs', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--debug', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='load part of the dataset to run quickly')
parser.add_argument('-y','--length_latent', type=int, default=512, help='length_latent')
parser.add_argument('--weight_file', default='', help='weights to load')
#parser.add_argument('--num_grid_point', type=int, default=2048, help='num of grid points')
parser.add_argument('--name', type=str, default='default', help='name of experiment (continue training if existing)')
parser.add_argument('-s','--seed',type=str,help="seed string"
parser.add_argument('--log_interval')
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'

dataset = ModelNet_aligned('dataset/modelnet40_off_aligned',mode='train',subsample_rate=None)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)

latent_vecs = []
latent_size = opt.latent_size

for i in range(len(dataset)):
    vec = (torch.ones(latent_size).normal_(0, 0.8).to(device))
    vec.requires_grad = True
    latent_vecs.append(vec)
    
model = LatentDiscriptor(latent_size = latent_size).to(device)
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
                    

for epoch in range(opt.epochs):
    training_loss= 0.0
    model.train()
    for index,(shape_batch,latent_indices) in enumerate(loader):
        latent_inputs = torch.zeros(0).to(device)
        for i_lat in latent_indices.cpu().detach().numpy():
            latent = latent_vecs[i_lat] 
            latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(1)], 1)
        latent_inputs = latent_inputs.transpose(0,1)
        shape_batch = shape_batch.to(device)
        loss = model(latent_inputs,shape_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss_epoch = training_loss/len(loader)
   
    
    if (epoch+1) % opt.log_interval == 0:
        print('[{}/{}], training loss: {:.4f}'.format(
            epoch+1,opt.epochs,training_loss_epoch))

        obs_global_est_np = []
        pose_est_np = []
        with torch.no_grad():
            model.eval()
            for index,(obs_batch,valid_pt,index_latents) in enumerate(loader):
                latent_inputs = torch.zeros(0).cuda()
                for i_lat in index_latents.cpu().detach().numpy():
                    latent = latent_vecs[i_lat]
                    latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(1)], 1)
                latent_inputs = latent_inputs.transpose(0,1)
                obs_batch = obs_batch.to(device)
                valid_pt = valid_pt.to(device)
                model(obs_batch,valid_pt,latent_inputs)
                obs_global_est_np.append(model.obs_global_est.cpu().detach().numpy())
                pose_est_np.append(model.pose_est.cpu().detach().numpy())
            
            pose_est_np = np.concatenate(pose_est_np)
            if init_pose is not None:
                pose_est_np = utils.cat_pose_2D(init_pose_np,pose_est_np)

            save_name = os.path.join(checkpoint_dir,'model_best.pth')
            utils.save_checkpoint(save_name,model,optimizer)

            obs_global_est_np = np.concatenate(obs_global_est_np)
            kwargs = {'e':epoch+1}
            valid_pt_np = dataset.valid_points.cpu().detach().numpy()
     
            #save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
            #np.save(save_name,obs_global_est_np)

            save_name = os.path.join(checkpoint_dir,'pose_est.npy')
            np.save(save_name,pose_est_np)
            
    if (epoch+1) % (opt.log_interval*4) == 0:
       
    






