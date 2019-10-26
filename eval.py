import argparse
import os
import numpy as np
import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim

from data.dataloader import *
from model.DeepLatent_naive import *
from model.DeepLatent_couple import *
from model.networks import *

from utils.utils import *

parser = argparse.ArgumentParser(description='3D auto decoder for tracking')
parser.add_argument('-r','--root', type=str, default='/home/mmvc/mmvc-ny-nas/Yi_Shi/data/modelnet40_off_aligned', help='data_root')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--debug', default=True, type=lambda x: (str(x).lower() == 'true'),help='load part of the dataset to run quickly')
parser.add_argument('-y','--latent_size', type=int, default=64, help='length_latent')
parser.add_argument('--weight_file', default='', help='path to weights to load')
parser.add_argument('--name', type=str, default='default', help='name of experiment (continue training if existing)')
parser.add_argument('-s','--seed',type=str,default=42,help="seed string")
parser.add_argument('--log_interval',type=str, default=1,help="log_interval")
parser.add_argument('--sample_num',type=int, default=2048,help="num_point")
parser.add_argument('--resume',type=bool, default=True,help="if load model")
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
root = opt.root

num_sigmas = 1
dataset = ModelNet_aligned(root,device,mode='train',downsample_num=opt.sample_num)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)

checkpoint_dir = 'results/'
latent_vecs = []
latent_size = opt.latent_size
model = DeepLatent(latent_length = latent_size, n_samples = opt.sample_num)

model, latent_vecs, _ = load_checkpoint(os.path.join(checkpoint_dir,'model_best'),model, None) 
model.to(device)
def draw(pt,pt_gt,index):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pt[0,:], pt[1,:], pt[2,:]) 
    fig.savefig(os.path.join(checkpoint_dir,'pics', 'shape_%d.png'%index), bbox_inches='tight')

    
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.scatter(pt_gt[0,:], pt_gt[1,:],pt_gt[2,:])
    fig2.savefig(os.path.join(checkpoint_dir,'pics', 'shape_%d_gt.png'%index), bbox_inches='tight')
 
    
total_loss = 0.0
total_chamfer = 0.0
total_l2 = 0.0

model.eval()
for index,(shape_batch,shape_gt_batch,latent_indices) in enumerate(loader):
    print(index)
    latent_inputs = torch.zeros(0).to(device)
    for i_lat in latent_indices.cpu().detach().numpy():
        latent = latent_vecs[i_lat] 
        latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(0)], 0)
    latent_repeat = latent_inputs.unsqueeze(-1).repeat(1,1,shape_batch.size()[-1])
    shape_batch = shape_batch.to(device)
    shape_gt_batch = shape_gt_batch.to(device)
    #print(shape_batch.size(),shape_gt_batch.size(),latent_repeat.size())
    loss,chamfer,l2 = model(shape_batch,shape_gt_batch,latent_repeat)
    if random.random()<0.5:
        shape = model.obs_est[0,:,:].cpu().detach().numpy()
        shape_gt = model.obs_gt[0,:,:].cpu().detach().numpy()
        draw(shape,shape_gt,index)
        
    print(l2.item(),chamfer.item())
    total_loss += loss.item()
    total_chamfer += chamfer.item()
    total_l2 += l2.item()
    
total_l2 = total_l2*1.0/len(dataset) 
total_chamfer = total_chamfer*1.0/len(dataset)
total_loss = total_loss*1.0/len(dataset)

print('l2:',total_l2,'chamfer:',total_chamfer,'total:',total_loss)
    

