import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.dataloader import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold.t_sne import TSNE
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

#parser.add_argument('-r','--root', type=str, default=, help='data_root')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ModelNet_aligned('/home/mmvc/mmvc-ny-nas/Yi_Shi/data/modelnet40_off_aligned',device,mode='train',downsample_num=2048,sigmas=[0.02])
categs = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower', 'glass', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night', 'person', 'piano', 'plant', 'radio', 'range', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv', 'vase', 'wardrobe', 'xbox']
print(categs)

name_instance = [os.path.split(i)[-1] for i in dataset.npys_gen]
categs_colors = mpl.cm.rainbow(np.linspace(0, 1, len(dataset.categ)))      
color_instance = [categs_colors[categs.index(i.split('_')[0])] for i in name_instance]
#print(color_instance)
                                      
def gen_tsne_plot_2D(latent_space, cs, path):
    # from MulticoreTSNE import MulticoreTSNE as TSNE
    tsne = TSNE(2)
    reduced = tsne.fit_transform(latent_space)
    np.save("2d.npy",reduced)
    fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    #ax = Axes2D(fig)
    plt.scatter(reduced[:, 0], reduced[:, 1],c=cs)
    #plt.view_init(30, 45)
    plt.savefig(os.path.join(path, '_TSNE.png'), bbox_inches='tight')
    plt.close()

def gen_tsne_plot_3D(latent_space, cs, path):
    tsne = TSNE(3)
    reduced = tsne.fit_transform(latent_space)
    np.save("3d.npy",reduced)
    fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(reduced[:, 0], reduced[:, 1],reduced[: , 2], c=cs)
    ax.view_init(30, 45)
    plt.savefig(os.path.join(path, '_TSNE_3D.png'), bbox_inches='tight')
    plt.close()
    
categ = glob
latents = torch.load('./results/model_best_latents.pt')
print(latents.size)
print(len(color_instance))
print(color_instance)
latents = latents.cpu().detach().numpy()
gen_tsne_plot_2D(latents,color_instance,'./results')
print("2d saved")
gen_tsne_plot_3D(latents,color_instance,'./results')
print("3d saved")
