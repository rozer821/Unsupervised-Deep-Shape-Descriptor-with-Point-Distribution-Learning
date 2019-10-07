import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from chamfer_distance import ChamferDistance
from data import ModelNet, Scape, ScapeInterpolation, FAUST
from network import *
from utils import *

parser = argparse.ArgumentParser(description='3D auto decoder for tracking')
parser.add_argument('--batch', type=int, default=3, help='training batch size')
parser.add_argument('--epochs', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--debug', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='load part of the dataset to run quickly')
parser.add_argument('--size_z', type=int, default=512, help='size_z')
parser.add_argument('--weight_file', default='', help='weights to load')
parser.add_argument('--num_grid_point', type=int, default=2048, help='num of grid points')
parser.add_argument('--ex', type=str, default='default', help='name of experiment (continue training if existing)')
opt = parser.parse_args()
print(opt)