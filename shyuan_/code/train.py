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

# no randomness
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

start_epoch = 0
size_z = opt.size_z
folder_out = opt.ex
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.makedirs(folder_out + '/ply/')

train_data = data(is_train=True, is_debug=opt.debug, is_normal_noise=False)
train_loader = DataLoader(train_data, batch_size=opt.batch, shuffle=False, drop_last=True)

test_data = data(is_train=False, is_debug=opt.debug, is_normal_noise=False)
test_loader = DataLoader(test_data, batch_size=opt.batch, shuffle=False, drop_last=True)

z_all = torch.zeros(10000, size_z)
z_all = z_all.type(torch.float32)
z_all = z_all.cuda()

z_all_val = torch.zeros(10000, size_z)
z_all_val = z_all_val.type(torch.float32)
z_all_val = z_all_val.cuda()

nn.init.normal_(z_all, mean=0, std=0.1)
nn.init.normal_(z_all_val, mean=0, std=0.1)

colors = []
intensities = []
for i in range(opt.num_grid_point):
    ind = int(255.0 / opt.num_grid_point * i)
    colors.append(cm.jet(ind))
    intensities.append(i)
colors = np.array(colors, dtype=np.float32)
colors *= 255.0
intensities = np.array(intensities, dtype=np.float32)
intensities /= (opt.num_grid_point + 0.0)

hand_feature = HandFeature(size_z=opt.size_z, num_point=opt.num_grid_point).cuda()
hand_joints = HandJoints(size_z=opt.size_z, num_point=opt.num_grid_point).cuda()

optim_f = optim.Adam(model.parameters(), lr=opt.lr)
optim_j = optim.Adam(hand_joints.parameters(), lr=opt.lr)
optim_z = optim.Adam([z_all.requires_grad_()], lr=opt.lr / 10.0)
optim_z_val = optim.Adam([z_all_val.requires_grad_()], lr=opt.lr / 10.0)

loss_train = AverageMeter()
loss_eval = AverageMeter()
loss_aver = None
error_train = AverageMeter()
error_eval = AverageMeter()
error_aver = None
his_train = AverageMeter()
his_eval = AverageMeter()
his_aver = None
global_counter = 0

def process_one_epoch(is_train=True):
    global global_counter, loss_aver
    if is_train:
        loader = train_loader
    else:
        loader = test_loader
    for i, batch in enumerate(loader):
        adjust_learning_rate(optim_f, global_counter, opt.batch, opt.lr, is_debug=opt.debug)
        if i == 0:
            print('lr=', lr_now)
        adjust_learning_rate(optim_z, global_counter, opt.batch, opt.lr / 10.0, is_debug=opt.debug)
        adjust_learning_rate(optim_z_val, global_counter, opt.batch, opt.lr / 10.0, is_debug=opt.debug)

        data = batch[0].cuda()
        idxs = batch[1].cuda()

        cube = get_cube().cude()

        hand = data[0]

        joints = data[1]
        joints = joints.transpose(0, 1)
        joints = joints.unsqueeze(0)

        idxs = idxs[1:2]
        if is_train:
            global_counter += 1
            md1 = hand_feature.train()
            md2 = hand_joints.train()
            z = z_all[idxs, :]
        else:
            md1 = hand_feature.eval()
            md2 = hand_joints.eval()
            z = z_all_val[idxs, :]

        nm = torch.norm(z, dim=1)
        nm = torch.max(nm, torch.ones_like(nm))
        nm = nm.unsqueeze(1)
        nm = nm.repeat(1, size_z)
        z = z / nm
        z = z.unsqueeze(2)
        z = z.repeat(1, 1, opt.num_grid_point)

        points_pred_comp = md1(cube, z)
        points_pred_join = md2(joints, z)
        points_pred_comp = points_pred_comp.transpose(1, 2)
        points_pred_join = points_pred_join.transpose(1, 2)

        chamfer_dist = ChamferDistance()

        hand_pred = cube.transpose(1, 2) + points_pred_comp
        joint_pred = joints.transpose(1, 2) + points_pred_join

        dist1, dist2 = chamfer_dist(hand, hand_pred)
        dist3, dist4 = chamfer_dist(joints, joint_pred)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))+(torch.mean(dist3)) + (torch.mean(dist4))

        if is_train:
            loss += torch.sqrt(torch.sum(torch.pow(joint_pred - joints, 2), -1)).mean()

        loss = loss * 100.0

        if is_train:
            optim_f.zero_grad()
            optim_j.zero_grad()
            optim_z.zero_grad()
            loss.backward(retain_graph=True)
            optim_f.step()
            optim_j.step()
            optim_z.step()
        else:
            optim_f.zero_grad()
            optim_j.zero_grad()
            optim_z_val.zero_grad()
            loss.backward(retain_graph=True)
            optim_z_val.step()

        if is_train:
            loss_train.update(loss.item())
            loss_aver = loss_train
            error_train.update(error_mean)
            error_aver = error_train
            his_train.update(probs_sum)
            his_aver = his_train
        else:
            loss_eval.update(loss.item())
            loss_aver = loss_eval
            error_eval.update(error_mean)
            error_aver = error_eval
            his_eval.update(probs_sum)
            his_aver = his_eval

        str_is_train = 'train' if is_train else 'eval'
        print(str_is_train, loss_aver.avg, loss_aver.val, error_aver.avg)

        if epoch % 20 == 0 and i < 32:
            print('Writing')
            pred = points_pred_next_abs.detach().cpu().numpy()
            a = 0
            write_to_ply_color(
                folder_out + "/ply/" + str_is_train + "_epoch_" + str(epoch) + "_batch_" + str(i) + "_" + str(
                    a) + "_pred.ply", pred[a], colors)
            write_to_pcd_with_intensity(
                folder_out + "/pcd/" + str_is_train + "_epoch_" + str(epoch) + "_batch_" + str(i) + "_" + str(
                    a) + "_pred.pcd", pred[a], intensities)

            prev = points_pred_cur_abs.detach().cpu().numpy()
            a = 0
            write_to_ply_color(
                folder_out + "/ply/" + str_is_train + "_epoch_" + str(epoch) + "_batch_" + str(i) + "_" + str(
                    a) + "_prev.ply", prev[a], colors)
            write_to_pcd_with_intensity(
                folder_out + "/pcd/" + str_is_train + "_epoch_" + str(epoch) + "_batch_" + str(i) + "_" + str(
                    a) + "_prev.pcd", prev[a], intensities)


for epoch in range(start_epoch, opt.epochs):
    print('epoch', epoch)
    process_one_epoch(True)
    process_one_epoch(False)

