import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapeFeature(nn.Module):
    # [ N*(3+z) ] -> # [ N * 3 ] 
    def __init__(self, size_z, num_point):
        super(ShapeFeature, self).__init__()
        size_kernel = 1
        size_pad = 0

        self.size_z = size_z
        self.num_point = num_point
        self.conv1 = torch.nn.Conv1d(3 + self.size_z, 256, size_kernel, padding=size_pad)
        self.conv2 = torch.nn.Conv1d(256, 128, size_kernel, padding=size_pad)
        self.conv3 = torch.nn.Conv1d(128, 3, size_kernel, padding=size_pad)

        self.conv4 = torch.nn.Conv1d(3 + self.size_z, 256, size_kernel, padding=size_pad)
        self.conv5 = torch.nn.Conv1d(256, 128, size_kernel, padding=size_pad)
        self.conv6 = torch.nn.Conv1d(128, 3, size_kernel, padding=size_pad)
        
        self.ln0 = nn.LayerNorm((self.size_z, num_point))
        self.ln1 = nn.LayerNorm((256, num_point))
        self.ln2 = nn.LayerNorm((128, num_point))
        self.ln3 = nn.LayerNorm((3, num_point))
        self.ln4 = nn.LayerNorm((256, num_point))
        self.ln5 = nn.LayerNorm((128, num_point))
        self.ln6 = nn.LayerNorm((3, num_point))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, z):
        z = self.ln0(z)
        x = torch.cat([z, x], 1)
        x = self.dropout(F.relu(self.ln1(self.conv1(x))))
        x = self.dropout(F.relu(self.ln2(self.conv2(x))))
        x = self.dropout(F.relu(self.ln3(self.conv3(x))))
        x = torch.cat([z, x], 1)
        x = self.dropout(F.relu(self.ln4(self.conv4(x))))
        x = self.dropout(F.relu(self.ln5(self.conv5(x))))
        x1 = self.dropout((self.ln6(self.conv6(x))))
        return x1
    
    
    
def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers

'''
class PointwiseMLP(nn.Sequential):
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)
'''


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.mlp = get_MLP_layers(dims, doLastRelu=False)

    def forward(self, x):
        return self.mlp.forward(x)
