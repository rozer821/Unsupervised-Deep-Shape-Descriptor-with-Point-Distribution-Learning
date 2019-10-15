import torch
import torch.nn as nn

INF = 1000000

class ChamfersDistance(nn.Module):
    '''
    Extensively search to compute the Chamfersdistance. 
    '''
    def forward(self, input1, input2, valid1=None, valid2=None):

        # input1, input2: BxNxK, BxMxK, K = 3
        B, N, K = input1.shape
        _, M, _ = input2.shape
        if valid1 is not None:
            # ignore invalid points
            valid1 = valid1.type(torch.float32)
            valid2 = valid2.type(torch.float32)

            invalid1 = 1 - valid1.unsqueeze(-1).expand(-1, -1, K)
            invalid2 = 1 - valid2.unsqueeze(-1).expand(-1, -1, K)

            input1 = input1 + invalid1 * INF * torch.ones_like(input1)
            input2 = input2 + invalid2 * INF * torch.ones_like(input2)

        # Repeat (x,y,z) M times in a row
        input11 = input1.unsqueeze(2)           # BxNx1xK
        input11 = input11.expand(B, N, M, K)    # BxNxMxK
        # Repeat (x,y,z) N times in a column
        input22 = input2.unsqueeze(1)           # Bx1xMxK
        input22 = input22.expand(B, N, M, K)    # BxNxMxK
        # compute the distance matrix
        D = input11 - input22                   # BxNxMxK
        D = torch.norm(D, p=2, dim=3)         # BxNxM

        dist0, _ = torch.min(D, dim=1)        # BxM
        dist1, _ = torch.min(D, dim=2)        # BxN

        if valid1 is not None:
            dist0 = torch.sum(dist0 * valid2, 1) / torch.sum(valid2, 1)
            dist1 = torch.sum(dist1 * valid1, 1) / torch.sum(valid1, 1)
        else:
            dist0 = torch.mean(dist0, 1)
            dist1 = torch.mean(dist1, 1)

        loss = dist0 + dist1  
        loss = torch.mean(loss)                 
        return loss