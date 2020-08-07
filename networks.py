import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bmtm, bmtv, bmmt, bbmv


class IMUNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)
        k0 = ks[0]
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension
        d0 = ds[0]
        d1 = ds[1]
        d2 = ds[2]

        # padding
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)

        # nets
        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((int(p0), 0)), # padding at start
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2),
            torch.nn.BatchNorm1d(c3, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c3, int(out_dim), 1, dilation=1),
            torch.nn.ReplicationPad1d((0, 0)), # padding at end
        )

        # for normalizing inputs
        self.mean_u = torch.nn.Parameter(torch.zeros(in_dim),
            requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(in_dim),
            requires_grad=False)

    def forward(self, us):
        u = self.norm(us).transpose(1, 2)
        return self.cnn(u).transpose(1, 2)

    def norm(self, us):
        return (us-self.mean_u)/self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)


class BBBNet(torch.nn.Module):
    """Compute velocity measurement covariance from IMU input"""

    def __init__(self, zupt_forward_std, lat_std, up_std):
        super(BBBNet, self).__init__()
        min_std = torch.Tensor([0.01])
        self.min_std = torch.nn.Parameter(min_std, requires_grad=False)

        self.cov_lin = torch.nn.Linear(6, 3, bias=False).float()
        self.cov_bias = torch.nn.Parameter(torch.zeros(3).float())

        # init to  zero
        self.cov_lin.weight.data *= 0
        self.cov_bias.data[0] = zupt_forward_std
        self.cov_bias.data[1] = lat_std
        self.cov_bias.data[2] = up_std
        
        self.cov_lin.weight.data[2, 0] = 0
        self.cov_lin.weight.data[2, 1] = 0
        self.cov_lin.weight.data[2, 2] = 0
        self.cov_lin.weight.data[1, 2] =  0

    def forward(self, us):
        us = us[:, :, :6]
        covs = self.cov_lin(us)**2 + self.cov_bias**2 + self.min_std**2
        return covs
