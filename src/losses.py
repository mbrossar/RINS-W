import torch
import numpy as np
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3


class VLoss(torch.nn.Module):

    def __init__(self, w=1, pos_weight=1, n0=0):
        super().__init__()
        # weights on different loss
        # pos_weight is weight of positive examples
        pos_weight = torch.Tensor([pos_weight]).cuda()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # where loss start in a training sequence ?
        self.n0 = n0
        # weight on speed loss
        self.w = w

    def forward(self, xs, hat_xs):
        zupts = xs[:, self.n0:].squeeze()
        hat_zupts = hat_xs[:, self.n0:].squeeze()
        return self.w*self.bce(hat_zupts, zupts)
