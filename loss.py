import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        loss = F.mse_loss(features, centers_batch)
        return loss