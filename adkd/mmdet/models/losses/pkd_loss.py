# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weighted_loss


def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variances.

    Args:
        feat (torch.Tensor): The original feature map with shape
            (N, C, H, W).
    """
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


@weighted_loss
def pkd_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    return F.mse_loss(pred, target, reduction='none') / 2






@MODELS.register_module()
class PKDLoss(nn.Module):

    def __init__(self, reduction='mean',
                 loss_weight=1.0,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65
                 ):
        super(PKDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1))

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # 进行修改 mgd
        loss = self.get_dis_loss(pred, target) * self.alpha_mgd


        # loss = self.loss_weight * pkd_loss(
        #    pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss
