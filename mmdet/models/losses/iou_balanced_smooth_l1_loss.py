import torch.nn as nn
from mmdet.core import weighted_iou_balanced_smoothl1

from ..registry import LOSSES


@LOSSES.register_module
class IoUbalancedSmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, delta=1.0, loss_weight=1.0):
        super(IoUbalancedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.delta = delta
        self.loss_weight = loss_weight

    def forward(self, pred, target, iou, weight, *args, **kwargs):
        # print("the loss_weight is ", self.loss_weight)
        loss_bbox = self.loss_weight * weighted_iou_balanced_smoothl1(
            pred, target, iou, weight, beta=self.beta, delta=self.delta, *args, **kwargs)
        return loss_bbox
