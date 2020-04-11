import torch.nn as nn
from mmdet.core import (iou_balanced_cross_entropy, iou_balanced_binary_cross_entropy,
                        mask_cross_entropy)

from ..registry import LOSSES


@LOSSES.register_module
class IOUbalancedCrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, loss_weight=1.0, eta=1.5):
        super(IOUbalancedCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        self.eta = eta

        if self.use_sigmoid:
            self.cls_criterion = iou_balanced_binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = iou_balanced_cross_entropy

    def forward(self, cls_score, label, label_weight, iou, *args, **kwargs):
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, label_weight, iou, eta =self.eta,  *args, **kwargs)
        return loss_cls
