import torch.nn as nn
from mmdet.core import iou_balanced_sigmoid_focal_loss

from ..registry import LOSSES


@LOSSES.register_module
class IOUbalancedSigmoidFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 loss_weight=1.0,
                 gamma=2.0,
                 alpha=0.25,
                 eta=1.0
                 # use_diff_thr=False
                 ):
        super(IOUbalancedSigmoidFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focaloss supported now.'
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.eta = eta
        self.cls_criterion = iou_balanced_sigmoid_focal_loss
        # self.use_diff_thr = use_diff_thr

    def forward(self, cls_score, label, label_weight, iou, *args, **kwargs):
        if self.use_sigmoid:
            # loss_cls = self.loss_weight * self.cls_criterion(
            #     cls_score,
            #     label,
            #     label_weight,
            #     iou,
            #     # iou_neg,
            #     gamma=self.gamma,
            #     alpha=self.alpha,
            #     eta=self.eta,
            #     # use_diff_thr=self.use_diff_thr,
            #     *args,
            #     **kwargs)
            loss_cls = self.cls_criterion(
                cls_score,
                label,
                label_weight,
                iou,
                # iou_neg,
                gamma=self.gamma,
                alpha=self.alpha,
                eta=self.eta,
                # use_diff_thr=self.use_diff_thr,
                *args,
                **kwargs)

        else:
            raise NotImplementedError
        return loss_cls
        # return
