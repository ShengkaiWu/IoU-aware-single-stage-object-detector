# TODO merge naive and weighted loss.
import numpy as np
import torch
import torch.nn.functional as F

def consistent_loss(pred, label, weight, iou, avg_factor=None, reduce=True):
    """
    :param pred: tesnor of shape (batch*num_samples, num_class)
    :param label: tensor of shape (batch*num_samples), store gt labels such as
               0, 1, 2, 80 for corresponding class(0 represent background).
    :param weight: tensor of shape (batch*num_samples), 1 for all the elements;
    :param iou: tensor of shape (batch*num_samples), iou between proposals and corresponding ground
        truth boxes for positives and 0 for negatives.
    :param avg_factor:
    :param reduce:
    :return:
    """
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw1 = F.cross_entropy(pred, label, reduction='none')
    target = iou.new_zeros(iou.size(0))
    pos_inds = (label > 0).nonzero().view(-1)
    target[pos_inds] = 1.0
    threshold = 0.5
    ones_weight = iou.new_ones(iou.size(0))
    iou_weights_1 = torch.where(iou > threshold, 1.0 + (iou - threshold), ones_weight)
    iou_weights = (1 - target) + target * iou_weights_1
    iou_weights = iou_weights.detach()
    raw = raw1 * iou_weights
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor