# TODO merge naive and weighted loss.
import numpy as np
import torch
import torch.nn.functional as F

from ..bbox import bbox_overlaps
from ...ops import sigmoid_focal_loss
from ..bbox.transforms import delta2bbox



def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor

# added by Shengkai Wu
# implement iou_balanced cross entropy loss.
def iou_balanced_cross_entropy(pred, label, weight, iou, eta = 1.5, avg_factor=None, reduce=True):
    """
    iou_balanced cross entropy loss to make the training process to focus more on positives with higher
    iou.
    :param pred: tesnor of shape (batch*num_samples, num_class)
    :param label: tensor of shape (batch*num_samples), store gt labels such as
               0, 1, 2, 80 for corresponding class(0 represent background).
    :param weight: tensor of shape (batch*num_samples), 1 for all the elements;
    :param iou: tensor of shape (batch*num_samples), iou between predicted boxes and corresponding ground
        truth boxes for positives and 0 for negatives.
    :param eta: control to which extent the training process focuses on the positives with high iou.
    :param avg_factor:
    :param reduce:
    :return:
    """
    # avg_factor = batch*num_samples
    # if avg_factor is None:
    #     avg_factor = max(torch.sum(weight > 0).float().item(), 1.)

    raw1 = F.cross_entropy(pred, label, reduction='none')

    target = iou.new_zeros(iou.size(0))
    # target_1 = iou.new_zeros(iou.size(0))
    # the way to get the indexes of positive example may be wrong; is it right?
    # pos_inds_1 = label > 0
    # target_1[pos_inds_1] = 1
    # modify the way to get the indexes
    pos_inds = (label > 0).nonzero().view(-1)
    # pos_inds = (label >= 1).nonzero().view(-1)
    target[pos_inds] = 1.0
    # pos_inds_test = target.nonzero().view(-1)

    method_1 = True
    normalization = True

    method_2 = False

    threshold = 0.66
    # threshold = torch.min(iou[pos_inds]).item()

    method_3 = False

    target = target.type_as(pred)
    if method_1:
        if normalization:
            iou_weights = (1 - target) + (target * iou).pow(eta)
            # normalized to keep the sum of loss for positive examples unchanged;
            raw2 = raw1*iou_weights
            normalizer = (raw1 * target).sum() / ((raw2 * target).sum() + 1e-6)
            normalized_iou_weights = (1 - target) + (target * iou).pow(eta) * normalizer
            normalized_iou_weights = normalized_iou_weights.detach()
            raw = raw1*normalized_iou_weights
        else:
            weight_pos = 1.8
            iou_weights = (1 - target) + (target * iou).pow(eta)*weight_pos
            iou_weights = iou_weights.detach()
            raw = raw1*iou_weights
    elif method_2:
        iou_weights = (1 - target) + (target*(1 + (iou - threshold))).pow(eta)
        iou_weights = iou_weights.detach()
        raw = raw1 * iou_weights
    elif method_3:
        ones_weight = iou.new_ones(iou.size(0))
        iou_weights_1 = torch.where(iou > threshold, 1.0 + (iou - threshold), ones_weight)
        # iou_weights = (1 - target) + (target * iou_weights_1).pow(eta)
        iou_weights = (1 - target) + target * iou_weights_1
        iou_weights = iou_weights.detach()
        raw = raw1 * iou_weights
        # raw = (raw1 * iou_weights +raw1)/2
        # print('test_loss')

    if avg_factor is None:
        # avg_factor = max(torch.sum(iou_weights).float().item(), 1.)
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)

    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor

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



def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    # print('test')
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor

def iou_balanced_binary_cross_entropy(pred, label, weight, iou, eta = 1.5, avg_factor=None, reduce=True):
    """

    :param pred: tensor of shape (num_examples, 1)
    :param label: tensor of shape (num_examples, 1)
    :param weight: tensor of shape (num_examples, 1)
    :param iou: tensor of shape (num_examples), containing the iou for all the regressed
          positive examples.
    :param eta:
    :param avg_factor:
    :return:
    """
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)

    raw1 = F.binary_cross_entropy_with_logits(pred, label.float(),reduction='none')

    target = label.new_zeros(label.size())
    # target_1 = iou.new_zeros(iou.size(0))
    # the way to get the indexes of positive example may be wrong; is it wright?
    # pos_inds_1 = label > 0
    # target_1[pos_inds_1] = 1
    # modify the way to get the indexes
    # label_squeeze = torch.squeeze(label)
    # pos_inds = (label > 0).nonzero().view(-1)
    # print('the size of label is ', label.size())
    pos_inds = (label > 0).nonzero()
    # print('the size of label_squeeze is ', label_squeeze.size())
    target[pos_inds] = 1

    # print('the num of positive examples is', torch.sum(target))
    # print('the num of positive examples for target_1 is', torch.sum(target_1))
    normalization = True
    if normalization:
        target = target.type_as(pred)
        iou = iou.unsqueeze(-1)
        # print('the size of target is ', target.size())
        # print('the size of iou is ', iou.size())
        # print('the size of iou_1 is ', iou_1.size())
        iou_weights = (1 - target) + (target * iou).pow(eta)
        # print('the size of iou_weights is ', iou_weights.size())
        # print('the size of raw1 is ', raw1.size())
        # iou_weights.unsqueeze(1)
        # normalized to keep the sum of loss for positive examples unchanged;
        raw2 = raw1 * iou_weights
        normalizer = (raw1 * target).sum() / ((raw2 * target).sum() + 1e-6)
        normalized_iou_weights = (1 - target) + (target * iou).pow(eta) * normalizer
        normalized_iou_weights = normalized_iou_weights.detach()
        raw = raw1 * normalized_iou_weights
    else:
        target = target.type_as(pred)
        weight_pos = 1.8
        iou_weights = (1 - target) + (target * iou).pow(eta) * weight_pos
        iou_weights = iou_weights.detach()
        raw = raw1 * iou_weights

    if reduce:

        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor

    # return F.binary_cross_entropy_with_logits(
    #     pred, label.float(), weight.float(),
    #     reduction='sum')[None] / avg_factor






# Known from the definition of weight in file anchor_target.py,
# all the elements of tensor 'weight' are 1.
def py_sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * weight
    # the value of reduction_enum is decided by arg 'reduction'
    # none: 0, mean:1, sum: 2
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


# added by Shengkai Wu
# The focal loss is only computed for negative examples, and the standard binary cross
# entropy loss is computed for the positive examples. This is designed to investigate
# whether hard example mining for positive examples is beneficial for the performance.
def py_sigmoid_focal_loss_for_negatives(pred,
                                       target,
                                       weight,
                                       gamma=2.0,
                                       alpha=0.25,
                                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = target + pred_sigmoid * (1 - target)
    weight = (alpha*target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
    # the value of reduction_enum is decided by arg 'reduction'
    # none: 0, mean:1, sum: 2
    # print("only compute the focal loss for negative examples")
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()



def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    """
    note that
    :param pred: tensor of shape (batch*A*width*height, num_class)
    :param target: tensor of shape (batch*A*width*height, num_class), only the element for the
        positive labels are 1.
    :param weight: tensor of shape (batch*A*width*height, num_class), 1 for pos and neg, 0 for the others
    :param gamma:
    :param alpha:
    :param avg_factor:
    :param num_classes:
    :return:
    """
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6

    return py_sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor

# added by Shengkai Wu
# iou-balanced classification loss is designed to strengthen the correlation between classificaiton and
# localization task. The goal is to make that the detections with high IOU with the ground truth boxes also have
# high classification scores.
def iou_balanced_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                iou,
                                gamma=2.0,
                                alpha=0.25,
                                eta=1.5,
                                avg_factor=None,
                                num_classes=80):
    """

    :param pred: tensor of shape (batch*A*width*height, num_class)
    :param target: tensor of shape (batch*A*width*height, num_class), only the positive label is
          assigned 1, 0 for others.
    :param weight: tensor of shape (batch*A*width*height, num_class), 1 for pos and neg, 0 for the others.
    :param iou: tensor of shape (batch*A*width*height), store the iou between predicted boxes and its
          corresponding ground truth boxes for the positives and the iou between the predicted boxes and
          anchors for negatives.
    :param gamma:
    :param alpha:
    :param eta: control the suppression for the positives of low iou.
    :param avg_factor: num_positive_samples. If None,
    :param num_classes:
    :return:
    """
    # if avg_factor is None:
    #     avg_factor = torch.sum(target).float().item() + 1e-6
    #     use_diff_thr = True
    # pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    loss1 = py_sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='none')

    IoU_balanced_Cls = True
    threshold = 0.5
    if IoU_balanced_Cls:
        # compute the normalized weights so that the loss produced by the positive examples
        # doesn't change.
        iou_expanded = iou.view(-1, 1).expand(-1, target.size()[1])
        iou_weights = (1 - target) + (target * iou_expanded).pow(eta)
        # iou_weights = iou_weights.detach()
        loss2 = loss1*iou_weights
        normalizer = (loss1*target).sum()/((loss2*target).sum()+1e-6)
        # normalizer = 2.1
        normalized_iou_weights = (1-target) + (target*iou_expanded).pow(eta)*normalizer
        normalized_iou_weights = normalized_iou_weights.detach()

        loss = loss1*normalized_iou_weights
        # print('test')
    else:
        # consistent loss
        iou_expanded = iou.view(-1, 1).expand(-1, target.size()[1])
        ones_weight = iou_expanded.new_ones(iou_expanded.size())
        # print('ones_weight.size() is ', ones_weight.size())
        iou_weights_1 = torch.where(iou_expanded > threshold, 1.0 + (iou_expanded - threshold), ones_weight)
        # iou_weights = (1 - target) + (target * iou_weights_1).pow(eta)
        iou_weights = (1 - target) + target * iou_weights_1
        iou_weights = iou_weights.detach()
        # loss = loss1 * iou_weights
        balance_factor = 0.6
        loss = loss1*balance_factor + loss1 * iou_weights*(1-balance_factor)


    return torch.sum(loss)[None] / avg_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    # the value of reduction_enum is decided by arg 'reduction'
    # none: 0, mean:1, sum: 2
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()

# Known from the definition of weight in file anchor_target.py,
# the elements of tensor 'weight' for positive proposals are one.
def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    # print('the size of pred is ', pred.size())
    # print('the size of target is ', target.size())
    # print('the size of weight is', weight.size())

    return torch.sum(loss * weight)[None] / avg_factor


# added by Shengkai Wu
# implement the focal loss for localization task.
def weighted_iou_balanced_smoothl1(pred, target, iou, weight, beta=1.0, delta=1.5, avg_factor=None):
    """

    :param pred: tensor of shape (batch*A*width*height, 4) or (batch*num_pos, 4)
    :param target: tensor of shape (batch*A*width*height, 4), store the parametrized coordinates of target boxes
          for the positive anchors and other values are set to be 0. Or tensor of shape (batch*num_pos, 4)
    :param iou: tensor of shape (batch*A*width*height)Or tensor of shape (batch*num_pos), store the iou between
          predicted boxes and its corresponding groundtruth boxes for the positives and the iou between the predicted
          boxes and anchors for negatives.
    :param weight: tensor of shape (batch*A*width*height, 4), only the weights for positive anchors are set to
          be 1 and other values are set to be 0. Or tensor of shape (batch*num_pos, 4), all the elements are 1.
    :param beta:
    :param delta: control the suppression for the outliers.
    :param avg_factor:
    :return:
    """
    # the pred and target are transformed to image domain and represented by top-left and bottom-right corners.
    assert pred.size() == target.size() and target.numel() > 0
    # ignore the positive examples of which the iou after regression is smaller
    # than 0.5;
    ignore_outliers = False
    iou_threshold = 0.5
    if ignore_outliers:
        filter = iou.new_zeros(iou.size())
        filter_extend = filter.view(-1, 1).expand(-1, 4)
        ind = (iou >= iou_threshold).nonzero()
        filter[ind] = 1
        iou = iou * filter

    iou_expanded = iou.view(-1, 1).expand(-1, 4)

    iou_weight = weight * iou_expanded.pow(delta)
    iou_weight = iou_weight.detach()


    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6

    loss1 = smooth_l1_loss(pred, target, beta, reduction='none')
    loss2 = loss1*iou_weight
    # loss2 = loss1 *filter_extend

    return torch.sum(loss2)[None] / avg_factor

def weighted_iou_regression_loss(iou_pred, iou_target, weight, avg_factor=None):
    """

    :param iou_pred: tensor of shape (batch*A*width*height) or (batch*num_pos)
    :param iou_target: tensor of shape (batch*A*width*height)Or tensor of shape (batch*num_pos), store the iou between
          predicted boxes and its corresponding groundtruth boxes for the positives and the iou between the predicted
          boxes and anchors for negatives.
    :param weight: tensor of shape (batch*A*width*height) or (batch*num_pos), 1 for positives and 0 for negatives and neutrals.
    :param avg_factor:
    :return:
    """
    # iou_pred_sigmoid = iou_pred.sigmoid()
    # iou_target = iou_target.detach()

    # L2 loss.
    # loss = torch.pow((iou_pred_sigmoid - iou_target), 2)*weight

    # Binary cross-entropy loss for the positive examples
    loss = F.binary_cross_entropy_with_logits(iou_pred, iou_target, reduction='none')* weight

    return torch.sum(loss)[None] / avg_factor

def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='none'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()

    return loss


def weighted_balanced_l1_loss(pred,
                              target,
                              weight,
                              beta=1.0,
                              alpha=0.5,
                              gamma=1.5,
                              avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = balanced_l1_loss(pred, target, beta, alpha, gamma, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3, reduction='mean'):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
        reduction (str): Reduction type.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_iou_loss(pred,
                      target,
                      weight,
                      style='naive',
                      beta=0.2,
                      eps=1e-3,
                      avg_factor=None):
    if style not in ['bounded', 'naive']:
        raise ValueError('Only support bounded iou loss and naive iou loss.')
    inds = torch.nonzero(weight[:, 0] > 0)
    if avg_factor is None:
        avg_factor = inds.numel() + 1e-6

    if inds.numel() > 0:
        inds = inds.squeeze(1)
    else:
        return (pred * weight).sum()[None] / avg_factor

    if style == 'bounded':
        loss = bounded_iou_loss(
            pred[inds], target[inds], beta=beta, eps=eps, reduction='sum')
    else:
        loss = iou_loss(pred[inds], target[inds], reduction='sum')
    loss = loss[None] / avg_factor
    return loss


def accuracy(pred, target, topk=1):
    """

    :param pred: (batch*num_sample, C)
    :param target: (batch*num_sample)
    :param topk:
    :return:
    """

    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True) # (batch*num_sample, 1)
    pred_label = pred_label.t() # (1, batch*num_sample)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label)) # (1, batch*num_sample)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def iou_loss(pred_bboxes, target_bboxes, reduction='mean'):
    ious = bbox_overlaps(pred_bboxes, target_bboxes, is_aligned=True)
    loss = -ious.log()

    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
