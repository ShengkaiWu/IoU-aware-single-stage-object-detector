import torch

from .transforms import bbox2delta
from ..utils import multi_apply


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    """

    :param pos_bboxes_list:
    :param neg_bboxes_list:
    :param pos_gt_bboxes_list:
    :param pos_gt_labels_list:
    :param cfg:
    :param reg_classes:
    :param target_means:
    :param target_stds:
    :param concat:
    :return:
    labels: tensor of shape (batch*num_samples), labels for positives; others are set to be 0;
    label_weights: tensor of shape (batch*num_samples), all the elements are set to be 1;
    bbox_targets: tensor of shape (batch*num_samples, 4), the parameterized coordinates of the
        corresponding ground truth boxes for positives; negatives are set to be 0;
    bbox_weights: tensor of shape (batch*num_samples, 4), the positives are set to be 1; others are 0;
    proposal_bboxes:tensor of shape (batch*num_samples, 4), positive proposals for positives and others
    are 0.
    """
    labels, label_weights, bbox_targets, bbox_weights, proposal_bboxes = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)

        # added by Shengkai Wu
        proposal_bboxes = torch.cat(proposal_bboxes, 0)

    return labels, label_weights, bbox_targets, bbox_weights, proposal_bboxes


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    """
    process for each image in the batch
    :param pos_bboxes: tensor of shape (num_pos, 4)
    :param neg_bboxes: tensor of shape (num_neg, 4)
    :param pos_gt_bboxes:
    :param pos_gt_labels:
    :param cfg:
    :param reg_classes:
    :param target_means:
    :param target_stds:
    :return:
    """
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    proposal_bboxes = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1

        # added by Shengkai Wu
        proposal_bboxes[:num_pos, :] = pos_bboxes

    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights, proposal_bboxes


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros((bbox_targets.size(0),
                                                  4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros((bbox_weights.size(0),
                                                  4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
