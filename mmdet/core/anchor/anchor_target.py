import torch

from ..bbox import assign_and_sample, build_assigner, PseudoSampler, bbox2delta
from ..utils import multi_apply


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image. anchor_list[i][j] represent the anchor set for the
            j-th level of feature pyramid of the i-th image and anchor_list[i][j].shape = (num_anchors, 4)
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image. gt_bboxes_list[i].size() = (num_truth, 4)
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
        labels_list: list of tensors. len(labels_list) equals to the num of layers of the feature pyramid and the
            labels_list[i].size() = (batch, A*width_i*height_i). class labels(1,2,...) for positives and 0 for the others.
        label_weights_list: the same as labels_list. 1 for positives and negatives, 0 for the others(invalid anchors and neutrals)
        bbox_targets_list: list of tensors. len(bbox_targets_list) equals to the num of layers of the feature pyramid
             and the bbox_targets_list[i].size() = (batch, A*width_i*height_i, 4). regression targets for positives and
             0 for the others
        bbox_weights_list: the same as bbox_targets_list. 1 for positive anchors, 0 for the others.
        num_total_pos: the number of positive examples in the batch;
        num_total_neg: the number of negative examples in the batch;
        level_anchor_list: list[Tensor]. level_anchor_list[i].size() is (batch, A*width_i*height_i, 4). store the original
             anchors.
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # the numbers of anchors for each feature pyramid level
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    # After tihs, anchor_list[i] is transformed from a list to a tensor and
    # anchor_list[i].shape equal to (num_anchors, 4), representing all the
    # anchors for the i-th image.
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    # all_labels: list[Tensor], all_labels[i].size() is (num_anchors) and stores the labels for all the
    #       anchors(valid and invalid) in the i-th image. 1,2...num_class for pos, 0 for the others
    # all_label_weights: list[Tensor], all_label_weights[i].size() is (num_anchors)
    #       1 for positives and negatives, 0 for the others(invalid anchors and neutrals)
    # all_bbox_targets: list[Tensor], all_bbox_targets[i].size() is (num_anchors, 4). the regression target
    #       for the positives, 0 for the others.
    # all_bbox_weights: list[Tensor], all_bbox_weights[i].size() is (num_anchors, 4). 1 for positive anchors,
    #      0 for others.
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single, # process each image respectively
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    # added by WSK
    # for each image, if the number of positives or negatives is 0, then adopt 1 for these two value
    # As a result, for a batch of image, num_total_pos >= num_images and num_total_neg >= num_images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

    # add by Shengkai Wu
    level_anchor_list = images_to_levels(anchor_list, num_level_anchors)

    return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg,
            level_anchor_list)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    target: list[Tensor], target[i].size() = (num_anchors,) for the i-th image;
    num_level_anchors: list of numbers, store the numbers of anchors of each level;

    level_targets[i].size() = (batch_size, A*width_i*height_i, ) for the i-th level of FPN
    """
    target = torch.stack(target, 0) # (batch_size, num_anchors, )
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    """

    :param flat_anchors: tensor of shape (num_anchors, 4), representing all the anchors for a image.
    :param valid_flags: tensor of shape (num_anchors), representing all the anchors' flags for a image.
          1 represents valid anchor and 0 represents invalid anchor.
    :param gt_bboxes: tensor of shape (num_truth, 4), representing all the ground truth boxes for a image.
    :param gt_bboxes_ignore: tensor
    :param gt_labels: tensor
    :param img_meta: dict
    :param target_means:
    :param target_stds:
    :param cfg:
    :param label_channels:
    :param sampling: whether to sample a subset examples from positive examples and negative examples.
    :param unmap_outputs:
    :return:
        labels: tensor of shape (num_anchors). num_anchors is the number of all anchors.
              store class label such as 1, 2 for positive examples and 0 for negatives or others
        label_weights: (num_anchors), 1 for positives and negatives, 0 for the others(invalid anchors and neutrals)
        bbox_targets: (num_anchors, 4), regression target(encoded ground truth box) for the positives
              and 0 for others.
        bbox_weights: (num_anchors, 4), 1 for positives and 0 for others.
        pos_inds: (num_pos), indexes of positive anchors;
        neg_inds: (num_neg), indexes of negative anchors;

    """
    # tensor of size(): (num_anchors), 1 for valid anchor, 0 for invalid anchor.
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors

    # only the valid anchors are kept and the invalid anchors are discarded.
    # For example, flat_anchors.size()=(102123, 4), valid_flags.size()=(102123)
    # anchors.size()=(69354, 4).
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)
    pos_inds = sampling_result.pos_inds  # (num_pos), indexes of positive anchors
    neg_inds = sampling_result.neg_inds  # (num_neg), indexes of negative anchors

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)


    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets

        # added by Shengkai Wu
        # store the correspondign anchors for every ground truth box. (x1, y1, x2, y2)
        # proposals_to_targets[pos_inds, :] =  sampling_result.pos_bboxes
        bbox_weights[pos_inds, :] = 1.0

        # gt_labels = none is used for rpn
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            # the labels for each positive example
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight

    if len(neg_inds) > 0:

        label_weights[neg_inds] = 1.0

    # map up to original set of anchors so that labels has corresponding relation with flat_anchors.
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)

        # added by Shengkai Wu
        # the following three line codes are copied from mmdetection-pytorch-0.4.1 and added by Shengkai Wu.
        # if label_channels > 1:
        #     labels, label_weights = expand_binary_labels(
        #         labels, label_weights, label_channels)

        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        # proposals_to_targets = unmap(proposals_to_targets, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)


# added by Shengkai Wu
# this function is copied from mmdetection-pytorch-0.4.1
def expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1 # assign 1 to the corresponding label
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights



def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
