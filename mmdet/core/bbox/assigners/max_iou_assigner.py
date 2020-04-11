import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care, neutral examples
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt. If set to be True, for example, the max IoU for
            a ground truth box is 0.8 and there exist multiple anchors that have IoU 0.8
            with that ground truth boxes. The ground truth box will be assigned to all
            these anchors.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself.
         added by Shengkai Wu: the 4th step will assign the anchors that have the maximum
         IOU with the ground truth boxes with the corresponding ground truth box's label.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
            num_gts: the number of ground truth boxes;
            assigned_gt_inds: (num_bboxes), store the indexes(>=1) of ground truth boxes for the
                  positives, 0 for negatives and -1 for neutrals.
            max_overlaps: (num_bboxes), store the IoU of each anchors with the nearest ground truth boxes;

        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default.
        # size=(num_bboxes), store ground truth boxes index(>=1) for positives,
        # 0 for negatives and -1 for neutrals.
        assigned_gt_inds = overlaps.new_full(
            (num_bboxes, ), -1, dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0
        # added by WSK
        # use different iou threshold to assign label for classification and localization task
        # respectively. The threshold for classification can be set lower than that for localization
        # so that more positive anchors can be used to train classification sub-network.
        # use_different_threshold = False
        # neg_iou_thr_cls = 0.4
        # pos_iou_thr_cls = 0.4
        # if use_different_threshold:
        #     # assigned_gt_inds_cls: (num_bboxes), -1 for neutrals, 0 for negatives, the indexes of
        #     # ground truth boxes for positive examples. This is used for classification task.
        #     assigned_gt_inds_cls = overlaps.new_full(
        #         (num_bboxes,), -1, dtype=torch.long)
        #     assigned_gt_inds_cls[(max_overlaps >= 0)
        #                      & (max_overlaps < neg_iou_thr_cls)] = 0
        #     pos_inds = max_overlaps >= pos_iou_thr_cls
        #     assigned_gt_inds_cls[pos_inds] = argmax_overlaps[pos_inds] + 1
        #     for i in range(num_gts):
        #         if gt_max_overlaps[i] >= self.min_pos_iou:
        #             if self.gt_max_assign_all:
        #                 max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
        #                 assigned_gt_inds_cls[max_iou_inds] = i + 1
        #             else:
        #                 assigned_gt_inds_cls[gt_argmax_overlaps[i]] = i + 1

        # 3. assign positive: above positive IoU threshold
        # added by WSK:
        # assigned_gt_inds: size = (num_bboxes), store gt index for examples with IoU > pos_iou_thr
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # pos_iou_thr_loc = 0.5
        # if use_different_threshold:
        #     pos_inds_loc = max_overlaps >= pos_iou_thr_loc



        # 4. assign fg: for each gt, proposals with highest IoU
        # added by WSK
        # this step can't ensure all the ground truth boxes have the corresponding anchors. For examples,
        # the object i have the nearest anchor j with IoU=0.4. however, object k < i also have the nearest
        # anchor j with IoU=0.45. this step will make the object k have no corresponding anchor.
        # Differently, YOLACT assign the ground truth boxes with the highest IoU firstly in sequence in this step.
        # This can ensure that every ground truth box have at least one corresponding anchor.
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
