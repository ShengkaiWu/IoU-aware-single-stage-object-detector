import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        """

        :param pos_inds: (num_pos), store the indexes for postives
        :param neg_inds: (num_neg), store the indexes for negatives;
        :param bboxes: (num_anchors, 4)
        :param gt_bboxes: (num_truth, 4)
        :param assign_result:
        :param gt_flags:
        """
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds] # (num_pos, 4), positive anchor boxes
        self.neg_bboxes = bboxes[neg_inds] # (num_neg, 4), negative anchor boxes
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1 # (num_pos), the indexes of ground truth boxes for positives
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :] # (num_pos, 4), corresponding ground truth boxes for positives
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
