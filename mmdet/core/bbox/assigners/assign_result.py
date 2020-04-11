import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        """

        :param num_gts: number of ground truth boxes
        :param gt_inds: (num_anchors), store indexes(>=1) of corresponding ground truth boxes
              for positive examples, -1 for neutrals and 0 for negatives.
        :param max_overlaps: (num_anchors), the max IoU for each anchors with the nearest
              ground truth box.
        :param labels:
        """
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        """

        :param gt_labels:
        :return:
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
