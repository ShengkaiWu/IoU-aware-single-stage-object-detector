import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (AnchorGenerator, anchor_target, weighted_smoothl1, weighted_iou_balanced_smoothl1,
                        multi_apply, delta2bbox, bbox_overlaps)
from mmdet.core.loss import weighted_iou_regression_loss
from .anchor_head import AnchorHead
from ..registry import HEADS


# TODO: add loss evaluator for SSD
@HEADS.register_module
class SSDHead(AnchorHead):

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes

        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]  # 4 or 6 ratio
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, anchors, num_total_samples, cfg):
        """
        compute the losses for one image
        :param cls_score: tensor of shape [num_total_examples, cls_out_channels]
        :param bbox_pred: tensor of shape [num_total_examples, 4] or [num_total_examples, 5]
        :param labels: tensor os shape [num_total_examples] storing gt labels such as
               0, 1, 2, 80 for corresponding class.
        :param label_weights: tensor of shape [num_total_examples]
        :param bbox_targets: tensor of shape [num_total_examples, 4], Store the parametrized
               coordinates of targets for positives and 0 for negatives.
        :param bbox_weights: tensor of shape [num_total_examples, 4], 1 for positives and 0 for negatives and neutrals.
        :param anchors: tensor of shape [num_total_examples, 4]
        :param num_total_samples: the number of positive examples.
        :param cfg:
        :return:
        """
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # index tensor of shape (Pos, 1), each row is a index for a non-zero element.
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1) # (Neg, 1)

        # added by Shengkai Wu
        IoU_balanced_Cls = False
        IoU_balanced_Loc = False
        eta = 1.5
        delta = 1.5
        cls_loss_weight = 1.0
        bbox_loss_weight = 1.0

        if IoU_balanced_Cls or IoU_balanced_Loc:
            pred_box = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds)
            # the negatives will stores the anchors information(x, y, w, h)
            target_box = delta2bbox(anchors, bbox_targets, self.target_means, self.target_stds)
            # iou between the regressed positive example and the corresponding ground truth box or
            # iou between the regressed negative example and the original negative example.
            iou = bbox_overlaps(target_box, pred_box, is_aligned=True)

        if IoU_balanced_Cls:
            target = iou.new_zeros(iou.size(0))
            target[pos_inds] = 1
            # target = target.type_as(cls_score)
            iou_weights = (1 - target) + (target * iou).pow(eta)
            raw2 = loss_cls_all * iou_weights
            normalizer = (loss_cls_all * target).sum() / ((raw2 * target).sum() + 1e-6)
            normalized_iou_weights = (1 - target) + (target * iou).pow(eta) * normalizer
            normalized_iou_weights = normalized_iou_weights.detach()
            loss_cls_all = loss_cls_all * normalized_iou_weights

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if IoU_balanced_Loc:
            loss_bbox = bbox_loss_weight * weighted_iou_balanced_smoothl1(
                bbox_pred,
                bbox_targets,
                iou,
                bbox_weights,
                beta=cfg.smoothl1_beta,
                delta=delta,
                avg_factor=num_total_samples)
        else:
            loss_bbox = weighted_smoothl1(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                beta=cfg.smoothl1_beta,
                avg_factor=num_total_samples)

        return loss_cls[None], loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        """

        :param cls_scores: list of tensors. len(cls_scores) is the number of levels for the feature pyramid.
          cls_scores[i].size() = [num_images, A*cls_out_channels, width_i, height_i]
        :param bbox_preds: list of tensors. len(bbox_preds) is the number of levels for the feature pyramid.
              bbox_preds[i].size() = [num_images, A*4, width_i, height_i] or [num_images, A*5, width_i, height_i]
        :param gt_bboxes:
        :param gt_labels:
        :param img_metas:
        :param cfg:
        :param gt_bboxes_ignore:
        :return:
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        # modifyied by Shengkai Wu
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, level_anchor_list) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)


        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)

        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)
        # added by Shengkai Wu
        all_anchors = torch.cat(level_anchor_list, -2).view(num_images, -1, 4)
        # num_total_examples = num_total_pos + num_total_neg
        # all_cls_scores.size() = [num_images,num_total_examples , cls_out_channels]
        # all_bbox_preds.size() = [num_image, num_total_examples, 4]
        # all_labels.size() = [num_images, num_total_examples]
        # all_label_weights.size() = [num_images, num_total_examples]
        # all_bbox_preds.size() = [num_images, num_total_examples, 4]
        # all_bbox_targets.size() = [num_images, num_total_examples, 4]
        # all_bbox_weights.size() = [num_images, num_total_examples, 4]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            all_anchors, #added by Shengkai Wu
            num_total_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
