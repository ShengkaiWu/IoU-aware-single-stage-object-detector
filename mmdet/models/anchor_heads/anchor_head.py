from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, bbox_overlaps,
                        multi_apply, multiclass_nms,  weighted_cross_entropy,
                        weighted_smoothl1, weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss)
from mmdet.core.loss import weighted_iou_regression_loss
from ..builder import build_loss
from ..registry import HEADS

# from mmdet.core.bbox import bbox_overlaps
# from mmdet.core.bbox.transforms import delta2bbox
from mmdet.core.anchor.anchor_target import expand_binary_labels

@HEADS.register_module
class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC', 'IOUbalancedSigmoidFocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

        # added by Shengkai Wu
        self.IoU_balanced_Cls = loss_cls['type'] in ['IOUbalancedCrossEntropyLoss', 'IOUbalancedSigmoidFocalLoss']
        self.IoU_balanced_Loc = loss_bbox['type'] in ['IoUbalancedSmoothL1Loss']


    def _init_layers(self):
        self.use_iou_prediction = False
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
            anchor_list: list[list[Tensor]], anchor_list[i][j].size() is (num_anchor, 4) and
                  and stores the anchors for the j-th level feature map of the i-th image.
            valid_flag_list: list[list[Tensor]], valid_flag_list.size() is (num_anchor) and
                  stores the anchors' flags for the j-th level feature map of the i-th image.
                  1 represents valid anchor and 0 represents invalid anchors.

        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights,
                    level_anchor, # added by Shengkai Wu
                    num_total_samples,
                    gt_bboxes, # added by Shengkai Wu
                    cfg):
        """
        compute loss for a single layer of the prediction pyramid.
        :param cls_score: tensor of shape (batch, A*num_class, width_i, height_i)
        :param bbox_pred: tensor of shape (batch, A*4, width_i, height_i)
        :param labels: For RetinaNet, tensor of shape (batch, A*width*height) storing gt labels such as 1, 2, 80 for
              positive examples and 0 for negatives or others.
        :param label_weights: the same as labels. 1 for positive and negative examples, 0 for invalid anchors and neutrals.
        :param bbox_targets: tensor of shape (batch, A*width*height, 4). Store the parametrized coordinates of
              targets for positives and 0 for negatives and others.
        :param bbox_weights: tensor of shape (batch, A*width*height, 4). 1 for positives and 0 for negatives and others.
        :param level_anchor: tensor of shape (batch, A*width*height, 4)
        :param num_total_samples:
        :param gt_bboxes: list of tensor. gt_boxes[i].size() = (num_truth_i, 4), store the top-left and bottom-right corner of
              truth boxes for the i-th image.
        :param cfg:
        :return:
        """
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        # added by Shengkai Wu
        level_anchor = level_anchor.reshape(-1, 4)

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)


        IoU_balanced_Cls = self.IoU_balanced_Cls
        IoU_balanced_Loc = self.IoU_balanced_Loc
        # if IoU_balanced_Loc or self.use_iou_prediction:
        if IoU_balanced_Loc or IoU_balanced_Cls:
            pred_box = delta2bbox(level_anchor, bbox_pred, self.target_means, self.target_stds)
            # the negatives will stores the anchors information(x, y, w, h)
            target_box = delta2bbox(level_anchor, bbox_targets, self.target_means, self.target_stds)
            iou = bbox_overlaps(target_box, pred_box, is_aligned=True) # regressed IoU for positives
           # iou = bbox_overlaps(target_box, level_anchor, is_aligned=True) # original IoU for positives


        if IoU_balanced_Loc:
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                iou,
                bbox_weights,
                avg_factor=num_total_samples)
        else:
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

        # classification loss: focal loss for positive and negative examples
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # # added by Shengkai Wu
        if self.use_sigmoid_cls:
            # transform tensor 'label' from size (batch*A*width*height) to size (batch*A*width*height, num_class)
            #  and the same as tensor 'label_weights'. may be wrong for rpn
            labels, label_weights = expand_binary_labels(labels, label_weights, self.cls_out_channels)

        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if IoU_balanced_Cls:

            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                iou,
                #
                avg_factor=num_total_samples)
            # print('test')

        else:
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        """
        :param cls_scores: list[Tensor]. len(cls_scores) equals to the number of feature map levels.
              and cls_scores[i].size() is (batch, A*C, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param bbox_preds: list[Tensor]. len(bbox_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (batch, A*4, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param gt_bboxes: list[Tensor],Ground truth bboxes of each image. store the top-left and bottom-right corners
              in the image coordinte;
        :param gt_labels:
        :param img_metas: list[dict], Meta info of each image.
        :param cfg:
        :param gt_bboxes_ignore:
        :return:
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
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
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg,
         level_anchor_list) = cls_reg_targets
        # added by WSK
        # If sampling is adopted, num_total_samples = num_total_pos + num_total_neg;
        # otherwise, num_total_samples = num_total_pos. For 'FocalLoss', 'GHMC', 'IOUbalancedSigmoidFocalLoss',
        # sampling is not adopted.
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            level_anchor_list, # added by Shengkai Wu
            num_total_samples=num_total_samples,
            gt_bboxes = gt_bboxes, # added by Shengkai Wu
            cfg=cfg)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   gt_bboxes, # added by WSK
                   gt_labels, # aded by WSK
                   img_metas,
                   cfg,
                   rescale=False):
        """

        :param cls_scores: cls_scores: list[Tensor]. len(cls_scores) equals to the number of feature map levels.
              and cls_scores[i].size() is (batch, A*C, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param bbox_preds: list[Tensor]. len(bbox_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (batch, A*4, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param gt_bboxes:DC(mmcv.parallel.DataContainer), tensor of shape (num_gts, 4),represent
             the coordinates of top-left and bottom-right corner for each gt box,  (x_tl, y_tl, x_br, y_br)
        :param gt_labels: DC(mmcv.parallel.DataContainer), tensor of shape
        :param img_metas: list[dict], Meta info of each image. DC(mmcv.parallel.DataContainer), the imformation about one image
            dict{
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip}
        :param cfg: test.cfg
        :param rescale:
        :return:
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        # print('the number of images', len(img_metas))
        for img_id in range(len(img_metas)):
            # get the classification score and predicted box for each image
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # added by WSK
            gt_bboxes_oneimg = gt_bboxes[img_id]
            gt_labels_oneimg = gt_labels[img_id]


            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor,
                                               gt_bboxes_oneimg, gt_labels_oneimg, # added by WSK
                                               cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          gt_bboxes, # added by WSK
                          gt_labels, # added by WSK
                          cfg,
                          rescale=False):
        """
        process one image.
        :param cls_scores: list[Tensor]. len(cls_score) equals to the number of feature map levels.
              and cls_scores[i].size() is (A*C, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param bbox_preds: list[Tensor]. len(bbox_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (A*4, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map. if use_iou_branch is used, (A*5, width_i, height_i), the
              additional dim represents the predicted IoU.
        :param mlvl_anchors: list[Tensor]. len(mlvl_anchors) equals to the number of feature map levels.
              and mlvl_anchors[i].size() is (width_i*height_i*A, 4). width_i and height_i is the size
              of the i-th level feature map.
        :param img_shape: recale the image to ensure the short and long side is no longer than (short_max, long_max)
        :param scale_factor: the scaling factor.
        :param gt_bboxes: tensor of shape (num_gts, 4), the coordinates of top-left and bottom-right corners
        :param gt_labels:
        :param cfg:
        :param rescale:
        :return:
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors) # the number of levels for FPN
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            # added by WSK
            if self.use_iou_prediction:
                alpha = 0.2
                threshold = 0.96

                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
                bbox_pred_list = torch.split(bbox_pred, [4, 1], -1)
                bbox_pred = bbox_pred_list[0] # (width_i*height_i*A, 4)
                iou_pred = bbox_pred_list[1] # (width_i*height_i*A, 1)
                iou_pred = torch.squeeze(iou_pred)
                iou_pred = iou_pred.sigmoid()
                # iou_pred[iou_pred<threshold] = 0

                # compute the IoU between the regressed anchors and the ground truth boxes
                bboxes_pred_decode = delta2bbox(anchors, bbox_pred, self.target_means,
                                    self.target_stds, img_shape) # (width_i*height_i*A, 4)
                if gt_bboxes.size(0) >= 1:
                    overlaps = bbox_overlaps(gt_bboxes, bboxes_pred_decode) #(num_gt, width_i*height_i*A)
                    # iou_truth.size()=(width_i*height_i*A)
                    iou_truth, argmax_overlaps = overlaps.max(dim=0) # ()
                    # iou_truth[iou_truth< threshold] = 0
                else:
                    iou_truth = iou_pred.new_zeros(iou_pred.size())

                # multiply classification score with the class-agnostic IoU to compute the final
                # detection confidence
                # iou_expanded = iou_pred.view(-1, 1).expand(-1, scores.size(-1))
                iou_expanded = iou_truth.view(-1, 1).expand(-1, scores.size(-1))

                # scores = scores * iou_expanded
                scores = scores.pow(alpha) * iou_expanded.pow(1 - alpha)
                # print('test')

            else:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            # Added by WSK
            # multiply the classification score with predicted iou to compute the final
            # detection confidence.
            # if self.use_iou_prediction:
            #     iou_expanded = iou_pred.view(-1, 1).expand(-1, scores.size(-1))
            #     gamma = 1.0
            #     scores = scores * iou_expanded.pow(gamma)
                # print('the predicted iou is ', iou_expanded)
            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)


            # select the top-k detections for each feature level respectively.
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1) # (width_i*height_i*A)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)


                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes) #(n, 4)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores) #(n, C)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)# (n, 1+C)

        # apply nms to all the detections from all the feature levels
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
