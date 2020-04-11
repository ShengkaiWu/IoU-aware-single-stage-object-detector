import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from .anchor_head import AnchorHead
from ..utils import bias_init_with_prob, ConvModule

# added by WSK
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, bbox_overlaps,
                        multi_apply, multiclass_nms,  weighted_cross_entropy,
                        weighted_smoothl1, weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss)
from mmdet.core.loss import weighted_iou_regression_loss
from ..registry import HEADS
from ..builder import build_loss

from mmdet.core.anchor.anchor_target import expand_binary_labels
from mmdet.ops import DeformConv, MaskedConv2d

class FeatureAlignment(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlignment, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x




@HEADS.register_module
class IoUawareRetinaHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_iou=dict(type='GHMIoU', bins=30, momentum=0.75, use_sigmoid=True, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(IoUawareRetinaHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors*4, 3, padding=1)

        # added by WSK
        # analyze the effect of the shared conv layers between regression head and
        # IoU prediction head. The number of conv layers to extract features for
        # IoU prediction have to be kept to be 4.
        self.shared_conv = 4
        if self.shared_conv < 4:
            self.iou_convs = nn.ModuleList()
            for i in range(4-self.shared_conv):
                chn = self.in_channels if (self.shared_conv==0 and i == 0) else self.feat_channels
                self.iou_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
                )

        # feature alignment for IoU prediction
        self.use_feature_alignment = False
        self.deformable_groups = 4
        if self.use_feature_alignment:
            self.feature_alignment = FeatureAlignment(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deformable_groups = self.deformable_groups)
            self.retina_iou = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        else:
            self.retina_iou = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)



    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
        # added by WSK
        if self.shared_conv < 4:
            for m in self.iou_convs:
                normal_init(m.conv, std=0.01)
        normal_init(self.retina_iou, std=0.01)

        # added by WSK
        if self.use_feature_alignment:
            self.feature_alignment.init_weights()

    def forward_single(self, x):
        """
        process one level of FPN
        :param x: one feature level of FPN. tensor of size (batch, self.feat_channels, width_i, height_i)
        :return:
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        reg_feat_list = []
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            reg_feat_list.append(reg_feat)
        cls_score = self.retina_cls(cls_feat) # (batch, A*num_class, width_i, height_i)
        bbox_pred = self.retina_reg(reg_feat) # (batch, A*4, width_i, height_i)

        #added by WSK
        # concatenation of regression prediction and feature map for the input of
        # IoU prediction head
        # bbox_pred_clone = bbox_pred.clone()
        # bbox_pred_clone = bbox_pred_clone.detach()
        # reg_feat = torch.cat([reg_feat_list[-1],bbox_pred_clone], 1)

        # analyze the effect of the shared conv layers between regression head and
        # IoU prediction head.
        if self.shared_conv == 0:
            iou_feat = x
        else:
            iou_feat = reg_feat_list[self.shared_conv - 1]
        if self.shared_conv < 4:
            for iou_conv in self.iou_convs:
                iou_feat = iou_conv(iou_feat)
        # iou_pred = self.retina_iou(iou_feat) # (batch, A, width_i, height_i)

        # feature alignment for iou prediction
        if self.use_feature_alignment:
            bbox_pred_list = torch.split(bbox_pred, 4, dim=1)
            iou_pred_list = []
            for i in range(len(bbox_pred_list)):
                iou_feat_aligned = self.feature_alignment(iou_feat, bbox_pred_list[i])
                iou_pred_single_anchor = self.retina_iou(iou_feat_aligned) # (batch, 1, width_i, height_i)
                iou_pred_list.append(iou_pred_single_anchor)
            iou_pred = torch.cat(iou_pred_list, 1) # (batch, A, width_i, height_i)
        else:
            iou_pred = self.retina_iou(iou_feat)  # (batch, A, width_i, height_i)

        return cls_score, bbox_pred, iou_pred

    def loss_single(self, cls_score, bbox_pred, iou_pred,
                    labels, label_weights, bbox_targets, bbox_weights,
                    level_anchor,
                    num_total_samples,
                    gt_bboxes,  # added by Shengkai Wu
                    cfg):
        """
        compute loss for a single layer of the prediction pyramid.
        :param cls_score: tensor of shape (batch, A*num_class, width_i, height_i)
        :param bbox_pred: tensor of shape (batch, A*4, width_i, height_i)
        :param iou_pred: tensor of shape (batch, A, width_i, height_i), sigmoid layer has no been applied.
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

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) # (batch*A*width_i*height_i, 4)

        IoU_balanced_Cls = self.IoU_balanced_Cls
        IoU_balanced_Loc = self.IoU_balanced_Loc

        pred_box = delta2bbox(level_anchor, bbox_pred, self.target_means, self.target_stds)
        # the negatives will stores the anchors information(x, y, w, h)
        target_box = delta2bbox(level_anchor, bbox_targets, self.target_means, self.target_stds)
        iou = bbox_overlaps(target_box, pred_box, is_aligned=True) # (batch*width_i*height_i*A)

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

        # added by WSK
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1) # (batch*width_i*height_i*A)
        bbox_weight_list = torch.split(bbox_weights, 1, -1)
        bbox_weight = bbox_weight_list[0]
        bbox_weight = torch.squeeze(bbox_weight) # (batch*A*width_i*height_i)
        weight_iou = 1.0
        loss_iou = weight_iou*weighted_iou_regression_loss(iou_pred, iou, bbox_weight, avg_factor=num_total_samples)


        # classification loss: focal loss for positive and negative examples
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # # added by Shengkai Wu
        if self.use_sigmoid_cls:
            # transform tensor 'label' from size (batch*A*width*height) to size (batch*A*width*height, num_class)
            #  and the same as tensor 'label_weights'. may be wrong for rpn
            labels, label_weights = expand_binary_labels(labels, label_weights, self.cls_out_channels)

        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
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

        return loss_cls, loss_bbox, loss_iou

    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
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
        :param iou_preds: list[Tensor]. len(iou_preds) equals to the number of feature map levels.
              and iou_preds[i].size() is (batch, A, width_i, height_i). width_i and height_i is the size
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

        losses_cls, losses_bbox, losses_iou= multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            iou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            level_anchor_list, # added by Shengkai Wu
            num_total_samples=num_total_samples,
            gt_bboxes = gt_bboxes, # added by Shengkai Wu
            cfg=cfg)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, losses_iou=losses_iou)


    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   gt_bboxes, # added by WSK
                   gt_labels, # aded by WSK
                   img_metas,
                   cfg,
                   rescale=False):
        """

        :param cls_scores: list[Tensor]. len(cls_scores) equals to the number of feature map levels.
              and cls_scores[i].size() is (batch, A*C, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param bbox_preds: list[Tensor]. len(bbox_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (batch, A*4, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param iou_preds: list[Tensor]. len(iou_preds) equals to the number of feature map levels.
              and iou_preds[i].size() is (batch, A, width_i, height_i). width_i and height_i is the size
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

            # added by WSK
            iou_pred_list = [
                iou_preds[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # added by WSK
            gt_bboxes_oneimg = gt_bboxes[img_id]
            gt_labels_oneimg = gt_labels[img_id]


            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, iou_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor,
                                               gt_bboxes_oneimg, gt_labels_oneimg, # added by WSK
                                               cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          iou_preds,
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
              of the i-th level feature map.
        :param iou_preds: list[Tensor]. len(iou_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (A, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
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
        assert len(cls_scores) == len(bbox_preds) == len(iou_preds) == len(mlvl_anchors) # the number of levels for FPN
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(cls_scores, bbox_preds, iou_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            # added by WSK
            alpha = 0.5
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1) # (width_i*height_i*A)
            iou_pred = iou_pred.sigmoid()
            # print('the size of iou_pred is ', iou_pred.size())

            # compute the IoU between the regressed anchors and the ground truth boxes
            bboxes_pred_decode = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape) # (width_i*height_i*A, 4)
            if gt_bboxes.size(0) >= 1:
                overlaps = bbox_overlaps(gt_bboxes, bboxes_pred_decode) #(num_gt, width_i*height_i*A)
                # iou_truth.size()=(width_i*height_i*A)
                iou_truth, argmax_overlaps = overlaps.max(dim=0) # (width_i*height_i*A)
            else:
                iou_truth = iou_pred.new_zeros(iou_pred.size())

            # multiply classification score with the class-agnostic IoU to compute the final
            # detection confidence
            iou_expanded = iou_pred.view(-1, 1).expand(-1, scores.size(-1))
            # iou_expanded = iou_truth.view(-1, 1).expand(-1, scores.size(-1))
            # scores = scores * iou_expanded
            scores = scores.pow(alpha) * iou_expanded.pow(1 - alpha)
            #


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

