import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (sigmoid_focal_loss, iou_loss, multi_apply,
                        multiclass_nms, distance2bbox, bbox_overlaps)
from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule

INF = 1e8


@HEADS.register_module
class IoUawareFCOSHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(IoUawareFCOSHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
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
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # added by WSK
        self.fcos_iou = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

        # added by WSK
        normal_init(self.fcos_iou, std=0.01)


    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # scale the bbox_pred of different level
        bbox_pred = scale(self.fcos_reg(reg_feat)).exp()

        # added by WSK
        iou = self.fcos_iou(reg_feat)

        return cls_score, bbox_pred, centerness, iou

    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             ious,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        """

        :param cls_scores: list of tensor. cls_scores[i].size() = (batch, num_class, width_i, height_i) for
              the i-th level of FPN.
        :param bbox_preds: list of tensor. bbox_preds[i].size() = (batch, 4, width_i, height_i) for
              the i-th level of FPN.
        :param centernesses: list of tensor. centernesses[i].size() = (batch, 1, width_i, height_i) for
              the i-th level of FPN.
        :param iou: list of tensor. centernesses[i].size() = (batch, 1, width_i, height_i) for
              the i-th level of FPN.
        :param gt_bboxes:
        :param gt_labels:
        :param img_metas:
        :param cfg:
        :param gt_bboxes_ignore:
        :return:
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) ==len(ious)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        # added by WSK
        flatten_iou = [
            iou.permute(0, 2, 3, 1).reshape(-1)
            for iou in ious
        ]
        flatten_iou = torch.cat(flatten_iou) # (num_all)


        flatten_cls_scores = torch.cat(flatten_cls_scores) # (num_all, num_class)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds) # (num_all, 4)
        flatten_centerness = torch.cat(flatten_centerness) # (num_all)
        flatten_labels = torch.cat(labels) # (num_all)
        flatten_bbox_targets = torch.cat(bbox_targets) # (num_all, 4)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = sigmoid_focal_loss(
            flatten_cls_scores, flatten_labels, cfg.gamma, cfg.alpha,
            'none').sum()[None] / (num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds] # (num_pos, 4)
        pos_bbox_targets = flatten_bbox_targets[pos_inds] # (num_pos, 4)
        pos_centerness = flatten_centerness[pos_inds] # (num_pos)
        pos_centerness_targets = self.centerness_target(pos_bbox_targets) # (num_pos)

        # added by WSK
        pos_iou = flatten_iou[pos_inds] # (num_pos)


        if num_pos > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds) # (num_pos, 4)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets) # (num_pos, 4)
            # centerness weighted iou loss
            loss_reg = ((iou_loss(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                reduction='none') * pos_centerness_targets).sum() /
                        pos_centerness_targets.sum())[None]
            loss_centerness = F.binary_cross_entropy_with_logits(
                pos_centerness, pos_centerness_targets, reduction='mean')[None]

            # added by WSK
            pos_iou_target = bbox_overlaps(pos_decoded_target_preds, pos_decoded_bbox_preds, is_aligned=True)
            # pos_iou_target = pos_iou_target.detach()
            loss_iou = F.binary_cross_entropy_with_logits(
                pos_iou, pos_iou_target, reduction='mean')[None]

        else:
            loss_reg = pos_bbox_preds.sum()[None]
            loss_centerness = pos_centerness.sum()[None]

            # added by WSK
            loss_iou = pos_iou.sum()[None]


        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_centerness=loss_centerness,
            loss_iou = loss_iou
        )


    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   ious,
                   gt_bboxes,
                   gt_labels,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]

            #added by WSK
            iou_pred_list = [
                ious[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, centerness_pred_list, iou_pred_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          ious,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """

        :param cls_scores: list[Tensor]. len(cls_score) equals to the number of feature map levels.
              and cls_scores[i].size() is (C, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param bbox_preds: list[Tensor]. len(bbox_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (4, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param centernesses: list[Tensor]. len(centernesses) equals to the number of feature map levels.
              and centernesses[i].size() is (1, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param ious: list[Tensor]. len(ious) equals to the number of feature map levels.
              and ious[i].size() is (1, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param mlvl_points:
        :param img_shape:
        :param scale_factor:
        :param cfg:
        :param rescale:
        :return:
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_iou = []
        for cls_score, bbox_pred, centerness, iou, points in zip(
                cls_scores, bbox_preds, centernesses, ious, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            # added by WSK
            alpha = 0.3
            iou = iou.permute(1, 2, 0).reshape(-1).sigmoid()
            iou_expanded = iou.view(-1, 1).expand(-1, scores.size(-1))
            # scores = scores * iou_expanded
            scores = scores.pow(alpha) * iou_expanded.pow(1 - alpha)



            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                # max_scores, _ = (scores * iou[:, None]).max(dim=1)
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                iou = iou[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_iou.append(iou)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_iou = torch.cat(mlvl_iou)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)

        # det_bboxes, det_labels = multiclass_nms(
        #     mlvl_bboxes,
        #     mlvl_scores,
        #     cfg.score_thr,
        #     cfg.nms,
        #     cfg.max_per_img,
        #     score_factors=mlvl_iou
        # )


        # det_bboxes, det_labels = multiclass_nms(
        #     mlvl_bboxes,
        #     mlvl_scores,
        #     cfg.score_thr,
        #     cfg.nms,
        #     cfg.max_per_img,
        #     score_factors=mlvl_centerness)

        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """

        :param pos_bbox_targets: tensor of shape (num, 4), store the top-left and bottom-right
              corner of the corresponding ground truth boxes for the positives.
        :return:
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]] #(num, 2)
        top_bottom = pos_bbox_targets[:, [1, 3]] #(num, 2)
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
