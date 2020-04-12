import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from .guided_anchor_head import GuidedAnchorHead, FeatureAdaption
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from mmdet.ops import MaskedConv2d

# added by WSK
from mmdet.core import (AnchorGenerator, anchor_target, anchor_inside_flags,
                        ga_loc_target, ga_shape_target, delta2bbox,  bbox_overlaps,
                        multi_apply, multiclass_nms)
from mmdet.core.loss import weighted_iou_regression_loss
from ..builder import build_loss
from mmdet.core.anchor.anchor_target import expand_binary_labels


@HEADS.register_module
class IoUawareGARetinaHead(GuidedAnchorHead):
    """IoU-aware Guided-Anchor-based RetinaNet head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(IoUawareGARetinaHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=1,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=1,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        self.feature_adaption_cls = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.feature_adaption_reg = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.retina_cls = MaskedConv2d(self.feat_channels,
                                       self.num_anchors *
                                       self.cls_out_channels,
                                       3,
                                       padding=1)
        self.retina_reg = MaskedConv2d(self.feat_channels,
                                       self.num_anchors * 4,
                                       3,
                                       padding=1)

        # added by WSK
        self.retina_iou = MaskedConv2d(self.feat_channels,
                                       self.num_anchors,
                                       3,
                                       padding=1)


    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        self.feature_adaption_cls.init_weights()
        self.feature_adaption_reg.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

        # added by WSK
        normal_init(self.retina_iou, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        cls_feat = self.feature_adaption_cls(cls_feat, shape_pred)
        reg_feat = self.feature_adaption_reg(reg_feat, shape_pred)

        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.retina_cls(cls_feat, mask)
        bbox_pred = self.retina_reg(reg_feat, mask)

        # added by WSK
        iou_pred = self.retina_iou(reg_feat, mask)

        return cls_score, bbox_pred, iou_pred, shape_pred, loc_pred


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
        :param iou_pred: tensor of shape (batch, A, width_i, height_i), sigmoid layer has not been applied.
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

        batch_size = level_anchor.size(0)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        # added by Shengkai Wu
        level_anchor = level_anchor.reshape(-1, 4)



        # compute the target for the IoU prediction head.
        # bbox_pred_1 = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)  # (batch, A*width_i*height_i, 4)
        # level_anchor_1 = level_anchor.reshape(batch_size, -1, 4) # (batch, A*width_i*height_i, 4)
        # max_iou_list = []
        # for i in range(batch_size):
        #     level_anchor_i = level_anchor_1[i, :, :].reshape(-1, 4)  # (A*width_i*height_i, 4)
        #     bbox_pred__i = bbox_pred_1[i, :, :].reshape(-1, 4)  # (A*width_i*height_i, 4)
        #     pred_bbox_i = delta2bbox(level_anchor_i, bbox_pred__i, self.target_means, self.target_stds)
        #     regressed_iou = bbox_overlaps(gt_bboxes[i], pred_bbox_i)  # (num_truth, A*width_i*height_i)
        #     # max_overlaps.size() = (A*width_i*height_i)
        #     max_iou, argmax_overlaps = regressed_iou.max(dim=0)
        #     max_iou_list.append(max_iou)
        # max_iou_target = torch.cat(max_iou_list, dim=0)  # (batch*A*width_i*height_i)

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) # (batch*A*width_i*height_i, 4)

        IoU_balanced_Cls = self.IoU_balanced_Cls
        IoU_balanced_Loc = self.IoU_balanced_Loc
        # if IoU_balanced_Loc or IoU_balanced_Cls:

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
        # for the positives, use the iou between each pos and its corresponding truth; for negatives and neutrals,
        # use the iou between each example and its nearest(with max iou) truth.
        # iou_target = iou*bbox_weight + max_iou_target*(1-bbox_weight)

        weight_iou = 1.0
        loss_iou = weight_iou*weighted_iou_regression_loss(iou_pred, iou, bbox_weight, avg_factor=num_total_samples)
        # GHM for IoU prediction loss
        # loss_iou = self.loss_iou(iou_pred, iou, bbox_weight, avg_factor=num_total_samples)

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
             shape_preds,
             loc_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.approx_generators)

        # get loc targets
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.octave_base_scale,
            self.anchor_strides,
            center_ratio=cfg.center_ratio,
            ignore_ratio=cfg.ignore_ratio)

        # get sampled approxes
        approxs_list, inside_flag_list = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg)
        # get squares and guided anchors
        squares_list, guided_anchors_list, _ = self.get_anchors(
            featmap_sizes, shape_preds, loc_preds, img_metas)

        # get shape targets
        sampling = False if not hasattr(cfg, 'ga_sampler') else True
        shape_targets = ga_shape_target(
            approxs_list,
            inside_flag_list,
            squares_list,
            gt_bboxes,
            img_metas,
            self.approxs_per_octave,
            cfg,
            sampling=sampling)
        if shape_targets is None:
            return None
        (bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num,
         anchor_bg_num) = shape_targets
        anchor_total_num = (
            anchor_fg_num if not sampling else anchor_fg_num + anchor_bg_num)

        # get anchor targets
        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            guided_anchors_list,
            inside_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        # (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        #  num_total_pos, num_total_neg) = cls_reg_targets

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg,
         level_anchor_list) = cls_reg_targets


        num_total_samples = (
            num_total_pos if self.cls_focal_loss else num_total_pos +
            num_total_neg)

        # get classification and bbox regression losses
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            iou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            level_anchor_list,  # added by Shengkai Wu
            num_total_samples=num_total_samples,
            gt_bboxes=gt_bboxes,  # added by Shengkai Wu
            cfg=cfg)

        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],
                loc_targets[i],
                loc_weights[i],
                loc_avg_factor=loc_avg_factor,
                cfg=cfg)
            losses_loc.append(loss_loc)

        # get anchor shape loss
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(
                shape_preds[i],
                bbox_anchors_list[i],
                bbox_gts_list[i],
                anchor_weights_list[i],
                anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_iou=losses_iou,
            loss_shape=losses_shape,
            loss_loc=losses_loc)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   shape_preds,
                   loc_preds,
                   gt_bboxes,  # added by WSK
                   gt_labels,  # aded by WSK
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
        :param shape_preds:
        :param loc_preds:
        :param gt_bboxes: DC(mmcv.parallel.DataContainer), tensor of shape (num_gts, 4),represent
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
        assert len(cls_scores) == len(bbox_preds) == len(iou_preds) == \
               len(shape_preds) == len(loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # get guided anchors
        _, guided_anchors, loc_masks = self.get_anchors(
            featmap_sizes,
            shape_preds,
            loc_preds,
            img_metas,
            use_loc_filter=not self.training)
        result_list = []
        for img_id in range(len(img_metas)):
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

            guided_anchor_list = [
                guided_anchors[img_id][i].detach() for i in range(num_levels)
            ]
            loc_mask_list = [
                loc_masks[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            # added by WSK
            gt_bboxes_oneimg = gt_bboxes[img_id]
            gt_labels_oneimg = gt_labels[img_id]

            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               iou_pred_list, # added by WSK
                                               guided_anchor_list,
                                               loc_mask_list, img_shape,
                                               scale_factor,
                                               gt_bboxes_oneimg, gt_labels_oneimg,  # added by WSK
                                               cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          iou_preds,
                          mlvl_anchors,
                          mlvl_masks,
                          img_shape,
                          scale_factor,
                          gt_bboxes,  # added by WSK
                          gt_labels,  # added by WSK
                          cfg,
                          rescale=False):
        """

        :param cls_scores: list[Tensor]. len(cls_scores) equals to the number of feature map levels.
              and cls_scores[i].size() is (A*C, width_i, height_i). width_i and height_i is the size
              of the i-th level feature map.
        :param bbox_preds: list[Tensor]. len(bbox_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (A*4, width_i, height_i).
        :param iou_preds: list[Tensor]. len(iou_preds) equals to the number of feature map levels.
              and bbox_preds[i].size() is (A, width_i, height_i).
        :param mlvl_anchors: list[Tensor]. len(mlvl_anchors) equals to the number of feature map levels.
              and mlvl_anchors[i].size() is (width_i*height_i*A, 4).
        :param mlvl_masks: list[Tensor]. len(mlvl_anchors) equals to the number of feature map levels.
              and mlvl_anchors[i].size() is (width_i*height_i*A).
        :param img_shape:
        :param scale_factor:
        :param gt_bboxes: tensor of shape (num_gts, 4), the coordinates of top-left and bottom-right corners
        :param gt_labels:
        :param cfg:
        :param rescale:
        :return:
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, iou_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       iou_preds, # added by WSK
                                                       mlvl_anchors,
                                                       mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == iou_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and bbox_pred
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # added by WSK
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1)
            iou_pred = iou_pred.sigmoid()
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            # added by WSK
            iou_pred = iou_pred[mask]

            # multiply classification score with the class-agnostic IoU to compute the final
            # detection confidence
            alpha = 1.0
            iou_expanded = iou_pred.view(-1, 1).expand(-1, scores.size(-1))
            scores = scores * iou_expanded
            # scores = scores.pow(alpha) * iou_expanded.pow(1 - alpha)

            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)

            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
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
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels