import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import delta2bbox, multiclass_nms, bbox_target, accuracy, bbox_overlaps
from mmdet.core.loss import weighted_iou_regression_loss
from ..builder import build_loss
from ..registry import HEADS

import numpy as np


@HEADS.register_module
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        # added by WSK
        self.use_iou_prediction = False
        self.use_class_agnostic_iou = True

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:

            # added by WSK
            # use regression to predict the iou
            if self.use_iou_prediction:
                out_dim_reg = 5 if reg_class_agnostic else 5 * num_classes
            else:
                out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
                # print("test: bbox_head")

            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        # added by Shengkai Wu
        self.IoU_balanced_Cls = loss_cls['type'] in ['IOUbalancedCrossEntropyLoss', 'IOUbalancedSigmoidFocalLoss']
        self.IoU_balanced_Loc = loss_bbox['type'] in ['IoUbalancedSmoothL1Loss']

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             proposal_bboxes, # added by Shengkai Wu
             reduce=True):
        """

        :param cls_score: tensor of shape (batch*num_samples, num_classes),num_samples = num_pos+num_neg;
            store the classification scores;
        :param bbox_pred: tensor of shape (batch*num_samples, 4) or (batch*num_samples, num_classes*4)/
            if use_iou_prediction is True, (batch*num_samples, 5) or (batch*num_samples, num_classes*5)
        :param labels: tensor of shape (batch*num_samples), contain labels for the positives
            and 0 for negatives;
        :param label_weights: tensor of shape (batch*num_samples), 1 for all the elements;
        :param bbox_targets: tensor of shape (batch*num_samples, 4), parametrized coordinates of
            ground truth boxes for positives and 0 for negatives;
        :param bbox_weights: tensor of shape (batch*num_samples, 4), 1 for positives and 0 for
            negatives;
        :param proposal_bboxes: tensor of shape (batch*num_samples, 4), coordinates of positive proposals
            and 0 for negatives.
        :param reduce:
        :return:
        """
        losses = dict()
        # the original implementation of mmdetection
        # if cls_score is not None:
        #     losses['loss_cls'] = self.loss_cls(
        #         cls_score, labels, label_weights, reduce=reduce)
        #     losses['acc'] = accuracy(cls_score, labels)
        # if bbox_pred is not None:
        #     pos_inds = labels > 0
        #     if self.reg_class_agnostic:
        #         pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
        #     else:
        #         pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
        #                                        4)[pos_inds, labels[pos_inds]]
        #     losses['loss_bbox'] = self.loss_bbox(
        #         pos_bbox_pred,
        #         bbox_targets[pos_inds],
        #         bbox_weights[pos_inds],
        #         avg_factor=bbox_targets.size(0))

        # add by Shengkai Wu, modified from the original implementation of mmdetection.
        # pos_inds = labels > 0

        # another way to get the pos_inds
        pos_inds = (labels>0).nonzero().view(-1)

        # added by WSK
        if self.use_iou_prediction:
            # print('the size of pos_bbox_pred is ', bbox_pred.size())
            if self.use_class_agnostic_iou:
                bbox_pred_list = torch.split(bbox_pred, [4 * self.num_classes, 1], -1)
                bbox_pred = bbox_pred_list[0]
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]  # (num_pos, 4)
                iou_pred = bbox_pred_list[1].squeeze()[pos_inds] # (num_pos)
                # print('test')
            else:
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds] # (num_pos, 5)
                    # print('test')
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                                   5)[pos_inds, labels[pos_inds]] # (num_pos, 5)

                pos_bbox_pred_list = torch.split(pos_bbox_pred, [4, 1], -1)
                pos_bbox_pred = pos_bbox_pred_list[0] # (num_pos, 4)
                iou_pred = pos_bbox_pred_list[1] # (num_pos, 1)
                iou_pred = iou_pred.squeeze()
                # print('test2')
        else:
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds] # (num_pos, 4)
                # print('test')
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]] # (num_pos, 4)
            # print("test: bbox_head")

        # compute the iou between predicted boxes and corresponding ground truth
        # boxes for positives. The shape of iou is (batch*num_pos)
        IoU_balanced_Cls = self.IoU_balanced_Cls
        IoU_balanced_Loc = self.IoU_balanced_Loc


        if IoU_balanced_Cls or IoU_balanced_Loc or self.use_iou_prediction:
            # ignoring target_mean and target_std will result in wrong target_box and pred_box and thus wrong
            # iou.
            pred_box = delta2bbox(proposal_bboxes[pos_inds], pos_bbox_pred, self.target_means, self.target_stds)
            target_box = delta2bbox(proposal_bboxes[pos_inds], bbox_targets[pos_inds], self.target_means, self.target_stds)
            iou = bbox_overlaps(target_box, pred_box, is_aligned=True) # (num_pos)
            # iou = bbox_overlaps(target_box,  proposal_bboxes[pos_inds], is_aligned=True)

            # print('pos_inds.size() is', pos_inds.size())
            # print('pos_inds is ', pos_inds)
            # iou_cpu = iou.cpu()
            # iou_cpu = iou_cpu.detach()
            # np_iou = iou_cpu.numpy()
            # np.savetxt('iou.csv', np_iou, delimiter=',')
            #
            # iou_original_cpu = iou_original.cpu()
            # iou_original_cpu = iou_original_cpu.detach()
            # np_iou_original = iou_original_cpu.numpy()
            # np.savetxt('iou_original.csv', np_iou_original, delimiter=',')


            # print('iou is', iou)
            # print('pos_bbox_pred.size() is', pos_bbox_pred.size())
            # print('iou.size() is', iou.size())
            # print('proposal_iou is ', proposal_iou)


        if bbox_pred is not None:
            if IoU_balanced_Loc:
                # print('the size of bbox_weights is ', bbox_weights[pos_inds].size())
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    iou,
                    bbox_weights[pos_inds],
                    avg_factor=bbox_targets.size(0))

            else:
                # print('the size of bbox_weights is ', bbox_weights[pos_inds].size())
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    bbox_weights[pos_inds],
                    avg_factor=bbox_targets.size(0))

        # added by WSK
        if self.use_iou_prediction:
            bbox_weight_list = torch.split(bbox_weights[pos_inds], 1, -1)
            bbox_weight = bbox_weight_list[0]
            bbox_weight = torch.squeeze(bbox_weight)
            weight_iou = 1.0
            loss_iou = weight_iou * weighted_iou_regression_loss(iou_pred, iou, bbox_weight,
                                                                 avg_factor=bbox_targets.size(0))
            losses['loss_bbox'] = losses['loss_bbox'] + loss_iou

        if cls_score is not None:
            if IoU_balanced_Cls:
                iou_extended = bbox_pred.new_zeros(bbox_pred.size(0))
                iou_extended[pos_inds] = iou
                # pos_inds_test = (iou_extended > 0).nonzero().view(-1)
                #
                # print('iou_extended.size() is', iou_extended.size())
                # print('pos_inds_test.size() is', pos_inds_test.size())
                # print('pos_inds is', pos_inds)
                # print('pos_inds_test is', pos_inds_test)
                # print('iou_extended is', iou_extended)

                losses['loss_cls'] = self.loss_cls(
                    cls_score, labels, label_weights, iou_extended, reduce=reduce)
                losses['acc'] = accuracy(cls_score, labels)
            else:
                losses['loss_cls'] = self.loss_cls(
                    cls_score, labels, label_weights, reduce=reduce)
                losses['acc'] = accuracy(cls_score, labels)

        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        """
        :param rois:
        :param cls_score: tensor of shape (batch*num_samples, num_classes),num_samples = num_pos+num_neg;
            store the classification scores;
        :param bbox_pred: tensor of shape (batch*num_samples, 4) or (batch*num_samples, num_classes*4)/
            if use_iou_prediction is True, (batch*num_samples, 5) or (batch*num_samples, num_classes*5);
        :param img_shape:
        :param scale_factor:
        :param rescale:
        :param cfg:
        :return:
        """
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:

            # added by WSK
            if self.use_iou_prediction:
                alpha = 1.0
                if bbox_pred.size(1)==5:
                    bbox_pred_list = torch.split(bbox_pred, [4,1], -1)
                    bbox_pred = bbox_pred_list[0] # (batch*num_samples, 4)
                    iou_pred = bbox_pred_list[1] # (batch*num_samples, 1)
                    iou_pred = iou_pred.sigmoid()
                    iou_pred = iou_pred.expand(-1, scores.size(-1))
                    # scores = scores * iou_pred

                    scores = scores.pow(alpha)*iou_pred.pow(1-alpha)
                    # scores = alpha*scores + (1-alpha)*iou_pred
                    # print('test')

                elif self.use_class_agnostic_iou:
                    bbox_pred_list = torch.split(bbox_pred, [4 * self.num_classes, 1], -1)
                    bbox_pred = bbox_pred_list[0] # (batch*num_samples, num_class*4)
                    iou_pred = bbox_pred_list[1] # (batch*num_sample, 1)
                    iou_pred = iou_pred.sigmoid()
                    iou_pred = iou_pred.expand(-1, scores.size(-1))
                    # scores = scores * iou_pred
                    # scores = scores.pow(alpha) * iou_pred.pow(1 - alpha)
                    scores = alpha * scores + (1 - alpha) * iou_pred
                    # print('test')

                else:
                    bbox_pred = bbox_pred.reshape(-1, cls_score.size(1), 5)
                    bbox_pred_list = torch.split(bbox_pred, [4, 1], -1)
                    bbox_pred = bbox_pred_list[0].reshape(-1, cls_score.size(-1)*4) # (batch*num_samples, num_class*4)
                    iou_pred = bbox_pred_list[1].squeeze() # (batch*num_samples, num_class)
                    iou_pred = iou_pred.sigmoid()
                    # only select the predicted iou correspondding to the largest scores.
                    # _ , ind = scores.max(dim=1)
                    # print('the size of ind is ', ind.size())
                    # iou_target = iou_pred[ind]
                    # print('the size of iou_pred[ind] is', iou_target.size())
                    # iou_target = iou_target.expand(-1, scores.size(-1))
                    # print('the size of iou_target is', iou_target.size())
                    # iou_selected = []
                    # for i in range(iou_pred.size(0)):
                    #     iou_selected.append(iou_pred[i, ind[i]])
                    # # print('the len of iou_selected is ', len(iou_selected))
                    # iou_final = torch.stack(iou_selected)
                    # # print('the size of iou_final0 is ', iou_final.size())
                    # iou_final = iou_final.view(-1, 1).expand(-1, cls_score.size(-1))
                    # # print('the size of iou_final1 is ', iou_final.size())
                    #
                    # scores *= iou_final

                    # scores = scores * iou_pred
                    scores = scores.pow(alpha) * iou_pred.pow(1 - alpha)
                    # scores = alpha * scores + (1 - alpha) * iou_pred
                    # print('test')


            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]
            # TODO: add clip here

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
