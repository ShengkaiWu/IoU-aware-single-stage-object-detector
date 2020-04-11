import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # if len(outs) == 3:
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # else:
        #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        #     losses = self.bbox_head.loss_with_iou(
        #         *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses

    def simple_test(self,
                    img,
                    img_meta,
                    gt_bboxes,
                    gt_labels,
                    rescale=False):
        """
        only one image is test.
        :param img: tesnor
        :param img_meta: DC(mmcv.parallel.DataContainer), the imformation about one image
            dict{
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip}
        :param gt_bboxes: DC(mmcv.parallel.DataContainer), tensor of shape (num_gts, 4),represent
             the coordinates of top-left and bottom-right corner for each gt box,  (x_tl, y_tl, x_br, y_br)
        :param gt_labels:  DC(mmcv.parallel.DataContainer), tensor of shape
        :param rescale:
        :return:
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x) # cls_score, bbox_pred
        # bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_inputs = outs + (gt_bboxes, gt_labels, img_meta, self.test_cfg, rescale)

        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
