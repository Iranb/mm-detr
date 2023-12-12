# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, OptMultiConfig
import torch
import math
import torch.nn as nn
from torch import Tensor
from mmdet.models.layers.transformer import inverse_sigmoid

from mmdet.registry import MODELS

from mmdet.models.dense_heads.conditional_detr_head import ConditionalDETRHead
from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList
from mmdet.structures.bbox import (
    bbox_cxcywh_to_xyxy,
    bbox_overlaps,
    bbox_xyxy_to_cxcywh,
)
from mmdet.utils import (
    ConfigType,
    InstanceList,
    OptInstanceList,
    OptMultiConfig,
    reduce_mean,
)
from mmdet.models.losses import QualityFocalLoss
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob
from mmdet.models.losses.utils import weighted_loss
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN


@weighted_loss
def l2_loss(input, target):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond**2 / pos_inds.shape[0]
    else:
        loss = input * 0.0
    return loss.sum()


@MODELS.register_module()
class QueryDistallationDETRHead(ConditionalDETRHead):
    def __init__(self, *args, pred_layer=-1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc_iou = nn.Linear(self.embed_dims, 1)
        self.pred_layer = pred_layer

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.activate = nn.ReLU()
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            dict(type="ReLU", inplace=True),
            dropout=0.0,
            add_residual=False,
        )
        # NOTE the activations of reg_branch here is the same as
        # those in transformer, but they are actually different
        # in DAB-DETR (prelu in transformer and relu in reg_branch)
        self.fc_reg = Linear(self.embed_dims, 4)

    def init_weights(self):
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.fc_iou.bias, bias_init)

    def forward(
        self, hidden_states: Tensor, references: Tensor
    ) -> Tuple[Tensor, Tensor]:
        references_unsigmoid = inverse_sigmoid(references)
        layers_bbox_preds = []
        for layer_id in range(hidden_states.shape[0]):
            tmp_reg_preds = self.fc_reg(
                self.activate(self.reg_ffn(hidden_states[layer_id]))
            )
            tmp_reg_preds[..., :2] += references_unsigmoid
            outputs_coord = tmp_reg_preds.sigmoid()
            layers_bbox_preds.append(outputs_coord)
        layers_bbox_preds = torch.stack(layers_bbox_preds)
        layers_iou_scores = self.fc_iou(hidden_states)
        layers_cls_scores = self.fc_cls(hidden_states)
        return layers_cls_scores, layers_bbox_preds, layers_iou_scores

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_iou_scores: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        assert batch_gt_instances_ignore is None, (
            f"{self.__class__.__name__} only supports "
            "for batch_gt_instances_ignore setting to None."
        )

        losses_cls, losses_bbox, losses_iou, losses_giou = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_iou_scores,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_giou_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_giou[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_giou"] = loss_giou_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        iou_scores: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores), label_weights, avg_factor=cls_avg_factor
            )
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor
            )

        ##########################
        # NOTE Compute iou loss
        ##########################
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        target_iou = bbox_overlaps(
            pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
        )

        pos_iou_pred = iou_scores.view(-1)[pos_inds].sigmoid()

        loss_fc_iou = l2_loss(
            pos_iou_pred,
            target_iou,
            torch.ones_like(pos_iou_pred),
            avg_factor=cls_avg_factor,
        )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            (
                img_h,
                img_w,
            ) = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
        return loss_cls, loss_bbox, loss_iou, loss_fc_iou

    def predict_by_feat(
        self,
        layer_cls_scores: Tensor,
        layer_bbox_preds: Tensor,
        layer_iou_scores: Tensor,
        batch_img_metas: List[dict],
        rescale: bool = True,
    ) -> InstanceList:
        cls_scores = layer_cls_scores[self.pred_layer]
        bbox_preds = layer_bbox_preds[self.pred_layer]
        iou_scores = layer_iou_scores[self.pred_layer]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            iou_score = iou_scores[img_id]
            results = self._predict_by_feat_single(
                cls_score, bbox_pred, iou_score, img_meta, rescale
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        iou_score: Tensor,
        img_meta: dict,
        rescale: bool = True,
    ) -> InstanceData:
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get("max_per_img", len(cls_score))
        img_shape = img_meta["img_shape"]
        # exclude background

        cls_score = cls_score.sigmoid()
        ious = iou_score.sigmoid().repeat(1, 1, cls_score.shape[-1])
        cls_score = cls_score * ious
        scores, indexes = cls_score.view(-1).topk(max_per_img)
        det_labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes
        bbox_pred = bbox_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get("scale_factor") is not None
            det_bboxes /= det_bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

    def predict(
        self,
        hidden_states: Tensor,
        references: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        last_layer_hidden_state = hidden_states
        outs = self(last_layer_hidden_state, references)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale
        )

        return predictions
