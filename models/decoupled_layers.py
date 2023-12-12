from typing import Dict, Tuple
import torch
from functools import partial
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from torch import Tensor
from torch.nn import ModuleList
from mmdet.models.layers.transformer.utils import (
    MLP,
    ConditionalAttention,
    coordinate_to_encoding,
)
from mmdet.models.layers.transformer.conditional_detr_layers import (
    ConditionalDetrTransformerDecoderLayer,
)
from mmdet.models.detectors.conditional_detr import ConditionalDETR
from torch import nn
from .query_distillation_layers import QueryDistllationDetrTransformerDecoder, multi_apply_v2
from .query_distillation_detr import QueryDsitillationDETR

from torch import nn
from mmdet.models.detectors.conditional_detr import ConditionalDETR
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers import (
    ConditionalDetrTransformerDecoder,
    DetrTransformerEncoder,
    SinePositionalEncoding,
)
from mmdet.models.layers.transformer import inverse_sigmoid

class DecoupledDecoderLayer(ConditionalDetrTransformerDecoderLayer):
    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, and
        normalization."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cls_cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.box_cross_attn = ConditionalAttention(**self.cross_attn_cfg)

        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims // 2)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
    
    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                ref_sine_embed: Tensor = None,
                is_first: bool = False):
        
        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks)
        # NOTE split
        cls_query, box_query = query.split(self.embed_dims // 2, dim=-1)
        cls_query = self.norms[0](cls_query)
        box_query = self.norms[0](box_query)

        query_list = multi_apply_v2(
            self.forward_cross_attn,
            [self.cls_cross_attn, self.box_cross_attn],
            [cls_query, box_query],
            query_pos.split(self.embed_dims // 2, dim=-1),
            ref_sine_embed.split(self.embed_dims // 2, dim=-1),
            key=key,
            key_pos=key_pos,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            is_first=is_first
        )
        query = torch.cat(query_list, dim=-1)
        return query

    def forward_cross_attn(self,
            query_func,
            query,
            query_pos,
            ref_sine_embed,
            key,
            key_pos,
            attn_mask,
            key_padding_mask,
            is_first):
        
        query = query_func(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            ref_sine_embed=ref_sine_embed,
            is_first=is_first
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        return query

class DecoupledQueryDistllationDetrTransformerDecoder(QueryDistllationDetrTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers and other layers."""

        self.layers = ModuleList(
            [
                DecoupledDecoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ]
        )
        self.embed_dims = self.layers[0].embed_dims

        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]
        # conditional detr affline
        self.query_scale = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 2)
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims, 2, 2)
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cls_cross_attn.qpos_proj = None
            self.layers[layer_id + 1].box_cross_attn.qpos_proj = None


    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        query_pos: Tensor = None,
        key_pos: Tensor = None,
        key_padding_mask: Tensor = None,
    ):
        reference_unsigmoid = self.ref_point_head(query_pos)
        reference = reference_unsigmoid.sigmoid()
        reference_xy = reference[..., :2]
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)
            # get sine embedding for the query reference
            ref_sine_embed = coordinate_to_encoding(coord_tensor=reference_xy)
            # apply transformation
            ref_sine_embed = ref_sine_embed.repeat(1, 1, 2) * pos_transformation
            query = layer(
                query,
                key=key,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                ref_sine_embed=ref_sine_embed,
                is_first=(layer_id == 0),
            )
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))

        ##########################
        # NOTE Cascaded Roll back #
        ##########################
        layer_ids = [i for i in range(1, len(self.layers))]
        (queries_list) = multi_apply_v2(
            self.forward_layer,
            layer_ids,
            query=query,
            reference_xy=reference_xy,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
        )
        if self.return_intermediate:
            intermediate += queries_list
        ##########################
        # NOTE Cascaded Roll back #
        ##########################

        if self.return_intermediate:
            return torch.stack(intermediate), reference

        query = self.post_norm(query)
        return query.unsqueeze(0), reference
    
    def forward_layer(
        self,
        i,
        query,
        reference_xy,
        key,
        query_pos,
        key_pos,
        key_padding_mask,
    ):
        """
        Note a wraper for map func
        """

        pos_transformation = self.query_scale(query)
        # get sine embedding for the query reference
        ref_sine_embed = coordinate_to_encoding(coord_tensor=reference_xy)
        # apply transformation
        ref_sine_embed = ref_sine_embed.repeat(1, 1, 2) * pos_transformation
        # print(query.size())
        out_query = self.layers[i](
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
            ref_sine_embed=ref_sine_embed,
            is_first=False,
        )  # [bs,num_queries, emb_dim]
        return out_query


@MODELS.register_module()
class DecoupledConditionalDETR(QueryDsitillationDETR):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DecoupledQueryDistllationDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.cls_query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        self.box_query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )
    
    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        query_pos = torch.cat(
            [self.cls_query_embedding.weight, self.box_query_embedding.weight], 
            dim=-1)
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        # NOTE cls query and position query
        query = torch.zeros_like(query_pos)
        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

from query_distillation_head import QueryDistallationDETRHead
from mmcv.cnn import Linear

@MODELS.register_module()
class DecoupledQueryDistallationDETRHead(QueryDistallationDETRHead):

    def __init__(self, *args, pred_layer=-1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc_iou = Linear(self.embed_dims, 1)
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
        # NOTE Fusion layer for iou 
        self.fusion_iou = MLP(self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)
         

        # NOTE the activations of reg_branch here is the same as
        # those in transformer, but they are actually different
        # in DAB-DETR (prelu in transformer and relu in reg_branch)
        
        self.fc_reg = Linear(self.embed_dims, 4)
        # self.norm_layer = build_norm_layer({'type': 'LN'}, self.embed_dims * 2)[1]
    
    def forward(
        self, hidden_states: Tensor, references: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        0.340

        Args:
            hidden_states (Tensor): _description_
            references (Tensor): _description_

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        
        cls_hs = hidden_states[..., :self.embed_dims]
        box_hs =  hidden_states[..., self.embed_dims:]

        references_unsigmoid = inverse_sigmoid(references)
        layers_bbox_preds = []
        for layer_id in range(hidden_states.shape[0]):
            tmp_reg_preds = self.fc_reg(
                self.activate(self.reg_ffn(box_hs[layer_id]))
            )
            tmp_reg_preds[..., :2] += references_unsigmoid
            outputs_coord = tmp_reg_preds.sigmoid()
            layers_bbox_preds.append(outputs_coord)
        layers_bbox_preds = torch.stack(layers_bbox_preds)

        layers_iou_scores = self.fc_iou(
            self.fusion_iou(hidden_states)
        )

        layers_cls_scores = self.fc_cls(
           cls_hs
        )
        return layers_cls_scores, layers_bbox_preds, layers_iou_scores

