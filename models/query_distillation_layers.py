# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union
from mmdet.utils import ConfigType, OptConfigType
from mmengine import ConfigDict
import torch
from functools import partial
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from torch import Tensor
from torch.nn import ModuleList

from mmdet.models.layers.transformer.detr_layers import (
    DetrTransformerDecoder,
    DetrTransformerDecoderLayer,
)
from mmdet.models.layers.transformer.utils import (
    MLP,
    ConditionalAttention,
    coordinate_to_encoding,
)
from mmdet.models.layers.transformer.conditional_detr_layers import (
    ConditionalDetrTransformerDecoderLayer,
)
from mmdet.models.layers.transformer.conditional_detr_layers import (
    ConditionalDetrTransformerDecoder,
)


def multi_apply_v2(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return list(map_results)


class QueryDistllationDetrTransformerDecoder(DetrTransformerDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers and other layers."""

        self.layers = ModuleList(
            [
                ConditionalDetrTransformerDecoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ]
        )
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]
        # conditional detr affline
        self.query_scale = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 2)
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims, 2, 2)
        # we have substitute 'qpos_proj' with 'qpos_sine_proj' except for
        # the first decoder layer), so 'qpos_proj' should be deleted
        # in other layers.
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cross_attn.qpos_proj = None

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
            ref_sine_embed = ref_sine_embed * pos_transformation
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
        ref_sine_embed = ref_sine_embed * pos_transformation
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
