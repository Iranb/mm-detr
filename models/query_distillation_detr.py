from torch import nn
from mmdet.models.detectors.conditional_detr import ConditionalDETR
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers import (
    ConditionalDetrTransformerDecoder,
    DetrTransformerEncoder,
    SinePositionalEncoding,
)

from .query_distillation_layers import QueryDistllationDetrTransformerDecoder


@MODELS.register_module()
class QueryDsitillationDETR(ConditionalDETR):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = QueryDistllationDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )
