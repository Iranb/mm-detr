# from .query_distillation_layers import QueryDistllationDetrTransformerDecoder
# from .query_distillation_head import QueryDistallationDETRHead
# from .query_distillation_detr import QueryDsitillationDETR

from .decoupled_layers import DecoupledConditionalDETR, DecoupledQueryDistallationDETRHead

from mmdet.utils import register_all_modules

# __all__ = ["QueryDistllationDetrTransformerDecoder", "QueryDistallationDETRHead", "QueryDsitillationDETR"]

__all__ = ["DecoupledConditionalDETR", "DecoupledQueryDistallationDETRHead", ]


register_all_modules()
