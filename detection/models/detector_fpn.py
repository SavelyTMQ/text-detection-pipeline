"""Многоуровневый якорный детектор с полной FPN и decoupled heads."""
import math
import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator

from detection.models.backbone import Backbone
from detection.models.fpn import FPN
from detection.models.heads import DecoupledHead


class FPNDetector(nn.Module):
    """
    Многоуровневый детектор с FPN (в стиле RetinaNet).
    
    Использует 3 уровня feature maps + decoupled heads (cls/reg/obj).
    """
    
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        backbone_out_indices=(-3, -2, -1),
        neck_channels=256,
        num_classes=1,
        anchor_sizes=(8, 16, 32, 64),
        anchor_ratios=(0.1, 0.2, 0.5, 1.0, 2.0),
        input_size=(512, 512),
        pretrained=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone
        self.backbone = Backbone(
            model_name=backbone_name,
            out_indices=backbone_out_indices,
            pretrained=pretrained,
            freeze=False
        )
        in_channels_list = self.backbone.backbone.feature_info.channels()
        
        # Neck
        self.neck = FPN(in_channels_list, out_channels=neck_channels)
        
        # Head (одна голова на все уровни)
        num_anchors_per_cell = len(anchor_sizes) * len(anchor_ratios)
        self.head = DecoupledHead(
            in_channels=neck_channels,
            num_anchors=num_anchors_per_cell,
            num_classes=num_classes
        )
        
        # Init biases для objectness (RetinaNet-style prior)
        self._init_head_bias()
        
        # Anchors для всех уровней
        self._init_anchors(anchor_sizes, anchor_ratios, input_size, in_channels_list)
    
    def _init_head_bias(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.head.obj_head.bias, bias_value)
        nn.init.normal_(self.head.obj_head.weight, std=0.01)
        nn.init.normal_(self.head.reg_head.weight, std=0.01)
        nn.init.constant_(self.head.reg_head.bias, 0.0)
        nn.init.normal_(self.head.cls_head.weight, std=0.01)
        nn.init.constant_(self.head.cls_head.bias, 0.0)
    
    def _init_anchors(self, anchor_sizes, anchor_ratios, input_size, in_channels_list):
        num_levels = len(in_channels_list)
        reductions = self.backbone.backbone.feature_info.reduction()
        
        multi_level_sizes = tuple([anchor_sizes] * num_levels)
        multi_level_ratios = tuple([anchor_ratios] * num_levels)
        
        anchor_gen = AnchorGenerator(
            sizes=multi_level_sizes,
            aspect_ratios=multi_level_ratios
        )
        
        grid_sizes = [[input_size[0] // r, input_size[1] // r] for r in reductions]
        strides = [[r, r] for r in reductions]
        
        anchors = anchor_gen.grid_anchors(grid_sizes, strides=strides)
        anchors = torch.cat(anchors, dim=0).unsqueeze(0)
        
        centers = (anchors[..., :2] + anchors[..., 2:]) / 2
        sizes = anchors[..., 2:] - anchors[..., :2]
        
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchor_centers", centers)
        self.register_buffer("anchor_sizes", sizes)
    
    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features)
        
        N = x.shape[0]
        all_cls, all_bbox, all_obj = [], [], []
        
        for level_feat in neck_features:
            cls_l, bbox_p, obj_l = self.head(level_feat)
            
            cls_l = cls_l.permute(0, 2, 3, 1).contiguous().view(N, -1, self.num_classes)
            bbox_p = bbox_p.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
            obj_l = obj_l.permute(0, 2, 3, 1).contiguous().view(N, -1)
            
            all_cls.append(cls_l)
            all_bbox.append(bbox_p)
            all_obj.append(obj_l)
        
        cls_logits = torch.cat(all_cls, dim=1)
        bbox_offsets = torch.cat(all_bbox, dim=1)
        confidence_logits = torch.cat(all_obj, dim=1)
        
        if self.training:
            return bbox_offsets, confidence_logits, cls_logits
        
        bboxes = self.decode_bboxes(bbox_offsets)
        confidence = torch.sigmoid(confidence_logits)
        cls_probs = torch.ones_like(cls_logits)  # single-class
        
        return bboxes, confidence, cls_probs
    
    def decode_bboxes(self, bbox_offsets):
        bbox_offsets = torch.clamp(bbox_offsets, -10, 10)
        
        tx = bbox_offsets[..., 0]
        ty = bbox_offsets[..., 1]
        tw = bbox_offsets[..., 2]
        th = bbox_offsets[..., 3]
        
        center_x = self.anchor_centers[..., 0] + tx * self.anchor_sizes[..., 0]
        center_y = self.anchor_centers[..., 1] + ty * self.anchor_sizes[..., 1]
        w = torch.exp(tw) * self.anchor_sizes[..., 0]
        h = torch.exp(th) * self.anchor_sizes[..., 1]
        
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        
        return torch.stack([x_min, y_min, w, h], dim=-1)
