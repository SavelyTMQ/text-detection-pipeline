"""Одноуровневый якорный детектор с Simplified FPN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator

from detection.models.backbone import Backbone
from detection.models.fpn import SimplifiedFPN
from detection.models.heads import DetectionHead


class SimpleDetector(nn.Module):
    """
    Baseline детектор: EfficientNet + Simplified FPN + одна голова.
    
    Использует один уровень feature map — простая архитектура для baseline.
    """
    
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        backbone_out_indices=(-2,),
        neck_channels=256,
        num_classes=1,
        anchor_sizes=(16, 32, 64, 128, 256),
        anchor_ratios=(0.25, 0.5, 1.0, 2.0, 4.0),
        input_size=(640, 640),
        pretrained=True,
        freeze_backbone=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone
        self.backbone = Backbone(
            model_name=backbone_name,
            out_indices=backbone_out_indices,
            pretrained=pretrained,
            freeze=freeze_backbone
        )
        in_channels = self.backbone.backbone.feature_info.channels()[0]
        
        # Neck (простая FPN)
        self.neck = SimplifiedFPN(in_channels, neck_channels)
        
        # Head
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        self.head = DetectionHead(neck_channels, num_anchors, num_classes)
        
        # Anchors
        self._init_anchors(anchor_sizes, anchor_ratios, input_size)
    
    def _init_anchors(self, anchor_sizes, anchor_ratios, input_size):
        anchor_gen = AnchorGenerator(
            sizes=(anchor_sizes,),
            aspect_ratios=(anchor_ratios,)
        )
        
        reduction = self.backbone.backbone.feature_info.reduction()[0]
        grid_h = input_size[0] // reduction
        grid_w = input_size[1] // reduction
        
        anchors = anchor_gen.grid_anchors(
            [[grid_h, grid_w]], [[reduction, reduction]]
        )
        anchors = torch.stack(anchors, dim=0)
        
        centers = (anchors[:, :, :2] + anchors[:, :, 2:]) / 2
        sizes = anchors[:, :, 2:] - anchors[:, :, :2]
        
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchor_centers", centers)
        self.register_buffer("anchor_sizes", sizes)
    
    def forward(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)
        cls_logits, bbox_preds = self.head(neck_out)
        
        N = x.shape[0]
        
        # Reshape: [B, A*C, H, W] -> [B, A*H*W, C]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(N, -1, self.head.num_classes)
        
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous()
        bbox_preds = bbox_preds.view(N, -1, 5)
        
        bbox_offsets = bbox_preds[:, :, :4]
        confidence_logits = bbox_preds[:, :, 4]
        
        if self.training:
            return bbox_offsets, confidence_logits, cls_logits
        
        bboxes = self.decode_bboxes(bbox_offsets)
        confidence = torch.sigmoid(confidence_logits)
        cls_probs = torch.sigmoid(cls_logits)
        
        return bboxes, confidence, cls_probs
    
    def decode_bboxes(self, bbox_offsets):
        """Декодирует offsets в bboxes формата xywh."""
        tx, ty, tw, th = bbox_offsets.unbind(dim=-1)
        
        center_x = self.anchor_centers[..., 0] + tx * self.anchor_sizes[..., 0]
        center_y = self.anchor_centers[..., 1] + ty * self.anchor_sizes[..., 1]
        w = torch.exp(tw).clamp(max=1e4) * self.anchor_sizes[..., 0]
        h = torch.exp(th).clamp(max=1e4) * self.anchor_sizes[..., 1]
        
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        
        return torch.stack([x_min, y_min, w, h], dim=-1)
