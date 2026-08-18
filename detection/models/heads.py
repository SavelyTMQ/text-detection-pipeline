"""Головы детектора: классификация, регрессия, objectness."""
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Simple head: общая свёртка + классификация + регрессия+objectness.
    Регрессия и objectness совмещены (5 значений на якорь).
    """
    
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        # 4 для bbox + 1 для objectness
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 5, 1)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        cls_logits = self.cls_head(x)
        bbox_preds = self.reg_head(x)
        return cls_logits, bbox_preds


class DecoupledHead(nn.Module):
    """
    Decoupled head: три отдельные ветви (cls, reg, obj).
    Используется в многоуровневом детекторе.
    """
    
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.cls_head(x), self.reg_head(x), self.obj_head(x)
