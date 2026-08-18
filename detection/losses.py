"""Loss-функция для детекции с hard negative mining."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComputeLoss:
    """
    Loss для детекции с тремя компонентами:
    - bbox regression (Smooth L1)
    - objectness (BCE + hard negative mining)
    - classification (BCE)
    """
    
    def __init__(
        self,
        weight_bbox=5.0,
        weight_obj=1.0,
        weight_cls=0.0,           # 0 для single-class
        obj_pos_weight=10.0,
        neg_pos_ratio=10,
        min_neg=100
    ):
        self.bbox_loss = nn.SmoothL1Loss()
        self.weight_bbox = weight_bbox
        self.weight_obj = weight_obj
        self.weight_cls = weight_cls
        self.obj_pos_weight = obj_pos_weight
        self.neg_pos_ratio = neg_pos_ratio
        self.min_neg = min_neg
    
    def __call__(self, predicts, targets):
        pred_offsets, pred_obj_logits, pred_cls_logits = predicts
        target_offsets, target_obj, target_cls = targets
        
        pos_mask = target_obj == 1
        neg_mask = target_obj == 0
        
        # ─── Objectness loss с hard negative mining ───
        obj_targets = target_obj.clamp(0, 1).float()
        
        raw_obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj_logits, obj_targets, reduction="none"
        )
        
        pos_loss = raw_obj_loss[pos_mask]
        neg_loss = raw_obj_loss[neg_mask]
        
        num_pos = int(pos_mask.sum().item())
        
        # Оставляем только самые "трудные" негативы
        if neg_loss.numel() > 0:
            num_neg_keep = min(
                neg_loss.numel(),
                max(self.min_neg, self.neg_pos_ratio * max(num_pos, 1))
            )
            neg_loss = torch.topk(neg_loss, k=num_neg_keep, largest=True).values
        else:
            num_neg_keep = 0
        
        denom = max(pos_loss.numel() + num_neg_keep, 1)
        loss_obj = (pos_loss.sum() + neg_loss.sum()) / denom
        
        # ─── Bbox regression loss ───
        if pos_mask.sum() > 0:
            loss_bbox = self.bbox_loss(
                pred_offsets[pos_mask], target_offsets[pos_mask]
            )
        else:
            loss_bbox = torch.tensor(0.0, device=pred_offsets.device)
        
        # ─── Classification loss ───
        if self.weight_cls > 0 and pos_mask.sum() > 0:
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_cls_logits[pos_mask], target_cls[pos_mask]
            )
        else:
            loss_cls = torch.tensor(0.0, device=pred_offsets.device)
        
        total_loss = (
            self.weight_bbox * loss_bbox
            + self.weight_obj * loss_obj
            + self.weight_cls * loss_cls
        )
        
        return total_loss
