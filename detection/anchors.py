"""Anchor assignment и вычисление target offsets."""
import torch
from torchvision.ops import box_iou


def get_target_offset(anchor_box, gt_box):
    """
    Вычисляет target offsets для регрессии bbox.
    
    Args:
        anchor_box: [x_min, y_min, x_max, y_max]
        gt_box: [x_min, y_min, x_max, y_max]
    
    Returns:
        Тензор [tx, ty, tw, th]
    """
    gt_center = (gt_box[:2] + gt_box[2:]) / 2
    gt_size = gt_box[2:] - gt_box[:2]
    
    anchor_center = (anchor_box[:2] + anchor_box[2:]) / 2
    anchor_size = anchor_box[2:] - anchor_box[:2]
    
    tx = (gt_center[0] - anchor_center[0]) / anchor_size[0]
    ty = (gt_center[1] - anchor_center[1]) / anchor_size[1]
    tw = torch.log(gt_size[0] / anchor_size[0])
    th = torch.log(gt_size[1] / anchor_size[1])
    
    return torch.stack([tx, ty, tw, th])


def assign_target(anchors, gt_boxes, gt_labels, num_classes,
                  pos_th=0.4, neg_th=0.2):
    """
    Сопоставляет якоря с GT-боксами.
    
    Логика:
    - IoU >= pos_th → положительный якорь
    - neg_th <= IoU < pos_th → игнорировать
    - IoU < neg_th → отрицательный якорь
    - Каждому GT гарантированно назначается хотя бы один якорь
    
    Args:
        anchors: [num_anchors, 4] в формате xyxy
        gt_boxes: [num_gt, 4] в формате xywh (пиксели)
        gt_labels: [num_gt] метки классов
        num_classes: количество классов
        pos_th, neg_th: пороги IoU
    
    Returns:
        target_offsets: [num_anchors, 4]
        target_objectness: [num_anchors] (1=pos, 0=neg, -1=ignore)
        target_cls: [num_anchors, num_classes] (one-hot)
    """
    num_anchors = anchors.shape[0]
    device = anchors.device
    
    target_objectness = torch.zeros(num_anchors, device=device)
    target_offsets = torch.zeros((num_anchors, 4), device=device)
    target_cls = torch.zeros((num_anchors, num_classes), device=device)
    
    if gt_boxes.numel() == 0:
        return target_offsets, target_objectness, target_cls
    
    # xywh -> xyxy
    gt_xyxy = gt_boxes.clone()
    gt_xyxy[:, 2:] = gt_xyxy[:, :2] + gt_xyxy[:, 2:]
    
    # IoU каждый anchor с каждым GT
    ious = box_iou(anchors, gt_xyxy)  # [num_anchors, num_gt]
    best_iou, best_gt_idx = ious.max(dim=1)
    
    # Игнорируемые якоря
    ignore_mask = (best_iou >= neg_th) & (best_iou < pos_th)
    target_objectness[ignore_mask] = -1
    
    # Положительные якоря
    pos_mask = best_iou >= pos_th
    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    
    for pos in pos_indices:
        gt_idx = best_gt_idx[pos]
        gt_box = gt_xyxy[gt_idx]
        anchor_box = anchors[pos]
        
        target_offsets[pos] = get_target_offset(anchor_box, gt_box)
        target_objectness[pos] = 1
        
        label = gt_labels[gt_idx].long()
        target_cls[pos, label] = 1
    
    # Гарантируем, что каждый GT получит хотя бы один якорь
    for gt_idx in range(gt_xyxy.shape[0]):
        already_assigned = (
            (target_objectness == 1) & (best_gt_idx == gt_idx)
        ).any()
        
        if not already_assigned:
            best_anchor_idx = torch.argmax(ious[:, gt_idx])
            target_offsets[best_anchor_idx] = get_target_offset(
                anchors[best_anchor_idx], gt_xyxy[gt_idx]
            )
            target_objectness[best_anchor_idx] = 1
            
            label = gt_labels[gt_idx].long()
            target_cls[best_anchor_idx, label] = 1
    
    return target_offsets, target_objectness, target_cls
