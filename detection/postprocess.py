"""Постобработка предсказаний детектора: score threshold + NMS."""
import torch
from torchvision.ops import nms


def filter_predictions(
    predictions,
    score_threshold=0.1,
    nms_threshold=0.5,
    max_boxes_per_cls=50,
    return_type="list"
):
    """
    Фильтрует и применяет NMS к предсказаниям.
    
    Args:
        predictions: tuple (bboxes, confidences, cls_probs)
            bboxes: [B, N, 4] в формате xywh
            confidences: [B, N]
            cls_probs: [B, N, C]
        score_threshold: минимальный скор для сохранения
        nms_threshold: IoU threshold для NMS
        max_boxes_per_cls: максимум боксов на класс
        return_type: "list" или "torch"
    
    Returns:
        Список словарей {boxes, labels, scores} для каждой картинки в батче
    """
    bboxes, confidences, cls_probs = predictions
    all_final_scores = confidences[:, :, None] * cls_probs
    num_classes = cls_probs.shape[-1]
    
    final_predictions = []
    
    for boxes, final_scores in zip(bboxes, all_final_scores):
        preds = {"boxes": [], "labels": [], "scores": []}
        
        for cls in range(num_classes):
            cls_scores = final_scores[:, cls]
            keep_ixs = cls_scores > score_threshold
            
            if keep_ixs.sum() == 0:
                continue
            
            cls_boxes = boxes[keep_ixs]
            cls_scores = cls_scores[keep_ixs]
            
            # Топ-K до NMS
            if len(cls_boxes) > max_boxes_per_cls:
                pos = torch.argsort(cls_scores, descending=True)
                cls_boxes = cls_boxes[pos[:max_boxes_per_cls]]
                cls_scores = cls_scores[pos[:max_boxes_per_cls]]
            
            # xywh -> xyxy для NMS
            boxes_xyxy = cls_boxes.clone()
            boxes_xyxy[:, 2:] = boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]
            
            pred_ixs = nms(boxes_xyxy, cls_scores, nms_threshold)
            
            for ix in pred_ixs:
                preds["boxes"].append(cls_boxes[ix].detach().cpu().tolist())
                preds["labels"].append(cls)
                preds["scores"].append(cls_scores[ix].detach().cpu().item())
        
        if return_type == "torch":
            if len(preds["boxes"]) == 0:
                preds["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                preds["labels"] = torch.zeros((0,), dtype=torch.long)
                preds["scores"] = torch.zeros((0,), dtype=torch.float32)
            else:
                preds["boxes"] = torch.tensor(preds["boxes"], dtype=torch.float32)
                preds["labels"] = torch.tensor(preds["labels"], dtype=torch.long)
                preds["scores"] = torch.tensor(preds["scores"], dtype=torch.float32)
        
        final_predictions.append(preds)
    
    return final_predictions
