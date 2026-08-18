"""Визуализация предсказаний детектора."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from detection.postprocess import filter_predictions


def denormalize_image(img_tensor):
    """Возвращает изображение из нормализованного тензора."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1)


@torch.no_grad()
def visualize_prediction(model, dataset, device, idx=0, score_threshold=0.2, save_path=None):
    """Показывает GT (зелёный) и предсказание (красный) на одной картинке."""
    model.eval()
    image, target = dataset[idx]
    
    outputs = model(image.unsqueeze(0).to(device))
    preds = filter_predictions(
        outputs, score_threshold=score_threshold,
        nms_threshold=0.5, max_boxes_per_cls=50, return_type="torch"
    )[0]
    
    img_show = denormalize_image(image).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_show)
    
    # GT — зелёные
    for box in target["boxes"]:
        x, y, w, h = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
        ))
    
    # Predicted — красные
    for box, score in zip(preds["boxes"], preds["scores"]):
        x, y, w, h = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        ))
        ax.text(x, y, f"{score:.2f}", color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.5))
    
    ax.set_title("Green = GT, Red = Prediction")
    ax.axis("off")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
