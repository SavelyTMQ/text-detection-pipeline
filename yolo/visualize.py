"""Визуализация результатов YOLOv8: графики + GT vs Predicted."""
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(run_dir, save_path=None):
    """Строит графики loss и mAP по эпохам."""
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = df.columns.str.strip()
    
    epochs = df["epoch"].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, df["train/box_loss"], label="Train box loss", color="blue")
    axes[0].plot(epochs, df["val/box_loss"], label="Val box loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Box Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="green")
    axes[1].plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].set_title("Validation mAP")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def draw_yolo_boxes(image_path, label_path, class_names, color=(0, 255, 0)):
    """Рисует YOLO-боксы на копии изображения."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    
    label_path = Path(label_path)
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = class_names[cls] if cls < len(class_names) else str(cls)
                cv2.putText(img, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def plot_gt_vs_predicted(val_img_dir, val_lbl_dir, pred_lbl_dir,
                          class_names, n_show=4, save_path=None):
    """Показывает GT и предсказания бок о бок."""
    val_img_dir = Path(val_img_dir)
    val_lbl_dir = Path(val_lbl_dir)
    pred_lbl_dir = Path(pred_lbl_dir)
    
    val_images = sorted(val_img_dir.glob("*.png"))
    n_show = min(n_show, len(val_images))
    sample = random.sample(val_images, n_show)
    
    fig, axes = plt.subplots(n_show, 2, figsize=(14, 5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    
    for i, img_path in enumerate(sample):
        stem = img_path.stem
        
        gt_img = draw_yolo_boxes(img_path, val_lbl_dir / f"{stem}.txt",
                                  class_names, color=(0, 255, 0))
        pred_img = draw_yolo_boxes(img_path, pred_lbl_dir / f"{stem}.txt",
                                    class_names, color=(0, 0, 255))
        
        axes[i, 0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Ground Truth — {stem}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Predicted — {stem}")
        axes[i, 1].axis("off")
    
    plt.suptitle("Ground Truth (зелёный) vs Predicted (красный)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
