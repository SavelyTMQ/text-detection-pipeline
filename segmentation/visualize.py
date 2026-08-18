"""Визуализация результатов сегментации."""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


def plot_history(history, save_path=None):
    """График loss и IoU по эпохам."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], label='Train loss', color='blue')
    ax1.plot(epochs, history['val_loss'], label='Val loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['val_iou'], label='Val IoU', color='green', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сохранён: {save_path}")
    
    plt.show()


def verify_predictions(dataset, save_dir, num_examples=4, save_path=None):
    """Показывает оригинал, GT и предсказание рядом."""
    save_dir = Path(save_dir)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    
    for i in range(num_examples):
        orig_img = Image.open(dataset.images[i])
        gt_mask = Image.open(dataset.segmentation_masks[i])
        
        img_name = Path(dataset.images[i]).stem + '.png'
        pred_path = save_dir / img_name
        pred_mask = Image.open(pred_path)
        
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original: {Path(dataset.images[i]).name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
