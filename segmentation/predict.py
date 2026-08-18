"""Сохранение предсказаний сегментации как картинок (вход для детекции)."""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def save_predictions(model, dataset, dataloader, save_dir, device, threshold=0.5):
    """
    Сохраняет предсказания модели как PNG-маски с именами исходных файлов.
    
    Args:
        model: обученная модель сегментации
        dataset: датасет (для получения оригинальных имён файлов)
        dataloader: DataLoader с shuffle=False!
        save_dir: папка для сохранения
        device: cuda/cpu
        threshold: порог бинаризации sigmoid(logits)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    img_idx = 0
    
    with torch.no_grad():
        for X_batch, _ in tqdm(dataloader, desc=f"Saving to {save_dir.name}"):
            X_batch = X_batch.to(device)
            
            Y_pred = model(X_batch)
            pred_mask = (torch.sigmoid(Y_pred) > threshold).int()
            
            for i in range(X_batch.shape[0]):
                original_path = dataset.images[img_idx]
                filename = Path(original_path).stem + '.png'
                save_path = save_dir / filename
                
                mask_np = pred_mask[i].squeeze(0).cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
                
                Image.fromarray(mask_np, mode='L').save(save_path)
                img_idx += 1
    
    print(f"✓ Сохранено {img_idx} предсказаний в {save_dir}")
