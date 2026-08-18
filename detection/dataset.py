"""Датасет для детекции с YOLO-разметкой."""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class TextYoloDataset(Dataset):
    """
    Датасет для детекции текста с YOLO-разметкой.
    
    Формат метки: class_id x_center y_center width height (нормализовано в [0,1]).
    Возвращает bboxes в формате xywh в пикселях.
    """
    
    def __init__(self, images_dir, labels_dir, input_size=(512, 512),
                 image_ext=".png", normalize=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = input_size
        self.image_ext = image_ext
        self.normalize = normalize
        
        self.image_paths = sorted(list(self.images_dir.glob(f"*{image_ext}")))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Не найдено изображений в: {self.images_dir}")
        
        # ImageNet статистика для нормализации
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def _read_yolo_label(self, label_path, width, height):
        """Читает YOLO-метку и возвращает bboxes (xywh в пикселях) и labels."""
        boxes = []
        labels = []
        
        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long)
            )
        
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                # Для single-class задачи (текст) всегда class_id = 0
                class_id = 0
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
                
                # YOLO normalized -> pixel coords
                x_center_px = x_center * width
                y_center_px = y_center * height
                box_w_px = box_w * width
                box_h_px = box_h * height
                
                x_min = x_center_px - box_w_px / 2
                y_min = y_center_px - box_h_px / 2
                
                # Защита от плохих боксов
                x_min = max(0.0, min(x_min, width - 1))
                y_min = max(0.0, min(y_min, height - 1))
                box_w_px = max(1.0, min(box_w_px, width - x_min))
                box_h_px = max(1.0, min(box_h_px, height - y_min))
                
                boxes.append([x_min, y_min, box_w_px, box_h_px])
                labels.append(class_id)
        
        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long)
            )
        
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        image = Image.open(image_path).convert("RGB")
        target_h, target_w = self.input_size
        image = image.resize((target_w, target_h))
        image = TF.to_tensor(image)
        
        if self.normalize:
            image = (image - self.mean) / self.std
        
        boxes, labels = self._read_yolo_label(label_path, target_w, target_h)
        
        target = {
            "boxes": boxes,      # [N, 4], формат xywh
            "labels": labels     # [N]
        }
        
        return image, target


def detection_collate_fn(batch):
    """Collate функция для детекции (разное число объектов на картинках)."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = list(targets)
    return images, targets
