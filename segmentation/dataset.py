"""Датасет для семантической сегментации Total-Text."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SelfSegmTextDataset(Dataset):
    """Датасет для сегментации текста. Возвращает пары (изображение, маска)."""
    
    def __init__(self, images_paths, masks_paths, image_size=(256, 256)):
        self.images = images_paths
        self.segmentation_masks = masks_paths
        
        self.images_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.masks_transforms = transforms.Compose([
            transforms.Resize(
                image_size, 
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.segmentation_masks[index])
        
        img = self.images_transforms(img)
        mask = self.masks_transforms(mask)
        
        return img, mask
    
    def __len__(self):
        return len(self.images)


def get_data_paths(images_dir, masks_dir):
    """Возвращает отсортированные пути к изображениям и маскам."""
    images = sorted(Path(images_dir).iterdir())
    masks = sorted(Path(masks_dir).iterdir())
    assert len(images) == len(masks), \
        f"Число изображений и масок не совпадает: {len(images)} vs {len(masks)}"
    return images, masks
