"""Основной скрипт для запуска обучения моделей сегментации."""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from segmentation.dataset import SelfSegmTextDataset, get_data_paths
from segmentation.models.unet import UNet
from segmentation.models.segnet import SegNet_Tiny
from segmentation.train import train
from segmentation.visualize import plot_history


def get_model(model_name, device):
    """Фабрика моделей."""
    if model_name == "unet":
        return UNet(in_channels=3, out_channels=1).to(device)
    elif model_name == "segnet":
        return SegNet_Tiny().to(device)
    elif model_name == "deeplabv3":
        import torchvision
        deeplab = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None, weights_backbone=None, num_classes=1
        )
        
        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x)["out"]
        
        return Wrapper(deeplab).to(device)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "segnet", "deeplabv3"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_SEGM)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="Вес для положительного класса в BCE")
    args = parser.parse_args()
    
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Обучаю модель: {args.model}")
    
    # Данные
    train_images, train_masks = get_data_paths(
        config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR
    )
    test_images, test_masks = get_data_paths(
        config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR
    )
    
    train_dataset = SelfSegmTextDataset(train_images, train_masks, config.IMAGE_SIZE)
    test_dataset = SelfSegmTextDataset(test_images, test_masks, config.IMAGE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Модель
    model = get_model(args.model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Обучение
    ckpt_dir = config.SEGM_CHECKPOINTS / args.model
    model, history = train(
        model, optimizer, criterion, args.epochs,
        train_loader, test_loader, device, ckpt_dir
    )
    
    # Финальные метрики
    print(f"\n=== Результаты {args.model} ===")
    print(f"Best IoU: {max(history['val_iou']):.4f}")
    print(f"Best epoch: {history['val_iou'].index(max(history['val_iou'])) + 1}")
    
    # Графики
    plot_history(history, save_path=f"results/{args.model}_history.png")


if __name__ == "__main__":
    main()
