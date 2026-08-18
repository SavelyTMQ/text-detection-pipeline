"""Запуск обучения кастомных детекторов.

Использование:
    python -m detection.run --model simple --epochs 30
    python -m detection.run --model fpn --epochs 50
"""
import argparse
import torch
from torch.utils.data import DataLoader

import config
from detection.dataset import TextYoloDataset, detection_collate_fn
from detection.models import SimpleDetector, FPNDetector
from detection.losses import ComputeLoss
from detection.anchors import assign_target
from detection.runner import Runner
from detection.visualize import visualize_prediction


def get_model(model_name, device):
    if model_name == "simple":
        return SimpleDetector(
            backbone_name="efficientnet_b0",
            backbone_out_indices=(-2,),
            neck_channels=256,
            num_classes=1,
            anchor_sizes=(16, 32, 64, 128, 256, 512),
            anchor_ratios=(0.25, 0.5, 1.0, 2.0, 4.0),
            input_size=config.DETECTION_IMAGE_SIZE,
            pretrained=True,
            freeze_backbone=False
        ).to(device)
    elif model_name == "fpn":
        return FPNDetector(
            backbone_name="efficientnet_b0",
            backbone_out_indices=(-3, -2, -1),
            neck_channels=256,
            num_classes=1,
            anchor_sizes=(8, 16, 32, 64),
            anchor_ratios=(0.1, 0.2, 0.5, 1.0, 2.0),
            input_size=config.DETECTION_IMAGE_SIZE
        ).to(device)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="simple",
                        choices=["simple", "fpn"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_DETECTION)
    parser.add_argument("--batch-size", type=int, default=config.DETECTION_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None,
                        help="Путь к чекпоинту для продолжения")
    args = parser.parse_args()
    
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Модель: {args.model}, epochs: {args.epochs}")
    
    # Данные (используем предсказания сегментации как входы!)
    train_dataset = TextYoloDataset(
        images_dir=config.SEGM_TRAIN_PREDS,
        labels_dir=config.YOLO_TRAIN_LABELS,
        input_size=config.DETECTION_IMAGE_SIZE,
        image_ext=".png"
    )
    val_dataset = TextYoloDataset(
        images_dir=config.SEGM_TEST_PREDS,
        labels_dir=config.YOLO_VAL_LABELS,
        input_size=config.DETECTION_IMAGE_SIZE,
        image_ext=".png"
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=detection_collate_fn
    )
    
    # Модель
    model = get_model(args.model, device)
    
    compute_loss = ComputeLoss(
        weight_bbox=5.0, weight_obj=1.0, weight_cls=0.0,
        obj_pos_weight=10.0, neg_pos_ratio=10, min_neg=100
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    # Runner
    ckpt_dir = config.DET_CHECKPOINTS / args.model
    runner = Runner(
        model=model,
        compute_loss=compute_loss,
        optimizer=optimizer,
        train_dataloader=train_loader,
        assign_target_method=assign_target,
        assign_target_kwargs={"pos_th": 0.4, "neg_th": 0.2},
        device=device,
        scheduler=scheduler,
        val_dataloader=val_loader,
        val_every=1,
        score_threshold=0.1,
        nms_threshold=0.5,
        max_boxes_per_cls=50,
        checkpoint_dir=ckpt_dir
    )
    
    # Обучение
    runner.train(num_epochs=args.epochs, resume_from=args.resume)
    
    # Визуализация
    print(f"\n=== Best mAP@0.5: {runner.best_val_metric:.4f} ===")
    visualize_prediction(
        model, val_dataset, device, idx=0,
        save_path=config.RESULTS_DIR / f"{args.model}_prediction.png"
    )


if __name__ == "__main__":
    main()
