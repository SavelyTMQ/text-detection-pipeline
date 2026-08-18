"""Обучение YOLOv8 на подготовленном датасете.

Использование:
    python -m yolo.train --epochs 200
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

import config
from yolo.prepare_dataset import prepare_yolo_split, create_data_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Базовая модель YOLOv8")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", type=str, default="text_detector")
    args = parser.parse_args()
    
    # ─── Подготовка датасета ────────────────────
    print("[1/3] Подготовка датасета в формате YOLO...")
    yolo_root = config.PROJECT_ROOT / "data" / "yolo_dataset"
    
    prepare_yolo_split(
        src_images_dir=config.SEGM_TRAIN_PREDS,
        src_labels_dir=config.YOLO_TRAIN_LABELS,
        dst_images_dir=yolo_root / "images" / "train",
        dst_labels_dir=yolo_root / "labels" / "train",
        image_ext=".png"
    )
    prepare_yolo_split(
        src_images_dir=config.SEGM_TEST_PREDS,
        src_labels_dir=config.YOLO_VAL_LABELS,
        dst_images_dir=yolo_root / "images" / "val",
        dst_labels_dir=yolo_root / "labels" / "val",
        image_ext=".png"
    )
    
    # ─── Создание data.yaml ─────────────────────
    print("\n[2/3] Создание data.yaml...")
    data_yaml_path = yolo_root / "data.yaml"
    create_data_yaml(yolo_root, data_yaml_path)
    
    # ─── Обучение ───────────────────────────────
    print("\n[3/3] Обучение YOLOv8...")
    model = YOLO(args.model)
    
    results = model.train(
        data=str(data_yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(config.CHECKPOINTS_DIR / "yolo"),
        name=args.name,
        exist_ok=True
    )
    
    # ─── Валидация ──────────────────────────────
    print("\nВалидация...")
    metrics = model.val(data=str(data_yaml_path), imgsz=args.imgsz)
    
    print(f"\n=== Результаты ===")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
