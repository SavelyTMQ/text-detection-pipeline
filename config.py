"""Общие настройки проекта."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent

# Пути к данным (Total-Text)
DATA_DIR = Path(os.getenv("DATA_DIR", "./data/Total-Text"))
TRAIN_IMAGES_DIR = DATA_DIR / "Train"
TEST_IMAGES_DIR = DATA_DIR / "Test"
TRAIN_MASKS_DIR = DATA_DIR / "Annotation" / "groundtruth_pixel" / "Train"
TEST_MASKS_DIR = DATA_DIR / "Annotation" / "groundtruth_pixel" / "Test"

# YOLO формат (создаётся после segmentation)
YOLO_LABELS_DIR = Path("./data/YOLO_format")

# Куда сохраняем предсказания сегментации (вход для детекции)
SEGM_PREDICTIONS_DIR = Path("./predictions/segmentation")
SEGM_TRAIN_PREDS = SEGM_PREDICTIONS_DIR / "train"
SEGM_TEST_PREDS = SEGM_PREDICTIONS_DIR / "test"

# Чекпоинты
CHECKPOINTS_DIR = Path("./checkpoints")
SEGM_CHECKPOINTS = CHECKPOINTS_DIR / "segmentation"
DET_CHECKPOINTS = CHECKPOINTS_DIR / "detection"

# Создаём папки
for d in [SEGM_PREDICTIONS_DIR, SEGM_TRAIN_PREDS, SEGM_TEST_PREDS,
          CHECKPOINTS_DIR, SEGM_CHECKPOINTS, DET_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# Гиперпараметры
IMAGE_SIZE = (256, 256)          # для сегментации
DETECTION_IMAGE_SIZE = (512, 512) # для детекции
BATCH_SIZE = 32
DETECTION_BATCH_SIZE = 4
EPOCHS_SEGM = 30
EPOCHS_DETECTION = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
