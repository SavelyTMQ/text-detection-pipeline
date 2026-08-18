# 📝 Text Detection Pipeline: от сегментации к детекции

Двухэтапная система детекции текста на естественных изображениях 
на датасете **Total-Text** (изогнутые линии текста, 1555 изображений).

**Pipeline:** Сегментация текста (U-Net/DeepLabV3) → Детекция bounding boxes (YOLOv8/кастомный детектор)

## 🎯 Задача

Total-Text — сложный датасет с текстом в трёх ориентациях: горизонтальный, 
многонаправленный, **изогнутый**. Задача — точно локализовать области текста.

## 📊 Результаты

### Этап 1: Семантическая сегментация

| Модель | IoU | Эпох | Примечание |
|--------|:---:|:----:|------------|
| SegNet_Tiny | 0.17 | 30 | Baseline, простая архитектура |
| DeepLabV3 (ResNet50) | 0.31 | 30 | Готовая модель |
| **U-Net (from scratch)** | **0.44** | 30 | **Лучший результат** |

### Этап 2: Детекция bounding boxes

| Модель | mAP@0.5 | Эпох | Примечание |
|--------|:-------:|:----:|------------|
| Simple Detector (EfficientNet + FPN) | 0.27 | 30 | Одноуровневый, свой код |
| FPN Detector (multi-level) | 0.08 | 50 | Не сошёлся |
| **YOLOv8n** | **0.59** | 200 | **Лучший результат** |

**Ключевой вывод:** U-Net + YOLOv8 дают самый качественный pipeline.

## 🏗️ Что реализовано с нуля

- **U-Net** — классическая архитектура для сегментации
- **SegNet-Tiny** — упрощённый baseline
- **Simple Detector** — одноуровневый якорный детектор (EfficientNet + FPN)
- **FPN Detector** — многоуровневый детектор с decoupled heads
- **Loss с hard negative mining** — для несбалансированных данных
- **Runner с чекпоинтами** — универсальный тренер для детекции

## 🚀 Установка

```bash
git clone https://github.com/YOUR_NICK/text-detection-pipeline.git
cd text-detection-pipeline

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Данные (Total-Text): см. data/README.md
```

## 📖 Запуск

### Полный pipeline

```bash
# 1. Обучение сегментации + сохранение предсказаний
python -m segmentation.run --model unet --epochs 30 --save-predictions

# 2. Обучение YOLOv8
python -m yolo.train --epochs 200

# Или обучение кастомного детектора
python -m detection.run --model simple --epochs 30
```

### Сравнение архитектур

```bash
# Сегментация
python -m segmentation.run --model segnet
python -m segmentation.run --model unet
python -m segmentation.run --model deeplabv3

# Детекция
python -m detection.run --model simple
python -m detection.run --model fpn
```

## 🛠️ Стек

Python 3.10, PyTorch 2.0, timm, torchmetrics, torchvision, ultralytics (YOLOv8), OpenCV

## 🔗 Датасет

Total-Text: https://www.kaggle.com/datasets/ipythonx/totaltextstr
