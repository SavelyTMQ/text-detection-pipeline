# 📝 Text Detection Pipeline: от сегментации к детекции

Двухэтапная система детекции текста на естественных изображениях 
на датасете **Total-Text** (изогнутые линии текста, 1555 изображений).

<img width="1220" height="576" alt="image" src="https://github.com/user-attachments/assets/b9eea4f8-7c83-411c-a060-147ea8297808" />


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

<details>
  <summary>SegNet_Tiny</summary>
  
<img width="553" height="435" alt="image" src="https://github.com/user-attachments/assets/5d11c892-ad49-4148-ade3-57e8d7ee5cfd" />
<img width="993" height="339" alt="image" src="https://github.com/user-attachments/assets/dafe2586-c87e-4dab-a002-0d4501b666d9" />

</details>

<details>
  <summary>DeepLabV3</summary>
  
<img width="562" height="435" alt="image" src="https://github.com/user-attachments/assets/2140d728-1bfc-4d63-9b1c-812c24996182" />
<img width="993" height="339" alt="image" src="https://github.com/user-attachments/assets/2f0a4063-9fe8-41ee-b2a9-f402f74ffb84" />

</details>

<details>
  <summary>U-Net</summary>
  
<img width="553" height="435" alt="image" src="https://github.com/user-attachments/assets/2455f532-4ea1-47e3-89d9-d2edfb01292c" />
<img width="993" height="339" alt="image" src="https://github.com/user-attachments/assets/84d407de-3c31-4609-b074-b791ad623133" />

</details>

#### Результаты сегментации, которые дальше применялись для детекции

<img width="1169" height="1589" alt="image" src="https://github.com/user-attachments/assets/25d362b0-070c-4e8b-821b-48e30101394b" />

### Этап 2: Детекция bounding boxes

| Модель | mAP@0.5 | Эпох | Примечание |
|--------|:-------:|:----:|------------|
| Simple Detector (EfficientNet + FPN) | 0.27 | 30 | Одноуровневый, свой код |
| FPN Detector (multi-level) | 0.08 | 50 | Не сошёлся |
| **YOLOv8n** | **0.59** | 200 | **Лучший результат** |

<details>
  <summary>Simple Detector</summary>
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/deaf5647-10dd-42b1-b85e-72eac5512a4a" />
<img width="790" height="812" alt="image" src="https://github.com/user-attachments/assets/4fc6bfdf-7f61-4dc0-bdaa-4b1872034f99" />
</details>

<details>
  <summary>FPN Detector</summary>
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/47e17b32-234b-4f49-a626-d462e191df0d" />
<img width="972" height="844" alt="image" src="https://github.com/user-attachments/assets/fda9bf63-0f48-4bc2-a16e-98415c9e3378" />
</details>

<details>
  <summary>YOLOv8n</summary>
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/0fd86559-9b56-4a53-8079-5f452e3b7871" />
<img width="1085" height="2025" alt="image" src="https://github.com/user-attachments/assets/b89ce77b-fe55-45db-aeff-18b4801a0d55" />
</details>

**Ключевой вывод:** U-Net + YOLOv8 дают самый качественный pipeline.

<img width="1085" height="2025" alt="image" src="https://github.com/user-attachments/assets/b89ce77b-fe55-45db-aeff-18b4801a0d55" />

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
