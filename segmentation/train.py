"""Обучение моделей сегментации."""
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm


def train_single_epoch(model, optimizer, criterion, train_dataloader, device):
    """Одна эпоха обучения."""
    model.train()
    avg_loss = 0
    
    for X_batch, Y_batch in tqdm(train_dataloader, desc="Train", leave=False):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        avg_loss += loss.item()
    
    return avg_loss / len(train_dataloader)


def validate_single_epoch(model, criterion, valid_dataloader, device):
    """Одна эпоха валидации с расчётом IoU."""
    iou_score = BinaryJaccardIndex().to(device)
    model.eval()
    
    avg_loss = 0
    avg_iou = 0
    
    with torch.no_grad():
        for X_batch, Y_batch in tqdm(valid_dataloader, desc="Val", leave=False):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            Y_pred = model(X_batch)
            pred_mask = (torch.sigmoid(Y_pred) > 0.5).int()
            
            iou = iou_score(pred_mask.squeeze(1), Y_batch.int().squeeze(1))
            loss = criterion(Y_pred, Y_batch)
            
            avg_loss += loss.item()
            avg_iou += iou.item()
    
    return avg_loss / len(valid_dataloader), avg_iou / len(valid_dataloader)


def train(model, optimizer, criterion, epochs, 
          train_dataloader, valid_dataloader, device, ckpt_path):
    """Полный цикл обучения с сохранением лучшего чекпоинта."""
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0
    
    for epoch in range(epochs):
        train_loss = train_single_epoch(model, optimizer, criterion, train_dataloader, device)
        val_loss, val_iou = validate_single_epoch(model, criterion, valid_dataloader, device)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_iou={val_iou:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Сохраняем чекпоинт каждой эпохи
        torch.save(model.state_dict(), ckpt_path / f"epoch_{epoch:02d}.pt")
        
        # Отдельно — лучший по IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), ckpt_path / "best.pt")
            print(f"  ✓ New best IoU: {val_iou:.4f}")
    
    return model, history
