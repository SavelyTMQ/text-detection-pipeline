"""Runner: цикл обучения с чекпоинтами и валидацией."""
from pathlib import Path
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from detection.postprocess import filter_predictions


class Runner:
    """Универсальный тренер для детекторов."""
    
    def __init__(
        self,
        model,
        compute_loss,
        optimizer,
        train_dataloader,
        assign_target_method,
        device=None,
        scheduler=None,
        assign_target_kwargs=None,
        val_dataloader=None,
        val_every=1,
        score_threshold=0.1,
        nms_threshold=0.5,
        max_boxes_per_cls=50,
        checkpoint_dir="./checkpoints/detection"
    ):
        self.model = model
        self.compute_loss = compute_loss
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        
        assign_target_kwargs = assign_target_kwargs or {}
        self.assign_target_method = partial(assign_target_method, **assign_target_kwargs)
        
        self.device = torch.device("cpu" if device is None else device)
        self.model.to(self.device)
        
        self.scheduler = scheduler
        self.val_dataloader = val_dataloader
        self.val_every = val_every
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_per_cls = max_boxes_per_cls
        
        # История
        self.batch_loss = []
        self.epoch_loss = []
        self.val_metric = []
        self.epoch_numbers = []
        self.val_epochs = []
        self.batches_per_epoch = []
        
        self.best_val_metric = -float("inf")
        self.start_epoch = 1
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "batch_loss": self.batch_loss,
            "epoch_loss": self.epoch_loss,
            "val_metric": self.val_metric,
            "best_val_metric": self.best_val_metric,
            "epoch_numbers": self.epoch_numbers,
            "val_epochs": self.val_epochs,
            "batches_per_epoch": self.batches_per_epoch,
        }
        
        torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, self.checkpoint_dir / "last.pt")
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
    
    def load_checkpoint(self, path="last.pt", load_optimizer=True):
        path = Path(path)
        if not path.is_absolute():
            path = self.checkpoint_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint не найден: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.batch_loss = checkpoint.get("batch_loss", [])
        self.epoch_loss = checkpoint.get("epoch_loss", [])
        self.val_metric = checkpoint.get("val_metric", [])
        self.best_val_metric = checkpoint.get("best_val_metric", -float("inf"))
        self.epoch_numbers = checkpoint.get("epoch_numbers", [])
        self.val_epochs = checkpoint.get("val_epochs", [])
        self.batches_per_epoch = checkpoint.get("batches_per_epoch", [])
        
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✓ Загружен checkpoint: {path}. Продолжаем с эпохи {self.start_epoch}")
    
    def _run_train_epoch(self):
        self.model.train()
        batch_loss = []
        anchors = self.model.anchors.view(-1, 4).to(self.device)
        
        for images, targets in tqdm(self.train_dataloader, desc="Train", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            
            accum_loss = torch.tensor(0.0, device=self.device)
            
            for ix in range(images.shape[0]):
                gt_boxes = targets[ix]["boxes"].to(self.device)
                gt_labels = targets[ix]["labels"].to(self.device)
                
                assigned_targets = self.assign_target_method(
                    anchors, gt_boxes, gt_labels,
                    num_classes=self.model.num_classes
                )
                
                outputs_ix = [out[ix] for out in outputs]
                loss = self.compute_loss(outputs_ix, assigned_targets)
                accum_loss += loss
            
            accum_loss = accum_loss / images.shape[0]
            
            self.optimizer.zero_grad()
            accum_loss.backward()
            self.optimizer.step()
            
            batch_loss.append(accum_loss.detach().cpu().item())
        
        return batch_loss
    
    def train(self, num_epochs=10, resume_from=None):
        if resume_from:
            self.load_checkpoint(resume_from)
        
        end_epoch = self.start_epoch + num_epochs - 1
        
        for epoch in range(self.start_epoch, end_epoch + 1):
            batch_loss = self._run_train_epoch()
            
            self.batch_loss.extend(batch_loss)
            self.batches_per_epoch.append(len(batch_loss))
            
            epoch_loss = np.mean(batch_loss)
            self.epoch_loss.append(epoch_loss)
            self.epoch_numbers.append(epoch)
            
            is_best = False
            val_desc = ""
            
            if self.val_dataloader and epoch % self.val_every == 0:
                val_metric = self.validate()
                self.val_metric.append(val_metric)
                self.val_epochs.append(epoch)
                val_desc = f", val mAP@0.5={val_metric:.4f}"
                
                if val_metric > self.best_val_metric:
                    self.best_val_metric = val_metric
                    is_best = True
            
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}{val_desc}")
            
            if self.scheduler:
                self.scheduler.step()
            
            self.save_checkpoint(epoch, is_best=is_best)
        
        self.start_epoch = end_epoch + 1
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
        
        for images, targets in tqdm(self.val_dataloader, desc="Val", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            
            predicts = filter_predictions(
                outputs,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                max_boxes_per_cls=self.max_boxes_per_cls,
                return_type="torch"
            )
            
            cpu_targets = [
                {"boxes": t["boxes"].detach().cpu().float(),
                 "labels": t["labels"].detach().cpu().long()}
                for t in targets
            ]
            
            metric.update(predicts, cpu_targets)
        
        result = metric.compute()
        print(f"  mAP={result['map']:.4f}, "
              f"mAP@0.5={result['map_50']:.4f}, "
              f"mAP@0.75={result['map_75']:.4f}")
        
        return result["map_50"].item()
