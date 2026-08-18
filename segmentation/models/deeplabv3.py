"""Обёртка для готовой DeepLabV3 из torchvision."""
import torch.nn as nn
import torchvision


class DeepLabV3Wrapper(nn.Module):
    """
    Обёртка вокруг torchvision.models.segmentation.deeplabv3_resnet50.
    
    Torchvision-модель возвращает словарь с ключом "out", 
    а нам нужен просто тензор для совместимости с train loop.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)["out"]


def get_deeplabv3(num_classes=1, pretrained_backbone=False):
    """Создаёт DeepLabV3 с ResNet50 backbone."""
    deeplab = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone="DEFAULT" if pretrained_backbone else None,
        num_classes=num_classes
    )
    return DeepLabV3Wrapper(deeplab)
