"""Backbone на базе timm-моделей."""
import torch.nn as nn
import timm


class Backbone(nn.Module):
    """
    Обёртка вокруг timm.create_model для извлечения feature maps.
    
    По умолчанию извлекает предпоследнюю feature map, чтобы сохранить
    пространственное разрешение (важно для мелких объектов вроде текста).
    """
    
    def __init__(
        self,
        model_name="efficientnet_b0",
        out_indices=(-2,),
        pretrained=True,
        freeze=False
    ):
        super().__init__()
        
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )
        except Exception as e:
            print(f"Не удалось загрузить pretrained: {e}")
            print("Использую pretrained=False")
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                features_only=True,
                out_indices=out_indices
            )
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)
