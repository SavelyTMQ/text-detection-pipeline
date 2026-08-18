"""Feature Pyramid Network — простая и полная версии."""
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedFPN(nn.Module):
    """Упрощённая FPN: одна свёртка 3x3 + BN + ReLU."""
    
    def __init__(self, in_channels, out_channels, use_activation=True):
        super().__init__()
        self.fpn_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if use_activation else nn.Identity()
    
    def forward(self, features):
        """features — список feature maps; используем только features[0]."""
        x = features[0]
        x = self.fpn_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class FPN(nn.Module):
    """
    Классическая FPN (Lin et al., 2017).
    
    Latéral 1x1 + top-down upsample + 3x3 smoothing.
    """
    
    def __init__(self, in_channels_list, out_channels=256, extra_conv=True):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            
            if extra_conv:
                self.fpn_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
                )
            else:
                self.fpn_convs.append(nn.Identity())
    
    def forward(self, features):
        """features: [C2, C3, C4, C5] от крупного к мелкому разрешению."""
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        fpn_features = []
        prev = laterals[-1]
        fpn_features.append(prev)
        
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                prev, size=laterals[i].shape[-2:], mode='nearest'
            )
            combined = laterals[i] + upsampled
            combined = self.fpn_convs[i](combined)
            fpn_features.append(combined)
            prev = combined
        
        # [P5, P4, P3, P2] -> [P2, P3, P4, P5]
        return fpn_features[::-1]
