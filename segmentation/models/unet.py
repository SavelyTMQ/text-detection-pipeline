"""U-Net архитектура для семантической сегментации."""
import torch
import torch.nn as nn


class EncoderBlockUNet(nn.Module):
    """Блок энкодера: Conv-BN-ReLU x depth + MaxPool."""
    
    def __init__(self, in_channels, out_channels, depth=2, 
                 kernel_size=3, padding=1):
        super().__init__()
        layers = []
        
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        for _ in range(depth - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        self.layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.layers(x)  # для skip connection
        x = self.pool(skip)
        return skip, x


class DecoderBlockUNet(nn.Module):
    """Блок декодера: ConvTranspose + Concat + Conv-BN-ReLU x depth."""
    
    def __init__(self, in_channels, skip_channels, out_channels, depth=2,
                 kernel_size=3, padding=1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        
        combined_channels = in_channels // 2 + skip_channels
        layers = [
            nn.Conv2d(combined_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(depth - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.layers(x)


class UNet(nn.Module):
    """
    Классическая U-Net архитектура для сегментации.
    
    Reference: Ronneberger et al. "U-Net: Convolutional Networks for 
    Biomedical Image Segmentation" (2015)
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.encoder0 = EncoderBlockUNet(in_channels, 64)
        self.encoder1 = EncoderBlockUNet(64, 128)
        self.encoder2 = EncoderBlockUNet(128, 256)
        self.encoder3 = EncoderBlockUNet(256, 512)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.decoder0 = DecoderBlockUNet(1024, 512, 512)
        self.decoder1 = DecoderBlockUNet(512, 256, 256)
        self.decoder2 = DecoderBlockUNet(256, 128, 128)
        self.decoder3 = DecoderBlockUNet(128, 64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        sk0, x = self.encoder0(x)
        sk1, x = self.encoder1(x)
        sk2, x = self.encoder2(x)
        sk3, x = self.encoder3(x)
        
        x = self.bottleneck(x)
        
        x = self.decoder0(x, sk3)
        x = self.decoder1(x, sk2)
        x = self.decoder2(x, sk1)
        x = self.decoder3(x, sk0)
        
        return self.final_conv(x)
