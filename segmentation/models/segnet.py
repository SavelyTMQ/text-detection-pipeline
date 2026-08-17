"""SegNet-Tiny — простая архитектура для baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet_Tiny(nn.Module):
    """
    Упрощённая версия SegNet для сегментации.
    Используется как baseline для сравнения.
    
    Reference: Badrinarayanan et al. "SegNet: A Deep Convolutional 
    Encoder-Decoder Architecture" (2015)
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc_conv0 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.enc_bn0 = nn.BatchNorm2d(8)
        
        self.enc_conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_bn1 = nn.BatchNorm2d(8)
        
        self.enc_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_bn2 = nn.BatchNorm2d(16)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(16, 16, kernel_size=1)
        
        # Decoder
        self.upsample0 = nn.Upsample(scale_factor=2)
        self.dec_bn0 = nn.BatchNorm2d(32)
        self.dec_conv0 = nn.ConvTranspose2d(32, 8, 3, padding=1)
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dec_bn1 = nn.BatchNorm2d(16)
        self.dec_conv1 = nn.ConvTranspose2d(16, 8, 3, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, 3, padding=1)
    
    def forward(self, x):
        # Encoder
        e0 = self.enc_bn0(self.pool0(F.relu(self.enc_conv0(x))))
        e1 = self.enc_bn1(self.pool1(F.relu(self.enc_conv1(e0))))
        e2 = self.enc_bn2(self.pool2(F.relu(self.enc_conv2(e1))))
        
        # Bottleneck
        b = self.bottleneck_conv(e2)
        
        # Decoder с skip connections
        d0 = self.dec_conv0(self.dec_bn0(F.relu(self.upsample0(torch.cat([b, e2], 1)))))
        d1 = self.dec_conv1(self.dec_bn1(F.relu(self.upsample1(torch.cat([d0, e1], 1)))))
        d2 = self.dec_conv2(self.dec_bn2(F.relu(self.upsample2(torch.cat([d1, e0], 1)))))
        
        return d2
