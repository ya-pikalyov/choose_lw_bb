import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Context Enhancement Module (CEM)
class CEM(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.local_conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        local = self.local_conv(x)
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        out = self.relu(local + global_feat)
        return out

# Upsampling + Conv
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# WLUSNet Model
class WLUSNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, 16),
            ConvBlock(16, 16)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(16, 32),
            ConvBlock(32, 32)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck + CEM
        self.bottleneck = ConvBlock(64, 128)
        self.cem = CEM(128)

        # Decoder
        self.up3 = UpBlock(128 + 64, 64)
        self.up2 = UpBlock(64 + 32, 32)
        self.up1 = UpBlock(32 + 16, 16)

        # Output
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        # Bottleneck
        b = self.bottleneck(self.pool3(x3))
        b = self.cem(b)

        # Decoder
        #d3 = self.up3(b, x3)
        #d2 = self.up2(d3, x2)
        #d1 = self.up1(d2, x1)

        #out = self.classifier(d1)
        return b