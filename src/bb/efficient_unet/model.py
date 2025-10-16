import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Conv Block
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Depthwise Separable Conv (Efficient block)
class DWConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            DWConv(in_c, out_c),
            DWConv(out_c, out_c)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.block(x)
        p = self.pool(f)
        return f, p

# Decoder block with nested skip connections (UNet++)
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DWConv(in_c, out_c),
            DWConv(out_c, out_c)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# EfficientUNet++ model
class EfficientUNetPP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        self.center = DWConv(256, 512)

        # Decoder
        #self.dec4 = DecoderBlock(512 + 256, 256)
        #self.dec3 = DecoderBlock(256 + 128, 128)
        #self.dec2 = DecoderBlock(128 + 64, 64)
        #self.dec1 = DecoderBlock(64 + 32, 32)

        #self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        center = self.center(p4)

        '''d4 = self.dec4(center, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        '''

        #out = self.out_conv(d1)
        return center
