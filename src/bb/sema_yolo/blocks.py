import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3K2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, k=3)
        self.conv2 = ConvBNAct(out_channels, out_channels, k=3)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class RFA_C3K2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.c3k2 = C3K2(in_channels, out_channels)

    def forward(self, x):
        w = self.attn(x)
        x = x * w
        return self.c3k2(x)

class GCP_ASFF(nn.Module):
    def __init__(self, c2=64, c3=128, c4=256, c5=512, out_c=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # project each feature to same dimension
        self.compress2 = nn.Conv2d(c2, out_c, 1)
        self.compress3 = nn.Conv2d(c3, out_c, 1)
        self.compress4 = nn.Conv2d(c4, out_c, 1)
        self.compress5 = nn.Conv2d(c5, out_c, 1)

        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Conv2d(out_c * 4, out_c, 1)

    def forward(self, p2, p3, p4, p5):
        w2 = self.sigmoid(self.compress2(self.pool(p2)))
        w3 = self.sigmoid(self.compress3(self.pool(p3)))
        w4 = self.sigmoid(self.compress4(self.pool(p4)))
        w5 = self.sigmoid(self.compress5(self.pool(p5)))

        fused = torch.cat([p2 * w2, p3 * w3, p4 * w4, p5 * w5], dim=1)
        return self.fusion(fused)

