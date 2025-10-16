import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 1. Detail Branch
# ------------------------------
class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x  # shape: [B, 128, H/8, W/8]

# ------------------------------
# 2. Semantic Branch
# ------------------------------
class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Conv2d(16, 8, 1)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.concat = nn.Conv2d(24, 16, 3, 1, 1)  # concat(16+8)

    def forward(self, x):
        x1 = self.conv1(x)            # [B, 16, H/2, W/2]
        x2 = self.pool(x1)            # [B, 16, H/4, W/4]
        x3 = self.conv_1x1(x1)        # [B, 8, H/2, W/2]
        x3 = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        out = torch.cat([x2, x3], dim=1)  # [B, 24, H/4, W/4]
        return self.concat(out)           # [B, 16, H/4, W/4]


class CEBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_gap = self.gap(x)
        x = self.conv(x_gap)
        x = self.bn(x)
        x = self.relu(x)
        return x * x_gap

class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock()
        self.stage3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ce = CEBlock(128, 128)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.ce(x)

# ------------------------------
# 3. BGA Fusion Layer
# ------------------------------
class BGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.semantic_conv = nn.Conv2d(128, 128, 3, padding=1)

        self.detail_att = nn.Conv2d(128, 128, 1)
        self.semantic_att = nn.Conv2d(128, 128, 1)

    def forward(self, d, s):
        # Resize semantic features to match detail branch
        s_up = F.interpolate(s, size=d.shape[2:], mode="bilinear", align_corners=True)

        d_conv = self.detail_conv(d)
        s_conv = self.semantic_conv(s_up)

        d_att = torch.sigmoid(self.detail_att(d))
        s_att = torch.sigmoid(self.semantic_att(s_conv))

        out = d_conv * s_att + s_conv * d_att
        return out

# ------------------------------
# 4. BiSeNetV2 (Final)
# ------------------------------
class BiSeNetV2(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGA()
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        d = self.detail(x)
        s = self.semantic(x)
        x = self.bga(d, s)
        #x = self.head(x)
        #x = F.interpolate(x, size=(x.shape[2]*8, x.shape[3]*8), mode='bilinear', align_corners=True)
        return x
