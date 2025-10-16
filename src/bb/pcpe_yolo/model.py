import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Partial Convolution (PConv)
# ---------------------------
class PartialConv(nn.Module):
    def __init__(self, in_c, out_c, ratio=0.25, k=3):
        super().__init__()
        p_c = int(in_c * ratio)
        self.partial_conv = nn.Conv2d(p_c, out_c, k, stride=1,
                                      padding=k//2, groups=p_c, bias=False)
        self.pass_through = nn.Identity()
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        c = self.partial_conv.in_channels
        x_p = self.partial_conv(x[:, :c])
        x_id = x[:, c:]
        out = torch.cat([x_p, x_id], dim=1)
        return self.bn(out)


# ---------------------------
# Inception Depthwise Conv (IDConv)
# ---------------------------
class IDConv(nn.Module):
    def __init__(self, in_c, k_small=3, k_band=11):
        super().__init__()
        g = in_c // 4
        self.dw_small = nn.Conv2d(g, g, k_small, padding=k_small//2, groups=g)
        self.dw_wide  = nn.Conv2d(g, g, (1, k_band), padding=(0, k_band//2), groups=g)
        self.dw_tall  = nn.Conv2d(g, g, (k_band, 1), padding=(k_band//2, 0), groups=g)
        self.identity = nn.Identity()

    def forward(self, x):
        g = x.size(1) // 4
        x1, x2, x3, x4 = x[:, :g], x[:, g:2*g], x[:, 2*g:3*g], x[:, 3*g:]
        return torch.cat([self.dw_small(x1), self.dw_wide(x2),
                          self.dw_tall(x3), self.identity(x4)], dim=1)


# ---------------------------
# GhostBottleneckV2
# ---------------------------
class GhostModule(nn.Module):
    def __init__(self, in_c, out_c, ratio=2):
        super().__init__()
        init_ch = out_c // ratio
        new_ch = out_c - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, init_ch, 1, bias=False),
            nn.BatchNorm2d(init_ch), nn.SiLU()
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, 3, 1, 1, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch), nn.SiLU()
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], 1)

class DFC_Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h_fc = nn.Conv2d(c, c, 1)
        self.v_fc = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.h_fc(x) + self.v_fc(x)

class GhostBottleneckV2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.ghost1 = GhostModule(in_c, out_c)
        self.attn   = DFC_Attention(out_c)
        self.ghost2 = GhostModule(out_c, out_c)
        self.shortcut = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.ghost1(x)
        x = self.attn(x) * x
        x = self.ghost2(x)
        return x + identity


# ---------------------------
# C2f_PIG (dynamic bottleneck switch)
# ---------------------------
class C2f_PIG(nn.Module):
    def __init__(self, in_c, out_c, n=3, threshold=3):
        super().__init__()
        blocks = []
        # first block always projects to out_c
        blocks.append(GhostBottleneckV2(in_c, out_c))
        for i in range(1, n):
            if n <= threshold:
                # High-parameter path
                blocks.append(IDConv(out_c))
            else:
                # Lightweight path
                blocks.append(GhostBottleneckV2(out_c, out_c))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ---------------------------
# CAA (Context Anchor Attention)
# ---------------------------
class CAA(nn.Module):
    def __init__(self, in_c, stage_idx=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_c, in_c, 1)
        k = 11 + 2 * stage_idx
        self.dw_h = nn.Conv2d(in_c, in_c, (1, k), padding=(0, k//2), groups=in_c)
        self.dw_v = nn.Conv2d(in_c, in_c, (k, 1), padding=(k//2, 0), groups=in_c)
        self.refine = nn.Conv2d(in_c, in_c, 1)

    def forward(self, x):
        pooled = self.conv1(self.pool(x))
        h = self.dw_h(pooled)
        v = self.dw_v(pooled)
        attn = torch.sigmoid(self.refine(h + v))
        return x * attn


# ---------------------------
# EUCB (Efficient Up-Convolution Block)
# ---------------------------
class EUCB(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dwconv = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.proj(x)


# ---------------------------
# Example: Backbone/Neck
# ---------------------------
class PCPE_YOLO_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.stage1 = C2f_PIG(32, 64, n=2)
        self.stage2 = C2f_PIG(64, 128, n=4)
        self.stage3 = C2f_PIG(128, 256, n=6)
        self.stage4 = C2f_PIG(256, 512, n=6)
        #self.caa = CAA(512, stage_idx=4)
        #self.eu_cb = EUCB(512, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        #x = self.caa(x)
        #x = self.eu_cb(x)
        return x



class PCPE_YOLO_Light(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.stage1 = C2f_PIG(24, 48, n=1)  # lighter depth
        self.stage2 = C2f_PIG(48, 96, n=2)
        self.stage3 = C2f_PIG(96, 192, n=3)
        self.stage4 = C2f_PIG(192, 384, n=3)
        self.caa = CAA(384, stage_idx=4)    # with capped kernel (≤9)
        self.eu_cb = EUCB(384, 128) # smaller up-proj

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.caa(x)
        x = self.eu_cb(x)
        return x

if __name__ == "__main__":
    model = PCPE_YOLO_Backbone()
    y = model(torch.randn(1,3,224,224))
    print(y.shape)  # [1, 256, H/16, W/16] approx
