import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAAttention(nn.Module):
    def __init__(self, dim, heads=4, pool_ratio=8):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.pool_ratio = pool_ratio

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = qkv

        # --- horizontal squeeze ---
        Hh = H // self.pool_ratio
        q_h = F.adaptive_avg_pool2d(q, (Hh, W))
        k_h = F.adaptive_avg_pool2d(k, (Hh, W))
        v_h = F.adaptive_avg_pool2d(v, (Hh, W))

        q_h = q_h.view(B, self.heads, C // self.heads, -1)
        k_h = k_h.view(B, self.heads, C // self.heads, -1)
        v_h = v_h.view(B, self.heads, C // self.heads, -1)
        attn_h = torch.softmax(torch.einsum('bhci,bhcj->bhij', q_h, k_h) / (C ** 0.5), dim=-1)
        out_h = torch.einsum('bhij,bhcj->bhci', attn_h, v_h)
        out_h = out_h.reshape(B, C, Hh, W)  # back to pooled grid
        out_h = F.interpolate(out_h, size=(H, W), mode='bilinear', align_corners=False)

        # --- vertical squeeze ---
        Ww = W // self.pool_ratio
        q_w = F.adaptive_avg_pool2d(q, (H, Ww))
        k_w = F.adaptive_avg_pool2d(k, (H, Ww))
        v_w = F.adaptive_avg_pool2d(v, (H, Ww))

        q_w = q_w.view(B, self.heads, C // self.heads, -1)
        k_w = k_w.view(B, self.heads, C // self.heads, -1)
        v_w = v_w.view(B, self.heads, C // self.heads, -1)
        attn_w = torch.softmax(torch.einsum('bhci,bhcj->bhij', q_w, k_w) / (C ** 0.5), dim=-1)
        out_w = torch.einsum('bhij,bhcj->bhci', attn_w, v_w)
        out_w = out_w.reshape(B, C, H, Ww)  # back to pooled grid
        out_w = F.interpolate(out_w, size=(H, W), mode='bilinear', align_corners=False)

        # --- fuse ---
        out = out_h + out_w
        out = self.dw_conv(out)
        return self.proj(out)


class SeaFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, heads=4, pool_ratio=8, drop=0.0):
        super().__init__()
        self.attn = SEAAttention(dim, heads, pool_ratio)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SeaFormerBackbone(nn.Module):
    def __init__(self, in_ch=3, dims=[64, 128, 256], depths=[2, 2, 6], **kwargs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            block = nn.Sequential(*[SeaFormerBlock(dim, **kwargs) for _ in range(depth)])
            self.stages.append(block)
            if i < len(dims) - 1:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(dim, dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dims[i + 1]),
                    nn.ReLU(inplace=True),
                ))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for layer in self.stages:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and any(isinstance(m, SeaFormerBlock) for m in layer):
                features.append(x)
        return features  # multi-scale features
