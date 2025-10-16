import torch
import torch.nn as nn
import torch.nn.functional as F

class SBCFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, shrink_ratio=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shrink_ratio = shrink_ratio
        self.dim_shrink = dim // shrink_ratio

        # Shrunken attention stream
        self.shrink_proj = nn.Conv2d(dim, self.dim_shrink, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.dim_shrink, 1, 1))
        self.attn = nn.MultiheadAttention(self.dim_shrink, num_heads, dropout=dropout)
        self.expand_proj = nn.Conv2d(self.dim_shrink, dim, kernel_size=1)

        # Local pass-through stream
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # Fusion and MLP
        self.norm = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Shrink stream: Downsample spatially
        x_shrink = F.adaptive_avg_pool2d(x, (H // self.shrink_ratio, W // self.shrink_ratio))
        x_shrink = self.shrink_proj(x_shrink) + self.pos_embed

        # Prepare tokens for attention: (HW) × B, embedding dim
        tokens = x_shrink.flatten(2).permute(2, 0, 1)  # (N, B, dim_shrink)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.permute(1, 2, 0).view(B, self.dim_shrink, H // self.shrink_ratio, W // self.shrink_ratio)
        attn_out = self.expand_proj(attn_out)
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # Pass-through stream: local detail
        local = self.local_conv(x)

        # Fusion and MLP residual connection
        x = x + attn_out + local
        x = x + self.mlp(self.norm(x))

        return x

class SBCFormerBackbone(nn.Module):
    def __init__(self, in_channels=3, dims=[64, 128, 256], depths=[2, 2, 4], **block_kwargs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = [SBCFormerBlock(dim, **block_kwargs) for _ in range(depth)]
            self.stages.append(nn.Sequential(*blocks))
            if i < len(dims) - 1:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(dim, dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dims[i+1]),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for layer in self.stages:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and isinstance(layer[0], SBCFormerBlock):
                features.append(x)
        return features
