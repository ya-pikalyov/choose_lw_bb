import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic CNN Block
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# Lightweight CNN Encoder (like MobileNet blocks)
class MobileEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, 3, 2, 1)
        self.block1 = ConvBNReLU(32, 64, 3, 2, 1)
        self.block2 = ConvBNReLU(64, 128, 3, 2, 1)
        self.block3 = ConvBNReLU(128, 256, 3, 2, 1)

    def forward(self, x):
        x1 = self.stem(x)  # [B, 32, H/2, W/2]
        x2 = self.block1(x1)  # [B, 64, H/4, W/4]
        x3 = self.block2(x2)  # [B, 128, H/8, W/8]
        x4 = self.block3(x3)  # [B, 256, H/16, W/16]
        return x1, x2, x3, x4


# Transformer Decoder Block
class SimpleTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x2 = self.attn(x, x, x)[0]
        x = self.norm1(x + x2)
        x2 = self.ffn(x)
        x = self.norm2(x + x2)
        return x

# Full Mob
