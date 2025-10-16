import torch
import torch.nn as nn
#from torch.nn.modules.utils import _pair as to_2tuple

import math
from functools import partial
from torch.nn import functional as F

from src.bb.stripnet.stripnet_blocks import OverlapPatchEmbed, Block
from src.utils.utils_nn import trunc_normal_init, constant_init, normal_init, autopad


class StripNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], k1s=[1,1,1,1], k2s=[19,19,19,19], drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, r2=None,
                 pretrained=False,
                 norm_cfg=None):
        super().__init__()


        self.pretrained = pretrained
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims


        self.head = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                                            embed_dim=self.embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=self.embed_dims[i], mlp_ratio=mlp_ratios[i], k1=k1s[i],k2=k2s[i],drop=drop_rate, drop_path=dpr[cur + j],
                norm_cfg=norm_cfg)

                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            if not self.pretrained:
                self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(
                    m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)#[-1]
        return x


def get_stripnet_small():
    model_ft = StripNet(img_size=224, in_chans=3,
                        embed_dims=[64, 128, 320, 512],
                        mlp_ratios=[8, 8, 4, 4],
                        k1s=[1,1,1,1],
                        k2s=[19,19,19,19],
                        drop_rate=0.1,
                        drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        depths=[2,2,4,2], num_stages=4,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
    return model_ft
def get_stripnet_tiny():
    model_ft = StripNet(img_size=224, in_chans=3,
                        embed_dims=[32, 64, 160, 256],
                        mlp_ratios=[8, 8, 4, 4],
                        k1s=[1,1,1,1],
                        k2s=[19,19,19,19],
                        drop_rate=0.1,
                        drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        depths=[3,3,5,2], num_stages=4,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
    return model_ft


def get_stripnet_tiny_tiny():
    model_ft = StripNet(img_size=224, in_chans=3,
                        embed_dims=[16, 32, 80, 128],
                        mlp_ratios=[8, 8, 4, 4],
                        k1s=[1,1,1,1],
                        k2s=[19,19,19,19],
                        drop_rate=0.1,
                        drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        depths=[3,3,5,2], num_stages=4,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
    return model_ft

def func():
    model = get_stripnet_small().to('cuda')
    def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

    for H,W in [(224,224),(384,384)]:
        x = torch.randn(1,3,H,W).to('cuda')
        _ = model.eval()(x)
        print(H,W, count_params(model))



if __name__ == "__main__":
    func()