import torch
import torch.nn as nn
import torch.nn.functional as F
from src.bb.sema_yolo.blocks import ConvBNAct, C3K2, RFA_C3K2, GCP_ASFF


class SEMA_YOLO_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3x384x384
        self.stem = ConvBNAct(3, 32, k=3, s=1)

        self.stage1 = C3K2(32, 64)   # Output P2
        self.stage2 = C3K2(64, 128)  # Output P3
        self.stage3 = C3K2(128, 256) # Output P4
        self.stage4 = C3K2(256, 512) # Output P5

        self.rfa_p2 = RFA_C3K2(64, 64)
        self.rfa_p3 = RFA_C3K2(128, 128)
        self.rfa_p4 = RFA_C3K2(256, 256)
        self.rfa_p5 = RFA_C3K2(512, 512)

        self.gcp_asff = GCP_ASFF(64)


    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage1(x)      # 96x96
        p3 = self.stage2(p2)     # 48x48
        p4 = self.stage3(p3)     # 24x24
        p5 = self.stage4(p4)     # 12x12

        p2 = self.rfa_p2(p2)
        p3 = self.rfa_p3(p3)
        p4 = self.rfa_p4(p4)
        p5 = self.rfa_p5(p5)

        self.gcp_asff = GCP_ASFF(c2=64, c3=128, c4=256, c5=512, out_c=64)


        return p2, p3, p4, p5