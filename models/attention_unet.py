import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpWithAttention(nn.Module):
    def __init__(self, gate_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.att = AttentionGate(F_g=gate_ch, F_l=skip_ch, F_int=min(gate_ch, skip_ch) // 2)
        self.conv = DoubleConv(gate_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = [x2.size(d) - x1.size(d) for d in range(2, 5)]
        x1 = F.pad(x1, [diff[2] // 2, diff[2] - diff[2] // 2,
                         diff[1] // 2, diff[1] - diff[1] // 2,
                         diff[0] // 2, diff[0] - diff[0] // 2])
        x2_att = self.att(g=x1, x=x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        features: List[int] = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        features = features or [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])

        self.up1 = UpWithAttention(gate_ch=features[4], skip_ch=features[3], out_ch=features[3])
        self.up2 = UpWithAttention(gate_ch=features[3], skip_ch=features[2], out_ch=features[2])
        self.up3 = UpWithAttention(gate_ch=features[2], skip_ch=features[1], out_ch=features[1])
        self.up4 = UpWithAttention(gate_ch=features[1], skip_ch=features[0], out_ch=features[0])

        self.outc = nn.Conv3d(features[0], out_channels, 1)

        if deep_supervision:
            self.ds_out1 = nn.Conv3d(features[3], out_channels, 1)
            self.ds_out2 = nn.Conv3d(features[2], out_channels, 1)
            self.ds_out3 = nn.Conv3d(features[1], out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        if self.deep_supervision:
            ds1 = self.ds_out1(x)

        x = self.up2(x, x3)
        if self.deep_supervision:
            ds2 = self.ds_out2(x)

        x = self.up3(x, x2)
        if self.deep_supervision:
            ds3 = self.ds_out3(x)

        x = self.up4(x, x1)
        logits = self.outc(x)

        if self.deep_supervision and self.training:
            return [logits, ds1, ds2, ds3]
        return logits
