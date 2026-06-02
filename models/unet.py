import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


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


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = [x2.size(d) - x1.size(d) for d in range(2, 5)]
        x1 = F.pad(x1, [diff[2] // 2, diff[2] - diff[2] // 2,
                         diff[1] // 2, diff[1] - diff[1] // 2,
                         diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        features: List[int] = None,
        trilinear: bool = True,
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

        self.up1 = Up(features[4] + features[3], features[3], trilinear)
        self.up2 = Up(features[3] + features[2], features[2], trilinear)
        self.up3 = Up(features[2] + features[1], features[1], trilinear)
        self.up4 = Up(features[1] + features[0], features[0], trilinear)

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
