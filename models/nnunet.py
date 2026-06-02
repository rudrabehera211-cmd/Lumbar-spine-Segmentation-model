import torch
import torch.nn as nn
from typing import List, Optional


class ConvDropoutNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.lrelu(self.norm(self.dropout(self.conv(x))))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, convs_per_stage=2, dropout=0.0):
        super().__init__()
        self.pool = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        convs = [ConvDropoutNorm(out_ch, out_ch, dropout=dropout)]
        for _ in range(convs_per_stage - 1):
            convs.append(ConvDropoutNorm(out_ch, out_ch, dropout=dropout))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, skip_ch, dec_ch, out_ch, convs_per_stage=2, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(dec_ch, dec_ch, 2, stride=2)
        convs = [ConvDropoutNorm(dec_ch + skip_ch, out_ch, dropout=dropout)]
        for _ in range(convs_per_stage - 1):
            convs.append(ConvDropoutNorm(out_ch, out_ch, dropout=dropout))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, skip):
        x = self.up(x)
        diff = [skip.size(d) - x.size(d) for d in range(2, 5)]
        x = nn.functional.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                                   diff[1] // 2, diff[1] - diff[1] // 2,
                                   diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([skip, x], dim=1)
        return self.convs(x)


class nnUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        features: List[int] = None,
        deep_supervision: bool = True,
    ):
        super().__init__()
        features = features or [32, 64, 128, 256, 320, 320]
        self.deep_supervision = deep_supervision

        self.encoder_stages = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        self.encoder_stages.append(nn.Sequential(
            ConvDropoutNorm(in_channels, features[0]),
            ConvDropoutNorm(features[0], features[0]),
        ))

        for i in range(len(features) - 1):
            self.encoder_stages.append(
                DownBlock(features[i], features[i + 1], dropout=0.0 if i < 2 else 0.2)
            )

        for i in range(len(features) - 2, -1, -1):
            skip_ch = features[i]
            dec_ch = features[i + 1]
            out_ch = features[i]
            self.decoder_stages.append(
                UpBlock(skip_ch, dec_ch, out_ch, dropout=0.2)
            )

        self.out = nn.Conv3d(features[0], out_channels, 1)

        if deep_supervision:
            self.ds_outs = nn.ModuleList([
                nn.Conv3d(f, out_channels, 1)
                for f in reversed(features[1:-1])
            ])

    def forward(self, x):
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        deep_outputs = []
        for i, stage in enumerate(self.decoder_stages):
            skip = skips[-(i + 2)]
            x = stage(x, skip)
            if self.deep_supervision and i < len(self.ds_outs):
                deep_outputs.append(nn.functional.interpolate(
                    self.ds_outs[i](x), size=skips[0].shape[2:], mode='trilinear', align_corners=False
                ))

        logits = self.out(x)

        if self.deep_supervision and self.training and deep_outputs:
            return [logits] + deep_outputs
        return logits
