import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, max_tokens: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_tokens = max_tokens

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x1.shape
        N = D * H * W
        do_pool = N > self.max_tokens
        if do_pool:
            ratio = (N / self.max_tokens) ** (1 / 3)
            pd = max(1, int(D / ratio))
            ph = max(1, int(H / ratio))
            pw = max(1, int(W / ratio))
            size = (pd, ph, pw)
            x1_pool, x2_pool = F.adaptive_avg_pool3d(x1, size), F.adaptive_avg_pool3d(x2, size)
            x1_flat = x1_pool.flatten(2).transpose(1, 2)
            x2_flat = x2_pool.flatten(2).transpose(1, 2)
        else:
            x1_flat = x1.flatten(2).transpose(1, 2)
            x2_flat = x2.flatten(2).transpose(1, 2)

        Q = self.q_proj(x1_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x2_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x2_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, C, *size) if do_pool else out.transpose(1, 2).reshape(B, C, D, H, W)

        if do_pool:
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)

        return x1 + self.gamma * out


class MultiScaleContextBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scales: List[int] = (1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList()
        for s in scales:
            if s == 1:
                self.convs.append(nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.convs.append(nn.Sequential(
                    nn.AvgPool3d(s, stride=s),
                    nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=s, mode='trilinear', align_corners=False),
                ))
        self.fusion = nn.Sequential(
            nn.Conv3d(out_ch * len(scales), out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.shape
        outs = []
        for s, conv in zip(self.scales, self.convs):
            if D < s or H < s or W < s:
                outs.append(outs[-1] if outs else x)
            else:
                outs.append(conv(x))
        out = torch.cat(outs, dim=1)
        return self.fusion(out)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1, bias=False), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1, bias=False), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1, bias=False), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ATMNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        features: List[int] = None,
        num_heads: int = 4,
        deep_supervision: bool = True,
    ):
        super().__init__()
        features = features or [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.inc = nn.Sequential(
            nn.Conv3d(in_channels, features[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(features[0], features[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
        )

        self.msc1 = MultiScaleContextBlock(features[0], features[0])
        self.cross_attn1 = CrossAttentionFusion(features[0], num_heads)

        self.down1 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(features[0], features[1], 1))
        self.msc2 = MultiScaleContextBlock(features[1], features[1])
        self.cross_attn2 = CrossAttentionFusion(features[1], num_heads)

        self.down2 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(features[1], features[2], 1))
        self.msc3 = MultiScaleContextBlock(features[2], features[2])
        self.cross_attn3 = CrossAttentionFusion(features[2], num_heads)

        self.down3 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(features[2], features[3], 1))
        self.msc4 = MultiScaleContextBlock(features[3], features[3])
        self.cross_attn4 = CrossAttentionFusion(features[3], num_heads)

        self.down4 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(features[3], features[4], 1))
        self.msc_bottleneck = MultiScaleContextBlock(features[4], features[4])

        self.att_gates = nn.ModuleList([
            AttentionGate(features[i], features[i], features[i] // 2)
            for i in range(3, -1, -1)
        ])
        self.up_convs = nn.ModuleList([
            self._make_conv_block(features[i + 1] + features[i], features[i])
            for i in range(3, -1, -1)
        ])

        self.out = nn.Conv3d(features[0], out_channels, 1)

        if deep_supervision:
            self.ds_out1 = nn.Conv3d(features[3], out_channels, 1)
            self.ds_out2 = nn.Conv3d(features[2], out_channels, 1)
            self.ds_out3 = nn.Conv3d(features[1], out_channels, 1)

    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.cross_attn1(x1, self.msc1(x1))

        x2 = self.down1(x1)
        x2 = self.cross_attn2(x2, self.msc2(x2))

        x3 = self.down2(x2)
        x3 = self.cross_attn3(x3, self.msc3(x3))

        x4 = self.down3(x3)
        x4 = self.cross_attn4(x4, self.msc4(x4))

        x5 = self.down4(x4)
        x5 = self.msc_bottleneck(x5)

        skip_features = [x1, x2, x3, x4]
        decoder_features = [x5]
        ds_outputs = []

        for i in range(4):
            skip = skip_features[3 - i]
            d_up = F.interpolate(decoder_features[-1], size=skip.shape[2:], mode='trilinear', align_corners=True)
            d_att = self.att_gates[i](g=skip, x=skip)
            d = torch.cat([d_att, d_up], dim=1)
            d = self.up_convs[i](d)
            decoder_features.append(d)

            if self.deep_supervision and i < 3:
                ds_outputs.append(F.interpolate(
                    [self.ds_out1, self.ds_out2, self.ds_out3][i](d),
                    size=x1.shape[2:], mode='trilinear', align_corners=False
                ))

        logits = self.out(decoder_features[-1])

        if self.deep_supervision and self.training:
            return [logits] + ds_outputs
        return logits
