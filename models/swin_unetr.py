import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=(8, 8, 8), qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8, 8), mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinUNETR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        window_size: Tuple[int, int, int] = (8, 8, 8),
        embed_dim: int = 48,
        depths: List[int] = (2, 2, 2, 2),
        num_heads: List[int] = (3, 6, 12, 24),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=2, stride=2)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_dim = embed_dim

        for i in range(len(depths)):
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(
                    SwinTransformerBlock3D(current_dim, num_heads[i], window_size, mlp_ratio, qkv_bias)
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            if i < len(depths) - 1:
                self.downsamples.append(PatchMerging3D(current_dim, current_dim * 2))
                current_dim *= 2

        self.bottleneck_dim = current_dim

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(current_dim, current_dim // 2, 2, stride=2),
            nn.Conv3d(current_dim // 2 + current_dim // 2, current_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(current_dim // 2),
            nn.ReLU(inplace=True),
        )
        current_dim = current_dim // 2

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(current_dim, current_dim // 2, 2, stride=2),
        )
        self.decoder3_conv = nn.Sequential(
            nn.Conv3d(current_dim // 2 + current_dim // 4, current_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(current_dim // 2),
            nn.ReLU(inplace=True),
        )
        current_dim = current_dim // 2

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(current_dim, current_dim // 2, 2, stride=2),
        )
        self.decoder2_conv = nn.Sequential(
            nn.Conv3d(current_dim // 2 + embed_dim, current_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(current_dim // 2),
            nn.ReLU(inplace=True),
        )
        current_dim = current_dim // 2

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(current_dim, current_dim, 2, stride=2),
            nn.Conv3d(current_dim, current_dim, 3, padding=1, bias=False),
            nn.BatchNorm3d(current_dim),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv3d(current_dim, out_channels, 1)

        if deep_supervision:
            self.ds_out3 = nn.Conv3d(current_dim * 2, out_channels, 1)
            self.ds_out2 = nn.Conv3d(current_dim, out_channels, 1)

    def forward(self, x):
        x0 = self.patch_embed(x)
        B, C, D, H, W = x0.shape

        features = [x0]

        for i in range(len(self.stages)):
            stage = self.stages[i]
            B, C, D, H, W = x0.shape
            x_flat = x0.flatten(2).transpose(1, 2)
            x_flat = stage(x_flat)
            x0 = x_flat.transpose(1, 2).reshape(B, C, D, H, W)
            features.append(x0)

            if i < len(self.downsamples):
                B, C, D, H, W = x0.shape
                x_flat = x0.flatten(2).transpose(1, 2)
                x_flat = self.downsamples[i](x_flat)
                C_new = x_flat.shape[-1]
                x0 = x_flat.transpose(1, 2).reshape(B, C_new, D // 2, H // 2, W // 2)

        x = self.decoder4(x0)
        x = self.decoder3(x)
        x = self.decoder3_conv(x)
        x = self.decoder2(x)
        x = self.decoder2_conv(x)
        x = self.decoder1(x)
        logits = self.out(x)

        return logits
