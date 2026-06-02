import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.deconv(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNETR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        features: List[int] = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.deep_supervision = deep_supervision

        features = features or [32, 64, 128, 256, 512]

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        patches_per_dim = [max(img_size[i] // patch_size, 1) for i in range(3)]
        self.pos_embed = nn.Parameter(torch.zeros(1, patches_per_dim[0] * patches_per_dim[1] * patches_per_dim[2], embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, qkv_bias=True)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.decoder0 = nn.Sequential(
            nn.Conv3d(in_channels, features[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = DeconvBlock(embed_dim, features[1], features[0])
        self.decoder2 = DeconvBlock(embed_dim, features[2], features[1])
        self.decoder3 = DeconvBlock(embed_dim, features[3], features[2])
        self.decoder4 = DeconvBlock(embed_dim, features[4], features[3])

        self.out = nn.Conv3d(features[4], out_channels, 1)

        if deep_supervision:
            self.ds_out1 = nn.Conv3d(features[3], out_channels, 1)
            self.ds_out2 = nn.Conv3d(features[2], out_channels, 1)

    def forward(self, x):
        B = x.shape[0]

        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pd = max(D // self.patch_size, 1)
        ph = max(H // self.patch_size, 1)
        pw = max(W // self.patch_size, 1)

        x0 = self.decoder0(x)

        x_patch = self.patch_embed(x)
        num_patches = x_patch.shape[1]
        x_patch = x_patch + self.pos_embed[:, :num_patches, :]

        hidden_states = []
        for blk in self.blocks:
            x_patch = blk(x_patch)
            hidden_states.append(x_patch)

        x_patch = self.norm(x_patch)

        x_patch = rearrange(x_patch, 'b (d h w) c -> b c d h w', d=pd, h=ph)

        x1 = self.decoder1(x_patch, x0)
        x2 = self.decoder2(x_patch, x1)
        x3 = self.decoder3(x_patch, x2)
        x4 = self.decoder4(x_patch, x3)

        logits = self.out(x4)

        if self.deep_supervision and self.training:
            ds1 = self.ds_out1(x3)
            ds2 = self.ds_out2(x2)
            return [logits, ds1, ds2]

        return logits
