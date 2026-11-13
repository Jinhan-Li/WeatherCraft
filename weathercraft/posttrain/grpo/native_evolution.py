import torch
from torch import nn
import math
from timm.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from .native_attention import ConvBlock, SelfAttentionBlock


def UniformerSubBlock(
    embed_dims, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_value=1e-06, block_type="Conv"
):
    assert block_type in ["Conv", "MHSA"]
    if block_type == "Conv":
        return SelfAttentionBlock(
            dim=embed_dims,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value,
        )
    else:
        return SelfAttentionBlock(
            dim=embed_dims,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value,
        )


class SpatioTemporalEvolutionBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        input_resolution=None,
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        layer_i=0,
    ):
        super(SpatioTemporalEvolutionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        block_type = "MHSA" if in_channels == out_channels and layer_i > 0 else "Conv"
        self.block = UniformerSubBlock(
            in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, block_type=block_type
        )
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        z = self.block(x)
        if self.in_channels != self.out_channels:
            z = self.reduction(z)
        return z


class SpatioTemporalEvolution(nn.Module):

    def __init__(
        self,
        channel_in,
        channel_hid,
        N2,
        input_resolution=None,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
    ):
        super(SpatioTemporalEvolution, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(0.01, drop_path, self.N2)]
        evolution_layers = [
            SpatioTemporalEvolutionBlock(
                channel_in,
                channel_hid,
                input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=dpr[0],
                layer_i=0,
            )
        ]
        for i in range(1, N2 - 1):
            evolution_layers.append(
                SpatioTemporalEvolutionBlock(
                    channel_hid,
                    channel_hid,
                    input_resolution,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[i],
                    layer_i=i,
                )
            )
        evolution_layers.append(
            SpatioTemporalEvolutionBlock(
                channel_hid,
                channel_in,
                input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                layer_i=N2 - 1,
            )
        )
        self.enc = nn.Sequential(*evolution_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        y = z.reshape(B, T, C, H, W)
        return y
