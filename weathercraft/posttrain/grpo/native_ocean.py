import torch
from torch import nn
import math
from timm.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from .native_evolution import SpatioTemporalEvolution


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        transpose=False,
        act_norm=False,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2,
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvDynamicsLayer(nn.Module):

    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvDynamicsLayer, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm,
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class AtmosphericEncoder(nn.Module):

    def __init__(self, C_in, spatial_hidden_dim, num_spatial_layers):
        super(AtmosphericEncoder, self).__init__()
        strides = stride_generator(num_spatial_layers)
        self.enc = nn.Sequential(
            ConvDynamicsLayer(C_in, spatial_hidden_dim, stride=strides[0]),
            *[
                ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s)
                for s in strides[1:]
            ]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return (latent, enc1)


class AtmosphericDecoder(nn.Module):

    def __init__(self, spatial_hidden_dim, C_out, num_spatial_layers):
        super(AtmosphericDecoder, self).__init__()
        strides = stride_generator(num_spatial_layers, reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s, transpose=True)
                for s in strides[:-1]
            ],
            ConvDynamicsLayer(
                2 * spatial_hidden_dim, spatial_hidden_dim, stride=strides[-1], transpose=True
            )
        )
        self.readout = nn.Conv2d(spatial_hidden_dim, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Triton(nn.Module):

    def __init__(
        self,
        params,
        input_channel=97,
        spatial_hidden_dim=256,
        output_channels=93,
        temporal_hidden_dim=512,
        num_spatial_layers=4,
        num_temporal_layers=8,
        in_time_seq_length=10,
        out_time_seq_length=10,
        gradient_checkpointing=False,
    ):
        super(Triton, self).__init__()
        C = input_channel
        T = 1
        self.output_dim = output_channels
        self.input_time_seq_length = in_time_seq_length
        self.output_time_seq_length = out_time_seq_length
        self.atmospheric_encoder = AtmosphericEncoder(C, spatial_hidden_dim, num_spatial_layers)
        self.temporal_evolution = SpatioTemporalEvolution(
            T * spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_layers,
            input_resolution=None,
            mlp_ratio=4.0,
            drop_path=0.1,
        )
        self.atmospheric_decoder = AtmosphericDecoder(
            spatial_hidden_dim, self.output_dim, num_spatial_layers
        )
        self.gradient_checkpointing = gradient_checkpointing

    def custom_forward(self, input_state):
        input_state = input_state.unsqueeze(1)
        batch_size, temporal_length, channels, height, width = input_state.shape
        reshaped_input = input_state.view(batch_size * temporal_length, channels, height, width)
        encoded_features, skip_connection = self.atmospheric_encoder(reshaped_input)
        _, encoded_channels, encoded_height, encoded_width = encoded_features.shape
        encoded_features = encoded_features.view(
            batch_size, temporal_length, encoded_channels, encoded_height, encoded_width
        )
        temporal_bias = encoded_features
        temporal_hidden = self.temporal_evolution(temporal_bias)
        reshaped_hidden = temporal_hidden.view(
            batch_size * temporal_length, encoded_channels, encoded_height, encoded_width
        )
        decoded_output = self.atmospheric_decoder(reshaped_hidden, skip_connection)
        final_output = decoded_output.view(batch_size, temporal_length, -1, height, width)
        final_output = final_output.squeeze(1)
        return final_output

    def forward(self, input_state):
        if self.gradient_checkpointing and self.training:
            outputs = torch.utils.checkpoint.checkpoint(
                self.custom_forward, input_state, use_reentrant=False
            )
        else:
            outputs = self.custom_forward(input_state)
        return outputs
