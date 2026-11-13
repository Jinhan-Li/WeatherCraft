"""Minimal two-observed-day adapters for three pinned upstream models."""

import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent


def upstream(name):
    spec = importlib.util.spec_from_file_location(
        "turbl1_" + name, HERE / "upstream/turbl1" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimVPForecast(nn.Module):
    def __init__(self, channels, height, width, options):
        super().__init__()
        self.channels = channels
        self.net = upstream("simvp").SimVP(
            shape_in=(2, channels, height, width),
            output_dim=channels,
            hid_S=options["hidden_spatial"],
            hid_T=options["hidden_temporal"],
            N_S=options["spatial_depth"],
            N_T=options["temporal_depth"],
        )

    def forward(self, history):
        b, _, h, w = history.shape
        sequence = history.reshape(b, 2, self.channels, h, w)
        # One next-day objective. The last output slot is the next-day readout;
        # no second future target, reconstruction loss or future input is used.
        return self.net(sequence)[:, -1]


class ConvLSTMForecast(nn.Module):
    def __init__(self, channels, options):
        super().__init__()
        self.channels = channels
        hidden = options["hidden_channels"]
        self.net = upstream("convlstm").ConvLSTM(
            input_dim=channels,
            hidden_dim=hidden,
            kernel_size=(3, 3),
            num_layers=len(hidden),
            batch_first=True,
            return_all_layers=False,
        )
        self.readout = nn.Conv2d(hidden[-1], channels, kernel_size=1)

    def forward(self, history):
        b, _, h, w = history.shape
        output, _ = self.net(history.reshape(b, 2, self.channels, h, w))
        # Hidden states start at zero for each two-day forecasting window.
        return self.readout(output[:, -1])


def sfno_module():
    directory = str(HERE / "vendor_sfno")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from torch_harmonics.examples.models import sfno

    return sfno


def real_parameter_spectral_class(base_class):
    class RealParameterSpectralConv(base_class):
        """Complex weights stored as real pairs for NCCL and fused AdamW.

        The upstream contraction, SHT and inverse SHT remain unchanged.
        view_as_complex is differentiable and retains both complex components.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            parameter = self._parameters.pop("weight")
            self.register_parameter(
                "weight_ri", nn.Parameter(torch.view_as_real(parameter.detach()).clone())
            )

        def __getattr__(self, name):
            if name == "weight" and "weight_ri" in self._parameters:
                return torch.view_as_complex(self._parameters["weight_ri"])
            return super().__getattr__(name)

    return RealParameterSpectralConv


class SphericalForecast(nn.Module):
    def __init__(self, channels, height, width, options, latitudes):
        super().__init__()
        self.flip_latitude = bool(latitudes[0] < latitudes[-1])
        expected = (
            -90.0 + np.arange(height) * (180.0 / height)
            if self.flip_latitude
            else 90.0 - np.arange(height) * (180.0 / height)
        )
        np.testing.assert_allclose(latitudes, expected, rtol=0, atol=1e-5)
        module = sfno_module()
        original = module.SpectralConvS2
        module.SpectralConvS2 = real_parameter_spectral_class(original)
        try:
            self.net = module.SphericalFourierNeuralOperatorNet(
                img_size=(height + 1, width),
                in_chans=2 * channels,
                out_chans=channels,
                embed_dim=options["width"],
                num_layers=options["depth"],
                scale_factor=options["scale_factor"],
                operator_type="driscoll-healy",
                grid="equiangular",
                grid_internal="legendre-gauss",
                activation_function="gelu",
                normalization_layer="none",
                use_mlp=True,
                mlp_ratio=2.0,
                big_skip=True,
                pos_embed="latlon",
            )
        finally:
            module.SpectralConvS2 = original
        assert all(not p.is_complex() for p in self.parameters())

    def forward(self, history):
        # Read the actual latitude coordinates. This project's crop is ascending
        # -90 ... +88.5, whereas official SHT is north-to-south including both
        # poles. Complete the absent pole using the input last-ring zonal mean,
        # orient for SHT, then restore/crop before loss, scores and own feedback.
        pole = history[:, :, -1:, :].mean(-1, keepdim=True).expand(-1, -1, -1, history.shape[-1])
        sphere = torch.cat((history, pole), dim=-2)
        if self.flip_latitude:
            sphere = sphere.flip(-2)
        prediction = self.net(sphere)
        if self.flip_latitude:
            prediction = prediction.flip(-2)
        return prediction[:, :, :-1, :]


class TwoDayForecast(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        model = cfg["model"]
        height, width = model["img_size"]
        channels = len(dataset.variable_names)
        assert channels == 69
        self.shape = (2 * channels, height, width)
        self.activation_checkpointing = model["activation_checkpointing"]
        if model["name"] == "simvp":
            self.net = SimVPForecast(channels, height, width, model)
        elif model["name"] == "convlstm":
            self.net = ConvLSTMForecast(channels, model)
        elif model["name"] == "sfno":
            self.net = SphericalForecast(channels, height, width, model, dataset.latitudes.numpy())
        else:
            raise ValueError(model["name"])

    def forward(self, history):
        assert tuple(history.shape[1:]) == self.shape
        if self.activation_checkpointing and self.training:
            result = checkpoint(self.net, history, use_reentrant=False)
        else:
            result = self.net(history)
        assert tuple(result.shape) == (history.shape[0], 69, *self.shape[1:])
        return result


def build_model(cfg, dataset, device):
    return TwoDayForecast(cfg, dataset).to(device)
