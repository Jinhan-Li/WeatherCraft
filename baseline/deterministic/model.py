"""Thin adapters around pinned OneForecast implementations, not architecture replacements."""

import importlib
import types
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
CHANNEL_NAMES = [f"{v}_{p}" for v in ("z", "t", "q", "u", "v") for p in LEVELS] + [
    "t2m",
    "u10",
    "v10",
    "msl",
]
UPSTREAM_NAMES = [f"{v}_{p}" for v in ("z", "q", "t", "u", "v") for p in LEVELS] + [
    "u10",
    "v10",
    "t2m",
    "msl",
]


def checkpoint_blocks(net):
    """Wrap only existing transformer blocks; state keys/parameters/RNG unchanged."""
    for module in net.modules():
        if type(module).__name__ in (
            "Transformer2DBlock",
            "Transformer3DBlock",
            "EarthSpecificBlock",
            "SwinTransformerV2Block",
        ):
            original = module.forward

            def wrapped(self, *args, _forward=original, **kwargs):
                if self.training and torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)

            module.forward = types.MethodType(wrapped, module)


class BatchedGraphMixin:
    """Disjoint graph union. No attention/message edge connects different samples.

    A forward is immediately followed by its backward before changing the batch
    size. This invariant also makes upstream activation-checkpoint closures safe.
    """

    _graph_keys = ("g2m_graph", "mesh_graph", "m2g_graph")
    _feature_keys = ("g2m_edata", "m2g_edata", "mesh_ndata", "mesh_edata")

    def to(self, *args, **kwargs):
        # Restore the single graph before the upstream graph-aware .to().
        if hasattr(self, "_single_geometry"):
            for k, v in self._single_geometry.items():
                setattr(self, k, v)
        super().to(*args, **kwargs)
        self._single_geometry = {k: getattr(self, k) for k in self._graph_keys + self._feature_keys}
        self._batch_geometry = {1: self._single_geometry}
        return self

    def prepare_input(self, invar, expect_partitioned_input=False, global_features_on_rank_0=False):
        import dgl

        assert (
            not self.is_distributed
            and not expect_partitioned_input
            and not global_features_on_rank_0
        )
        b, c, h, w = invar.shape
        assert (c, h, w) == (self.input_dim_grid_nodes, *self.input_res)
        if not hasattr(self, "_single_geometry"):
            self.to(invar.device)
        if b not in self._batch_geometry:
            base = self._single_geometry
            self._batch_geometry[b] = {k: dgl.batch([base[k]] * b) for k in self._graph_keys}
            self._batch_geometry[b].update({k: base[k].repeat(b, 1) for k in self._feature_keys})
        for k, v in self._batch_geometry[b].items():
            setattr(self, k, v)
        return invar.permute(0, 2, 3, 1).reshape(b * h * w, c)

    def prepare_output(
        self, outvar, produce_aggregated_output=True, produce_aggregated_output_on_all_ranks=True
    ):
        h, w = self.input_res
        return outvar.reshape(-1, h, w, self.output_dim_grid_nodes).permute(0, 3, 1, 2).contiguous()


def graph_class(name):
    clsname = {"graphcast": "GraphCast", "oneforecast": "OneForecast"}[name]
    upstream = getattr(importlib.import_module("baseline.deterministic.vendor." + clsname), clsname)
    return type("Batched" + clsname, (BatchedGraphMixin, upstream), {})


class DailyModel(nn.Module):
    def __init__(self, names, cfg, latitudes, longitudes):
        super().__init__()
        assert list(names) == CHANNEL_NAMES, "Unexpected variable order"
        self.name = cfg["name"]
        self.register_buffer("to_upstream", torch.tensor([names.index(n) for n in UPSTREAM_NAMES]))
        self.register_buffer(
            "to_weathercraft", torch.tensor([UPSTREAM_NAMES.index(n) for n in names])
        )
        h, w = cfg["img_size"]
        if self.name == "pangu":
            from .vendor.pangu import Pangu

            self.net = Pangu(
                img_size=(h, w),
                patch_size=(2, 4, 4),
                embed_dim=192,
                num_heads=(6, 12, 12, 6),
                window_size=(2, 6, 12),
                drop_path_rate=0.2,
            )
        elif self.name == "fengwu":
            from .vendor.Fengwu import Fengwu

            self.net = Fengwu(in_shape=(1, 69, h, w))
        elif self.name == "fuxi":
            from .vendor.Fuxi import Fuxi
            from .fuxi_stability import stabilize_fuxi

            self.net = Fuxi(in_shape=(1, 69, h, w))
            stabilize_fuxi(self.net)
        else:
            # Coordinate values must describe the array rows/columns, not a
            # synthetic inclusive latitude linspace or shifted longitude grid.
            cls = graph_class(self.name)
            options = dict(
                input_res=(h, w),
                recompute_activation=False,
                grid_latitudes=np.asarray(latitudes),
                grid_longitudes=np.asarray(longitudes),
            )
            if self.name == "oneforecast":
                options["params"] = None  # unused in the pinned upstream model
            self.net = cls(**options)
        if cfg.get("activation_checkpointing", False):
            if self.name in ("graphcast", "oneforecast"):
                self.net.set_checkpoint_encoder(True)
                self.net.set_checkpoint_processor(7)  # 14 intermediate message-passing layers
                self.net.set_checkpoint_decoder(True)
            else:
                checkpoint_blocks(self.net)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # nn.Module._apply does not call the graph implementation's .to().
        self.net.to(*args, **kwargs)
        return self

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1] == 69
        return self.net(x.index_select(1, self.to_upstream)).index_select(1, self.to_weathercraft)


class TwoDayOneForecast(nn.Module):
    def __init__(self, c, days):
        super().__init__()
        names = days.variable_names
        assert list(names) == CHANNEL_NAMES
        order = torch.tensor([names.index(n) for n in UPSTREAM_NAMES])
        self.register_buffer("input_order", torch.cat((order, order + 69)))
        self.register_buffer("output_order", torch.tensor([UPSTREAM_NAMES.index(n) for n in names]))
        self.net = graph_class("oneforecast")(
            params=None,
            input_res=tuple(c["model"]["img_size"]),
            input_dim_grid_nodes=138,
            output_dim_grid_nodes=69,
            recompute_activation=False,
            grid_latitudes=days.latitudes.numpy(),
            grid_longitudes=np.load(c["data"]["longitude_path"]).reshape(-1),
        )
        if c["model"]["activation_checkpointing"]:
            self.net.set_checkpoint_encoder(True)
            self.net.set_checkpoint_processor(7)
            self.net.set_checkpoint_decoder(True)

    def to(self, *a, **kw):
        super().to(*a, **kw)
        self.net.to(*a, **kw)
        return self

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1] == 138
        return self.net(x.index_select(1, self.input_order)).index_select(1, self.output_order)


def build_model(config, days, device):
    name = config["model"]["name"]
    if name in ("simvp", "convlstm", "sfno"):
        from .adapters import build_model as adapted_model

        return adapted_model(config, days, device)
    if name == "storm":
        from .storm import TwoDayModel

        return TwoDayModel(config, days).to(device)
    if config["model"].get("history_days", 1) == 2 and name in ("fuxi", "fengwu", "graphcast"):
        from .history import TwoDayModel

        return TwoDayModel(config, days).to(device)
    if config["model"].get("history_days", 1) == 2:
        if config["model"]["name"] != "oneforecast":
            raise ValueError("Two-day baseline support is configured for OneForecast")
        return TwoDayOneForecast(config, days).to(device)
    longitude = np.load(config["data"]["longitude_path"]).reshape(-1)
    return DailyModel(days.variable_names, config["model"], days.latitudes.numpy(), longitude).to(
        device
    )
