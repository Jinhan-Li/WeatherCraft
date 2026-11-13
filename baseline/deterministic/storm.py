import torch
from torch import nn
from .model import CHANNEL_NAMES


class TwoDayModel(nn.Module):

    def __init__(self, c, days):
        super().__init__()
        from .storm_core import STORM

        assert list(days.variable_names) == CHANNEL_NAMES
        self.net = STORM(
            n_var=69,
            lon=120,
            lat=240,
            seq_len=2,
            pred_len=1,
            num_layers=c["model"]["num_layers"],
            d_model=c["model"]["d_model"],
            scales_number_M=c["model"]["scales_number_M"],
        )
        self.net.encoder.activation_checkpointing = c["model"]["activation_checkpointing"]
        for layer in self.net.encoder.layers:
            layer.cross_message[-1].requires_grad_(False)

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1:] == (138, 120, 240)
        out = self.net(x.reshape(x.shape[0], 2, 69, 120, 240))
        assert out.shape == (x.shape[0], 1, 69, 120, 240)
        return out[:, 0]
