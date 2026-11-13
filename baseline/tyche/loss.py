"""Original Tyche MeanFlow pretraining objective, without DDP-specific access."""

import numpy as np
import torch
from torch import nn


class MeanFlowLoss(nn.Module):
    def __init__(self, flow_ratio=0.5, time_dist="lognorm", time_mu=-0.4, time_sigma=1.0):
        super().__init__()
        if not 0 <= flow_ratio <= 1 or time_dist not in ("lognorm", "uniform") or time_sigma <= 0:
            raise ValueError("Invalid MeanFlow time distribution")
        self.flow_ratio, self.time_dist = flow_ratio, time_dist
        self.time_mu, self.time_sigma = time_mu, time_sigma

    def sample_t_r(self, batch_size, device):
        if self.time_dist == "uniform":
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        else:
            normal = (
                np.random.randn(batch_size, 2).astype(np.float32) * self.time_sigma + self.time_mu
            )
            samples = 1 / (1 + np.exp(-normal))
        t = np.maximum(samples[:, 0], samples[:, 1])
        r = np.minimum(samples[:, 0], samples[:, 1])
        chosen = np.random.permutation(batch_size)[: int(self.flow_ratio * batch_size)]
        r[chosen] = t[chosen]
        return torch.tensor(t, device=device), torch.tensor(r, device=device)

    def forward(self, net, target, condition):
        t, r = self.sample_t_r(target.shape[0], target.device)
        t_view = t.reshape(-1, 1, 1, 1).detach().clone()
        r_view = r.reshape(-1, 1, 1, 1).detach().clone()
        noise = torch.randn_like(target)
        z = (1 - t_view) * target + t_view * noise
        v = noise - target

        def model_fn(z_in, t_in, r_in):
            # The bare backbone accepts jvp; the inference wrapper does not.
            # jvp=True selects ordinary attention, needed for double backward.
            return net(z_in, t_in, condition, r=r_in, jvp=True)

        u, du_dt = torch.autograd.functional.jvp(
            model_fn,
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=self.training,
        )
        target_velocity = (v - (t_view - r_view) * du_dt).detach()
        error_sq = (u - target_velocity).square().mean(dim=(1, 2, 3))
        weight = (error_sq + 1e-3).pow(-0.5).detach()
        return (weight * error_sq).mean()
