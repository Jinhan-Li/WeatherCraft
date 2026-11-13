import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import model as original_model
from .vendor.Fuxi import Fuxi
from .vendor.Fengwu import Fengwu
from .fuxi_stability import stabilize_fuxi


class HistoryFuXi(Fuxi):
    def __init__(self, h, w):
        super().__init__(in_shape=(2, 69, h, w), patch_size=(2, 4, 4))
        stabilize_fuxi(self)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 5 and x.shape[1:3] == (2, 69)
        x = x.permute(0, 2, 1, 3, 4)
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        x = self.u_transformer(x)
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)

        return x


class HistoryFengWu(Fengwu):
    def __init__(self, h, w):
        super().__init__(in_shape=(2, 69, h, w))
        for name in ("surface", "z", "r", "u", "v", "t"):
            enc = getattr(self, "encoder_" + name)
            old = enc.patchembed2d.proj
            enc.in_chans *= 2
            enc.patchembed2d.proj = nn.Conv2d(
                old.in_channels * 2,
                old.out_channels,
                old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None,
            )

    def forward(self, x):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """
        surface = torch.cat((x[:, 65:69], x[:, 134:138]), dim=1)
        u = torch.cat((x[:, 39:52], x[:, 108:121]), dim=1)  # 从第4通道开始，每隔5个通道提取u
        v = torch.cat((x[:, 52:65], x[:, 121:134]), dim=1)  # 从第5通道开始，每隔5个通道提取v
        t = torch.cat((x[:, 26:39], x[:, 95:108]), dim=1)  # 从第6通道开始，每隔5个通道提取t
        r = torch.cat((x[:, 13:26], x[:, 82:95]), dim=1)  # 从第7通道开始，每隔5个通道提取r
        z = torch.cat((x[:, 0:13], x[:, 69:82]), dim=1)  # 从第8通道开始，每隔5个通道提取z
        # print(f"surface shape: {surface.shape}")
        # print(f"u shape: {u.shape}")
        # print(f"v shape: {v.shape}")
        # print(f"t shape: {t.shape}")
        # print(f"r shape: {r.shape}")
        # print(f"z shape: {z.shape}")

        surface, skip_surface = self.encoder_surface(surface)
        z, skip_z = self.encoder_z(z)
        r, skip_r = self.encoder_r(r)
        u, skip_u = self.encoder_u(u)
        v, skip_v = self.encoder_v(v)
        t, skip_t = self.encoder_t(t)

        x = torch.cat(
            [
                surface.unsqueeze(1),
                z.unsqueeze(1),
                r.unsqueeze(1),
                u.unsqueeze(1),
                v.unsqueeze(1),
                t.unsqueeze(1),
            ],
            dim=1,
        )
        B, PL, L_SIZE, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.fuser(x)

        x = x.reshape(B, PL, L_SIZE, C)
        surface, z, r, u, v, t = (
            x[:, 0, :, :],
            x[:, 1, :, :],
            x[:, 2, :, :],
            x[:, 3, :, :],
            x[:, 4, :, :],
            x[:, 5, :, :],
        )

        surface = self.decoder_surface(surface, skip_surface)
        z = self.decoder_z(z, skip_z)
        r = self.decoder_r(r, skip_r)
        u = self.decoder_u(u, skip_u)
        v = self.decoder_v(v, skip_v)
        t = self.decoder_t(t, skip_t)

        # stached = torch.stack((z,r,u,v,t),dim=1)
        # reshaped = stached.permute(0,1,3,4,2).reshape(B,13*5,180,360)
        # reshaped = stached.permute(0,1,3,4,2).reshape(B,13*5,180,360)
        reshaped = torch.cat((z, r, t, u, v), dim=1)
        final_output = torch.cat([reshaped, surface], dim=1)
        return final_output


class TwoDayModel(nn.Module):
    def __init__(self, c, days):
        super().__init__()
        names = days.variable_names
        assert list(names) == original_model.CHANNEL_NAMES
        self.name = c["model"]["name"]
        h, w = c["model"]["img_size"]
        order = torch.tensor([names.index(n) for n in original_model.UPSTREAM_NAMES])
        self.register_buffer("input_order", torch.cat((order, order + 69)))
        self.register_buffer(
            "output_order", torch.tensor([original_model.UPSTREAM_NAMES.index(n) for n in names])
        )
        if self.name == "fuxi":
            self.net = HistoryFuXi(h, w)
        elif self.name == "fengwu":
            self.net = HistoryFengWu(h, w)
        else:
            self.net = original_model.graph_class("graphcast")(
                input_res=(h, w),
                input_dim_grid_nodes=138,
                output_dim_grid_nodes=69,
                recompute_activation=False,
                grid_latitudes=days.latitudes.numpy(),
                grid_longitudes=np.load(c["data"]["longitude_path"]).reshape(-1),
            )
        if c["model"]["activation_checkpointing"]:
            if self.name == "graphcast":
                self.net.set_checkpoint_encoder(True)
                self.net.set_checkpoint_processor(7)
                self.net.set_checkpoint_decoder(True)
            else:
                original_model.checkpoint_blocks(self.net)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.net.to(*args, **kwargs)
        return self

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1:] == (138, 120, 240)
        x = x.index_select(1, self.input_order)
        if self.name == "fuxi":
            x = x.reshape(x.shape[0], 2, 69, 120, 240)
        return self.net(x).index_select(1, self.output_order)
