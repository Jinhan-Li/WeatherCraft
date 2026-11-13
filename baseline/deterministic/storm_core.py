import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super(ConvMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(ConvBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===== Patch Embed 模块 =====
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_chans * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.proj_back = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        # 切成 patch tokens
        patches = self.to_patch_embedding(x)  # [B, num_patches, embed_dim]
        return patches

    def reverse(self, patches, H, W):
        # 把 patch token 还原回 feature map
        patch_dim = self.proj_back.out_features
        patch_size = self.patch_size
        patches = self.proj_back(patches)  # [B, num_patches, patch_dim]
        x = Rearrange(
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=H // patch_size,
            w=W // patch_size,
            p1=patch_size,
            p2=patch_size,
        )(patches)
        return x


# ===== Self-Attention Block with Patch =====
class SpatialEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        patch_size=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_value=1e-6,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(dim, dim, patch_size)

        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.STORM_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.STORM_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Pad only the per-scale patch boundary; crop back before cross-scale messaging.
        ph, pw = (-H) % self.patch_embed.patch_size, (-W) % self.patch_embed.patch_size
        x = F.pad(x, (0, pw, 0, ph), mode="replicate") if ph or pw else x
        patches = self.patch_embed(x)  # [B, num_patches, dim]

        # attention + mlp
        patches = patches + self.drop_path(
            self.STORM_1
            * self.attn(
                self.norm1(patches), self.norm1(patches), self.norm1(patches), need_weights=False
            )[0]
        )
        patches = patches + self.drop_path(self.STORM_2 * self.mlp(self.norm2(patches)))

        # reverse to feature map
        x = self.patch_embed.reverse(patches, H + ph, W + pw)[..., :H, :W]  # [B, C, H, W]
        return x


def stride_generator(reverse=False):
    strides = [2, 1] * 10
    if reverse:
        return list(reversed(strides))
    else:
        return strides


class MultiScaleSampler(nn.Module):
    def __init__(self, C, d_model, M):
        super().__init__()
        self.M = M
        self.C = C
        self.d_model = d_model
        strides = stride_generator()
        self.enc_scale0 = ConvDynamicsLayer(self.C, self.d_model, stride=1)
        self.convs = nn.ModuleDict()
        for m in range(M):
            self.convs[f"m{m+1}"] = nn.Sequential(
                *[
                    ConvDynamicsLayer(self.d_model, self.d_model, stride=s)
                    for s in strides[2 * m : 2 * m + 2]
                ]
            )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        xt_flat = x.view(B * T, C, H, W)
        enc = self.enc_scale0(xt_flat)
        outs = [enc]
        for m in range(self.M):
            conv = self.convs[f"m{m+1}"]
            enc = conv(enc)
            outs.append(enc)
        #     print('multi_scale_sampler | ',enc.shape)
        return outs


# class ChannelEmbedding(nn.Module):
#     def __init__(self, C, d_model, M):
#         super().__init__()
#         self.linears = nn.ModuleList([nn.Linear(C, d_model) for _ in range(M+1)])
#         self.d_model = d_model
#     def forward(self, multiscale_x):
#         outs = []
#         for idx, x in enumerate(multiscale_x):
#             B, T, C, Hs, Ws = x.shape
#             x_perm = x.permute(0,1,3,4,2).contiguous()  # [B,T,Hs,Ws,C]
#             x_flat = x_perm.view(-1, C)
#             proj = self.linears[idx](x_flat)
#             proj = proj.view(B, T, Hs, Ws, self.d_model)
#             proj = proj.permute(0,1,4,2,3).contiguous()  # [B,T,d_model,Hs,Ws]
#             outs.append(proj)
#         return outs


class SpatialEncoderPerScale(nn.Module):
    def __init__(self, d_model, Hs, Ws, drop_path=0):
        super().__init__()
        self.block = SpatialEncoderBlock(
            dim=d_model, num_heads=8, mlp_ratio=4, qkv_bias=True, drop=0, drop_path=drop_path
        )

    def forward(self, x):
        # x: [B*T, D, Hs, Ws]
        B, D, H, W = x.shape
        z = self.block(x)
        return z


class TemporalEncoderPerScale(nn.Module):
    """
    Temporal encoding per scale:
      - MLP across T (Linear T->T, GELU, Linear T->T) applied for each (B,D,H,W)
    """

    def __init__(self, T, drop_path=0.1):
        super().__init__()
        self.T = T
        self.mlp_t = nn.Sequential(
            nn.Linear(T, 4 * T), nn.GELU(), nn.Linear(4 * T, T), DropPath(drop_path)
        )

    def forward(self, x):
        # x: [B, T, D, H, W]

        _, D, H, W = x.shape
        x = x.view(-1, self.T, D, H, W)
        B, T, D, H, W = x.shape
        # assert T == self.T, f"Expected T={self.T} got T={T}"
        xt = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, T]
        xt_flat = xt.view(-1, self.T)  # [B*D*H*W, T]
        xt_flat = self.mlp_t(xt_flat)  # [.., T]
        xt = xt_flat.view(B, D, H, W, T).permute(0, 4, 1, 2, 3).contiguous()  # [B, T, D, H, W]
        xt = xt.view(B * T, D, H, W).contiguous()
        return xt


class SpatioTemporalLayerPerScale(nn.Module):
    def __init__(self, d_model, scales_hw, T, drop_path=0.1):
        super().__init__()
        # per-scale spatial and temporal blocks
        self.spatial_blocks = nn.ModuleList(
            [SpatialEncoderPerScale(d_model, Hs, Ws, drop_path=drop_path) for (Hs, Ws) in scales_hw]
        )
        self.temporal_blocks = nn.ModuleList(
            [TemporalEncoderPerScale(T, drop_path=drop_path) for _ in scales_hw]
        )
        self.cross_message = nn.ModuleList(
            [ConvDynamicsLayer(d_model, d_model, stride=2) for _ in scales_hw]
        )

    def forward(self, multiscale_x):
        outs = []

        fine_sx_list = []
        for i, x in enumerate(multiscale_x):

            sx = self.spatial_blocks[i](x)
            fine_sx_list.append(sx)
            if i > 0:
                #    print(fine_sx_list[i-1].shape,' | ',sx.shape)
                sx = sx + self.cross_message[i - 1](fine_sx_list[i - 1])
            tx = self.temporal_blocks[i](sx)
            outs.append(tx + x)  # residual connection between each encoding layer
        return outs


class MultiLayerEncoder(nn.Module):
    def __init__(self, d_model, N, scales_hw, T, drop_path=0.1):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, N)]
        self.layers = nn.ModuleList(
            [SpatioTemporalLayerPerScale(d_model, scales_hw, T, drop_path=dpr[n]) for n in range(N)]
        )

    def forward(self, multiscale_x):
        x = multiscale_x
        for layer in self.layers:
            if (
                getattr(self, "activation_checkpointing", False)
                and self.training
                and torch.is_grad_enabled()
            ):
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = layer(x)
        return x


class MultiScalePredictor(nn.Module):
    def __init__(self, d_model, C, M, H, W, seq_len, pred_len):
        super().__init__()
        self.M = M
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.temporal_proj = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(M + 1)])
        self.deconvs = nn.ModuleList()
        strides = stride_generator(reverse=True)

        self.skips = nn.ModuleList()
        for m in range(M + 1):
            k = 2**m
            if m == 0:
                # self.deconvs.append(nn.ConvTranspose2d(d_model, d_model, kernel_size=1, stride=1))
                self.deconvs.append(ConvDynamicsLayer(d_model, d_model, stride=1, transpose=True))
            else:
                self.deconvs.append(
                    nn.Sequential(
                        *[
                            ConvDynamicsLayer(d_model, d_model, stride=s, transpose=True)
                            for s in strides[: 2 * m]
                        ]
                    )
                )
                # self.deconvs.append(nn.ConvTranspose2d(d_model, d_model, kernel_size=k, stride=k))
            self.skips.append(ConvDynamicsLayer(2 * d_model, d_model, stride=1))
        self.out_proj = nn.Linear(d_model, C)

    def forward(self, encoded_multiscale, res):
        B = encoded_multiscale[0].shape[0]
        total = None
        for m, enc in enumerate(encoded_multiscale):
            enc = self.skips[m](torch.cat([res[m], enc], dim=1))
            _, D, Hs, Ws = enc.shape
            enc = enc.view(-1, self.seq_len, D, Hs, Ws)
            B, T, D, Hs, Ws = enc.shape
            enc_perm = enc.permute(0, 2, 3, 4, 1).contiguous()  # [B,d,Hs,Ws,T]
            rep = self.temporal_proj[m](enc_perm)  # [B,Hs,Ws,D,L]
            rep_ch = rep.permute(0, 4, 1, 2, 3).contiguous().view(B * self.pred_len, D, Hs, Ws)
            deconv = self.deconvs[m](rep_ch)
            H_up, W_up = deconv.shape[-2], deconv.shape[-1]

            deconv = deconv.view(B, self.pred_len, D, H_up, W_up)
            deconv_perm = deconv.permute(0, 1, 3, 4, 2).contiguous()  # [B,L,H,W,D]
            deconv_flat = deconv_perm.view(-1, D)
            outC = self.out_proj(deconv_flat)
            outC = (
                outC.view(B, self.pred_len, H_up, W_up, -1).permute(0, 1, 4, 2, 3).contiguous()
            )  # [B,L,C,H,W]
            if total is None:
                total = outC
            else:
                # if total.shape[-2:] != outC.shape[-2:]:
                #     outC = F.interpolate(outC.view(B*self.pred_len, outC.shape[2], outC.shape[3], outC.shape[4]),
                #                          size=(total.shape[-2], total.shape[-1]), mode='bilinear', align_corners=False)
                #     outC = outC.view(B, self.pred_len, outC.shape[1], outC.shape[2], outC.shape[3])
                total = total + outC
        return total


class STORM(nn.Module):
    def __init__(
        self, n_var, lon, lat, seq_len, pred_len, num_layers=3, d_model=256, scales_number_M=3
    ):
        super().__init__()
        self.C = n_var
        self.H = lon
        self.W = lat
        self.T = seq_len
        self.L = pred_len
        self.M = scales_number_M
        self.sampler = MultiScaleSampler(C=n_var, d_model=d_model, M=self.M)
        # compute per-scale shapes (Hs, Ws)
        scales_hw = []
        for m in range(self.M + 1):
            k = 2**m
            Hs = self.H // k
            Ws = self.W // k
            scales_hw.append((Hs, Ws))
        #    print(scales_hw)
        # self.embed = ChannelEmbedding(C=n_var, d_model=d_model, M=self.M)
        self.encoder = MultiLayerEncoder(
            d_model=d_model, N=num_layers, scales_hw=scales_hw, T=seq_len
        )
        self.predictor = MultiScalePredictor(
            d_model=d_model,
            C=n_var,
            M=self.M,
            H=self.H,
            W=self.W,
            seq_len=seq_len,
            pred_len=pred_len,
        )

    def forward(self, x):

        scales = self.sampler(x)
        embeds = scales
        encoded = self.encoder(embeds)
        out = self.predictor(encoded, scales)
        # out = out*std+mean
        return out
