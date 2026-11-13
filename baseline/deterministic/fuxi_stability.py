"""Numerical boundary fix for padded SwinV2 tokens; no shared-package edits.

The pinned FuXi constructs a bare timm stage. Apply timm's native residual
initialization, and bound cosine-normalization's zero-vector derivative using
a fixed 1e-6 floor. Ordinary vectors and all layer/parameter shapes are retained.
"""

import types
import torch.nn.functional as functional

COSINE_EPS = 1e-6


def stable_normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    return functional.normalize(input, p=p, dim=dim, eps=max(eps, COSINE_EPS), out=out)


class _FunctionalView:
    normalize = staticmethod(stable_normalize)

    def __getattr__(self, name):
        return getattr(functional, name)


def stabilize_fuxi(net):
    stage = net.u_transformer.layer
    stage._init_respostnorm()
    for block in stage.blocks:
        attention = block.attn
        original = type(attention).forward
        # Reuse the installed timm forward bytecode, with only a private F view.
        # Neither torch.nn.functional nor the shared timm class is modified.
        namespace = dict(original.__globals__)
        namespace["F"] = _FunctionalView()
        forward = types.FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        forward.__kwdefaults__ = original.__kwdefaults__
        attention.forward = types.MethodType(forward, attention)
    return net
