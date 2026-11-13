from .model import SwinV2_Tyche

IMAGE_SIZE = (120, 240)
CHANNELS = 69


def build_model():
    return SwinV2_Tyche(
        img_resolution=IMAGE_SIZE,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        window_size=(10, 10),
        shift_size=(5, 5),
        patch_size=(2, 2),
        depth=12,
        dim=1024,
        heads=12,
        flash=True,
        logvar=False,
    )
