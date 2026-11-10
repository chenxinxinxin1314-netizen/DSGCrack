"""Public model entry point for DSGCrack with the default DCG + SRG configuration."""
from torch import nn
from .encoder import TransMixerEncoder
from .decoder import MultiScaleDecoder

ARCHITECTURE = "dsgcrack_v1"


class DSGCrack(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        if image_size != 512:
            raise ValueError("This release reproduces the paper's 512x512 input setting.")
        self.backbone = TransMixerEncoder()
        self.decoder = MultiScaleDecoder([16, 32, 64, 128], mid_dim=8, size=(512, 512))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or tuple(image.shape[-2:]) != (512, 512):
            raise ValueError("Expected an RGB tensor with shape [B,3,512,512].")
        return self.decoder(self.backbone(image))


def build_model(image_size=512):
    return DSGCrack(image_size=image_size)
