"""The four-stage TransMixer encoder, with one complete DCG block per stage."""
import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from .backbone.transmixer import TransMixer
from .dcg import DCG
from .layers import LayerNorm2D, Stem, PatchMerging


class EncoderStage(nn.Module):
    def __init__(self, channels, state_dim):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(
            TransMixer(hidden_dim=channels, ssm_d_state=state_dim,
                       mlp_ratio=2., channel_first=True),
            DCG(in_dim=channels, cell_size=(8, 8)))])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TransMixerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dimensions = [16, 32, 64, 128]
        states = [8, 8, 16, 16]
        self.stem = Stem(in_dim=3, dim=dimensions[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (channels, state) in enumerate(zip(dimensions, states)):
            self.stages.append(EncoderStage(channels, state))
            if index < len(dimensions)-1:
                self.downsamples.append(PatchMerging(channels, dimensions[index+1]))
        self.apply(self._init_weights)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, LayerNorm2D):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return features
