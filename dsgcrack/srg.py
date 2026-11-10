"""Scale-Aware Residual Gating (SRG)."""
import torch
from torch import nn
from torch.nn import functional as F

class SRG(nn.Module):
    """Keep the shallow gate; learn scale-specific corrections to its logits."""

    def __init__(self, dim):
        super().__init__()
        self.reference = nn.Conv2d(dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # Do not change the initialization of downstream, unchanged modules.
        with torch.random.fork_rng(devices=[]):
            self.corrections = nn.ModuleList([
                nn.Conv2d(2 * dim, 1, kernel_size=1) for _ in range(3)
            ])
        self.alpha = nn.Parameter(torch.zeros(3))

    def forward(self, x1, x2, x3, x4):
        size = x1.shape[-2:]
        base_logits = self.reference(x1)
        outputs = []
        # Three feature scales, not a loop over pixels or samples.
        for index, feature in enumerate((x2, x3, x4)):
            feature = F.interpolate(feature, size=size, mode='bilinear',
                                    align_corners=False)
            correction = self.corrections[index](torch.cat([x1, feature], dim=1))
            # Match autocast dtype: the FP32 parameter must not upcast the gate.
            strength = self.alpha[index].tanh().to(dtype=correction.dtype)
            gate = self.sigmoid(base_logits + strength * correction)
            outputs.append(feature * gate)
        return tuple(outputs)
