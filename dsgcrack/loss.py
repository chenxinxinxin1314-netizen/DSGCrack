"""The full method's original 0.87 BCE-with-logits + 0.13 Dice objective."""
from torch import nn


class SegmentationLoss(nn.Module):
    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        probability = logits.sigmoid()
        axes = (-2, -1)
        tp = (probability*target).sum(axes)
        fp = (probability*(1-target)).sum(axes)
        fn = ((1-probability)*target).sum(axes)
        dice = 1-((2*tp+1.)/(2*tp+fp+fn+1.)).mean()
        return .87*bce+.13*dice
