import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels):
    target = max(1, channels // 8)
    return next(groups for groups in range(target, 0, -1) if channels % groups == 0)


def _check_spatial(x, cell_size, blocks=1):
    if x.ndim != 4:
        raise ValueError(f'Expected [B,C,H,W], got shape={tuple(x.shape)}, cell_size={cell_size}')
    h, w = x.shape[-2:]
    ch, cw = cell_size
    if h <= 0 or w <= 0 or h % (blocks * ch) or w % (blocks * cw):
        raise ValueError(f'H={h}, W={w} must be divisible by {blocks}*cell_size={cell_size} '
                         'for spatial blocks and complete direction cells')


def _sqrt_nonnegative(value):
    """Exact zero with a finite zero subgradient; never differentiate sqrt(0)."""
    value = value.clamp_min(0)
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(value))


class EdgeConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 bias=True):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1,
            bias=bias)
        self.w_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(1, kernel_size), 
            stride=1, 
            padding=(0, kernel_size//2),
            groups=mid_channels)
        
        self.h_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(kernel_size, 1), 
            stride=1, 
            padding=(kernel_size//2, 0),
            groups=mid_channels
        )
        
        self.out_proj = nn.Conv2d(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x



class DCG(nn.Module):
    """Directional Confidence Gating (DCG) with continuous directional statistics."""
    def __init__(self,
                 in_dim,
                 cell_size = (8, 8)):
        super().__init__()

        if type(in_dim) is not int or in_dim < 1:
            raise ValueError(f'in_dim must be a positive integer, got {in_dim}')
        if type(cell_size) is int:
            cell_size = (cell_size, cell_size)
        if not isinstance(cell_size, (tuple, list)) or len(cell_size) != 2 or any(
                type(size) is not int or size < 1 for size in cell_size):
            raise ValueError(f'cell_size must contain two positive integers, got {cell_size}')
        self.cell_size = tuple(cell_size)
        groups = _group_count(in_dim)
        mid_dim = max(1, in_dim // 2)
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.direction_proj = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(groups, in_dim),
            nn.ReLU(inplace=True),  
            nn.AdaptiveAvgPool2d((1, 1))   
        )


        self.gate_branch = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=mid_dim, out_channels=in_dim),
            nn.GroupNorm(groups, in_dim)
        )

        self.value_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(groups, in_dim)
        )

        self.refine = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=mid_dim, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(groups, in_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = split_spatial_blocks(x, self.cell_size)

        direction = self.direction_descriptor(x)
        direction = self.direction_proj(direction)

        x1 = self.sigmoid(self.gate_branch(x + direction))
        x2 = self.value_branch(x)
        # Scalar beta broadcasts across the unchanged 1x1 Conv + GN value branch.
        x = x + self.beta * x1 * x2

        x = merge_spatial_blocks(x)

        # The identity is already included inside each spatial block.
        x = self.refine(x)

        return x


    def descriptor_from_gradients(self, dx, dy, eps=1e-6):
        """Confidence-weighted unoriented tangents over complete 2-D cells.

        dx/dy: [4B, 1, h, w] -> descriptor: [4B, 3, h//ch, w//cw].
        atan2's undefined origin (including float32 underflow) is bypassed
        before evaluation, so the backward pass cannot divide by zero.
        An undefined orientation contributes zero to both directional moments.
        """
        _check_spatial(dx, self.cell_size)
        if dx.shape != dy.shape or dx.shape[1] != 1:
            raise ValueError('dx and dy must have the same [B,1,H,W] shape')
        if dx.dtype != dy.dtype or dx.device != dy.device or not dx.is_floating_point():
            raise ValueError('dx and dy must share a floating dtype and device')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        original_dtype = dx.dtype
        with torch.autocast(device_type=dx.device.type, enabled=False):
            dx, dy = dx.float(), dy.float()
            magnitude_sq = dx.square() + dy.square()
            valid = magnitude_sq > eps
            magnitude = _sqrt_nonnegative(torch.where(valid, magnitude_sq, torch.zeros_like(magnitude_sq)))
            safe_dx = torch.where(valid, dx, torch.ones_like(dx))
            safe_dy = torch.where(valid, dy, torch.zeros_like(dy))
            theta = torch.remainder(torch.atan2(safe_dy, safe_dx) + math.pi / 2, math.pi)
            cos2 = torch.where(valid, torch.cos(2 * theta), torch.zeros_like(theta))
            sin2 = torch.where(valid, torch.sin(2 * theta), torch.zeros_like(theta))
            # Spatial checks above prevent silently dropping incomplete cells.
            moments = torch.cat((magnitude, magnitude * cos2, magnitude * sin2), dim=1)
            means = F.avg_pool2d(moments, kernel_size=self.cell_size, stride=self.cell_size)
            mean_magnitude = means[:, 0:1]
            denominator = mean_magnitude + eps
            u = means[:, 1:2] / denominator
            v = means[:, 2:3] / denominator
            coherence = _sqrt_nonnegative(u.square() + v.square()).clamp(0, 1)
            confidence = mean_magnitude * coherence
            return torch.cat((u * confidence, v * confidence, confidence), dim=1).to(dtype=original_dtype)

    def direction_descriptor(self, x):
        # Keep channel mean, Sobel coefficients and zero padding unchanged.
        _check_spatial(x, self.cell_size)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_mean = x.float().mean(dim=1, keepdim=True)
            sobel_x = x_mean.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
            sobel_y = x_mean.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
            dx = F.conv2d(x_mean, sobel_x, padding=1)
            dy = F.conv2d(x_mean, sobel_y, padding=1)
            return self.descriptor_from_gradients(dx, dy).to(dtype=x.dtype)


def split_spatial_blocks(x, cell_size=(1, 1)):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    _check_spatial(x, cell_size, blocks=2)
    x = einops.rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def merge_spatial_blocks(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = einops.rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x
