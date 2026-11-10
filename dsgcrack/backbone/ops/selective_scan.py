"""VMamba oflex scan adapter. See third_party/README.md for source attribution."""
import torch

try:
    import selective_scan_cuda_oflex
except ImportError:
    selective_scan_cuda_oflex = None


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, u, delta, A, B, C, D, bias, softplus, oflex):
        ctx.softplus = softplus
        out, state, *_ = selective_scan_cuda_oflex.fwd(
            u, delta, A, B, C, D, bias, softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, bias, state)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        u, delta, A, B, C, D, bias, state = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, dbias, *_ = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, bias, dout, state, ctx.softplus, 1)
        return du, ddelta, dA, dB, dC, dD, dbias, None, None


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None,
                      delta_softplus=True, oflex=True, backend=None):
    if backend not in (None, "oflex"):
        raise ValueError("This release uses the paper-default oflex backend.")
    if not u.is_cuda:
        raise RuntimeError("Full-model forward requires an NVIDIA CUDA GPU.")
    if selective_scan_cuda_oflex is None:
        raise RuntimeError(
            "Build the scan extension first: "
            "python -m pip install --no-build-isolation ./kernels/selective_scan")
    return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)
