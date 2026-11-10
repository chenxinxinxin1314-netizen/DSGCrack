"""Strict loading for public checkpoints and an explicit adapter for trusted legacy files."""
import re
import torch
from .model import ARCHITECTURE


def canonical_config(config):
    result = dict(config)
    # Accept checkpoints/configs produced before the release-project rename.
    if result.get("architecture") == "dcg_srg_v1":
        result["architecture"] = ARCHITECTURE
    return result


def canonical_state_dict(state):
    result = {}
    for old_key, tensor in state.items():
        key = old_key.removeprefix("module.")
        key = key.replace("backbone.patch_embed.", "backbone.stem.")
        key = key.replace("backbone.vss_layers.", "backbone.stages.")
        if re.match(r"^backbone\.stages\.\d+\.blocks\.\d+\.1\.", key):
            key = key.replace(".hog_feat.", ".direction_proj.")
            key = key.replace(".weight.", ".gate_branch.")
            key = key.replace(".conv.", ".value_branch.")
            key = key.replace(".fuse_block.", ".refine.")
        key = key.replace("decoder.sr_module.", "decoder.srg.")
        key = key.replace("decoder.srg.att.", "decoder.srg.reference.")
        if key in result:
            raise ValueError(f"Checkpoint key collision after renaming: {key}")
        result[key] = tensor
    return result


def read_checkpoint(path, legacy=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=not legacy)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state dictionary or a checkpoint containing 'model'.")
    config = canonical_config(checkpoint.get("config", {}))
    if "architecture" in config and config["architecture"] != ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match this release.")
    if "config" in checkpoint:
        checkpoint["config"] = config
    saved = checkpoint.get("args")
    if saved is not None:
        options = saved if isinstance(saved, dict) else vars(saved)
        expected = {"srf_variant": "residual_scale", "disable_local_path": False,
                    "srf_dim": 8, "dcg_cell_sizes": [8, 8, 8, 8], "dcg_blocks": 2}
        for name, value in expected.items():
            actual = options.get(name, value)
            if name == "dcg_cell_sizes":
                actual = list(actual)
            if actual != value:
                raise ValueError(f"Legacy checkpoint is not the full default model: {name}={actual}")
    raw = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw, dict) or not all(torch.is_tensor(v) for v in raw.values()):
        raise ValueError("Checkpoint does not contain a tensor state dictionary.")
    return canonical_state_dict(raw), checkpoint


def validate_state_dict(model, state):
    expected = model.state_dict()
    missing, unexpected = set(expected)-set(state), set(state)-set(expected)
    if missing or unexpected:
        raise ValueError(f"Checkpoint keys differ: missing={sorted(missing)[:5]}, "
                         f"unexpected={sorted(unexpected)[:5]}")
    for key, value in state.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"Checkpoint tensor shape differs: {key}: "
                             f"{tuple(value.shape)} vs {tuple(expected[key].shape)}")


def load_weights(model, path, legacy=False):
    state, checkpoint = read_checkpoint(path, legacy=legacy)
    validate_state_dict(model, state)
    model.load_state_dict(state, strict=True)
    return checkpoint
