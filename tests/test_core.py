import numpy as np
import torch
import pytest
from dsgcrack import DSGCrack, DCG, SRG, build_model
from dsgcrack.metrics import LegacyMetrics, ProbabilityMetrics
from dsgcrack.checkpoint import validate_state_dict, canonical_config
from dsgcrack.model import ARCHITECTURE


def test_default_model_parameter_count_and_public_names():
    model = build_model()
    assert isinstance(model, DSGCrack)
    assert sum(p.numel() for p in model.parameters()) == 2499460
    assert any(isinstance(m, DCG) for m in model.modules())
    assert any(isinstance(m, SRG) for m in model.modules())
    assert not any("hog_feat" in k or "sr_module" in k for k in model.state_dict())


def test_dcg_gradients_are_finite_for_nonconstant_and_zero_features():
    for value in [torch.randn(2, 16, 16, 16), torch.zeros(2, 16, 16, 16)]:
        value.requires_grad_()
        module = DCG(16)
        output = module(value)
        assert output.shape == value.shape and torch.isfinite(output).all()
        output.square().mean().backward()
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_srg_initial_gates_share_the_shallow_reference():
    module = SRG(8)
    gates = []
    handle = module.sigmoid.register_forward_hook(lambda m, i, output: gates.append(output.detach()))
    features = [torch.randn(2, 8, size, size) for size in [8, 4, 2, 1]]
    result = module(*features)
    handle.remove()
    assert len(result) == len(gates) == 3
    assert all(x.shape == features[0].shape for x in result)
    torch.testing.assert_close(gates[0], gates[1], rtol=0, atol=0)
    torch.testing.assert_close(gates[1], gates[2], rtol=0, atol=0)
    assert all(((g >= 0) & (g <= 1)).all() for g in gates)


def test_fixed_probability_metrics_and_historical_f1_are_distinct():
    probability = ProbabilityMetrics(.5)
    probability.update(np.array([[.9, .4], [.6, .1]]), np.array([[1, 1], [0, 0]]))
    result = probability.result()
    assert result["F1"] == .5
    np.testing.assert_allclose(result["mIoU"], 1/3)
    legacy = LegacyMetrics()
    legacy.update(np.array([[0, 127], [128, 255]], dtype=np.uint8), np.array([[0, 1], [0, 1]]))
    assert legacy.result()["F1"] == .8
    assert legacy.result()["ODS"] == .8


def test_checkpoint_shapes_are_checked_before_any_layer_can_reshape_them():
    model = torch.nn.Linear(2, 2)
    state = model.state_dict()
    state["weight"] = state["weight"].reshape(4, 1)
    with pytest.raises(ValueError, match="shape differs"):
        validate_state_dict(model, state)


def test_project_rename_preserves_previously_exported_config_compatibility():
    previous = {"architecture": "dcg_srg_v1", "image_size": 512}
    result = canonical_config(previous)
    assert result == {"architecture": ARCHITECTURE, "image_size": 512}
    assert previous["architecture"] == "dcg_srg_v1"
