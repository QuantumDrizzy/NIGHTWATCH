"""Validate the TrackFormer (MobileViT-XT) architecture: it builds and runs.

Offline, CPU-only (no CUDA/sensor). Checks the forward pass produces the right
shapes and well-formed outputs, is deterministic in eval mode, and is
batch-independent (BatchNorm uses running stats, not batch stats).
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trackformer import MobileViT_XT  # noqa: E402

INPUT = (3, 16, 64, 64)   # channels, time, H, W


@pytest.fixture(scope="module")
def model():
    m = MobileViT_XT()
    m.eval()
    return m


def test_forward_shapes(model):
    x = torch.randn(2, *INPUT)
    with torch.no_grad():
        p_det, seed = model(x)
    assert p_det.shape == (2, 1)          # detection confidence
    assert seed.shape == (2, 4)           # confidence-gated [x, y, vx, vy]


def test_detection_is_a_probability(model):
    x = torch.randn(4, *INPUT)
    with torch.no_grad():
        p_det, _ = model(x)
    assert torch.all(p_det >= 0.0) and torch.all(p_det <= 1.0)


def test_outputs_finite(model):
    x = torch.randn(2, *INPUT)
    with torch.no_grad():
        p_det, seed = model(x)
    assert torch.all(torch.isfinite(p_det))
    assert torch.all(torch.isfinite(seed))


def test_deterministic_in_eval(model):
    x = torch.randn(1, *INPUT)
    with torch.no_grad():
        a_p, a_s = model(x)
        b_p, b_s = model(x)
    assert torch.allclose(a_p, b_p) and torch.allclose(a_s, b_s)


def test_batch_independence(model):
    x = torch.randn(3, *INPUT)
    with torch.no_grad():
        pd_batch, seed_batch = model(x)
        pd_one, seed_one = model(x[:1])
    assert torch.allclose(pd_batch[:1], pd_one, atol=1e-4)
    assert torch.allclose(seed_batch[:1], seed_one, atol=1e-4)


def test_parameter_count_is_lightweight(model):
    n_params = sum(p.numel() for p in model.parameters())
    assert 0 < n_params < 5_000_000        # MobileViT-XT is a small net by design


def test_gradients_flow(model):
    # training path sanity: a backward pass populates gradients
    m = MobileViT_XT()
    m.train()
    x = torch.randn(1, *INPUT)
    p_det, seed = m(x)
    (p_det.sum() + seed.sum()).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in m.parameters())
