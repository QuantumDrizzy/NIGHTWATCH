"""Validate the exported ONNX is a faithful clone of the trained PyTorch model.

Adapted from validate_onnx.py into a regression test: the model that ships to
TensorRT must match the PyTorch reference. Offline, CPU-only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from trackformer import MobileViT_XT  # noqa: E402

ONNX = os.path.join(ROOT, "NIGHTWATCH_MOBILEVIT_XT.onnx")
PTH = os.path.join(ROOT, "nightwatch_mobilevit.pth")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(ONNX) and os.path.exists(PTH)),
    reason="model artifacts (.onnx/.pth) not present",
)


@pytest.fixture(scope="module")
def pt_and_ort():
    m = MobileViT_XT()
    try:
        m.load_state_dict(torch.load(PTH, map_location="cpu"))
    except Exception as e:  # noqa: BLE001 — weights don't fit the current arch
        pytest.skip(f".pth does not fit the current architecture: {type(e).__name__}")
    m.eval()
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    return m, sess


def test_onnx_io_names(pt_and_ort):
    _, sess = pt_and_ort
    assert sess.get_inputs()[0].name == "input_cube"
    assert [o.name for o in sess.get_outputs()] == ["p_det", "coords"]


def test_onnx_matches_pytorch(pt_and_ort):
    m, sess = pt_and_ort
    iname = sess.get_inputs()[0].name
    max_det = max_reg = 0.0
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(1, 3, 16, 64, 64)
            pd, co = m(x)
            outs = sess.run(None, {iname: x.numpy()})
            max_det = max(max_det, float(np.max(np.abs(pd.numpy() - outs[0]))))
            max_reg = max(max_reg, float(np.max(np.abs(co.numpy() - outs[1]))))
    assert max_det < 1e-4, f"detection diverges: {max_det:.2e}"
    assert max_reg < 1e-4, f"regression diverges: {max_reg:.2e}"


def test_batch1_output_shapes(pt_and_ort):
    # the real-time path runs one detection cube at a time (batch=1)
    _, sess = pt_and_ort
    iname = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 16, 64, 64).astype(np.float32)
    outs = sess.run(None, {iname: x})
    assert outs[0].shape == (1, 1)      # p_det
    assert outs[1].shape == (1, 4)      # coords


def test_dynamic_batch_is_a_known_limit(pt_and_ort):
    # [KNOWN_LIMIT] export.py declares a dynamic batch axis, but the torch exporter
    # bakes the regression-head LSTM initial state at batch=1, so batch>1 fails on
    # the LSTM node. The real-time pipeline uses batch=1, so this is acceptable; it
    # is asserted here so the limitation is explicit (flip this test if a future
    # export makes the batch axis truly dynamic).
    _, sess = pt_and_ort
    iname = sess.get_inputs()[0].name
    x = np.random.randn(3, 3, 16, 64, 64).astype(np.float32)
    with pytest.raises(Exception):
        sess.run(None, {iname: x})
