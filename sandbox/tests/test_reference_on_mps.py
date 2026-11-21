"""NVIDIA's PyTorch MotionBricks, moved with ``.cuda()`` as on an NVIDIA box, runs on the
Apple GPU through the shim and agrees with the CPU golden fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cudashim import reference, torch_cuda  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.reference,
    pytest.mark.skipif(not reference.available(), reason="run sandbox/setup.sh"),
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"),
]


@pytest.mark.parametrize("name", ["unitree_g1", "soma"])
def test_upstream_model_on_cuda_alias(name):
    golden_path = ROOT / f"sandbox/golden/{name}/random.npz"
    if not golden_path.is_file() or not any(reference.checkpoint_dir(name).rglob("*.safetensors")):
        pytest.skip("needs checkpoints + golden fixtures")
    golden = dict(np.load(golden_path))
    request = {k[3:]: v for k, v in golden.items() if k.startswith("in.")}
    with torch_cuda.enabled():
        model = reference.build(name, "cpu").cuda()  # NVIDIA-style code path
        assert next(model.parameters()).device.type == "mps"
        out = reference.run(model, name, request, device="cuda")
    assert np.array_equal(out["pred_num_tokens"], golden["pred_num_tokens"])
    key = "pred_qpos" if name == "unitree_g1" else "pred_global_motions"
    for i, n in enumerate(golden["pred_num_tokens"]):
        assert np.abs(out[key][i, :4 * n] - golden[key][i, :4 * n]).max() < 5e-3
