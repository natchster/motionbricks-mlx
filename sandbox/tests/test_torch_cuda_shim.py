"""CUDA-targeting PyTorch code runs unchanged on the Apple GPU through cudashim.torch_cuda."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from cudashim import torch_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU (MPS)")


def test_plain_torch_has_no_cuda():
    assert not torch.cuda.is_available()
    with pytest.raises((AssertionError, RuntimeError)):
        torch.zeros(1).cuda()


def test_cuda_code_runs_on_mps():
    with torch_cuda.enabled() as device:
        assert device.type == "mps"
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64).cuda()
        c = torch.ones(3).to("cuda:0")
        d = torch.ones(3).to(torch.device("cuda"))
        assert {t.device.type for t in (a, b, c, d)} == {"mps"}
        layer = torch.nn.Linear(64, 8).cuda()
        assert next(layer.parameters()).device.type == "mps"
        out = layer(a @ b)
        torch.cuda.synchronize()
        assert out.shape == (64, 8) and out.device.type == "mps"
    assert not torch.cuda.is_available()  # cleanly restored


def test_cuda_events_time_kernels():
    with torch_cuda.enabled():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        x = torch.randn(512, 512, device="cuda")
        start.record()
        for _ in range(10):
            x = x @ x / 512
        end.record()
        torch.cuda.synchronize()
        assert start.elapsed_time(end) >= 0


def test_cuda_autocast_maps_to_mps():
    with torch_cuda.enabled():
        x = torch.randn(32, 32, device="cuda")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y = x @ x
        assert y.dtype == torch.float16


def test_matches_cpu_numerics():
    torch.manual_seed(0)
    x = torch.randn(128, 256)
    w = torch.randn(256, 64)
    ref = x @ w
    with torch_cuda.enabled():
        got = (x.cuda() @ w.cuda()).cpu()
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-4)
