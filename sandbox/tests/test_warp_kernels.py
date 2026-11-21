"""NVIDIA's upstream Warp CUDA kernels, compiled and run unchanged on the Apple CPU."""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from cudashim.warp_cpu import UPSTREAM, cuda_device_alias, upstream_on_path  # noqa: E402

pytestmark = pytest.mark.skipif(not (UPSTREAM / "motionbricks").is_dir(), reason="run sandbox/setup.sh")


def test_cuda_device_is_absent_but_aliasable():
    wp.init()
    assert not wp.is_cuda_available()
    with pytest.raises(ValueError, match="cuda:0"):
        wp.get_device("cuda:0")
    with cuda_device_alias() as device:
        wp.set_device("cuda:0")  # what upstream examples do
        assert wp.get_device("cuda").is_cpu and device.is_cpu
        assert wp.zeros(2).device.is_cpu  # allocations on the default device follow the alias
        with pytest.raises((RuntimeError, ValueError)):  # explicit device="cuda" does not (documented)
            wp.zeros(2, device="cuda")
    wp.set_device("cpu")


@pytest.mark.parametrize("num_targets", [1, 2, 4])
def test_upstream_condition_kernel(num_targets):
    with upstream_on_path():
        from motionbricks.model.kernels import write_motionbricks_conditions_kernel
    bits = 0b10101010101
    arrays = [wp.zeros((1, n), dtype=wp.int32, device="cpu") for n in (8, 8, 8, 11)]
    wp.launch(write_motionbricks_conditions_kernel, dim=11, inputs=[num_targets, bits, *arrays], device="cpu")
    has_global, has_local, has_poses, allowed = (a.numpy()[0] for a in arrays)
    expected = (np.arange(8) < 4 + num_targets).astype(np.int32)
    expected_local = expected.copy()
    expected_local[[3, 3 + num_targets]] = 0
    assert np.array_equal(has_global, expected) and np.array_equal(has_poses, expected)
    assert np.array_equal(has_local, expected_local)
    assert np.array_equal(allowed, (bits >> np.arange(11)) & 1)


def test_upstream_padding_kernel_is_generic_over_dtype():
    with upstream_on_path():
        from motionbricks.model.kernels import pad_token_poses_kernel
    poses = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    src = wp.array(poses, dtype=wp.float32, device="cpu")
    dst = wp.zeros((4, 5), dtype=wp.float32, device="cpu")
    wp.launch(pad_token_poses_kernel, dim=(4, 5), inputs=[src, 2, dst], device="cpu")
    assert np.array_equal(dst.numpy(), poses[[0, 1, 1, 1]])
