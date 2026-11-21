"""MLX vs NVIDIA's upstream PyTorch model, stage by stage, on golden fixtures.

Each sub-network is *teacher-forced* with the exact tensors the PyTorch run fed it, so a
mismatch localizes to one module instead of cascading through argmax decisions.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_helpers import CASES, golden_inputs, load_golden, load_model

pytestmark = pytest.mark.weights


def rel_err(got, want, mask=None):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if mask is not None:
        got, want = got[mask], want[mask]
    return np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)


def arr(g, key):
    return mx.array(g[key])


def flag(g, key):
    return mx.array(g[key]) != 0


@pytest.mark.parametrize("case", CASES)
def test_root_network(model_name, case):
    g, model = load_golden(model_name, case), load_model(model_name)
    logits, tokens, root = model.core.root(
        arr(g, "root_in.global_root"), flag(g, "root_in.has_global"), arr(g, "root_in.local_root"),
        flag(g, "root_in.has_local"), arr(g, "root_in.poses"), flag(g, "root_in.has_poses"), flag(g, "root_in.allowed"),
        arr(g, "root_in.num_tokens").reshape(-1),
    )
    want_logits = g["root_out.num_token_logits"]
    finite = np.isfinite(want_logits)
    assert np.array_equal(np.isfinite(np.array(logits)), finite), "masked (-inf) classes differ"
    assert rel_err(logits, want_logits, finite) < 1e-4
    assert np.array_equal(np.array(tokens), g["root_out.pred_num_tokens"])
    assert rel_err(root, g["root_out.pred_global_root_values"]) < 1e-4


@pytest.mark.parametrize("case", CASES)
def test_pose_network(model_name, case):
    g, model = load_golden(model_name, case), load_model(model_name)
    logits = model.core.pose(
        arr(g, "pose_in.pose_tokens").astype(mx.int32), arr(g, "pose_in.root_cond"), arr(g, "pose_in.pose_cond"),
        flag(g, "pose_in.has_pose_cond"), arr(g, "pose_in.num_tokens").astype(mx.int32).reshape(-1),
    )
    want = g["pose_out.pose_logits"]
    assert rel_err(logits, want) < 1e-4
    assert np.array_equal(np.array(mx.argmax(logits, -1)), want.argmax(-1))


@pytest.mark.parametrize("case", CASES)
def test_vqvae_decoder(model_name, case):
    g, model = load_golden(model_name, case), load_model(model_name)
    recon = model.core.vqvae.decode(
        arr(g, "vq_in.tokens").astype(mx.int32), arr(g, "vq_in.target_cond"), flag(g, "vq_in.has_target_cond"),
        arr(g, "vq_in.external_cond"), flag(g, "vq_in.token_mask"),
    )
    assert rel_err(recon, g["vq_out.recon_state"]) < 1e-4


def check_end_to_end(model_name, model, g, out):
    assert np.array_equal(out["pred_num_tokens"], g["pred_num_tokens"])
    for i, count in enumerate(g["pred_num_tokens"]):
        valid = slice(0, 4 * int(count))
        if model_name == "unitree_g1":
            assert np.abs(out["pred_qpos"][i, valid] - g["pred_qpos"][i, valid]).max() < 5e-4  # m / rad
            assert np.abs(out["pred_foot_contacts"][i, valid] - g["pred_foot_contacts"][i, valid]).max() < 5e-3
        else:
            idx = model.core.rep.global_.indices
            got, want = out["pred_global_motions"][i, valid], g["pred_global_motions"][i, valid]
            root = idx["root"]
            rot = idx["global_rot_data"]  # passes through a Gram-Schmidt: looser
            other = np.setdiff1d(idx["all"], np.concatenate([root, rot]))
            assert np.abs(got[:, root] - want[:, root]).max() < 1e-4
            assert np.abs(got[:, other] - want[:, other]).max() < 1e-3
            assert np.abs(got[:, rot] - want[:, rot]).max() < 3e-3


@pytest.mark.parametrize("case", CASES)
def test_end_to_end(model_name, case):
    g, model = load_golden(model_name, case), load_model(model_name)
    check_end_to_end(model_name, model, g, model.run(golden_inputs(g)))


def test_end_to_end_on_the_cpu_stream(model_name):
    """The CPU stream takes the op-based rig instead of the Metal kernels (eager: no compile wait)."""
    g, model = load_golden(model_name, "trace"), load_model(model_name)
    with mx.stream(mx.cpu):
        out = model.run(golden_inputs(g), compiled=False)
    check_end_to_end(model_name, model, g, out)
