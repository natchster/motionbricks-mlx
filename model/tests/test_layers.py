"""Weight-free unit tests: each building block against a NumPy re-derivation."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from motionbricks_mlx import layers, rig
from motionbricks_mlx.config import SOMA
from motionbricks_mlx.networks import MultiHeadQuantizer
from motionbricks_mlx.representation import MotionRepresentation

RNG = np.random.default_rng(0)


def randomize(module: nn.Module, scale=0.1):
    from mlx.utils import tree_flatten, tree_unflatten

    params = [(k, mx.array(RNG.normal(0, scale, v.shape).astype(np.float32))) for k, v in tree_flatten(module.parameters())]
    module.update(tree_unflatten(params))
    return dict(params)


def np_layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def np_encoder_layer(x, p, heads, pad):
    """PyTorch TransformerEncoderLayer (post-norm, ReLU) in float64 NumPy."""
    g = lambda k: np.asarray(p[k], np.float64)  # noqa: E731
    b, n, d = x.shape
    qkv = x @ g("self_attn.in_proj.weight").T + g("self_attn.in_proj.bias")
    q, k, v = np.split(qkv, 3, -1)
    split = lambda t: t.reshape(b, n, heads, d // heads).transpose(0, 2, 1, 3)  # noqa: E731
    q, k, v = split(q), split(k), split(v)
    s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(d // heads)
    s = np.where(pad[:, None, None, :], -np.inf, s)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(b, n, d) @ g("self_attn.out_proj.weight").T + g("self_attn.out_proj.bias")
    x = np_layer_norm(x + o, g("norm1.weight"), g("norm1.bias"))
    f = np.maximum(x @ g("linear1.weight").T + g("linear1.bias"), 0) @ g("linear2.weight").T + g("linear2.bias")
    return np_layer_norm(x + f, g("norm2.weight"), g("norm2.bias"))


def test_sincos_matches_formula():
    pe = np.array(layers.sincos_position_encoding(16, 8))
    pos = np.arange(16)[:, None]
    div = np.exp(np.arange(0, 8, 2) * -(math.log(10000.0) / 8))
    assert np.allclose(pe[:, 0::2], np.sin(pos * div), atol=1e-6)
    assert np.allclose(pe[:, 1::2], np.cos(pos * div), atol=1e-6)


def test_transformer_layer_matches_pytorch_semantics():
    layer = layers.TransformerEncoderLayer(64, 4, ffn_dim=96)
    params = randomize(layer)
    params = {k: np.array(v) for k, v in params.items()}
    x = RNG.normal(size=(2, 7, 64)).astype(np.float32)
    pad = np.zeros((2, 7), bool)
    pad[1, 5:] = True
    out = layers.TransformerEncoder(1, 64, 4)
    out.layers = [layer]
    got = np.array(out(mx.array(x), mx.array(pad)))
    want = np_encoder_layer(x.astype(np.float64), params, 4, pad)
    assert np.abs(got - want).max() < 2e-5


def test_resconv_block_masks_and_residual():
    block = layers.ResConv1DBlock(8, 8, dilation=3)
    randomize(block)
    x = mx.array(RNG.normal(size=(2, 10, 8)).astype(np.float32))
    mask = mx.array(np.arange(10)[None, :, None] < np.array([10, 6])[:, None, None]).astype(mx.float32)
    y = np.array(block(x, mask))
    assert np.all(y[1, 6:] == 0)
    # Without masking a zero block (zero weights) is the identity.
    zero = layers.ResConv1DBlock(8, 8)
    zero.update({"conv1": {"weight": mx.zeros_like(zero.conv1.weight), "bias": mx.zeros_like(zero.conv1.bias)},
                 "conv2": {"weight": mx.zeros_like(zero.conv2.weight), "bias": mx.zeros_like(zero.conv2.bias)}})
    assert np.allclose(np.array(zero(x)), np.array(x))


def test_decoder_shapes_and_target_override():
    dec = layers.DoubleCondDecoder(5, 32, down_t=2, width=16, depth=2, target_cond_dim=5, external_cond_dim=3)
    randomize(dec)
    x = mx.array(RNG.normal(size=(2, 16, 32)).astype(np.float32))
    ext = mx.array(RNG.normal(size=(2, 64, 3)).astype(np.float32))
    tgt = mx.array(RNG.normal(size=(2, 64, 5)).astype(np.float32))
    has = mx.zeros((2, 64), dtype=mx.bool_)
    mask = mx.array(np.arange(16)[None] < np.array([16, 10])[:, None])
    y = np.array(dec(x, ext, tgt, has, mask))
    assert y.shape == (2, 64, 5)
    assert np.all(y[1, 40:] == 0)  # 10 tokens * 4 frames


def test_quantizer_lookup_and_digits():
    from motionbricks_mlx.config import VQSpec

    vq = VQSpec(num_heads=3, codes_per_head=4, code_dim=6)
    q = MultiHeadQuantizer(vq)
    embed = RNG.normal(size=(3, 4, 2)).astype(np.float32)
    q.vq.codebook.embed = mx.array(embed)
    idx = RNG.integers(0, 4, size=(2, 5, 3))
    got = np.array(q.dequantize(mx.array(idx.astype(np.int32))))
    want = np.concatenate([embed[h][idx[..., h]] for h in range(3)], -1)
    assert np.array_equal(got, want)
    overall = (idx * 4 ** np.arange(3)).sum(-1)
    assert np.array_equal(np.array(q.overall_to_heads(mx.array(overall))), idx)


def test_rotation_round_trips():
    q = RNG.normal(size=(50, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.sign(q[:, :1])
    m = rig.quat_matrix(mx.array(q))
    back = np.array(rig.matrix_quat(m))
    back *= np.sign(back[:, :1])
    assert np.abs(back - q).max() < 1e-5
    six = np.array(rig.matrix_to_cont6d(m))
    assert np.abs(np.array(rig.cont6d_to_matrix(mx.array(six))) - np.array(m)).max() < 1e-5
    w = np.array(rig.wrap(mx.array([-7.0, -math.pi, 0.0, math.pi, 7.0])))
    assert np.all(w >= -math.pi) and np.all(w < math.pi)


def test_root_conversion_integrates_back():
    """global -> local (finite differences) then local -> global (integration) recovers the path."""
    rep = MotionRepresentation(SOMA.num_joints, np.zeros(9 + 87 + 180 + 90 + 4), np.ones(9 + 87 + 180 + 90 + 4))
    t = np.linspace(0, 2, 64)
    heading = 0.6 * np.sin(t)
    root = np.stack([np.cumsum(np.cos(heading)) / 30, 0.9 + 0.01 * t, np.cumsum(np.sin(heading)) / 30,
                     np.cos(heading), np.sin(heading)], -1)[None].astype(np.float32)
    scale = np.sqrt(1 + 1e-5)
    local = rep.global_root_to_local(mx.array(root / scale), mx.array([64]))
    body = mx.zeros((1, 64, rep.local.dim - 4))
    recon = np.array(rep.local_to_global(mx.concatenate([local, body], -1)))[0, :, :5]
    # removing_heading=False: velocities stay in the world frame, so integration returns the
    # path relative to frame 0, and headings relative to the first heading.
    expected = root[0].astype(np.float64).copy()
    expected[:, 0] -= expected[0, 0]
    expected[:, 2] -= expected[0, 2]
    expected[:, 3] = np.cos(heading - heading[0])
    expected[:, 4] = np.sin(heading - heading[0])
    assert np.abs(recon - expected).max() < 1e-4


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_low_precision_attention_is_finite(dtype):
    enc = layers.TransformerEncoder(2, 32, 4)
    randomize(enc)
    enc.set_dtype(dtype)
    pad = mx.array(np.arange(8)[None] >= np.array([8, 3])[:, None])
    y = enc(mx.array(RNG.normal(size=(2, 8, 32))).astype(dtype), pad)
    assert y.dtype == dtype and np.isfinite(np.array(y.astype(mx.float32))).all()


# --- custom Metal rig kernels vs the op-based reference --------------------------------------
import pytest as _pytest  # noqa: E402

from motionbricks_mlx import kernels as _kernels  # noqa: E402


@_pytest.mark.skipif(not _kernels.available(), reason="needs the Metal GPU")
def test_rig_kernels_match_ops():
    from motionbricks_mlx.model import load_rig_arrays
    from motionbricks_mlx.representation import Layout, GLOBAL_ROOT_KEYS, body_keys
    from motionbricks_mlx.rig import G1Rig, SomaRetarget

    rng = np.random.default_rng(3)
    rig = G1Rig(load_rig_arrays("unitree_g1"), Layout.build(GLOBAL_ROOT_KEYS, body_keys(34)).indices)
    qpos = rng.normal(0, 0.4, size=(3, 8, 36)).astype(np.float32)
    qpos[..., 2] += 0.75
    ctx, tgt = mx.array(qpos[:, :4]), mx.array(qpos[:, 4:])
    for got, want in zip(rig.preprocess(ctx, tgt), rig.preprocess_ops(ctx, tgt)):
        np.testing.assert_allclose(np.array(got), np.array(want), atol=2e-5, rtol=1e-5)
    feats = rng.normal(0, 1, size=(2, 64, 414)).astype(np.float32)
    for got, want in zip(rig.postprocess(mx.array(feats)), rig.postprocess_ops(mx.array(feats))):
        np.testing.assert_allclose(np.array(got), np.array(want), atol=2e-5, rtol=1e-5)

    soma = SomaRetarget(load_rig_arrays("soma"), 30)
    six = mx.array(rng.normal(0, 1, size=(2, 64, 180)).astype(np.float32))
    np.testing.assert_allclose(np.array(soma.capture_to_standard(six)),
                               np.array(soma.apply_ops(six, soma.capture, soma.standard)), atol=2e-5, rtol=1e-5)
