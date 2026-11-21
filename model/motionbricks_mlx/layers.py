# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/layers.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# The residual blocks and Encoder derive from T2M-GPT (Apache-2.0) through MotionGPT (MIT).
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""Building blocks: MLPs, transformer encoder, residual 1D convolutions, and the
conditioned convolutional decoder.

Layout convention: every sequence tensor is **channels-last** ``(batch, time, channels)``.
PyTorch's Conv1d works on ``(batch, channels, time)``; MLX convolutions are natively
channels-last, so the many ``transpose(1, 2)`` calls of the original disappear and the
weights are transposed once at load time instead (see :mod:`motionbricks_mlx.weights`).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _activation(name: str):
    if name == "relu":
        return nn.relu
    if name == "silu":
        return nn.silu
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name}")


class ReLU(nn.Module):
    """Parameter-free placeholder that keeps list indices aligned with the checkpoint."""

    def __call__(self, x):
        return nn.relu(x)


class FCBlock(nn.Module):
    """Stack of Linear + LeakyReLU(0.01) layers followed by a projection."""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int):
        super().__init__()
        self.fc_layers = [nn.Linear(size_in, layer_width)] + [
            nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)
        ]
        self.forward_projection = nn.Linear(layer_width, size_out)

    def __call__(self, x):
        for layer in self.fc_layers:
            x = nn.leaky_relu(layer(x), negative_slope=0.01)
        return self.forward_projection(x)


def sincos_position_encoding(seq_length: int, dim: int) -> mx.array:
    """Fixed sine/cosine encoding of shape ``(seq_length, dim)`` (the checkpoint also stores it)."""
    position = mx.arange(seq_length, dtype=mx.float32)[:, None]
    div_term = mx.exp(mx.arange(0, dim, 2, dtype=mx.float32) * -(math.log(10000.0) / dim))
    pe = mx.zeros((seq_length, dim))
    pe[:, 0::2] = mx.sin(position * div_term)
    pe[:, 1::2] = mx.cos(position * div_term)
    return pe


class PositionEmbedding(nn.Module):
    """Holds the checkpoint's ``embed`` parameter, shaped ``(seq, 1, dim)`` like PyTorch."""

    def __init__(self, seq_length: int, dim: int):
        super().__init__()
        self.embed = sincos_position_encoding(seq_length, dim)[:, None, :]

    def table(self, length: int) -> mx.array:
        return self.embed[:length, 0, :]


class MultiheadAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` (batch_first, packed in-projection) on fused SDPA.

    PyTorch's packed ``in_proj_weight``/``in_proj_bias`` live in an ``nn.Linear`` here
    (``in_proj``), so :func:`mlx.nn.quantize` can quantize it like any other projection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        batch, length, dim = x.shape
        head_dim = dim // self.num_heads
        # Split on the contiguous last axis (strided views). Indexing qkv[0..2] after a
        # transpose lowers to three Gather copy kernels per layer instead.
        q, k, v = (
            t.reshape(batch, length, self.num_heads, head_dim).transpose(0, 2, 1, 3)
            for t in mx.split(self.in_proj(x), 3, axis=-1)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(head_dim), mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class TransformerEncoderLayer(nn.Module):
    """Post-norm encoder layer with PyTorch defaults: ReLU FFN of width 2048, eps 1e-5."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int = 2048, eps: float = 1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)

    def __call__(self, x, mask):
        x = self.norm1(x + self.self_attn(x, mask))
        return self.norm2(x + self.linear2(nn.relu(self.linear1(x))))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int):
        super().__init__()
        self.layers = [TransformerEncoderLayer(dim, num_heads) for _ in range(num_layers)]

    def __call__(self, x: mx.array, key_padding_mask: mx.array | None = None) -> mx.array:
        """``key_padding_mask``: bool ``(batch, length)``, True marks padded keys (PyTorch semantics)."""
        mask = None
        if key_padding_mask is not None:
            # SDPA boolean masks use the opposite convention: True = attend.
            mask = mx.logical_not(key_padding_mask)[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x


def _apply_mask(x: mx.array, mask: mx.array | None) -> mx.array:
    return x if mask is None else x * mask


class ResConv1DBlock(nn.Module):
    """Residual block: act -> dilated conv(k=3) -> act -> conv(k=1), with token masking.

    The released checkpoints use ``norm="None"`` (identity), so only that path is ported.
    """

    def __init__(self, n_in: int, n_state: int, dilation: int = 1, activation: str = "relu"):
        super().__init__()
        self.act = _activation(activation)
        self.conv1 = nn.Conv1d(n_in, n_state, 3, stride=1, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1)

    def __call__(self, x, mask=None):
        x = _apply_mask(x, mask)
        residual = x
        x = _apply_mask(self.act(x), mask)
        x = self.conv1(x)
        x = _apply_mask(self.act(x), mask)
        x = self.conv2(x) + residual
        return _apply_mask(x, mask)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation="relu"):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate**depth, activation=activation)
            for depth in range(n_depth)
        ]
        self.model = blocks[::-1] if reverse_dilation else blocks

    def __call__(self, x, mask=None):
        for block in self.model:
            x = block(x, mask)
        return x


class Encoder(nn.Module):
    """Temporal downsampling encoder (motion features -> latent tokens). Not used by the
    released inference graph, ported for completeness and tokenization experiments."""

    def __init__(self, input_emb_width, output_emb_width=512, down_t=3, stride_t=2, width=512,
                 depth=3, dilation_growth_rate=3, activation="relu"):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.model = [nn.Conv1d(input_emb_width, width, 3, padding=1), ReLU()]
        for _ in range(down_t):
            self.model.append([
                nn.Conv1d(width, width, filter_t, stride=stride_t, padding=pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation),
            ])
        self.model.append(nn.Conv1d(width, output_emb_width, 3, padding=1))

    def __call__(self, x):
        for layer in self.model:
            if isinstance(layer, list):
                x = layer[1](layer[0](x))
            else:
                x = layer(x)
        return x


class Upsample2x(nn.Module):
    """Nearest-neighbour temporal upsampling ([a, b] -> [a, a, b, b])."""

    def __call__(self, x):
        return mx.repeat(x, 2, axis=1)


class DoubleCondDecoder(nn.Module):
    """Upsampling convolutional decoder conditioned on dense target poses and external
    per-frame features. Every level reshapes its hidden state so each frame owns a slice of
    the channels, then overwrites the slices of frames that carry a target pose."""

    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=3, width=512, depth=3,
                 dilation_growth_rate=3, activation="relu", target_cond_dim=-1, external_cond_dim=-1,
                 cond_fusion_last_layer=False):
        super().__init__()
        self.down_t, self.width = down_t, width
        self.has_external = external_cond_dim > 0
        self.has_target = target_cond_dim > 0
        self.levels = down_t + (1 if cond_fusion_last_layer else 0)
        self.model = [nn.Conv1d(output_emb_width, width, 3, padding=1), ReLU()]
        for _ in range(down_t):
            self.model.append([
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation),
                Upsample2x(),
                nn.Conv1d(width, width, 3, padding=1),
            ])
        self.model += [nn.Conv1d(width, width, 3, padding=1), ReLU(), nn.Conv1d(width, input_emb_width, 3, padding=1)]
        if self.has_external:
            self.external_cond_blocks = []
            for i in range(self.levels):
                cond_dim = 2 ** (down_t - i) * external_cond_dim
                self.external_cond_blocks += [nn.Linear(cond_dim + width, width), ReLU()]
        if self.has_target:
            assert width % 2**down_t == 0
            self.target_cond_blocks = []
            for i in range(self.levels):
                self.target_cond_blocks += [nn.Linear(target_cond_dim, width // 2 ** (down_t - i)), ReLU()]

    def __call__(self, x, external_cond=None, target_cond=None, has_target_cond=None, token_mask=None):
        """x: ``(B, P, C)`` quantized latents; token_mask: bool ``(B, P)``;
        target_cond/external_cond: ``(B, P * 2**down_t, D)``; has_target_cond: bool ``(B, P * 2**down_t)``."""
        batch = x.shape[0]
        mask = None if token_mask is None else token_mask[..., None].astype(x.dtype)
        x = _apply_mask(x, mask)
        h = nn.relu(self.model[0](x))
        for i in range(self.levels):
            frames_per_position = 2 ** (self.down_t - i)
            positions = h.shape[1]
            if self.has_target and target_cond is not None and has_target_cond is not None:
                h_target = nn.relu(self.target_cond_blocks[2 * i](target_cond))
                h = h.reshape(batch, positions * frames_per_position, self.width // frames_per_position)
                h = mx.where(has_target_cond[..., None], h_target, h)
                h = h.reshape(batch, positions, self.width)
            if self.has_external:
                h_cond = external_cond.reshape(batch, positions, -1)
                h = nn.relu(self.external_cond_blocks[2 * i](mx.concatenate([h, h_cond], axis=-1)))
            if i == self.down_t:
                continue
            resnet, upsample, conv = self.model[i + 2]
            h = upsample(resnet(h, mask))
            if mask is not None:
                mask = mx.repeat(mask, 2, axis=1)
            h = conv(_apply_mask(h, mask))
        h = _apply_mask(h, mask)
        h = nn.relu(self.model[2 + self.down_t](h))
        h = _apply_mask(h, mask)
        h = self.model[2 + self.down_t + 2](h)
        return _apply_mask(h, mask)
