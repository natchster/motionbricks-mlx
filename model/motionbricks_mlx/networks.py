# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/{g1,soma}/networks.py,
# vqvae.py, quantizer.py). Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
# Apache-2.0. The quantizer credits MotionGPT (MIT) and vector-quantize-pytorch (MIT).
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""The three learned modules: root predictor, masked pose-token transformer, VQ-VAE.

The SOMA and G1 upstream files differ only in hyper-parameters and in whether masks
arrive as INT32 or bool, so one implementation serves both. Masks are bool here.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import (
    FRAMES_PER_TOKEN,
    MASKED_NUM_TOKENS,
    MAX_FRAMES,
    MAX_TOKENS,
    MIN_TOKENS,
    NUM_TOKEN_CHOICES,
    DOWN_T,
    PoseSpec,
    RootSpec,
    VQSpec,
)
from .layers import DoubleCondDecoder, Encoder, FCBlock, PositionEmbedding, ReLU, TransformerEncoder

GLOBAL_ROOT_DIM = 5  # x, y, z, cos(heading), sin(heading)
LOCAL_ROOT_DIM = 4  # yaw rate, vx, vz, height


def _blend(value, missing, has):
    """``value * has + missing * (1 - has)`` exactly as upstream (has is bool)."""
    h = has[..., None].astype(value.dtype)
    return value * h + missing * (1 - h)


class RootNetwork(nn.Module):
    """Predicts the token count (duration) and the normalized global root path.

    Stage 1 encodes the token-count query plus the 8 conditioning frames and classifies
    the duration. Stage 2 embeds the chosen duration, attends over frames + 16 position
    queries, and a conditioned conv decoder upsamples to 64 root frames."""

    def __init__(self, spec: RootSpec, local_pose_dim: int):
        super().__init__()
        e = spec.n_embd
        self.shared_transformer_model = TransformerEncoder(spec.n_layers_shared, e, spec.n_head)
        self.root_token_transformer_model = TransformerEncoder(spec.n_layers_root_token, e, spec.n_head)
        self.conv_output = DoubleCondDecoder(
            GLOBAL_ROOT_DIM, e, down_t=DOWN_T, width=spec.width, depth=spec.depth,
            dilation_growth_rate=spec.dilation_growth_rate, target_cond_dim=GLOBAL_ROOT_DIM, external_cond_dim=e,
        )
        self.proj_local_pose = nn.Linear(local_pose_dim, spec.pose_feat_dim)
        self.proj_local_root_value = nn.Linear(LOCAL_ROOT_DIM, spec.local_root_feat_dim)
        self.proj_global_root_value = nn.Linear(GLOBAL_ROOT_DIM, spec.global_root_feat_dim)
        self.no_local_pose_emb = mx.zeros((spec.pose_feat_dim,))
        self.no_local_root_emb = mx.zeros((spec.local_root_feat_dim,))
        self.no_global_root_emb = mx.zeros((spec.global_root_feat_dim,))
        self.conv_no_frame_emb = mx.zeros((e,))
        proj_in = spec.pose_feat_dim + spec.local_root_feat_dim + spec.global_root_feat_dim
        self.proj_start_input = FCBlock(spec.input_feat_mlp_num_layers, e, proj_in, e)
        self.proj_end_input = FCBlock(spec.input_feat_mlp_num_layers, e, proj_in, e)
        self.input_position_emb = nn.Embedding(FRAMES_PER_TOKEN * 2, e)
        self.proj_input_num_tokens = nn.Embedding(NUM_TOKEN_CHOICES + 2, e)
        self.position_emb = PositionEmbedding(MAX_TOKENS, e)
        self.proj_num_token_output_logit = nn.Linear(e, NUM_TOKEN_CHOICES + 1)
        self.middle_token_emb = nn.Embedding(NUM_TOKEN_CHOICES + 1, e)

    def __call__(self, global_root, has_global, local_root, has_local, poses, has_poses, allowed, num_tokens=None):
        """All root/pose inputs normalized, ``(B, 8, D)``; masks bool ``(B, 8)``;
        ``allowed`` bool ``(B, 11)``; ``num_tokens`` optional int ``(B,)``: 6..16 forces the
        duration, ``MASKED_NUM_TOKENS`` (the default) lets the model choose.

        Returns ``(num_token_logits (B, 12), pred_num_tokens (B,) int32, pred_global_root (B, 64, 5))``."""
        f = FRAMES_PER_TOKEN
        batch = poses.shape[0]
        lp = _blend(self.proj_local_pose(poses), self.no_local_pose_emb, has_poses)
        lr = _blend(self.proj_local_root_value(local_root), self.no_local_root_emb, has_local)
        gr = _blend(self.proj_global_root_value(global_root), self.no_global_root_emb, has_global)
        start = self.proj_start_input(mx.concatenate([lp[:, :f], lr[:, :f], gr[:, :f]], axis=-1))
        end = self.proj_end_input(mx.concatenate([lp[:, -f:], lr[:, -f:], gr[:, -f:]], axis=-1))
        frame_emb = mx.concatenate([start, end], axis=1)
        positioned = frame_emb + self.input_position_emb.weight[None]

        if num_tokens is None:
            num_tokens = mx.full((batch,), MASKED_NUM_TOKENS, dtype=mx.int32)
        num_tokens = num_tokens.astype(mx.int32)
        query = self.proj_input_num_tokens(num_tokens - MIN_TOKENS)[:, None]
        first = self.shared_transformer_model(mx.concatenate([query, positioned], axis=1))
        logits = self.proj_num_token_output_logit(first[:, 0])
        # The out-of-reach class is never allowed; neither are durations the caller excluded.
        neg_inf = mx.array(-mx.inf, dtype=logits.dtype)
        logits = mx.concatenate(
            [mx.where(allowed, logits[:, :NUM_TOKEN_CHOICES], neg_inf), mx.full((batch, 1), neg_inf)], axis=-1
        )
        chosen = mx.where(
            num_tokens == MASKED_NUM_TOKENS, mx.argmax(logits, axis=-1).astype(mx.int32), num_tokens - MIN_TOKENS
        )
        pred_num_tokens = chosen + MIN_TOKENS

        positions = mx.arange(MAX_TOKENS)
        token_mask = positions[None, :] < pred_num_tokens[:, None]
        position_emb = mx.broadcast_to(self.position_emb.table(MAX_TOKENS)[None], (batch, MAX_TOKENS, frame_emb.shape[-1]))
        second_in = mx.concatenate([self.middle_token_emb(chosen)[:, None], positioned, position_emb], axis=1)
        padding = mx.concatenate([mx.zeros((batch, 1 + 2 * f), dtype=mx.bool_), mx.logical_not(token_mask)], axis=1)
        second = self.root_token_transformer_model(second_in, padding)[:, -MAX_TOKENS:]

        # Dense 64-frame conditions: context frames first, target frames at the chosen end.
        has_frame = has_poses | has_global | has_local
        frame = mx.where(has_frame[..., None], frame_emb, self.conv_no_frame_emb)
        rest = MAX_FRAMES - f
        dense_frame = mx.concatenate([frame[:, :f], mx.broadcast_to(self.conv_no_frame_emb, (batch, rest, frame.shape[-1]))], axis=1)
        dense_root = mx.concatenate([global_root[:, :f], mx.zeros((batch, rest, GLOBAL_ROOT_DIM), dtype=global_root.dtype)], axis=1)
        dense_has_root = mx.concatenate([has_global[:, :f], mx.zeros((batch, rest), dtype=mx.bool_)], axis=1)
        frames = mx.arange(MAX_FRAMES)[None, :]
        for i in range(f):
            at = frames == (pred_num_tokens * f - f + i)[:, None]
            dense_frame = mx.where(at[..., None], frame[:, f + i][:, None], dense_frame)
            dense_root = mx.where(at[..., None], global_root[:, f + i][:, None], dense_root)
            dense_has_root = mx.where(at, has_global[:, f + i][:, None], dense_has_root)
        pred_root = self.conv_output(
            second, external_cond=dense_frame, target_cond=dense_root, has_target_cond=dense_has_root, token_mask=token_mask
        )
        return logits, pred_num_tokens, pred_root


class PoseNetwork(nn.Module):
    """Masked token transformer (MaskGIT-style) that predicts multi-head pose codes for 16
    token positions, conditioned on the root path, context/target poses, and duration."""

    def __init__(self, spec: PoseSpec, vq: VQSpec, local_pose_dim: int, root_cond_dim: int):
        super().__init__()
        self.spec, self.vq = spec, vq
        self.num_with_mask = vq.codes_per_head + 1
        self.mask_id = vq.codes_per_head
        feat = spec.pose_feat_width // FRAMES_PER_TOKEN
        self.transformer_model = TransformerEncoder(spec.n_layers, spec.n_embd, spec.n_head)
        self.pose_token_emb = nn.Embedding(vq.num_heads * self.num_with_mask, vq.code_dim_per_head)
        self.proj_pose_token_emb = FCBlock(spec.pose_token_mlp_num_layers, spec.pose_feat_width, vq.code_dim, spec.pose_feat_width)
        self.proj_local_root_values = nn.Linear(root_cond_dim * FRAMES_PER_TOKEN, spec.root_feat_width)
        self.proj_local_pose = nn.Linear(local_pose_dim, feat)
        self.proj_num_valid_positions = nn.Embedding(NUM_TOKEN_CHOICES, spec.token_length_feat_width)
        self.proj_input = [nn.Linear(spec.pose_feat_width + spec.root_feat_width + spec.token_length_feat_width, spec.n_embd), ReLU()]
        self.position_emb = PositionEmbedding(MAX_TOKENS, spec.n_embd)
        self.proj_pose_output_logit = nn.Linear(spec.n_embd, vq.num_heads * vq.codes_per_head)

    def __call__(self, pose_tokens, root_cond, pose_cond, has_pose_cond, num_tokens):
        """pose_tokens int ``(B, 16, H)``; root_cond ``(B, 64, 4)``; pose_cond ``(B, 64, D)``;
        has_pose_cond bool ``(B, 64)``; num_tokens int ``(B,)``. Returns logits ``(B, 16, H, codes)``."""
        batch, positions, heads = pose_tokens.shape
        f = FRAMES_PER_TOKEN
        feat = self.spec.pose_feat_width // f
        root_emb = self.proj_local_root_values(root_cond.reshape(batch, positions, -1))
        cond_emb = self.proj_local_pose(pose_cond)
        offsets = mx.arange(heads, dtype=pose_tokens.dtype) * self.num_with_mask
        token_emb = self.pose_token_emb(pose_tokens + offsets).reshape(batch, positions, -1)
        token_emb = self.proj_pose_token_emb(token_emb).reshape(batch, positions * f, feat)
        pose_emb = _blend(cond_emb, token_emb, has_pose_cond).reshape(batch, positions, self.spec.pose_feat_width)
        length_emb = self.proj_num_valid_positions(num_tokens - MIN_TOKENS)[:, None]
        length_emb = mx.broadcast_to(length_emb, (batch, positions, length_emb.shape[-1]))
        x = self.proj_input[1](self.proj_input[0](mx.concatenate([pose_emb, root_emb, length_emb], axis=-1)))
        x = x + self.position_emb.table(positions)
        padding = mx.arange(positions)[None, :] >= num_tokens[:, None]
        out = self.transformer_model(x, padding)
        return self.proj_pose_output_logit(out).reshape(batch, positions, heads, -1)


class _CodebookStore(nn.Module):
    def __init__(self, heads, entries, width):
        super().__init__()
        self.embed = mx.zeros((heads, entries, width))


class _MultiHeadCodebook(nn.Module):
    def __init__(self, heads, entries, width):
        super().__init__()
        self.codebook = _CodebookStore(heads, entries, width)


class MultiHeadQuantizer(nn.Module):
    """Product codebook: each token is ``H`` independent indices into per-head codebooks."""

    def __init__(self, vq: VQSpec):
        super().__init__()
        self.vq = _MultiHeadCodebook(vq.num_heads, vq.codes_per_head, vq.code_dim_per_head)

    def dequantize(self, indices: mx.array) -> mx.array:
        """``(B, T, H)`` per-head indices -> ``(B, T, H * code_dim_per_head)``."""
        embed = self.vq.codebook.embed
        heads, entries, width = embed.shape
        flat = indices + mx.arange(heads, dtype=indices.dtype) * entries
        codes = embed.reshape(heads * entries, width)[flat]
        return codes.reshape(*indices.shape[:2], heads * width)

    def overall_to_heads(self, overall: mx.array) -> mx.array:
        """Split overall code indices (``codes_per_head ** H`` space) into per-head digits."""
        heads, entries = self.vq.codebook.embed.shape[:2]
        powers = mx.array([entries**h for h in range(heads)], dtype=mx.int64)
        return (overall.astype(mx.int64)[..., None] // powers) % entries


class VQVAE(nn.Module):
    """Pose VQ-VAE. Inference uses ``decode``; ``encoder`` is ported for completeness."""

    def __init__(self, vq: VQSpec, encoder_dim: int, decoder_dim: int, target_dim: int, external_dim: int):
        super().__init__()
        self.encoder = Encoder(encoder_dim, vq.code_dim, DOWN_T, 2, vq.width, vq.depth, vq.dilation_growth_rate)
        self.decoder = DoubleCondDecoder(
            decoder_dim, vq.code_dim, DOWN_T, vq.width, vq.depth, vq.dilation_growth_rate,
            target_cond_dim=target_dim, external_cond_dim=external_dim,
        )
        self.quantizer = MultiHeadQuantizer(vq)

    def decode(self, tokens, target_cond, has_target_cond, external_cond, token_mask):
        latents = self.quantizer.dequantize(tokens)
        return self.decoder(latents, external_cond, target_cond, has_target_cond, token_mask)
