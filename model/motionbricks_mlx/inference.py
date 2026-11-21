# SPDX-License-Identifier: Apache-2.0
# MLX port of MotionBricks (NVlabs/motionbricks, tools/onnx/models/{g1,soma}/inference.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modifications Copyright 2026 The motionbricks-mlx Authors: re-implemented in MLX (see NOTICE).
"""One generation: root path + duration -> pose tokens -> decoded motion.

Everything is static-shape (16 tokens / 64 frames, padded by masks), exactly like the
exported ONNX graph, so the whole ``predict`` traces into a single ``mx.compile`` graph.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import FRAMES_PER_TOKEN, MAX_FRAMES, MAX_TOKENS, ModelSpec
from .networks import VQVAE, PoseNetwork, RootNetwork, dense_conditions, target_frame_masks
from .representation import MotionRepresentation

HEIGHT_ERROR_THRESHOLD = 0.05  # metres, SOMA "from_height_constraint_violation"


class MotionBricksCore(nn.Module):
    """Root predictor, pose predictor and VQ-VAE decoder, chained for batched inference."""

    def __init__(self, spec: ModelSpec, rep: MotionRepresentation):
        super().__init__()
        self.spec, self.rep = spec, rep
        local_pose_dim = len(rep.local_pose_feature)
        self.root = RootNetwork(spec.root, local_pose_dim)
        self.pose = PoseNetwork(spec.pose, spec.vq, local_pose_dim, root_cond_dim=len(rep.global_root_no_height))
        self.vqvae = VQVAE(
            spec.vq,
            encoder_dim=local_pose_dim,
            decoder_dim=rep.local.dim,
            target_dim=local_pose_dim,
            external_dim=len(rep.local_root_velocity),
        )
        self.compute_dtype = mx.float32

    # -- helpers ------------------------------------------------------------------------
    def _remask(self, tokens, prob, num_tokens, step: int, steps: int):
        """MaskGIT re-masking (``_sample_tokens_with_highest_prob``): mask the
        ``cos(step / steps * pi / 2)`` fraction of valid tokens with the lowest confidence."""
        batch, positions, heads = tokens.shape
        ratio = math.cos(step / steps * math.pi * 0.5)
        count = mx.maximum((ratio * num_tokens * heads).astype(mx.int32), 1)
        valid = mx.arange(positions)[None, :, None] < num_tokens[:, None, None]
        prob = mx.where(valid, prob, mx.inf).reshape(batch, -1)
        order = mx.argsort(prob, axis=-1)
        selected = mx.arange(positions * heads)[None, :] < count[:, None]
        order = mx.where(selected, order, order[:, :1])
        hit = (order[:, :, None] == mx.arange(positions * heads)[None, None, :]).any(axis=1)
        flat = mx.where(hit, self.pose.mask_id, tokens.reshape(batch, -1))
        return flat.reshape(batch, positions, heads)

    def _dense_pose_conditions(self, poses, has_poses, num_tokens):
        """Context poses at frames 0..3, target poses at the last token's 4 frames."""
        f = FRAMES_PER_TOKEN
        batch, _, dim = poses.shape
        at = target_frame_masks(num_tokens)
        cond = dense_conditions(poses, mx.zeros((batch, MAX_FRAMES - f, dim), dtype=poses.dtype), at)
        has = mx.concatenate([has_poses[:, :f], mx.zeros((batch, MAX_FRAMES - f), dtype=mx.bool_)], axis=1)
        for i, mask in enumerate(at):  # a union like upstream's numeric_mask_union, not an overwrite
            has = has | (mask & has_poses[:, f + i][:, None])
        return cond, has

    # -- the generation -------------------------------------------------------------------
    def predict(self, global_root, has_global, local_root, has_local, local_poses, has_poses, allowed,
                num_tokens=None, num_inference_steps: int = 1):
        """Inputs mirror the ONNX "joint space" interface (before rig conversion):
        ``global_root (B,8,5)``, ``local_root (B,8,4)``, ``local_poses (B,8,ric+rot)`` float32;
        masks bool ``(B,8)``; ``allowed`` bool ``(B,11)``.
        Returns ``(pred_global (B,64,global_dim) float32, pred_num_tokens (B,) int32)``."""
        rep, dt, f32 = self.rep, self.compute_dtype, mx.float32
        f = FRAMES_PER_TOKEN

        # Recentre on the first context frame's ground position; remember its heading.
        offset_x, offset_z = global_root[:, :1, 0:1], global_root[:, :1, 2:3]
        heading0 = mx.arctan2(global_root[:, 0, 4], global_root[:, 0, 3])
        zero = mx.zeros_like(offset_x)
        recentred = global_root - mx.concatenate([offset_x, zero, offset_z, zero, zero], axis=-1)
        g_n = rep.normalize_global_root(recentred)
        l_n = rep.normalize_local_root(local_root)
        p_n = rep.normalize_local_pose(mx.concatenate([recentred[..., 1:2], local_poses], axis=-1))

        # 1) Duration + root path.
        _, num_tokens, pred_root = self.root(
            g_n.astype(dt), has_global, l_n.astype(dt), has_local, p_n.astype(dt), has_poses, allowed, num_tokens
        )
        pred_root = pred_root.astype(f32)
        lengths = num_tokens * f
        pred_local_root = rep.global_root_to_local(pred_root, lengths)
        last = (lengths - 1)[:, None]
        estimated = mx.take_along_axis(pred_local_root, last[..., None], axis=1)
        h = has_local[:, -1:, None].astype(f32)
        final = h * l_n[:, -1:] + (1 - h) * estimated
        at_last = (mx.arange(MAX_FRAMES)[None, :] == last)[..., None]
        pred_local_root = mx.where(at_last, final, pred_local_root)

        # 2) Pose tokens (release config: one MaskGIT step, argmax sampling).
        pose_cond, has_pose_cond = self._dense_pose_conditions(p_n, has_poses, num_tokens)
        root_cond = pred_root[..., rep.global_root_no_height_mx]
        heads = self.spec.vq.num_heads
        tokens = mx.full((global_root.shape[0], MAX_TOKENS, heads), self.pose.mask_id, dtype=mx.int32)
        prob = mx.full(tokens.shape, 1.0 / heads)
        for step in range(num_inference_steps):
            tokens = self._remask(tokens, prob, num_tokens, step, num_inference_steps)
            logits = self.pose(tokens, root_cond.astype(dt), pose_cond.astype(dt), has_pose_cond, num_tokens)
            tokens = mx.argmax(logits, axis=-1).astype(mx.int32)
            if step + 1 < num_inference_steps:
                prob = mx.take_along_axis(mx.softmax(logits.astype(f32), axis=-1), tokens[..., None], axis=-1)[..., 0]

        # 3) Decode tokens to local motion, integrate to global.
        token_mask = mx.arange(MAX_TOKENS)[None, :] < num_tokens[:, None]
        external = pred_local_root[..., rep.local_root_velocity_mx]
        recon = self.vqvae.decode(tokens, pose_cond.astype(dt), has_pose_cond, external.astype(dt), token_mask)
        pred_global = rep.local_to_global(recon.astype(f32))

        mode = self.spec.final_root_mode
        if mode != "from_pose_module":
            root_from_root_module = rep.unnormalize_global_root(pred_root)
            if mode == "from_root_module":
                use_root = mx.ones((pred_global.shape[0], 1, 1))
            elif mode == "from_height_constraint_violation":
                target_frames = (num_tokens[:, None] - 1) * f + mx.arange(f)[None, :]
                h_root = mx.take_along_axis(root_from_root_module[..., 1], target_frames, axis=1)
                h_pose = mx.take_along_axis(pred_global[..., 1], target_frames, axis=1)
                h_target = rep.unnormalize_global_root(g_n)[:, -f:, 1]
                w = has_global[:, -f:].astype(f32)
                err_pose = (mx.abs(h_pose - h_target) * w).sum(axis=1) / w.sum(axis=1)
                err_root = (mx.abs(h_root - h_target) * w).sum(axis=1) / w.sum(axis=1)
                use_root = ((err_root < err_pose) & (err_pose > HEIGHT_ERROR_THRESHOLD))[:, None, None].astype(f32)
            else:
                raise ValueError(f"Unsupported final_root_mode {mode!r}")
            root = use_root * root_from_root_module + (1 - use_root) * pred_global[..., :5]
            pred_global = mx.concatenate([root, pred_global[..., 5:]], axis=-1)

        # Undo the recentring: restore the ground offset and rotate headings back.
        angle = mx.arctan2(pred_global[..., 4], pred_global[..., 3]) + heading0[:, None]
        root = mx.stack(
            [pred_global[..., 0] + offset_x[:, :, 0], pred_global[..., 1], pred_global[..., 2] + offset_z[:, :, 0],
             mx.cos(angle), mx.sin(angle)],
            axis=-1,
        )
        return mx.concatenate([root, pred_global[..., 5:]], axis=-1), num_tokens
