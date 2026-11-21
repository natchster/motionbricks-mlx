// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Building blocks, ported from `motionbricks_mlx/layers.py`. Channels-last `(B, T, C)`.

use mlx_rs::{Array, fast, ops};

use crate::ops::*;
use crate::weights::Weights;

pub struct Linear {
    pub weight: Array,
    pub bias: Array,
}

impl Linear {
    pub fn load(w: &mut Weights, p: &str) -> R<Self> {
        Ok(Linear { weight: w.get(&format!("{p}.weight"))?, bias: w.get(&format!("{p}.bias"))? })
    }
    pub fn forward(&self, x: &Array) -> R<Array> {
        linear(x, &self.weight, &self.bias)
    }
}

/// `nn.Embedding`: `weight[ids]`.
pub struct Embedding {
    pub weight: Array,
}

impl Embedding {
    pub fn load(w: &mut Weights, p: &str) -> R<Self> {
        Ok(Embedding { weight: w.get(&format!("{p}.weight"))? })
    }
    pub fn forward(&self, ids: &Array) -> R<Array> {
        self.weight.take_axis(ids, 0)
    }
}

pub struct LayerNorm {
    weight: Array,
    bias: Array,
}

impl LayerNorm {
    fn load(w: &mut Weights, p: &str) -> R<Self> {
        Ok(LayerNorm { weight: w.get(&format!("{p}.weight"))?, bias: w.get(&format!("{p}.bias"))? })
    }
    fn forward(&self, x: &Array) -> R<Array> {
        fast::layer_norm(x, &self.weight, &self.bias, 1e-5)
    }
}

pub struct Conv1d {
    weight: Array, // (out, k, in)
    bias: Array,
    padding: i32,
    dilation: i32,
}

impl Conv1d {
    pub fn load(w: &mut Weights, p: &str, padding: i32, dilation: i32) -> R<Self> {
        Ok(Conv1d { weight: w.get(&format!("{p}.weight"))?, bias: w.get(&format!("{p}.bias"))?, padding, dilation })
    }
    pub fn forward(&self, x: &Array) -> R<Array> {
        ops::add(ops::conv1d(x, &self.weight, 1, self.padding, self.dilation, 1)?, &self.bias)
    }
}

/// Linear + LeakyReLU(0.01) stack followed by a projection.
pub struct FcBlock {
    layers: Vec<Linear>,
    projection: Linear,
}

impl FcBlock {
    pub fn load(w: &mut Weights, p: &str, num_layers: usize) -> R<Self> {
        let layers = (0..num_layers).map(|i| Linear::load(w, &format!("{p}.fc_layers.{i}"))).collect::<R<_>>()?;
        Ok(FcBlock { layers, projection: Linear::load(w, &format!("{p}.forward_projection"))? })
    }
    pub fn forward(&self, x: &Array) -> R<Array> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = leaky_relu(&layer.forward(&x)?, 0.01)?;
        }
        self.projection.forward(&x)
    }
}

/// Fixed sin/cos table stored in the checkpoint as `(seq, 1, dim)`; `table(n)` is `embed[:n, 0, :]`.
pub struct PositionEmbedding {
    embed: Array,
}

impl PositionEmbedding {
    pub fn load(w: &mut Weights, p: &str) -> R<Self> {
        Ok(PositionEmbedding { embed: w.get(&format!("{p}.embed"))? })
    }
    pub fn table(&self, length: i32) -> R<Array> {
        index(&slice(&self.embed, 0, 0, length)?, 1, 0)
    }
}

/// `torch.nn.MultiheadAttention` (batch_first) on fused SDPA, with `mx.split` for q/k/v.
pub struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: i32,
}

impl MultiheadAttention {
    fn forward(&self, x: &Array, mask: Option<&Array>) -> R<Array> {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2));
        let hd = d / self.num_heads;
        let qkv = self.in_proj.forward(x)?.split_equal(3, -1)?;
        let heads = |t: &Array| t.reshape(&[b, l, self.num_heads, hd])?.transpose_axes(&[0, 2, 1, 3]);
        let (q, k, v) = (heads(&qkv[0])?, heads(&qkv[1])?, heads(&qkv[2])?);
        let scale = (1.0 / (hd as f64).sqrt()) as f32; // Python: math.sqrt in float64
        let out = match mask {
            Some(m) => fast::scaled_dot_product_attention(&q, &k, &v, scale, m, None)?,
            None => fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?,
        };
        self.out_proj.forward(&out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, d])?)
    }
}

/// Post-norm encoder layer (ReLU FFN of width 2048, eps 1e-5).
pub struct TransformerEncoderLayer {
    attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    fn load(w: &mut Weights, p: &str, num_heads: i32) -> R<Self> {
        Ok(TransformerEncoderLayer {
            attn: MultiheadAttention {
                in_proj: Linear::load(w, &format!("{p}.self_attn.in_proj"))?,
                out_proj: Linear::load(w, &format!("{p}.self_attn.out_proj"))?,
                num_heads,
            },
            linear1: Linear::load(w, &format!("{p}.linear1"))?,
            linear2: Linear::load(w, &format!("{p}.linear2"))?,
            norm1: LayerNorm::load(w, &format!("{p}.norm1"))?,
            norm2: LayerNorm::load(w, &format!("{p}.norm2"))?,
        })
    }
    fn forward(&self, x: &Array, mask: Option<&Array>) -> R<Array> {
        let x = self.norm1.forward(&(x + self.attn.forward(x, mask)?))?;
        let ffn = self.linear2.forward(&relu(&self.linear1.forward(&x)?)?)?;
        self.norm2.forward(&(&x + ffn))
    }
}

pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    pub fn load(w: &mut Weights, p: &str, num_layers: usize, num_heads: i32) -> R<Self> {
        let layers = (0..num_layers)
            .map(|i| TransformerEncoderLayer::load(w, &format!("{p}.layers.{i}"), num_heads))
            .collect::<R<_>>()?;
        Ok(TransformerEncoder { layers })
    }
    /// `key_padding_mask`: bool `(B, L)`, true = padded key (PyTorch semantics).
    pub fn forward(&self, x: &Array, key_padding_mask: Option<&Array>) -> R<Array> {
        let mask = match key_padding_mask {
            Some(pad) => {
                let (b, l) = (pad.dim(0), pad.dim(1));
                Some(pad.logical_not()?.reshape(&[b, 1, 1, l])?)
            }
            None => None,
        };
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, mask.as_ref())?;
        }
        Ok(x)
    }
}

fn apply_mask(x: &Array, mask: Option<&Array>) -> R<Array> {
    match mask {
        Some(m) => ops::multiply(x, m),
        None => Ok(x.clone()),
    }
}

/// act -> dilated conv(k=3) -> act -> conv(k=1), residual, token masking (norm = identity).
pub struct ResConv1dBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResConv1dBlock {
    fn forward(&self, x: &Array, mask: Option<&Array>) -> R<Array> {
        let x = apply_mask(x, mask)?;
        let residual = x.clone();
        let h = apply_mask(&relu(&x)?, mask)?;
        let h = self.conv1.forward(&h)?;
        let h = apply_mask(&relu(&h)?, mask)?;
        let h = ops::add(self.conv2.forward(&h)?, &residual)?;
        apply_mask(&h, mask)
    }
}

/// Residual stack with reversed dilations: `model.{i}` has dilation `rate^(depth-1-i)`.
pub struct Resnet1d {
    blocks: Vec<ResConv1dBlock>,
}

impl Resnet1d {
    fn load(w: &mut Weights, p: &str, depth: usize, rate: i32) -> R<Self> {
        let blocks = (0..depth)
            .map(|i| {
                let dilation = rate.pow((depth - 1 - i) as u32);
                Ok(ResConv1dBlock {
                    conv1: Conv1d::load(w, &format!("{p}.model.{i}.conv1"), dilation, dilation)?,
                    conv2: Conv1d::load(w, &format!("{p}.model.{i}.conv2"), 0, 1)?,
                })
            })
            .collect::<R<_>>()?;
        Ok(Resnet1d { blocks })
    }
    fn forward(&self, x: &Array, mask: Option<&Array>) -> R<Array> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, mask)?;
        }
        Ok(x)
    }
}

/// Upsampling conv decoder conditioned on dense target poses and external per-frame features
/// (both present in every released decoder; `cond_fusion_last_layer` is false).
pub struct DoubleCondDecoder {
    down_t: i32,
    width: i32,
    input_conv: Conv1d,                          // model.0 (+ ReLU model.1)
    up: Vec<(Resnet1d, Conv1d)>,                 // model.{2+i}.0 / .2 (Upsample2x is .1)
    post_conv: Conv1d,                           // model.{2+down_t} (+ ReLU)
    out_conv: Conv1d,                            // model.{4+down_t}
    external: Vec<Linear>,                       // external_cond_blocks.{2i}
    target: Vec<Linear>,                         // target_cond_blocks.{2i}
}

impl DoubleCondDecoder {
    pub fn load(w: &mut Weights, p: &str, down_t: i32, width: i32, depth: usize, rate: i32) -> R<Self> {
        let up = (0..down_t)
            .map(|i| {
                Ok((
                    Resnet1d::load(w, &format!("{p}.model.{}.0", 2 + i), depth, rate)?,
                    Conv1d::load(w, &format!("{p}.model.{}.2", 2 + i), 1, 1)?,
                ))
            })
            .collect::<R<_>>()?;
        let levels = down_t; // cond_fusion_last_layer = False
        Ok(DoubleCondDecoder {
            down_t,
            width,
            input_conv: Conv1d::load(w, &format!("{p}.model.0"), 1, 1)?,
            up,
            post_conv: Conv1d::load(w, &format!("{p}.model.{}", 2 + down_t), 1, 1)?,
            out_conv: Conv1d::load(w, &format!("{p}.model.{}", 4 + down_t), 1, 1)?,
            external: (0..levels).map(|i| Linear::load(w, &format!("{p}.external_cond_blocks.{}", 2 * i))).collect::<R<_>>()?,
            target: (0..levels).map(|i| Linear::load(w, &format!("{p}.target_cond_blocks.{}", 2 * i))).collect::<R<_>>()?,
        })
    }

    /// x `(B, P, C)`; external/target `(B, P * 2^down_t, D)`; has_target bool `(B, P * 2^down_t)`;
    /// token_mask bool `(B, P)`.
    pub fn forward(&self, x: &Array, external: &Array, target: &Array, has_target: &Array, token_mask: &Array) -> R<Array> {
        let batch = x.dim(0);
        let mut mask = unsq(token_mask, -1)?.as_dtype(x.dtype())?;
        let x = apply_mask(x, Some(&mask))?;
        let mut h = relu(&self.input_conv.forward(&x)?)?;
        let has_target = unsq(has_target, -1)?;
        for i in 0..self.down_t as usize {
            let fpp = 1 << (self.down_t as usize - i);
            let positions = h.dim(1);
            let h_target = relu(&self.target[i].forward(target)?)?;
            h = h.reshape(&[batch, positions * fpp, self.width / fpp])?;
            h = wh(&has_target, &h_target, &h)?;
            h = h.reshape(&[batch, positions, self.width])?;
            let h_cond = external.reshape(&[batch, positions, -1])?;
            h = relu(&self.external[i].forward(&cat(&[&h, &h_cond], -1)?)?)?;
            let (resnet, conv) = &self.up[i];
            h = ops::repeat_axis::<f32>(resnet.forward(&h, Some(&mask))?, 2, 1)?;
            mask = ops::repeat_axis::<f32>(&mask, 2, 1)?;
            h = conv.forward(&apply_mask(&h, Some(&mask))?)?;
        }
        let h = apply_mask(&h, Some(&mask))?;
        let h = relu(&self.post_conv.forward(&h)?)?;
        let h = apply_mask(&h, Some(&mask))?;
        let h = self.out_conv.forward(&h)?;
        apply_mask(&h, Some(&mask))
    }
}
