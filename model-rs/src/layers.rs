// SPDX-License-Identifier: Apache-2.0
// Rust port of motionbricks_mlx (MLX port of NVlabs/motionbricks).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Building blocks, ported from `motionbricks_mlx/layers.py`. Channels-last `(B, T, C)`.

use mlx_rs::error::Exception;
use mlx_rs::ops::{QuantizeOptions, QuantizedMatmulOptions, quantize, quantized_matmul};
use mlx_rs::{Array, Dtype, fast, ops};

use crate::ops::*;
use crate::weights::Weights;

/// Affine-quantized weight (`mlx.nn.QuantizedLinear`): packed values, per-group scales and biases.
pub struct Quantized {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: i32,
    pub bits: i32,
}

pub struct Linear {
    pub weight: Array,
    pub bias: Array,
    /// Set by [`Linear::quantize`]; then `weight` is unused.
    pub quantized: Option<Quantized>,
}

impl Linear {
    pub fn new(weight: Array, bias: Array) -> Self {
        Linear { weight, bias, quantized: None }
    }
    pub fn load(w: &mut Weights, p: &str) -> R<Self> {
        Ok(Linear::new(w.get(&format!("{p}.weight"))?, w.get(&format!("{p}.bias"))?))
    }
    pub fn forward(&self, x: &Array) -> R<Array> {
        match &self.quantized {
            None => linear(x, &self.weight, &self.bias),
            Some(q) => {
                let options = QuantizedMatmulOptions { transpose: true, group_size: Some(q.group_size), bits: Some(q.bits), ..Default::default() };
                let y = quantized_matmul(x, &q.weight, &q.scales, &q.biases, options).map_err(|e| Exception::custom(format!("{e:?}")))?;
                ops::add(y, &self.bias)
            }
        }
    }
    /// Affine quantization like `mlx.nn.quantize` (the bias stays in the compute dtype).
    pub fn quantize(&mut self, group_size: i32, bits: i32) -> R<()> {
        let options = QuantizeOptions { group_size: Some(group_size), bits: Some(bits), ..Default::default() };
        let (weight, scales, biases) = quantize(&self.weight, options).map_err(|e| Exception::custom(format!("{e:?}")))?;
        let biases = biases.ok_or_else(|| Exception::custom("affine quantization returned no biases"))?;
        self.weight = ops::zeros::<f32>(&[0])?;
        self.quantized = Some(Quantized { weight, scales, biases, group_size, bits });
        Ok(())
    }
    /// Parameters, for evaluating them once (`mx.eval(model.parameters())`).
    pub fn parameters(&self) -> Vec<&Array> {
        match &self.quantized {
            None => vec![&self.weight, &self.bias],
            Some(q) => vec![&q.weight, &q.scales, &q.biases, &self.bias],
        }
    }
}

/// Fixed sine/cosine position encoding `(seq_length, dim)`, like `layers.sincos_position_encoding`
/// (the checkpoints also store it).
pub fn sincos_position_encoding(seq_length: i32, dim: i32) -> R<Array> {
    let position = ops::arange::<_, f32>(0, seq_length, 1)?.reshape(&[seq_length, 1])?;
    let div = ops::exp(&ops::multiply(ops::arange::<_, f32>(0, dim, 2)?, f(-(10000f64.ln() / dim as f64) as f32))?)?;
    let angle = ops::multiply(&position, &div)?; // (seq, dim/2)
    let pe = ops::stack(&[ops::sin(&angle)?, ops::cos(&angle)?], -1)?; // interleave sin/cos
    pe.reshape(&[seq_length, dim])?.as_dtype(Dtype::Float32)
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
    /// `weight` channels-last `(out, k, in)`.
    pub fn new(weight: Array, bias: Array, padding: i32, dilation: i32) -> Self {
        Conv1d { weight, bias, padding, dilation }
    }
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

/// Whether MLX 0.32's fused SDPA kernels handle this head dim and query length (otherwise
/// `fast::scaled_dot_product_attention` falls back to composite ops).
fn fused_sdpa(head_dim: i32, query_len: i32) -> bool {
    if query_len > 8 { [64, 72, 80, 96, 128, 192, 256].contains(&head_dim) } else { [64, 96, 128, 192, 256].contains(&head_dim) }
}

/// Most negative finite value of `dtype` (MLX's mask fill, `finfo(dtype).min`).
fn finfo_min(dtype: Dtype) -> Array {
    match dtype {
        Dtype::Float16 => Array::from_f32(-65504.0).as_dtype(dtype).unwrap(),
        Dtype::Bfloat16 => Array::from_f32(-3.389_531_4e38).as_dtype(dtype).unwrap(),
        _ => Array::from_f32(f32::MIN),
    }
}

/// `torch.nn.MultiheadAttention` (batch_first) on fused SDPA, with `mx.split` for q/k/v.
///
/// For head dims MLX's fused kernels don't cover (the root networks' 32), the composite
/// fallback MLX would run is written out here, op for op, with one difference: the softmax
/// scale is a `(head_dim,)` array made at load time instead of a scalar. MLX 0.32.2 (mlx-sys
/// 0.7) prints scalar constants into compiled kernels with too few digits, so the fallback's
/// `q * scale` came out one ulp off under `mx.compile` and compiled runs drifted from eager
/// ones. Eager results are bitwise unchanged.
pub struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: i32,
    scale: f32,
    scale_vec: Array,
}

impl MultiheadAttention {
    fn load(w: &mut Weights, p: &str, num_heads: i32) -> R<Self> {
        let in_proj = Linear::load(w, &format!("{p}.in_proj"))?;
        let head_dim = in_proj.weight.dim(1) / num_heads;
        let scale = (1.0 / (head_dim as f64).sqrt()) as f32; // Python: math.sqrt in float64
        let scale_vec = ops::full::<f32>(&[head_dim], &Array::from_f32(scale))?.as_dtype(w.dtype)?;
        scale_vec.eval()?;
        Ok(MultiheadAttention { in_proj, out_proj: Linear::load(w, &format!("{p}.out_proj"))?, num_heads, scale, scale_vec })
    }

    fn forward(&self, x: &Array, mask: Option<&Array>) -> R<Array> {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2));
        let hd = d / self.num_heads;
        let qkv = self.in_proj.forward(x)?.split_equal(3, -1)?;
        let heads = |t: &Array| t.reshape(&[b, l, self.num_heads, hd])?.transpose_axes(&[0, 2, 1, 3]);
        let (q, k, v) = (heads(&qkv[0])?, heads(&qkv[1])?, heads(&qkv[2])?);
        let out = if fused_sdpa(hd, l) {
            match mask {
                Some(m) => fast::scaled_dot_product_attention(&q, &k, &v, self.scale, m, None)?,
                None => fast::scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?,
            }
        } else {
            let q = ops::multiply(&self.scale_vec, &q)?;
            let mut scores = ops::matmul(&q, &k.swap_axes(-1, -2)?)?;
            if let Some(m) = mask {
                scores = wh(m, &scores, &finfo_min(scores.dtype()))?;
            }
            ops::matmul(&ops::softmax_axis(&scores, -1, true)?, &v)?
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
            attn: MultiheadAttention::load(w, &format!("{p}.self_attn"), num_heads)?,
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
    /// Quantize the four projections of every layer (`mlx.nn.quantize` on the encoder).
    pub fn quantize(&mut self, group_size: i32, bits: i32) -> R<()> {
        for l in &mut self.layers {
            for lin in [&mut l.attn.in_proj, &mut l.attn.out_proj, &mut l.linear1, &mut l.linear2] {
                lin.quantize(group_size, bits)?;
            }
        }
        Ok(())
    }
    pub fn parameters(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(|l| {
                let mut v = l.attn.in_proj.parameters();
                v.extend(l.attn.out_proj.parameters());
                v.extend(l.linear1.parameters());
                v.extend(l.linear2.parameters());
                v.extend([&l.norm1.weight, &l.norm1.bias, &l.norm2.weight, &l.norm2.bias]);
                v
            })
            .collect()
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
    pub conv1: Conv1d,
    pub conv2: Conv1d,
}

impl ResConv1dBlock {
    /// `mask`: `(B, T, 1)` float, 1 = valid frame.
    pub fn forward(&self, x: &Array, mask: Option<&Array>) -> R<Array> {
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
