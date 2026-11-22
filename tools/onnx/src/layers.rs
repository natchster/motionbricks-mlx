// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks tools/onnx (ONNX exporter).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! Shared building blocks, ported from upstream `tools/onnx/models/layers.py` (and
//! `models/soma/layers.py`) following the op sequences of the verified
//! `model-rs/src/layers.rs`. The conv blocks run channels-first `[B, C, T]` (ONNX Conv
//! layout, like the PyTorch original); the [`DoubleCondDecoder`] transposes to
//! channels-last for the conditioning stages exactly where upstream does.
//!
//! Scope convention: every `emit` takes the module's own scope (e.g.
//! `/core/backbone_net/transformer.layers.0`); sub-module emits extend it as
//! `{scope}/{child}` with the checkpoint key's last component (`self_attn`, `in_proj`,
//! `conv1`, `model.0`, ...), so node names mirror the traced reference graphs.
//! Initializers keep their checkpoint key as their name (unique by construction).

use anyhow::Result;

use crate::graph::{DType, Dim, Graph};
use crate::ops;
use crate::weights::Weights;

/// `nn.Linear` as `MatMul` + `Add` (the weight is stored pre-transposed `[in, out]`
/// under the checkpoint key, like upstream's constant folding).
pub struct Linear {
    pub w: String,
    pub b: String,
    pub out_dim: i64,
}

impl Linear {
    /// Reads `{key}.weight` `[out, in]` (registered via [`ops::matmul_weight`]) and
    /// `{key}.bias`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str) -> Result<Linear> {
        let wname = format!("{key}.weight");
        let t = w.get(&wname)?;
        let (out_dim, in_dim) = (t.shape[0], t.shape[1]);
        let wt = ops::matmul_weight(g, &wname, out_dim, in_dim, &t.data);
        let bname = format!("{key}.bias");
        let b = w.get(&bname)?;
        let bt = ops::c_f32(g, &bname, &b.shape, &b.data);
        Ok(Linear { w: wt, b: bt, out_dim })
    }

    /// `[.., in] -> [.., out_dim]`.
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str) -> String {
        let y = ops::matmul(g, scope, x, &self.w);
        ops::add(g, scope, &y, &self.b)
    }
}

/// `nn.LayerNorm` (eps 1e-5).
pub struct LayerNorm {
    pub scale: String,
    pub bias: String,
}

impl LayerNorm {
    /// Reads `{key}.weight`, `{key}.bias`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str) -> anyhow::Result<Self> {
        let wname = format!("{key}.weight");
        let t = w.get(&wname)?;
        let scale = ops::c_f32(g, &wname, &t.shape, &t.data);
        let bname = format!("{key}.bias");
        let t = w.get(&bname)?;
        let bias = ops::c_f32(g, &bname, &t.shape, &t.data);
        Ok(LayerNorm { scale, bias })
    }

    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str) -> String {
        ops::layer_norm(g, scope, x, &self.scale, &self.bias, 1e-5)
    }
}

/// `nn.Embedding`: `weight[ids]`.
pub struct Embedding {
    pub table: String,
}

impl Embedding {
    /// Reads `{key}.weight` `[num_embeddings, dim]`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str) -> anyhow::Result<Self> {
        let wname = format!("{key}.weight");
        let t = w.get(&wname)?;
        let table = ops::c_f32(g, &wname, &t.shape, &t.data);
        Ok(Embedding { table })
    }

    /// `Gather` at axis 0: `ids` `[..]` -> `[.., dim]`.
    pub fn emit(&self, g: &mut Graph, scope: &str, ids: &str) -> String {
        ops::gather(g, scope, &self.table, ids, 0)
    }
}

/// `nn.Conv1d` (stride 1), channels-first `[B, C, T]`; weight in the PyTorch layout
/// `[C_out, C_in, K]` as stored in the checkpoints.
pub struct Conv1d {
    pub w: String,
    pub b: String,
    pub padding: i64,
    pub dilation: i64,
}

impl Conv1d {
    /// Reads `{key}.weight` `[out, in, k]`, `{key}.bias`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, padding: i64, dilation: i64) -> anyhow::Result<Self> {
        let wname = format!("{key}.weight");
        let t = w.get(&wname)?;
        let wt = ops::c_f32(g, &wname, &t.shape, &t.data);
        let bname = format!("{key}.bias");
        let t = w.get(&bname)?;
        let bt = ops::c_f32(g, &bname, &t.shape, &t.data);
        Ok(Conv1d { w: wt, b: bt, padding, dilation })
    }

    /// `[B, C_in, T] -> [B, C_out, T']`, stride 1.
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str) -> String {
        ops::conv1d(g, scope, x, &self.w, &self.b, self.padding, self.dilation, 1)
    }
}

/// Stack of two `Linear` + `LeakyReLU(0.01)` layers followed by a projection
/// (upstream `FCBlock` with `num_layers = 2`).
pub struct FcBlock {
    layers: [Linear; 2],
    projection: Linear,
}

impl FcBlock {
    /// Reads `{key}.fc_layers.{0,1}` and `{key}.forward_projection`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str) -> anyhow::Result<Self> {
        let layers = [Linear::load(g, w, &format!("{key}.fc_layers.0"))?, Linear::load(g, w, &format!("{key}.fc_layers.1"))?];
        let projection = Linear::load(g, w, &format!("{key}.forward_projection"))?;
        Ok(FcBlock { layers, projection })
    }

    /// Linear -> LeakyReLU(0.01) -> Linear -> LeakyReLU(0.01) -> Linear.
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str) -> String {
        let mut h = x.to_string();
        for (i, layer) in self.layers.iter().enumerate() {
            let scope = format!("{scope}/fc_layers.{i}");
            h = layer.emit(g, &scope, &h);
            h = ops::leaky_relu(g, &scope, &h, 0.01);
        }
        let scope = format!("{scope}/forward_projection");
        self.projection.emit(g, &scope, &h)
    }
}

/// Fixed sine/cosine position encoding table stored in the checkpoint as
/// `[seq, 1, dim]`.
pub struct PositionEmbedding {
    table: String,
}

impl PositionEmbedding {
    /// Reads `{key}.embed` `[seq, 1, dim]`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str) -> anyhow::Result<Self> {
        let name = format!("{key}.embed");
        let t = w.get(&name)?;
        let table = ops::c_f32(g, &name, &t.shape, &t.data);
        Ok(PositionEmbedding { table })
    }

    /// Rows `[start, start+len)` squeezed to `[1, len, dim]` for broadcasting onto
    /// `[B, len, dim]`.
    pub fn emit_rows(&self, g: &mut Graph, scope: &str, start: i64, len: i64) -> String {
        let rows = ops::slice_axis(g, scope, &self.table, 0, start, start + len); // [len, 1, dim]
        let rows = ops::squeeze(g, scope, &rows, &[1]); // [len, dim]
        ops::unsqueeze(g, scope, &rows, &[0]) // [1, len, dim]
    }
}

/// `torch.nn.MultiheadAttention` (batch_first), self-attention only. The packed
/// `in_proj` `[3D, D]` is applied as one MatMul followed by a `Split` into q/k/v
/// (matching upstream's fused projection); q is scaled once by `1/sqrt(head_dim)`
/// before `QK^T` (SDPA style, numerically equal to torch's two-sided scaling).
pub struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    n_head: i64,
    d_model: i64,
}

impl MultiheadAttention {
    /// Reads `{key}.in_proj.{weight,bias}` (packed `[3D, D]`) and
    /// `{key}.out_proj.{weight,bias}`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, n_head: i64) -> anyhow::Result<Self> {
        let in_proj = Linear::load(g, w, &format!("{key}.in_proj"))?;
        let out_proj = Linear::load(g, w, &format!("{key}.out_proj"))?;
        assert_eq!(in_proj.out_dim % 3, 0, "{key}.in_proj is not packed [3D, D]");
        Ok(MultiheadAttention { d_model: in_proj.out_dim / 3, in_proj, out_proj, n_head })
    }

    /// `x`: `[B, L, D]` batch-first. `key_padding_mask`: bool `[B, L]`, True = PAD
    /// (excluded from attention); emitted as an additive `where(pad, -inf, 0)` bias of
    /// shape `[B, 1, 1, L]` on the attention scores, like torch's MHA.
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str, key_padding_mask: Option<&str>) -> String {
        let (d, h) = (self.d_model, self.n_head);
        let hd = d / h;
        let shape = g.shape(x).to_vec(); // [B, L, D]
        let (b, l) = (shape[0].clone(), shape[1].clone());

        let qkv = self.in_proj.emit(g, &format!("{scope}/in_proj"), x); // [B, L, 3D]
        let qkv = ops::split(g, scope, &qkv, 2, &[d, d, d]);
        let heads = |g: &mut Graph, t: &str| {
            let t = ops::reshape(g, scope, t, &[b.clone(), l.clone(), Dim::Static(h), Dim::Static(hd)]);
            ops::transpose(g, scope, &t, &[0, 2, 1, 3]) // [B, H, L, hd]
        };
        let q = heads(g, &qkv[0]);
        let k = heads(g, &qkv[1]);
        let v = heads(g, &qkv[2]);

        let scale = (1.0 / (hd as f64).sqrt()) as f32; // Python: math.sqrt in float64
        let scale = ops::c_scalar(g, scope, scale);
        let q = ops::mul(g, scope, &q, &scale);
        let kt = ops::transpose(g, scope, &k, &[0, 1, 3, 2]); // [B, H, hd, L]
        let mut scores = ops::matmul(g, scope, &q, &kt); // [B, H, L, L]
        if let Some(mask) = key_padding_mask {
            let neg_inf = ops::full_f32(g, scope, "neg_inf", &[], f32::NEG_INFINITY);
            let zero = ops::c_scalar(g, scope, 0.0);
            let bias = ops::where_(g, scope, mask, &neg_inf, &zero); // [B, L]
            let bias = ops::unsqueeze(g, scope, &bias, &[1, 2]); // [B, 1, 1, L]
            scores = ops::add(g, scope, &scores, &bias);
        }
        let probs = ops::softmax(g, scope, &scores, -1);
        let out = ops::matmul(g, scope, &probs, &v); // [B, H, L, hd]
        let out = ops::transpose(g, scope, &out, &[0, 2, 1, 3]); // [B, L, H, hd]
        let out = ops::reshape(g, scope, &out, &[b, l, Dim::Static(d)]);
        self.out_proj.emit(g, &format!("{scope}/out_proj"), &out)
    }
}

/// Post-norm encoder layer (ReLU FFN of width 2048, eps 1e-5):
/// `x = LN(x + MHA(x))`, then `x = LN(x + linear2(relu(linear1(x))))`.
pub struct TransformerEncoderLayer {
    self_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    /// Reads `{key}.{self_attn,linear1,linear2,norm1,norm2}`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, n_head: i64) -> anyhow::Result<Self> {
        Ok(TransformerEncoderLayer {
            self_attn: MultiheadAttention::load(g, w, &format!("{key}.self_attn"), n_head)?,
            linear1: Linear::load(g, w, &format!("{key}.linear1"))?,
            linear2: Linear::load(g, w, &format!("{key}.linear2"))?,
            norm1: LayerNorm::load(g, w, &format!("{key}.norm1"))?,
            norm2: LayerNorm::load(g, w, &format!("{key}.norm2"))?,
        })
    }

    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str, key_padding_mask: Option<&str>) -> String {
        let attn = self.self_attn.emit(g, &format!("{scope}/self_attn"), x, key_padding_mask);
        let x = ops::add(g, scope, x, &attn);
        let x = self.norm1.emit(g, &format!("{scope}/norm1"), &x);
        let ffn = self.linear1.emit(g, &format!("{scope}/linear1"), &x);
        let ffn = ops::relu(g, &format!("{scope}/linear1"), &ffn);
        let ffn = self.linear2.emit(g, &format!("{scope}/linear2"), &ffn);
        let x = ops::add(g, scope, &x, &ffn);
        self.norm2.emit(g, &format!("{scope}/norm2"), &x)
    }
}

/// A stack of [`TransformerEncoderLayer`]s.
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    /// Reads `{key}.layers.{i}` for `i` in `0..num_layers`.
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, num_layers: usize, n_head: i64) -> anyhow::Result<Self> {
        let layers = (0..num_layers)
            .map(|i| TransformerEncoderLayer::load(g, w, &format!("{key}.layers.{i}"), n_head))
            .collect::<Result<_>>()?;
        Ok(TransformerEncoder { layers })
    }

    /// `key_padding_mask`: bool `[B, L]`, True = PAD (PyTorch semantics).
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str, key_padding_mask: Option<&str>) -> String {
        let mut x = x.to_string();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.emit(g, &format!("{scope}/layers.{i}"), &x, key_padding_mask);
        }
        x
    }
}

/// `token_mask [B, T] * x [B, C, T]` via a `[B, 1, T]` unsqueeze; identity without a mask.
fn apply_mask(g: &mut Graph, scope: &str, x: &str, token_mask: Option<&str>) -> String {
    match token_mask {
        Some(m) => {
            let m = ops::unsqueeze(g, scope, m, &[1]); // [B, 1, T]
            ops::mul(g, scope, x, &m)
        }
        None => x.to_string(),
    }
}

/// act -> dilated conv(k=3) -> act -> conv(k=1), residual, token masking (norm =
/// identity), in the order of `model-rs/src/layers.rs::ResConv1dBlock`:
/// mask(x) -> relu -> mask -> conv1 -> relu -> mask -> conv2 -> +residual -> mask.
pub struct ResConv1dBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResConv1dBlock {
    /// Reads `{key}.conv1` (k=3, pad = dil = `dilation`) and `{key}.conv2` (k=1).
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, dilation: i64) -> anyhow::Result<Self> {
        Ok(ResConv1dBlock {
            conv1: Conv1d::load(g, w, &format!("{key}.conv1"), dilation, dilation)?,
            conv2: Conv1d::load(g, w, &format!("{key}.conv2"), 0, 1)?,
        })
    }

    /// `x`: `[B, C, T]`; `token_mask`: f32 `[B, T]`, 1 = valid frame.
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str, token_mask: Option<&str>) -> String {
        let x = apply_mask(g, scope, x, token_mask);
        let residual = x.clone();
        let r = ops::relu(g, scope, &x);
        let h = apply_mask(g, scope, &r, token_mask);
        let h = self.conv1.emit(g, &format!("{scope}/conv1"), &h);
        let r = ops::relu(g, scope, &h);
        let h = apply_mask(g, scope, &r, token_mask);
        let h = self.conv2.emit(g, &format!("{scope}/conv2"), &h);
        let h = ops::add(g, scope, &h, &residual);
        apply_mask(g, scope, &h, token_mask)
    }
}

/// Residual stack with reversed dilations: block `i` (`{key}.model.{i}`) has dilation
/// `rate^(depth-1-i)` (for depth 4, rate 3: 27, 9, 3, 1).
pub struct Resnet1d {
    blocks: Vec<ResConv1dBlock>,
}

impl Resnet1d {
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, depth: usize, rate: i64) -> anyhow::Result<Self> {
        let blocks = (0..depth)
            .map(|i| ResConv1dBlock::load(g, w, &format!("{key}.model.{i}"), rate.pow((depth - 1 - i) as u32)))
            .collect::<Result<_>>()?;
        Ok(Resnet1d { blocks })
    }

    /// `x`: `[B, C, T]`; `token_mask`: f32 `[B, T]`, 1 = valid frame.
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str, token_mask: Option<&str>) -> String {
        let mut x = x.to_string();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.emit(g, &format!("{scope}/model.{i}"), &x, token_mask);
        }
        x
    }
}

/// `repeat_interleave(2)` on the last axis of a `[B, T]` mask: `[B, 2T]` with the
/// pattern `[m0, m0, m1, m1, ...]` (matching the nearest-neighbour `upsample2x` of the
/// features — a plain `Tile` would repeat the whole axis instead).
fn repeat_interleave2(g: &mut Graph, scope: &str, x: &str) -> String {
    let s = g.shape(x).to_vec();
    let t = s[1].clone();
    let x = ops::unsqueeze(g, scope, x, &[2]); // [B, T, 1]
    let x = ops::expand(g, scope, &x, &[s[0].clone(), t.clone(), Dim::Static(2)]); // [B, T, 2]
    let t2 = match t {
        Dim::Static(t) => Dim::Static(2 * t),
        other => other,
    };
    ops::reshape(g, scope, &x, &[s[0].clone(), t2])
}

/// Upsampling conv decoder conditioned on dense target poses and external per-frame
/// features (upstream `DoubleCondDecoder` with `cond_fusion_last_layer = false`, both
/// conds present). Channels-first `[B, C, T]` inside, like the PyTorch original.
pub struct DoubleCondDecoder {
    down_t: i64,
    width: i64,
    input_conv: Conv1d,              // model.0 (+ ReLU model.1)
    up: Vec<(Resnet1d, Conv1d)>,     // model.{2+i}.0 / .2 (Upsample2x is .1)
    post_conv: Conv1d,               // model.{2+down_t} (+ ReLU model.{3+down_t})
    out_conv: Conv1d,                // model.{4+down_t}
    external: Vec<Linear>,           // external_cond_blocks.{2i} (+ ReLU at {2i+1})
    target: Vec<Linear>,             // target_cond_blocks.{2i} (+ ReLU at {2i+1})
}

impl DoubleCondDecoder {
    pub fn load(g: &mut Graph, w: &mut Weights, key: &str, down_t: i64, width: i64, depth: usize, rate: i64) -> anyhow::Result<Self> {
        let up = (0..down_t)
            .map(|i| {
                Ok((
                    Resnet1d::load(g, w, &format!("{key}.model.{}.0", 2 + i), depth, rate)?,
                    Conv1d::load(g, w, &format!("{key}.model.{}.2", 2 + i), 1, 1)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let levels = down_t; // cond_fusion_last_layer = false
        Ok(DoubleCondDecoder {
            down_t,
            width,
            input_conv: Conv1d::load(g, w, &format!("{key}.model.0"), 1, 1)?,
            up,
            post_conv: Conv1d::load(g, w, &format!("{key}.model.{}", 2 + down_t), 1, 1)?,
            out_conv: Conv1d::load(g, w, &format!("{key}.model.{}", 4 + down_t), 1, 1)?,
            external: (0..levels).map(|i| Linear::load(g, w, &format!("{key}.external_cond_blocks.{}", 2 * i))).collect::<Result<_>>()?,
            target: (0..levels).map(|i| Linear::load(g, w, &format!("{key}.target_cond_blocks.{}", 2 * i))).collect::<Result<_>>()?,
        })
    }

    /// `x`: `[B, width, T]`; `external`: `[B, T * 2^down_t, ext_dim]`; `target`:
    /// `[B, T * 2^down_t, target_dim]`; `has_target`: f32 or bool `[B, T * 2^down_t]`
    /// (cast to bool if needed); `token_mask`: f32 `[B, T]`, 1 = valid frame. Returns
    /// `[B, T * 2^down_t, out_dim]` channels-last (the `model-rs` layout; upstream's
    /// PyTorch module returns channels-first, so a final Transpose converts).
    pub fn emit(&self, g: &mut Graph, scope: &str, x: &str, external: &str, target: &str, has_target: &str, token_mask: &str) -> String {
        let b = g.shape(x)[0].clone();
        let width = self.width;
        // SOMA casts the (numeric) token mask to the activation dtype before
        // multiplying; the Cast is emitted unconditionally (a no-op for f32).
        let mut mask = ops::cast(g, scope, token_mask, g.dtype(x)); // [B, T]
        let m3 = ops::unsqueeze(g, scope, &mask, &[1]); // [B, 1, T]
        let mut h = ops::mul(g, scope, x, &m3);
        let scope0 = format!("{scope}/model.0");
        h = self.input_conv.emit(g, &scope0, &h);
        h = ops::relu(g, &scope0, &h);

        let has_target = match g.dtype(has_target) {
            DType::Bool => has_target.to_string(),
            _ => ops::cast(g, scope, has_target, DType::Bool),
        };
        let has_target = ops::unsqueeze(g, scope, &has_target, &[2]); // [B, T*2^down_t, 1]

        for i in 0..self.down_t as usize {
            let fpp = 1i64 << (self.down_t as usize - i); // frames per position
            let positions = g.shape(&h)[2].get().expect("static time dim");
            // Target-cond replacement: project the dense target, split the hidden
            // channels over the position's frames, then pick per frame.
            let h_target = self.target[i].emit(g, &format!("{scope}/target_cond_blocks.{}", 2 * i), target);
            let h_target = ops::relu(g, &format!("{scope}/target_cond_blocks.{}", 2 * i + 1), &h_target); // [B, Ttot, width/fpp]
            h = ops::transpose(g, scope, &h, &[0, 2, 1]); // [B, pos, width]
            h = ops::reshape(g, scope, &h, &[b.clone(), Dim::Static(positions * fpp), Dim::Static(width / fpp)]);
            h = ops::where_(g, scope, &has_target, &h_target, &h);
            h = ops::reshape(g, scope, &h, &[b.clone(), Dim::Static(positions), Dim::Static(width)]);
            h = ops::transpose(g, scope, &h, &[0, 2, 1]); // [B, width, pos]
            // External-cond fusion: concat channels-last + Linear + ReLU.
            let ext_dim = g.shape(external)[2].get().expect("static external dim");
            let h_cond = ops::reshape(g, scope, external, &[b.clone(), Dim::Static(positions), Dim::Static(ext_dim * fpp)]);
            h = ops::transpose(g, scope, &h, &[0, 2, 1]); // [B, pos, width]
            h = ops::concat(g, scope, &[&h, &h_cond], 2);
            h = self.external[i].emit(g, &format!("{scope}/external_cond_blocks.{}", 2 * i), &h);
            h = ops::transpose(g, scope, &h, &[0, 2, 1]); // [B, width, pos]
            h = ops::relu(g, &format!("{scope}/external_cond_blocks.{}", 2 * i + 1), &h);
            // Resnet at the current resolution, then 2x upsample + conv.
            let (resnet, conv) = &self.up[i];
            h = resnet.emit(g, &format!("{scope}/model.{}.0", 2 + i), &h, Some(&mask));
            h = ops::upsample2x(g, &format!("{scope}/model.{}.1", 2 + i), &h);
            mask = repeat_interleave2(g, scope, &mask);
            let m3 = ops::unsqueeze(g, scope, &mask, &[1]);
            h = ops::mul(g, scope, &h, &m3);
            h = conv.emit(g, &format!("{scope}/model.{}.2", 2 + i), &h);
        }
        let m3 = ops::unsqueeze(g, scope, &mask, &[1]);
        h = ops::mul(g, scope, &h, &m3);
        let post = format!("{scope}/model.{}", 2 + self.down_t);
        h = self.post_conv.emit(g, &post, &h);
        h = ops::relu(g, &post, &h);
        h = ops::mul(g, scope, &h, &m3);
        h = self.out_conv.emit(g, &format!("{scope}/model.{}", 4 + self.down_t), &h);
        h = ops::mul(g, scope, &h, &m3);
        ops::transpose(g, scope, &h, &[0, 2, 1]) // [B, T*2^down_t, out_dim]
    }
}
