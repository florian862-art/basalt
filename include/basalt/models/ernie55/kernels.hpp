// include/basalt/models/ernie55/kernels.hpp - ERNIE-5.5-Flash CUDA kernel declarations.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::ernie55 {

constexpr int kHcMult = 4;
constexpr int kHcCoef = 24;   // 4 pre + 4 post + 16 comb
constexpr int kHcSplits = 16;

struct RopeTable {
    float step[32];   ///< frequency per rotary pair (rope_dim = 64 -> 32 pairs)
    float mscale = 1.0f;
};

RopeTable make_rope_table(float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, int n_ctx_orig);
RopeTable rope_table_plain(float freq_base);
RopeTable rope_table_yarn(float freq_base, float factor, float beta_fast, float beta_slow, int n_ctx_orig);

// ---- mHC (ERNIE 5.5 exact RMS factor: 1 / (sqrt(mean) + rms_eps)) ---------------------------------------------

size_t mhc_workspace_bytes(int n_tokens);

/// Computes 24 mHC coefficients [pre(4), post(4), comb(16)] per token from x [n_tokens, 4 * n_embd].
void mhc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
               int n_embd, float rms_eps, float hc_eps, int sinkhorn_iters, float* workspace, float* coef,
               void* stream);

/// Collapses 4 residual streams x [n_tokens, 4, n_embd] -> out [n_tokens, n_embd] using pre = coef[:, 0:4].
void mhc_pre(const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd, float* out, void* stream);

/// Updates 4 residual streams: out[d] = sum_s comb[s, d] * x[s] + post[d] * f (matching vLLM mhc_post_torch).
void mhc_post(const float* f, const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd,
              float* out, void* stream);

/// Final mHC stream contraction: hc_head_fn [4, 4*H], hc_head_base [4], hc_head_scale [1] -> out [n_tokens, H].
void mhc_contract(const float* x, const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                  int n_tokens, int n_embd, float rms_eps, float hc_eps, float* workspace, float* coef4, float* out,
                  void* stream);

// ---- Elementwise & ERNIE-specific building blocks -------------------------------------------------------------

/// Broadcasts single-stream x [n_tokens, n_embd] to 4 identical mHC streams out [n_tokens, 4, n_embd].
void broadcast_streams(const float* x, int n_tokens, int n_embd, float* out, void* stream);

/// In-place FP32 LM-head multimax: logits += t0*relu(r0-x) + t1*relu(x-r1) + t2*relu(r2-x)^2 + t3*relu(x-r3)^2.
void multimax_f32(float* logits, const float* ranges, const float* ts, int n_tokens, int n_vocab, void* stream);

/// SiTU-GLU: out[i] = (beta * tanh(g / beta) * sigmoid(g)) * (linear_beta * tanh(u / linear_beta)).
void situ_glu_f32(const float* gate, const float* up, int n_tokens, int n_ff, float beta, float linear_beta,
                  float* out, void* stream);

/// Split-feature MoE router: logits [n_tokens, 2*E] -> scores = sigmoid(l0) + sigmoid(l1),
/// bias-corrected top-K, optional renorm, multiplied by learnable per-expert scale[id].
void split_sigmoid_route(const float* logits, const float* bias, const float* scale, int n_tokens, int n_expert,
                         int top_k, bool renorm, int32_t* ids, float* weights, void* stream);

/// VHA PostMix: out[t, h, d] = sum_g mix[h, g] * in[t, g, d]  (in/out [n_tokens, n_heads, head_dim]).
void vha_postmix_f32(const float* in, const float* mix, int n_tokens, int n_heads, int head_dim, float* out,
                     void* stream);

/// In-place sigmoid output gating: z[i] *= sigmoid(gate[i]).
void sigmoid_gate_mul_f32(float* z, const float* gate, int64_t n, void* stream);

/// Standard LayerNorm (with weight and bias) over rows [n_tokens, dim].
void layernorm_f32(const float* x, const float* weight, const float* bias, int n_tokens, int dim, float eps,
                   float* out, void* stream);

/// In-place orthonormal Walsh-Hadamard transform (dim = 128, scaled by 1/sqrt(128)) on rows [n_rows, 128].
void hadamard_128_f32(float* x, int n_rows, void* stream);

/// Weighted sum of routed expert outputs: out[t, c] = sum_{k=0}^{top_k-1} w[t, k] * exp_out[t * top_k + k, c].
void weighted_sum_f32(const float* exp_out, const float* weights, int n_tokens, int top_k, int dim, float* out,
                      void* stream);

/// Residual add: dst[i] += src[i].
void add_f32(float* dst, const float* src, int64_t n, void* stream);

// ---- HCA Attention (37 layers) --------------------------------------------------------------------------------

/// Per-head RMSNorm on q_b [n_tokens, n_heads, head_dim] (if norm_w != null) + forward interleaved RoPE on last 64 dims.
void hca_q_rope(float* q, const int32_t* pos, int n_tokens, int n_heads, int head_dim, const RopeTable& rt,
                void* stream);

/// RMSNorm on kv [n_tokens, 512] + forward interleaved RoPE on last 64 dims -> kv_rot [n_tokens, 512] (in-place safe).
void hca_kv_norm_rope(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                      float rms_eps, const RopeTable& rt, float* kv_rot, void* stream);

/// Commits the trailing min(n_tokens, win_size) rows of kv_rot into sliding-window ring [win, 512] (race-free).
void hca_win_commit(const float* kv_rot, const int32_t* pos, int n_tokens, int head_dim, int win_size,
                    uint8_t* win_cache, bool fp8, void* stream);

/// RMSNorm on kv [n_tokens, 512] + forward interleaved RoPE on last 64 dims, writing into sliding-window ring [win, 512].
void hca_kv_store_window(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                         int win_size, float rms_eps, const RopeTable& rt, uint8_t* win_cache, bool fp8, void* stream);

/// Streaming 128:1 HCA compressor with additive APE[pos % ratio]: updates state [ratio, 512] x 2 and writes closed
/// compressed blocks into comp_cache (RMSNorm + YaRN RoPE at block start position).
void hca_compress(const float* ckv, const float* cscore, const float* ape, const float* norm_w, const int32_t* pos,
                  int n_tokens, int ratio, int head_dim, float rms_eps, const RopeTable& comp_rt,
                  float* state_kv, float* state_score, uint8_t* comp_cache, bool fp8, void* stream);

/// HCA multi-query attention over compressed blocks [0 .. pos/ratio) + recent window [max(0, pos-win+1) .. pos],
/// with per-head attention sink `attn_sink[n_heads]`. When `kv_rot != nullptr`, positions >= pos[0] are read from
/// `kv_rot[(j - pos[0]) * head_dim]` so prefill chunks never race on `win_cache`.
void hca_attention(const float* q, const uint8_t* comp_cache, const uint8_t* win_cache, const float* kv_rot,
                   const float* attn_sink, const int32_t* pos, int n_tokens, int n_heads, int head_dim, int win_size,
                   int ratio, bool fp8, float* out, void* stream);

/// Inverse interleaved RoPE on last 64 dims of VHA-mixed attention output [n_tokens, n_heads, head_dim].
void hca_inv_rope(float* o, const int32_t* pos, int n_tokens, int n_heads, int head_dim, const RopeTable& rt,
                  void* stream);

// ---- MLA-DSA Attention + Lightning Indexer (6 layers) ---------------------------------------------------------

/// Applies interleaved RoPE to q[..., nope:nope+rope] and k_pe [n_tokens, rope], and writes normed kv_c + rotated k_pe
/// into DSA KV cache at positions `pos[0..n_tokens)`.
void dsa_qkv_rope_and_store(float* q, const float* kv_a, const float* kv_norm_w, const int32_t* pos, int n_tokens,
                            int n_heads, int qk_nope, int qk_rope, int kv_lora, float rms_eps, const RopeTable& rt,
                            uint8_t* kv_cache, bool fp8, void* stream);

/// Prepares Lightning Indexer queries (partial NeoX RoPE + Hadamard-128) and keys (LayerNorm + partial NeoX RoPE +
/// Hadamard-128), storing keys into indexer K cache (MXFP4 when fp8, else FP16).
void dsa_indexer_prepare(float* iq, float* ik, const float* k_norm_w, const float* k_norm_b, const int32_t* pos,
                         int n_tokens, int idx_heads, int idx_dim, int rope_dim, float eps, const RopeTable& rt,
                         uint8_t* idx_k_cache, bool fp8, void* stream);

/// Scores older tokens [0 .. pos - win) with Lightning Indexer (ReLU(iq . ik) * weights), selects top-K, and merges
/// with the trailing sliding window [max(0, pos - win + 1) .. pos] into `out_idx [n_tokens, top_k + win]`.
/// `scores_buf` (optional, stride `scores_stride >= max(pos) + 1` per token) caches scores for single-pass selection.
void dsa_indexer_topk_merge(const float* iq, const float* iweights, const uint8_t* idx_k_cache, const int32_t* pos,
                            int n_tokens, int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                            int32_t* out_idx, void* stream, float* scores_buf = nullptr, int scores_stride = 0);

/// Sparse MLA attention over selected positions `sel_idx [n_tokens, width]` with `attn_sink[n_heads]`,
/// reading `kv_b` (Q8_0 [n_heads * (qk_nope + v_dim), kv_lora]) or pre-expanded per-head K/V, writing
/// `out [n_tokens, n_heads, v_dim]`.
void dsa_mla_attention(const float* q, const uint8_t* kv_cache, const void* kv_b_q8, const float* attn_sink,
                       const int32_t* sel_idx, int n_tokens, int n_heads, int qk_nope, int qk_rope, int kv_lora,
                       int v_dim, int sel_width, bool fp8, float* out, void* stream);

}  // namespace basalt::models::ernie55
