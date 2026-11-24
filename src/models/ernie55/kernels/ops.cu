// src/models/ernie55/kernels/ops.cu - ERNIE-5.5-Flash CUDA kernels:
//   - mHC with ERNIE's exact RMS factor 1 / (sqrt(mean) + rms_eps) and row-major destination-first comb[d, s]
//   - mHC final contraction head (hc_head_fn, hc_head_scale, hc_head_base)
//   - FP32 LM-head multimax
//   - SiTU-GLU (beta * tanh(g / beta) * sigmoid(g) * linear_beta * tanh(u / linear_beta))
//   - Split-feature MoE router (sigmoid(l0) + sigmoid(l1) + bias + top-K + renorm + learnable per-expert scale)
//   - VHA PostMix, sigmoid output gate, LayerNorm, orthonormal Hadamard-128, weighted sum, RoPE tables
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace basalt::models::ernie55 {

// ---- RoPE tables ----------------------------------------------------------------------------------------------

namespace {

float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * std::log((float) n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * std::log(base));
}

}  // namespace

RopeTable make_rope_table(float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, int n_ctx_orig) {
    constexpr int n_dims = 64;
    const float lo = ext_factor != 0.0f ? std::max(0.0f, std::floor(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base))) : 0.0f;
    const float hi = ext_factor != 0.0f ? std::min((float) (n_dims - 1), std::ceil(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base))) : 0.0f;
    const float theta_scale = std::pow(freq_base, -2.0f / (float) n_dims);
    RopeTable t{};
    float extrap = 1.0f;
    for (int i = 0; i < n_dims / 2; ++i) {
        float step = freq_scale * extrap;
        if (ext_factor != 0.0f) {
            const float y = ((float) i - lo) / std::max(0.001f, hi - lo);
            const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
            step = step * (1.0f - ramp) + extrap * ramp;
        }
        t.step[i] = step;
        extrap *= theta_scale;
    }
    t.mscale = attn_factor;
    if (ext_factor != 0.0f) t.mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    return t;
}

RopeTable rope_table_plain(float freq_base) { return make_rope_table(freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0); }

RopeTable rope_table_yarn(float freq_base, float factor, float beta_fast, float beta_slow, int n_ctx_orig) {
    const float freq_scale = 1.0f / factor;
    const float attn_factor = 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
    return make_rope_table(freq_base, freq_scale, 1.0f, attn_factor, beta_fast, beta_slow, n_ctx_orig);
}

namespace {

constexpr int kThreads = 256;
constexpr int kSlots = kHcCoef + 1;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) r += red[i];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoidf_ieee(float v) { return 1.0f / (1.0f + expf(-v)); }

// ---- mHC kernels ----------------------------------------------------------------------------------------------

__global__ void mhc_partial_kernel(const float* __restrict__ x, const float* __restrict__ fn, int flat, int n_mix,
                                   float* __restrict__ ws) {
    __shared__ float red[kThreads / 32];
    const int split = blockIdx.x, mix = blockIdx.y, t = blockIdx.z;
    const int len = flat / kHcSplits, base = split * len;
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * flat + base);
    const float4* wv = reinterpret_cast<const float4*>(fn + (size_t) mix * flat + base);
    float dot = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < len / 4; i += kThreads) {
        const float4 a = xv[i], w = wv[i];
        dot = fmaf(a.x, w.x, dot); dot = fmaf(a.y, w.y, dot);
        dot = fmaf(a.z, w.z, dot); dot = fmaf(a.w, w.w, dot);
        if (mix == 0) {
            ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss);
            ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss);
        }
    }
    dot = block_sum(dot, red);
    if (mix == 0) ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        float* part = ws + ((size_t) t * kHcSplits + split) * (n_mix + 1);
        part[mix] = dot;
        if (mix == 0) part[n_mix] = ss;
    }
}

// One warp per token: computes ERNIE 5.5 mHC coefficients with RMS = 1 / (sqrt(mean) + rms_eps).
// Output layout per token (24 floats): [pre(0..3), post(4..7), comb(8..23)] where comb[d*4 + s] is row d, col s.
__global__ void mhc_coef_ernie_kernel(const float* __restrict__ ws, const float* __restrict__ scale,
                                      const float* __restrict__ base, int flat, float rms_eps, float hc_eps,
                                      int sinkhorn_iters, float* __restrict__ coef) {
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const float* w = ws + (size_t) t * kHcSplits * kSlots;
    float ss = 0.f;
    for (int s = 0; s < kHcSplits; ++s) ss += w[(size_t) s * kSlots + kHcCoef];
    // ERNIE 5.5 exact RMS factor: eps outside sqrt
    const float inv_rms = 1.0f / (sqrtf(ss / (float) flat) + rms_eps);

    __shared__ float mix[kHcCoef];
    if (lane < kHcCoef) {
        float acc = 0.f;
        for (int s = 0; s < kHcSplits; ++s) acc += w[(size_t) s * kSlots + lane];
        mix[lane] = acc * inv_rms;
    }
    __syncwarp();

    float* out = coef + (size_t) t * kHcCoef;
    if (lane < 4) {
        out[lane] = sigmoidf_ieee(mix[lane] * scale[0] + base[lane]) + hc_eps;
    } else if (lane < 8) {
        out[lane] = sigmoidf_ieee(mix[lane] * scale[1] + base[lane]) * 2.0f;
    }

    // Sinkhorn-Knopp on 4x4 matrix comb[d, s] = mix[8 + d*4 + s] * scale[2] + base[8 + d*4 + s]
    __shared__ float cm[4][4];
    if (lane < 16) {
        const int r = lane >> 2, c = lane & 3;
        cm[r][c] = mix[8 + lane] * scale[2] + base[8 + lane];
    }
    __syncwarp();

    if (lane == 0) {
        // Row softmax + hc_eps
        for (int r = 0; r < 4; ++r) {
            float m = fmaxf(fmaxf(cm[r][0], cm[r][1]), fmaxf(cm[r][2], cm[r][3]));
            float s = 0.f;
            for (int c = 0; c < 4; ++c) { cm[r][c] = expf(cm[r][c] - m); s += cm[r][c]; }
            const float inv = 1.0f / s;
            for (int c = 0; c < 4; ++c) cm[r][c] = cm[r][c] * inv + hc_eps;
        }
        // First column normalization
        for (int c = 0; c < 4; ++c) {
            const float cs = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + hc_eps;
            const float inv = 1.0f / cs;
            for (int r = 0; r < 4; ++r) cm[r][c] *= inv;
        }
        // Remaining sinkhorn_iters - 1 iterations (row norm then col norm)
        for (int it = 1; it < sinkhorn_iters; ++it) {
            for (int r = 0; r < 4; ++r) {
                const float rs = cm[r][0] + cm[r][1] + cm[r][2] + cm[r][3] + hc_eps;
                const float inv = 1.0f / rs;
                for (int c = 0; c < 4; ++c) cm[r][c] *= inv;
            }
            for (int c = 0; c < 4; ++c) {
                const float cs = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + hc_eps;
                const float inv = 1.0f / cs;
                for (int r = 0; r < 4; ++r) cm[r][c] *= inv;
            }
        }
    }
    __syncwarp();
    if (lane < 16) out[8 + lane] = cm[lane >> 2][lane & 3];
}

__global__ void mhc_contract_coef_kernel(const float* __restrict__ ws, const float* __restrict__ scale,
                                         const float* __restrict__ base, int flat, float rms_eps, float hc_eps,
                                         float* __restrict__ coef4) {
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const float* w = ws + (size_t) t * kHcSplits * 5;
    float ss = 0.f;
    for (int s = 0; s < kHcSplits; ++s) ss += w[(size_t) s * 5 + 4];
    const float inv_rms = 1.0f / (sqrtf(ss / (float) flat) + rms_eps);
    if (lane < 4) {
        float acc = 0.f;
        for (int s = 0; s < kHcSplits; ++s) acc += w[(size_t) s * 5 + lane];
        coef4[(size_t) t * 4 + lane] = sigmoidf_ieee((acc * inv_rms) * scale[0] + base[lane]) + hc_eps;
    }
}

__global__ void mhc_pre_kernel(const float* __restrict__ x, const float* __restrict__ coef, int coef_stride,
                               int n_embd, float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n_embd >> 2;
    if (i >= n4) return;
    const float* p = coef + (size_t) t * coef_stride;
    const float4* xt = reinterpret_cast<const float4*>(x + (size_t) t * kHcMult * n_embd);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) {
        const float w = p[s];
        const float4 v = xt[(size_t) s * n4 + i];
        acc.x = fmaf(w, v.x, acc.x); acc.y = fmaf(w, v.y, acc.y);
        acc.z = fmaf(w, v.z, acc.z); acc.w = fmaf(w, v.w, acc.w);
    }
    reinterpret_cast<float4*>(out + (size_t) t * n_embd)[i] = acc;
}

__global__ void mhc_post_kernel(const float* __restrict__ f, const float* __restrict__ x,
                                const float* __restrict__ coef, int coef_stride, int n_embd,
                                float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n_embd >> 2;
    if (i >= n4) return;
    const float* cf = coef + (size_t) t * coef_stride;
    const float4 fv = reinterpret_cast<const float4*>(f + (size_t) t * n_embd)[i];
    const float4* xt = reinterpret_cast<const float4*>(x + (size_t) t * kHcMult * n_embd);
    float4 xs[kHcMult];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) xs[s] = xt[(size_t) s * n4 + i];
    float4* ot = reinterpret_cast<float4*>(out + (size_t) t * kHcMult * n_embd);
#pragma unroll
    for (int d = 0; d < kHcMult; ++d) {
        const float post = cf[4 + d];
        float4 o = make_float4(fv.x * post, fv.y * post, fv.z * post, fv.w * post);
#pragma unroll
        for (int s = 0; s < kHcMult; ++s) {
            // vLLM mhc_post_torch: out[d] = post[d] * f + sum_s comb[s, d] * x[s]
            const float c = cf[8 + s * 4 + d];
            o.x = fmaf(xs[s].x, c, o.x); o.y = fmaf(xs[s].y, c, o.y);
            o.z = fmaf(xs[s].z, c, o.z); o.w = fmaf(xs[s].w, c, o.w);
        }
        ot[(size_t) d * n4 + i] = o;
    }
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, int n_embd, float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float v = x[(size_t) t * n_embd + i];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) out[((size_t) t * kHcMult + s) * n_embd + i] = v;
}

// ---- Elementwise kernels --------------------------------------------------------------------------------------

__device__ __forceinline__ float relu_nan(float x) { return x <= 0.0f ? 0.0f : x; }

__global__ void multimax_kernel(float* __restrict__ logits, const float* __restrict__ r,
                                const float* __restrict__ ts, int n_vocab) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vocab) return;
    float* p = logits + (size_t) t * n_vocab + i;
    const float x = *p;
    const float a = relu_nan(r[0] - x);
    const float b = relu_nan(x - r[1]);
    const float c = relu_nan(r[2] - x);
    const float d = relu_nan(x - r[3]);
    float y = __fadd_rn(x, __fmul_rn(ts[0], a));
    y = __fadd_rn(y, __fmul_rn(ts[1], b));
    y = __fadd_rn(y, __fmul_rn(ts[2], __fmul_rn(c, c)));
    y = __fadd_rn(y, __fmul_rn(ts[3], __fmul_rn(d, d)));
    *p = y;
}

__global__ void situ_glu_kernel(const float* __restrict__ gate, const float* __restrict__ up, int64_t n,
                                float beta, float linear_beta, float* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i], u = up[i];
    const float sg = __fmul_rn(beta * tanhf(g / beta), sigmoidf_ieee(g));
    const float su = linear_beta > 0.0f ? (linear_beta * tanhf(u / linear_beta)) : u;
    out[i] = __fmul_rn(sg, su);
}

// One block (256 threads) per token: split-feature sigmoid sum + bias-corrected top-K + renorm + per-expert scale.
__global__ void split_sigmoid_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                           const float* __restrict__ scale, int n_expert, int top_k, bool renorm,
                                           int32_t* __restrict__ ids, float* __restrict__ weights) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* scores = smem;               // [n_expert]
    float* sel = smem + n_expert;       // [n_expert]
    const float* row = logits + (size_t) t * (2 * n_expert);
    for (int e = tid; e < n_expert; e += blockDim.x) {
        const float s = sigmoidf_ieee(row[e]) + sigmoidf_ieee(row[n_expert + e]);
        scores[e] = s;
        sel[e] = s + bias[e];
    }
    __syncthreads();

    if (tid == 0) {
        int32_t* oi = ids + (size_t) t * top_k;
        float* ow = weights + (size_t) t * top_k;
        float total = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            float best = -FLT_MAX;
            int best_id = 0;
            for (int e = 0; e < n_expert; ++e) {
                if (sel[e] > best) {
                    best = sel[e];
                    best_id = e;
                }
            }
            oi[k] = best_id;
            const float w = scores[best_id];
            ow[k] = w;
            total += w;
            sel[best_id] = -FLT_MAX;
        }
        if (renorm) {
            const float denom = fmaxf(total, 1e-20f);
            for (int k = 0; k < top_k; ++k) ow[k] /= denom;
        }
        for (int k = 0; k < top_k; ++k) ow[k] *= scale[oi[k]];
    }
}

__global__ void vha_postmix_kernel(const float* __restrict__ in, const float* __restrict__ mix,
                                   int n_heads, int head_dim, float* __restrict__ out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (d >= head_dim) return;
    const float* base = in + (size_t) t * n_heads * head_dim + d;
    const float* mrow = mix + (size_t) h * n_heads;
    float acc = 0.0f;
    for (int g = 0; g < n_heads; ++g) acc = fmaf(mrow[g], base[(size_t) g * head_dim], acc);
    out[((size_t) t * n_heads + h) * head_dim + d] = acc;
}

__global__ void sigmoid_gate_mul_kernel(float* __restrict__ z, const float* __restrict__ gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] *= sigmoidf_ieee(gate[i]);
}

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                 const float* __restrict__ b, int dim, float eps, float* __restrict__ out) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float red[8];
    const float* xr = x + (size_t) t * dim;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) sum += xr[i];
    sum = warp_sum(sum);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    float mean = 0.0f;
    for (int w_i = 0; w_i < blockDim.x / 32; ++w_i) mean += red[w_i];
    mean /= (float) dim;
    __syncthreads();

    float var = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float d = xr[i] - mean;
        var += d * d;
    }
    var = warp_sum(var);
    if ((tid & 31) == 0) red[tid >> 5] = var;
    __syncthreads();
    float vsum = 0.0f;
    for (int w_i = 0; w_i < blockDim.x / 32; ++w_i) vsum += red[w_i];
    const float inv = rsqrtf(vsum / (float) dim + eps);
    float* yr = out + (size_t) t * dim;
    for (int i = tid; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Orthonormal 128-point Walsh-Hadamard transform: one block of 128 threads per row (7 butterfly stages).
__global__ void hadamard_128_kernel(float* __restrict__ x) {
    __shared__ float s[128];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    s[tid] = x[(size_t) row * 128 + tid];
    __syncthreads();
#pragma unroll
    for (int step = 1; step < 128; step <<= 1) {
        const int partner = tid ^ step;
        const float a = s[tid], b = s[partner];
        __syncthreads();
        s[tid] = (tid < partner) ? (a + b) : (b - a);
        __syncthreads();
    }
    // 1 / sqrt(128) = 0.08838834764831845
    x[(size_t) row * 128 + tid] = s[tid] * 0.08838834764831845f;
}

__global__ void weighted_sum_kernel(const float* __restrict__ exp_out, const float* __restrict__ weights,
                                    int top_k, int dim, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= dim) return;
    const float* wt = weights + (size_t) t * top_k;
    const float* eb = exp_out + (size_t) t * top_k * dim + i;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) acc = fmaf(wt[k], eb[(size_t) k * dim], acc);
    out[(size_t) t * dim + i] = acc;
}

__global__ void add_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

}  // namespace

size_t mhc_workspace_bytes(int n_tokens) {
    return (size_t) n_tokens * kHcSplits * kSlots * sizeof(float);
}

void mhc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
               int n_embd, float rms_eps, float hc_eps, int sinkhorn_iters, float* workspace, float* coef,
               void* stream) {
    const int flat = kHcMult * n_embd;
    if (flat % (kHcSplits * 4) || sinkhorn_iters < 1) throw std::invalid_argument("ernie55 mhc_mixes: geometry");
    cudaStream_t st = (cudaStream_t) stream;
    mhc_partial_kernel<<<dim3(kHcSplits, kHcCoef, n_tokens), kThreads, 0, st>>>(x, hc_fn, flat, kHcCoef, workspace);
    mhc_coef_ernie_kernel<<<n_tokens, 32, 0, st>>>(workspace, hc_scale, hc_base, flat, rms_eps, hc_eps, sinkhorn_iters,
                                                   coef);
}

void mhc_pre(const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd, float* out, void* stream) {
    const int n4 = n_embd / 4;
    mhc_pre_kernel<<<dim3((n4 + 127) / 128, n_tokens), 128, 0, (cudaStream_t) stream>>>(x, coef, coef_stride, n_embd,
                                                                                        out);
}

void mhc_post(const float* f, const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd,
              float* out, void* stream) {
    const int n4 = n_embd / 4;
    mhc_post_kernel<<<dim3((n4 + 127) / 128, n_tokens), 128, 0, (cudaStream_t) stream>>>(f, x, coef, coef_stride,
                                                                                         n_embd, out);
}

void mhc_contract(const float* x, const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                  int n_tokens, int n_embd, float rms_eps, float hc_eps, float* workspace, float* coef4, float* out,
                  void* stream) {
    const int flat = kHcMult * n_embd;
    cudaStream_t st = (cudaStream_t) stream;
    mhc_partial_kernel<<<dim3(kHcSplits, 4, n_tokens), kThreads, 0, st>>>(x, hc_head_fn, flat, 4, workspace);
    mhc_contract_coef_kernel<<<n_tokens, 32, 0, st>>>(workspace, hc_head_scale, hc_head_base, flat, rms_eps, hc_eps,
                                                      coef4);
    mhc_pre(x, coef4, 4, n_tokens, n_embd, out, stream);
}

void broadcast_streams(const float* x, int n_tokens, int n_embd, float* out, void* stream) {
    broadcast_streams_kernel<<<dim3((n_embd + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(x, n_embd, out);
}

void multimax_f32(float* logits, const float* ranges, const float* ts, int n_tokens, int n_vocab, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    multimax_kernel<<<dim3((n_vocab + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(logits, ranges, ts,
                                                                                              n_vocab);
}

void situ_glu_f32(const float* gate, const float* up, int n_tokens, int n_ff, float beta, float linear_beta,
                  float* out, void* stream) {
    const int64_t n = (int64_t) n_tokens * n_ff;
    if (n <= 0) return;
    situ_glu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, n, beta, linear_beta,
                                                                                     out);
}

void split_sigmoid_route(const float* logits, const float* bias, const float* scale, int n_tokens, int n_expert,
                         int top_k, bool renorm, int32_t* ids, float* weights, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (2 * n_expert) * sizeof(float);
    split_sigmoid_route_kernel<<<n_tokens, 256, smem, (cudaStream_t) stream>>>(logits, bias, scale, n_expert, top_k,
                                                                               renorm, ids, weights);
}

void vha_postmix_f32(const float* in, const float* mix, int n_tokens, int n_heads, int head_dim, float* out,
                     void* stream) {
    if (n_tokens <= 0) return;
    vha_postmix_kernel<<<dim3((head_dim + 127) / 128, n_heads, n_tokens), 128, 0, (cudaStream_t) stream>>>(
        in, mix, n_heads, head_dim, out);
}

void sigmoid_gate_mul_f32(float* z, const float* gate, int64_t n, void* stream) {
    if (n <= 0) return;
    sigmoid_gate_mul_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(z, gate, n);
}

void layernorm_f32(const float* x, const float* weight, const float* bias, int n_tokens, int dim, float eps,
                   float* out, void* stream) {
    if (n_tokens <= 0) return;
    const int threads = dim <= 128 ? 128 : 256;
    layernorm_kernel<<<n_tokens, threads, 0, (cudaStream_t) stream>>>(x, weight, bias, dim, eps, out);
}

void hadamard_128_f32(float* x, int n_rows, void* stream) {
    if (n_rows <= 0) return;
    hadamard_128_kernel<<<n_rows, 128, 0, (cudaStream_t) stream>>>(x);
}

void weighted_sum_f32(const float* exp_out, const float* weights, int n_tokens, int top_k, int dim, float* out,
                      void* stream) {
    if (n_tokens <= 0) return;
    weighted_sum_kernel<<<dim3((dim + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(exp_out, weights, top_k,
                                                                                              dim, out);
}

void add_f32(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    add_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(dst, src, n);
}

}  // namespace basalt::models::ernie55
