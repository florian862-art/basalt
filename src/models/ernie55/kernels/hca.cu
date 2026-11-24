// src/models/ernie55/kernels/hca.cu - ERNIE-5.5 Heavily Compressed Attention (HCA) CUDA kernels:
//   - Q interleaved RoPE (last 64 dims of head_dim=512)
//   - KV RMSNorm + interleaved RoPE + sliding-window ring store (FP8 or FP16)
//   - 128:1 streaming compressor with additive APE[pos % 128], softmax-weighted sum, RMSNorm, YaRN RoPE, cache store
//   - Multi-query attention over compressed blocks + sliding window with per-head attn_sink
//   - Inverse interleaved RoPE on VHA-mixed attention output
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace basalt::models::ernie55 {
namespace {

constexpr int kRopeDim = 64;
constexpr int kRopePairs = 32;
// FP8 HCA row: nope_dim E4M3 codes (448 B) + 7 FP32 scales (28 B, per 64 values) + 64 BF16 rope (128 B) = 604 B -> 608 B
constexpr int kHcaFp8Stride = 608;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step, float mscale, bool inverse) {
    float s, c;
    sincosf(pos * step, &s, &c);
    c *= mscale;
    s *= mscale;
    if (inverse) s = -s;
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

__device__ __forceinline__ void store_hca_row(uint8_t* cache, size_t slot, const float* val, int head_dim, bool fp8,
                                              float* sh_red) {
    if (!fp8) {
        __half* dst = reinterpret_cast<__half*>(cache) + slot * (size_t) head_dim;
        for (int c = threadIdx.x; c < head_dim; c += blockDim.x) dst[c] = __float2half_rn(val[c]);
        return;
    }
    // FP8: nope [0..head_dim-64) quantized per 64 values with power-of-two UE8M0-compatible FP32 scale, rope [head_dim-64..head_dim) BF16
    const int nope = head_dim - kRopeDim;
    const int ngrp = nope / 64;
    uint8_t* row = cache + slot * (size_t) kHcaFp8Stride;
    float* scales = reinterpret_cast<float*>(row + nope);
    uint16_t* rope_out = reinterpret_cast<uint16_t*>(row + nope + ngrp * 4);

    // One warp per 64-value group (requires blockDim.x >= 256; handles arbitrary head_dim via loop)
    for (int g = threadIdx.x / 32; g < ngrp; g += blockDim.x / 32) {
        const int lane = threadIdx.x & 31;
        const int c0 = g * 64 + lane;
        const int c1 = c0 + 32;
        const float v0 = __bfloat162float(__float2bfloat16_rn(val[c0]));
        const float v1 = __bfloat162float(__float2bfloat16_rn(val[c1]));
        const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
        const float sc = exp2f(ceilf(log2f(fmaxf(am / 448.0f, 1e-4f))));
        if (lane == 0) scales[g] = sc;
        const float inv = 1.0f / sc;
        row[c0] = __nv_cvt_float_to_fp8(v0 * inv, __NV_SATFINITE, __NV_E4M3);
        row[c1] = __nv_cvt_float_to_fp8(v1 * inv, __NV_SATFINITE, __NV_E4M3);
    }
    for (int i = threadIdx.x; i < kRopeDim; i += blockDim.x) {
        rope_out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(val[nope + i]));
    }
    (void) sh_red;
}

__device__ __forceinline__ float load_hca_elem(const uint8_t* cache, size_t slot, int c, int head_dim, bool fp8) {
    if (!fp8) {
        const __half* src = reinterpret_cast<const __half*>(cache) + slot * (size_t) head_dim;
        return __half2float(src[c]);
    }
    const int nope = head_dim - kRopeDim;
    const uint8_t* row = cache + slot * (size_t) kHcaFp8Stride;
    if (c < nope) {
        const float* scales = reinterpret_cast<const float*>(row + nope);
        const float sc = scales[c >> 6];
        const __nv_fp8_e4m3 q = *reinterpret_cast<const __nv_fp8_e4m3*>(row + c);
        return (float) q * sc;
    }
    const int ngrp = nope / 64;
    const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + nope + ngrp * 4);
    return __bfloat162float(__ushort_as_bfloat16(rope_in[c - nope]));
}

__global__ void hca_rope_kernel(float* __restrict__ x, const int32_t* __restrict__ pos, int n_heads, int head_dim,
                                RopeTable rt, bool inverse) {
    const int pr = threadIdx.x;   // 0..31
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (pr >= kRopePairs) return;
    float* base = x + ((size_t) t * n_heads + h) * head_dim + (head_dim - kRopeDim) + 2 * pr;
    float x0 = base[0], x1 = base[1];
    rotate_pair(x0, x1, (float) pos[t], rt.step[pr], rt.mscale, inverse);
    base[0] = x0;
    base[1] = x1;
}

__global__ void hca_kv_norm_rope_kernel(const float* __restrict__ kv_in, const float* __restrict__ norm_w,
                                        const int32_t* __restrict__ pos, int head_dim, float rms_eps, RopeTable rt,
                                        float* __restrict__ kv_rot) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const float* src = kv_in + (size_t) t * head_dim;
    float ss = 0.0f;
    for (int c = tid; c < head_dim; c += blockDim.x) {
        const float v = src[c];
        sh[c] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < blockDim.x / 32; ++w) tot += red[w];
    const float inv = rsqrtf(tot / (float) head_dim + rms_eps);
    __syncthreads();

    for (int c = tid; c < head_dim; c += blockDim.x) sh[c] = sh[c] * inv * norm_w[c];
    __syncthreads();

    if (tid < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * tid;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid], rt.mscale, false);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();

    float* dst = kv_rot + (size_t) t * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) dst[c] = sh[c];
}

__global__ void hca_win_commit_kernel(const float* __restrict__ kv_rot, const int32_t* __restrict__ pos, int start_t,
                                      int head_dim, int win_size, uint8_t* __restrict__ win_cache, bool fp8) {
    const int t = start_t + blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const float* src = kv_rot + (size_t) t * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) sh[c] = src[c];
    __syncthreads();
    const int slot = pos[t] % win_size;
    store_hca_row(win_cache, (size_t) slot, sh, head_dim, fp8, red);
}

__global__ void hca_kv_store_kernel(const float* __restrict__ kv_in, const float* __restrict__ norm_w,
                                    const int32_t* __restrict__ pos, int start_t, int head_dim, int win_size,
                                    float rms_eps, RopeTable rt, uint8_t* __restrict__ win_cache, bool fp8) {
    const int t = start_t + blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const float* src = kv_in + (size_t) t * head_dim;
    float ss = 0.0f;
    for (int c = tid; c < head_dim; c += blockDim.x) {
        const float v = src[c];
        sh[c] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < blockDim.x / 32; ++w) tot += red[w];
    const float inv = rsqrtf(tot / (float) head_dim + rms_eps);
    __syncthreads();

    for (int c = tid; c < head_dim; c += blockDim.x) sh[c] = sh[c] * inv * norm_w[c];
    __syncthreads();

    if (tid < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * tid;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid], rt.mscale, false);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();

    const int slot = pos[t] % win_size;
    store_hca_row(win_cache, (size_t) slot, sh, head_dim, fp8, red);
}

// Single-block sequential streaming compressor over n_tokens: guarantees exact ordering when a chunk crosses
// multiple 128-token block boundaries. One thread per channel (blockDim.x = head_dim, e.g. 512).
__global__ void hca_compress_kernel(const float* __restrict__ ckv, const float* __restrict__ cscore,
                                    const float* __restrict__ ape, const float* __restrict__ norm_w,
                                    const int32_t* __restrict__ pos, int n_tokens, int ratio, int head_dim,
                                    float rms_eps, RopeTable comp_rt, float* __restrict__ state_kv,
                                    float* __restrict__ state_score, uint8_t* __restrict__ comp_cache, bool fp8) {
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const int c = threadIdx.x;
    if (c >= head_dim) return;

    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        const int slot = p % ratio;
        state_kv[(size_t) slot * head_dim + c] = ckv[(size_t) t * head_dim + c];
        state_score[(size_t) slot * head_dim + c] =
            cscore[(size_t) t * head_dim + c] + ape[(size_t) slot * head_dim + c];
        __syncthreads();

        if ((p + 1) % ratio != 0) continue;

        // Softmax-weighted sum over the closed group [0 .. ratio)
        float m = -FLT_MAX;
        for (int j = 0; j < ratio; ++j) m = fmaxf(m, state_score[(size_t) j * head_dim + c]);
        float den = 0.0f, num = 0.0f;
        for (int j = 0; j < ratio; ++j) {
            const float e = expf(state_score[(size_t) j * head_dim + c] - m);
            den += e;
            num += state_kv[(size_t) j * head_dim + c] * e;
        }
        const float comp = num / den;
        float ss = warp_sum(comp * comp);
        if ((c & 31) == 0) red[c >> 5] = ss;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < head_dim / 32; ++w) tot += red[w];
        const float y = comp * rsqrtf(tot / (float) head_dim + rms_eps) * norm_w[c];
        sh[c] = y;
        __syncthreads();

        if (c < kRopePairs) {
            const int idx = head_dim - kRopeDim + 2 * c;
            float x0 = sh[idx], x1 = sh[idx + 1];
            rotate_pair(x0, x1, (float) (p + 1 - ratio), comp_rt.step[c], comp_rt.mscale, false);
            sh[idx] = x0;
            sh[idx + 1] = x1;
        }
        __syncthreads();

        const size_t blk_id = (size_t) (p / ratio);
        store_hca_row(comp_cache, blk_id, sh, head_dim, fp8, red);
        __syncthreads();
    }
}

// One block (256 threads) per (head, token): online softmax attention over compressed blocks [0..n_comp)
// + sliding window [win_start..pos] + per-head attn_sink.
__global__ void hca_attention_kernel(const float* __restrict__ q, const uint8_t* __restrict__ comp_cache,
                                     const uint8_t* __restrict__ win_cache, const float* __restrict__ kv_rot,
                                     const float* __restrict__ attn_sink, const int32_t* __restrict__ pos,
                                     int n_heads, int head_dim, int win_size, int ratio, bool fp8,
                                     float* __restrict__ out) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int p = pos[t];
    const int pos0 = pos[0];
    const int win_start = p - (win_size - 1) > 0 ? p - (win_size - 1) : 0;
    const int win_n = p - win_start + 1;
    const int n_comp = (p + 1) / ratio;
    const float scale = rsqrtf((float) head_dim);

    extern __shared__ float smem[];   // [head_dim (q) + head_dim (acc) + 16 (red)]
    float* sq = smem;
    float* acc = smem + head_dim;
    float* red = acc + head_dim;
    const float* qrow = q + ((size_t) t * n_heads + h) * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) {
        sq[c] = qrow[c];
        acc[c] = 0.0f;
    }
    __syncthreads();

    // Initialize online softmax with attention sink (no V contribution, only logit mass exp(sink))
    float m = attn_sink[h];
    float l = 1.0f;   // exp(sink - sink) = 1

    auto step_cache_row = [&](const uint8_t* cache, size_t slot) {
        float dot = 0.0f;
        for (int c = tid; c < head_dim; c += blockDim.x) dot += sq[c] * load_hca_elem(cache, slot, c, head_dim, fp8);
        dot = warp_sum(dot);
        if ((tid & 31) == 0) red[tid >> 5] = dot;
        __syncthreads();
        float s = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) s += red[w];
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float alpha = expf(m - m_new);
        const float beta = expf(s - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (int c = tid; c < head_dim; c += blockDim.x) {
            const float v = load_hca_elem(cache, slot, c, head_dim, fp8);
            acc[c] = acc[c] * alpha + beta * v;
        }
        __syncthreads();
    };

    auto step_f32_row = [&](const float* row) {
        float dot = 0.0f;
        for (int c = tid; c < head_dim; c += blockDim.x) dot += sq[c] * row[c];
        dot = warp_sum(dot);
        if ((tid & 31) == 0) red[tid >> 5] = dot;
        __syncthreads();
        float s = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) s += red[w];
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float alpha = expf(m - m_new);
        const float beta = expf(s - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (int c = tid; c < head_dim; c += blockDim.x) acc[c] = acc[c] * alpha + beta * row[c];
        __syncthreads();
    };

    for (int b = 0; b < n_comp; ++b) step_cache_row(comp_cache, (size_t) b);
    for (int j = 0; j < win_n; ++j) {
        const int jp = win_start + j;
        if (kv_rot && jp >= pos0) step_f32_row(kv_rot + (size_t) (jp - pos0) * head_dim);
        else step_cache_row(win_cache, (size_t) (jp % win_size));
    }

    const float inv_l = 1.0f / l;
    float* orow = out + ((size_t) t * n_heads + h) * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) orow[c] = acc[c] * inv_l;
}

}  // namespace

void hca_q_rope(float* q, const int32_t* pos, int n_tokens, int n_heads, int head_dim, const RopeTable& rt,
                void* stream) {
    if (n_tokens <= 0) return;
    hca_rope_kernel<<<dim3(n_heads, n_tokens), 32, 0, (cudaStream_t) stream>>>(q, pos, n_heads, head_dim, rt, false);
}

void hca_inv_rope(float* o, const int32_t* pos, int n_tokens, int n_heads, int head_dim, const RopeTable& rt,
                  void* stream) {
    if (n_tokens <= 0) return;
    hca_rope_kernel<<<dim3(n_heads, n_tokens), 32, 0, (cudaStream_t) stream>>>(o, pos, n_heads, head_dim, rt, true);
}

void hca_kv_norm_rope(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                      float rms_eps, const RopeTable& rt, float* kv_rot, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_kv_norm_rope_kernel<<<n_tokens, 256, smem, (cudaStream_t) stream>>>(kv_in, kv_norm_w, pos, head_dim, rms_eps,
                                                                            rt, kv_rot);
}

void hca_win_commit(const float* kv_rot, const int32_t* pos, int n_tokens, int head_dim, int win_size,
                    uint8_t* win_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    const int n_write = n_tokens < win_size ? n_tokens : win_size;
    const int start_t = n_tokens - n_write;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_win_commit_kernel<<<n_write, 256, smem, (cudaStream_t) stream>>>(kv_rot, pos, start_t, head_dim, win_size,
                                                                         win_cache, fp8);
}

void hca_kv_store_window(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                         int win_size, float rms_eps, const RopeTable& rt, uint8_t* win_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    const int n_write = n_tokens < win_size ? n_tokens : win_size;
    const int start_t = n_tokens - n_write;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_kv_store_kernel<<<n_write, 256, smem, (cudaStream_t) stream>>>(kv_in, kv_norm_w, pos, start_t, head_dim,
                                                                       win_size, rms_eps, rt, win_cache, fp8);
}

void hca_compress(const float* ckv, const float* cscore, const float* ape, const float* norm_w, const int32_t* pos,
                  int n_tokens, int ratio, int head_dim, float rms_eps, const RopeTable& comp_rt,
                  float* state_kv, float* state_score, uint8_t* comp_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_compress_kernel<<<1, head_dim, smem, (cudaStream_t) stream>>>(ckv, cscore, ape, norm_w, pos, n_tokens, ratio,
                                                                      head_dim, rms_eps, comp_rt, state_kv,
                                                                      state_score, comp_cache, fp8);
}

void hca_attention(const float* q, const uint8_t* comp_cache, const uint8_t* win_cache, const float* kv_rot,
                   const float* attn_sink, const int32_t* pos, int n_tokens, int n_heads, int head_dim, int win_size,
                   int ratio, bool fp8, float* out, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (2 * head_dim + 16) * sizeof(float);
    hca_attention_kernel<<<dim3(n_heads, n_tokens), 256, smem, (cudaStream_t) stream>>>(
        q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, head_dim, win_size, ratio, fp8, out);
}

}  // namespace basalt::models::ernie55
