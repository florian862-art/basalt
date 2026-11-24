// src/models/ernie55/kernels/mla_dsa.cu - ERNIE-5.5 MLA-DSA + Lightning Indexer CUDA kernels:
//   - Q/K interleaved RoPE + latent KV cache store (FP8 fp8_ds_mla or FP16)
//   - Lightning Indexer prepare: LayerNorm(k), partial NeoX RoPE (leading 64 dims), orthonormal Hadamard-128,
//     MXFP4 (FP8 mode) or FP16 indexer K cache store
//   - Lightning Indexer scoring (ReLU(iq . ik) * weights * (1 / sqrt(idx_dim * idx_heads))), top-K over [0..pos-win),
//     merged with trailing sliding window [max(0, pos-win+1)..pos]
//   - Absorbed sparse MLA attention over selected positions with per-head softmax_offset sink and Q8_0 kv_b_proj
#include "basalt/models/ernie55/kernels.hpp"
#include "basalt/kernels/mxfp4_scale.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace basalt::models::ernie55 {
namespace {

// fp8_ds_mla row: 512 E4M3 codes (512 B) + 4 fp32 scales (16 B) + 64 BF16 rope (128 B) = 656 B
constexpr int kDsaFp8RowBytes = 656;
// MXFP4 indexer K row (dim = 128): 64 packed nibble bytes + 4 UE8M0 scale bytes = 68 B
constexpr int kIdxFp4RowBytes = 68;

__device__ __constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ int warp_sum_i32(int v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step) {
    float s, c;
    sincosf(pos * step, &s, &c);
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

__device__ __forceinline__ uint8_t quant_e2m1(float x) {
    const uint8_t sign = x < 0.0f ? 8u : 0u;
    const float a = fabsf(x);
    uint8_t mag = 0;
    if (a >= 5.0f) mag = 7;
    else if (a >= 3.5f) mag = 6;
    else if (a >= 2.5f) mag = 5;
    else if (a >= 1.75f) mag = 4;
    else if (a >= 1.25f) mag = 3;
    else if (a >= 0.75f) mag = 2;
    else if (a >= 0.25f) mag = 1;
    return sign | mag;
}

__global__ void dsa_q_rope_kernel(float* __restrict__ q, const int32_t* __restrict__ pos, int n_heads,
                                  int qk_nope, int qk_rope, RopeTable rt) {
    const int pr = threadIdx.x;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (pr >= qk_rope / 2) return;
    const int qd = qk_nope + qk_rope;
    float* base = q + ((size_t) t * n_heads + h) * qd + qk_nope + 2 * pr;
    float x0 = base[0], x1 = base[1];
    rotate_pair(x0, x1, (float) pos[t], rt.step[pr]);
    base[0] = x0;
    base[1] = x1;
}

__global__ void dsa_kv_store_kernel(const float* __restrict__ kv_a, const float* __restrict__ kv_norm_w,
                                    const int32_t* __restrict__ pos, int kv_lora, int qk_rope, float rms_eps,
                                    RopeTable rt, uint8_t* __restrict__ kv_cache, bool fp8) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [kv_lora + qk_rope + 16]
    float* rp = sh + kv_lora;
    float* red = rp + qk_rope;
    const int kv_dim = kv_lora + qk_rope;
    const float* src = kv_a + (size_t) t * kv_dim;

    float ss = 0.0f;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        const float v = src[c];
        sh[c] = v;
        ss += v * v;
    }
    for (int c = tid; c < qk_rope; c += blockDim.x) rp[c] = src[kv_lora + c];
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < blockDim.x / 32; ++w) tot += red[w];
    const float inv = rsqrtf(tot / (float) kv_lora + rms_eps);
    __syncthreads();

    for (int c = tid; c < kv_lora; c += blockDim.x) sh[c] = sh[c] * inv * kv_norm_w[c];
    if (tid < qk_rope / 2) {
        float x0 = rp[2 * tid], x1 = rp[2 * tid + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid]);
        rp[2 * tid] = x0;
        rp[2 * tid + 1] = x1;
    }
    __syncthreads();

    const size_t cell = (size_t) pos[t];
    if (!fp8) {
        __half* dst = reinterpret_cast<__half*>(kv_cache) + cell * (size_t) kv_dim;
        for (int c = tid; c < kv_lora; c += blockDim.x) dst[c] = __float2half_rn(sh[c]);
        for (int c = tid; c < qk_rope; c += blockDim.x) dst[kv_lora + c] = __float2half_rn(rp[c]);
    } else {
        // fp8_ds_mla: 512 E4M3 codes (4 groups of 128), 4 fp32 scales, 64 BF16 rope
        uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
        float* scales = reinterpret_cast<float*>(row + kv_lora);
        uint16_t* rope_out = reinterpret_cast<uint16_t*>(row + kv_lora + (kv_lora / 128) * 4);
        const int ngrp = kv_lora / 128;
        for (int g = tid / 32; g < ngrp; g += blockDim.x / 32) {
            const int lane = tid & 31;
            float vals[4];
            float am = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                vals[k] = __bfloat162float(__float2bfloat16_rn(sh[g * 128 + k * 32 + lane]));
                am = fmaxf(am, fabsf(vals[k]));
            }
            am = warp_max(am);
            const float sc = exp2f(ceilf(log2f(fmaxf(am / 448.0f, 1e-4f))));
            if (lane == 0) scales[g] = sc;
            const float inv_s = 1.0f / sc;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                row[g * 128 + k * 32 + lane] = __nv_cvt_float_to_fp8(vals[k] * inv_s, __NV_SATFINITE, __NV_E4M3);
        }
        for (int c = tid; c < qk_rope; c += blockDim.x)
            rope_out[c] = __bfloat16_as_ushort(__float2bfloat16_rn(rp[c]));
    }
}

// Partial NeoX RoPE on leading `rope_dim` (64) elements of each row [n_rows, idx_dim]:
// pairs (i, i + rope_dim / 2) for i in 0..31.
__global__ void neox_partial_rope_kernel(float* __restrict__ x, const int32_t* __restrict__ pos,
                                         int rows_per_tok, int idx_dim, int rope_dim, RopeTable rt) {
    const int half = rope_dim / 2;
    const int i = threadIdx.x;
    const int r = blockIdx.x;
    const int t = blockIdx.y;
    if (i >= half) return;
    float* row = x + ((size_t) t * rows_per_tok + r) * idx_dim;
    float x0 = row[i], x1 = row[i + half];
    rotate_pair(x0, x1, (float) pos[t], rt.step[i]);
    row[i] = x0;
    row[i + half] = x1;
}

__global__ void store_idx_k_kernel(const float* __restrict__ ik, const int32_t* __restrict__ pos, int idx_dim,
                                   uint8_t* __restrict__ idx_k_cache, bool fp8) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;   // 0..127
    const float* src = ik + (size_t) t * idx_dim;
    const size_t cell = (size_t) pos[t];
    if (!fp8) {
        __half* dst = reinterpret_cast<__half*>(idx_k_cache) + cell * (size_t) idx_dim;
        if (tid < idx_dim) dst[tid] = __float2half_rn(src[tid]);
        return;
    }
    // MXFP4 (32-element blocks, 4 blocks for idx_dim=128): 64 packed bytes (even/odd nibbles) + 4 UE8M0 bytes
    __shared__ float vals[128];
    __shared__ uint8_t nibs[128];
    vals[tid] = src[tid];
    const int blk = tid >> 5;
    const int lane = tid & 31;
    const float am = warp_max(fabsf(vals[tid]));
    // E8M0 exponent: scale = 2^(e - 127) >= am / 6.0
    int exp_i = 127;
    if (am > 0.0f) {
        exp_i = (int) ceilf(log2f(am / 6.0f)) + 127;
        exp_i = max(0, min(254, exp_i));
    } else {
        exp_i = 0;
    }
    const float sc = ldexpf(1.0f, exp_i - 127);
    const float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
    nibs[tid] = quant_e2m1(vals[tid] * inv);
    __syncthreads();

    uint8_t* row = idx_k_cache + cell * (size_t) kIdxFp4RowBytes;
    if (tid < 64) row[tid] = nibs[2 * tid] | (uint8_t) (nibs[2 * tid + 1] << 4);
    if (lane == 0 && blk < 4) row[64 + blk] = (uint8_t) exp_i;
}

__device__ __forceinline__ float load_idx_k_elem(const uint8_t* cache, size_t cell, int d, int idx_dim, bool fp8) {
    if (!fp8) {
        const __half* src = reinterpret_cast<const __half*>(cache) + cell * (size_t) idx_dim;
        return __half2float(src[d]);
    }
    const uint8_t* row = cache + cell * (size_t) kIdxFp4RowBytes;
    const uint8_t byte = row[d >> 1];
    const uint8_t nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    const uint8_t e = row[64 + (d >> 5)];
    const float sc = ldexpf(1.0f, (int) e - 127);
    return kE2M1[nib] * sc;
}

// Multi-block single-pass scorer across all SMs: computes score_pos(j) once into scores_buf[t * stride + j]
// whenever older_n > top_k (zero work on fast path older_n <= top_k).
__global__ void dsa_indexer_score_kernel(const float* __restrict__ iq, const float* __restrict__ iweights,
                                         const uint8_t* __restrict__ idx_k_cache, const int32_t* __restrict__ pos,
                                         int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                                         float* __restrict__ scores_buf, int scores_stride) {
    const int t = blockIdx.y;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    if (older_n <= top_k) return;

    __shared__ float sw[64];
    const float wscale = rsqrtf((float) (idx_dim * idx_heads));
    if (threadIdx.x < idx_heads) sw[threadIdx.x] = iweights[(size_t) t * idx_heads + threadIdx.x] * wscale;
    __syncthreads();

    const float* iq_t = iq + (size_t) t * idx_heads * idx_dim;
    float* out_s = scores_buf + (size_t) t * scores_stride;
    const int step = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < older_n; j += step) {
        float k_reg[128];
        for (int d = 0; d < idx_dim; ++d) k_reg[d] = load_idx_k_elem(idx_k_cache, (size_t) j, d, idx_dim, fp8);
        float total = 0.0f;
        for (int h = 0; h < idx_heads; ++h) {
            const float* qh = iq_t + (size_t) h * idx_dim;
            float dot = 0.0f;
            for (int d = 0; d < idx_dim; ++d) dot += qh[d] * k_reg[d];
            if (dot > 0.0f) total += dot * sw[h];
        }
        out_s[j] = total;
    }
}

// Fast top-K selection reading precomputed FP32 scores from scores_buf (L2-resident, ~20 us).
__global__ void dsa_indexer_select_cached_kernel(const float* __restrict__ scores_buf, int scores_stride,
                                                 const int32_t* __restrict__ pos, int top_k, int win_size,
                                                 int32_t* __restrict__ out_idx) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int width = top_k + win_size;
    int32_t* dst = out_idx + (size_t) t * width;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    const int swa_n = seq_len < win_size ? seq_len : win_size;
    const int keep_k = older_n < top_k ? older_n : top_k;

    if (older_n <= top_k) {
        for (int i = tid; i < width; i += blockDim.x) {
            if (i < seq_len) dst[i] = i;
            else dst[i] = -1;
        }
        return;
    }

    const float* sc_row = scores_buf + (size_t) t * scores_stride;
    float local_lo = FLT_MAX, local_hi = -FLT_MAX;
    for (int j = tid; j < older_n; j += blockDim.x) {
        const float sc = sc_row[j];
        local_lo = fminf(local_lo, sc);
        local_hi = fmaxf(local_hi, sc);
    }
    __shared__ float s_lo[8], s_hi[8];
    __shared__ int s_cnt;
    local_lo = -warp_max(-local_lo);
    local_hi = warp_max(local_hi);
    if ((tid & 31) == 0) { s_lo[tid >> 5] = local_lo; s_hi[tid >> 5] = local_hi; }
    __syncthreads();
    float lo = FLT_MAX, hi = -FLT_MAX;
    for (int w = 0; w < 8; ++w) { lo = fminf(lo, s_lo[w]); hi = fmaxf(hi, s_hi[w]); }

    for (int iter = 0; iter < 20; ++iter) {
        const float mid = 0.5f * (lo + hi);
        int cnt = 0;
        for (int j = tid; j < older_n; j += blockDim.x) {
            if (sc_row[j] > mid) ++cnt;
        }
        cnt = warp_sum_i32(cnt);
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        if ((tid & 31) == 0) atomicAdd(&s_cnt, cnt);
        __syncthreads();
        if (s_cnt > keep_k) lo = mid;
        else hi = mid;
        __syncthreads();
    }
    const float thresh = hi;

    if (tid == 0) s_cnt = 0;
    __syncthreads();
    for (int j = tid; j < older_n; j += blockDim.x) {
        if (sc_row[j] > thresh) {
            const int slot = atomicAdd(&s_cnt, 1);
            if (slot < keep_k) dst[slot] = j;
        }
    }
    __syncthreads();
    if (s_cnt < keep_k) {
        for (int j = tid; j < older_n; j += blockDim.x) {
            const float sc = sc_row[j];
            if (sc <= thresh && sc >= lo) {
                const int slot = atomicAdd(&s_cnt, 1);
                if (slot < keep_k) dst[slot] = j;
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < width - keep_k; i += blockDim.x) {
        if (i < swa_n) dst[keep_k + i] = seq_len - swa_n + i;
        else dst[keep_k + i] = -1;
    }
}

// One block (256 threads) per token fallback when no scores_buf is supplied.
__global__ void dsa_indexer_topk_kernel(const float* __restrict__ iq, const float* __restrict__ iweights,
                                        const uint8_t* __restrict__ idx_k_cache, const int32_t* __restrict__ pos,
                                        int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                                        int32_t* __restrict__ out_idx) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int width = top_k + win_size;
    int32_t* dst = out_idx + (size_t) t * width;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    const int swa_n = seq_len < win_size ? seq_len : win_size;
    const int keep_k = older_n < top_k ? older_n : top_k;

    // Fast path when all older tokens fit inside top_k (no ranking needed)
    if (older_n <= top_k) {
        for (int i = tid; i < width; i += blockDim.x) {
            if (i < seq_len) dst[i] = i;
            else dst[i] = -1;
        }
        return;
    }

    // Load per-head weights scaled by 1 / sqrt(idx_dim * idx_heads)
    __shared__ float sw[64];
    const float wscale = rsqrtf((float) (idx_dim * idx_heads));
    if (tid < idx_heads) sw[tid] = iweights[(size_t) t * idx_heads + tid] * wscale;
    __syncthreads();

    // Maintain thread-local streaming top-K? For top_k = 2048 and 256 threads, each thread tracks top-8 candidates
    // (256 * 8 = 2048!) or for general older_n we score blocks of positions and merge via shared bitonic/selection.
    // Clean, exact streaming insertion per thread: each thread owns stride positions j = tid, tid+256, ...
    // and keeps its best `per_t = (top_k + 255) / 256` (8) pairs, then block reduces?
    // Wait: stride-partitioned local top-8 can miss if all true top-K fall in one stride!
    // Exact two-pass score-and-select: we have up to 1M older_n, so use an exact shared min-heap / bucket threshold
    // or thread-cooperative scoring into a compact histogram threshold!
    // Even simpler & 100% exact: 32-bin radix / binary search on float score threshold to find exact k-th score!
    const float* iq_t = iq + (size_t) t * idx_heads * idx_dim;

    auto score_pos = [&](int j) -> float {
        float k_reg[128];
        for (int d = 0; d < idx_dim; ++d) k_reg[d] = load_idx_k_elem(idx_k_cache, (size_t) j, d, idx_dim, fp8);
        float total = 0.0f;
        for (int h = 0; h < idx_heads; ++h) {
            const float* qh = iq_t + (size_t) h * idx_dim;
            float dot = 0.0f;
            for (int d = 0; d < idx_dim; ++d) dot += qh[d] * k_reg[d];
            if (dot > 0.0f) total += dot * sw[h];
        }
        return total;
    };

    // Pass 1: find [lo, hi] range of scores across [0 .. older_n)
    float local_lo = FLT_MAX, local_hi = -FLT_MAX;
    for (int j = tid; j < older_n; j += blockDim.x) {
        const float sc = score_pos(j);
        local_lo = fminf(local_lo, sc);
        local_hi = fmaxf(local_hi, sc);
    }
    __shared__ float s_lo[8], s_hi[8];
    __shared__ int s_cnt;
    local_lo = -warp_max(-local_lo);
    local_hi = warp_max(local_hi);
    if ((tid & 31) == 0) { s_lo[tid >> 5] = local_lo; s_hi[tid >> 5] = local_hi; }
    __syncthreads();
    float lo = FLT_MAX, hi = -FLT_MAX;
    for (int w = 0; w < 8; ++w) { lo = fminf(lo, s_lo[w]); hi = fmaxf(hi, s_hi[w]); }

    // Pass 2: 20-step binary search on score threshold `mid` so count(score > mid) <= keep_k <= count(score >= mid)
    for (int iter = 0; iter < 20; ++iter) {
        const float mid = 0.5f * (lo + hi);
        int cnt = 0;
        for (int j = tid; j < older_n; j += blockDim.x) {
            if (score_pos(j) > mid) ++cnt;
        }
        cnt = warp_sum_i32(cnt);
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        if ((tid & 31) == 0) atomicAdd(&s_cnt, cnt);
        __syncthreads();
        if (s_cnt > keep_k) lo = mid;
        else hi = mid;
        __syncthreads();
    }
    const float thresh = hi;

    // Pass 3: emit all strictly > thresh, then fill ties == thresh up to keep_k
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    for (int j = tid; j < older_n; j += blockDim.x) {
        if (score_pos(j) > thresh) {
            const int slot = atomicAdd(&s_cnt, 1);
            if (slot < keep_k) dst[slot] = j;
        }
    }
    __syncthreads();
    if (s_cnt < keep_k) {
        for (int j = tid; j < older_n; j += blockDim.x) {
            const float sc = score_pos(j);
            if (sc <= thresh && sc >= lo) {
                const int slot = atomicAdd(&s_cnt, 1);
                if (slot < keep_k) dst[slot] = j;
            }
        }
    }
    __syncthreads();

    // Append trailing sliding window [seq_len - swa_n .. seq_len - 1], then pad with -1
    for (int i = tid; i < width - keep_k; i += blockDim.x) {
        if (i < swa_n) dst[keep_k + i] = seq_len - swa_n + i;
        else dst[keep_k + i] = -1;
    }
}

// Dequantize one Q8_0 row (block_size = 32, 34 bytes/block: fp16 scale + 32 int8) dot with float vector `x`.
__device__ __forceinline__ float q8_0_row_dot(const uint8_t* row, const float* x, int cols) {
    const int nb = cols >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) s = fmaf((float) qs[k], x[b * 32 + k], s);
        acc = fmaf(d, s, acc);
    }
    return acc;
}

__device__ __forceinline__ void load_dsa_kv_row(const uint8_t* kv_cache, size_t cell, int kv_lora, int qk_rope,
                                                bool fp8, float* c_kv, float* k_pe) {
    const int tid = threadIdx.x;
    if (!fp8) {
        const __half* row = reinterpret_cast<const __half*>(kv_cache) + cell * (size_t) (kv_lora + qk_rope);
        for (int c = tid; c < kv_lora; c += blockDim.x) c_kv[c] = __half2float(row[c]);
        for (int c = tid; c < qk_rope; c += blockDim.x) k_pe[c] = __half2float(row[kv_lora + c]);
    } else {
        const uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
        const float* scales = reinterpret_cast<const float*>(row + kv_lora);
        const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + kv_lora + (kv_lora / 128) * 4);
        for (int c = tid; c < kv_lora; c += blockDim.x) {
            const __nv_fp8_e4m3 q = *reinterpret_cast<const __nv_fp8_e4m3*>(row + c);
            c_kv[c] = (float) q * scales[c >> 7];
        }
        for (int c = tid; c < qk_rope; c += blockDim.x)
            k_pe[c] = __bfloat162float(__ushort_as_bfloat16(rope_in[c]));
    }
}

// Absorbed sparse MLA attention: one block (256 threads) per (head, token).
//   1. Absorb W_UK[h]^T into q_nope[h] -> q_lat[512]
//   2. Online softmax over selected positions sel_idx[t, 0..sel_width) with attn_sink[h] -> o_lat[512]
//   3. Project o[h] = W_UV[h] @ o_lat[512] -> out[t, h, v_dim]
__global__ void dsa_mla_attention_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kv_cache,
                                         const uint8_t* __restrict__ kv_b_q8, const float* __restrict__ attn_sink,
                                         const int32_t* __restrict__ sel_idx, int n_heads, int qk_nope, int qk_rope,
                                         int kv_lora, int v_dim, int sel_width, bool fp8, float* __restrict__ out) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int qd = qk_nope + qk_rope;
    const size_t q8_row_bytes = (size_t) (kv_lora / 32) * 34;
    const uint8_t* head_kv_b = kv_b_q8 + (size_t) h * (size_t) (qk_nope + v_dim) * q8_row_bytes;
    const uint8_t* w_uk = head_kv_b;
    const uint8_t* w_uv = head_kv_b + (size_t) qk_nope * q8_row_bytes;

    extern __shared__ float smem[];
    float* q_nope = smem;                       // [qk_nope]
    float* q_pe = q_nope + qk_nope;             // [qk_rope]
    float* q_lat = q_pe + qk_rope;              // [kv_lora]
    float* o_lat = q_lat + kv_lora;             // [kv_lora]
    float* c_kv = o_lat + kv_lora;              // [kv_lora]
    float* k_pe = c_kv + kv_lora;               // [qk_rope]
    float* red = k_pe + qk_rope;                // [16]

    const float* qrow = q + ((size_t) t * n_heads + h) * qd;
    for (int i = tid; i < qk_nope; i += blockDim.x) q_nope[i] = qrow[i];
    for (int i = tid; i < qk_rope; i += blockDim.x) q_pe[i] = qrow[qk_nope + i];
    for (int c = tid; c < kv_lora; c += blockDim.x) { q_lat[c] = 0.0f; o_lat[c] = 0.0f; }
    __syncthreads();

    // 1. Absorb W_UK^T: q_lat[c] = sum_{r=0}^{qk_nope-1} q_nope[r] * W_UK[r, c]
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        const int blk = c >> 5;
        const int off = c & 31;
        float acc = 0.0f;
        for (int r = 0; r < qk_nope; ++r) {
            const uint8_t* b = w_uk + (size_t) r * q8_row_bytes + (size_t) blk * 34;
            const float d = __half2float(*reinterpret_cast<const __half*>(b));
            const int8_t qv = reinterpret_cast<const int8_t*>(b + 2)[off];
            acc = fmaf(q_nope[r], d * (float) qv, acc);
        }
        q_lat[c] = acc;
    }
    __syncthreads();

    // 2. Online softmax over selected positions with softmax_offset sink
    const float scale = rsqrtf((float) qd);
    float m = attn_sink[h];
    float l = 1.0f;
    const int32_t* sel = sel_idx + (size_t) t * sel_width;

    for (int s_i = 0; s_i < sel_width; ++s_i) {
        const int32_t cell = sel[s_i];
        if (cell < 0) break;
        load_dsa_kv_row(kv_cache, (size_t) cell, kv_lora, qk_rope, fp8, c_kv, k_pe);
        __syncthreads();

        float dot = 0.0f;
        for (int c = tid; c < kv_lora; c += blockDim.x) dot = fmaf(q_lat[c], c_kv[c], dot);
        for (int c = tid; c < qk_rope; c += blockDim.x) dot = fmaf(q_pe[c], k_pe[c], dot);
        dot = warp_sum(dot);
        if ((tid & 31) == 0) red[tid >> 5] = dot;
        __syncthreads();
        float sc = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) sc += red[w];
        sc *= scale;

        const float m_new = fmaxf(m, sc);
        const float alpha = expf(m - m_new);
        const float beta = expf(sc - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (int c = tid; c < kv_lora; c += blockDim.x) o_lat[c] = o_lat[c] * alpha + beta * c_kv[c];
        __syncthreads();
    }

    const float inv_l = 1.0f / l;
    for (int c = tid; c < kv_lora; c += blockDim.x) o_lat[c] *= inv_l;
    __syncthreads();

    // 3. Project o[h, r] = W_UV[r, :] . o_lat[:]
    float* orow = out + ((size_t) t * n_heads + h) * v_dim;
    for (int r = tid; r < v_dim; r += blockDim.x) {
        orow[r] = q8_0_row_dot(w_uv + (size_t) r * q8_row_bytes, o_lat, kv_lora);
    }
}

}  // namespace

void dsa_qkv_rope_and_store(float* q, const float* kv_a, const float* kv_norm_w, const int32_t* pos, int n_tokens,
                            int n_heads, int qk_nope, int qk_rope, int kv_lora, float rms_eps, const RopeTable& rt,
                            uint8_t* kv_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    dsa_q_rope_kernel<<<dim3(n_heads, n_tokens), 32, 0, st>>>(q, pos, n_heads, qk_nope, qk_rope, rt);
    const size_t smem = (size_t) (kv_lora + qk_rope + 16) * sizeof(float);
    dsa_kv_store_kernel<<<n_tokens, 256, smem, st>>>(kv_a, kv_norm_w, pos, kv_lora, qk_rope, rms_eps, rt, kv_cache,
                                                     fp8);
}

void dsa_indexer_prepare(float* iq, float* ik, const float* k_norm_w, const float* k_norm_b, const int32_t* pos,
                         int n_tokens, int idx_heads, int idx_dim, int rope_dim, float eps, const RopeTable& rt,
                         uint8_t* idx_k_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    // LayerNorm on ik in-place
    layernorm_f32(ik, k_norm_w, k_norm_b, n_tokens, idx_dim, eps, ik, stream);
    // Partial NeoX RoPE on leading rope_dim (64) dims of iq and ik
    neox_partial_rope_kernel<<<dim3(idx_heads, n_tokens), 32, 0, st>>>(iq, pos, idx_heads, idx_dim, rope_dim, rt);
    neox_partial_rope_kernel<<<dim3(1, n_tokens), 32, 0, st>>>(ik, pos, 1, idx_dim, rope_dim, rt);
    // Orthonormal 128-dim Walsh-Hadamard transform on iq and ik
    hadamard_128_f32(iq, n_tokens * idx_heads, stream);
    hadamard_128_f32(ik, n_tokens, stream);
    // Store ik into indexer K cache
    store_idx_k_kernel<<<n_tokens, 128, 0, st>>>(ik, pos, idx_dim, idx_k_cache, fp8);
}

void dsa_indexer_topk_merge(const float* iq, const float* iweights, const uint8_t* idx_k_cache, const int32_t* pos,
                            int n_tokens, int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                            int32_t* out_idx, void* stream, float* scores_buf, int scores_stride) {
    if (n_tokens <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    if (scores_buf && scores_stride > 0) {
        dsa_indexer_score_kernel<<<dim3(64, n_tokens), 256, 0, st>>>(
            iq, iweights, idx_k_cache, pos, idx_heads, idx_dim, top_k, win_size, fp8, scores_buf, scores_stride);
        dsa_indexer_select_cached_kernel<<<n_tokens, 256, 0, st>>>(
            scores_buf, scores_stride, pos, top_k, win_size, out_idx);
        return;
    }
    dsa_indexer_topk_kernel<<<n_tokens, 256, 0, st>>>(iq, iweights, idx_k_cache, pos, idx_heads,
                                                      idx_dim, top_k, win_size, fp8, out_idx);
}

void dsa_mla_attention(const float* q, const uint8_t* kv_cache, const void* kv_b_q8, const float* attn_sink,
                       const int32_t* sel_idx, int n_tokens, int n_heads, int qk_nope, int qk_rope, int kv_lora,
                       int v_dim, int sel_width, bool fp8, float* out, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (qk_nope + 2 * qk_rope + 3 * kv_lora + 16) * sizeof(float);
    dsa_mla_attention_kernel<<<dim3(n_heads, n_tokens), 256, smem, (cudaStream_t) stream>>>(
        q, kv_cache, (const uint8_t*) kv_b_q8, attn_sink, sel_idx, n_heads, qk_nope, qk_rope, kv_lora, v_dim,
        sel_width, fp8, out);
}

}  // namespace basalt::models::ernie55
