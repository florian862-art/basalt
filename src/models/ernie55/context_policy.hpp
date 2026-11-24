// src/models/ernie55/context_policy.hpp - KV cache precision & context sizing for ERNIE-5.5-Flash (CPU-only).
#pragma once

#include <cstdint>
#include <string>

namespace basalt::models::ernie55::ctxp {

enum class CacheFormat { Fp8, Fp16 };

inline bool cache_format(const std::string& s, CacheFormat& out) {
    if (s.empty() || s == "fp8") { out = CacheFormat::Fp8; return true; }
    if (s == "fp16") { out = CacheFormat::Fp16; return true; }
    return false;
}

/// Bytes per context position across all HCA + MLA-DSA layers (excluding small fixed 128-slot sliding windows).
inline uint64_t bytes_per_position(int n_hca, int n_dsa, int hca_ratio, int hca_head_dim, int mla_kv_lora,
                                   int mla_rope, int idx_dim, CacheFormat fmt) {
    // HCA compressed KV: 1 entry per `hca_ratio` (128) positions; row = 512 values
    const uint64_t hca_row = fmt == CacheFormat::Fp8 ? (uint64_t) (hca_head_dim + (hca_head_dim / 64) * 4)
                                                     : (uint64_t) hca_head_dim * 2;
    const uint64_t hca_per_pos = (uint64_t) n_hca * ((hca_row + (uint64_t) hca_ratio - 1) / (uint64_t) hca_ratio);
    // MLA-DSA latent KV (512 + 64) + indexer K (128):
    // FP8 mode: 512 E4M3 + 4x fp32 scales (16 B) + 64 BF16 rope (128 B) = 656 B + indexer MXFP4 (68 B) = 724 B
    // FP16 mode: (512 + 64) * 2 = 1152 B + indexer FP16 (256 B) = 1408 B
    const uint64_t dsa_kv = fmt == CacheFormat::Fp8 ? (uint64_t) (mla_kv_lora + (mla_kv_lora / 128) * 4 + mla_rope * 2)
                                                    : (uint64_t) (mla_kv_lora + mla_rope) * 2;
    const uint64_t idx_k = fmt == CacheFormat::Fp8 ? (uint64_t) (idx_dim / 2 + idx_dim / 32)
                                                   : (uint64_t) idx_dim * 2;
    const uint64_t dsa_per_pos = (uint64_t) n_dsa * (dsa_kv + idx_k);
    return hca_per_pos + dsa_per_pos;
}

struct Plan {
    int64_t context = 0;
    uint64_t kv_bytes = 0;
    std::string error;
};

inline Plan plan_context(int64_t asked, int64_t max_pos, uint64_t free_vram, int n_hca, int n_dsa, int hca_ratio,
                         int hca_head_dim, int mla_kv_lora, int mla_rope, int idx_dim, CacheFormat fmt) {
    Plan p;
    if (asked < 1024 || asked > max_pos) {
        p.error = "--max-context " + std::to_string(asked) + " out of range [1024, " + std::to_string(max_pos) + "]";
        return p;
    }
    const uint64_t per_pos = bytes_per_position(n_hca, n_dsa, hca_ratio, hca_head_dim, mla_kv_lora, mla_rope, idx_dim, fmt);
    p.context = asked;
    p.kv_bytes = (uint64_t) asked * per_pos;
    if (free_vram > 0 && p.kv_bytes > free_vram / 2) {
        p.error = "--max-context " + std::to_string(asked) + " needs " + std::to_string(p.kv_bytes >> 20) +
                  " MiB KV cache, more than half of " + std::to_string(free_vram >> 20) + " MiB free VRAM";
        return p;
    }
    return p;
}

}  // namespace basalt::models::ernie55::ctxp
