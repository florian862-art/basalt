// src/models/ernie55/session.cpp - ERNIE-5.5-Flash inference session:
//   - 43-layer hybrid HCA (37x) + MLA-DSA (6x) + mHC (4 streams) + Latent MXFP4 MoE (42x 512 top-10)
//   - Multi-GPU peer expert tier + pinned RAM CPU pool via core::ExpertTier
//   - Decode windows (1..8 rows) and prefill chunks (up to kPrefillChunk rows) using Q8_0 MMVQ/MMQ + MXFP4 grouped/CPU
#include "models/ernie55/session.hpp"

#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/models/ernie55/kernels.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace basalt::models::ernie55 {
namespace {

constexpr int kMaxDecodeWindow = 8;
constexpr int kPrefillChunk = 128;
constexpr int kHcaFp8RowBytes = 608;
constexpr int kDsaFp8RowBytes = 656;
constexpr int kIdxFp4RowBytes = 68;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeMXFP4 = 39;

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("ernie55: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

template <typename T>
bool dalloc(T*& p, size_t count, const char* what, std::string& err) {
    p = nullptr;
    if (count == 0) return true;
    return ck(cudaMalloc(&p, count * sizeof(T)), what, err);
}

template <typename T>
void dfree(T*& p) {
    if (p) { cudaFree(p); p = nullptr; }
}

struct HcaLayerState {
    uint8_t* win_cache = nullptr;
    uint8_t* comp_cache = nullptr;
    float* state_kv = nullptr;
    float* state_score = nullptr;
    size_t win_bytes = 0, comp_bytes = 0, state_bytes = 0;
};

struct DsaLayerState {
    uint8_t* kv_cache = nullptr;
    uint8_t* idx_k_cache = nullptr;
    size_t kv_bytes = 0, idx_bytes = 0;
};

class ErnieSession final : public model::ModelSession {
public:
    explicit ErnieSession(ErnieModel& m) : m_(m), g_(m.geo()) {}

    ~ErnieSession() override { release(); }

    bool init(std::string& err) {
        cudaSetDevice(m_.device());
        if (!ck(cudaStreamCreate(&stream_), "stream", err)) return false;
        if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
            err = "ernie55: cublasCreate failed";
            return false;
        }
        cublasSetStream(cublas_, stream_);

        // Build RoPE tables
        hca_rope_ = rope_table_yarn(g_.comp_rope_theta, g_.yarn_factor, g_.yarn_beta_fast, g_.yarn_beta_slow, g_.yarn_orig_ctx);
        dsa_rope_ = rope_table_plain(g_.rope_theta);

        // Allocate KV & compressor state caches for this conversation
        const int64_t ctx = m_.context();
        const bool fp8 = m_.cache_fp8();
        const size_t hca_row = fp8 ? (size_t) kHcaFp8RowBytes : (size_t) g_.hca_head_dim * 2;
        const size_t dsa_row = fp8 ? (size_t) kDsaFp8RowBytes : (size_t) (g_.mla_kv_lora + g_.mla_qk_rope) * 2;
        const size_t idx_row = fp8 ? (size_t) kIdxFp4RowBytes : (size_t) g_.idx_dim * 2;

        hca_.resize((size_t) g_.n_layers);
        dsa_.resize((size_t) g_.n_layers);
        kv_bytes_ = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!g_.is_dsa(l)) {
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                const size_t n_comp = (size_t) (ctx / ratio + 2);
                auto& s = hca_[(size_t) l];
                s.win_bytes = (size_t) g_.window * hca_row;
                s.comp_bytes = n_comp * hca_row;
                s.state_bytes = (size_t) ratio * (size_t) g_.hca_head_dim * sizeof(float);
                if (!ck(cudaMalloc(&s.win_cache, s.win_bytes), "hca win_cache", err) ||
                    !ck(cudaMalloc(&s.comp_cache, s.comp_bytes), "hca comp_cache", err) ||
                    !ck(cudaMalloc(&s.state_kv, s.state_bytes), "hca state_kv", err) ||
                    !ck(cudaMalloc(&s.state_score, s.state_bytes), "hca state_score", err))
                    return false;
                cudaMemsetAsync(s.win_cache, 0, s.win_bytes, stream_);
                cudaMemsetAsync(s.comp_cache, 0, s.comp_bytes, stream_);
                cudaMemsetAsync(s.state_kv, 0, s.state_bytes, stream_);
                cudaMemsetAsync(s.state_score, 0, s.state_bytes, stream_);
                kv_bytes_ += (int64_t) (s.win_bytes + s.comp_bytes + 2 * s.state_bytes);
            } else {
                auto& s = dsa_[(size_t) l];
                s.kv_bytes = (size_t) ctx * dsa_row;
                s.idx_bytes = (size_t) ctx * idx_row;
                if (!ck(cudaMalloc(&s.kv_cache, s.kv_bytes), "dsa kv_cache", err) ||
                    !ck(cudaMalloc(&s.idx_k_cache, s.idx_bytes), "dsa idx_k_cache", err))
                    return false;
                cudaMemsetAsync(s.kv_cache, 0, s.kv_bytes, stream_);
                cudaMemsetAsync(s.idx_k_cache, 0, s.idx_bytes, stream_);
                kv_bytes_ += (int64_t) (s.kv_bytes + s.idx_bytes);
            }
        }

        // Allocate forward scratch buffers sized for kPrefillChunk rows
        const int maxT = kPrefillChunk;
        const int H = g_.n_embd;
        const int q_dim_max = std::max(g_.hca_heads * g_.hca_head_dim, g_.mla_heads * (g_.mla_qk_nope + g_.mla_qk_rope));
        const int o_dim_max = std::max(g_.hca_heads * g_.hca_head_dim, g_.mla_heads * g_.mla_v_dim);
        const int gate_dim_max = std::max(g_.hca_o_groups * g_.hca_o_lora, g_.mla_heads * g_.mla_v_dim);
        const int ff_max = std::max(g_.n_ff, g_.n_ff_exp);
        const int q8_cols_max = std::max({H, gate_dim_max, ff_max, g_.hca_q_lora, g_.n_latent});
        const int max_entries = maxT * g_.n_used;

        if (!dalloc(d_pos_, maxT, "d_pos", err) ||
            !dalloc(d_res_a_, (size_t) maxT * kHcMult * H, "d_res_a", err) ||
            !dalloc(d_res_b_, (size_t) maxT * kHcMult * H, "d_res_b", err) ||
            !dalloc(d_x_in_, (size_t) maxT * H, "d_x_in", err) ||
            !dalloc(d_x_norm_, (size_t) maxT * H, "d_x_norm", err) ||
            !dalloc(d_sub_out_, (size_t) maxT * H, "d_sub_out", err) ||
            !dalloc(d_mhc_ws_, mhc_workspace_bytes(maxT) / sizeof(float), "d_mhc_ws", err) ||
            !dalloc(d_mhc_coef_, (size_t) maxT * kHcCoef, "d_mhc_coef", err) ||
            !dalloc(d_qr_, (size_t) maxT * std::max(g_.hca_q_lora, g_.mla_q_lora), "d_qr", err) ||
            !dalloc(d_q_, (size_t) maxT * q_dim_max, "d_q", err) ||
            !dalloc(d_kv_, (size_t) maxT * std::max(g_.hca_head_dim, g_.mla_kv_lora + g_.mla_qk_rope), "d_kv", err) ||
            !dalloc(d_ckv_, (size_t) maxT * g_.hca_head_dim, "d_ckv", err) ||
            !dalloc(d_cscore_, (size_t) maxT * g_.hca_head_dim, "d_cscore", err) ||
            !dalloc(d_attn_o_, (size_t) maxT * o_dim_max, "d_attn_o", err) ||
            !dalloc(d_attn_o2_, (size_t) maxT * o_dim_max, "d_attn_o2", err) ||
            !dalloc(d_gate_, (size_t) maxT * gate_dim_max, "d_gate", err) ||
            !dalloc(d_z_, (size_t) maxT * gate_dim_max, "d_z", err) ||
            !dalloc(d_iq_, (size_t) maxT * g_.idx_heads * g_.idx_dim, "d_iq", err) ||
            !dalloc(d_ik_, (size_t) maxT * g_.idx_dim, "d_ik", err) ||
            !dalloc(d_iw_, (size_t) maxT * g_.idx_heads, "d_iw", err) ||
            !dalloc(d_sel_idx_, (size_t) maxT * (g_.idx_top_k + g_.window), "d_sel_idx", err) ||
            !dalloc(d_idx_scores_, (size_t) kMaxDecodeWindow * (size_t) ctx, "d_idx_scores", err) ||
            !dalloc(d_ff_g_, (size_t) maxT * ff_max, "d_ff_g", err) ||
            !dalloc(d_ff_u_, (size_t) maxT * ff_max, "d_ff_u", err) ||
            !dalloc(d_ff_h_, (size_t) maxT * ff_max, "d_ff_h", err) ||
            !dalloc(d_rlogits_, (size_t) maxT * 2 * g_.n_expert, "d_rlogits", err) ||
            !dalloc(d_exp_ids_, (size_t) max_entries, "d_exp_ids", err) ||
            !dalloc(d_exp_w_, (size_t) max_entries, "d_exp_w", err) ||
            !dalloc(d_x_lat_, (size_t) maxT * g_.n_latent, "d_x_lat", err) ||
            !dalloc(d_lat_sum_, (size_t) maxT * g_.n_latent, "d_lat_sum", err) ||
            !dalloc(d_lat_norm_, (size_t) maxT * g_.n_latent, "d_lat_norm", err) ||
            !dalloc(d_routed_out_, (size_t) maxT * H, "d_routed_out", err) ||
            !dalloc(d_exp_out_, (size_t) max_entries * g_.n_latent, "d_exp_out", err) ||
            !dalloc(d_logits_, (size_t) kMaxDecodeWindow * g_.n_vocab, "d_logits", err) ||
            !dalloc(d_picks_, kMaxDecodeWindow, "d_picks", err) ||
            !dalloc(d_lp_scratch_, kernels::row_logprobs_scratch_floats(kMaxDecodeWindow, g_.n_vocab), "d_lp_scratch", err) ||
            !dalloc(d_lp_out_, (size_t) kMaxDecodeWindow * kernels::kLogprobStride, "d_lp_out", err) ||
            !dalloc(d_mmq_bounds_, 2, "d_mmq_bounds", err) ||
            !dalloc(d_mmq_iota_, maxT, "d_mmq_iota", err))
            return false;

        // Quantized activation scratch for both MMVQ (up to 8 cols) and MMQ (up to maxT cols)
        const size_t q8_bytes = std::max(kernels::native_q8_1_bytes(q8_cols_max, kMaxDecodeWindow),
                                         prefill::mmq::q8_bytes(maxT, q8_cols_max));
        if (!ck(cudaMalloc(&d_xq_, q8_bytes), "d_xq", err)) return false;

        // Grouped GPU expert dispatch buffers (sized for max_entries)
        const size_t exp_scratch = kernels::native_expert_scratch_bytes(max_entries, g_.n_ff_exp);
        if (!ck(cudaMalloc(&d_exp_scratch_, exp_scratch), "d_exp_scratch", err) ||
            !dalloc(d_grp_ptr_, max_entries, "d_grp_ptr", err) ||
            !dalloc(d_grp_start_, max_entries + 1, "d_grp_start", err) ||
            !dalloc(d_n_groups_, 1, "d_n_groups", err) ||
            !dalloc(d_ent_dst_, max_entries, "d_ent_dst", err) ||
            !dalloc(d_ent_tok_, max_entries, "d_ent_tok", err))
            return false;

        // Host staging buffers
        h_emb_.resize((size_t) maxT * H);
        h_pos_.resize(maxT);
        h_ids_.resize(max_entries);
        h_grp_ptr_.resize(max_entries);
        h_grp_start_.resize(max_entries + 1);
        h_ent_dst_.resize(max_entries);
        h_ent_tok_.resize(max_entries);
        prefill::mmq::iota(d_mmq_iota_, maxT, stream_);

        return ck(cudaStreamSynchronize(stream_), "init sync", err);
    }

    model::Result prepare() override {
        auto& st = m_.shared_tier();
        if (st.tier) return {};
        cudaSetDevice(m_.device());
        std::string err;
        const size_t x_bytes = (size_t) kPrefillChunk * (size_t) g_.n_latent * sizeof(float);
        const size_t row_bytes = (size_t) kPrefillChunk * (size_t) g_.n_used * (size_t) g_.n_latent * sizeof(float);
        if (!ck(cudaHostAlloc(&st.h_x, x_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "h_x", err) ||
            !ck(cudaHostAlloc(&st.h_rows, row_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "h_rows", err))
            return {model::Status::Error, err};

        auto tier = std::make_shared<core::ExpertTier>(m_.tier_spec());
        core::ExpertTier::Reserve res;
        res.copies = 512ull << 20;
        res.homes = 256ull << 20;
        res.beside_homes = 256ull << 20;
        res.modules = 128ull << 20;
        core::LoadPlan* lp = m_.planning();
        core::LoadPlan local_lp;
        if (!tier->open(res, st.h_rows, lp ? *lp : local_lp, err)) return {model::Status::Error, err};
        st.tier = std::move(tier);
        return {};
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens <= 0 || r.n_tokens > kMaxDecodeWindow)
            return {model::Status::Error, "ernie55: decode_window n_tokens out of range"};
        if (r.pos < 0 || r.pos + r.n_tokens > m_.context())
            return {model::Status::Refused, "ernie55: context limit reached"};
        std::string err;
        if (!forward_rows(r.tokens, r.pos, r.n_tokens, /*want_logits=*/true, err))
            return {model::Status::Error, err};
        last_window_n_ = r.n_tokens;

        // Sample next token(s)
        kernels::SamplerParams sp = r.sampling ? *r.sampling : kernels::SamplerParams{};
        if (!r.sampling) sp.greedy = true;
        if (logit_bias_) sp.logit_bias = logit_bias_;

        int32_t* d_hist = nullptr;
        if (r.history && r.history_len > 0) {
            const size_t hbytes = (size_t) r.n_tokens * (size_t) r.history_len * sizeof(int32_t);
            if (!ck(cudaMalloc(&d_hist, hbytes), "history", err)) return {model::Status::Error, err};
            cudaMemcpyAsync(d_hist, r.history, hbytes, cudaMemcpyHostToDevice, stream_);
        }
        kernels::sample_tokens(d_logits_, r.n_tokens, g_.n_vocab, d_hist, r.history_len, sp, d_picks_, stream_);
        std::vector<int32_t> picks((size_t) r.n_tokens);
        cudaMemcpyAsync(picks.data(), d_picks_, (size_t) r.n_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        if (r.logits) {
            cudaMemcpyAsync(r.logits, d_logits_ + (size_t) (r.n_tokens - 1) * g_.n_vocab,
                            (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        }
        if (!ck(cudaStreamSynchronize(stream_), "decode sync", err)) {
            if (d_hist) cudaFree(d_hist);
            return {model::Status::Error, err};
        }
        if (d_hist) cudaFree(d_hist);
        if (r.next_token) *r.next_token = picks.back();
        if (r.row_tokens) std::memcpy(r.row_tokens, picks.data(), (size_t) r.n_tokens * sizeof(int32_t));
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.n_tokens <= 0) return {};
        if (r.pos < 0 || r.pos + r.n_tokens > m_.context())
            return {model::Status::Refused, "ernie55: prefill exceeds context"};
        if (auto& st = m_.shared_tier(); st.tier) st.tier->forget_use();
        int done = 0;
        std::string err;
        while (done < r.n_tokens) {
            if (r.stop && r.stop()) break;
            const int take = std::min(kPrefillChunk, r.n_tokens - done);
            if (!forward_rows(r.tokens + done, r.pos + done, take, /*want_logits=*/false, err))
                return {model::Status::Error, err};
            done += take;
            if (r.progress) r.progress(done);
        }
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        logit_bias_ = bias;
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        if (row < 0 || row >= last_window_n_) return {model::Status::Error, "ernie55: resample_row out of range"};
        kernels::SamplerParams sp;
        sp.greedy = true;
        sp.logit_bias = bias;
        kernels::sample_tokens(d_logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab, nullptr, 0, sp, d_picks_, stream_);
        std::string err;
        cudaMemcpyAsync(&token, d_picks_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        if (!ck(cudaStreamSynchronize(stream_), "resample sync", err)) return {model::Status::Error, err};
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n <= 0 || row0 + n > last_window_n_)
            return {model::Status::Error, "ernie55: row_logprobs out of range"};
        std::string err;
        cudaMemcpyAsync(d_picks_, picks, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        kernels::row_logprobs(d_logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_picks_, top, d_lp_out_,
                              d_lp_scratch_, stream_);
        cudaMemcpyAsync(out, d_lp_out_, (size_t) n * kernels::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_);
        if (!ck(cudaStreamSynchronize(stream_), "logprobs sync", err)) return {model::Status::Error, err};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_window_n_) return {model::Status::Error, "ernie55: copy_logits out of range"};
        std::string err;
        cudaMemcpyAsync(out, d_logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        if (!ck(cudaStreamSynchronize(stream_), "copy_logits sync", err)) return {model::Status::Error, err};
        return {};
    }

    size_t running_bytes() const override {
        size_t tot = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!g_.is_dsa(l)) {
                const auto& s = hca_[(size_t) l];
                tot += s.win_bytes + 2 * s.state_bytes;
            }
        }
        return tot;
    }

    model::Result running_state(int64_t, bool, core::RunningState& rs) override {
        rs.stream = stream_;
        rs.regions.clear();
        rs.head.clear();
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!g_.is_dsa(l)) {
                const auto& s = hca_[(size_t) l];
                rs.regions.push_back({s.win_cache, s.win_bytes});
                rs.regions.push_back({s.state_kv, s.state_bytes});
                rs.regions.push_back({s.state_score, s.state_bytes});
            }
        }
        return {};
    }

    ExpertCounts expert_counts() const override {
        const auto& t = m_.shared_tier().tier;
        if (!t) return {};
        return {t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
    }

    std::vector<CardExperts> experts_by_card() const override {
        const auto& t = m_.shared_tier().tier;
        if (!t) return {};
        std::vector<CardExperts> out;
        for (const auto& c : t->slots_by_card()) out.push_back({c.device, c.slots, c.bytes});
        return out;
    }

    int64_t kv_bytes() const override { return kv_bytes_; }

private:
    const DevTensor* req_t(const std::string& name, std::string& err) const {
        const DevTensor* t = m_.tensor(name);
        if (!t) err = "ernie55: missing tensor " + name;
        return t;
    }
    const DevTensor* req_t(int l, const char* suf, std::string& err) const {
        return req_t("blk." + std::to_string(l) + "." + suf, err);
    }

    // Q8_0 matmul y[T, n_out] = x[T, n_in] @ W[n_out, n_in]^T (MMVQ for T <= 8, MMQ for T > 8).
    void matmul_q8(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int T) {
        if (T <= kMaxDecodeWindow) {
            kernels::native_quantize_q8_1(x, d_xq_, n_in, T, stream_);
            kernels::native_q8_0_mmvq(w->ptr, d_xq_, y, n_in, n_out, T, stream_);
            return;
        }
        prefill::mmq::quantize(x, nullptr, d_xq_, kTypeQ8_0, n_in, n_in, T, stream_);
        const int32_t bounds[2] = {0, T};
        cudaMemcpyAsync(d_mmq_bounds_, bounds, sizeof(bounds), cudaMemcpyHostToDevice, stream_);
        prefill::mmq::Product p;
        p.w = w->ptr;
        p.type = kTypeQ8_0;
        p.w_rows = n_out;
        p.w_cols = n_in;
        p.expert_bytes = w->bytes;
        p.n = 1;
        p.xq = d_xq_;
        p.bounds = d_mmq_bounds_;
        p.ids = d_mmq_iota_;
        p.total_rows = T;
        p.max_rows = T;
        p.dst = y;
        p.ld_dst = n_out;
        mmq_ctx_.run(p, stream_);
    }

    // BF16 matmul y[T, n_out] = x[T, n_in] @ W[n_out, n_in]^T in slices of up to 8 rows.
    void matmul_bf16(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int T) {
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int take = std::min(8, T - t0);
            kernels::bf16_gemv_fp32_mmvf_cols(x + (size_t) t0 * n_in, (const uint16_t*) w->ptr,
                                              y + (size_t) t0 * n_out, n_in, n_out, take, stream_);
        }
    }

    // F32 matmul y[T, n_out] = x[T, n_in] @ W[n_out, n_in]^T via cuBLAS SGEMM.
    void matmul_f32(const float* w, const float* x, float* y, int n_in, int n_out, int T) {
        const float alpha = 1.0f, beta = 0.0f;
        // Column-major view: Y^T [n_out, T] = W [n_in, n_out]^T @ X [n_in, T]
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &alpha, w, n_in, x, n_in, &beta, y, n_out);
    }

    // Runs routed MXFP4 experts for MoE layer `moe_l` (0..n_layers-n_dense-1) over `T` tokens,
    // sliced into windows of at most kMaxDecodeWindow (8) to match ExpertTier/PeerExperts/MMVQ contracts.
    bool run_moe_experts(int moe_l, int T, std::string& err) {
        auto& st = m_.shared_tier();
        core::ExpertTier* tier = st.tier.get();
        if (!tier) { err = "ernie55: expert tier not prepared"; return false; }

        const int K = g_.n_used;
        const int Ldim = g_.n_latent;
        const auto L = [&]() {
            auto lay = kernels::native_expert_layout(kTypeMXFP4, kTypeMXFP4, Ldim, g_.n_ff_exp, 0.0f);
            lay.gate_softcap = g_.gate_softcap;
            lay.up_softcap = g_.up_softcap;
            return lay;
        }();

        for (int t0 = 0; t0 < T; t0 += kMaxDecodeWindow) {
            const int tw = std::min(kMaxDecodeWindow, T - t0);
            const int entries = tw * K;
            const float* x_lat_w = d_x_lat_ + (size_t) t0 * (size_t) Ldim;
            const int32_t* ids_w = d_exp_ids_ + (size_t) t0 * (size_t) K;
            float* exp_out_w = d_exp_out_ + (size_t) t0 * (size_t) K * (size_t) Ldim;

            // Copy window routing ids and latent activations to host for tier lookup + CPU/peer dispatch
            cudaMemcpyAsync(h_ids_.data(), ids_w, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(st.h_x, x_lat_w, (size_t) tw * (size_t) Ldim * sizeof(float), cudaMemcpyDeviceToHost,
                            stream_);
            if (!ck(cudaStreamSynchronize(stream_), "moe sync", err)) return false;

            tier->begin_token(tw);
            // Deduplicate expert ids across [0..tw) into unique experts + Use records (u.nt <= tw <= 8)
            std::vector<int32_t> uniq;
            std::vector<core::ExpertTier::Use> uses;
            uniq.reserve((size_t) entries);
            uses.reserve((size_t) entries);
            for (int idx = 0; idx < entries; ++idx) {
                const int32_t eid = h_ids_[(size_t) idx];
                const int tok = idx / K;
                int found = -1;
                for (int u = 0; u < (int) uniq.size(); ++u) {
                    if (uniq[(size_t) u] == eid) { found = u; break; }
                }
                if (found < 0) {
                    found = (int) uniq.size();
                    uniq.push_back(eid);
                    uses.push_back(core::ExpertTier::Use{});
                }
                auto& u = uses[(size_t) found];
                u.tok[u.nt] = tok;
                u.row[u.nt] = idx;
                ++u.nt;
            }

            // Pre-quantize this window's latent activations for GPU 0 grouped experts (1 <= tw <= 8)
            kernels::native_quantize_q8_1(x_lat_w, d_xq_, Ldim, tw, stream_);

            std::vector<int> cpu_rows;
            cpu_rows.reserve((size_t) entries);

            int offset = 0;
            core::PeerExperts* peer = tier->peer_for(moe_l);
            while (offset < (int) uniq.size()) {
                const int batch = std::min((int) uniq.size() - offset, core::ExpertTier::kMaxK);
                if (peer && !peer->arm(L, st.h_x, tw, K, 8, err)) return false;
                core::ExpertTier::Lookup lk;
                if (!tier->lookup(moe_l, uniq.data() + offset, batch, lk, err)) return false;
                tier->fetch(moe_l, lk);

                // Raise Peer GPU (e.g. GPU 1) plan first (handles both resident kind=2 and bus-fetched kind=3)
                if (peer) {
                    std::vector<int32_t> p_ids((size_t) entries, 0), p_kind((size_t) entries, 0);
                    for (int i = 0; i < batch; ++i) {
                        if (!lk.peer[i]) continue;
                        const int kind_v = lk.fetched[i] ? 3 : 2;
                        const auto& u = uses[(size_t) (offset + i)];
                        for (int j = 0; j < u.nt; ++j) {
                            p_ids[(size_t) u.row[j]] = uniq[(size_t) (offset + i)];
                            p_kind[(size_t) u.row[j]] = kind_v;
                            cpu_rows.push_back(u.row[j]);   // peer writes into mapped st.h_rows
                        }
                    }
                    if (!peer->raise(moe_l, p_ids.data(), p_kind.data(), err)) return false;
                }

                // Start CPU pool on host-only misses
                bool any_cpu = false;
                for (int i = 0; i < batch; ++i) {
                    if (!lk.dev[i] && !lk.peer[i] && !lk.dropped[i] && !lk.reading[i]) {
                        any_cpu = true;
                        const auto& u = uses[(size_t) (offset + i)];
                        for (int j = 0; j < u.nt; ++j) cpu_rows.push_back(u.row[j]);
                    }
                }
                if (any_cpu) tier->start_cpu_window(moe_l, st.h_x, lk, uses.data() + offset, st.h_rows, false);

                // Launch GPU 0 grouped experts for main-card hits + bus-fetched experts
                int n_gpu_grp = 0, n_gpu_ent = 0;
                for (int i = 0; i < batch; ++i) {
                    if (!lk.dev[i]) continue;
                    h_grp_ptr_[(size_t) n_gpu_grp] = (unsigned long long) lk.dev[i];
                    h_grp_start_[(size_t) n_gpu_grp] = n_gpu_ent;
                    const auto& u = uses[(size_t) (offset + i)];
                    for (int j = 0; j < u.nt; ++j) {
                        h_ent_dst_[(size_t) n_gpu_ent] = u.row[j];
                        h_ent_tok_[(size_t) n_gpu_ent] = u.tok[j];
                        ++n_gpu_ent;
                    }
                    ++n_gpu_grp;
                }
                h_grp_start_[(size_t) n_gpu_grp] = n_gpu_ent;
                if (n_gpu_grp > 0) {
                    tier->await_fetch(stream_);
                    cudaMemcpyAsync(d_grp_ptr_, h_grp_ptr_.data(), (size_t) n_gpu_grp * sizeof(unsigned long long),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_grp_start_, h_grp_start_.data(), (size_t) (n_gpu_grp + 1) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_n_groups_, &n_gpu_grp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_ent_dst_, h_ent_dst_.data(), (size_t) n_gpu_ent * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_ent_tok_, h_ent_tok_.data(), (size_t) n_gpu_ent * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    kernels::native_expert_grouped(L, d_grp_ptr_, d_grp_start_, d_n_groups_, d_ent_dst_, d_ent_tok_,
                                                   n_gpu_grp, n_gpu_ent, d_xq_, d_exp_scratch_, exp_out_w, stream_);
                }
                if (peer && !peer->run_fetched(err)) return false;

                if (any_cpu) tier->finish_cpu();
                if (peer) {
                    if (!peer->finish(st.h_rows, err)) return false;
                    cudaSetDevice(m_.device());
                }

                // Handle any file-read misses if RAM didn't fit all experts
                bool any_read = false;
                for (int i = 0; i < batch; ++i) {
                    if (lk.reading[i]) {
                        any_read = true;
                        const auto& u = uses[(size_t) (offset + i)];
                        for (int j = 0; j < u.nt; ++j) cpu_rows.push_back(u.row[j]);
                    }
                }
                if (any_read) {
                    if (!tier->await_reads(err)) return false;
                    tier->start_cpu_window(moe_l, st.h_x, lk, uses.data() + offset, st.h_rows, true);
                    tier->finish_cpu();
                }

                tier->promote(moe_l, lk);

                // Upload this batch's CPU + peer outputs before the next batch touches st.h_rows
                for (int row : cpu_rows) {
                    cudaMemcpyAsync(exp_out_w + (size_t) row * Ldim, st.h_rows + (size_t) row * Ldim,
                                    (size_t) Ldim * sizeof(float), cudaMemcpyHostToDevice, stream_);
                }
                cpu_rows.clear();
                offset += batch;
            }
        }
        return true;
    }

    bool forward_rows(const int32_t* tokens, int64_t pos0, int T, bool want_logits, std::string& err) {
        cudaSetDevice(m_.device());
        const int H = g_.n_embd;
        const bool fp8 = m_.cache_fp8();

        for (int t = 0; t < T; ++t) {
            h_pos_[(size_t) t] = (int32_t) (pos0 + t);
            if (!m_.embedding_row(tokens[t], h_emb_.data() + (size_t) t * H)) {
                err = "ernie55: token id out of vocabulary range";
                return false;
            }
        }
        cudaMemcpyAsync(d_pos_, h_pos_.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_x_in_, h_emb_.data(), (size_t) T * (size_t) H * sizeof(float), cudaMemcpyHostToDevice, stream_);
        broadcast_streams(d_x_in_, T, H, d_res_a_, stream_);

        float* res = d_res_a_;
        float* res_next = d_res_b_;

        for (int l = 0; l < g_.n_layers; ++l) {
            // ---- Attention sublayer ----
            const DevTensor *hc_fn = req_t(l, "hc_attn_fn.weight", err),
                            *hc_sc = req_t(l, "hc_attn_scale.weight", err),
                            *hc_bs = req_t(l, "hc_attn_base.weight", err),
                            *anorm = req_t(l, "attn_norm.weight", err),
                            *vha_m = req_t(l, "attn_vha_mix.weight", err),
                            *agate = req_t(l, "attn_gate.weight", err),
                            *asink = req_t(l, "attn_sinks.weight", err);
            if (!hc_fn || !hc_sc || !hc_bs || !anorm || !vha_m || !agate || !asink) return false;

            mhc_mixes(res, (const float*) hc_fn->ptr, (const float*) hc_sc->ptr, (const float*) hc_bs->ptr, T, H,
                      g_.rms_eps, g_.hc_eps, g_.sinkhorn_iters, d_mhc_ws_, d_mhc_coef_, stream_);
            mhc_pre(res, d_mhc_coef_, kHcCoef, T, H, d_x_in_, stream_);
            kernels::rms_norm_rows(d_x_in_, (const float*) anorm->ptr, T, H, g_.rms_eps, d_x_norm_, stream_);

            if (!g_.is_dsa(l)) {
                const DevTensor *qa = req_t(l, "attn_q_a.weight", err), *qan = req_t(l, "attn_q_a_norm.weight", err),
                                *qb = req_t(l, "attn_q_b.weight", err), *wkv = req_t(l, "attn_kv.weight", err),
                                *kvn = req_t(l, "attn_kv_a_norm.weight", err), *oa = req_t(l, "attn_output_a.weight", err),
                                *ob = req_t(l, "attn_output_b.weight", err), *cape = req_t(l, "attn_compressor_ape.weight", err),
                                *ckv = req_t(l, "attn_compressor_kv.weight", err), *cg = req_t(l, "attn_compressor_gate.weight", err),
                                *cn = req_t(l, "attn_compressor_norm.weight", err);
                if (!qa || !qan || !qb || !wkv || !kvn || !oa || !ob || !cape || !ckv || !cg || !cn) return false;

                matmul_q8(qa, d_x_norm_, d_qr_, H, g_.hca_q_lora, T);
                kernels::rms_norm_rows(d_qr_, (const float*) qan->ptr, T, g_.hca_q_lora, g_.rms_eps, d_qr_, stream_);
                matmul_q8(qb, d_qr_, d_q_, g_.hca_q_lora, g_.hca_heads * g_.hca_head_dim, T);
                hca_q_rope(d_q_, d_pos_, T, g_.hca_heads, g_.hca_head_dim, hca_rope_, stream_);
                matmul_q8(agate, d_qr_, d_gate_, g_.hca_q_lora, g_.hca_o_groups * g_.hca_o_lora, T);

                matmul_q8(wkv, d_x_norm_, d_kv_, H, g_.hca_head_dim, T);
                auto& hs = hca_[(size_t) l];
                hca_kv_norm_rope(d_kv_, (const float*) kvn->ptr, d_pos_, T, g_.hca_head_dim, g_.rms_eps, hca_rope_,
                                 d_kv_, stream_);

                matmul_bf16(ckv, d_x_norm_, d_ckv_, H, g_.hca_head_dim, T);
                matmul_bf16(cg, d_x_norm_, d_cscore_, H, g_.hca_head_dim, T);
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                hca_compress(d_ckv_, d_cscore_, (const float*) cape->ptr, (const float*) cn->ptr, d_pos_, T, ratio,
                             g_.hca_head_dim, g_.rms_eps, hca_rope_, hs.state_kv, hs.state_score, hs.comp_cache, fp8,
                             stream_);

                hca_attention(d_q_, hs.comp_cache, hs.win_cache, d_kv_, (const float*) asink->ptr, d_pos_, T,
                              g_.hca_heads, g_.hca_head_dim, g_.window, ratio, fp8, d_attn_o_, stream_);
                hca_win_commit(d_kv_, d_pos_, T, g_.hca_head_dim, g_.window, hs.win_cache, fp8, stream_);
                vha_postmix_f32(d_attn_o_, (const float*) vha_m->ptr, T, g_.hca_heads, g_.hca_head_dim, d_attn_o2_, stream_);
                hca_inv_rope(d_attn_o2_, d_pos_, T, g_.hca_heads, g_.hca_head_dim, hca_rope_, stream_);

                // Grouped output projection wo_a: 8 groups, each group maps [heads/8 * head_dim] -> [o_lora]
                const int hpg = g_.hca_heads / g_.hca_o_groups;
                const int gin = hpg * g_.hca_head_dim;
                const int gout = g_.hca_o_lora;
                const size_t oa_grp_bytes = (size_t) gout * (size_t) (gin / 32) * 34;
                for (int grp = 0; grp < g_.hca_o_groups; ++grp) {
                    DevTensor sub_w{(uint8_t*) oa->ptr + (size_t) grp * oa_grp_bytes, kTypeQ8_0, oa_grp_bytes, {}};
                    for (int t = 0; t < T; ++t) {
                        cudaMemcpyAsync(d_x_in_ + (size_t) t * gin,
                                        d_attn_o2_ + ((size_t) t * g_.hca_heads + grp * hpg) * g_.hca_head_dim,
                                        (size_t) gin * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
                    }
                    matmul_q8(&sub_w, d_x_in_, d_ff_h_, gin, gout, T);
                    for (int t = 0; t < T; ++t) {
                        cudaMemcpyAsync(d_z_ + (size_t) t * (g_.hca_o_groups * gout) + (size_t) grp * gout,
                                        d_ff_h_ + (size_t) t * gout, (size_t) gout * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream_);
                    }
                }
                sigmoid_gate_mul_f32(d_z_, d_gate_, (int64_t) T * g_.hca_o_groups * g_.hca_o_lora, stream_);
                matmul_q8(ob, d_z_, d_sub_out_, g_.hca_o_groups * g_.hca_o_lora, H, T);
            } else {
                const DevTensor *qa = req_t(l, "attn_q_a.weight", err), *qan = req_t(l, "attn_q_a_norm.weight", err),
                                *qb = req_t(l, "attn_q_b.weight", err), *kva = req_t(l, "attn_kv_a_mqa.weight", err),
                                *kvn = req_t(l, "attn_kv_a_norm.weight", err), *kvb = req_t(l, "attn_kv_b.weight", err),
                                *wo = req_t(l, "attn_output.weight", err), *iqb = req_t(l, "indexer.attn_q_b.weight", err),
                                *ikw = req_t(l, "indexer.attn_k.weight", err), *iknw = req_t(l, "indexer.k_norm.weight", err),
                                *iknb = req_t(l, "indexer.k_norm.bias", err), *ipr = req_t(l, "indexer.proj.weight", err);
                if (!qa || !qan || !qb || !kva || !kvn || !kvb || !wo || !iqb || !ikw || !iknw || !iknb || !ipr)
                    return false;

                matmul_q8(qa, d_x_norm_, d_qr_, H, g_.mla_q_lora, T);
                kernels::rms_norm_rows(d_qr_, (const float*) qan->ptr, T, g_.mla_q_lora, g_.rms_eps, d_qr_, stream_);
                matmul_q8(agate, d_qr_, d_gate_, g_.mla_q_lora, g_.mla_heads * g_.mla_v_dim, T);
                matmul_q8(qb, d_qr_, d_q_, g_.mla_q_lora, g_.mla_heads * (g_.mla_qk_nope + g_.mla_qk_rope), T);
                matmul_q8(kva, d_x_norm_, d_kv_, H, g_.mla_kv_lora + g_.mla_qk_rope, T);

                auto& ds = dsa_[(size_t) l];
                dsa_qkv_rope_and_store(d_q_, d_kv_, (const float*) kvn->ptr, d_pos_, T, g_.mla_heads, g_.mla_qk_nope,
                                       g_.mla_qk_rope, g_.mla_kv_lora, g_.rms_eps, dsa_rope_, ds.kv_cache, fp8, stream_);

                matmul_q8(iqb, d_qr_, d_iq_, g_.mla_q_lora, g_.idx_heads * g_.idx_dim, T);
                matmul_bf16(ikw, d_x_norm_, d_ik_, H, g_.idx_dim, T);
                matmul_bf16(ipr, d_x_norm_, d_iw_, H, g_.idx_heads, T);
                dsa_indexer_prepare(d_iq_, d_ik_, (const float*) iknw->ptr, (const float*) iknb->ptr, d_pos_, T,
                                    g_.idx_heads, g_.idx_dim, g_.mla_qk_rope, g_.rms_eps, dsa_rope_, ds.idx_k_cache, fp8,
                                    stream_);
                const int sel_w = g_.idx_top_k + g_.window;
                for (int t0 = 0; t0 < T; t0 += kMaxDecodeWindow) {
                    const int tw = std::min(kMaxDecodeWindow, T - t0);
                    dsa_indexer_topk_merge(d_iq_ + (size_t) t0 * g_.idx_heads * g_.idx_dim,
                                           d_iw_ + (size_t) t0 * g_.idx_heads, ds.idx_k_cache, d_pos_ + t0, tw,
                                           g_.idx_heads, g_.idx_dim, g_.idx_top_k, g_.window, fp8,
                                           d_sel_idx_ + (size_t) t0 * sel_w, stream_, d_idx_scores_, (int) m_.context());
                }

                dsa_mla_attention(d_q_, ds.kv_cache, kvb->ptr, (const float*) asink->ptr, d_sel_idx_, T, g_.mla_heads,
                                  g_.mla_qk_nope, g_.mla_qk_rope, g_.mla_kv_lora, g_.mla_v_dim, g_.idx_top_k + g_.window,
                                  fp8, d_attn_o_, stream_);
                vha_postmix_f32(d_attn_o_, (const float*) vha_m->ptr, T, g_.mla_heads, g_.mla_v_dim, d_attn_o2_, stream_);
                sigmoid_gate_mul_f32(d_attn_o2_, d_gate_, (int64_t) T * g_.mla_heads * g_.mla_v_dim, stream_);
                matmul_q8(wo, d_attn_o2_, d_sub_out_, g_.mla_heads * g_.mla_v_dim, H, T);
            }

            mhc_post(d_sub_out_, res, d_mhc_coef_, kHcCoef, T, H, res_next, stream_);
            std::swap(res, res_next);

            // ---- FFN / Latent MoE sublayer ----
            const DevTensor *fh_fn = req_t(l, "hc_ffn_fn.weight", err),
                            *fh_sc = req_t(l, "hc_ffn_scale.weight", err),
                            *fh_bs = req_t(l, "hc_ffn_base.weight", err),
                            *fnorm = req_t(l, "ffn_norm.weight", err);
            if (!fh_fn || !fh_sc || !fh_bs || !fnorm) return false;

            mhc_mixes(res, (const float*) fh_fn->ptr, (const float*) fh_sc->ptr, (const float*) fh_bs->ptr, T, H,
                      g_.rms_eps, g_.hc_eps, g_.sinkhorn_iters, d_mhc_ws_, d_mhc_coef_, stream_);
            mhc_pre(res, d_mhc_coef_, kHcCoef, T, H, d_x_in_, stream_);
            kernels::rms_norm_rows(d_x_in_, (const float*) fnorm->ptr, T, H, g_.rms_eps, d_x_norm_, stream_);

            if (l < g_.n_dense) {
                const DevTensor *fg = req_t(l, "ffn_gate.weight", err),
                                *fu = req_t(l, "ffn_up.weight", err),
                                *fd = req_t(l, "ffn_down.weight", err);
                if (!fg || !fu || !fd) return false;
                matmul_q8(fg, d_x_norm_, d_ff_g_, H, g_.n_ff, T);
                matmul_q8(fu, d_x_norm_, d_ff_u_, H, g_.n_ff, T);
                situ_glu_f32(d_ff_g_, d_ff_u_, T, g_.n_ff, g_.gate_softcap, g_.up_softcap, d_ff_h_, stream_);
                matmul_q8(fd, d_ff_h_, d_sub_out_, g_.n_ff, H, T);
            } else {
                const DevTensor *rg = req_t(l, "ffn_gate_inp.weight", err),
                                *rb = req_t(l, "exp_probs_b.bias", err),
                                *rs = req_t(l, "exp_probs_scale.weight", err),
                                *ld = req_t(l, "ffn_routed_down.weight", err),
                                *ln = req_t(l, "ffn_latent_norm.weight", err),
                                *lu = req_t(l, "ffn_routed_up.weight", err),
                                *sg = req_t(l, "ffn_gate_shexp.weight", err),
                                *su = req_t(l, "ffn_up_shexp.weight", err),
                                *sd = req_t(l, "ffn_down_shexp.weight", err);
                if (!rg || !rb || !rs || !ld || !ln || !lu || !sg || !su || !sd) return false;

                // Split-feature routing
                matmul_f32((const float*) rg->ptr, d_x_norm_, d_rlogits_, H, 2 * g_.n_expert, T);
                split_sigmoid_route(d_rlogits_, (const float*) rb->ptr, (const float*) rs->ptr, T, g_.n_expert,
                                    g_.n_used, g_.norm_topk_prob, d_exp_ids_, d_exp_w_, stream_);

                // Shared expert
                matmul_q8(sg, d_x_norm_, d_ff_g_, H, g_.n_ff_exp, T);
                matmul_q8(su, d_x_norm_, d_ff_u_, H, g_.n_ff_exp, T);
                situ_glu_f32(d_ff_g_, d_ff_u_, T, g_.n_ff_exp, g_.gate_softcap, g_.up_softcap, d_ff_h_, stream_);
                matmul_q8(sd, d_ff_h_, d_sub_out_, g_.n_ff_exp, H, T);

                // Routed experts in latent space
                matmul_q8(ld, d_x_norm_, d_x_lat_, H, g_.n_latent, T);
                if (!run_moe_experts(l - g_.n_dense, T, err)) return false;
                weighted_sum_f32(d_exp_out_, d_exp_w_, T, g_.n_used, g_.n_latent, d_lat_sum_, stream_);
                kernels::rms_norm_rows(d_lat_sum_, (const float*) ln->ptr, T, g_.n_latent, g_.rms_eps, d_lat_norm_, stream_);
                matmul_q8(lu, d_lat_norm_, d_routed_out_, g_.n_latent, H, T);
                add_f32(d_sub_out_, d_routed_out_, (int64_t) T * H, stream_);
            }

            mhc_post(d_sub_out_, res, d_mhc_coef_, kHcCoef, T, H, res_next, stream_);
            std::swap(res, res_next);
        }

        if (want_logits) {
            const DevTensor *hfn = req_t("hc_head_fn.weight", err),
                            *hsc = req_t("hc_head_scale.weight", err),
                            *hbs = req_t("hc_head_base.weight", err),
                            *onorm = req_t("output_norm.weight", err),
                            *head = req_t("output.weight", err),
                            *mmr = req_t("output_multimax_ranges", err),
                            *mmt = req_t("output_multimax_ts", err);
            if (!hfn || !hsc || !hbs || !onorm || !head || !mmr || !mmt) return false;

            mhc_contract(res, (const float*) hfn->ptr, (const float*) hsc->ptr, (const float*) hbs->ptr, T, H,
                         g_.rms_eps, g_.hc_eps, d_mhc_ws_, d_mhc_coef_, d_x_in_, stream_);
            kernels::rms_norm_rows(d_x_in_, (const float*) onorm->ptr, T, H, g_.rms_eps, d_x_norm_, stream_);
            if (head->type == kTypeQ8_0) matmul_q8(head, d_x_norm_, d_logits_, H, g_.n_vocab, T);
            else matmul_bf16(head, d_x_norm_, d_logits_, H, g_.n_vocab, T);
            multimax_f32(d_logits_, (const float*) mmr->ptr, (const float*) mmt->ptr, T, g_.n_vocab, stream_);
        }
        return true;
    }

    void release() {
        for (auto& s : hca_) {
            dfree(s.win_cache); dfree(s.comp_cache); dfree(s.state_kv); dfree(s.state_score);
        }
        for (auto& s : dsa_) {
            dfree(s.kv_cache); dfree(s.idx_k_cache);
        }
        dfree(d_pos_); dfree(d_res_a_); dfree(d_res_b_); dfree(d_x_in_); dfree(d_x_norm_); dfree(d_sub_out_);
        dfree(d_mhc_ws_); dfree(d_mhc_coef_); dfree(d_qr_); dfree(d_q_); dfree(d_kv_); dfree(d_ckv_); dfree(d_cscore_);
        dfree(d_attn_o_); dfree(d_attn_o2_); dfree(d_gate_); dfree(d_z_); dfree(d_iq_); dfree(d_ik_); dfree(d_iw_);
        dfree(d_sel_idx_); dfree(d_idx_scores_); dfree(d_ff_g_); dfree(d_ff_u_); dfree(d_ff_h_); dfree(d_rlogits_); dfree(d_exp_ids_);
        dfree(d_exp_w_); dfree(d_x_lat_); dfree(d_lat_sum_); dfree(d_lat_norm_); dfree(d_routed_out_); dfree(d_exp_out_);
        dfree(d_logits_); dfree(d_picks_); dfree(d_lp_scratch_); dfree(d_lp_out_); dfree(d_mmq_bounds_); dfree(d_mmq_iota_);
        dfree(d_xq_); dfree(d_exp_scratch_); dfree(d_grp_ptr_); dfree(d_grp_start_); dfree(d_n_groups_);
        dfree(d_ent_dst_); dfree(d_ent_tok_);
        if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }

    ErnieModel& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    prefill::mmq::Context mmq_ctx_;
    RopeTable hca_rope_{}, dsa_rope_{};

    std::vector<HcaLayerState> hca_;
    std::vector<DsaLayerState> dsa_;
    int64_t kv_bytes_ = 0;
    int last_window_n_ = 0;
    const float* logit_bias_ = nullptr;

    int32_t *d_pos_ = nullptr, *d_sel_idx_ = nullptr, *d_exp_ids_ = nullptr, *d_picks_ = nullptr;
    int32_t *d_mmq_bounds_ = nullptr, *d_mmq_iota_ = nullptr;
    int32_t *d_grp_start_ = nullptr, *d_n_groups_ = nullptr, *d_ent_dst_ = nullptr, *d_ent_tok_ = nullptr;
    unsigned long long* d_grp_ptr_ = nullptr;
    float *d_res_a_ = nullptr, *d_res_b_ = nullptr, *d_x_in_ = nullptr, *d_x_norm_ = nullptr, *d_sub_out_ = nullptr;
    float *d_mhc_ws_ = nullptr, *d_mhc_coef_ = nullptr;
    float *d_qr_ = nullptr, *d_q_ = nullptr, *d_kv_ = nullptr, *d_ckv_ = nullptr, *d_cscore_ = nullptr;
    float *d_attn_o_ = nullptr, *d_attn_o2_ = nullptr, *d_gate_ = nullptr, *d_z_ = nullptr;
    float *d_iq_ = nullptr, *d_ik_ = nullptr, *d_iw_ = nullptr, *d_idx_scores_ = nullptr;
    float *d_ff_g_ = nullptr, *d_ff_u_ = nullptr, *d_ff_h_ = nullptr;
    float *d_rlogits_ = nullptr, *d_exp_w_ = nullptr, *d_x_lat_ = nullptr, *d_lat_sum_ = nullptr;
    float *d_lat_norm_ = nullptr, *d_routed_out_ = nullptr, *d_exp_out_ = nullptr;
    float *d_logits_ = nullptr, *d_lp_scratch_ = nullptr, *d_lp_out_ = nullptr;
    void *d_xq_ = nullptr, *d_exp_scratch_ = nullptr;

    std::vector<float> h_emb_;
    std::vector<int32_t> h_pos_, h_ids_, h_grp_start_, h_ent_dst_, h_ent_tok_;
    std::vector<unsigned long long> h_grp_ptr_;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(ErnieModel& model, std::string& err) {
    auto s = std::make_unique<ErnieSession>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::ernie55
