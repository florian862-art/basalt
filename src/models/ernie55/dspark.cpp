// src/models/ernie55/dspark.cpp - ERNIE-5.5-Flash DSpark speculative drafter runtime.
#include "models/ernie55/dspark.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/log.hpp"
#include "basalt/models/ernie55/kernels.hpp"
#include "models/ernie55/model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <vector>

namespace basalt::models::ernie55 {
namespace {

constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeMXFP4 = 39;
constexpr int kMaxInjectBatch = 8;

inline float bf16_to_f32(uint16_t u) {
    uint32_t bits = (uint32_t) u << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("ernie55 dspark: ") + what + ": " + cudaGetErrorString(e);
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

struct Window {
    std::vector<uint16_t*> ctx_ring;   ///< [n_stages], each [kDsparkWinCap, head_dim] FP16
    int64_t ctx_start = 0;
    int64_t ctx_end = 0;
};

}  // namespace

struct ErnieDsparkDrafter::Impl {
    const ErnieModel* main = nullptr;
    Geometry g;
    DsparkPack dp;
    int device = 0;
    RopeTable rope{};

    std::map<std::string, DevTensor> dev;
    uint64_t dev_bytes = 0;
    std::vector<uint16_t> markov_w1;    ///< BF16 [n_vocab, markov_rank] in host RAM
    std::vector<uint8_t> experts_ram;   ///< MXFP4 [n_stages, n_expert, blob_bytes] in host RAM
    kernels::cpu::NativeFmt nat_fmt{};

    std::vector<Window> windows;
    int next_win = 0;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudaEvent_t sync_ev = nullptr;

    // Device scratch buffers sized for max(kDsparkBlock, kMaxInjectBatch) = 8 rows
    int32_t* d_pos = nullptr;
    float *d_res_a = nullptr, *d_res_b = nullptr, *d_x_in = nullptr, *d_x_norm = nullptr, *d_sub_out = nullptr;
    float *d_mhc_ws = nullptr, *d_mhc_coef = nullptr;
    float *d_qr = nullptr, *d_q = nullptr, *d_kv = nullptr;
    float *d_attn_o = nullptr, *d_attn_o2 = nullptr, *d_gate = nullptr, *d_z = nullptr;
    float *d_ff_g = nullptr, *d_ff_u = nullptr, *d_ff_h = nullptr;
    float *d_rlogits = nullptr, *d_exp_w = nullptr;
    int32_t* d_exp_ids = nullptr;
    float *d_x_lat = nullptr, *d_exp_out = nullptr, *d_lat_sum = nullptr, *d_lat_norm = nullptr, *d_routed_out = nullptr;
    float *d_hcout = nullptr, *d_hnorm = nullptr;
    float *d_base_logits = nullptr, *d_mbias = nullptr, *d_me = nullptr;
    float *d_amax_v = nullptr, *d_conf = nullptr;
    int32_t *d_amax_i = nullptr, *d_tok = nullptr;
    void* d_xq = nullptr;

    // Pinned host staging buffers for fast GPU<->CPU transfers
    float* h_emb = nullptr;
    float* h_x_lat = nullptr;
    float* h_exp_out = nullptr;
    int32_t* h_exp_ids = nullptr;
    float* h_me = nullptr;
    int32_t* h_tok = nullptr;
    float* h_conf = nullptr;
    std::vector<uint8_t> h_nact;

    ~Impl() {
        for (auto& [k, t] : dev)
            if (t.ptr) cudaFree(t.ptr);
        for (auto& w : windows)
            for (uint16_t* r : w.ctx_ring) dfree(r);
        dfree(d_pos); dfree(d_res_a); dfree(d_res_b); dfree(d_x_in); dfree(d_x_norm); dfree(d_sub_out);
        dfree(d_mhc_ws); dfree(d_mhc_coef); dfree(d_qr); dfree(d_q); dfree(d_kv);
        dfree(d_attn_o); dfree(d_attn_o2); dfree(d_gate); dfree(d_z);
        dfree(d_ff_g); dfree(d_ff_u); dfree(d_ff_h); dfree(d_rlogits); dfree(d_exp_w); dfree(d_exp_ids);
        dfree(d_x_lat); dfree(d_exp_out); dfree(d_lat_sum); dfree(d_lat_norm); dfree(d_routed_out);
        dfree(d_hcout); dfree(d_hnorm); dfree(d_base_logits); dfree(d_mbias); dfree(d_me);
        dfree(d_amax_v); dfree(d_conf); dfree(d_amax_i); dfree(d_tok); dfree(d_xq);
        if (h_emb) cudaFreeHost(h_emb);
        if (h_x_lat) cudaFreeHost(h_x_lat);
        if (h_exp_out) cudaFreeHost(h_exp_out);
        if (h_exp_ids) cudaFreeHost(h_exp_ids);
        if (h_me) cudaFreeHost(h_me);
        if (h_tok) cudaFreeHost(h_tok);
        if (h_conf) cudaFreeHost(h_conf);
        if (sync_ev) cudaEventDestroy(sync_ev);
        if (cublas) cublasDestroy(cublas);
        if (stream) cudaStreamDestroy(stream);
    }

    const DevTensor* req(const std::string& name, std::string& err) const {
        auto it = dev.find(name);
        if (it == dev.end()) {
            err = "ernie55 dspark: missing tensor " + name;
            return nullptr;
        }
        return &it->second;
    }

    const DevTensor* req(int s, const char* suf, std::string& err) const {
        return req("blk." + std::to_string(s) + "." + suf, err);
    }

    void matmul_q8_on(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int T, cudaStream_t st) const {
        kernels::native_quantize_q8_1(x, d_xq, n_in, T, st);
        kernels::native_q8_0_mmvq(w->ptr, d_xq, y, n_in, n_out, T, st);
    }

    void matmul_f32_on(const float* w, const float* x, float* y, int n_in, int n_out, int T, cublasHandle_t cb) const {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &alpha, w, n_in, x, n_in, &beta, y, n_out);
    }

    bool run_stage_experts(int s, int B, std::string& err) {
        auto& st = main->shared_tier();
        if (!st.tier || !st.tier->pool()) {
            err = "ernie55 dspark: CPU expert pool not ready";
            return false;
        }
        kernels::cpu::ExpertPool* pool = st.tier->pool();
        const int K = g.n_used;
        const int L = g.n_latent;
        const int entries = B * K;

        cudaMemcpyAsync(h_exp_ids, d_exp_ids, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_x_lat, d_x_lat, (size_t) B * (size_t) L * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (!ck(cudaStreamSynchronize(stream), "stage expert D2H sync", err)) return false;

        // Quantize latent activations for each block row
        for (int t = 0; t < B; ++t) {
            kernels::cpu::native_quant_act(nat_fmt, h_x_lat + (size_t) t * L,
                                           h_nact.data() + (size_t) t * nat_fmt.act_bytes);
        }

        // Deduplicate experts across the B block rows into ExpertJobMulti jobs
        struct Slot {
            int expert = -1;
            int nt = 0;
            int tok[kernels::cpu::MAXT] = {};
            int row[kernels::cpu::MAXT] = {};
        };
        std::array<Slot, 128> slots{};
        int n_slots = 0;
        for (int t = 0; t < B; ++t) {
            for (int k = 0; k < K; ++k) {
                const int e = h_exp_ids[t * K + k];
                if (e < 0 || e >= dp.n_expert) {
                    err = "ernie55 dspark: routed expert id out of range";
                    return false;
                }
                int idx = -1;
                for (int i = 0; i < n_slots; ++i) {
                    if (slots[(size_t) i].expert == e) {
                        idx = i;
                        break;
                    }
                }
                if (idx < 0) {
                    idx = n_slots++;
                    slots[(size_t) idx].expert = e;
                    slots[(size_t) idx].nt = 0;
                }
                auto& sl = slots[(size_t) idx];
                sl.tok[sl.nt] = t;
                sl.row[sl.nt] = t * K + k;
                sl.nt++;
            }
        }

        const uint8_t* stage_base = experts_ram.data() + (size_t) s * dp.stage_stride;
        std::array<kernels::cpu::ExpertJobMulti, 128> jobs{};
        for (int i = 0; i < n_slots; ++i) {
            const auto& sl = slots[(size_t) i];
            auto& jb = jobs[(size_t) i];
            jb.blob = stage_base + (size_t) sl.expert * dp.blob_bytes;
            jb.nt = sl.nt;
            for (int j = 0; j < sl.nt; ++j) {
                jb.nact[j] = h_nact.data() + (size_t) sl.tok[j] * nat_fmt.act_bytes;
                jb.out[j] = h_exp_out + (size_t) sl.row[j] * L;
            }
        }

        pool->run_split_multi_native(nat_fmt, jobs.data(), n_slots);

        cudaMemcpyAsync(d_exp_out, h_exp_out, (size_t) entries * (size_t) L * sizeof(float), cudaMemcpyHostToDevice,
                        stream);
        return true;
    }
};

ErnieDsparkDrafter::ErnieDsparkDrafter(std::unique_ptr<Impl> p) : p_(std::move(p)) {}
ErnieDsparkDrafter::~ErnieDsparkDrafter() = default;

std::unique_ptr<ErnieDsparkDrafter> ErnieDsparkDrafter::plan_load(const ErnieModel& main, int file_handle,
                                                                  core::LoadPlan& reads, int max_windows,
                                                                  std::string& err) {
    const Pack& pack = main.pack();
    if (!pack.dspark.present) {
        err = "ernie55 dspark: pack has no dspark section";
        return nullptr;
    }
    auto p = std::make_unique<Impl>();
    p->main = &main;
    p->g = pack.geo;
    p->dp = pack.dspark;
    p->device = main.device();
    cudaSetDevice(p->device);

    if (p->dp.block_size <= 0 || p->dp.block_size > kDsparkBlock) {
        err = "ernie55 dspark: unsupported block_size " + std::to_string(p->dp.block_size);
        return nullptr;
    }

    p->rope = rope_table_plain(p->dp.rope_theta);

    if (!kernels::cpu::native_fmt(kTypeMXFP4, kTypeMXFP4, p->g.n_latent, p->g.n_ff_exp, p->nat_fmt, err)) {
        err = "ernie55 dspark: " + err;
        return nullptr;
    }
    p->nat_fmt.gate_softcap = p->g.gate_softcap;
    p->nat_fmt.up_softcap = p->g.up_softcap;

    if (!ck(cudaStreamCreate(&p->stream), "stream", err) ||
        !ck(cudaEventCreateWithFlags(&p->sync_ev, cudaEventDisableTiming), "event", err))
        return nullptr;
    if (cublasCreate(&p->cublas) != CUBLAS_STATUS_SUCCESS) {
        err = "ernie55 dspark: cublasCreate failed";
        return nullptr;
    }
    cublasSetStream(p->cublas, p->stream);

    // Load dense tensors: markov_w1 to host RAM (BF16), everything else to GPU 0
    for (const auto& [name, t] : p->dp.tensors) {
        if (name == "markov_w1.weight") {
            if (t.type != kTypeBF16 || t.shape.size() != 2 || t.shape[0] != (uint64_t) p->dp.markov_rank ||
                t.shape[1] != (uint64_t) p->g.n_vocab) {
                err = "ernie55 dspark: markov_w1.weight shape/type mismatch";
                return nullptr;
            }
            p->markov_w1.resize(t.bytes / 2);
            core::LoadItem it;
            it.file = file_handle;
            it.off = t.offset;
            it.bytes = t.bytes;
            it.host = reinterpret_cast<uint8_t*>(p->markov_w1.data());
            reads.add(it);
            continue;
        }
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        const uint64_t pad = (d.type == kTypeQ8_0) ? 256 : 0;
        if (!ck(cudaMalloc(&d.ptr, d.bytes + pad), name.c_str(), err)) return nullptr;
        if (pad) cudaMemset((uint8_t*) d.ptr + d.bytes, 0, pad);
        core::LoadItem it;
        it.file = file_handle;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = p->device;
        reads.add(it);
        p->dev_bytes += d.bytes + pad;
        p->dev.emplace(name, std::move(d));
    }

    // Load all stage MXFP4 routed experts into host RAM
    p->experts_ram.resize(p->dp.experts_bytes);
    {
        // Split into 64 MiB LoadItems for parallel pipelined reading
        constexpr uint64_t kChunk = 64ull << 20;
        uint64_t done = 0;
        while (done < p->dp.experts_bytes) {
            const uint64_t take = std::min(kChunk, p->dp.experts_bytes - done);
            core::LoadItem it;
            it.file = file_handle;
            it.off = p->dp.experts_offset + done;
            it.bytes = take;
            it.host = p->experts_ram.data() + done;
            reads.add(it);
            done += take;
        }
    }

    // Allocate per-session FP16 context rings
    const int nwin = std::max(1, max_windows);
    p->windows.resize((size_t) nwin);
    const size_t ring_elems = (size_t) kDsparkWinCap * (size_t) p->g.hca_head_dim;
    for (int w = 0; w < nwin; ++w) {
        p->windows[(size_t) w].ctx_ring.resize((size_t) p->dp.n_stages, nullptr);
        for (int s = 0; s < p->dp.n_stages; ++s) {
            if (!dalloc(p->windows[(size_t) w].ctx_ring[(size_t) s], ring_elems, "ctx_ring", err)) return nullptr;
            cudaMemsetAsync(p->windows[(size_t) w].ctx_ring[(size_t) s], 0, ring_elems * sizeof(uint16_t), p->stream);
            p->dev_bytes += ring_elems * sizeof(uint16_t);
        }
    }

    // Allocate device scratch buffers for max(block_size, kMaxInjectBatch) = 8 rows
    const int maxR = std::max(kDsparkBlock, kMaxInjectBatch);
    const int H = p->g.n_embd;
    const int q_dim = p->g.hca_heads * p->g.hca_head_dim;
    const int gate_dim = p->g.hca_o_groups * p->g.hca_o_lora;
    const int feat_dim = (int) p->dp.target_layers.size() * H;
    const int q8_cols_max = std::max({feat_dim, H, gate_dim, p->g.n_ff_exp, p->g.hca_q_lora, p->g.n_latent,
                                      p->dp.markov_rank});
    const int max_entries = maxR * p->g.n_used;

    if (!dalloc(p->d_pos, maxR, "d_pos", err) ||
        !dalloc(p->d_res_a, (size_t) maxR * kHcMult * H, "d_res_a", err) ||
        !dalloc(p->d_res_b, (size_t) maxR * kHcMult * H, "d_res_b", err) ||
        !dalloc(p->d_x_in, (size_t) maxR * std::max(H, (p->g.hca_heads / p->g.hca_o_groups) * p->g.hca_head_dim), "d_x_in", err) ||
        !dalloc(p->d_x_norm, (size_t) maxR * H, "d_x_norm", err) ||
        !dalloc(p->d_sub_out, (size_t) maxR * H, "d_sub_out", err) ||
        !dalloc(p->d_mhc_ws, mhc_workspace_bytes(maxR) / sizeof(float), "d_mhc_ws", err) ||
        !dalloc(p->d_mhc_coef, (size_t) maxR * kHcCoef, "d_mhc_coef", err) ||
        !dalloc(p->d_qr, (size_t) maxR * p->g.hca_q_lora, "d_qr", err) ||
        !dalloc(p->d_q, (size_t) maxR * q_dim, "d_q", err) ||
        !dalloc(p->d_kv, (size_t) maxR * p->g.hca_head_dim, "d_kv", err) ||
        !dalloc(p->d_attn_o, (size_t) maxR * q_dim, "d_attn_o", err) ||
        !dalloc(p->d_attn_o2, (size_t) maxR * q_dim, "d_attn_o2", err) ||
        !dalloc(p->d_gate, (size_t) maxR * gate_dim, "d_gate", err) ||
        !dalloc(p->d_z, (size_t) maxR * gate_dim, "d_z", err) ||
        !dalloc(p->d_ff_g, (size_t) maxR * p->g.n_ff_exp, "d_ff_g", err) ||
        !dalloc(p->d_ff_u, (size_t) maxR * p->g.n_ff_exp, "d_ff_u", err) ||
        !dalloc(p->d_ff_h, (size_t) maxR * std::max(p->g.n_ff_exp, p->g.hca_o_lora), "d_ff_h", err) ||
        !dalloc(p->d_rlogits, (size_t) maxR * 2 * p->g.n_expert, "d_rlogits", err) ||
        !dalloc(p->d_exp_w, (size_t) max_entries, "d_exp_w", err) ||
        !dalloc(p->d_exp_ids, (size_t) max_entries, "d_exp_ids", err) ||
        !dalloc(p->d_x_lat, (size_t) maxR * p->g.n_latent, "d_x_lat", err) ||
        !dalloc(p->d_exp_out, (size_t) max_entries * p->g.n_latent, "d_exp_out", err) ||
        !dalloc(p->d_lat_sum, (size_t) maxR * p->g.n_latent, "d_lat_sum", err) ||
        !dalloc(p->d_lat_norm, (size_t) maxR * p->g.n_latent, "d_lat_norm", err) ||
        !dalloc(p->d_routed_out, (size_t) maxR * H, "d_routed_out", err) ||
        !dalloc(p->d_hcout, (size_t) maxR * H, "d_hcout", err) ||
        !dalloc(p->d_hnorm, (size_t) maxR * H, "d_hnorm", err) ||
        !dalloc(p->d_base_logits, (size_t) kDsparkBlock * (size_t) p->g.n_vocab, "d_base_logits", err) ||
        !dalloc(p->d_mbias, (size_t) p->g.n_vocab, "d_mbias", err) ||
        !dalloc(p->d_me, (size_t) p->dp.markov_rank, "d_me", err) ||
        !dalloc(p->d_amax_v, 256, "d_amax_v", err) ||
        !dalloc(p->d_amax_i, 256, "d_amax_i", err) ||
        !dalloc(p->d_tok, 1, "d_tok", err) ||
        !dalloc(p->d_conf, 1, "d_conf", err))
        return nullptr;

    const size_t q8_bytes = kernels::native_q8_1_bytes(q8_cols_max, maxR);
    if (!ck(cudaMalloc(&p->d_xq, q8_bytes), "d_xq", err)) return nullptr;

    // Pinned host staging buffers
    if (!ck(cudaHostAlloc(&p->h_emb, (size_t) maxR * H * sizeof(float), cudaHostAllocPortable), "h_emb", err) ||
        !ck(cudaHostAlloc(&p->h_x_lat, (size_t) maxR * p->g.n_latent * sizeof(float), cudaHostAllocPortable), "h_x_lat", err) ||
        !ck(cudaHostAlloc(&p->h_exp_out, (size_t) max_entries * p->g.n_latent * sizeof(float), cudaHostAllocPortable), "h_exp_out", err) ||
        !ck(cudaHostAlloc(&p->h_exp_ids, (size_t) max_entries * sizeof(int32_t), cudaHostAllocPortable), "h_exp_ids", err) ||
        !ck(cudaHostAlloc(&p->h_me, (size_t) p->dp.markov_rank * sizeof(float), cudaHostAllocPortable), "h_me", err) ||
        !ck(cudaHostAlloc(&p->h_tok, sizeof(int32_t), cudaHostAllocPortable), "h_tok", err) ||
        !ck(cudaHostAlloc(&p->h_conf, sizeof(float), cudaHostAllocPortable), "h_conf", err))
        return nullptr;
    p->h_nact.resize((size_t) maxR * p->nat_fmt.act_bytes);

    return std::unique_ptr<ErnieDsparkDrafter>(new ErnieDsparkDrafter(std::move(p)));
}

int ErnieDsparkDrafter::open_window() {
    if (p_->next_win >= (int) p_->windows.size()) return -1;
    return p_->next_win++;
}

int ErnieDsparkDrafter::feature_slot(int layer) const {
    for (size_t i = 0; i < p_->dp.target_layers.size(); ++i)
        if (p_->dp.target_layers[i] == layer) return (int) i;
    return -1;
}

int ErnieDsparkDrafter::feature_floats() const { return (int) p_->dp.target_layers.size() * p_->g.n_embd; }
int ErnieDsparkDrafter::block_size() const { return p_->dp.block_size; }
int ErnieDsparkDrafter::n_stages() const { return p_->dp.n_stages; }
uint64_t ErnieDsparkDrafter::device_bytes() const { return p_->dev_bytes; }
uint64_t ErnieDsparkDrafter::host_bytes() const {
    return (uint64_t) p_->markov_w1.size() * 2 + (uint64_t) p_->experts_ram.size();
}

void ErnieDsparkDrafter::forget_from(int w, int64_t pos) {
    if (w < 0 || w >= (int) p_->windows.size()) return;
    auto& win = p_->windows[(size_t) w];
    if (pos <= win.ctx_start) {
        win.ctx_start = pos;
        win.ctx_end = pos;
    } else if (pos < win.ctx_end) {
        win.ctx_end = pos;
    }
}

void ErnieDsparkDrafter::reset(int w) {
    if (w < 0 || w >= (int) p_->windows.size()) return;
    auto& win = p_->windows[(size_t) w];
    win.ctx_start = 0;
    win.ctx_end = 0;
}

bool ErnieDsparkDrafter::inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream,
                                       std::string& err) {
    if (n_rows <= 0) return true;
    if (w < 0 || w >= (int) p_->windows.size()) {
        err = "ernie55 dspark: invalid window";
        return false;
    }
    cudaSetDevice(p_->device);
    cudaStream_t st = (cudaStream_t) stream;
    const DevTensor *fproj = p_->req("feature_proj.weight", err),
                    *fnorm = p_->req("feature_norm.weight", err);
    if (!fproj || !fnorm) return false;

    const int H = p_->g.n_embd;
    const int F = feature_floats();
    auto& win = p_->windows[(size_t) w];

    for (int r0 = 0; r0 < n_rows; r0 += kMaxInjectBatch) {
        const int tw = std::min(kMaxInjectBatch, n_rows - r0);
        const float* fslice = d_features + (size_t) r0 * (size_t) F;
        p_->matmul_q8_on(fproj, fslice, p_->d_x_norm, F, H, tw, st);
        kernels::rms_norm_rows(p_->d_x_norm, (const float*) fnorm->ptr, tw, H, p_->g.rms_eps, p_->d_x_norm, st);
        for (int s = 0; s < p_->dp.n_stages; ++s) {
            const DevTensor *wkv = p_->req(s, "attn_kv.weight", err),
                            *kvn = p_->req(s, "attn_kv_a_norm.weight", err);
            if (!wkv || !kvn) return false;
            p_->matmul_q8_on(wkv, p_->d_x_norm, p_->d_kv, H, p_->g.hca_head_dim, tw, st);
            dspark_kv_ctx_store(p_->d_kv, (const float*) kvn->ptr, pos0 + r0, tw, p_->g.hca_head_dim, kDsparkWinCap,
                                p_->g.rms_eps, p_->rope, win.ctx_ring[(size_t) s], st);
        }
    }

    if (win.ctx_end == 0 && pos0 > 0) win.ctx_start = pos0;
    else if (pos0 > win.ctx_end) win.ctx_start = pos0;
    win.ctx_end = pos0 + n_rows;
    if (win.ctx_end - win.ctx_start > kDsparkWinCap) win.ctx_start = win.ctx_end - kDsparkWinCap;
    return true;
}

bool ErnieDsparkDrafter::draft(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream, DsparkDraft& out,
                               std::string& err) {
    out.n = 0;
    if (w < 0 || w >= (int) p_->windows.size()) {
        err = "ernie55 dspark: invalid window";
        return false;
    }
    cudaSetDevice(p_->device);

    // Order DSpark stream after caller_stream (e.g. inject_device)
    if (caller_stream && caller_stream != (void*) p_->stream) {
        if (!ck(cudaEventRecord(p_->sync_ev, (cudaStream_t) caller_stream), "record sync_ev", err) ||
            !ck(cudaStreamWaitEvent(p_->stream, p_->sync_ev, 0), "wait sync_ev", err))
            return false;
    }

    const int B = p_->dp.block_size;
    const int n_gen = std::clamp(n_out, 1, B);
    const int H = p_->g.n_embd;
    const auto& g = p_->g;
    const auto& win = p_->windows[(size_t) w];

    // Prepare block input tokens [anchor, noise, ..., noise] and positions [pos, pos+1, ..., pos+B-1]
    int32_t h_pos[kDsparkBlock];
    for (int r = 0; r < B; ++r) {
        const int32_t tok = (r == 0) ? anchor : p_->dp.noise_token_id;
        h_pos[r] = (int32_t) (pos + r);
        if (!p_->main->embedding_row(tok, p_->h_emb + (size_t) r * H)) {
            err = "ernie55 dspark: token id out of range";
            return false;
        }
    }
    cudaMemcpyAsync(p_->d_pos, h_pos, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, p_->stream);
    cudaMemcpyAsync(p_->d_x_in, p_->h_emb, (size_t) B * (size_t) H * sizeof(float), cudaMemcpyHostToDevice, p_->stream);
    broadcast_streams(p_->d_x_in, B, H, p_->d_res_a, p_->stream);

    float* res = p_->d_res_a;
    float* res_next = p_->d_res_b;
    const int64_t eff_pos0 = std::min(pos, win.ctx_end);

    for (int s = 0; s < p_->dp.n_stages; ++s) {
        // ---- Attention sublayer (SWA + bidirectional block) ----
        const DevTensor *hc_fn = p_->req(s, "hc_attn_fn.weight", err),
                        *hc_sc = p_->req(s, "hc_attn_scale.weight", err),
                        *hc_bs = p_->req(s, "hc_attn_base.weight", err),
                        *anorm = p_->req(s, "attn_norm.weight", err),
                        *qa = p_->req(s, "attn_q_a.weight", err),
                        *qan = p_->req(s, "attn_q_a_norm.weight", err),
                        *qb = p_->req(s, "attn_q_b.weight", err),
                        *wkv = p_->req(s, "attn_kv.weight", err),
                        *kvn = p_->req(s, "attn_kv_a_norm.weight", err),
                        *asink = p_->req(s, "attn_sinks.weight", err),
                        *vha_m = p_->req(s, "attn_vha_mix.weight", err),
                        *agate = p_->req(s, "attn_gate.weight", err),
                        *oa = p_->req(s, "attn_output_a.weight", err),
                        *ob = p_->req(s, "attn_output_b.weight", err);
        if (!hc_fn || !hc_sc || !hc_bs || !anorm || !qa || !qan || !qb || !wkv || !kvn || !asink || !vha_m || !agate ||
            !oa || !ob)
            return false;

        mhc_mixes(res, (const float*) hc_fn->ptr, (const float*) hc_sc->ptr, (const float*) hc_bs->ptr, B, H,
                  g.rms_eps, g.hc_eps, g.sinkhorn_iters, p_->d_mhc_ws, p_->d_mhc_coef, p_->stream);
        mhc_pre(res, p_->d_mhc_coef, kHcCoef, B, H, p_->d_x_in, p_->stream);
        kernels::rms_norm_rows(p_->d_x_in, (const float*) anorm->ptr, B, H, g.rms_eps, p_->d_x_norm, p_->stream);

        p_->matmul_q8_on(qa, p_->d_x_norm, p_->d_qr, H, g.hca_q_lora, B, p_->stream);
        kernels::rms_norm_rows(p_->d_qr, (const float*) qan->ptr, B, g.hca_q_lora, g.rms_eps, p_->d_qr, p_->stream);
        p_->matmul_q8_on(qb, p_->d_qr, p_->d_q, g.hca_q_lora, g.hca_heads * g.hca_head_dim, B, p_->stream);
        hca_q_rope(p_->d_q, p_->d_pos, B, g.hca_heads, g.hca_head_dim, p_->rope, p_->stream);
        p_->matmul_q8_on(agate, p_->d_qr, p_->d_gate, g.hca_q_lora, g.hca_o_groups * g.hca_o_lora, B, p_->stream);

        p_->matmul_q8_on(wkv, p_->d_x_norm, p_->d_kv, H, g.hca_head_dim, B, p_->stream);
        hca_kv_norm_rope(p_->d_kv, (const float*) kvn->ptr, p_->d_pos, B, g.hca_head_dim, g.rms_eps, p_->rope,
                         p_->d_kv, p_->stream);

        dspark_swa_block_attn(p_->d_q, win.ctx_ring[(size_t) s], p_->d_kv, (const float*) asink->ptr, eff_pos0,
                              win.ctx_start, B, g.hca_heads, g.hca_head_dim, p_->dp.window, kDsparkWinCap,
                              p_->d_attn_o, p_->stream);
        vha_postmix_f32(p_->d_attn_o, (const float*) vha_m->ptr, B, g.hca_heads, g.hca_head_dim, p_->d_attn_o2,
                        p_->stream);
        hca_inv_rope(p_->d_attn_o2, p_->d_pos, B, g.hca_heads, g.hca_head_dim, p_->rope, p_->stream);

        // Grouped wo_a (8 groups of [1024, 4096]) -> sigmoid gate -> wo_b
        const int hpg = g.hca_heads / g.hca_o_groups;
        const int gin = hpg * g.hca_head_dim;
        const int gout = g.hca_o_lora;
        const size_t oa_grp_bytes = (size_t) gout * (size_t) (gin / 32) * 34;
        for (int grp = 0; grp < g.hca_o_groups; ++grp) {
            DevTensor sub_w{(uint8_t*) oa->ptr + (size_t) grp * oa_grp_bytes, kTypeQ8_0, oa_grp_bytes, {}};
            for (int r = 0; r < B; ++r) {
                cudaMemcpyAsync(p_->d_x_in + (size_t) r * gin,
                                p_->d_attn_o2 + ((size_t) r * g.hca_heads + grp * hpg) * g.hca_head_dim,
                                (size_t) gin * sizeof(float), cudaMemcpyDeviceToDevice, p_->stream);
            }
            p_->matmul_q8_on(&sub_w, p_->d_x_in, p_->d_ff_h, gin, gout, B, p_->stream);
            for (int r = 0; r < B; ++r) {
                cudaMemcpyAsync(p_->d_z + (size_t) r * (g.hca_o_groups * gout) + (size_t) grp * gout,
                                p_->d_ff_h + (size_t) r * gout, (size_t) gout * sizeof(float),
                                cudaMemcpyDeviceToDevice, p_->stream);
            }
        }
        sigmoid_gate_mul_f32(p_->d_z, p_->d_gate, (int64_t) B * g.hca_o_groups * g.hca_o_lora, p_->stream);
        p_->matmul_q8_on(ob, p_->d_z, p_->d_sub_out, g.hca_o_groups * g.hca_o_lora, H, B, p_->stream);

        mhc_post(p_->d_sub_out, res, p_->d_mhc_coef, kHcCoef, B, H, res_next, p_->stream);
        std::swap(res, res_next);

        // ---- Latent MoE sublayer ----
        const DevTensor *fh_fn = p_->req(s, "hc_ffn_fn.weight", err),
                        *fh_sc = p_->req(s, "hc_ffn_scale.weight", err),
                        *fh_bs = p_->req(s, "hc_ffn_base.weight", err),
                        *fnorm = p_->req(s, "ffn_norm.weight", err),
                        *rg = p_->req(s, "ffn_gate_inp.weight", err),
                        *rb = p_->req(s, "exp_probs_b.bias", err),
                        *rs = p_->req(s, "exp_probs_scale.weight", err),
                        *ld = p_->req(s, "ffn_routed_down.weight", err),
                        *ln = p_->req(s, "ffn_latent_norm.weight", err),
                        *lu = p_->req(s, "ffn_routed_up.weight", err),
                        *sg = p_->req(s, "ffn_gate_shexp.weight", err),
                        *su = p_->req(s, "ffn_up_shexp.weight", err),
                        *sd = p_->req(s, "ffn_down_shexp.weight", err);
        if (!fh_fn || !fh_sc || !fh_bs || !fnorm || !rg || !rb || !rs || !ld || !ln || !lu || !sg || !su || !sd)
            return false;

        mhc_mixes(res, (const float*) fh_fn->ptr, (const float*) fh_sc->ptr, (const float*) fh_bs->ptr, B, H,
                  g.rms_eps, g.hc_eps, g.sinkhorn_iters, p_->d_mhc_ws, p_->d_mhc_coef, p_->stream);
        mhc_pre(res, p_->d_mhc_coef, kHcCoef, B, H, p_->d_x_in, p_->stream);
        kernels::rms_norm_rows(p_->d_x_in, (const float*) fnorm->ptr, B, H, g.rms_eps, p_->d_x_norm, p_->stream);

        p_->matmul_f32_on((const float*) rg->ptr, p_->d_x_norm, p_->d_rlogits, H, 2 * g.n_expert, B, p_->cublas);
        split_sigmoid_route(p_->d_rlogits, (const float*) rb->ptr, (const float*) rs->ptr, B, g.n_expert, g.n_used,
                            g.norm_topk_prob, p_->d_exp_ids, p_->d_exp_w, p_->stream);

        // Shared expert
        p_->matmul_q8_on(sg, p_->d_x_norm, p_->d_ff_g, H, g.n_ff_exp, B, p_->stream);
        p_->matmul_q8_on(su, p_->d_x_norm, p_->d_ff_u, H, g.n_ff_exp, B, p_->stream);
        situ_glu_f32(p_->d_ff_g, p_->d_ff_u, B, g.n_ff_exp, g.gate_softcap, g.up_softcap, p_->d_ff_h, p_->stream);
        p_->matmul_q8_on(sd, p_->d_ff_h, p_->d_sub_out, g.n_ff_exp, H, B, p_->stream);

        // Routed MXFP4 experts on shared CPU ExpertPool
        p_->matmul_q8_on(ld, p_->d_x_norm, p_->d_x_lat, H, g.n_latent, B, p_->stream);
        if (!p_->run_stage_experts(s, B, err)) return false;

        weighted_sum_f32(p_->d_exp_out, p_->d_exp_w, B, g.n_used, g.n_latent, p_->d_lat_sum, p_->stream);
        kernels::rms_norm_rows(p_->d_lat_sum, (const float*) ln->ptr, B, g.n_latent, g.rms_eps, p_->d_lat_norm,
                               p_->stream);
        p_->matmul_q8_on(lu, p_->d_lat_norm, p_->d_routed_out, g.n_latent, H, B, p_->stream);
        add_f32(p_->d_sub_out, p_->d_routed_out, (int64_t) B * H, p_->stream);

        mhc_post(p_->d_sub_out, res, p_->d_mhc_coef, kHcCoef, B, H, res_next, p_->stream);
        std::swap(res, res_next);
    }

    // ---- Final contraction + shared target LM head (BEFORE multimax) ----
    const DevTensor *hfn = p_->req("hc_head_fn.weight", err),
                    *hsc = p_->req("hc_head_scale.weight", err),
                    *hbs = p_->req("hc_head_base.weight", err),
                    *onorm = p_->req("output_norm.weight", err),
                    *mw2 = p_->req("markov_w2.weight", err),
                    *conf_w = p_->req("conf_proj.weight", err);
    const DevTensor *head = p_->main->tensor("output.weight"),
                    *mmr = p_->main->tensor("output_multimax_ranges"),
                    *mmt = p_->main->tensor("output_multimax_ts");
    if (!hfn || !hsc || !hbs || !onorm || !mw2 || !conf_w) return false;
    if (!head || !mmr || !mmt) {
        err = "ernie55 dspark: main model missing output.weight / multimax tensors";
        return false;
    }

    mhc_contract(res, (const float*) hfn->ptr, (const float*) hsc->ptr, (const float*) hbs->ptr, n_gen, H, g.rms_eps,
                 g.hc_eps, p_->d_mhc_ws, p_->d_mhc_coef, p_->d_hcout, p_->stream);
    kernels::rms_norm_rows(p_->d_hcout, (const float*) onorm->ptr, n_gen, H, g.rms_eps, p_->d_hnorm, p_->stream);

    if (head->type == kTypeQ8_0) {
        p_->matmul_q8_on(head, p_->d_hnorm, p_->d_base_logits, H, g.n_vocab, n_gen, p_->stream);
    } else {
        kernels::bf16_gemv_fp32_mmvf_cols(p_->d_hnorm, (const uint16_t*) head->ptr, p_->d_base_logits, H, g.n_vocab,
                                          n_gen, p_->stream);
    }

    // ---- Autoregressive Markov chain + multimax + confidence ----
    const int rank = p_->dp.markov_rank;
    int32_t prev = anchor;
    for (int i = 0; i < n_gen; ++i) {
        if (prev < 0 || prev >= g.n_vocab) {
            err = "ernie55 dspark: markov prev token out of range";
            return false;
        }
        const uint16_t* w1_row = p_->markov_w1.data() + (size_t) prev * (size_t) rank;
        for (int j = 0; j < rank; ++j) p_->h_me[j] = bf16_to_f32(w1_row[j]);
        cudaMemcpyAsync(p_->d_me, p_->h_me, (size_t) rank * sizeof(float), cudaMemcpyHostToDevice, p_->stream);

        p_->matmul_q8_on(mw2, p_->d_me, p_->d_mbias, rank, g.n_vocab, 1, p_->stream);
        dspark_markov_multimax_argmax(p_->d_base_logits + (size_t) i * g.n_vocab, p_->d_mbias,
                                      (const float*) mmr->ptr, (const float*) mmt->ptr, g.n_vocab, p_->d_amax_v,
                                      p_->d_amax_i, p_->d_tok, p_->stream);
        dspark_confidence((const float*) conf_w->ptr, p_->d_hcout + (size_t) i * H, p_->d_me, H, rank, p_->d_conf,
                          p_->stream);
        cudaMemcpyAsync(p_->h_tok, p_->d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost, p_->stream);
        cudaMemcpyAsync(p_->h_conf, p_->d_conf, sizeof(float), cudaMemcpyDeviceToHost, p_->stream);
        if (!ck(cudaStreamSynchronize(p_->stream), "markov step sync", err)) return false;

        out.token[i] = *p_->h_tok;
        out.conf[i] = *p_->h_conf;
        prev = out.token[i];
    }
    out.n = n_gen;
    return true;
}

}  // namespace basalt::models::ernie55
