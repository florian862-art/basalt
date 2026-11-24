// src/models/ernie55/kernels/kernels_parity.cpp - GPU parity tests for all ERNIE-5.5-Flash CUDA kernels
// against FP64 CPU references:
//   1. mHC mixes (1 / (sqrt(mean) + rms_eps)), pre, post (destination-first comb[d,s]), contract
//   2. FP32 LM-head multimax
//   3. SiTU-GLU (beta=4.0, linear_beta=25.0)
//   4. Split-feature sigmoid MoE router (512 experts, top-10, bias, renorm, per-expert scale)
//   5. VHA PostMix, sigmoid output gate, LayerNorm, orthonormal Walsh-Hadamard-128
//   6. HCA attention pipeline: q_rope, kv_store_window, 128:1 streaming compressor w/ additive APE,
//      multi-query attention w/ attn_sink, inv_rope
//   7. MLA-DSA + Lightning Indexer pipeline: dsa_qkv_rope_and_store, dsa_indexer_prepare,
//      dsa_indexer_topk_merge, dsa_mla_attention w/ Q8_0 kv_b and attn_sink
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace basalt::models::ernie55;

#define CUDA_CHECK(expr)                                                                                 \
    do {                                                                                                 \
        cudaError_t _e = (expr);                                                                         \
        if (_e != cudaSuccess) {                                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            std::exit(2);                                                                                \
        }                                                                                                \
    } while (0)

template <typename T>
struct DevBuf {
    T* ptr = nullptr;
    size_t count = 0;
    explicit DevBuf(size_t n = 0) : count(n) {
        if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DevBuf() {
        if (ptr) cudaFree(ptr);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void upload(const T* src, size_t n) { CUDA_CHECK(cudaMemcpy(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice)); }
    void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }
    void download(std::vector<T>& dst) const {
        CUDA_CHECK(cudaMemcpy(dst.data(), ptr, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (count) CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }
};

std::vector<float> rand_vec(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (float& v : out) v = dist(rng);
    return out;
}

double max_abs_err(const std::vector<float>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double) a[i] - b[i]));
    return m;
}

double rel_l2_err(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double) a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

int g_failures = 0;

void check_tol(const char* name, double err, double tol) {
    if (err <= tol && std::isfinite(err)) {
        std::printf("[PASS] %-36s err=%.3e (tol=%.1e)\n", name, err, tol);
    } else {
        std::printf("[FAIL] %-36s err=%.3e (tol=%.1e)\n", name, err, tol);
        ++g_failures;
    }
}

void quantize_q8_0(const std::vector<float>& src, int rows, int cols, std::vector<uint8_t>& dst) {
    const int nb = cols / 32;
    dst.resize((size_t) rows * nb * 34);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* x = src.data() + (size_t) r * cols + b * 32;
            float am = 0.0f;
            for (int k = 0; k < 32; ++k) am = std::max(am, std::abs(x[k]));
            const float d = am / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            uint8_t* out = dst.data() + ((size_t) r * nb + b) * 34;
            const __half dh = __float2half_rn(d);
            std::memcpy(out, &dh, 2);
            int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
            for (int k = 0; k < 32; ++k) {
                int q = (int) std::lrint(x[k] * inv);
                qs[k] = (int8_t) std::max(-127, std::min(127, q));
            }
        }
    }
}

std::vector<double> dequantize_q8_0(const std::vector<uint8_t>& src, int rows, int cols) {
    const int nb = cols / 32;
    std::vector<double> out((size_t) rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const uint8_t* in = src.data() + ((size_t) r * nb + b) * 34;
            __half dh;
            std::memcpy(&dh, in, 2);
            const double d = __half2float(dh);
            const int8_t* qs = reinterpret_cast<const int8_t*>(in + 2);
            for (int k = 0; k < 32; ++k) out[(size_t) r * cols + b * 32 + k] = d * (double) qs[k];
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// 1. mHC mixes, pre, post, contract
// -----------------------------------------------------------------------------
void test_mhc() {
    const int T = 3, H = 256, flat = kHcMult * H;
    // Large rms_eps intentionally distinguishes ERNIE 1/(sqrt(mean)+eps) from rsqrt(mean+eps)
    const float rms_eps = 1e-2f, hc_eps = 1e-4f;
    const int sink_iters = 20;

    auto x = rand_vec((size_t) T * flat, -1.5f, 1.5f, 101);
    auto fn = rand_vec((size_t) kHcCoef * flat, -0.1f, 0.1f, 102);
    auto sc = std::vector<float>{0.15f, 0.25f, 0.35f};
    auto bs = rand_vec(kHcCoef, -0.5f, 0.5f, 103);
    auto f_in = rand_vec((size_t) T * H, -1.0f, 1.0f, 104);

    // Host FP64 reference
    std::vector<double> ref_coef((size_t) T * kHcCoef, 0.0);
    std::vector<double> ref_pre((size_t) T * H, 0.0);
    std::vector<double> ref_post((size_t) T * flat, 0.0);
    for (int t = 0; t < T; ++t) {
        double ss = 0.0;
        for (int i = 0; i < flat; ++i) ss += (double) x[t * flat + i] * x[t * flat + i];
        const double inv_rms = 1.0 / (std::sqrt(ss / flat) + rms_eps);
        double mix[kHcCoef];
        for (int m = 0; m < kHcCoef; ++m) {
            double dot = 0.0;
            for (int i = 0; i < flat; ++i) dot += (double) x[t * flat + i] * fn[m * flat + i];
            mix[m] = dot * inv_rms;
        }
        auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
        for (int k = 0; k < 4; ++k) ref_coef[t * kHcCoef + k] = sig(mix[k] * sc[0] + bs[k]) + hc_eps;
        for (int k = 0; k < 4; ++k) ref_coef[t * kHcCoef + 4 + k] = sig(mix[4 + k] * sc[1] + bs[4 + k]) * 2.0;

        double cm[4][4];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) cm[r][c] = mix[8 + r * 4 + c] * sc[2] + bs[8 + r * 4 + c];
        for (int r = 0; r < 4; ++r) {
            double mx = std::max({cm[r][0], cm[r][1], cm[r][2], cm[r][3]});
            double sum = 0.0;
            for (int c = 0; c < 4; ++c) { cm[r][c] = std::exp(cm[r][c] - mx); sum += cm[r][c]; }
            for (int c = 0; c < 4; ++c) cm[r][c] = cm[r][c] / sum + hc_eps;
        }
        for (int c = 0; c < 4; ++c) {
            double cs = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + hc_eps;
            for (int r = 0; r < 4; ++r) cm[r][c] /= cs;
        }
        for (int it = 1; it < sink_iters; ++it) {
            for (int r = 0; r < 4; ++r) {
                double rs = cm[r][0] + cm[r][1] + cm[r][2] + cm[r][3] + hc_eps;
                for (int c = 0; c < 4; ++c) cm[r][c] /= rs;
            }
            for (int c = 0; c < 4; ++c) {
                double cs = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + hc_eps;
                for (int r = 0; r < 4; ++r) cm[r][c] /= cs;
            }
        }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) ref_coef[t * kHcCoef + 8 + r * 4 + c] = cm[r][c];

        for (int i = 0; i < H; ++i) {
            double acc = 0.0;
            for (int s = 0; s < 4; ++s) acc += ref_coef[t * kHcCoef + s] * x[(t * 4 + s) * H + i];
            ref_pre[t * H + i] = acc;
        }
        for (int d = 0; d < 4; ++d) {
            for (int i = 0; i < H; ++i) {
                double acc = ref_coef[t * kHcCoef + 4 + d] * f_in[t * H + i];
                for (int s = 0; s < 4; ++s) acc += cm[s][d] * x[(t * 4 + s) * H + i];
                ref_post[(t * 4 + d) * H + i] = acc;
            }
        }
    }

    DevBuf<float> dx(x.size()), dfn(fn.size()), dsc(sc.size()), dbs(bs.size()), df(f_in.size());
    DevBuf<float> dws(mhc_workspace_bytes(T) / sizeof(float));
    DevBuf<float> dcoef((size_t) T * kHcCoef), dpre((size_t) T * H), dpost((size_t) T * flat);
    dx.upload(x); dfn.upload(fn); dsc.upload(sc); dbs.upload(bs); df.upload(f_in);

    mhc_mixes(dx.ptr, dfn.ptr, dsc.ptr, dbs.ptr, T, H, rms_eps, hc_eps, sink_iters, dws.ptr, dcoef.ptr, nullptr);
    mhc_pre(dx.ptr, dcoef.ptr, kHcCoef, T, H, dpre.ptr, nullptr);
    mhc_post(df.ptr, dx.ptr, dcoef.ptr, kHcCoef, T, H, dpost.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hcoef(ref_coef.size()), hpre(ref_pre.size()), hpost(ref_post.size());
    dcoef.download(hcoef); dpre.download(hpre); dpost.download(hpost);
    check_tol("mhc_mixes (ERNIE rms_eps outside)", max_abs_err(hcoef, ref_coef), 1e-5);
    check_tol("mhc_pre", max_abs_err(hpre, ref_pre), 1e-5);
    check_tol("mhc_post (comb[d,s])", max_abs_err(hpost, ref_post), 1e-5);

    // Final mhc_contract
    auto hfn = rand_vec((size_t) 4 * flat, -0.1f, 0.1f, 105);
    auto hsc = std::vector<float>{0.2f};
    auto hbs = rand_vec(4, -0.4f, 0.4f, 106);
    std::vector<double> ref_con((size_t) T * H, 0.0);
    for (int t = 0; t < T; ++t) {
        double ss = 0.0;
        for (int i = 0; i < flat; ++i) ss += (double) x[t * flat + i] * x[t * flat + i];
        const double inv_rms = 1.0 / (std::sqrt(ss / flat) + rms_eps);
        double w4[4];
        for (int k = 0; k < 4; ++k) {
            double dot = 0.0;
            for (int i = 0; i < flat; ++i) dot += (double) x[t * flat + i] * hfn[k * flat + i];
            w4[k] = 1.0 / (1.0 + std::exp(-((dot * inv_rms) * hsc[0] + hbs[k]))) + hc_eps;
        }
        for (int i = 0; i < H; ++i) {
            double acc = 0.0;
            for (int s = 0; s < 4; ++s) acc += w4[s] * x[(t * 4 + s) * H + i];
            ref_con[t * H + i] = acc;
        }
    }
    DevBuf<float> dhfn(hfn.size()), dhsc(1), dhbs(4), dcoef4((size_t) T * 4), dcon((size_t) T * H);
    dhfn.upload(hfn); dhsc.upload(hsc); dhbs.upload(hbs);
    mhc_contract(dx.ptr, dhfn.ptr, dhsc.ptr, dhbs.ptr, T, H, rms_eps, hc_eps, dws.ptr, dcoef4.ptr, dcon.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hcon(ref_con.size());
    dcon.download(hcon);
    check_tol("mhc_contract", max_abs_err(hcon, ref_con), 1e-5);
}

// -----------------------------------------------------------------------------
// 2. Multimax & SiTU-GLU & Split-Sigmoid Router & VHA / Gate / LN / Hadamard
// -----------------------------------------------------------------------------
void test_elementwise_and_router() {
    // Multimax
    const int T = 2, V = 4096;
    auto logits = rand_vec((size_t) T * V, -25.0f, 25.0f, 201);
    std::vector<float> ranges = {-10.0f, 12.0f, -15.0f, 18.0f};
    std::vector<float> ts = {0.25f, -0.3f, 0.05f, -0.02f};
    std::vector<double> ref_mm(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        double x = logits[i];
        double a = std::max(0.0, (double) ranges[0] - x);
        double b = std::max(0.0, x - (double) ranges[1]);
        double c = std::max(0.0, (double) ranges[2] - x);
        double d = std::max(0.0, x - (double) ranges[3]);
        ref_mm[i] = x + ts[0] * a + ts[1] * b + ts[2] * c * c + ts[3] * d * d;
    }
    DevBuf<float> dlog(logits.size()), dr(4), dts(4);
    dlog.upload(logits); dr.upload(ranges); dts.upload(ts);
    multimax_f32(dlog.ptr, dr.ptr, dts.ptr, T, V, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hlog(logits.size());
    dlog.download(hlog);
    check_tol("multimax_f32", max_abs_err(hlog, ref_mm), 1e-4);

    // SiTU-GLU (beta=4.0, linear_beta=25.0)
    const int FF = 2048;
    auto gate = rand_vec((size_t) T * FF, -10.0f, 10.0f, 202);
    auto up = rand_vec((size_t) T * FF, -40.0f, 40.0f, 203);
    std::vector<double> ref_situ(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        double g = gate[i], u = up[i];
        double sg = 4.0 * std::tanh(g / 4.0) * (1.0 / (1.0 + std::exp(-g)));
        double su = 25.0 * std::tanh(u / 25.0);
        ref_situ[i] = sg * su;
    }
    DevBuf<float> dg(gate.size()), du(up.size()), do_situ(gate.size());
    dg.upload(gate); du.upload(up);
    situ_glu_f32(dg.ptr, du.ptr, T, FF, 4.0f, 25.0f, do_situ.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hsitu(gate.size());
    do_situ.download(hsitu);
    check_tol("situ_glu_f32 (4.0 / 25.0)", max_abs_err(hsitu, ref_situ), 1e-4);

    // Split-sigmoid MoE router (512 experts, top-10)
    const int E = 512, K = 10;
    auto rlog = rand_vec((size_t) T * 2 * E, -3.0f, 3.0f, 204);
    auto rbias = rand_vec(E, -0.5f, 0.5f, 205);
    auto rscale = rand_vec(E, 0.5f, 2.5f, 206);
    std::vector<int32_t> ref_ids((size_t) T * K);
    std::vector<double> ref_w((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<double> sc(E), sel(E);
        for (int e = 0; e < E; ++e) {
            double l0 = rlog[t * 2 * E + e], l1 = rlog[t * 2 * E + E + e];
            sc[e] = 1.0 / (1.0 + std::exp(-l0)) + 1.0 / (1.0 + std::exp(-l1));
            sel[e] = sc[e] + rbias[e];
        }
        std::vector<int> idx(E);
        for (int e = 0; e < E; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return sel[a] > sel[b]; });
        double tot = 0.0;
        for (int k = 0; k < K; ++k) { ref_ids[t * K + k] = idx[k]; tot += sc[idx[k]]; }
        for (int k = 0; k < K; ++k) ref_w[t * K + k] = (sc[idx[k]] / tot) * rscale[idx[k]];
    }
    DevBuf<float> drl(rlog.size()), drb(E), drs(E), drw((size_t) T * K);
    DevBuf<int32_t> dri((size_t) T * K);
    drl.upload(rlog); drb.upload(rbias); drs.upload(rscale);
    split_sigmoid_route(drl.ptr, drb.ptr, drs.ptr, T, E, K, true, dri.ptr, drw.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> hri(ref_ids.size());
    std::vector<float> hrw(ref_w.size());
    dri.download(hri); drw.download(hrw);
    double id_mismatch = 0.0;
    for (size_t i = 0; i < ref_ids.size(); ++i) id_mismatch += (hri[i] != ref_ids[i]) ? 1.0 : 0.0;
    check_tol("split_sigmoid_route (ids)", id_mismatch, 0.0);
    check_tol("split_sigmoid_route (weights)", max_abs_err(hrw, ref_w), 1e-5);

    // VHA PostMix + Sigmoid gate + LayerNorm + Hadamard-128
    const int NH = 16, HD = 64;
    auto vin = rand_vec((size_t) T * NH * HD, -1.0f, 1.0f, 207);
    auto vmix = rand_vec((size_t) NH * NH, -0.5f, 0.5f, 208);
    auto gvec = rand_vec((size_t) T * NH * HD, -2.0f, 2.0f, 209);
    std::vector<double> ref_vha(vin.size(), 0.0);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h)
            for (int d = 0; d < HD; ++d) {
                double acc = 0.0;
                for (int g = 0; g < NH; ++g) acc += (double) vmix[h * NH + g] * vin[(t * NH + g) * HD + d];
                ref_vha[(t * NH + h) * HD + d] = acc * (1.0 / (1.0 + std::exp(-(double) gvec[(t * NH + h) * HD + d])));
            }
    DevBuf<float> dvin(vin.size()), dvmix(vmix.size()), dvout(vin.size()), dgvec(gvec.size());
    dvin.upload(vin); dvmix.upload(vmix); dgvec.upload(gvec);
    vha_postmix_f32(dvin.ptr, dvmix.ptr, T, NH, HD, dvout.ptr, nullptr);
    sigmoid_gate_mul_f32(dvout.ptr, dgvec.ptr, (int64_t) vin.size(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hvout(vin.size());
    dvout.download(hvout);
    check_tol("vha_postmix + sigmoid_gate_mul", max_abs_err(hvout, ref_vha), 1e-5);

    // LayerNorm + Hadamard-128 (orthonormal involution: H(H(x)) == x)
    const int D = 128;
    auto lnx = rand_vec((size_t) T * D, -2.0f, 3.0f, 210);
    auto lnw = rand_vec(D, 0.5f, 1.5f, 211);
    auto lnb = rand_vec(D, -0.5f, 0.5f, 212);
    std::vector<double> ref_ln(lnx.size());
    for (int t = 0; t < T; ++t) {
        double mean = 0.0;
        for (int i = 0; i < D; ++i) mean += lnx[t * D + i];
        mean /= D;
        double var = 0.0;
        for (int i = 0; i < D; ++i) { double d = lnx[t * D + i] - mean; var += d * d; }
        double inv = 1.0 / std::sqrt(var / D + 1e-6);
        for (int i = 0; i < D; ++i) ref_ln[t * D + i] = (lnx[t * D + i] - mean) * inv * lnw[i] + lnb[i];
    }
    DevBuf<float> dlnx(lnx.size()), dlnw(D), dlnb(D), dlno(lnx.size());
    dlnx.upload(lnx); dlnw.upload(lnw); dlnb.upload(lnb);
    layernorm_f32(dlnx.ptr, dlnw.ptr, dlnb.ptr, T, D, 1e-6f, dlno.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hlno(lnx.size());
    dlno.download(hlno);
    check_tol("layernorm_f32", max_abs_err(hlno, ref_ln), 1e-5);

    // Hadamard-128 exact Sylvester check + H(H(x)) == x
    std::vector<double> ref_had(lnx.size());
    const double inv_sqrt128 = 1.0 / std::sqrt(128.0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> v(D);
        for (int i = 0; i < D; ++i) v[i] = lnx[t * D + i];
        for (int len = 1; len < D; len <<= 1) {
            for (int i = 0; i < D; i += 2 * len) {
                for (int j = 0; j < len; ++j) {
                    double a = v[i + j], b = v[i + j + len];
                    v[i + j] = a + b;
                    v[i + j + len] = a - b;
                }
            }
        }
        for (int i = 0; i < D; ++i) ref_had[t * D + i] = v[i] * inv_sqrt128;
    }
    hadamard_128_f32(dlnx.ptr, T, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hhad(lnx.size());
    dlnx.download(hhad);
    check_tol("hadamard_128_f32 (Sylvester)", max_abs_err(hhad, ref_had), 1e-5);
    hadamard_128_f32(dlnx.ptr, T, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    dlnx.download(hhad);
    std::vector<double> lnx_d(lnx.begin(), lnx.end());
    check_tol("hadamard_128_f32 (involution H^2=I)", max_abs_err(hhad, lnx_d), 1e-5);
}

// -----------------------------------------------------------------------------
// 3. HCA Attention Pipeline (128:1 streaming compress w/ APE + window + sink + inv_rope)
// -----------------------------------------------------------------------------
void test_hca() {
    const int N = 160;   // crosses 128-token boundary -> 1 closed compressed block + 128 window
    const int ratio = 128, win = 128, HD = 512, NH = 4;
    const float rms_eps = 1e-6f;
    const RopeTable rt = rope_table_plain(100000.0f);
    const RopeTable comp_rt = rope_table_yarn(160000.0f, 8.0f, 32.0f, 1.0f, 131072);

    auto q_all = rand_vec((size_t) N * NH * HD, -0.5f, 0.5f, 301);
    auto kv_in = rand_vec((size_t) N * HD, -0.5f, 0.5f, 302);
    auto kv_nw = rand_vec(HD, 0.8f, 1.2f, 303);
    auto ckv = rand_vec((size_t) N * HD, -0.5f, 0.5f, 304);
    auto csc = rand_vec((size_t) N * HD, -1.0f, 1.0f, 305);
    auto ape = rand_vec((size_t) ratio * HD, -0.5f, 0.5f, 306);
    auto comp_nw = rand_vec(HD, 0.8f, 1.2f, 307);
    auto sink = rand_vec(NH, -1.0f, 1.0f, 308);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;

    auto rot_pair = [](double& x0, double& x1, double p, float step, float mscale, bool inv) {
        double ang = p * (double) step;
        double c = std::cos(ang) * mscale, s = std::sin(ang) * mscale;
        if (inv) s = -s;
        double a = x0 * c - x1 * s, b = x0 * s + x1 * c;
        x0 = a; x1 = b;
    };

    // Host FP64 reference for the last token (t = N - 1 = 159: n_comp = 1, win = [32..159])
    std::vector<std::vector<double>> win_ref(win, std::vector<double>(HD));
    for (int t = 0; t < N; ++t) {
        double ss = 0.0;
        for (int c = 0; c < HD; ++c) ss += (double) kv_in[t * HD + c] * kv_in[t * HD + c];
        double inv = 1.0 / std::sqrt(ss / HD + rms_eps);
        std::vector<double> row(HD);
        for (int c = 0; c < HD; ++c) row[c] = kv_in[t * HD + c] * inv * kv_nw[c];
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(row[HD - 64 + 2 * pr], row[HD - 64 + 2 * pr + 1], (double) t, rt.step[pr], rt.mscale, false);
        win_ref[t % win] = row;
    }
    // Compressed block 0: positions 0..127
    std::vector<double> comp0(HD);
    for (int c = 0; c < HD; ++c) {
        double mx = -1e30;
        for (int j = 0; j < ratio; ++j) mx = std::max(mx, (double) csc[j * HD + c] + ape[j * HD + c]);
        double num = 0.0, den = 0.0;
        for (int j = 0; j < ratio; ++j) {
            double e = std::exp((double) csc[j * HD + c] + ape[j * HD + c] - mx);
            den += e;
            num += (double) ckv[j * HD + c] * e;
        }
        comp0[c] = num / den;
    }
    {
        double ss = 0.0;
        for (int c = 0; c < HD; ++c) ss += comp0[c] * comp0[c];
        double inv = 1.0 / std::sqrt(ss / HD + rms_eps);
        for (int c = 0; c < HD; ++c) comp0[c] = comp0[c] * inv * comp_nw[c];
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(comp0[HD - 64 + 2 * pr], comp0[HD - 64 + 2 * pr + 1], 0.0, comp_rt.step[pr], comp_rt.mscale, false);
    }

    const int t_last = N - 1;
    std::vector<double> ref_out((size_t) NH * HD);
    for (int h = 0; h < NH; ++h) {
        std::vector<double> qh(HD);
        for (int c = 0; c < HD; ++c) qh[c] = q_all[(t_last * NH + h) * HD + c];
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(qh[HD - 64 + 2 * pr], qh[HD - 64 + 2 * pr + 1], (double) t_last, rt.step[pr], rt.mscale, false);
        double scale = 1.0 / std::sqrt((double) HD);
        std::vector<double> scores;
        std::vector<const std::vector<double>*> rows;
        auto add_row = [&](const std::vector<double>& r) {
            double dot = 0.0;
            for (int c = 0; c < HD; ++c) dot += qh[c] * r[c];
            scores.push_back(dot * scale);
            rows.push_back(&r);
        };
        add_row(comp0);
        for (int j = t_last - win + 1; j <= t_last; ++j) add_row(win_ref[j % win]);
        double mx = sink[h];
        for (double s : scores) mx = std::max(mx, s);
        double l = std::exp((double) sink[h] - mx);
        std::vector<double> acc(HD, 0.0);
        for (size_t i = 0; i < scores.size(); ++i) {
            double w = std::exp(scores[i] - mx);
            l += w;
            for (int c = 0; c < HD; ++c) acc[c] += w * (*rows[i])[c];
        }
        for (int c = 0; c < HD; ++c) acc[c] /= l;
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(acc[HD - 64 + 2 * pr], acc[HD - 64 + 2 * pr + 1], (double) t_last, rt.step[pr], rt.mscale, true);
        for (int c = 0; c < HD; ++c) ref_out[h * HD + c] = acc[c];
    }

    // Run GPU HCA pipeline (FP16 cache for tight parity check)
    DevBuf<float> dq(q_all.size()), dkv(kv_in.size()), dkvnw(HD), dckv(ckv.size()), dcsc(csc.size());
    DevBuf<float> dape(ape.size()), dcnw(HD), dsink(NH), dst_kv((size_t) ratio * HD), dst_sc((size_t) ratio * HD);
    DevBuf<float> dout((size_t) NH * HD);
    DevBuf<int32_t> dpos(N), dpos_last(1);
    DevBuf<uint8_t> dwin((size_t) win * HD * 2), dcomp((size_t) 4 * HD * 2);
    dq.upload(q_all); dkv.upload(kv_in); dkvnw.upload(kv_nw); dckv.upload(ckv); dcsc.upload(csc);
    dape.upload(ape); dcnw.upload(comp_nw); dsink.upload(sink); dpos.upload(pos);
    int32_t plast = t_last;
    dpos_last.upload(&plast, 1);
    dst_kv.zero(); dst_sc.zero(); dwin.zero(); dcomp.zero();

    hca_q_rope(dq.ptr, dpos.ptr, N, NH, HD, rt, nullptr);
    hca_kv_store_window(dkv.ptr, dkvnw.ptr, dpos.ptr, N, HD, win, rms_eps, rt, dwin.ptr, false, nullptr);
    hca_compress(dckv.ptr, dcsc.ptr, dape.ptr, dcnw.ptr, dpos.ptr, N, ratio, HD, rms_eps, comp_rt,
                 dst_kv.ptr, dst_sc.ptr, dcomp.ptr, false, nullptr);
    hca_attention(dq.ptr + (size_t) t_last * NH * HD, dcomp.ptr, dwin.ptr, nullptr, dsink.ptr, dpos_last.ptr,
                  1, NH, HD, win, ratio, false, dout.ptr, nullptr);
    hca_inv_rope(dout.ptr, dpos_last.ptr, 1, NH, HD, rt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hout(ref_out.size());
    dout.download(hout);
    check_tol("HCA pipeline (compress+APE+attn+inv_rope)", rel_l2_err(hout, ref_out), 2e-3);
}

// -----------------------------------------------------------------------------
// 4. MLA-DSA + Lightning Indexer Pipeline
// -----------------------------------------------------------------------------
void test_mla_dsa() {
    const int N = 40;
    const int win = 8, top_k = 6, width = top_k + win;
    const int idx_h = 8, idx_d = 128, rope_d = 64;
    const int NH = 4, qk_nope = 64, qk_rope = 64, kv_lora = 128, v_dim = 64;
    const float eps = 1e-6f;
    const RopeTable rt = rope_table_plain(100000.0f);

    // Indexer prepare + topk_merge (FP16 cache)
    auto iq = rand_vec((size_t) N * idx_h * idx_d, -0.5f, 0.5f, 401);
    auto ik = rand_vec((size_t) N * idx_d, -0.5f, 0.5f, 402);
    auto knw = rand_vec(idx_d, 0.8f, 1.2f, 403);
    auto knb = rand_vec(idx_d, -0.2f, 0.2f, 404);
    auto iw = rand_vec((size_t) N * idx_h, 0.1f, 1.0f, 405);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;

    DevBuf<float> diq(iq.size()), dik(ik.size()), dknw(idx_d), dknb(idx_d), diw(iw.size());
    DevBuf<int32_t> dpos(N), dsel((size_t) N * width);
    DevBuf<uint8_t> didx_k((size_t) N * idx_d * 2);
    DevBuf<float> dscores((size_t) N * N);
    diq.upload(iq); dik.upload(ik); dknw.upload(knw); dknb.upload(knb); diw.upload(iw); dpos.upload(pos);
    dsa_indexer_prepare(diq.ptr, dik.ptr, dknw.ptr, dknb.ptr, dpos.ptr, N, idx_h, idx_d, rope_d, eps, rt,
                        didx_k.ptr, false, nullptr);
    dsa_indexer_topk_merge(diq.ptr, diw.ptr, didx_k.ptr, dpos.ptr, N, idx_h, idx_d, top_k, win, false,
                           dsel.ptr, nullptr, dscores.ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download transformed iq, ik and verify top-K for last token t = N - 1 against CPU scoring
    std::vector<float> hiq(iq.size()), hik(ik.size());
    std::vector<int32_t> hsel(N * width);
    diq.download(hiq); dik.download(hik); dsel.download(hsel);

    const int t = N - 1;
    const int older_n = N - win;
    const double wscale = 1.0 / std::sqrt((double) (idx_d * idx_h));
    std::vector<std::pair<double, int>> scored;
    for (int j = 0; j < older_n; ++j) {
        double tot = 0.0;
        for (int h = 0; h < idx_h; ++h) {
            double dot = 0.0;
            for (int d = 0; d < idx_d; ++d) dot += (double) hiq[(t * idx_h + h) * idx_d + d] * hik[j * idx_d + d];
            if (dot > 0.0) tot += dot * iw[t * idx_h + h] * wscale;
        }
        scored.push_back({tot, j});
    }
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<int> gpu_older(hsel.begin() + t * width, hsel.begin() + t * width + top_k);
    std::vector<int> ref_older;
    for (int k = 0; k < top_k; ++k) ref_older.push_back(scored[k].second);
    std::sort(gpu_older.begin(), gpu_older.end());
    std::sort(ref_older.begin(), ref_older.end());
    double topk_err = 0.0;
    for (int k = 0; k < top_k; ++k) topk_err += (gpu_older[k] != ref_older[k]) ? 1.0 : 0.0;
    // Also verify trailing sliding window [N - win .. N - 1]
    for (int k = 0; k < win; ++k) topk_err += (hsel[t * width + top_k + k] != (N - win + k)) ? 1.0 : 0.0;
    check_tol("dsa_indexer_prepare + topk_merge", topk_err, 0.0);

    // Absorbed sparse MLA attention parity (FP16 KV cache + Q8_0 kv_b)
    auto q_mla = rand_vec((size_t) N * NH * (qk_nope + qk_rope), -0.5f, 0.5f, 406);
    auto kv_a = rand_vec((size_t) N * (kv_lora + qk_rope), -0.5f, 0.5f, 407);
    auto kv_nw = rand_vec(kv_lora, 0.8f, 1.2f, 408);
    auto kv_b_f = rand_vec((size_t) NH * (qk_nope + v_dim) * kv_lora, -0.3f, 0.3f, 409);
    auto sink = rand_vec(NH, -1.0f, 1.0f, 410);
    std::vector<uint8_t> kv_b_q8;
    quantize_q8_0(kv_b_f, NH * (qk_nope + v_dim), kv_lora, kv_b_q8);
    auto kv_b_deq = dequantize_q8_0(kv_b_q8, NH * (qk_nope + v_dim), kv_lora);

    DevBuf<float> dqm(q_mla.size()), dkva(kv_a.size()), dkvnw(kv_lora), dsink(NH), dmo((size_t) NH * v_dim);
    DevBuf<uint8_t> dkv_cache((size_t) N * (kv_lora + qk_rope) * 2), dkvb(kv_b_q8.size());
    dqm.upload(q_mla); dkva.upload(kv_a); dkvnw.upload(kv_nw); dsink.upload(sink); dkvb.upload(kv_b_q8);

    dsa_qkv_rope_and_store(dqm.ptr, dkva.ptr, dkvnw.ptr, dpos.ptr, N, NH, qk_nope, qk_rope, kv_lora, eps, rt,
                           dkv_cache.ptr, false, nullptr);
    dsa_mla_attention(dqm.ptr + (size_t) t * NH * (qk_nope + qk_rope), dkv_cache.ptr, dkvb.ptr, dsink.ptr,
                      dsel.ptr + (size_t) t * width, 1, NH, qk_nope, qk_rope, kv_lora, v_dim, width, false,
                      dmo.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download post-RoPE q and FP16 kv_cache to compare attention math directly
    std::vector<float> hqm(q_mla.size()), hmo((size_t) NH * v_dim);
    std::vector<uint8_t> hkvc(dkv_cache.count);
    dqm.download(hqm); dmo.download(hmo); dkv_cache.download(hkvc);
    const __half* hkv_h = reinterpret_cast<const __half*>(hkvc.data());

    std::vector<double> ref_mla((size_t) NH * v_dim, 0.0);
    const int qd = qk_nope + qk_rope;
    const double scale = 1.0 / std::sqrt((double) qd);
    for (int h = 0; h < NH; ++h) {
        const float* qr = hqm.data() + (t * NH + h) * qd;
        const double* w_uk = kv_b_deq.data() + (size_t) h * (qk_nope + v_dim) * kv_lora;
        const double* w_uv = w_uk + (size_t) qk_nope * kv_lora;
        std::vector<double> q_lat(kv_lora, 0.0);
        for (int c = 0; c < kv_lora; ++c)
            for (int r = 0; r < qk_nope; ++r) q_lat[c] += (double) qr[r] * w_uk[r * kv_lora + c];
        std::vector<double> scs;
        std::vector<int> cells;
        for (int s_i = 0; s_i < width; ++s_i) {
            int cell = hsel[t * width + s_i];
            if (cell < 0) break;
            const __half* row = hkv_h + (size_t) cell * (kv_lora + qk_rope);
            double dot = 0.0;
            for (int c = 0; c < kv_lora; ++c) dot += q_lat[c] * __half2float(row[c]);
            for (int c = 0; c < qk_rope; ++c) dot += (double) qr[qk_nope + c] * __half2float(row[kv_lora + c]);
            scs.push_back(dot * scale);
            cells.push_back(cell);
        }
        double mx = sink[h];
        for (double s : scs) mx = std::max(mx, s);
        double l = std::exp((double) sink[h] - mx);
        std::vector<double> o_lat(kv_lora, 0.0);
        for (size_t i = 0; i < scs.size(); ++i) {
            double w = std::exp(scs[i] - mx);
            l += w;
            const __half* row = hkv_h + (size_t) cells[i] * (kv_lora + qk_rope);
            for (int c = 0; c < kv_lora; ++c) o_lat[c] += w * __half2float(row[c]);
        }
        for (int c = 0; c < kv_lora; ++c) o_lat[c] /= l;
        for (int r = 0; r < v_dim; ++r) {
            double acc = 0.0;
            for (int c = 0; c < kv_lora; ++c) acc += w_uv[r * kv_lora + c] * o_lat[c];
            ref_mla[h * v_dim + r] = acc;
        }
    }
    check_tol("dsa_mla_attention (absorbed Q8_0 + sink)", rel_l2_err(hmo, ref_mla), 5e-4);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 77;
    }
    test_mhc();
    test_elementwise_and_router();
    test_hca();
    test_mla_dsa();
    std::printf("ernie55_kernels_parity: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
