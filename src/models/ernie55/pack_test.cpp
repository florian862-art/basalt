// src/models/ernie55/pack_test.cpp - unit test for ERNIE-5.5 context policy and pack/meta validation (CPU-only).
#include "models/ernie55/context_policy.hpp"
#include "models/ernie55/pack.hpp"

#include <cstdio>

namespace {

int g_fail = 0;

void expect(bool cond, const char* msg) {
    if (!cond) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", msg);
    }
}

}  // namespace

int main() {
    using namespace basalt::models::ernie55;

    ctxp::CacheFormat fmt = ctxp::CacheFormat::Fp16;
    expect(ctxp::cache_format("", fmt) && fmt == ctxp::CacheFormat::Fp8, "default kv is fp8");
    expect(ctxp::cache_format("fp8", fmt) && fmt == ctxp::CacheFormat::Fp8, "fp8 parses");
    expect(ctxp::cache_format("fp16", fmt) && fmt == ctxp::CacheFormat::Fp16, "fp16 parses");
    expect(!ctxp::cache_format("int4", fmt), "unknown kv rejected");

    // 37 HCA (ratio 128, dim 512) + 6 MLA-DSA (512 + 64 + 128 idx) at 262,144 context in FP8
    const uint64_t bpp_fp8 = ctxp::bytes_per_position(37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    const uint64_t bpp_fp16 = ctxp::bytes_per_position(37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp16);
    expect(bpp_fp8 > 4000 && bpp_fp8 < 5000, "fp8 bytes/pos ~4.5 KiB");
    expect(bpp_fp16 > 8000 && bpp_fp16 < 9500, "fp16 bytes/pos ~8.7 KiB");

    const auto p_ok = ctxp::plan_context(262144, 1048576, 6ull << 30, 37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    expect(p_ok.error.empty() && p_ok.context == 262144, "262K fp8 fits easily in 6 GiB free VRAM");

    const auto p_oob = ctxp::plan_context(2000000, 1048576, 6ull << 30, 37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    expect(!p_oob.error.empty(), "context past 1M rejected");

    const auto p_oom = ctxp::plan_context(1048576, 1048576, 2ull << 30, 37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    expect(!p_oom.error.empty(), "1M context exceeding half of 2 GiB free VRAM rejected");

    // Bad pack path returns clean error
    Pack pk;
    std::string err;
    expect(!open_pack("/nonexistent/ernie55.basalt", pk, err) && !err.empty(), "missing pack returns error");

    std::printf("ernie55_pack_test: %d failures (fp8=%llu B/pos, fp16=%llu B/pos, 262K fp8=%.2f GiB)\n",
                g_fail, (unsigned long long) bpp_fp8, (unsigned long long) bpp_fp16,
                (double) p_ok.kv_bytes / (double) (1ull << 30));
    return g_fail ? 1 : 0;
}
