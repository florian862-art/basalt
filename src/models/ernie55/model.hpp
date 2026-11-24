// src/models/ernie55/model.hpp - loaded ERNIE-5.5-Flash model: dense weights on GPU 0, BF16 embeddings in RAM,
// Q8_0 (or BF16) output head, and shared multi-GPU/RAM MXFP4 ExpertTier.
#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/model/adapter.hpp"
#include "models/ernie55/context_policy.hpp"
#include "models/ernie55/pack.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::ernie55 {

struct DevTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

constexpr int kMaxSecondCards = 3;
constexpr int64_t kDefaultContext = 8192;
constexpr uint64_t kExpertLendSegment = 256ull << 20;

class ErnieModel final : public model::LoadedModel {
public:
    ErnieModel();
    ~ErnieModel() override;

    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req,
                         std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;

    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }
    model::PeerHooks* peer() const override;

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const {
        return tensor("blk." + std::to_string(layer) + "." + suffix);
    }

    /// Reads one BF16 embedding row from host memory into `out` (n_embd floats).
    bool embedding_row(int32_t token, float* out) const;

    /// ExpertTier descriptor for the 42 MXFP4 MoE layers (latent dim 2048, ff 2048, SiTU-GLU 4/25).
    core::ExpertTierSpec tier_spec() const;
    size_t expert_stride() const;
    core::LoadPlan* planning() const { return planning_; }

    struct SharedTier {
        std::shared_ptr<core::ExpertTier> tier;
        float* h_x = nullptr;
        float* h_rows = nullptr;
    };
    SharedTier& shared_tier() const { return shared_tier_; }

    const model::LoadRequest& flags() const { return flags_; }
    const std::string& path() const { return path_; }
    int device() const { return device_; }
    bool cache_fp8() const { return cache_fmt_ == ctxp::CacheFormat::Fp8; }
    ctxp::CacheFormat cache_format() const { return cache_fmt_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    core::LoadPlan* planning_ = nullptr;
    mutable SharedTier shared_tier_;
    model::GpuCards gpus_;
    model::LoadRequest flags_;
    std::string path_;
    std::vector<uint16_t> embd_;   ///< BF16 [n_vocab, n_embd] in host RAM
    int device_ = 0, fd_ = -1;
    ctxp::CacheFormat cache_fmt_ = ctxp::CacheFormat::Fp8;
    int64_t ctx_ = kDefaultContext;
};

std::unique_ptr<model::ModelSession> make_session(ErnieModel& model, std::string& err);

}  // namespace basalt::models::ernie55
