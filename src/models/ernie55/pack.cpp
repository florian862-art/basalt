// src/models/ernie55/pack.cpp - reads an ERNIE-5.5-Flash .basalt pack's meta and experts.layout sections.
#include "models/ernie55/pack.hpp"

#include "basalt/ggml/ggml.h"

#include <sstream>

namespace basalt::models::ernie55 {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

std::vector<int> ints(const Json* j) {
    std::vector<int> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back((int) x.num());
    return v;
}

std::vector<uint64_t> unsigneds(const Json* j) {
    std::vector<uint64_t> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.u64());
    return v;
}

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "ernie55") {
        err = path + ": not an ernie55 pack";
        return false;
    }
    const Json* kv = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!kv || !kv->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    auto k = [&](const char* s) { return kv->get(std::string("ernie55.") + s); };
    auto u = [&](const char* s, double d = 0) { return (int) num_or(k(s), d); };
    auto f = [&](const char* s, double d = 0) { return (float) num_or(k(s), d); };
    auto b = [&](const char* s, bool d = false) {
        const Json* j = k(s);
        if (!j) return d;
        return j->is_number() ? j->num() != 0 : j->b;
    };

    Geometry& g = p.geo;
    g.n_layers = u("block_count");
    g.n_dense = u("first_k_dense_replace", 1);
    g.n_embd = u("embedding_length");
    g.n_ff = u("feed_forward_length");
    g.context_length = (int64_t) num_or(k("context_length"), 1048576);
    g.rms_eps = f("attention.layer_norm_rms_epsilon", 1e-6);
    g.window = u("attention.sliding_window", 128);
    g.compress_ratios = ints(k("attention.compress_ratios"));

    g.hca_heads = u("attention.head_count", 64);
    g.hca_kv_heads = u("attention.head_count_kv", 1);
    g.hca_head_dim = u("attention.key_length", 512);
    g.hca_v_dim = u("attention.value_length", 512);
    g.hca_q_lora = u("attention.q_lora_rank", 1024);
    g.hca_kv_lora = u("attention.kv_lora_rank", 512);
    g.hca_o_groups = u("attention.output_group_count", 8);
    g.hca_o_lora = u("attention.output_lora_rank", 1024);
    g.hca_rope_dim = u("rope.dimension_count", 64);
    g.rope_theta = f("rope.freq_base", 8000000.0);
    g.comp_rope_theta = f("attention.compress_rope_freq_base", 160000.0);
    g.yarn_factor = f("rope.scaling.factor", 16.0);
    g.yarn_orig_ctx = u("rope.scaling.original_context_length", 65536);
    g.yarn_beta_fast = f("rope.scaling.beta_fast", 32.0);
    g.yarn_beta_slow = f("rope.scaling.beta_slow", 1.0);

    g.mla_heads = u("mla.head_count", 64);
    g.mla_q_lora = u("mla.q_lora_rank", 1024);
    g.mla_kv_lora = u("mla.kv_lora_rank", 512);
    g.mla_qk_nope = u("mla.qk_nope_head_dim", 192);
    g.mla_qk_rope = u("mla.qk_rope_head_dim", 64);
    g.mla_v_dim = u("mla.v_head_dim", 256);
    g.idx_heads = u("attention.indexer.head_count", 64);
    g.idx_dim = u("attention.indexer.key_length", 128);
    g.idx_top_k = u("attention.indexer.top_k", 2048);

    g.n_expert = u("expert_count", 512);
    g.n_used = u("expert_used_count", 10);
    g.n_shared = u("expert_shared_count", 1);
    g.n_ff_exp = u("expert_feed_forward_length", 2048);
    g.n_latent = u("expert_latent_length", 2048);
    g.norm_topk_prob = b("expert_weights_norm", true);
    g.gate_softcap = f("expert_gate_softcap", 4.0);
    g.up_softcap = f("expert_up_softcap", 25.0);

    g.hc_mult = u("hyper_connection.count", 4);
    g.sinkhorn_iters = u("hyper_connection.sinkhorn_iterations", 20);
    g.hc_eps = f("hyper_connection.epsilon", 1e-6);
    g.has_flashindexer = b("has_flashindexer", false);

    if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_vocab <= 0) g.n_vocab = u("vocab_size");

    if (g.n_layers <= 0 || (int) g.compress_ratios.size() != g.n_layers || g.n_embd <= 0 || g.n_vocab <= 0 ||
        g.n_dense < 0 || g.n_dense > g.n_layers || g.hc_mult != 4) {
        err = path + ": incomplete or unsupported ernie55 geometry";
        return false;
    }

    const core::ModelFile::Section* gguf = p.file.find("gguf");
    const core::ModelFile::Section* exps = p.file.find("experts");
    if (!gguf || !exps) {
        err = path + ": pack misses gguf / experts";
        return false;
    }
    p.experts_range = p.file.range("experts");

    p.tensors.clear();
    for (const auto& [name, val] : tj->members) {
        if (!val.is_object()) continue;
        TensorRef tr;
        tr.name = name;
        tr.type = (int) num_or(val.get("type_id"), -1);
        tr.shape = unsigneds(val.get("shape"));
        tr.offset = gguf->offset + (uint64_t) num_or(val.get("offset"), 0);
        tr.bytes = (uint64_t) num_or(val.get("bytes"), 0);
        if (tr.type < 0 || tr.bytes == 0) {
            err = path + ": malformed tensor entry " + name;
            return false;
        }
        p.tensors.emplace(name, std::move(tr));
    }

    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    std::vector<uint8_t> layout_raw;
    if (!p.file.read("experts.layout", layout_raw, err)) return false;
    std::istringstream iss(std::string(layout_raw.begin(), layout_raw.end()));
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty() || line[0] == '#') continue;
        int l = -1, gu = 0, dn = 0;
        uint64_t off = 0, blob = 0;
        std::istringstream ls(line);
        if (!(ls >> l >> gu >> dn >> off >> blob)) {
            err = path + ": malformed experts.layout line: " + line;
            return false;
        }
        if (l < 0 || l >= g.n_layers) {
            err = path + ": experts.layout layer out of range: " + std::to_string(l);
            return false;
        }
        if (gu != GGML_TYPE_MXFP4 || dn != GGML_TYPE_MXFP4) {
            err = path + ": ernie55 expects MXFP4 experts (type 39), got " + std::to_string(gu) + "/" + std::to_string(dn);
            return false;
        }
        if (off + blob * (uint64_t) g.n_expert > p.experts_range.bytes) {
            err = path + ": experts.layout exceeds experts section at layer " + std::to_string(l);
            return false;
        }
        p.experts[(size_t) l] = {true, gu, dn, off, blob};
    }
    for (int l = g.n_dense; l < g.n_layers; ++l) {
        if (!p.experts[(size_t) l].present) {
            err = path + ": experts.layout misses MoE layer " + std::to_string(l);
            return false;
        }
    }
    return true;
}

}  // namespace basalt::models::ernie55
