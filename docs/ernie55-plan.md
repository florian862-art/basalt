# ERNIE-5.5-Flash-Preview in Basalt — Umsetzungsplan (`feat/ernie55`)

Dieses Dokument beschreibt das geplante Vorgehen, um **`baidu/ERNIE-5.5-Flash-Preview`** (280B gesamt / 14B aktiv, MXFP4-Routed-Experts, 1M Kontext) nativ, qualitätstreu und hardwareoptimiert in **Basalt** zu integrieren.

**Ziel-Hardware:** 2× NVIDIA RTX 5070 Ti (32 GiB VRAM, Blackwell SM 12.0), Intel Core Ultra 7 270K Plus (24 Kerne, AVX2 + AVX-VNNI), 183 GiB DDR5-RAM.  
**Git-Basis:** Feature-Branch `feat/ernie55` direkt ab `origin/main` (`jesdga95/basalt`, inkl. MXFP4-Grundsupport `#481`, `HTTP_BODY_S` `#425`, `</think>` `#448`, Elastic-KV `#543`/`#549`). Der laufende Produktiv-Container `basalt` bleibt während der gesamten Entwicklung unberührt.

---

## 1. Architektur- & Speicher-Profil

| Eigenschaft | Wert |
|---|---|
| Architektur | `Ernie5_5_MoeForCausalLM` (`ernie5_5_moe`) → Basalt-ID `ernie55` |
| Layer | 43 (Layer 0 Dense-MLP; Layer 1–42 Latent-MoE) |
| Hidden / Vocab | `hidden_size = 4096`, `vocab_size = 201216`, `max_position_embeddings = 1,048,576` |
| Attention (Hybrid) | **37× HCA** (`compress_ratio = 128`, `head_dim = 512`, `o_groups = 8`, `o_lora = 1024`, YaRN `160000`, `attn_sink`, VHA PostMix, Output-Gate) + **6× MLA-DSA** (Layer `7, 14, 21, 28, 35, 42`; `q_lora = 1024`, `kv_lora = 512`, `qk_nope = 192`, `qk_rope = 64`, `v_head_dim = 256`, Top-2048 Lightning Indexer + 128-Token Sliding Window + `softmax_offset`, VHA PostMix, Output-Gate) |
| MoE (Latent) | 512 Routed Experts (`2048×2048`, Top-10, Split-Feature Sigmoid-Routing + Bias + lernbare Per-Expert-Skala), `fc1_latent_proj` (`4096→2048`), `latent_norm` + `fc2_latent_proj` (`2048→4096`), 1 Shared Expert |
| Aktivierung | **SiTU-GLU** (`activation_situ_beta = 4.0`, `activation_situ_linear_beta = 25.0`) |
| Residual / Head | **mHC** (`hc_mult = 4`, 20 Sinkhorn-Iterationen) + `mhc_contract` + FP32 **LM-Head Multimax** |
| Drafter | Gebündelter 3-Layer **DSpark** unter `model.mtp.0..2` (512 MXFP4-Experten/Stage) |

### Speicherbudget auf 2× 16 GiB VRAM + 183 GiB RAM

| Teil | Pack-Format | Größe | Laufzeit-Ort |
|---|---|---|---|
| 42× 512 Routed Experts (`w1`/`w2`/`w3`) | **MXFP4 (GGML Typ 39)** bitgenau aus HF `I8`+`F8_E8M0` | **133,9 GiB** | ~15 GiB GPU 1 (Peer) + ~3–5 GiB GPU 0 + ~115 GiB gepinnt im RAM (**0 SSD-Spill**) |
| Dense / Attention / Shared / Latent-Proj | **Q8_0** (aus blockweisem 128×128 FP8) | **~8,0 GiB** | GPU 0 |
| `embed_tokens` | **BF16** | **1,54 GiB** | RAM (CPU-Lookup) |
| `lm_head` | **BF16 im Pack**, beim Laden **Q8_0** (~0,82 GiB, optional BF16) | **1,54 GiB** (Pack) / **0,82 GiB** (VRAM) | GPU 0 |
| KV-Cache (`--kv fp8` Default) | HCA komprimiert + MLA-DSA FP8 + Indexer FP4 | **~1,0–1,2 GiB** @ 262K | GPU 0 |
| DSpark (Stufe 2, Top-32/64 Experts) | MXFP4 + Q8_0 | **~0,6–1,2 GiB** | GPU 0 |

---

## 2. Qualitäts-Entscheidungen

1. **Routed Experts bleiben Original-MXFP4:** Keine Re-Quantisierung — bitgenau zu Baidus W4A8-QAT-Gewichten.
2. **Dense/Attention als Q8_0 (nicht per-tensor FP8):** Basalts vorhandener `fp8_gemm` arbeitet per-tensor, ERNIE 5.5 nutzt 128×128 blockweises FP8 (`ue8m0`). Q8_0 (1×32 Block, 8-Bit linear) vermeidet jeglichen Per-Tensor-Qualitätsverlust und nutzt Basalts fertige Q8-/TMA-Kernels.
3. **`lm_head` BF16 im Pack + Multimax strikt in FP32:** Verlustfreie Pack-Ablage; Q8_0-Load spart ~720 MiB VRAM innerhalb des Placement-Rauschens.
4. **Verifikation vor Optimierung:** Jeder neue Kernel und Layer wird gegen PyTorch/`vLLM-ERNIE`-Referenzwerte paritätsgetestet (Max-Abs/Rel-RMS + End-to-End Logit-/KLD-Abgleich).

---

## 3. Umsetzungs-Phasen

### Stufe 1 — Vollständige, verifizierte Basis (schnell zum ersten lokalen Token)

1. **Kernels: MXFP4 Gate/Up + SiTU-GLU (CUDA + CPU)**
   - MXFP4 (Typ 39) zusätzlich zu `D_FMTS` auch für `GU_FMTS` (Decode-Fused-Gate/Up, Prefill `mmq_mxfp4`, CPU `ggml_cpu_dots` / `native_expert`).
   - **SiTU-GLU** (`beta=4.0`, `linear_beta=25.0`) für Dense-MLP, Shared Experts und MoE Experts auf GPU und CPU.
   - Paritätstests gegen FP64/FP32-Referenz.
2. **Konverter (`tools/convert/checkpoint_ernie55.py` + `tools/convert/arch/ernie55.py`)**
   - Direkte Konvertierung aus den 128 HF-Safetensors (in `~/.cache/huggingface/hub/models--baidu--ERNIE-5.5-Flash-Preview/`).
   - Bitgenaues Umpacken von HF MXFP4 (`I8 [2048, 1024]` + `F8_E8M0 [2048, 64]`) nach GGML Typ 39 (`block_mxfp4`).
   - Dequantisierung 128×128 FP8 → Q8_0 für Dense/Attention/Shared/Latent-Proj.
   - Optionale Mitnahme von `model.mtp.*` (DSpark) und `flashindexer/flashindexer.safetensors` im Pack-Schema.
   - Unit-Tests mit synthetischen Mini-Checkpoints + Bit-Exakt-Check der MXFP4-Repack-Logik.
3. **Modell-Runtime (`src/models/ernie55/` + `include/basalt/models/ernie55/`)**
   - Baut auf den bewährten Mustern von `deepseek41` (HCA, mHC, MLA-DSA) und `hy4flash` (Hybrid-Layer, Latent-MoE, Dual-GPU Peer-Tier, Doorbell-Handoff) auf.
   - ERNIE-5.5-Erweiterungen:
     - **VHA PostMix** (`m = I + V·Uᵀ`, beim Laden gefaltet) + **Attention-Output-Gate** (`sigmoid(o_gate(qr))`) für HCA und MLA-DSA.
     - **6× MLA-DSA** mit Sliding-Window-128 + Top-2048 Indexer-Merge + `softmax_offset`.
     - **Latent MoE** mit Split-Feature-Routing (`sigmoid(l0) + sigmoid(l1)` + Bias + lernbare Per-Expert-Skala), `fc1_latent_proj`, `latent_norm`, `fc2_latent_proj`.
     - **mHC-Kontraktion** (`model.mhc_contract.*`) + **FP32 LM-Head Multimax**.
   - Dual-GPU Peer-Tier (`--gpu 0,1`) + `peer_share` im Prefill + `--kv fp8` / `--kv fp16`.
   - Layer- & End-to-End-Paritätstests gegen `vLLM-ERNIE`/PyTorch.
4. **Serve-Dialekt (`serve/models/ernie55.py`)**
   - Segment-basiertes Chat-Rendering (entsprechend `encoding_ernie.py` / `decoding_ernie.py`), Thinking (`enable_thinking`, `reasoning_effort`) und Tool-Call-Parser + Pytest-Suite.

### Stufe 2 — Maximal-Performance für 2× RTX 5070 Ti + Core Ultra 7 270K Plus

5. **Multi-Row MXFP4 AVX2-VNNI CPU-Kernel:** Liest RAM-Experten bei 2–8 Token (Verify-Window) nur 1× aus DDR5.
6. **DSpark-Drafter (`model.mtp.*`):** 3 SWA-mHC-Stages auf GPU 0 (Top-32/64 Experten) → amortisiert DDR5-Bandbreite über 5–7 Token/Pass.
7. **FlashIndexer (`flashindexer.safetensors`):** Top-8 Indexer-Head-Routing im Prefill (~8× schnellerer Indexer bei 64K–262K+ Kontext).
8. **Nativer Blackwell SM 12.0 W4A8 (MXFP4×MXFP8) Prefill-MoE-GEMM:** Abgeleitet aus `moe_mxfp8.cu` / `moe_fp4.cu` für volle Blackwell-Tensor-Core-Rate und exakte Baidu-W4A8-Numerik.
