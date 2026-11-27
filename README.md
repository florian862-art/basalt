<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

# Basalt — ERNIE-5.5-Flash-Preview Port

> **Status: Merged into upstream [`jesdga95/basalt`](https://github.com/jesdga95/basalt)!**
> This port was integrated into upstream `main` via **[Pull Request #687](https://github.com/jesdga95/basalt/pull/687)** (merge commit [`5fe1928d`](https://github.com/jesdga95/basalt/commit/5fe1928dcc82ad62286d2a622bcbb5d85ebfba2b)).
>
> **Links:** [Merged Upstream PR #687](https://github.com/jesdga95/basalt/pull/687) · [Ready-Built Pack on Hugging Face](https://huggingface.co/florian862-art/ERNIE-5.5-Flash-Preview-Basalt) · [Model Guide (`docs/models/ernie55.md`)](docs/models/ernie55.md) · [Precision & Benchmarks (`docs/precision/ernie55.md`)](docs/precision/ernie55.md)

This repository is a showcase fork of **[`jesdga95/basalt`](https://github.com/jesdga95/basalt)** ([basalt.jgulfo.com](https://basalt.jgulfo.com)) documenting the native support port for Baidu's **[`ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)** on NVIDIA Blackwell GPUs (SM 12.0).

For active development, releases, and all supported models (Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.8-27B, Hy4-flash, Inkling 2 Mini, and ERNIE-5.5-Flash), please use upstream **[`jesdga95/basalt`](https://github.com/jesdga95/basalt)**.

---

## What This Port Added ([PR #687](https://github.com/jesdga95/basalt/pull/687))

- **Full Native `ERNIE-5.5-Flash-Preview` Runtime (`src/models/ernie55/`):**
  - 1 dense + 42 Latent MoE layers (`512` MXFP4 routed experts, top-`10` active + shared expert, SiTU-GLU `4.0/25.0`), 37× HCA + 6× MLA-DSA hybrid attention, mHC 4-stream Sinkhorn-Knopp residuals, native 1M YaRN context, reasoning/thinking control, and parallel XML tool calling (`<ernie:tool_call>`).
  - Weightless per-head HCA Q-RMSNorm across all 37 HCA layers and all 3 DSpark layers (critical for long-context coherence).
  - Ultra-compact FP8 KV cache (~`4.5 KB/token`, ~`1.13 GiB` at 262K context) and full hybrid HCA/DSA conversation cache (`save_cells` / `restore_cells`, ~`169 MB` per 35K-token conversation).
- **Decode & Prefill Acceleration:**
  - Per-window CUDA graphs, mapped doorbell, PDL fused kernels, GPU-1 peer RAM fetches (prioritizing multi-token experts), cross-layer routing hints + L3-grouped CPU cache warming, and a shipped 180-request hot-expert profile (`data/profiles/ernie55-43x512.prof`).
  - Tensor-Core MMQ (`GGML_TYPE_MXFP4`) + 50/50 dual-GPU `PeerShare` prefill with `R = 8192` aliased chunks, cross-layer `PeerShare::prepare()`, async GPU-0 staging, flat CSR bucketing, 64-head BF16 Tensor-Core Lightning Indexer, and deterministic 256-bin radix top-K.
  - Multi-row AVX2 + AVX-VNNI (`vpdpbusd`) MXFP4 CPU expert kernel (**8.3× faster** than ggml, bitwise `memcmp == 0`).
- **Native 3-Stage DSpark Speculative Drafter (`4.27 ms` draft latency):**
  - Directly attaches the bundled `model.mtp.*` DSpark tensors from the official Hugging Face checkpoint (`--spec 8` default, `--spec 0` off).
- **End-to-End Conversion Tools:**
  - [`tools/convert/convert.py`](tools/convert/convert.py) ([`checkpoint_ernie55.py`](tools/convert/checkpoint_ernie55.py)) + [`tools/convert/dspark_ernie55_attach.py`](tools/convert/dspark_ernie55_attach.py).

*(Note: Baidu's `ERNIE-5.5-Flash-Preview` release is text-only and does not include a vision encoder.)*

---

## Requirements

- **GPU:** 1 or 2 NVIDIA Blackwell GPUs (SM 12.0, CUDA 13).
- **CPU:** Linux x86-64 with AVX2 (AVX-VNNI `vpdpbusd` recommended) or Windows x64.
- **Memory:** ~`152.6 GiB` loaded (`~9.1 GiB` GPU 0 + remaining experts across GPU 0/1 VRAM and system RAM). For 16–32 GB GPUs, **192 GB system RAM** is recommended.
- **Disk:** ~`156 GB` for the converted `.basalt` pack (plus temporary space for the HF snapshot during conversion).
- **Build tools:** CMake 3.24+, GCC 13+ (Linux) or VS 2022 + Ninja (Windows), Python 3.10+ (`numpy`).

---

## How To Build, Convert & Run ERNIE-5.5-Flash-Preview

### 1. Clone and Build (from Upstream)

```sh
git clone https://github.com/jesdga95/basalt.git
cd basalt

# Build ERNIE-5.5 only (fast) or pass -DBASALT_MODELS=all for every Basalt model:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=ernie55
cmake --build build -j

pip install -r requirements.txt
```

### 2. Download Ready-Built `.basalt` Pack (Fastest)

A verified ready-built pack (`166.7 GB` / `155.3 GiB`, with DSpark attached, SHA-256 `7af8cd7a90ce86c269c4fbe5a24f0b5f40266d5f72d054ba9999a7edda088728`) is available on **[`florian862-art/ERNIE-5.5-Flash-Preview-Basalt`](https://huggingface.co/florian862-art/ERNIE-5.5-Flash-Preview-Basalt)**:

```sh
hf download florian862-art/ERNIE-5.5-Flash-Preview-Basalt ERNIE-5.5-Flash-Preview-MXFP4.basalt --local-dir models
```

*(If you use the ready-built pack, skip straight to Step 5 below.)*

### 2b. Or Self-Convert from the Official Hugging Face Checkpoint

Download **[`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)** (Apache-2.0 preview weights). The bundled **DSpark** drafter (`model.mtp.*`) is included in the **same** repository — no second download is required:

```sh
pip install -U huggingface_hub
hf download baidu/ERNIE-5.5-Flash-Preview --local-dir ./ERNIE-5.5-Flash-Preview
```

### 3. Convert to `.basalt`

```sh
mkdir -p models
python3 tools/convert/convert.py \
  --checkpoint ./ERNIE-5.5-Flash-Preview \
  --out models/ERNIE-5.5-Flash-Preview-MXFP4.basalt \
  --workers 16
```

### 4. Attach the Bundled DSpark Drafter (~15 s)

```sh
python3 tools/convert/dspark_ernie55_attach.py \
  --checkpoint ./ERNIE-5.5-Flash-Preview \
  --into models/ERNIE-5.5-Flash-Preview-MXFP4.basalt \
  --workers 16
```

This appends a SHA-256-verified `dspark` section to `ERNIE-5.5-Flash-Preview-MXFP4.basalt` (~`156 GB` total) without reconverting the main weights. You can delete `./ERNIE-5.5-Flash-Preview` once conversion finishes.

### 5. Configure and Serve

Create `basalt_config.json` (example for 2× 16 GB Blackwell GPUs in `peer` mode; for a single GPU set `"gpu": "0"` and omit `"gpu-mode"`):

```json
{
  "version": 2,
  "models": [
    {
      "model": "models/ERNIE-5.5-Flash-Preview-MXFP4.basalt",
      "alias": "ernie-5.5-flash-preview-mxfp4",
      "gpu": "0,1",
      "gpu-mode": "peer",
      "max-context": 262144,
      "kv": "fp8",
      "spec": 8,
      "ram-reserve-mib": 24576
    }
  ],
  "engine": {
    "pool-workers": 20,
    "peer-reserve-mib": 350,
    "conversation-cache-disk-mib": 16384
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080
  }
}
```

Start the OpenAI / Anthropic compatible server:

```sh
bin/basalt serve --config basalt_config.json
```

All flags, KV cache math, and rollback options: [`docs/models/ernie55.md`](docs/models/ernie55.md) · Precision & quality measurements: [`docs/precision/ernie55.md`](docs/precision/ernie55.md).

---

## Benchmarks

- **Hardware:** 2× NVIDIA RTX 5070 Ti (16 GB each, SM 12.0), Intel Core Ultra 7 270K Plus (AVX2 + AVX-VNNI), 192 GB DDR5-6200
- **Footprint:** `152.6 GiB` loaded in `37.9–46 s` (`GPU 0: 9.1 GiB`, `GPU 1: 14.6 GiB`, `RAM: 128.9 GiB`, `--gpu 0,1 --gpu-mode peer --kv fp8`)

### `basalt-bench` (`--kv fp8`, `power limit 300 W`)

| Test | Throughput | Accept % | Draft Len |
|---|---:|---:|---:|
| `tg conv` | **40.5 t/s** | 72.4% | 1.71 |
| `tg struct` | **88.8 t/s** | 99.5% | 6.60 |
| `tg prose` | **46.4 t/s** | 65.7% | 1.60 |

### Prefill & Long-Context Serve (`--gpu 0,1 --gpu-mode peer --kv fp8`)

| Workload | Throughput | Details |
|---|---:|---|
| **Prefill `0–8K` (`R=8192`)** | **836–844 t/s** | `pp=7,554` in `9.0 s`, `pp=8,192` in `10 s` |
| **Prefill `4K` (3,876 tok)** | **691 t/s** | `5.6 s`, exact SHA-1 |
| **Prefill `16K` (`pp16384`)** | **751 t/s** | `basalt-bench` (`749 t/s` wall) |
| **Prefill `24K` (23,076 tok)** | **606 t/s** | `38.1 s`, exact SHA-1 |
| **Prefill `64K` (`pp65536`)** | **621 t/s** | `basalt-bench` (BF16 Tensor-Core Lightning Indexer) |
| **Prefill `131K` (131,072 tok)** | **489 t/s** | `278 s` (`ctx=262144`) |
| **Prefill `170K` (170,328 tok)** | **433 t/s** | `393.3 s`, `tg=27.7 t/s`, `acc=82%`, `dl=2.1` (`ctx=262144`) |
| **Sustained Decode (`7.5K` ctx, 1,305 tok)** | **46.4 t/s** | `acc=74%`, `dl=2.7`, `hit=28.8%`, `pcie=44.8%` |
| **Decode @ `16K / 24K` ctx** | **40.9 / 42.0 t/s** | `ctx=65536` |
| **DSpark Draft Latency** | **4.27 ms** | `conv --spec 3` |
| **CPU MXFP4 Kernel (1-thread, 2-tok down)** | **700 µs (8.95 GB/s)** | **8.3×** vs. ggml (`5805 µs`), bitwise `memcmp == 0` |

---

## Credits & License

- **Upstream Engine:** **[`jesdga95/basalt`](https://github.com/jesdga95/basalt)** by `@jesdga95` and contributors (integrated via **[PR #687](https://github.com/jesdga95/basalt/pull/687)**, including `@jesdga95`'s CUDA-graph decode, PDL fused kernels, L3-grouped CPU warming, and shipped expert profile).
- **Model:** [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (Apache-2.0 preview weights).
- **License:** MIT, see [LICENSE](LICENSE).
