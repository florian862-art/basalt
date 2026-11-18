# Fork notes — our three fixes, and how we run 0.4.1 on this machine

This page is ours, not upstream documentation. It records which branches carry our
fixes, how we apply them to a release archive (we do not build from source), how the
engine is configured on this host, and the measurements behind those values. When we
change one of these, we change this page.

The host, so the numbers can be placed: two GPUs with 16.3 GiB each, 184 GiB RAM, a
CPU without AVX-512 (the loader's `expert kernels run on AVX-2` warning is normal
here), the Qwen3.8 Flash-Next NVFP4/DQ6_K pack, one serving slot.

## Branches and where each one stands

| branch | files | status (18.11.) |
|---|---|---|
| `parser-code-state` | `serve/frontend.py`, `serve/tests/test_stray_backtick.py` | [PR #424](https://github.com/jesdga95/basalt/pull/424) open |
| `http-body-timeout` | `serve/handler.py`, `serve/tests/test_body_timeout.py` | [PR #425](https://github.com/jesdga95/basalt/pull/425) open |
| `think-end-quote` | `serve/frontend.py`, `serve/tests/test_quoted_think_end.py` | pushed here, **no PR yet** |
| `local-all3` | all three together | local only — this is what our running image contains |

[Issue #426](https://github.com/jesdga95/basalt/issues/426) is ours too:
`tool_call_recovery` is in the parser but no caller can turn it on.

`parser-code-state` and `think-end-quote` both touch `serve/frontend.py`, on
different parts of it; `local-all3` stacks them and has been the file our agents
have run against for days.

## Applying them to a release archive

We run the shipped `bin/basalt-v0.4.1-linux-x64.tar.gz` and overlay two files, so a
test image is one build away and no engine internals move:

```dockerfile
COPY patch/serve/frontend.py /opt/basalt/serve/frontend.py
COPY patch/serve/handler.py  /opt/basalt/serve/handler.py
```

`patch/serve-orig/` holds the archive's own copies, so `diff -r` shows exactly what
we changed. Because the overlay starts from the **0.4.1 file** and the branches are
on `main`, our files are not byte-identical to the branch files: the added code
lines match (checked line by line), comment wording differs where the two base
versions differ. A rebuild for a newer release has to re-apply the changes, not copy
the files.

Checks after every image build: the upstream suites that cover this code
(`serve/tests/test_content_fences.py`, `test_quoted_call_values.py`,
`test_quoted_tool_tags.py`) and the three shapes each of our fixes is for.

## Configuration on this host

`basalt_config.json`, `engine` block — keys are engine flag names without the
leading dashes; every one was looked up in `docs/configuration/index.md`, since an
unknown name stops the start:

```json
{
  "spec": 8,
  "prefill": "auto",
  "resident-experts": true,
  "pool-workers": 20,
  "vram-reserve-mib": 700,
  "peer-reserve-mib": 600,
  "ram-reserve-mib": 8192,
  "conversation-cache-mib": 24576,
  "conversation-cache-slots": 6,
  "conversation-cache-min-free-mib": 16384
}
```

Model entry: `max-context 262144`, `kv int8`, `parallel 1`, `vision 1`,
`fit-max-tokens true`.

Why, in one line each:

- **`kv int8`** — measured lossless against fp16 (dKLD +0,0000 [−0,0010, +0,0010])
  in `docs/precision`; `k8v4` (+0,0056) and `q4_0` (+0,0093) cost visible quality,
  and fp16 would only spend VRAM the expert slots need.
- **`pool-workers 20`** — 24 physical cores without SMT; `auto` picked 15 (P-cores
  minus the host thread plus half the E-cores). This is a value carried over from
  the previous engine on this box, not measured per pack.
- **`parallel 1`** — the conversation we serve daily is ~200 K tokens. Two slots at
  `max-context 262144` would give each 131,072 and answer that conversation with a
  400.
- **`ram-reserve-mib 8192`** — headroom for the rest of the system under the expert
  tiers, which take what is above the floor. Generous here because the host has
  184 GiB; on the 91 GiB machine this was the tight value.
- **`conversation-cache-*`** — a cold re-read of a 200 K-token conversation costs
  29 s on this host. Not proven useful yet, see the last section.

Flags we tested and **dropped for quality cost**: `--route-prior` and
`--route-tail-skip` (top-1 91.81 → 85.00 % for +2.6 % speed on this NVFP4 pack) and
`--relaxed-drafting`. Not built at all: `--shared-expert-arena` (our container's
`/dev/shm` is the Docker default 64 MiB and the arena file is ~54 GiB — that is a
house problem, not an engine one).

## Measurements behind them

Method: same pack, same client, `temperature 0`, cold prompts, one slot, nothing
else asking the engine during a run. Prompt rate is `prompt_tokens ÷ time to first
token`, for the reason at the end of this section; decode rate is the server's own
`done` line. This is one host and one pack, not a benchmark suite.

| | 0.4.1 with our patches | previous engine on this box |
|---|---|---|
8,216-token prompt, first token | 2.58 s (3,189 tok/s) | 3.28 s (2,505) |
32,715-token prompt | 5.34 s (6,121) | 11.02 s (2,970) |
157,942-token prompt | 20.1 s (7,857) | 56.6 s (2,788) |
short prose decode (75 → 220 tokens) | 151.8 / 158.7 tok/s | 92.9 tok/s |
first token, small prompt | 0.28–0.40 s | 0.68–1.61 s |

**`spec` 4 → 8 did not buy anything here.** Widening the verify window from 4 to the
default 8, same short prose prompt: decode 151.8 / 158.7 → 122.3 / 126.6 tok/s,
draft acceptance 69–77 % → 59–63 %, experts over PCIe steady at 29–35 %, 8K prompt
rate 3,189 → 2,588 / 3,087 tok/s. The extra depths look mostly rejected while their
verification still costs traffic on a host whose decode sits on the PCIe expert
stream. Not a clean single-variable run — worker count, `ram-reserve` and the
conversation cache changed in the same start, and a second conversation shared the
slot. We are repeating it with only `spec` moved.

**Why we do not read the prompt rate from `timings.prompt_per_second`.** On cold
prompts it reported 82,160,000 tok/s for 8,216 tokens (exactly `prompt_tokens × 10⁴`)
and 197,427,500 tok/s for 157,942 tokens, and 0 when the whole prefix was cached.
Nothing is claimed about the cause; the field is just not usable for a comparison,
so the rate above comes from the client's own time to first token. Say the word if a
reproducible issue would help.

## Conversation cache: configured, not yet seen working

24 GiB RAM, 6 slots, 16 GiB floor are accepted and reported by `/metrics`, but the
log holds no parked conversation, and RAM parking does not survive a restart: one
conversation had to re-read 43 K tokens cold (`pp=50836/7512c 9.1s`) after we
stopped and started the engine. `conversation-cache-disk-mib` with
`conversation-cache-dir` is the next thing we try; the disk parking is the part that
would have to outlive a restart.
