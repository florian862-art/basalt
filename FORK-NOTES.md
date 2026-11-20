# Fork notes — our three fixes, and how we run the release build on this machine

This page is ours, not upstream documentation. It records which branches carry our
fixes, how we apply them to a release archive (we do not build from source), how the
engine is configured on this host, and the measurements behind those values. We run
0.5.0 since 20.11., 0.4.1 before; each release entry says which one a number came
from. When we change one of these, we change this page.

The host, so the numbers can be placed: two GPUs with 16.3 GiB each, 184 GiB RAM, a
CPU without AVX-512 (the loader's `expert kernels run on AVX-2` warning is normal
here), the Qwen3.8 Flash-Next NVFP4/DQ6_K pack, one serving slot.

## Branches and where each one stands

| branch | files | status (20.11.) |
|---|---|---|
| `parser-code-state` | `serve/frontend.py`, `serve/tests/test_stray_backtick.py` | [PR #424](https://github.com/jesdga95/basalt/pull/424) open |
| `http-body-timeout` | `serve/handler.py`, `serve/tests/test_body_timeout.py` | [PR #425](https://github.com/jesdga95/basalt/pull/425) **merged** ([`d7a9a0f5`](https://github.com/jesdga95/basalt/commit/d7a9a0f5)) |
| `think-end-quote` | `serve/frontend.py`, `serve/tests/test_quoted_think_end.py` | **superseded** — upstream merged [`bc7f2a2a`](https://github.com/jesdga95/basalt/commit/bc7f2a2a) (PR #448, 19.11.) fixing [issue #428](https://github.com/jesdga95/basalt/issues/428), which we reported; our branch is no longer needed once we run a build that contains it |
| `local-all3` | all three together, on 0.4.1 | local only |
| `try-0.5.0` | all three rebased onto the 0.5.0 code | local only — this is what our running image contains since 20.11. |

[Issue #426](https://github.com/jesdga95/basalt/issues/426) is ours too:
`tool_call_recovery` is in the parser but no caller can turn it on.

`parser-code-state` and `think-end-quote` both touch `serve/frontend.py`, on
different parts of it; the stacked branches are the files our agents have run against
for days.

## Applying them to a release archive

We run the shipped archive (`bin/basalt-v0.5.0-linux-x64.tar.gz` since 20.11.,
0.4.1 before) and overlay two files, so a test image is one build away and no engine
internals move:

```dockerfile
COPY patch/serve/frontend.py /opt/basalt/serve/frontend.py
COPY patch/serve/handler.py  /opt/basalt/serve/handler.py
```

`patch/serve-orig/` holds the archive's own copies, so `diff -r` shows exactly what we
changed. The overlay is re-applied per release (branch `try-0.5.0`), never copied from
the previous release's file: a rebuild has to re-create the changes on the new base.

**What each release still needs, checked rather than assumed.** We run our three cases
against the *unpatched* archive (`PYTHONPATH` on the extracted archive, the image only
as a runtime) before deciding what to overlay. On 0.5.0:

| our fix | on bare 0.5.0 | overlay? |
|---|---|---|
| parser code rules (three shapes) | three of four shapes are fine — upstream fixed those; **a call written in an indented block still fires as a call** | yes, until #424 lands |
| `HTTP_BODY_S` | absent (0 occurrences in the 0.4.1 and the 0.5.0 archive alike) | yes — but #425 is merged upstream and the merge is **after** the 0.5.0 cut, so the next archive must go **without** this overlay, or we would revert upstream's own fix |
| quoted `</think>` | still broken in the 0.5.0 archive | yes **for this release** — upstream merged [PR #448](https://github.com/jesdga95/basalt/pull/448) on 19.11. (our issue #428), and we verified the shape passes on current `main`: the next archive goes without it |

So of the three fixes, **one is still ours to carry**: the parser code rules, until
#424 lands. #425 and #448 are both merged **after** the 0.5.0 cut, which is why our
0.5.0 image still overlays both files.

And the other direction, checked on the built image: everything upstream changed in
these two files for 0.5.0 (`EFFORT`, `hold_back` in `frontend.py`;
`checked_return_progress` in `handler.py`) is present in the files we ship, so the
rebase rolled nothing back.

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

| | 0.5.0, `spec 8` (20.11.) | 0.4.1, `spec 4` (18.11.) | previous engine on this box |
|---|---|---|---|
| ≈8,200-token prompt, first token | 2.62 / 3.05 s (3,134 / 2,697 tok/s) | 2.58 s (3,189) | 3.28 s (2,505) |
| ≈32,700-token prompt | 5.60 s (5,844) | 5.34 s (6,121) | 11.02 s (2,970) |
| ≈158,000-token prompt | 20.62 s (7,661) | 20.1 s (7,857) | 56.6 s (2,788) |
| short prose decode (75 → 220 tokens) | 122.7 / 108.3 tok/s | 151.8 / 158.7 tok/s | 92.9 tok/s |
| first token, small prompt | 0.32–0.41 s | 0.28–0.40 s | 0.68–1.61 s |

**0.4.1 → 0.5.0 changed prompt processing by nothing we can see.** 158 K: 7,857 →
7,661 tok/s (−2.5 %), 32 K: 6,121 → 5,844 (−4.5 %). Both cold (`pp=…/0c`), same pack,
same prompts. That is below our own noise floor: two 8 K readings 13 s apart inside one
run differ by 16 % (2,697 against 3,134). Load time did move: 26.5 s → 31.8 s for the
same 69.1 GiB (page-cache state counts in there).

The short decode row is **not** a 0.5.0 regression: the 0.4.1 pair above ran
`spec 4`, and the 0.4.1 `spec 8` arm measured 122.3 / 126.6 tok/s — today's
122.7 / 108.3 sit in that band. Three `spec 8` samples (108–127) against two
`spec 4` samples (152–159) now.

**`spec` 4 → 8 did not buy anything here, and 0.5.0 shows why.** Widening the verify
window from 4 to the default 8, same short prose prompt: decode 151.8 / 158.7 →
122.3 / 126.6 tok/s, draft acceptance 69–77 % → 59–63 %, experts over PCIe steady at
29–35 %. 0.5.0 prints `dl` in the `done` line — the code computes it as generated
tokens (after the first) over verify windows, i.e. what a window actually delivers —
and that is the number that settles it. A controlled probe on 0.5.0 (220-token
answers, nothing else in the single slot, two prompt shapes, two runs each):

| prompt shape | decode | drafts accepted | `dl` | experts over PCIe |
|---|---|---|---|---|
| prose | 116.6 / 148.5 tok/s | 61 / 76 % | 2.9 | 28.4 % |
| a 14-row table | 122.0 / 228.3 tok/s | 93 / 96 % | 2.9 | 21.9 % |

Then the same prompt shapes at `spec 4`, same engine, same day, the config change as
the only variable:

| prompt shape | decode | drafts accepted | `dl` | experts over PCIe |
|---|---|---|---|---|
| prose | 105.7 / 132.2 tok/s | 71 / 68 % | 2.6 / 2.5 | 25.4 / 30.2 % |
| a 14-row table | 136.4 / 195.8 tok/s | 99 / 98 % | 2.6 | 24.3 / 25.4 % |

`dl` is 2.9 in all four `spec 8` runs and in every one of the 69 session windows we
read that morning (2.1 to 5.2). With `spec 8` the window may hold one accepted token
plus up to seven drafts, but a window delivers ~2.9 tokens whatever the acceptance
rate: the window is not the binding limit, the drafter's depth is. So widening it
cannot multiply anything (the documentation's 183.9 → 319.8 tok/s compares `spec 0`
against `spec 7` — off against on, not 4 against 8). And `spec 4` confirms it from the
other side: `dl` drops only 2.9 → 2.6 — which is also the proof the narrower window is
really running, since `GET /metrics` labels these fields unclearly (it pairs `spec 6 ·
mtp_max 4` with a start line and config of `spec=4`) — while throughput stays inside
the noise (the two prose runs of one stage alone spread 27 %, the stage mean differs by
10 %) and acceptance rises at the structured prompt (99–98 % against 93–96 %: fewer
deep drafts offered, fewer rejected). **We keep `spec 4`**: half the verify work,
nothing measurable given up. Acceptance, not window size, is where the speed is: same
length, same `dl`, structured content drafts to 175 tok/s against prose's 133.

**Why we do not read the prompt rate from `timings.prompt_per_second`.** On cold
prompts it reported 82,160,000 tok/s for 8,216 tokens (exactly `prompt_tokens × 10⁴`)
and 197,427,500 tok/s for 157,942 tokens on 0.4.1; on 0.5.0 the same prompt reports
263,230,000 tok/s, and it is 0 when the whole prefix was cached. On the same 0.5.0 row
`acc` says 100 % while `dl` says 2.3 tokens per window, which is another pair of fields
we cannot reconcile. Nothing is claimed about the cause; the fields are simply not
usable for a comparison, so the rate above comes from the client's own time to first
token. Say the word if a reproducible issue would help.

## Conversation cache: configured, not yet seen working

24 GiB RAM, 6 slots, 16 GiB floor are accepted and reported by `/metrics`, but the
log holds no parked conversation, and RAM parking does not survive a restart: one
conversation had to re-read 43 K tokens cold (`pp=50836/7512c 9.1s`) after we
stopped and started the engine. `conversation-cache-disk-mib` with
`conversation-cache-dir` is the next thing we try; the disk parking is the part that
would have to outlive a restart.
