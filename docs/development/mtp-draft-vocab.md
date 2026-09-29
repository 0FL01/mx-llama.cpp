# Experimental Qwen4EXP MTP vocabulary restriction

`LLAMA_MTP_DRAFT_VOCAB_RANGES` is an opt-in, process-start setting for the
Qwen4EXP **draft** head. Unset or empty keeps the original full-head path.
For example:

```sh
LLAMA_MTP_DRAFT_VOCAB_RANGES=0:65536,248044:248320
```

Ranges are half-open, ordered, non-overlapping token-ID intervals. Malformed,
negative, reversed, overlapping, or out-of-vocabulary ranges are rejected.
`0:<vocabulary-size>` is allowed as a full-range control. Set the environment
before creating any MTP context; changing it after graph creation is unsupported.

The implementation projects full-K row views before matrix multiplication, then
fills a full-size F32 logits tensor with negative infinity and inserts the kept
rows at their original token IDs. It neither copies nor requantizes the output
weights. Target logits, target sampling, verification, and the pre-head hidden
handoff are unchanged. Active LoRA or a scaled output head is rejected on the
restricted path rather than silently ignored.

The tested shared sidecar borrows the target's **Q6_K** output weight:
`[2560,248320]`, 521,472,000 bytes. The sidecar filename's Q8_0 label does not
describe this shared head. The example keeps 65,812 of 248,320 rows.

## Diagnostics and checks

- `test-mtp-vocab`: strict parser cases and selected-row/masked-gap comparisons,
  columns 1/2/3/4/8, three graph executions, CPU and optional ROCm backend.
  Qualification exercised 90 CPU/ROCm comparisons with zero observed error.
- `LLAMA_MTP_HEAD_PROFILE=1`: MTP-only completion-inclusive head interval.
  It installs an evaluation callback and refuses to replace a user callback.
  This is not a GPU-kernel-only timer; disable it for throughput measurements.
- `LLAMA_SPECULATIVE_STATS=1`: structured cumulative per-drafter statistics at
  existing request reporting points. `accepted` excludes replay corrective
  tokens; `callback_accepted` preserves raw state-update bookkeeping, and
  `replay_corrections` explains the difference. Per-position accepted counts use
  the same correction. Existing human-readable counters are unchanged.

## Qualification status

The narrow example reduced bounded head-interval medians from about 1.40 ms to
0.39 ms. It is **not a universal TG setting**: Russian baseline token coverage
was only 52.93%, acceptance fell from 67.58% to 46.25%, and observed TG fell
15.16%. English, Rust/C++, and JSON observed TG changes were +6.00%, +1.84%, and
+14.29%, but full/restricted output streams differed in every class. Those are
not matched-output speedup claims. Within each variant the two repeats matched.

An independent, same-binary wide-range control `0:196608,248044:248320` keeps
196,884 rows and reduces head medians from 1.40 ms to about 1.12 ms. Its fresh
A/B/B/A results were:

| Class | Full TG | Wide TG | Change | Acceptance full / wide | Equal A/B tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| English | 15.001 | 14.579 | -2.81% | 62.90% / 62.90% | 512/512 |
| Russian | 16.208 | 15.683 | -3.24% | 67.58% / 63.96% | 32/512 |
| Rust/C++ | 21.821 | 18.779 | -13.94% | 89.88% / 81.36% | 208/512 |
| JSON/tool | 20.989 | 21.044 | +0.26% | 88.03% / 93.37% | 52/512 |

TG is weighted committed generation throughput using the server's timing
convention. Two repeats per variant are not a significance test; the English
wide repeats themselves vary substantially. Every A/A and B/B stream matches.
Only English wide has a matched A/B stream. Other observed runtime comparisons
include different verification, replay, routing and cache histories.

Decision: the head-cost mechanism is verified, but neither range is qualified
as a universal TG improvement. Retain only the explicitly experimental,
default-off knob. No production preset has been changed. Raw manifests,
requests, logs, statistics, and resource samples are in the target checkout's
ignored `build-throughput-qualification/` directory (`r3v2-*` and `r3-wide-*`).
