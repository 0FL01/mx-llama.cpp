# Same-device expert-cache reuse for prefill

`LLAMA_MOE_CACHE_PREFILL_D2D=1` enables an experimental, default-off source
selection for the scheduler's selective expert-weight upload. Set it before
creating the context. Unset or `=0` preserves the original H2D path.

The existing MoE cache must be enabled and contain canonical resident experts.
For same-backend hits, the scheduler copies the exact cached expert slice with
the backend's asynchronous D2D interface into its existing full-layout weight
temporary. Misses, foreign owners, unsupported layouts and failed async-copy
eligibility retain H2D. The private lookup is read-only: routing IDs, placement,
admissions, cache maps, arithmetic and temporary allocation are unchanged.

The original consecutive-expert groups are retained. Their extra padding of up
to 512 bytes always comes from the canonical host neighbour, not the next cache
slot. The latter may contain an unrelated expert. No new GPU staging allocation
is introduced, and the zero dummy slot is untouched.

## Qualification on two gfx906 / MI50 16 GiB devices

The 2026-09-28 comparison used Qwen3.8 Flash Next legacy GGUF, layer split 1,1,
canonical CPU routed experts and PLE, ranked cache112/inserts2, MTP2 on ROCm1,
context131072, batch/ubatch1024, threads16, FA on, q4_0 KV and DIO/lazy off.
The production environment was held fixed; Q4 planar repack was off. Requests
used identical token arrays, greedy generation and an identical explicit warmup.
Four fresh blocks A/B/B/A used one frozen binary and diagnostics were off.

| Warm MTP2 fixture | A PP tok/s | B PP tok/s | PP change | Post-PP TG change |
|---|---:|---:|---:|---:|
| PP512 + TG512 | 96.976 | 104.111 | +7.36% | -1.54% |
| PP4096 + TG512 | 170.896 | 182.952 | +7.05% | +0.31% |
| PP16384 + TG512 | 164.064 | 176.013 | +7.28% | +0.02% |

Both B PP repeats exceeded both A repeats for each fixture. All 512-token outputs
and speculative counters matched. This is a workload-specific warm-prefill win,
not an established TG improvement or broad statistical significance.

A separate profile conserved the 42,542,158,336-byte PP512 expert-transfer budget:
6,369,894,400 bytes moved from H2D to D2D, with identical canonical tails and cache
metadata. Fingerprints describe residency metadata, not weight checksums.

Server startup is not an empty-cache control: its rollback-capability probe
decodes two zeros and admits 96 experts even with `--no-warmup` and no sidecar.
`test-moe-cache-prefill` instead loads the model without that probe and creates a
fresh target context for each fixture. Empty-map profiles had zero D2D bytes,
identical H2D/tail budgets and byte-identical full-vocabulary F32 logits.

| Truly empty target-cache fixture | A PP tok/s | B PP tok/s | PP change | Inclusive post-PP TG change |
|---|---:|---:|---:|---:|
| PP512 + TG512 | 81.868 | 81.139 | -0.89% | -0.43% |
| PP4096 + TG512 | 176.952 | 177.548 | +0.34% | -1.69% |
| PP16384 + TG512 | 167.952 | 168.069 | +0.07% | -0.95% |

All cold logits and 512-token streams matched across A/A, B/B and A/B. Cold
throughput is neutral at this repetition count. Manual post-TG uses inclusive
generation wall time: `llama_perf` excludes cache-update uploads and must not be
presented as whole-generation throughput. These manual numbers are not directly
comparable to the HTTP/MTP2 category.

## Checks and diagnostics

Build with HIP tests enabled, then run on gfx906:

```sh
ctest --test-dir build-throughput -R '^(test-arg-parser|test-moe-cache-sched)$' --output-on-failure
```

The cache-copy GPU test covers Q4_0/Q4_1, widths1/4/64, broadcast/per-expert inputs,
cold/hit/foreign-owner cases, actual async-copy calls, canonical payload/tails and
unchanged cache/dummy bytes. All 144 comparisons were bit-exact. Router IDs are
distinct within each token; repetitions across tokens are tested.

The manual test requires `LLAMA_MOE_PREFILL_FIXTURES` (fixed request JSON) and
`LLAMA_MOE_PREFILL_OUTPUT_PREFIX`, plus the ordinary model/server-style preset
arguments and `--n-predict 512`. It writes initial full-vocabulary F32 snapshots
and structured PP, perf-TG, inclusive generation-wall and token records.

For separate diagnostics, `GGML_SCHED_TIMING=1` reports H2D/D2D/tail bytes and
eligible cache-hit/miss counts. `LLAMA_THROUGHPUT_TRACE=1` reports reason-tagged
waits and cache snapshots. Disable both for throughput measurements.

Raw commands, fixtures, binary/library hashes, 24 server records, 12 cold records,
resource samples and summaries are retained on the qualification host under
`build-throughput-qualification/`: `r2-manifest.json`, `r2-summary.json`,
`r2-cold-manifest.json`, `r2-cold-profile-summary.json` and `r2-final-evidence.json`.
The opt-in is qualified for this workload; it remains off by default and no
production preset/image was changed.
