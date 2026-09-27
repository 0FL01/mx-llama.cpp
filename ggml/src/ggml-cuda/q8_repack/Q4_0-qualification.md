# Q4_0 planar qualification — 2026-09-27

The opt-in implementation is qualified on the workload below, but **this run
does not demonstrate a performance win**. Q4_0 remains default-off; no production
image or configuration was changed. Q4_0 fusion remains deliberately disabled.

## Configuration and method

- Two gfx906 / MI50 GPUs, 17,163,091,968 VRAM bytes per device; ROCm 7.14.
- Canonical `master`, layout `10bf832a9d`, compute `e2ba23b3c6`, tests
  `e474b39462`, actual verify-width telemetry `772ec4b051`.
- Real legacy Q4_0 `qwen38-keep1-Q4_0.gguf` (Qwen3.8 Flash Next, about 75.4 GB
  of tensor data, 512 experts). No weights were changed.
- Target: `-sm tensor -tps 2 -ngl 999 -ncmoe 36 -c 8192 -b 512 -ub 512
  -fa on -ctk q4_0 -ctv q4_0 -t 16 --fit off -np 1 --no-warmup`.
  Thirty-six expert layers are host-resident; twelve are GPU-resident.
  The dynamic canonical expert cache is not used.
- MTP adds `--spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
  --spec-draft-n-max 3 -ngld 999`.
- One immutable binary, A/B/B/A separately for plain and MTP. Only
  `GGML_CUDA_REPACK_Q4_0=0/1` varies, before loading. Diagnostics are off.
- Fixed code/prose token IDs, temperature 0, seed 42, identical untimed warmup,
  `cache_prompt=false`, `ignore_eos=true`. Twenty-four measured responses.
  PP4096 occupies about 4K context; 16K/64K contexts were not measured.
- Aggregate rates divide summed tokens by summed times. Generation follows
  the server's `(predicted_n - 1) / predicted_ms` convention, including MTP
  committed generation, not draft-token throughput.

## Results

| Workload | A canonical Q4 (tok/s) | B planar Q4 (tok/s) | Change |
|---|---:|---:|---:|
| PP512 | 13.983 | 13.683 | -2.14% |
| PP4096 | 13.901 | 13.801 | -0.72% |
| Plain code TG128 | 6.404 | 6.371 | -0.51% |
| Plain prose TG512 | 6.577 | 6.460 | -1.78% |
| MTP code TG128 | 5.654 | 5.460 | -3.42% |
| MTP prose TG512 | 4.202 | 4.029 | -4.11% |

All A/A, B/B and A/B token streams agree within each workload. MTP acceptance
and actual verify-width histograms also agree exactly. Across two repeats per
variant: code accepts 154/212 draft tokens (72.64%), with widths
`2:12, 3:10, 4:78`; prose accepts 482/940 (51.28%), with widths
`2:86, 3:74, 4:382`. Histograms include checkpoint replays, not just committing
steps (72 code / 314 prose). Baseline MTP and plain generation are not asserted
bit-identical to each other.

No measured class crosses the predeclared 5% aggregate TG regression gate.
There are only two repeats per variant; no statistical-significance claim is
made. The mixed CPU/GPU result does not isolate a GPU-kernel speedup.
Sampled peak VRAM is 14,727,696,384 / 14,198,525,952 bytes, leaving
2,435,395,584 / 2,964,566,016 bytes. These are sampled peaks and aggregate free
memory, not exact transient peaks or largest contiguous free blocks. No OOM
occurred.

## Correctness evidence and limits

- Exact packed-nibble/FP16-scale host layout, including padding; GPU dense/MoE
  parity for single/narrow/MMQ32/MMQ64, random weights and real inventory shapes.
- Row/expert views, asynchronous upload, grouped-MMQ 4/4/1 workspace remainder,
  shared Q8_0/MXFP4 regression, and 56 TP2 split/reduction cases pass.
- Real model width4/5 logits, routing and 16 greedy tokens match exactly.
  Width6 matches all 16 tokens but not all logits/routing; universal bit-exact
  output is not claimed. Global `--no-repack` makes Q4 opt-in off/on identical.
- An initial association change was rejected for a 40.69% code PPL regression.
  Keeping stock scale-after-correction association restores fixed-fixture code
  PPL exactly to 15.6211071619; prose PPL is 6.12152268976 / 5.64316389934
  (A/B). Each fixture scores 511 teacher-forced tokens, not a general quality suite.

Reproduction: `tests/benchmark-q4-repack-model.py` for fixed HTTP fixtures and
`test-q4-repack-model` for routing/logit/teacher-forced checks. Raw commands,
responses, logs and GPU samples are in the target's ignored
`build-q4-qualification/qualified-*` artifacts and `qualified-summary.json`.
Measured binary SHA256: server
`1a9478771edda1bca758d0df4f6f164525a6fbb2a27517cef7512fd291752976`, HIP library
`da45ae5393d4d98b4f92fddf49eb6257f1dbf3d91853ac928751945d916b42f4`.
