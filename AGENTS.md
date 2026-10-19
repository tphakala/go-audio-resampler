# AGENTS.md

Orientation for coding agents and automated reviewers working on this repository. README.md is the user-facing reference (usage, quality tables, CLI tools). This file is the map for changing and reviewing the code.

## What this is

`github.com/tphakala/go-audio-resampler` (package `resampler`; the README imports it under the alias `resampling`) is a pure-Go, high-quality audio sample rate converter. It is a reimplementation of the multi-stage design of libsoxr (the SoX Resampler library, by Rob Sykes): a DFT pre-stage plus a polyphase FIR stage with Kaiser-window filters and cubic coefficient interpolation. It aims to:

1. **Build anywhere Go builds.** No cgo, no C toolchain. SIMD comes from `github.com/tphakala/simd` plus assembly kernels in `internal/simdops`, each with a pure-Go fallback. Only `cmd/resample-wav` uses `go-audio/wav` and `go-audio/audio`.
2. **Match libsoxr quality.** At engine level `engine.QualityHigh` is libsoxr HQ and `engine.QualityVeryHigh` is VHQ, and the quality tests exercise those. The root presets reach them differently: `NewEngine*` maps both High and VeryHigh to `engine.QualityHigh`, and `New` maps High and VeryHigh through precision 24 and 32 bits to `Quality24Bit` and `Quality32Bit` (`convenience.go`, `stages.go`); `doc.go` and the README table still say 20-bit and 28-bit. Keep those in line with whatever mapping is intended. THD, SNR and aliasing are measured against libsoxr reference data, and the README quality tables are backed by committed tests.
3. **Be safe for streaming and real-time use.** Chunked `Process` calls plus one `Flush` give the same audio as a one-shot resample, with documented latency, an allocation-free `ProcessInto` path, and per-channel state so channels can run in parallel.

Packages:

| Package | Role |
|---|---|
| `resampler` (root) | Public API: `Config`, `QualitySpec`, presets, `New`, `Resampler`, `MultiFlusher`, `SimpleResampler` and `SimpleResamplerFloat32`, `ResampleMono[Float32]`, `ResampleStereo[Float32]`, convenience constructors, interleave helpers, `GetInfo`, `NewEngine`/`NewEngineFloat32`, `GetPresetSpec`, and the pipeline types `Pipeline`, `StageSpec`, `Stage`, `RingBuffer` |
| `internal/engine` | The resampling algorithms: DFT stage (with a half-band fast path), polyphase and cubic stages, the single-ratio `Resampler`, filter parameters, stage adapter |
| `internal/pipeline` | Stage interface and the multi-stage pipeline, buffers, zero-copy processing |
| `internal/filter` | Kaiser-window low-pass filter design |
| `internal/mathutil` | Bessel I0, Kaiser beta, filter length estimation, shared constants |
| `internal/simdops` | SIMD dot products and the fused cubic polyphase kernel: AVX+FMA (amd64) and NEON (arm64) assembly, pure-Go fallback elsewhere |
| `internal/testutil` | Shared test helpers |
| `cmd/resample-wav` | CLI: resample a WAV file |
| `cmd/resample` | Demo and benchmark tool |
| `examples/basic`, `examples/streaming` | Runnable examples; `streaming` shows the real-time FIFO pattern |
| `rules` | gocritic ruleguard matchers (lint-only) |
| `test-reference` | C programs and data generators that need libsoxr; not part of the Go build |

Non-goals: wrapping libsoxr, bit-exact equality with libsoxr output (quality is compared by THD, SNR and spectra), variable-rate (time-varying) conversion, WAV or other file I/O in the library itself (only the CLI reads files).

### API stability

The module is v1 (see CHANGELOG.md for the current release). The exported API of the root package is the contract: do not change a signature, remove a symbol or change documented behaviour without calling it out as breaking and discussing a major version first. Additive changes go in the `[Unreleased]` section of CHANGELOG.md. Everything under `internal/` may change freely, but its output must stay what the public tests pin.

## Design rules you must not break

A change that violates one of these is wrong even if tests pass.

1. **No cgo, ever.** Everything builds with `CGO_ENABLED=0`. The core library has one runtime dependency (`tphakala/simd`); do not add another. WAV and PCM libraries belong to the CLI only.
2. **Every assembly kernel has a pure-Go twin that is bit-identical.** `internal/simdops` selects a kernel by CPU feature at runtime; the amd64, arm64 and `other` files must agree, tier for tier. A SIMD change that alters output bits is a quality change and needs the quality tests, not just the equivalence tests.
3. **Quality is measured, not assumed.** A change to filter design, preset parameters, stage selection, phase arithmetic or interpolation is checked against the quality-regression suite (THD, SNR, aliasing, including the active sub-phase interpolation ratios). Do not loosen a threshold to make a change pass; explain it in the CHANGELOG entry.
4. **Streaming equals one-shot.** Feeding input in any chunking, then calling `Flush` once, yields the same samples as one call. `Process` and `ProcessZeroCopy` must give identical output (`process_equivalence_test.go` pins this).
5. **`Flush` is end-of-stream only.** It pushes padding through the filter to drain the tail, so calling it per chunk injects padding mid-stream and clicks at every boundary. It is called once after the last `Process`; a second `Flush` is a no-op returning an empty slice. `Reset` discards all state and is for starting an unrelated stream. Keep `doc.go` and the README true to this.
6. **Startup latency is real and reported.** The first `Process` calls withhold about `Latency()` samples, which equals the filters' group delay. `Latency()` matches the measured deficit to within about 2 samples; keep that accurate when changing filter lengths or stage layout, and keep `examples/streaming` working.
7. **Output is time-aligned.** Output sample `n` corresponds to input time `n/ratio` for every preset (a unit impulse at input `p` peaks at output `p*ratio`), as in libsoxr. Each FIR stage (`DFTStage`, `DFTDecimationStage`, `PolyphaseStage`) starts with a deterministic zero prefill and initial phase for its own group delay, so chains are aligned by composition; `Flush` pads less and truncates so the total length is unchanged. `alignment_test.go` and `internal/engine/alignment_stage_test.go` pin it. A change to filter lengths, stage layout, prefill, `Flush` padding or `Reset` must keep them passing; shifting the signal to compensate elsewhere is not a fix.
8. **`ProcessInto` does not advance state on failure.** When the output buffer is too small it returns `ErrBufferTooSmall` before touching any state, so the caller can retry with a larger buffer. `EstimateOutput` is a conservative upper bound. The warm `ProcessInto` path allocates nothing; do not add allocations to it.
9. **One channel, one instance, independent state.** `SimpleResampler` and `SimpleResamplerFloat32` handle one channel. Multi-channel state is per channel, which is what makes `EnableParallel` safe. `Process` and `Flush` on one instance must be serialized by the caller; nothing may share mutable state across channels or instances.
10. **Validate inputs at the door.** `Config.Validate` and the constructors reject NaN, non-positive and out-of-range rates and channel counts (up to 256 channels) with an error rather than a panic or a hang (`nan_validation_test.go`). Parameters reaching `internal/` code are already validated; keep it that way.
11. **float32 and float64 paths stay consistent.** `SimpleResamplerFloat32` is float32-native (`Process` and `Flush` return `[]float32`). The `Resampler` from `New` runs float64 internally and `ProcessFloat32` is a conversion wrapper. A behaviour fix in one precision path is made in the other, or the reason it does not apply is stated.
12. **Attribution and licence.** The code is LGPL-2.1-or-later and derives from libsoxr. Go files carry the SPDX header. Keep the attribution section of `doc.go` accurate when new libsoxr-derived components are added.

## Layout

```
doc.go               Package overview: features, streaming/latency/Flush contract, presets, thread safety, attribution
resample.go          Config, QualitySpec, QualityPreset, New, Resampler, MultiFlusher, Info, GetInfo
convenience.go       SimpleResampler[Float32], ProcessInto, one-shot helpers, constructors, interleave helpers
pipeline_builder.go  Stage selection and pipeline assembly for the root API
stages.go            Stage constructors that bridge to internal/engine
constant.go    the constantRateResampler returned by New (per-channel stages, parallel path)
constants.go   Shared constants

internal/engine/     resampler.go, dft_stage.go, polyphase.go, polyphase_stage.go, cubic.go, filter_params.go, stage_adapter.go
internal/pipeline/   pipeline.go, buffer.go
internal/filter/     kaiser.go
internal/mathutil/   bessel.go, constants.go
internal/simdops/    ops.go, polyphase_resample*.go and .s (amd64, arm64, other)

cmd/resample-wav/    CLI (main.go, helpers.go)
cmd/resample/        Demo and benchmark tool
test-reference/      libsoxr C reference programs, generators and README
```

## Testing approach

Tests run without audio hardware. Tests that need libsoxr reference data skip when it is not available, and the committed reference data lives in `internal/engine/testdata/soxr_reference_data.json`.

- **Regression tests.** New behaviour gets a test. A bug fix gets a regression test that fails before the fix. A test must be able to fail: break the line it pins and confirm.
- **Quality suite.** `internal/engine` holds the quality-regression, preset-comparison, anti-aliasing, soxr-comparison and severe-ratio tests; `TestQualityRegression_THD` and `TestQualityRegression_THD_ActiveInterpolation` back the README numbers. If a README table changes, the test that backs it changes with it.
- **Streaming tests.** Root-package tests cover flush lifecycle, multi-stage and multi-channel flush, flush length, chunking equivalence (`streaming_equivalence_test.go`), latency, `ProcessInto`, aliasing and parallel use.
- **Fuzzing.** `FuzzResamplerNew` and `FuzzResampleMono` (root), `FuzzDesignLowPassFilter` and `FuzzKaiserWindow` (`internal/filter`), `FuzzBesselI0`, `FuzzKaiserBeta`, `FuzzKaiserBetaWithTrBw` and `FuzzEstimateFilterLength` (`internal/mathutil`). A `-fuzz` pattern that matches nothing passes without running, so run a new target in the package that defines it. Crashers are committed under `testdata/fuzz`. CI does not run fuzzing.
- **Race.** `go test -race` covers the parallel channel path; keep new concurrency tests race-clean.
- **Benchmarks.** Benchmarks live beside the code (`internal/engine`, `internal/simdops`, root). Hot paths (convolution, the cubic kernel, `ProcessInto`) are allocation-sensitive; compare before and after with `task bench`. CI runs benchstat on pull requests and posts the comparison.
- **SIMD tiers.** Kernel changes are exercised on both amd64 and arm64 hardware where possible; the pure-Go twin is the oracle.

## Commands

Go version is read from `go.mod` (currently 1.27.0). CI pins golangci-lint in `.github/workflows/ci.yml` (currently v2.15.0); use the same major version locally.

```
task build         # build resample-wav and the demo into ./bin (GOAMD64=v3)
task test          # go test ./...
task test:verbose  # go test -v ./...
task test:race     # go test -race ./...
task test:coverage # coverage profile and HTML report
task test:soxr     # soxr comparison tests in internal/engine (builds the C reference tools first)
task bench         # all benchmarks with allocation counts
task bench:engine  # engine benchmarks only (also bench:profile and bench:compare)
task lint          # golangci-lint run ./...
task fmt           # go fmt ./...
task vet           # go vet ./...
task tidy          # go mod tidy
task check         # fmt, vet, lint, test
task release:check # tidy, check, race tests
```

Run `task check` before pushing. CI runs: lint; tests on Linux, macOS and Windows; `go test -race -short`; build of the library and both CLIs plus a `go mod tidy` cleanliness check; a Linux job that builds the libsoxr reference tools and runs the `Soxr` comparison tests and the per-preset THD, SNR and passband ripple comparisons, failing if any of them skips; coverage upload; CodeQL; govulncheck; OpenSSF Scorecard; and the PR benchmark comparison. CONTRIBUTING.md describes the PR workflow.

## Conventions

- **Lint.** Config is `.golangci.yaml`: a long explicit list of linters (gocognit at 50, `mnd`, `exhaustive`, `testifylint`, `godoclint`, `modernize`, among others), with gocritic running the ruleguard matchers in `rules/`.
- **No `//nolint` without a linter name and a reason** (`nolintlint` is on). If a linter is wrong often enough to silence globally, change `.golangci.yaml` with a comment saying why.
- **Magic numbers** go in the constants files (`constants.go`, `internal/*/constants.go`) with a comment on where the value comes from (libsoxr recipe, measurement).
- **Tests** are table-driven with `testify` where that reads well; helpers call `t.Helper()`.
- **Comments.** Explain why: the libsoxr behaviour being matched, the failure being avoided, the measurement behind a constant. Match the existing density.
- **Docs follow code.** A behaviour change updates `doc.go`, the README and the relevant godoc in the same change. A sentence that is false for some path (float32 versus float64, mono versus multi-channel) is a defect.
- **Changelog.** CHANGELOG.md follows Keep a Changelog and semantic versioning. User-visible changes get an entry under `[Unreleased]` in the same PR, with the PR number; performance claims quote the benchmark that backs them.
- **Commits** use an imperative subject with a type prefix (`feat:`, `fix:`, `perf:`, `docs:`, `build:`, `ci:`, `chore:`, `refactor:`, `test:`), optionally scoped.
- **Dependencies.** Dependabot keeps modules and Actions current; workflow actions are pinned by SHA. The default branch is `master`.

## Review guidance

For CodeRabbit, Copilot and human reviewers. Rate findings on what a caller of the library would see.

Flag these, they are real defects here:

- Any break of the design rules above: cgo, a new core dependency, a SIMD path that no longer matches its pure-Go twin, a hot-path allocation in `ProcessInto`.
- Streaming output that depends on chunk size, `Flush` semantics that change, or a `Flush` that is called implicitly mid-stream.
- A quality regression: a lowered threshold, a changed preset parameter or filter design step without quality-suite evidence, or a README quality number no test backs.
- A panic, hang or unbounded allocation reachable from `Config` values, rates, channel counts or input slice lengths.
- State shared between channels or instances, or a data race on the parallel path.
- A float32 and float64 path that diverge in behaviour without a stated reason.
- A broken `ErrBufferTooSmall` contract (state advanced on failure) or an `EstimateOutput` that under-reports.
- A test that cannot fail on the behaviour it names, or a doc sentence that is false for some path.

Intentional, do not flag:

- Output that is not bit-identical to libsoxr; quality is compared statistically, and the Low and Medium presets deliberately track libsoxr within a few dB of THD rather than beating it.
- Startup latency (the first `Process` calls returning fewer samples than expected); it is documented and exposed through `Latency()`.
- The package being named `resampler` while the README imports it as `resampling`.
- `internal/` packages exporting more than they need; they are not API.
- The C and Python material in `test-reference`, and the `cmd/resample-wav` WAV dependencies.
- Tests that skip when libsoxr reference data is unavailable.

## Things that do not belong in the repo

- Design docs, specs, plans, research notes and handoff prompts (`PLAN.md`, `SPEC.md`, `*-plan.md`, `*-spec.md`, `*-design.md`, and `docs/plans/`, `docs/specs/`, `docs/design/`, `docs/research/`, `docs/superpowers/` are gitignored on purpose).
- Benchmark result dumps (`benchmark_results*.txt`) and CPU or memory profiles; keep them in release notes or PR comments.
- Build outputs and coverage files (`*.test`, `*.out`, `*.prof`, `*.exe`, `bin/`), editor and tool state (`.claude/`, `.serena/`, `.task/`).
- Raw captures or large audio files. Committed test data is limited to small fixtures, the soxr reference JSON and fuzz corpora under `testdata/`.
