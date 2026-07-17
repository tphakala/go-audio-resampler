# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Latency()` on `SimpleResampler` and `SimpleResamplerFloat32`, returning the
  startup deficit in output samples so a caller can prime a real-time FIFO
  before the first output. (#51)
- Streaming documentation (latency, the `Flush` contract, per-channel instance
  requirements) and a real-time FIFO example at `examples/streaming`. (#51)

### Fixed

- Polyphase phase-boundary coefficient interpolation used a wrapped neighbor,
  degrading THD+N by 77 to 111 dB at ratios with active sub-phase
  interpolation (for example 44100 to 64000 or 32000 to 44100); exact-rational
  ratios such as 44100 to 48000 were unaffected.
- Severe non-integer downsampling (beyond roughly 1:16) corrupted output with
  repeated stale samples and grew internal history without bound.
- `Flush` over-padded each filter stage by one zero, emitting about 2 phantom
  samples; `Process` plus `Flush` now totals within
  `[floor(n*ratio), ceil(n*ratio)+1]`. (#51)
- At unity ratio (`inputRate == outputRate`), `Process` returned the caller's
  own input slice; it now returns an owned buffer.
- Cubic (`QualityQuick`) resampling computed its first output segments from a
  fictional zero history and never emitted the final segments; output is now
  aligned to real data, with the first output after 2 input samples.
- NaN sample rates are now rejected by constructors.
- Half-band stage construction errors now propagate instead of silently
  substituting a nearest-neighbor stub.
- `GetLatency` now accounts for decimation and cubic stages.

### Changed

- `Flush` is now terminal: a second `Flush` returns an empty slice, and a
  `Process` call after `Flush` starts a fresh stream instead of convolving
  against leftover padding. (#51)
- `QualityQuick` through `NewEngine` and `NewEngineFloat32` now uses cubic
  interpolation (matching `New()` and the documented contract) instead of a
  full FIR pipeline; latency drops accordingly.

## [1.4.0] - 2026-05-29

### Added

- Zero-allocation float32 streaming. `SimpleResamplerFloat32` (from `NewEngineFloat32`)
  now has `ProcessInto(input, output []float32) (int, error)` and
  `EstimateOutput(inputLen int) int`. The engine is float32-native, so this path
  has no float64 round-trip and reports 0 allocs/op once warm. The `New(config)`
  path gains the float32 counterpart `ProcessFloat32Into`, which reuses grow-only
  scratch buffers for the conversion. Both produce output bit-identical to
  `Process` / `ProcessFloat32` and return `ErrBufferTooSmall` before advancing
  state, so a too-small call can be retried safely. (#31, fixes #28)
- `FlushMulti() ([][]float64, error)` on a new optional `MultiFlusher` interface,
  draining every channel's pipeline independently after `ProcessMulti`. Exposed as
  an optional interface (matching the existing `infoProvider` pattern) so it does
  not break external implementors of `Resampler`. (#41, #39)

### Fixed

- Multi-stage pipelines now flush inter-stage tails correctly. Each stage's pending
  input, including the previous stage's flushed tail, is processed through the stage
  before its delay line is drained, so the tail propagates all the way to the final
  output. Polyphase over-padding was also removed. (#40, #37, #30)
- `Flush()` previously drained only channel 0, silently dropping the delay-line
  tails of channels 1..N-1 after `ProcessMulti`. Use `FlushMulti` for multi-channel
  streams. (#41, #39)

### Changed

- Removed the `gonum.org/v1/gonum` dependency. The DFT pre-stage is FIR-polyphase
  based and no longer uses an external FFT, so `github.com/tphakala/simd` is now the
  core library's only runtime dependency. (#29)
- Stereo processing reuses a single engine and reads all available samples in the
  per-channel path; a dead mutex was removed. (#38, #32, #33, #34)
- Engine internals: dead-code cleanup, warm-path zero-allocation, hot-loop
  bounds-check elimination, and a split of the oversized polyphase source. No public
  API change. (#29)
- Bumped `github.com/tphakala/simd` to v1.1.0. (#20)

## [1.3.0] - 2026-05-04

- Zero-allocation float64 streaming via caller-owned output buffers:
  `SimpleResampler.ProcessInto` and `EstimateOutput`.

## [1.2.0] - 2026-03-15

- Security, fuzzing, and CI hardening: fuzz tests, Kaiser window overflow fix for
  extreme parameters, OpenSSF Scorecard compliance, and Dependabot auto-merge.

## [1.1.0] - 2025-11-26

- Float32-native API: `SimpleResamplerFloat32`, `ResampleMonoFloat32`, and
  `ResampleStereoFloat32` for consistent float32 workflows.

## [1.0.1] - 2025-11-25

- Bug fixes.

## [1.0.0] - 2025-11-24

- Initial release. Pure Go, multi-stage polyphase FIR resampling with Kaiser window
  design, quality presets, multi-channel and streaming support, validated against
  libsoxr.

[Unreleased]: https://github.com/tphakala/go-audio-resampler/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/tphakala/go-audio-resampler/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tphakala/go-audio-resampler/releases/tag/v1.0.0
