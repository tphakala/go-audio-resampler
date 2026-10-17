# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Behaviour change: non-integer downsampling now anti-aliases.** Before, a
  conversion such as 48000 to 32000, 48000 to 44100 or 384000 to 256000 had no
  lowpass at the output Nyquist: a tone between the output Nyquist and the input
  Nyquist passed at 0 dB and folded straight into the output band, and a 120 dB
  `sox spectrogram` of a 384000 to 256000 `QualityHigh` clip differed from
  `sox rate` in 27 percent of its pixels (raster-equal 0.7285). The 2x DFT
  pre-stage is now designed as libsoxr designs it (`cr.c:383-387`), as a lowpass
  with its stopband at the output Nyquist, using the same recipe as the integer
  decimation stage; the polyphase stage after it still only rejects images.
  Measured alias rejection (single tones from 1.001 to 1.5 times the output
  Nyquist, float64): 0 dB before; now -108 (Low), -110 (Medium), -135 (High), -180
  (engine VeryHigh), -157 (24-bit) and -204 dB (32-bit), each at or past the
  preset's stopband. The passband edge matches the integer decimation path (High:
  -3 dB at 0.953 of the output Nyquist; libsoxr HQ 0.950) and is flat to 0.01 dB
  up to 0.9 of it for Medium and above. The same 120 dB spectrogram is now
  raster-equal 0.9966 with 26 of 132098 pixels more than two palette steps
  apart.
  - **Conversions that change:** non-integer downsampling on every entry point
    (`NewEngine`, `NewEngineFloat32`, `New`, `ResampleMono*`, `ResampleStereo*`,
    `cmd/resample-wav`), and `New` downsampling by any ratio that is not a power
    of two, including integer ones such as 48000 to 16000 (`New` factors out 2:1
    half-band stages and hands the non-integer remainder to the engine).
    Output samples change, and content between roughly 0.95 of the output
    Nyquist and the input Nyquist is now removed instead of folded.
  - **Latency grows.** `Latency()` and `GetLatency()` follow the longer filter and
    still match the measured startup deficit. `NewEngine`, High, 48000 to 44100:
    89 to 201; 384000 to 256000: 66 to 198; `QualityMedium` 49 to 152 (384000 to
    256000); `QualityLow` falls (68 to 48 at 48000 to 44100). On `New`,
    48000 to 44100 High goes 105 to 241 and VeryHigh 135 to 319. Hard-coded
    `Latency()` values break. The startup transient also lasts longer in input
    samples, so a first small chunk can return no output; a FIFO primed with
    `Latency()` samples still self-heals as documented. `GetInfo()` reports the
    longer filter length and memory use.
  - **CPU cost rises for the affected conversions.** Measured with
    `go test -bench` and benchstat (10 runs on amd64, 5 on arm64): the float32
    `ProcessInto` benchmark `BenchmarkProcessIntoFloat32_Warm_384to256_High`
    (65536-sample chunks, `QualityHigh`) goes from 6.24 ms to 14.49 ms per chunk
    on a Raspberry Pi 5 (about 37 ms to 85 ms of CPU per second of audio, still
    about 12x real time on one core) and from 1.70 ms to 3.36 ms on an i7-1260P
    (noisy, 10 runs, +98 percent). `Throughput_{Medium,High,VeryHigh}_48kTo32k`
    rise 2.5x on arm64 (Pi 5: 6.09, 7.94, 10.98 ms to 15.45, 20.16, 28.32 ms).
    `QualityLow` and `QualityQuick`, upsampling and integer decimation are
    unchanged (`Throughput_Low_48kTo32k` is 4.7 percent faster on the Pi), and
    `ProcessInto` still allocates nothing once warm.
  - **Ratio limit.** The pre-stage filter grows about linearly with the
    input/output ratio and is capped at 8191 taps, so the preset's stopband is
    only guaranteed up to the ratio that cap supports. Measured (`NewEngine`,
    float64): `QualityHigh` meets 126 dB at 11.0:1 (-134.6 dB) and 17.4:1 (-131.5
    dB) but reaches only -42.4 dB at 34.8:1 (384000 to 11025); `QualityMedium`
    is -109.3 dB, -109.5 dB and -55.8 dB at the same three ratios. This was 0 dB
    before. Ratios between 17.4:1 and 34.8:1 are not measured. `New` never hands
    the engine more than 2:1. Integer decimation is limited by the same cap
    (384000 to 8000, High: -89.9 dB measured).
  - **`New` at `QualityMedium`** reaches the engine's Low filter (the 16-bit
    Medium preset maps to `engine.QualityLow`), so it is -3 dB at 0.824 of the
    output Nyquist, while `NewEngine` Medium is flat to 0.01 dB up to 0.9 and -3 dB at 0.951. This mapping is
    unchanged and tracked in #76.
  - **Unchanged:** `QualityQuick` (cubic only, no anti-alias filter when
    downsampling, as libsoxr QQ), unity conversions, upsampling, integer
    decimation on the engine path (all pinned bit-for-bit by new golden
    checksums captured on amd64 and arm64 hardware before the change), total
    output length for any input length and chunking, and streaming equals
    one-shot.

## [1.8.0] - 2026-10-17

### Changed

- **Behaviour change: output is now time-aligned.** For `QualityLow`,
  `QualityMedium`, `QualityHigh` and `QualityVeryHigh` on every entry point
  (`NewEngine`, `NewEngineFloat32`, `New`, `ResampleMono` and the other one-shot
  helpers), output sample `n` now corresponds to input time `n/ratio`, as in
  libsoxr and `sox rate`. Before, every FIR stage emitted its first sample only
  once it held a full window of real input, so the signal ran early by one
  filter group delay: for example 48000 to 24000 `QualityHigh` was 187.5 output
  samples early. Each FIR stage now starts with a deterministic zero prefill and
  initial phase for its own group delay, so a unit impulse at input sample `p`
  peaks at output sample `p*ratio` (within 0.005 input samples), and the first
  samples of a stream carry the filter's onset ramp. `Latency()` and
  `GetLatency()` are now the filters' group delay, roughly half their previous
  value (48000 to 24000 `QualityHigh`: 375 to 188), and still match the measured
  startup deficit. Total output length is unchanged for any input length and
  chunking, streaming still equals one-shot bit for bit, `QualityQuick` and unity
  conversions are bit-identical to before, and `ProcessInto` still allocates
  nothing once warm. Callers that shifted the output to compensate for the old
  early signal must stop doing so, and anything that hard-coded a `Latency()`
  value must read it at run time ([#77](https://github.com/tphakala/go-audio-resampler/pull/77)).

## [1.7.0] - 2026-08-06

### Changed

- `Process` now runs the same zero-copy stage pipeline as `ProcessZeroCopy` and
  copies only the final result into caller-owned memory, instead of copying
  every intermediate stage output. On multi-stage conversions (an upsampling
  pre-stage feeding a decimation or polyphase stage) this drops the intermediate
  copy and its allocation: the 44100 to 48000 `QualityHigh` benchmark goes from
  2 to 1 allocation per call and from 25.3 to 9.3 KiB/op (-63%), about 9% faster,
  with output bit-identical to before. Public API and behavior are unchanged; a
  new equivalence test pins `Process` output to `ProcessZeroCopy` across the
  rational, integer-decimation, and cubic paths. (#66)
- Bumped `github.com/tphakala/simd` to v1.8.0 (from v1.6.0). v1.7.0 and v1.8.0
  are additive, drop-in upgrades; the resampler consumes only the `cpu`, `f32`,
  and `f64` primitive groups, which are unchanged, so resampled output stays
  bit-identical tier for tier. (#65)

## [1.6.0] - 2026-08-05

### Changed

- Cubic (`QualityQuick`) resampling now runs through a fused, whole-block
  polyphase kernel (`ResampleCubic32` / `ResampleCubic64` in `internal/simdops`)
  instead of a per-output `CubicInterpDot` loop, with AVX+FMA (amd64) and NEON
  (arm64) kernels under the same CPU-feature gate as the `CubicInterpDot`
  primitive they build on. The block result is bit-identical to the previous
  per-output loop on every CPU, tier for tier, and the path still allocates
  nothing. The isolated kernel benchmark shows roughly 1.6x (64 taps) to 2.4x
  (20 taps) on the inner cubic dot; on the full pipeline the end-to-end delta is
  within run-to-run noise. No public API change. (#63)
- Bumped `github.com/tphakala/simd` to v1.6.0. (#58)

## [1.5.0] - 2026-07-17

### Added

- `Latency()` on `SimpleResampler` and `SimpleResamplerFloat32`, returning the
  startup deficit in output samples so a caller can prime a real-time FIFO
  before the first output. (#51)
- Streaming documentation (latency, the `Flush` contract, per-channel instance
  requirements) and a real-time FIFO example at `examples/streaming`. (#51)

### Fixed

- Polyphase phase-boundary coefficient interpolation used a wrapped neighbor,
  degrading THD+N at ratios with active sub-phase interpolation. The committed
  measurement shows an 86.26 dB improvement (wrapped -54.46 dB versus flat
  -140.72 dB) at 44100 to 64000, with similar magnitude at other
  active-interpolation ratios measured during the investigation but not
  committed as tests; exact-rational ratios such as 44100 to 48000 were
  unaffected. (#51)
- Severe non-integer downsampling (beyond roughly 1:16) corrupted output with
  repeated stale samples and grew internal history without bound. (#51)
- `Flush` over-padded each filter stage by one zero, emitting about 2 phantom
  samples; `Process` plus `Flush` now totals within
  `[floor(n*ratio), ceil(n*ratio)+1]`. (#51)
- At unity ratio (`inputRate == outputRate`), `Process` returned the caller's
  own input slice; it now returns an owned buffer. (#51)
- Cubic (`QualityQuick`) resampling computed its first output segments from a
  fictional zero history and never emitted the final segments; output is now
  aligned to real data, with the first output after 2 input samples. Its
  `Process` also returned an aliased empty slice during priming; it now returns
  an owned buffer. (#51)
- NaN sample rates are now rejected by all constructors, covering both the
  `NewEngine`/`NewEngineFloat32` engine path and the `New(config)` pipeline
  path (`New`, `NewMultiChannel`, `NewStereo`, `NewSimple`, and the preset
  helpers). (#51)
- Half-band stage construction errors now propagate instead of silently
  substituting a nearest-neighbor stub. (#51)
- `GetLatency` now accounts for decimation and cubic stages. (#51)
- `GetLatency` on the `New(config)` path now reports the measured startup
  deficit in output samples; it previously mixed rate domains across pipeline
  stages and mis-reported multi-stage ratios (672 reported vs 703 measured at
  44100 to 96000 QualityHigh; now within 2 samples). (#52)

### Changed

- `Flush` is now terminal: a second `Flush` returns an empty slice, and a
  `Process` call after `Flush` starts a fresh stream instead of convolving
  against leftover padding. (#51)
- `QualityQuick` through `NewEngine` and `NewEngineFloat32` now uses cubic
  interpolation (matching `New()` and the documented contract) instead of a
  full FIR pipeline; latency drops accordingly. (#51)

### Removed

- The internal duplicate polyphase filter design path
  (`internal/filter/polyphase.go`) and the `cmd/analyze-filter` diagnostic
  that consumed it; the engine's design code in `internal/engine` is the
  single polyphase implementation. (#55)
- The production-dead `LinearStage` from `internal/engine`; only its own
  tests referenced it. (#55)

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

[Unreleased]: https://github.com/tphakala/go-audio-resampler/compare/v1.8.0...HEAD
[1.8.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/tphakala/go-audio-resampler/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/tphakala/go-audio-resampler/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tphakala/go-audio-resampler/releases/tag/v1.0.0
