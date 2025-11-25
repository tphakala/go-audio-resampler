# go-audio-resampler

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/go-audio-resampler.svg)](https://pkg.go.dev/github.com/tphakala/go-audio-resampler)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/go-audio-resampler)](https://goreportcard.com/report/github.com/tphakala/go-audio-resampler)

The **only pure Go** high-quality audio resampling library (that I know of) - **no CGO dependencies** required.

Based on [libsoxr](https://sourceforge.net/projects/soxr/) (the SoX Resampler library) by Rob Sykes, this is a complete Go reimplementation delivering professional-grade quality without any C code or external dependencies.

The library implements polyphase FIR filtering with Kaiser window design for professional-grade sample rate conversion, following the multi-stage architecture that makes libsoxr one of the highest-quality resamplers available.

## Features

- **100% Pure Go** - No CGO, no external C libraries, no build complexity. Just `go get` and build anywhere
- **Cross-platform** - Works on Linux, macOS, Windows, ARM, WebAssembly - anywhere Go runs
- **Multiple quality presets** - From quick/low-latency to very high quality mastering (8-bit to 32-bit precision)
- **Selectable precision** - Generic float32/float64 processing paths with type-safe SIMD operations
- **Polyphase FIR filtering** - Efficient multi-stage resampling architecture with cubic coefficient interpolation
- **Kaiser window design** - Optimal stopband attenuation with configurable parameters
- **SIMD acceleration** - AVX2/SSE optimizations via [tphakala/simd](https://github.com/tphakala/simd) for both float32 and float64
- **Multi-channel support** - Process stereo, surround, and multi-channel audio (up to 256 channels)
- **Parallel channel processing** - ~1.7x speedup for stereo, up to 8x for 7.1 surround audio
- **Streaming API** - Process audio in chunks with proper state management
- **soxr-quality algorithms** - Implements libsoxr's multi-stage architecture for professional-grade quality

## Installation

```bash
go get github.com/tphakala/go-audio-resampler
```

Requires Go 1.25 or later.

## Quick Start

### Simple One-Shot Resampling

```go
package main

import (
    "fmt"
    "log"

    resampling "github.com/tphakala/go-audio-resampler"
)

func main() {
    // Resample mono audio from 44.1kHz to 48kHz
    input := generateSineWave(44100, 1000, 1.0) // 1 second of 1kHz sine

    output, err := resampling.ResampleMono(input, 44100, 48000, resampling.QualityHigh)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Resampled %d samples to %d samples\n", len(input), len(output))
}
```

### Streaming Resampling

```go
package main

import (
    "log"

    resampling "github.com/tphakala/go-audio-resampler"
)

func main() {
    // Create a resampler for CD to DAT conversion
    config := &resampling.Config{
        InputRate:  44100,
        OutputRate: 48000,
        Channels:   2,
        Quality:    resampling.QualitySpec{Preset: resampling.QualityHigh},
    }

    r, err := resampling.New(config)
    if err != nil {
        log.Fatal(err)
    }

    // Process audio in chunks
    for chunk := range audioChunks {
        output, err := r.Process(chunk)
        if err != nil {
            log.Fatal(err)
        }
        writeOutput(output)
    }

    // Flush remaining samples
    final, _ := r.Flush()
    writeOutput(final)
}
```

### Convenience Functions

```go
// Common sample rate conversions
r, _ := resampling.NewCDtoDAT(resampling.QualityHigh)    // 44.1kHz -> 48kHz
r, _ := resampling.NewDATtoCD(resampling.QualityHigh)    // 48kHz -> 44.1kHz
r, _ := resampling.NewCDtoHiRes(resampling.QualityHigh)  // 44.1kHz -> 88.2kHz

// Simple mono/stereo resamplers
r, _ := resampling.NewSimple(inputRate, outputRate)
r, _ := resampling.NewStereo(inputRate, outputRate, resampling.QualityMedium)
r, _ := resampling.NewMultiChannel(inputRate, outputRate, 6, resampling.QualityHigh)

// Direct engine access for maximum performance
r, _ := resampling.NewEngine(44100, 48000, resampling.QualityHigh)
```

### Stereo Processing

```go
// Process stereo as separate channels
leftOut, rightOut, err := resampling.ResampleStereo(
    leftChannel, rightChannel,
    44100, 48000,
    resampling.QualityHigh,
)

// Interleave/deinterleave helpers
interleaved := resampling.InterleaveToStereo(left, right)
left, right := resampling.DeinterleaveFromStereo(interleaved)
```

## Quality Presets

| Preset            | Precision | Attenuation | Use Case                        |
| ----------------- | --------- | ----------- | ------------------------------- |
| `QualityQuick`    | 8-bit     | ~54 dB      | Preview, real-time with low CPU |
| `QualityLow`      | 16-bit    | ~102 dB     | Speech, low-bandwidth audio     |
| `QualityMedium`   | 16-bit    | ~102 dB     | General music playback          |
| `QualityHigh`     | 20-bit    | ~126 dB     | Studio production, streaming    |
| `QualityVeryHigh` | 28-bit    | ~175 dB     | Mastering, archival             |

### Quality Validation vs libsoxr

The following THD (Total Harmonic Distortion) measurements compare this library against libsoxr reference implementation for 44.1kHz → 48kHz conversion at 1kHz test tone:

| Preset   | libsoxr THD | Go THD     | Difference |
| -------- | ----------- | ---------- | ---------- |
| Low      | -146.25 dB  | -145.17 dB | +1.1 dB    |
| Medium   | -130.61 dB  | -145.17 dB | -14.6 dB ✓ |
| High     | -155.19 dB  | -156.73 dB | -1.5 dB ✓  |
| VeryHigh | -162.22 dB  | -162.19 dB | +0.03 dB   |

_Negative difference means Go implementation has better THD (lower distortion)._

**Key findings:**

- **High and VeryHigh presets match or exceed libsoxr quality** with THD differences under 2 dB
- All presets achieve SNR (Signal-to-Noise Ratio) matching libsoxr within measurement tolerance
- Downsampling (e.g., 48kHz → 32kHz) often shows even better performance due to optimized anti-aliasing filters

### Custom Quality Settings

```go
config := &resampling.Config{
    InputRate:  44100,
    OutputRate: 48000,
    Channels:   1,
    Quality: resampling.QualitySpec{
        Preset:        resampling.QualityCustom,
        Precision:     24,          // Bits of precision
        PhaseResponse: 50,          // 0=minimum, 50=linear, 100=maximum
        PassbandEnd:   0.95,        // Preserve up to 95% of Nyquist
        StopbandBegin: 0.99,        // Attenuate above 99% of Nyquist
    },
}
```

## Precision Modes (float32 vs float64)

The library supports both float32 and float64 processing through Go generics:

```go
// float64 for maximum precision (default)
r64, _ := engine.NewResampler[float64](44100, 48000, engine.QualityHigh)
output64, _ := r64.Process(input64)

// float32 for ~2x SIMD throughput
r32, _ := engine.NewResampler[float32](44100, 48000, engine.QualityHigh)
output32, _ := r32.Process(input32)
```

**Precision comparison (44.1kHz → 48kHz, High quality):**

| Mode    | THD @ 1kHz | Use Case                          |
| ------- | ---------- | --------------------------------- |
| float64 | -144.95 dB | Maximum precision, critical audio |
| float32 | -144.96 dB | Higher throughput, general use    |

Both modes achieve equivalent quality for most use cases. Use float32 when processing large amounts of audio data where throughput is more important than theoretical precision limits.

## Architecture

The library implements a multi-stage resampling architecture similar to libsoxr:

```text
Input -> [DFT Pre-Stage] -> [Polyphase FIR] -> Output
              (2x)            (fine ratio)
```

- **Integer ratios** (2x, 3x, etc.): Single DFT stage for efficiency
- **Non-integer ratios** (44.1kHz→48kHz): DFT pre-upsampling + polyphase stage
- **Polyphase decomposition**: Reduces computation by processing only needed output phases
- **Cubic coefficient interpolation**: Sub-phase interpolation for excellent high-frequency THD
- **Kaiser window**: Optimal FIR filter design with configurable stopband attenuation

## API Reference

### Core Types

```go
// Main resampler interface
type Resampler interface {
    Process(input []float64) ([]float64, error)
    ProcessFloat32(input []float32) ([]float32, error)
    ProcessMulti(input [][]float64) ([][]float64, error)
    Flush() ([]float64, error)
    GetLatency() int
    Reset()
    GetRatio() float64
}

// Configuration
type Config struct {
    InputRate      float64      // Input sample rate in Hz
    OutputRate     float64      // Output sample rate in Hz
    Channels       int          // Number of audio channels
    Quality        QualitySpec  // Quality settings
    MaxInputSize   int          // Optional buffer size hint
    EnableSIMD     bool         // Enable SIMD optimizations
    EnableParallel bool         // Enable parallel channel processing
}
```

### Getting Resampler Info

```go
r, _ := resampling.New(config)
info := resampling.GetInfo(r)

fmt.Printf("Algorithm: %s\n", info.Algorithm)
fmt.Printf("Filter length: %d taps\n", info.FilterLength)
fmt.Printf("Latency: %d samples\n", info.Latency)
fmt.Printf("Memory: %d bytes\n", info.MemoryUsage)
fmt.Printf("SIMD: %v (%s)\n", info.SIMDEnabled, info.SIMDType)
```

## Command-Line Tools

### resample-wav

Resample WAV audio files:

```bash
# Build
go build -o bin/resample-wav ./cmd/resample-wav

# Usage
resample-wav -rate 48 input.wav output.wav              # Resample to 48kHz
resample-wav -rate 16 -quality high speech.wav out.wav  # Downsample speech
resample-wav -rate 96 music.wav hires.wav               # Upsample to hi-res
resample-wav -rate 48 -fast input.wav output.wav        # Float32 precision (~40% faster)
resample-wav -rate 48 -parallel=false input.wav out.wav # Disable parallel processing
```

Parallel channel processing is enabled by default for stereo/multichannel files.

### resample (demo)

Interactive demo showing library capabilities:

```bash
go build -o bin/resample ./cmd/resample
./bin/resample -demo
```

## Performance

Filter complexity comparison for 44.1kHz → 48kHz conversion:

| Quality  | DFT Stage          | Polyphase Stage     | Total Taps |
| -------- | ------------------ | ------------------- | ---------- |
| Low      | 132 taps × 2 phase | 32 taps × 80 phase  | ~2820      |
| Medium   | 132 taps × 2 phase | 32 taps × 80 phase  | ~2820      |
| High     | 166 taps × 2 phase | 64 taps × 80 phase  | ~5452      |
| VeryHigh | 166 taps × 2 phase | 100 taps × 80 phase | ~8332      |

_Actual performance varies by CPU. The [simd](https://github.com/tphakala/simd) library automatically uses AVX2/SSE when available. Setting `GOAMD64=v3` can provide minor additional speedup for non-SIMD code paths._

### Parallel Channel Processing

For multi-channel audio, enabling parallel processing provides significant speedup:

```go
config := &resampling.Config{
    InputRate:      44100,
    OutputRate:     48000,
    Channels:       2,
    Quality:        resampling.QualitySpec{Preset: resampling.QualityHigh},
    EnableParallel: true,  // Process channels concurrently
}
```

| Configuration | Processing Time | Speedup |
| ------------- | --------------- | ------- |
| Stereo (sequential) | 15.7 ms | 1.0x |
| Stereo (parallel) | 9.4 ms | 1.67x |
| 5.1 Surround (parallel) | ~17.1 ms | ~5.5x vs sequential |
| 7.1 Surround (parallel) | ~20.0 ms | ~7.0x vs sequential |

_Benchmark: 1 second of audio, 44.1kHz → 48kHz, High quality, Intel i7-1260P_

Parallel processing is safe because each channel maintains independent filter state. Mono audio is unaffected (no channels to parallelize).

## Dependencies

- [github.com/tphakala/simd](https://github.com/tphakala/simd) - SIMD operations for filter convolution
- [gonum.org/v1/gonum](https://gonum.org) - FFT for DFT stages
- [github.com/go-audio/wav](https://github.com/go-audio/wav) - WAV file I/O (CLI tools only)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `go test ./...` passes
5. Submit a pull request

## License

This library is licensed under the **GNU Lesser General Public License v2.1 (LGPL-2.1)**.

See the [LICENSE](LICENSE) file for the full license text.

## Acknowledgments

This library is based on [libsoxr](https://sourceforge.net/projects/soxr/) by Rob Sykes (licensed under LGPL-2.1). The following were derived from libsoxr:

- Multi-stage resampling pipeline (DFT pre-stage + polyphase)
- Polyphase filter bank decomposition
- Quality preset parameters and filter specifications
- Integer phase arithmetic for polyphase interpolation

The Kaiser window filter design follows well-established DSP literature, particularly the work of James Kaiser on optimal window functions.

We gratefully acknowledge Rob Sykes' excellent work on libsoxr.
