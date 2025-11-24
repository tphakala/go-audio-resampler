# go-audio-resampler

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/go-audio-resampler.svg)](https://pkg.go.dev/github.com/tphakala/go-audio-resampler)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/go-audio-resampler)](https://goreportcard.com/report/github.com/tphakala/go-audio-resampler)


High-quality audio resampling library for Go, based on [libsoxr](https://sourceforge.net/projects/soxr/) (the SoX Resampler library) by Rob Sykes.

The library implements polyphase FIR filtering with Kaiser window design for professional-grade sample rate conversion, following the multi-stage architecture that makes libsoxr one of the highest-quality resamplers available.

## Features

- **Multiple quality presets** - From quick/low-latency to very high quality mastering
- **Polyphase FIR filtering** - Efficient multi-stage resampling architecture
- **Kaiser window design** - Optimal stopband attenuation with configurable parameters
- **SIMD acceleration** - Optional AVX2/SSE optimizations via [tphakala/simd](https://github.com/tphakala/simd)
- **Multi-channel support** - Process stereo, surround, and multi-channel audio
- **Streaming API** - Process audio in chunks with proper state management
- **Pure Go** - No CGO dependencies, cross-platform compatible

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

| Preset            | Precision | Use Case                        |
| ----------------- | --------- | ------------------------------- |
| `QualityQuick`    | 8-bit     | Preview, real-time with low CPU |
| `QualityLow`      | 16-bit    | Speech, low-bandwidth audio     |
| `QualityMedium`   | 16-bit    | General music playback          |
| `QualityHigh`     | 24-bit    | Studio production, streaming    |
| `QualityVeryHigh` | 32-bit    | Mastering, archival             |

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

## Architecture

The library implements a multi-stage resampling architecture similar to libsoxr:

```
Input -> [DFT Pre-Stage] -> [Polyphase FIR] -> Output
              (2x)            (fine ratio)
```

- **Integer ratios** (2x, 3x, etc.): Single DFT stage for efficiency
- **Non-integer ratios** (44.1kHz→48kHz): DFT pre-upsampling + polyphase stage
- **Polyphase decomposition**: Reduces computation by processing only needed output phases
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
    InputRate    float64      // Input sample rate in Hz
    OutputRate   float64      // Output sample rate in Hz
    Channels     int          // Number of audio channels
    Quality      QualitySpec  // Quality settings
    MaxInputSize int          // Optional buffer size hint
    EnableSIMD   bool         // Enable SIMD optimizations
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
resample-wav -rate 48 input.wav output.wav           # Resample to 48kHz
resample-wav -rate 16 -quality high speech.wav out.wav  # Downsample speech
resample-wav -rate 96 music.wav hires.wav            # Upsample to hi-res
```

### resample (demo)

Interactive demo showing library capabilities:

```bash
go build -o bin/resample ./cmd/resample
./bin/resample -demo
```

## Performance

Benchmarks comparing different quality levels (44.1kHz → 48kHz, mono):

| Quality  | Throughput    | Filter Taps | Latency     |
| -------- | ------------- | ----------- | ----------- |
| Quick    | ~50x realtime | 4           | 2 samples   |
| Medium   | ~20x realtime | 64          | 32 samples  |
| High     | ~10x realtime | 128         | 64 samples  |
| VeryHigh | ~5x realtime  | 256         | 128 samples |

_Actual performance varies by CPU and SIMD support._

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
