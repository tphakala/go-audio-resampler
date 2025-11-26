// Package resampler provides the only pure Go high-quality audio resampling library.
//
// This library is a complete Go reimplementation based on libsoxr (the SoX Resampler
// library) by Rob Sykes, delivering professional-grade sample rate conversion with
// no CGO dependencies. It works anywhere Go runs: Linux, macOS, Windows, ARM,
// WebAssembly, and more.
//
// # Features
//
//   - 100% Pure Go - No CGO, no external C libraries, cross-compiles effortlessly
//   - Multiple quality presets from quick/low-latency to very high quality mastering
//   - Selectable float32/float64 precision paths with type-safe SIMD operations
//   - Polyphase FIR filtering with cubic coefficient interpolation
//   - Kaiser window design for optimal stopband attenuation
//   - SIMD acceleration (AVX2/SSE) via github.com/tphakala/simd for both precisions
//   - Multi-channel support for stereo, surround, and multi-channel audio (up to 256 channels)
//   - Parallel channel processing for ~1.7x speedup on stereo, up to 8x on 7.1 surround
//   - Streaming API for processing audio in chunks with proper state management
//   - Quality validated against libsoxr reference implementation
//
// # Quick Start
//
// For simple one-shot resampling (float64):
//
//	output, err := resampler.ResampleMono(input, 44100, 48000, resampler.QualityHigh)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// For float32 audio data (common in real-time applications):
//
//	output, err := resampler.ResampleMonoFloat32(input, 44100, 48000, resampler.QualityHigh)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// For streaming resampling with a reusable resampler:
//
//	config := &resampler.Config{
//	    InputRate:  44100,
//	    OutputRate: 48000,
//	    Channels:   2,
//	    Quality:    resampler.QualitySpec{Preset: resampler.QualityHigh},
//	}
//	r, err := resampler.New(config)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Process audio chunks
//	for chunk := range audioChunks {
//	    output, err := r.Process(chunk)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    writeOutput(output)
//	}
//
//	// Flush remaining samples
//	final, _ := r.Flush()
//
// # Quality Presets
//
// The library provides several quality presets for common use cases:
//
//   - [QualityQuick]: 8-bit precision (~54 dB), lowest CPU usage. Suitable for preview
//     or real-time with low CPU.
//   - [QualityLow]: 16-bit precision (~102 dB). Good for speech and low-bandwidth audio.
//   - [QualityMedium]: 16-bit precision (~102 dB). Suitable for general music playback.
//   - [QualityHigh]: 20-bit precision (~126 dB). Recommended for studio production and
//     high-quality streaming. Matches libsoxr HQ preset.
//   - [QualityVeryHigh]: 28-bit precision (~175 dB). For mastering and archival applications.
//     Matches libsoxr VHQ preset.
//
// Custom quality settings can be specified using [QualitySpec] with
// [QualityCustom] preset.
//
// # Convenience Functions
//
// The package provides convenience constructors for common sample rate conversions:
//
//   - [NewCDtoDAT]: 44.1kHz to 48kHz (CD to DAT/DVD)
//   - [NewDATtoCD]: 48kHz to 44.1kHz (DAT/DVD to CD)
//   - [NewCDtoHiRes]: 44.1kHz to 88.2kHz (CD to high-resolution)
//   - [NewSimple]: Simple mono resampler with high quality defaults
//   - [NewStereo]: Stereo resampler with configurable quality
//   - [NewMultiChannel]: Multi-channel resampler
//
// # Float32 vs Float64 Precision
//
// The library provides two precision paths:
//
// Float64 (default): Maximum precision for mastering, archival, and critical applications.
// Use [NewEngine], [SimpleResampler], [ResampleMono], and [ResampleStereo].
//
// Float32: ~2x SIMD throughput with 32-bit precision. Ideal for real-time applications,
// game audio, streaming, and when memory bandwidth is a concern.
// Use [NewEngineFloat32], [SimpleResamplerFloat32], [ResampleMonoFloat32], and [ResampleStereoFloat32].
//
// The float32 path provides a fully consistent API where both Process and Flush
// return []float32, eliminating type conversion overhead:
//
//	r, err := resampler.NewEngineFloat32(44100, 48000, resampler.QualityHigh)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for chunk := range audioChunks {
//	    output, _ := r.Process(chunk)  // []float32 in, []float32 out
//	    writeOutput(output)
//	}
//	final, _ := r.Flush()  // Returns []float32 (not []float64!)
//
// Helper functions for float32 stereo interleaving are also provided:
// [InterleaveToStereoFloat32] and [DeinterleaveFromStereoFloat32].
//
// # Architecture
//
// The library implements a multi-stage resampling architecture similar to libsoxr:
//
//	Input -> [DFT Pre-Stage] -> [Polyphase FIR] -> Output
//	              (2x)            (fine ratio)
//
// Integer ratios (2x, 3x) use efficient single-stage processing, while
// non-integer ratios (44.1kHz→48kHz) use DFT pre-upsampling combined with
// a polyphase stage. Polyphase decomposition reduces computation by processing
// only the needed output phases. Cubic coefficient interpolation provides
// smooth sub-phase transitions for excellent high-frequency THD performance.
//
// # Stereo Processing
//
// For stereo audio, use [ResampleStereo] for one-shot processing or configure
// a multi-channel resampler:
//
//	leftOut, rightOut, err := resampler.ResampleStereo(
//	    leftChannel, rightChannel,
//	    44100, 48000,
//	    resampler.QualityHigh,
//	)
//
// For float32 stereo audio, use [ResampleStereoFloat32]:
//
//	leftOut, rightOut, err := resampler.ResampleStereoFloat32(
//	    leftChannel, rightChannel,
//	    44100, 48000,
//	    resampler.QualityHigh,
//	)
//
// Helper functions [InterleaveToStereo] and [DeinterleaveFromStereo] are provided
// for converting between interleaved and planar audio formats. Float32 variants
// [InterleaveToStereoFloat32] and [DeinterleaveFromStereoFloat32] are also available.
//
// # Parallel Processing
//
// For multi-channel audio (stereo, 5.1, 7.1), channels can be processed
// concurrently for significant performance gains:
//
//	config := &resampling.Config{
//	    InputRate:      44100,
//	    OutputRate:     48000,
//	    Channels:       2,
//	    Quality:        resampling.QualitySpec{Preset: resampling.QualityHigh},
//	    EnableParallel: true,  // Process channels concurrently
//	}
//
// Parallel processing is safe because each channel maintains independent state.
// Benchmarks show ~1.7x speedup for stereo and up to 8x for 7.1 surround audio.
// Mono audio is unaffected as there are no channels to parallelize.
//
// # Thread Safety
//
// Individual [Resampler] instances are safe for concurrent use by multiple
// goroutines when processing different channels via [Resampler.ProcessMulti].
// However, calls to [Resampler.Process] and [Resampler.Flush] on the same
// instance should be serialized.
//
// # Attribution
//
// This library is based on libsoxr (https://sourceforge.net/projects/soxr/)
// by Rob Sykes, licensed under LGPL-2.1. The following components were derived
// from libsoxr:
//
//   - Multi-stage resampling pipeline (DFT pre-stage + polyphase)
//   - Polyphase filter bank decomposition
//   - Quality preset parameters and filter specifications
//   - Integer phase arithmetic for polyphase interpolation
//
// The Kaiser window filter design follows well-established DSP literature,
// particularly the work of James Kaiser on optimal window functions.
package resampler
