// Package resampler provides high-quality audio resampling in pure Go.
//
// This library is based on libsoxr (the SoX Resampler library) by Rob Sykes,
// implementing polyphase FIR filtering with Kaiser window design for
// professional-grade sample rate conversion.
//
// # Features
//
//   - Multiple quality presets from quick/low-latency to very high quality mastering
//   - Polyphase FIR filtering with efficient multi-stage architecture
//   - Kaiser window design for optimal stopband attenuation
//   - Optional SIMD acceleration (AVX2/SSE) via github.com/tphakala/simd
//   - Multi-channel support for stereo, surround, and multi-channel audio
//   - Streaming API for processing audio in chunks with proper state management
//   - Pure Go implementation with no CGO dependencies
//
// # Quick Start
//
// For simple one-shot resampling:
//
//	output, err := resampling.ResampleMono(input, 44100, 48000, resampling.QualityHigh)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// For streaming resampling with a reusable resampler:
//
//	config := &resampling.Config{
//	    InputRate:  44100,
//	    OutputRate: 48000,
//	    Channels:   2,
//	    Quality:    resampling.QualitySpec{Preset: resampling.QualityHigh},
//	}
//	r, err := resampling.New(config)
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
//   - [QualityQuick]: 8-bit precision, lowest CPU usage. Suitable for preview
//     or real-time with low CPU.
//   - [QualityLow]: 16-bit precision. Good for speech and low-bandwidth audio.
//   - [QualityMedium]: 16-bit precision. Suitable for general music playback.
//   - [QualityHigh]: 24-bit precision. Recommended for studio production and
//     high-quality streaming.
//   - [QualityVeryHigh]: 32-bit precision. For mastering and archival applications.
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
// only the needed output phases.
//
// # Stereo Processing
//
// For stereo audio, use [ResampleStereo] for one-shot processing or configure
// a multi-channel resampler:
//
//	leftOut, rightOut, err := resampling.ResampleStereo(
//	    leftChannel, rightChannel,
//	    44100, 48000,
//	    resampling.QualityHigh,
//	)
//
// Helper functions [InterleaveToStereo] and [DeinterleaveFromStereo] are provided
// for converting between interleaved and planar audio formats.
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
