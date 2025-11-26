package resampler

import (
	"github.com/tphakala/go-audio-resampler/internal/engine"
)

// Common sample rates for convenience functions.
const (
	// RateCD is the CD quality sample rate (Red Book standard).
	RateCD = 44100

	// RateDAT is the DAT/DVD sample rate.
	RateDAT = 48000

	// RateHiRes88 is the high-resolution 2x CD sample rate.
	RateHiRes88 = 88200

	// RateHiRes96 is the high-resolution 2x DAT sample rate.
	RateHiRes96 = 96000

	// RateHiRes176 is the very high resolution 4x CD sample rate.
	RateHiRes176 = 176400

	// RateHiRes192 is the very high resolution 4x DAT sample rate.
	RateHiRes192 = 192000

	// RateTelephony is the telephony (PSTN narrowband) sample rate.
	RateTelephony = 8000

	// RateVoIP is the VoIP wideband sample rate.
	RateVoIP = 16000

	// RateSpeech is the speech recognition common sample rate.
	RateSpeech = 22050

	// RateVideo is the video production sample rate (matches many video formats).
	RateVideo = 48000
)

// NewCDtoDAT creates a resampler for CD (44.1kHz) to DAT (48kHz) conversion.
// This is one of the most common professional audio conversions.
func NewCDtoDAT(quality QualityPreset) (Resampler, error) {
	return New(&Config{
		InputRate:  RateCD,
		OutputRate: RateDAT,
		Channels:   1,
		Quality:    QualitySpec{Preset: quality},
	})
}

// NewDATtoCD creates a resampler for DAT (48kHz) to CD (44.1kHz) conversion.
func NewDATtoCD(quality QualityPreset) (Resampler, error) {
	return New(&Config{
		InputRate:  RateDAT,
		OutputRate: RateCD,
		Channels:   1,
		Quality:    QualitySpec{Preset: quality},
	})
}

// NewCDtoHiRes creates a resampler for CD (44.1kHz) to high-res (88.2kHz) conversion.
func NewCDtoHiRes(quality QualityPreset) (Resampler, error) {
	return New(&Config{
		InputRate:  RateCD,
		OutputRate: RateHiRes88,
		Channels:   1,
		Quality:    QualitySpec{Preset: quality},
	})
}

// NewHiRestoCD creates a resampler for high-res (88.2kHz) to CD (44.1kHz) conversion.
func NewHiRestoCD(quality QualityPreset) (Resampler, error) {
	return New(&Config{
		InputRate:  RateHiRes88,
		OutputRate: RateCD,
		Channels:   1,
		Quality:    QualitySpec{Preset: quality},
	})
}

// NewSimple creates a simple mono resampler with sensible defaults.
// Uses QualityHigh for professional-grade audio quality.
func NewSimple(inputRate, outputRate float64) (Resampler, error) {
	return New(&Config{
		InputRate:  inputRate,
		OutputRate: outputRate,
		Channels:   1,
		Quality:    QualitySpec{Preset: QualityHigh},
	})
}

// NewStereo creates a stereo resampler with the specified quality.
func NewStereo(inputRate, outputRate float64, quality QualityPreset) (Resampler, error) {
	return New(&Config{
		InputRate:  inputRate,
		OutputRate: outputRate,
		Channels:   stereoChannels,
		Quality:    QualitySpec{Preset: quality},
	})
}

// NewMultiChannel creates a multi-channel resampler.
func NewMultiChannel(inputRate, outputRate float64, channels int, quality QualityPreset) (Resampler, error) {
	return New(&Config{
		InputRate:  inputRate,
		OutputRate: outputRate,
		Channels:   channels,
		Quality:    QualitySpec{Preset: quality},
	})
}

// SimpleResampler provides a simplified interface for basic resampling tasks.
// It wraps the engine.Resampler directly for maximum performance.
// Uses float64 precision for maximum quality.
type SimpleResampler struct {
	engine *engine.Resampler[float64]
}

// NewEngine creates a SimpleResampler using the engine directly.
// This bypasses the pipeline infrastructure for simpler use cases.
// Uses float64 precision for maximum quality.
func NewEngine(inputRate, outputRate float64, quality QualityPreset) (*SimpleResampler, error) {
	engineQuality := presetToEngineQuality(quality)
	r, err := engine.NewResampler[float64](inputRate, outputRate, engineQuality)
	if err != nil {
		return nil, err
	}
	return &SimpleResampler{engine: r}, nil
}

// Process resamples the input samples.
func (r *SimpleResampler) Process(input []float64) ([]float64, error) {
	return r.engine.Process(input)
}

// Flush returns any remaining buffered samples.
func (r *SimpleResampler) Flush() ([]float64, error) {
	return r.engine.Flush()
}

// Reset clears internal state.
func (r *SimpleResampler) Reset() {
	r.engine.Reset()
}

// GetRatio returns the resampling ratio.
func (r *SimpleResampler) GetRatio() float64 {
	return r.engine.GetRatio()
}

// GetStatistics returns processing statistics.
func (r *SimpleResampler) GetStatistics() map[string]int64 {
	return r.engine.GetStatistics()
}

// presetToEngineQuality converts a QualityPreset to engine.Quality.
func presetToEngineQuality(preset QualityPreset) engine.Quality {
	switch preset {
	case QualityQuick, QualityLow:
		return engine.QualityLow
	case QualityMedium:
		return engine.QualityMedium
	case QualityHigh, QualityVeryHigh:
		return engine.QualityHigh
	default:
		return engine.QualityMedium
	}
}

// ResampleMono is a convenience function for one-shot mono resampling.
// It creates a resampler, processes the input, flushes, and returns the result.
func ResampleMono(input []float64, inputRate, outputRate float64, quality QualityPreset) ([]float64, error) {
	r, err := NewEngine(inputRate, outputRate, quality)
	if err != nil {
		return nil, err
	}

	output, err := r.Process(input)
	if err != nil {
		return nil, err
	}

	flushed, err := r.Flush()
	if err != nil {
		return nil, err
	}

	return append(output, flushed...), nil
}

// ResampleStereo is a convenience function for one-shot stereo resampling.
// Input is expected as [left, right] channels.
func ResampleStereo(left, right []float64, inputRate, outputRate float64, quality QualityPreset) (leftOut, rightOut []float64, err error) {
	// Process left channel
	leftOut, err = ResampleMono(left, inputRate, outputRate, quality)
	if err != nil {
		return nil, nil, err
	}

	// Process right channel
	rightOut, err = ResampleMono(right, inputRate, outputRate, quality)
	if err != nil {
		return nil, nil, err
	}

	return leftOut, rightOut, nil
}

// InterleaveToStereo converts two mono channels to interleaved stereo.
// Output format: [L0, R0, L1, R1, L2, R2, ...]
func InterleaveToStereo(left, right []float64) []float64 {
	minLen := min(len(left), len(right))
	result := make([]float64, minLen*stereoChannels)
	for i := range minLen {
		result[i*stereoChannels] = left[i]
		result[i*stereoChannels+1] = right[i]
	}
	return result
}

// DeinterleaveFromStereo converts interleaved stereo to two mono channels.
// Input format: [L0, R0, L1, R1, L2, R2, ...]
func DeinterleaveFromStereo(interleaved []float64) (left, right []float64) {
	numSamples := len(interleaved) / stereoChannels
	left = make([]float64, numSamples)
	right = make([]float64, numSamples)
	for i := range numSamples {
		left[i] = interleaved[i*stereoChannels]
		right[i] = interleaved[i*stereoChannels+1]
	}
	return left, right
}

// =============================================================================
// Float32 Native API
// =============================================================================
//
// The following types and functions provide a float32-native resampling path.
// Use these when working with float32 audio data for:
//   - ~2x SIMD throughput (256-bit SIMD processes 8×float32 vs 4×float64)
//   - Reduced memory bandwidth
//   - Consistent float32 throughout the pipeline (no type conversions)
//
// For maximum precision (mastering, archival), use the float64 API instead.

// SimpleResamplerFloat32 provides a simplified float32-native interface for
// basic resampling tasks. It wraps the engine.Resampler[float32] directly
// for maximum performance with float32 audio data.
//
// Unlike using ProcessFloat32 on the main Resampler interface (where Flush
// returns float64), SimpleResamplerFloat32 keeps everything in float32,
// eliminating type conversion overhead.
//
// Example:
//
//	r, err := resampler.NewEngineFloat32(44100, 48000, resampler.QualityHigh)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for chunk := range audioChunks {
//	    output, _ := r.Process(chunk)  // []float32 in, []float32 out
//	    writeOutput(output)
//	}
//	final, _ := r.Flush()  // Returns []float32!
type SimpleResamplerFloat32 struct {
	engine *engine.Resampler[float32]
}

// NewEngineFloat32 creates a SimpleResamplerFloat32 using the engine directly.
// This bypasses the pipeline infrastructure for simpler use cases.
// Uses float32 precision throughout for ~2x SIMD throughput compared to float64.
//
// Parameters:
//   - inputRate: Input sample rate in Hz (e.g., 44100)
//   - outputRate: Output sample rate in Hz (e.g., 48000)
//   - quality: Quality preset controlling filter design and precision
//
// For maximum precision, use NewEngine (float64) instead.
func NewEngineFloat32(inputRate, outputRate float64, quality QualityPreset) (*SimpleResamplerFloat32, error) {
	engineQuality := presetToEngineQuality(quality)
	r, err := engine.NewResampler[float32](inputRate, outputRate, engineQuality)
	if err != nil {
		return nil, err
	}
	return &SimpleResamplerFloat32{engine: r}, nil
}

// Process resamples the input samples.
// Input and output are both float32, with no type conversion overhead.
func (r *SimpleResamplerFloat32) Process(input []float32) ([]float32, error) {
	return r.engine.Process(input)
}

// Flush returns any remaining buffered samples as float32.
// Unlike the main Resampler.Flush() which returns float64, this returns
// float32 for a consistent float32 workflow.
func (r *SimpleResamplerFloat32) Flush() ([]float32, error) {
	return r.engine.Flush()
}

// Reset clears internal state, allowing the resampler to be reused.
func (r *SimpleResamplerFloat32) Reset() {
	r.engine.Reset()
}

// GetRatio returns the resampling ratio (outputRate / inputRate).
func (r *SimpleResamplerFloat32) GetRatio() float64 {
	return r.engine.GetRatio()
}

// GetStatistics returns processing statistics including samples processed.
func (r *SimpleResamplerFloat32) GetStatistics() map[string]int64 {
	return r.engine.GetStatistics()
}

// ResampleMonoFloat32 is a convenience function for one-shot mono resampling
// with float32 samples. It creates a resampler, processes the input, flushes,
// and returns the result.
//
// This is the float32 equivalent of ResampleMono. Use this when:
//   - Your audio data is already in float32 format
//   - You want ~2x SIMD throughput compared to float64
//   - 32-bit precision is sufficient (most real-time applications)
//
// For maximum precision (mastering, archival), use ResampleMono instead.
func ResampleMonoFloat32(input []float32, inputRate, outputRate float64, quality QualityPreset) ([]float32, error) {
	r, err := NewEngineFloat32(inputRate, outputRate, quality)
	if err != nil {
		return nil, err
	}

	output, err := r.Process(input)
	if err != nil {
		return nil, err
	}

	flushed, err := r.Flush()
	if err != nil {
		return nil, err
	}

	return append(output, flushed...), nil
}

// ResampleStereoFloat32 is a convenience function for one-shot stereo resampling
// with float32 samples. Input is expected as [left, right] channels.
//
// This is the float32 equivalent of ResampleStereo.
func ResampleStereoFloat32(left, right []float32, inputRate, outputRate float64, quality QualityPreset) (leftOut, rightOut []float32, err error) {
	// Process left channel
	leftOut, err = ResampleMonoFloat32(left, inputRate, outputRate, quality)
	if err != nil {
		return nil, nil, err
	}

	// Process right channel
	rightOut, err = ResampleMonoFloat32(right, inputRate, outputRate, quality)
	if err != nil {
		return nil, nil, err
	}

	return leftOut, rightOut, nil
}

// InterleaveToStereoFloat32 converts two mono float32 channels to interleaved stereo.
// Output format: [L0, R0, L1, R1, L2, R2, ...]
//
// This is the float32 equivalent of InterleaveToStereo.
func InterleaveToStereoFloat32(left, right []float32) []float32 {
	minLen := min(len(left), len(right))
	result := make([]float32, minLen*stereoChannels)
	for i := range minLen {
		result[i*stereoChannels] = left[i]
		result[i*stereoChannels+1] = right[i]
	}
	return result
}

// DeinterleaveFromStereoFloat32 converts interleaved stereo float32 to two mono channels.
// Input format: [L0, R0, L1, R1, L2, R2, ...]
//
// This is the float32 equivalent of DeinterleaveFromStereo.
func DeinterleaveFromStereoFloat32(interleaved []float32) (left, right []float32) {
	numSamples := len(interleaved) / stereoChannels
	left = make([]float32, numSamples)
	right = make([]float32, numSamples)
	for i := range numSamples {
		left[i] = interleaved[i*stereoChannels]
		right[i] = interleaved[i*stereoChannels+1]
	}
	return left, right
}
