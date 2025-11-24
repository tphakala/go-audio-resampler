package resampler

import (
	"github.com/tphakala/go-audio-resampler/internal/engine"
	"github.com/tphakala/go-audio-resampler/internal/pipeline"
)

// Stage creation constants.
const (
	// defaultFFTPhases is the default number of phases for FFT fallback.
	defaultFFTPhases = 256

	// halfBandFactor is the up/downsampling factor for half-band filters (always 2).
	halfBandFactor = 2
)

// newCubicStage creates a cubic interpolation stage.
func newCubicStage(ratio float64) pipeline.Stage {
	return engine.NewCubicStage(ratio)
}

// newHalfBandStage creates a half-band decimation/interpolation stage.
// Half-band filters are specialized FIR filters where every other coefficient is zero,
// enabling 2x up/downsampling with half the multiplications of a general FIR filter.
// For now, we use the polyphase engine which is functionally equivalent but less optimized.
// A dedicated half-band implementation would store only non-zero coefficients
// (see soxr/src/half-coefs.h for reference coefficients).
func newHalfBandStage(ratio float64, filterLength, precision int) pipeline.Stage {
	// Use polyphase stage with factor=2 - functionally equivalent to half-band
	// The polyphase implementation handles 2x ratios efficiently
	stage, err := newPolyphaseStage(ratio, filterLength, halfBandFactor, precision)
	if err != nil {
		// Fallback to stub if polyphase creation fails
		return &stubStage{
			ratio:        ratio,
			filterLength: filterLength,
			name:         "halfband",
		}
	}
	return stage
}

// newPolyphaseStage creates a polyphase FIR filtering stage using engine.Resampler.
// Uses float64 precision for maximum quality in pipeline stages.
//
// Parameters:
//   - ratio: resampling ratio (output/input)
//   - filterLength: suggested filter length (may be adjusted by engine)
//   - phases: number of polyphase phases (may be adjusted by engine)
//   - precision: bit precision (8, 16, 20, 24, 28, 32) - maps to engine.Quality
func newPolyphaseStage(ratio float64, _, _, precision int) (pipeline.Stage, error) {
	// Map precision (in bits) to engine quality level
	quality := precisionToEngineQuality(precision)

	// Create the engine resampler
	// Use reference rates - only the ratio matters for filter design
	inputRate := 48000.0
	outputRate := inputRate * ratio

	resampler, err := engine.NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, err
	}

	// Wrap in stage adapter
	return engine.NewStageAdapter[float64](resampler), nil
}

// Precision thresholds for quality mapping (matching soxr's approach).
const (
	precisionThresholdQuick    = 8  // Up to 8-bit: QualityQuick
	precisionThresholdLow      = 16 // Up to 16-bit: QualityLow
	precisionThresholdHigh     = 20 // Up to 20-bit: QualityHigh (soxr's default HQ)
	precisionThresholdMedHigh  = 24 // Up to 24-bit: Quality24Bit
	precisionThresholdVeryHigh = 28 // Up to 28-bit: QualityVeryHigh (soxr's VHQ)
)

// precisionToEngineQuality converts bit precision to engine.Quality.
// This maps the top-level resampler's precision setting to the engine's
// quality presets, which control filter attenuation and tap count.
//
// Precision to Quality mapping (matching soxr's approach):
//   - 8 bits:  QualityQuick  (~54 dB attenuation)
//   - 16 bits: QualityLow    (~102 dB attenuation)
//   - 20 bits: QualityHigh   (~126 dB attenuation) - soxr's default HQ
//   - 24 bits: Quality24Bit  (~150 dB attenuation)
//   - 28 bits: QualityVeryHigh (~175 dB attenuation) - soxr's VHQ
//   - 32 bits: Quality32Bit  (~199 dB attenuation)
func precisionToEngineQuality(precision int) engine.Quality {
	switch {
	case precision <= precisionThresholdQuick:
		return engine.QualityQuick
	case precision <= precisionThresholdLow:
		return engine.QualityLow
	case precision <= precisionThresholdHigh:
		return engine.QualityHigh
	case precision <= precisionThresholdMedHigh:
		return engine.Quality24Bit
	case precision <= precisionThresholdVeryHigh:
		return engine.QualityVeryHigh
	default:
		return engine.Quality32Bit
	}
}

// newFFTStage creates an FFT-based resampling stage.
// FFT-based resampling (overlap-add/overlap-save) is beneficial for very long filters
// where time-domain FIR convolution becomes expensive (O(N×M) vs O(N log N)).
// For typical audio resampling with moderate filter lengths, polyphase FIR
// is more efficient and provides equivalent quality.
func newFFTStage(ratio float64, fftSize, precision int) (pipeline.Stage, error) {
	// Polyphase FIR provides equivalent quality and is more efficient
	// for typical audio filter lengths (< 1000 taps).
	// FFT-based would only benefit for extremely long filters.
	return newPolyphaseStage(ratio, fftSize, defaultFFTPhases, precision)
}

// stubStage is a temporary stub implementation for stages not yet implemented.
type stubStage struct {
	ratio        float64
	filterLength int
	phases       int
	name         string
}

func (s *stubStage) Process(input []float64) ([]float64, error) {
	// Simple passthrough with ratio adjustment for now
	outputSize := int(float64(len(input)) * s.ratio)
	output := make([]float64, outputSize)

	// Simple nearest-neighbor resampling for both up and downsampling
	for i := range output {
		srcIdx := int(float64(i) / s.ratio)
		if srcIdx >= len(input) {
			srcIdx = len(input) - 1
		}
		output[i] = input[srcIdx]
	}

	return output, nil
}

func (s *stubStage) Flush() ([]float64, error) {
	return []float64{}, nil
}

func (s *stubStage) Reset() {}

func (s *stubStage) GetRatio() float64 {
	return s.ratio
}

func (s *stubStage) GetLatency() int {
	if s.filterLength > 0 {
		return s.filterLength / latencyDivisor
	}
	return 0
}

func (s *stubStage) GetMinInput() int {
	return minInputSamples
}

func (s *stubStage) GetMemoryUsage() int64 {
	// Estimate based on filter length
	return int64(s.filterLength * bytesPerFloat64)
}

func (s *stubStage) GetFilterLength() int {
	return s.filterLength
}

func (s *stubStage) GetPhases() int {
	return s.phases
}

func (s *stubStage) GetSIMDInfo() string {
	return ""
}

// Ensure implementations satisfy the interface
var (
	_ pipeline.Stage = (*engine.CubicStage)(nil)
	_ pipeline.Stage = (*engine.StageAdapter[float64])(nil)
	_ pipeline.Stage = (*stubStage)(nil)
)
