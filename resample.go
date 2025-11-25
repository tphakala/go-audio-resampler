package resampler

import (
	"errors"
	"fmt"
)

// Resampler is the main interface for audio resampling.
// Implementations may use different algorithms (cubic, polyphase FIR, FFT)
// depending on quality settings and performance requirements.
type Resampler interface {
	// Process resamples a mono audio channel.
	// The input slice contains audio samples at the input rate.
	// Returns resampled audio at the output rate.
	Process(input []float64) ([]float64, error)

	// ProcessFloat32 is like Process but for float32 samples.
	ProcessFloat32(input []float32) ([]float32, error)

	// ProcessMulti processes multiple audio channels simultaneously.
	// Each slice in the input represents one channel.
	// Channels are processed independently but may share filter coefficients.
	ProcessMulti(input [][]float64) ([][]float64, error)

	// Flush returns any remaining samples in internal buffers.
	// Should be called when no more input will be provided.
	Flush() ([]float64, error)

	// GetLatency returns the resampler latency in samples.
	// This is the delay between input and output due to filtering.
	GetLatency() int

	// Reset clears all internal state and buffers.
	Reset()

	// GetRatio returns the resampling ratio (output_rate / input_rate).
	GetRatio() float64
}

// Config holds resampling configuration.
type Config struct {
	// InputRate is the sample rate of input audio in Hz.
	InputRate float64

	// OutputRate is the desired output sample rate in Hz.
	OutputRate float64

	// Channels is the number of audio channels to process.
	// Channels are processed in parallel when possible.
	Channels int

	// Quality determines the resampling algorithm and filter parameters.
	Quality QualitySpec

	// MaxInputSize hints at the maximum input buffer size for optimization.
	// Set to 0 to use default buffer sizes.
	MaxInputSize int

	// EnableSIMD allows the use of SIMD optimizations when available.
	// Set to false to force pure Go implementation.
	EnableSIMD bool

	// EnableParallel enables parallel channel processing.
	// When true, multiple channels are processed concurrently using goroutines.
	// This provides speedup proportional to channel count (2x for stereo, 8x for 7.1).
	// Has no effect on mono audio.
	EnableParallel bool
}

// QualitySpec defines resampling quality parameters.
// Users can either use a preset or customize individual parameters.
type QualitySpec struct {
	// Preset is a convenience setting for common quality levels.
	// When set, it overrides other fields unless they are explicitly set.
	Preset QualityPreset

	// Precision in bits (15-33). Higher precision means better SNR
	// but increased computational cost. 16 = CD quality, 24 = studio.
	Precision int

	// PhaseResponse controls the filter's phase characteristics.
	// 0 = minimum phase (lowest latency)
	// 50 = linear phase (symmetric impulse response)
	// 100 = maximum phase
	PhaseResponse float64

	// PassbandEnd is the normalized frequency (0-1) below which
	// the signal is preserved. Typically 0.8-0.99.
	PassbandEnd float64

	// StopbandBegin is the normalized frequency (0-1) above which
	// the signal is attenuated. Must be > PassbandEnd.
	StopbandBegin float64

	// Flags for additional options.
	Flags QualityFlags
}

// QualityPreset enumerates predefined quality levels.
// Each preset configures multiple parameters for typical use cases.
type QualityPreset int

const (
	// QualityQuick uses cubic interpolation. Fastest but lowest quality.
	// Suitable for preview, real-time with low CPU, or non-critical audio.
	QualityQuick QualityPreset = iota

	// QualityLow provides basic resampling with ~16-bit quality.
	// Good for speech, low-bandwidth audio, or when CPU is limited.
	QualityLow

	// QualityMedium provides good quality suitable for most music.
	// Equivalent to typical CD quality (16-bit, 44.1kHz).
	QualityMedium

	// QualityHigh provides professional quality with 24-bit precision.
	// Suitable for studio production and high-quality streaming.
	QualityHigh

	// QualityVeryHigh provides maximum quality with 32-bit precision.
	// For mastering, archival, and critical listening applications.
	QualityVeryHigh

	// QualityCustom indicates manual configuration of parameters.
	QualityCustom
)

// QualityFlags provides additional quality options.
type QualityFlags uint32

const (
	// FlagNoInterpolation disables coefficient interpolation in polyphase filters.
	// Reduces CPU usage but may introduce artifacts.
	FlagNoInterpolation QualityFlags = 1 << iota

	// FlagMinimumPhase forces minimum-phase filter design for lowest latency.
	FlagMinimumPhase

	// FlagLinearPhase forces linear-phase filter design for best transient response.
	FlagLinearPhase

	// FlagAllowAliasing permits some aliasing for better passband response.
	// Only use when input is known to be bandlimited.
	FlagAllowAliasing

	// FlagNoSIMD disables SIMD optimizations even when available.
	FlagNoSIMD
)

// Common errors returned by the resampler.
var (
	// ErrInvalidConfig indicates invalid configuration parameters.
	ErrInvalidConfig = errors.New("invalid resampler configuration")

	// ErrBufferTooSmall indicates the output buffer is too small.
	ErrBufferTooSmall = errors.New("output buffer too small")

	// ErrNotSupported indicates the requested operation is not supported.
	ErrNotSupported = errors.New("operation not supported")
)

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.InputRate <= 0 || c.OutputRate <= 0 {
		return fmt.Errorf("%w: sample rates must be positive", ErrInvalidConfig)
	}

	if c.Channels < 1 {
		return fmt.Errorf("%w: channels must be at least 1", ErrInvalidConfig)
	}

	if c.Channels > maxChannels {
		return fmt.Errorf("%w: too many channels (max %d)", ErrInvalidConfig, maxChannels)
	}

	ratio := c.OutputRate / c.InputRate
	if ratio < minRatioFactor || ratio > maxRatioFactor {
		return fmt.Errorf("%w: resampling ratio out of range (%v to %v)", ErrInvalidConfig, minRatioFactor, maxRatioFactor)
	}

	if err := c.Quality.Validate(); err != nil {
		return err
	}

	return nil
}

// Validate checks if the quality specification is valid.
func (q *QualitySpec) Validate() error {
	if q.Preset == QualityCustom {
		if q.Precision < precision8Bit || q.Precision > precision33Bit {
			return fmt.Errorf("%w: precision must be %d-%d bits", ErrInvalidConfig, precision8Bit, precision33Bit)
		}

		if q.PhaseResponse < 0 || q.PhaseResponse > 100 {
			return fmt.Errorf("%w: phase response must be 0-100", ErrInvalidConfig)
		}

		if q.PassbandEnd <= 0 || q.PassbandEnd >= 1 {
			return fmt.Errorf("%w: passband end must be in (0, 1)", ErrInvalidConfig)
		}

		if q.StopbandBegin <= q.PassbandEnd || q.StopbandBegin > 1 {
			return fmt.Errorf("%w: stopband begin must be in (passband_end, 1]", ErrInvalidConfig)
		}
	}

	return nil
}

// GetPresetSpec returns the quality specification for a preset.
func GetPresetSpec(preset QualityPreset) QualitySpec {
	switch preset {
	case QualityQuick:
		return QualitySpec{
			Preset:        QualityQuick,
			Precision:     precision8Bit,
			PhaseResponse: linearPhaseResponse,
			PassbandEnd:   quickPassbandEnd,
			StopbandBegin: quickStopbandBegin,
		}

	case QualityLow:
		return QualitySpec{
			Preset:        QualityLow,
			Precision:     precision16Bit,
			PhaseResponse: linearPhaseResponse,
			PassbandEnd:   lowPassbandEnd,
			StopbandBegin: lowStopbandBegin,
		}

	case QualityMedium:
		return QualitySpec{
			Preset:        QualityMedium,
			Precision:     precision16Bit,
			PhaseResponse: linearPhaseResponse,
			PassbandEnd:   mediumPassbandEnd,
			StopbandBegin: mediumStopbandBegin,
		}

	case QualityHigh:
		return QualitySpec{
			Preset:        QualityHigh,
			Precision:     precision24Bit,
			PhaseResponse: linearPhaseResponse,
			PassbandEnd:   highPassbandEnd,
			StopbandBegin: highStopbandBegin,
		}

	case QualityVeryHigh:
		return QualitySpec{
			Preset:        QualityVeryHigh,
			Precision:     precision32Bit,
			PhaseResponse: linearPhaseResponse,
			PassbandEnd:   veryHighPassbandEnd,
			StopbandBegin: veryHighStopbandBegin,
		}

	default:
		return QualitySpec{Preset: QualityMedium}
	}
}

// New creates a new resampler with the specified configuration.
// The actual implementation (cubic, polyphase, FFT) is chosen based on
// the quality settings and resampling ratio.
func New(config *Config) (Resampler, error) {
	if config == nil {
		return nil, fmt.Errorf("%w: config is nil", ErrInvalidConfig)
	}

	if err := config.Validate(); err != nil {
		return nil, err
	}

	// Apply preset if not custom
	if config.Quality.Preset != QualityCustom {
		config.Quality = GetPresetSpec(config.Quality.Preset)
	}

	// Select appropriate implementation based on quality and ratio
	ratio := config.OutputRate / config.InputRate

	// Create constant-rate resampler with multi-stage pipeline
	// The pipeline builder automatically selects optimal stages based on ratio and quality
	return newConstantRateResampler(config, ratio)
}

// Info returns information about the resampler implementation.
type Info struct {
	// Algorithm describes the resampling algorithm in use.
	Algorithm string

	// FilterLength is the number of filter taps.
	FilterLength int

	// Phases is the number of polyphase filter phases.
	Phases int

	// Latency is the processing latency in samples.
	Latency int

	// MemoryUsage is the approximate memory usage in bytes.
	MemoryUsage int64

	// SIMDEnabled indicates if SIMD optimizations are active.
	SIMDEnabled bool

	// SIMDType describes the SIMD instruction set in use.
	SIMDType string
}

// infoProvider is an optional interface for resamplers that can provide detailed info.
type infoProvider interface {
	GetInfo() Info
}

// GetInfo returns information about a resampler.
// If the resampler implements the infoProvider interface, it returns actual values.
// Otherwise, it returns basic info based on the resampler's public methods.
func GetInfo(r Resampler) Info {
	// Use type assertion to get actual info if available
	if provider, ok := r.(infoProvider); ok {
		return provider.GetInfo()
	}

	// Fallback for resamplers that don't implement infoProvider
	return Info{
		Algorithm:    "unknown",
		FilterLength: 0,
		Phases:       0,
		Latency:      r.GetLatency(),
		MemoryUsage:  0,
		SIMDEnabled:  false,
		SIMDType:     "none",
	}
}
