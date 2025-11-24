// Package pipeline implements the multi-stage resampling pipeline architecture.
// The pipeline automatically decomposes the resampling ratio into stages
// that can be processed efficiently using different algorithms.
package pipeline

import (
	"fmt"
	"math"
)

// Stage represents a single processing stage in the resampling pipeline.
// Each stage performs a specific transformation (decimation, interpolation, filtering).
type Stage interface {
	// Process transforms input samples to output samples.
	Process(input []float64) ([]float64, error)

	// Flush returns any remaining buffered samples.
	Flush() ([]float64, error)

	// Reset clears internal state.
	Reset()

	// GetRatio returns the stage's resampling ratio (output/input).
	GetRatio() float64

	// GetLatency returns the stage latency in samples.
	GetLatency() int

	// GetMinInput returns the minimum input size for processing.
	GetMinInput() int

	// GetMemoryUsage returns approximate memory usage in bytes.
	GetMemoryUsage() int64

	// GetFilterLength returns the filter length (0 if not applicable).
	GetFilterLength() int

	// GetPhases returns the number of polyphase filter phases (0 if not applicable).
	GetPhases() int

	// GetSIMDInfo returns SIMD optimization info (empty if none).
	GetSIMDInfo() string
}

// StageType identifies the type of processing stage.
type StageType int

const (
	// StageCubic performs cubic interpolation (fast, low quality).
	StageCubic StageType = iota

	// StageHalfBand performs half-band filtering (decimation by 2).
	StageHalfBand

	// StagePolyphase performs polyphase FIR filtering.
	StagePolyphase

	// StageFFT performs FFT-based resampling.
	StageFFT

	// StageDelay adds a fixed delay (for phase alignment).
	StageDelay
)

// StageSpec specifies parameters for creating a stage.
type StageSpec struct {
	Type          StageType
	Ratio         float64 // Resampling ratio for this stage
	Quality       int     // Quality level (0-100)
	FilterLength  int     // Number of filter taps
	Phases        int     // Number of phases for polyphase
	CutoffFactor  float64 // Filter cutoff frequency factor
	Interpolation int     // Coefficient interpolation order (0-3)
}

// Pipeline represents a multi-stage resampling pipeline.
type Pipeline struct {
	stages       []StageSpec
	totalRatio   float64
	totalLatency int
}

// QualityParams holds quality-related parameters for pipeline construction.
type QualityParams struct {
	Precision     int     // Bits of precision (8-32)
	PassbandEnd   float64 // Normalized frequency (0-1)
	StopbandBegin float64 // Normalized frequency (0-1)
	PhaseResponse float64 // Phase linearity (0-100)
	AllowAliasing bool    // Allow some aliasing for efficiency
}

// BuildPipeline constructs an optimal pipeline for the given ratio and quality.
// It decomposes the ratio into stages that can be processed efficiently.
func BuildPipeline(ratio float64, quality QualityParams) (*Pipeline, error) {
	if ratio <= 0 {
		return nil, fmt.Errorf("invalid ratio: %f", ratio)
	}

	p := &Pipeline{
		totalRatio: ratio,
		stages:     make([]StageSpec, 0, defaultStageCapacity),
	}

	// Quick mode: single cubic interpolation stage
	if quality.Precision <= precision8Bit {
		p.stages = append(p.stages, StageSpec{
			Type:  StageCubic,
			Ratio: ratio,
		})
		return p, nil
	}

	// Decompose ratio into stages
	remainingRatio := ratio

	// For downsampling (ratio < 1), use half-band stages for powers of 2
	if ratio < 1.0 {
		// Factor out powers of 2 for efficient half-band filtering
		for remainingRatio < halfRatio {
			p.stages = append(p.stages, StageSpec{
				Type:         StageHalfBand,
				Ratio:        halfRatio,
				Quality:      quality.Precision,
				FilterLength: calculateHalfBandTaps(quality),
			})
			remainingRatio *= doubleRatio
		}
	}

	// For upsampling (ratio > 1), use interpolation stages
	if ratio > 1.0 {
		// Factor out powers of 2 for efficient processing
		for remainingRatio > doubleRatio {
			p.stages = append(p.stages, StageSpec{
				Type:         StageHalfBand,
				Ratio:        doubleRatio,
				Quality:      quality.Precision,
				FilterLength: calculateHalfBandTaps(quality),
			})
			remainingRatio /= doubleRatio
		}
	}

	// Handle remaining ratio with polyphase or FFT
	if math.Abs(remainingRatio-1.0) > ratioTolerance {
		// Choose between polyphase and FFT based on ratio and quality
		if shouldUseFFT(remainingRatio, quality) {
			// FFT-based resampling for high-quality or exact ratios
			p.stages = append(p.stages, StageSpec{
				Type:         StageFFT,
				Ratio:        remainingRatio,
				Quality:      quality.Precision,
				FilterLength: calculateFFTSize(remainingRatio, quality),
			})
		} else {
			// Polyphase FIR for general resampling
			p.stages = append(p.stages, StageSpec{
				Type:          StagePolyphase,
				Ratio:         remainingRatio,
				Quality:       quality.Precision,
				FilterLength:  calculatePolyphaseTaps(remainingRatio, quality),
				Phases:        calculatePolyphasePhases(quality),
				CutoffFactor:  calculateCutoffFactor(remainingRatio, quality),
				Interpolation: calculateInterpolationOrder(quality),
			})
		}
	}

	// Calculate total latency
	p.calculateLatency()

	return p, nil
}

// calculateLatency computes the total pipeline latency.
func (p *Pipeline) calculateLatency() {
	totalLatency := 0
	cumulativeRatio := 1.0

	for _, stage := range p.stages {
		// Stage latency depends on filter length and position in pipeline
		stageLatency := 0

		switch stage.Type {
		case StageCubic:
			stageLatency = latencyCubic

		case StageHalfBand:
			stageLatency = stage.FilterLength / latencyHalfband

		case StagePolyphase:
			stageLatency = stage.FilterLength / latencyPolyphase

		case StageFFT:
			stageLatency = stage.FilterLength / latencyFFT

		case StageDelay:
			stageLatency = stage.FilterLength // Delay is the filter length itself
		}

		// Adjust for cumulative ratio change
		totalLatency += int(float64(stageLatency) / cumulativeRatio)
		cumulativeRatio *= stage.Ratio
	}

	p.totalLatency = totalLatency
}

// GetStages returns the pipeline stages.
func (p *Pipeline) GetStages() []StageSpec {
	return p.stages
}

// GetTotalRatio returns the combined ratio of all stages.
func (p *Pipeline) GetTotalRatio() float64 {
	return p.totalRatio
}

// GetTotalLatency returns the total pipeline latency in samples.
func (p *Pipeline) GetTotalLatency() int {
	return p.totalLatency
}

// Helper functions for pipeline construction

func calculateHalfBandTaps(quality QualityParams) int {
	// More taps for higher precision
	// Approximation: 4 taps per 6dB of attenuation
	attenuation := float64(quality.Precision) * dbPerBit
	taps := int(attenuation/attenuationDivisor) * simdAlignment

	// Ensure odd number of taps and reasonable bounds
	if taps%bufferGrowthFactor == 0 {
		taps++
	}
	if taps < minFilterTaps {
		taps = minFilterTaps
	}
	if taps > maxFilterTaps {
		taps = maxFilterTaps
	}

	return taps
}

func calculatePolyphaseTaps(ratio float64, quality QualityParams) int {
	// Base calculation on required attenuation and transition bandwidth
	attenuation := float64(quality.Precision) * dbPerBit
	transitionBandwidth := quality.StopbandBegin - quality.PassbandEnd

	// Kaiser formula approximation
	taps := int((attenuation - kaiserOffset) / (kaiserMultiplier * transitionBandwidth * kaiserTwoPi))

	// Adjust for ratio
	if ratio < 1 {
		taps = int(float64(taps) / ratio)
	}

	// Bounds checking
	if taps < minPolyphaseTaps {
		taps = minPolyphaseTaps
	}
	if taps > maxPolyphaseTaps {
		taps = maxPolyphaseTaps
	}

	// Round to multiple of 4 for SIMD efficiency
	taps = (taps + simdAlignmentMask) &^ simdAlignmentMask

	return taps
}

func calculatePolyphasePhases(quality QualityParams) int {
	// More phases for higher quality
	phases := phasesBase

	if quality.Precision >= precision24Bit {
		phases = phases24Bit
	}
	if quality.Precision >= precision32Bit {
		phases = phases32Bit
	}

	return phases
}

func calculateCutoffFactor(ratio float64, quality QualityParams) float64 {
	// Cutoff frequency relative to Nyquist
	cutoff := quality.PassbandEnd

	// Adjust for decimation to prevent aliasing
	if ratio < 1 {
		cutoff *= ratio
	}

	return cutoff
}

func calculateInterpolationOrder(quality QualityParams) int {
	// Higher order interpolation for better quality
	if quality.Precision >= precision24Bit {
		return interpOrderCubic
	}
	if quality.Precision >= precision16Bit {
		return interpOrderLinear
	}
	return interpOrderNone
}

func shouldUseFFT(ratio float64, quality QualityParams) bool {
	// Use FFT for very high quality or when ratio is close to simple fractions
	if quality.Precision >= precision28Bit {
		return true
	}

	// Check if ratio is close to a simple fraction (e.g., 44100/48000)
	for _, common := range commonAudioRatios {
		if math.Abs(ratio-common) < ratioToleranceFFT {
			return true
		}
	}

	return false
}

func calculateFFTSize(_ float64, quality QualityParams) int {
	// Base FFT size on quality requirements
	baseSize := fftSizeBase

	if quality.Precision >= precision24Bit {
		baseSize = fftSize24Bit
	}
	if quality.Precision >= precision32Bit {
		baseSize = fftSize32Bit
	}

	// Ensure power of 2
	size := 1
	for size < baseSize {
		size *= 2
	}

	return size
}

// OptimizePipeline attempts to optimize an existing pipeline by combining
// or reordering stages where beneficial.
// Current implementation returns the pipeline unchanged - optimization is
// a future enhancement that could combine adjacent half-band stages,
// reorder stages for better cache usage, or merge small ratio changes.
func OptimizePipeline(p *Pipeline) *Pipeline {
	// Pipeline optimization is not yet implemented.
	// The current stage selection already produces efficient pipelines
	// for common use cases.
	return p
}
