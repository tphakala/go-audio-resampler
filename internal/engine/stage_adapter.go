package engine

import (
	"github.com/tphakala/go-audio-resampler/internal/simdops"
	"github.com/tphakala/simd/cpu"
)

// Constants for stage adapter calculations.
const (
	// Latency calculation divisor (half the filter length for symmetric FIR).
	latencyDivisor = 2

	// Byte sizes for float types.
	bytesPerFloat32 = 4
	bytesPerFloat64 = 8
)

// StageAdapter wraps a Resampler to implement the pipeline.Stage interface.
// This allows the engine.Resampler to be used in the resample package's
// multi-stage pipeline architecture.
//
// Type parameter F controls the precision of sample processing.
type StageAdapter[F simdops.Float] struct {
	*Resampler[F]
}

// NewStageAdapter creates a StageAdapter wrapping the given Resampler.
func NewStageAdapter[F simdops.Float](r *Resampler[F]) *StageAdapter[F] {
	return &StageAdapter[F]{Resampler: r}
}

// GetRatio returns the resampling ratio.
func (s *StageAdapter[F]) GetRatio() float64 {
	return s.Resampler.GetRatio()
}

// GetLatency returns the stage latency in samples.
// This is the delay due to FIR filter buffering.
func (s *StageAdapter[F]) GetLatency() int {
	latency := 0

	// DFT stage latency: half the filter length (taps * factor for polyphase)
	if s.preStage != nil && s.preStage.factor > 1 {
		latency += (s.preStage.tapsPerPhase * s.preStage.factor) / latencyDivisor
	}

	// Polyphase stage latency
	if s.polyphaseStage != nil {
		latency += s.polyphaseStage.tapsPerPhase / latencyDivisor
	}

	return latency
}

// GetMinInput returns the minimum input size for processing.
func (s *StageAdapter[F]) GetMinInput() int {
	// Need at least 1 sample to start processing
	return 1
}

// GetMemoryUsage returns approximate memory usage in bytes.
func (s *StageAdapter[F]) GetMemoryUsage() int64 {
	var usage int64

	// Determine bytes per element based on type
	var zero F
	bytesPerElement := int64(bytesPerFloat32)
	if _, ok := any(zero).(float64); ok {
		bytesPerElement = bytesPerFloat64
	}

	// DFT stage memory
	if s.preStage != nil {
		// Polyphase coefficients: factor phases * tapsPerPhase
		for _, phase := range s.preStage.polyCoeffs {
			usage += int64(len(phase)) * bytesPerElement
		}
		usage += int64(cap(s.preStage.history)) * bytesPerElement
	}

	// Polyphase stage memory
	if s.polyphaseStage != nil {
		// Phase-first layout: polyCoeffs[phase][tap]
		for _, phase := range s.polyphaseStage.polyCoeffs {
			usage += int64(len(phase)) * bytesPerElement
		}
		usage += int64(cap(s.polyphaseStage.history)) * bytesPerElement
	}

	return usage
}

// GetFilterLength returns the total filter length.
func (s *StageAdapter[F]) GetFilterLength() int {
	length := 0

	if s.preStage != nil && s.preStage.factor > 1 {
		// Original filter length = tapsPerPhase * factor
		length += s.preStage.tapsPerPhase * s.preStage.factor
	}

	if s.polyphaseStage != nil {
		length += s.polyphaseStage.tapsPerPhase * s.polyphaseStage.numPhases
	}

	return length
}

// GetPhases returns the number of polyphase filter phases.
func (s *StageAdapter[F]) GetPhases() int {
	if s.polyphaseStage != nil {
		return s.polyphaseStage.numPhases
	}
	return 0
}

// GetSIMDInfo returns SIMD optimization info.
func (s *StageAdapter[F]) GetSIMDInfo() string {
	return cpu.Info()
}
