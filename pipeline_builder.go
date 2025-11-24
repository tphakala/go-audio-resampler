package resampler

import (
	"fmt"

	"github.com/tphakala/go-audio-resampler/internal/pipeline"
)

// Pipeline wraps the pipeline.Pipeline with resample-specific functionality.
type Pipeline struct {
	*pipeline.Pipeline
	stages []StageSpec
}

// StageSpec extends pipeline.StageSpec with resample-specific fields.
type StageSpec struct {
	pipeline.StageSpec
	engine string // Implementation: "cubic", "polyphase", "halfband", "fft"
}

// buildPipeline constructs an optimal pipeline for the given configuration and ratio.
func buildPipeline(config *Config, ratio float64) (*Pipeline, error) {
	// Convert quality spec to pipeline quality params
	qualityParams := pipeline.QualityParams{
		Precision:     config.Quality.Precision,
		PassbandEnd:   config.Quality.PassbandEnd,
		StopbandBegin: config.Quality.StopbandBegin,
		PhaseResponse: config.Quality.PhaseResponse,
		AllowAliasing: config.Quality.Flags&FlagAllowAliasing != 0,
	}

	// Build the pipeline
	p, err := pipeline.BuildPipeline(ratio, qualityParams)
	if err != nil {
		return nil, fmt.Errorf("failed to build pipeline: %w", err)
	}

	// Convert to resample-specific pipeline
	rp := &Pipeline{
		Pipeline: p,
		stages:   make([]StageSpec, 0),
	}

	// Convert pipeline stages to resample stages
	for _, stage := range p.GetStages() {
		rs := StageSpec{
			StageSpec: stage,
		}

		// Determine engine implementation
		switch stage.Type {
		case pipeline.StageCubic:
			rs.engine = "cubic"
		case pipeline.StageHalfBand:
			rs.engine = "halfband"
		case pipeline.StagePolyphase:
			rs.engine = "polyphase"
		case pipeline.StageFFT:
			rs.engine = "fft"
		default:
			rs.engine = "unknown"
		}

		rp.stages = append(rp.stages, rs)
	}

	return rp, nil
}

// createStage creates a Stage implementation based on the specification.
// The config.Quality.Precision is passed to all filter stages to ensure
// consistent quality throughout the pipeline.
func createStage(spec StageSpec, config *Config) (Stage, error) {
	precision := config.Quality.Precision

	switch spec.Type {
	case pipeline.StageCubic:
		return newCubicStage(spec.Ratio), nil

	case pipeline.StageHalfBand:
		return newHalfBandStage(spec.Ratio, spec.FilterLength, precision), nil

	case pipeline.StagePolyphase:
		return newPolyphaseStage(
			spec.Ratio,
			spec.FilterLength,
			spec.Phases,
			precision,
		)

	case pipeline.StageFFT:
		return newFFTStage(spec.Ratio, spec.FilterLength, precision)

	default:
		return nil, fmt.Errorf("unsupported stage type: %v", spec.Type)
	}
}

// Stage represents a processing stage in the resampling pipeline.
// This extends the pipeline.Stage interface with resample-specific functionality.
type Stage interface {
	pipeline.Stage
}

// RingBuffer wraps pipeline.RingBuffer for convenience.
type RingBuffer struct {
	*pipeline.RingBuffer
}

// NewRingBuffer creates a new ring buffer.
func NewRingBuffer(capacity int) *RingBuffer {
	return &RingBuffer{
		RingBuffer: pipeline.NewRingBuffer(capacity),
	}
}
