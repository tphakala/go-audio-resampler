// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

// Resampler implements high-quality audio resampling using soxr-style
// multi-stage architecture.
//
// Type parameter F must be float32 or float64, controlling the precision
// of internal processing. Use float32 for ~2x SIMD throughput, float64
// for maximum precision.
//
// Architecture (matching soxr):
//   - Integer ratios (e.g., 2×): Single DFT stage
//   - Non-integer ratios (e.g., CD→DAT): DFT pre-stage + Polyphase stage
//
// The polyphase stage uses integer division/modulo for phase calculation,
// exactly matching soxr's poly-fir0.h algorithm.
type Resampler[F simdops.Float] struct {
	// Configuration
	inputRate  float64
	outputRate float64
	ratio      float64 // outputRate / inputRate

	// Stages
	cubicStage      *CubicStage[F]         // Optional cubic interpolation stage (for QualityQuick)
	preStage        *DFTStage[F]           // Optional pre-upsampling stage
	decimationStage *DFTDecimationStage[F] // Optional decimation stage (for integer downsampling)
	polyphaseStage  *PolyphaseStage[F]     // Main polyphase resampling stage

	// SIMD operations for type F
	ops *simdops.Ops[F]

	// Statistics
	samplesIn  int64
	samplesOut int64
}

// NewResampler creates a new resampler for the given sample rates.
//
// This implements soxr's multi-stage architecture:
//   - For integer ratios: Uses only DFT stage
//   - For non-integer ratios: Uses DFT pre-stage (2×) + polyphase stage
func NewResampler[F simdops.Float](inputRate, outputRate float64, quality Quality) (*Resampler[F], error) {
	if inputRate <= 0 || outputRate <= 0 {
		return nil, fmt.Errorf("sample rates must be positive: input=%f, output=%f", inputRate, outputRate)
	}

	ratio := outputRate / inputRate

	// Validate ratio bounds to prevent memory exhaustion and integer overflow.
	// Following SOXR's pattern: ratios between 1/256 and 256 are practical for audio.
	// Extreme ratios can cause: (1) integer overflow in output size calculation,
	// (2) memory exhaustion from attempting to allocate huge output buffers.
	const minRatio = 1.0 / 256.0 // 256x downsampling
	const maxRatio = 256.0       // 256x upsampling
	if ratio < minRatio || ratio > maxRatio {
		return nil, fmt.Errorf("resampling ratio %.6f out of valid range [%.6f, %.0f]", ratio, minRatio, maxRatio)
	}
	ops := simdops.For[F]()

	r := &Resampler[F]{
		inputRate:  inputRate,
		outputRate: outputRate,
		ratio:      ratio,
		ops:        ops,
	}

	// QualityQuick uses cubic interpolation (matching SOXR_QQ)
	if quality == QualityQuick {
		cubicStage := NewCubicStage[F](ratio)
		r.cubicStage = cubicStage
		return r, nil
	}

	// Determine architecture based on ratio
	// Key insight: DFT pre-stage (upsampling) only makes sense for UPSAMPLING.
	// For downsampling, we should NOT upsample first - go directly to polyphase.
	if ratio >= 1.0 {
		// UPSAMPLING (ratio >= 1.0)
		if isIntegerRatio(ratio) {
			// Integer ratio: single DFT stage
			intRatio := int(math.Round(ratio))
			dftStage, err := NewDFTStage[F](intRatio, quality)
			if err != nil {
				return nil, fmt.Errorf("failed to create DFT stage: %w", err)
			}
			r.preStage = dftStage
			// No polyphase stage needed
		} else {
			// Non-integer upsampling: DFT pre-stage + polyphase stage
			// Pre-upsample by 2× to get better working ratio for polyphase
			preUpsampleFactor := 2
			intermediateRate := inputRate * float64(preUpsampleFactor)

			// Create DFT pre-stage (2× upsampling)
			dftStage, err := NewDFTStage[F](preUpsampleFactor, quality)
			if err != nil {
				return nil, fmt.Errorf("failed to create DFT pre-stage: %w", err)
			}
			r.preStage = dftStage

			// Create polyphase stage for remaining ratio
			// Polyphase operates on pre-upsampled signal
			polyphaseRatio := outputRate / intermediateRate
			// Pass total io_ratio for correct Fp1 calculation (soxr uses total ratio)
			totalIORatio := inputRate / outputRate
			// hasPreStage = true because we have a DFT pre-stage
			polyStage, err := NewPolyphaseStage[F](polyphaseRatio, totalIORatio, true, quality)
			if err != nil {
				return nil, fmt.Errorf("failed to create polyphase stage: %w", err)
			}
			r.polyphaseStage = polyStage
		}
	} else {
		// DOWNSAMPLING (ratio < 1.0)
		// soxr uses different approaches based on the ratio:
		// - Integer ratios (2:1, 3:1, 4:1): DFT decimation stage
		// - Non-integer ratios: 2x DFT pre-stage + polyphase filtering
		//
		// For integer ratios, DFT decimation achieves 150+ dB anti-aliasing
		// because it uses a very long filter with cutoff at output Nyquist.
		ioRatio := inputRate / outputRate // e.g., 2.0 for 96→48

		if isIntegerRatio(ioRatio) && ioRatio >= 2.0 {
			// Integer ratio downsampling: use DFT decimation stage
			// This matches soxr's approach for 96→48, 192→48, etc.
			decimFactor := int(math.Round(ioRatio))
			decimStage, err := NewDFTDecimationStage[F](decimFactor, quality)
			if err != nil {
				return nil, fmt.Errorf("failed to create DFT decimation stage: %w", err)
			}
			r.decimationStage = decimStage
			// No polyphase stage needed for integer decimation
		} else {
			// Non-integer downsampling: soxr uses 2x upsampling pre-stage + polyphase
			// This approach gives better filter characteristics because:
			// 1. The 2x upsampling moves frequencies up, giving more room for transition
			// 2. The polyphase stage then has a cleaner decimation task
			//
			// Example for 48kHz → 44.1kHz:
			//   Pre-stage: 2x upsample (48 → 96 kHz)
			//   Polyphase: 96 → 44.1 kHz (ratio = 0.459375)
			preUpsampleFactor := 2
			intermediateRate := inputRate * float64(preUpsampleFactor)

			// Create DFT pre-stage (2× upsampling)
			dftStage, err := NewDFTStage[F](preUpsampleFactor, quality)
			if err != nil {
				return nil, fmt.Errorf("failed to create DFT pre-stage: %w", err)
			}
			r.preStage = dftStage

			// Create polyphase stage for decimation from intermediate rate
			// Polyphase operates on pre-upsampled signal
			polyphaseRatio := outputRate / intermediateRate
			// totalIORatio is input/output ratio relative to ORIGINAL rates
			totalIORatio := ioRatio
			// hasPreStage = false because soxr uses preM=0 for upsampling pre-stages.
			// When the pre-stage is upsampling (not downsampling), the polyphase
			// filter design treats it as if there's no pre-stage in terms of
			// frequency normalization (Fn=1, not Fn=2*mult).
			polyStage, err := NewPolyphaseStage[F](polyphaseRatio, totalIORatio, false, quality)
			if err != nil {
				return nil, fmt.Errorf("failed to create polyphase stage: %w", err)
			}
			r.polyphaseStage = polyStage
		}
	}

	return r, nil
}

// Process resamples the input samples.
func (r *Resampler[F]) Process(input []F) ([]F, error) { //nolint:dupl // intentional parallel structure with ProcessZeroCopy
	if len(input) == 0 {
		return []F{}, nil
	}

	r.samplesIn += int64(len(input))

	// QualityQuick uses cubic interpolation only
	if r.cubicStage != nil {
		output, err := r.cubicStage.Process(input)
		if err != nil {
			return nil, fmt.Errorf("cubic stage processing failed: %w", err)
		}
		r.samplesOut += int64(len(output))
		return output, nil
	}

	// Stage 1: Pre-stage (DFT upsampling) - for upsampling only
	intermediate := input
	var err error
	if r.preStage != nil {
		intermediate, err = r.preStage.Process(input)
		if err != nil {
			return nil, fmt.Errorf("pre-stage processing failed: %w", err)
		}
	}

	// Stage 2: Decimation stage (for integer downsampling) OR Polyphase stage
	output := intermediate
	if r.decimationStage != nil {
		// Integer downsampling: use DFT decimation
		output, err = r.decimationStage.Process(intermediate)
		if err != nil {
			return nil, fmt.Errorf("decimation stage processing failed: %w", err)
		}
	} else if r.polyphaseStage != nil {
		// Non-integer ratio: use polyphase
		output, err = r.polyphaseStage.Process(intermediate)
		if err != nil {
			return nil, fmt.Errorf("polyphase stage processing failed: %w", err)
		}
	}

	r.samplesOut += int64(len(output))
	return output, nil
}

// ProcessZeroCopy resamples input using the zero-copy internal path.
// The returned slice aliases internal buffers and is only valid until the
// next Process, ProcessZeroCopy, or Flush call.
func (r *Resampler[F]) ProcessZeroCopy(input []F) ([]F, error) { //nolint:dupl // intentional parallel structure with Process
	if len(input) == 0 {
		return []F{}, nil
	}

	r.samplesIn += int64(len(input))

	if r.cubicStage != nil {
		output, err := r.cubicStage.Process(input)
		if err != nil {
			return nil, fmt.Errorf("cubic stage processing failed: %w", err)
		}
		r.samplesOut += int64(len(output))
		return output, nil
	}

	intermediate := input
	var err error
	if r.preStage != nil {
		intermediate, err = r.preStage.processZeroCopy(input)
		if err != nil {
			return nil, fmt.Errorf("pre-stage processing failed: %w", err)
		}
	}

	output := intermediate
	if r.decimationStage != nil {
		output, err = r.decimationStage.processZeroCopy(intermediate)
		if err != nil {
			return nil, fmt.Errorf("decimation stage processing failed: %w", err)
		}
	} else if r.polyphaseStage != nil {
		output, err = r.polyphaseStage.processZeroCopy(intermediate)
		if err != nil {
			return nil, fmt.Errorf("polyphase stage processing failed: %w", err)
		}
	}

	r.samplesOut += int64(len(output))
	return output, nil
}

// Flush returns any remaining buffered samples.
func (r *Resampler[F]) Flush() ([]F, error) {
	// QualityQuick cubic stage doesn't buffer
	if r.cubicStage != nil {
		return r.cubicStage.Flush()
	}

	var output []F
	var err error

	// Flush pre-stage
	if r.preStage != nil {
		intermediate, flushErr := r.preStage.Flush()
		if flushErr != nil {
			return nil, flushErr
		}

		// If we have a polyphase stage, process the flushed samples
		if r.polyphaseStage != nil && len(intermediate) > 0 {
			output, err = r.polyphaseStage.Process(intermediate)
			if err != nil {
				return nil, err
			}
		} else {
			output = intermediate
		}
	}

	// Flush decimation stage (for integer downsampling)
	if r.decimationStage != nil {
		decimFlush, err := r.decimationStage.Flush()
		if err != nil {
			return nil, err
		}
		output = append(output, decimFlush...)
	}

	// Flush polyphase stage
	if r.polyphaseStage != nil {
		polyFlush, err := r.polyphaseStage.Flush()
		if err != nil {
			return nil, err
		}
		output = append(output, polyFlush...)
	}

	r.samplesOut += int64(len(output))
	return output, nil
}

// Reset clears internal state.
func (r *Resampler[F]) Reset() {
	if r.cubicStage != nil {
		r.cubicStage.Reset()
	}
	if r.preStage != nil {
		r.preStage.Reset()
	}
	if r.decimationStage != nil {
		r.decimationStage.Reset()
	}
	if r.polyphaseStage != nil {
		r.polyphaseStage.Reset()
	}
	r.samplesIn = 0
	r.samplesOut = 0
}

// GetRatio returns the resampling ratio.
func (r *Resampler[F]) GetRatio() float64 {
	return r.ratio
}

// GetStatistics returns processing statistics.
func (r *Resampler[F]) GetStatistics() map[string]int64 {
	return map[string]int64{
		"samplesIn":  r.samplesIn,
		"samplesOut": r.samplesOut,
	}
}

// isIntegerRatio checks if the ratio is an integer (within tolerance).
func isIntegerRatio(ratio float64) bool {
	const tolerance = 1e-9
	rounded := math.Round(ratio)
	return math.Abs(ratio-rounded) < tolerance && rounded >= 1.0
}
