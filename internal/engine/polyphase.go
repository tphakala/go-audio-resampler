package engine

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/filter"
	"github.com/tphakala/go-audio-resampler/internal/simdops"
	"github.com/tphakala/simd/f64"
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
	preStage       *DFTStage[F]       // Optional pre-upsampling stage
	polyphaseStage *PolyphaseStage[F] // Main polyphase resampling stage

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
	ops := simdops.For[F]()

	r := &Resampler[F]{
		inputRate:  inputRate,
		outputRate: outputRate,
		ratio:      ratio,
		ops:        ops,
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
		// For downsampling, we use polyphase directly without DFT pre-stage.
		// This is correct because:
		// 1. DFT upsampling before polyphase downsampling is wasteful
		// 2. The polyphase filter can handle anti-aliasing directly
		// 3. soxr also uses this approach for downsampling
		totalIORatio := inputRate / outputRate
		// hasPreStage = false because we go directly to polyphase
		polyStage, err := NewPolyphaseStage[F](ratio, totalIORatio, false, quality)
		if err != nil {
			return nil, fmt.Errorf("failed to create polyphase stage: %w", err)
		}
		r.polyphaseStage = polyStage
		// No DFT pre-stage for downsampling
	}

	return r, nil
}

// Process resamples the input samples.
func (r *Resampler[F]) Process(input []F) ([]F, error) {
	if len(input) == 0 {
		return []F{}, nil
	}

	r.samplesIn += int64(len(input))

	// Stage 1: Pre-stage (DFT upsampling)
	intermediate := input
	var err error
	if r.preStage != nil {
		intermediate, err = r.preStage.Process(input)
		if err != nil {
			return nil, fmt.Errorf("pre-stage processing failed: %w", err)
		}
	}

	// Stage 2: Polyphase stage (if present)
	output := intermediate
	if r.polyphaseStage != nil {
		output, err = r.polyphaseStage.Process(intermediate)
		if err != nil {
			return nil, fmt.Errorf("polyphase stage processing failed: %w", err)
		}
	}

	r.samplesOut += int64(len(output))
	return output, nil
}

// Flush returns any remaining buffered samples.
func (r *Resampler[F]) Flush() ([]F, error) {
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
	if r.preStage != nil {
		r.preStage.Reset()
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

// Quality levels for resampling.
// These match soxr's quality presets for compatibility.
type Quality int

const (
	// QualityQuick provides quick cubic interpolation (8-bit equivalent).
	// Standard quality presets (matching soxr).
	QualityQuick Quality = iota
	// QualityLow provides low quality 16-bit resampling.
	QualityLow
	// QualityMedium provides medium quality 16-bit resampling.
	QualityMedium
	// QualityHigh provides high quality 20-bit resampling (soxr default).
	QualityHigh
	// QualityVeryHigh provides very high quality 28-bit resampling.
	QualityVeryHigh

	// Quality16Bit provides 16-bit precision for fine control.
	Quality16Bit
	// Quality20Bit provides 20-bit precision for fine control.
	Quality20Bit
	// Quality24Bit provides 24-bit precision for fine control.
	Quality24Bit
	// Quality28Bit provides 28-bit precision for fine control.
	Quality28Bit
	// Quality32Bit provides 32-bit precision for fine control.
	Quality32Bit
)

// Filter design constants.
const (
	// Stopband attenuation calculation: att = (bits + 1) * 6.02 dB
	// This matches soxr's formula: att = (bits1 + 1) * linear_to_dB(2.)
	dbPerBit = 6.0206 // 20 * log10(2) ≈ 6.02 dB per bit

	// Quality preset bit precisions (matching soxr)
	bitsQuick    = 8  // Quick quality
	bitsLow      = 16 // Low quality
	bitsMedium   = 16 // Medium quality (same bits, different transition)
	bitsHigh     = 20 // High quality (SOXR_HQ)
	bitsVeryHigh = 28 // Very high quality (SOXR_VHQ)

	// Explicit bit precision constants for Quality*Bit presets
	bits16Bit = 16 // 16-bit precision preset
	bits20Bit = 20 // 20-bit precision preset
	bits24Bit = 24 // 24-bit precision preset
	bits28Bit = 28 // 28-bit precision preset
	bits32Bit = 32 // 32-bit precision preset

	// Attenuation values derived from bit precision: (bits + 1) * 6.02 dB
	attenuationQuick    = (bitsQuick + 1) * dbPerBit    // ~54 dB
	attenuationLow      = (bitsLow + 1) * dbPerBit      // ~102 dB
	attenuationMedium   = (bitsMedium + 1) * dbPerBit   // ~102 dB
	attenuationHigh     = (bitsHigh + 1) * dbPerBit     // ~126 dB
	attenuationVeryHigh = (bitsVeryHigh + 1) * dbPerBit // ~175 dB

	// Filter cutoff frequency constants.
	nyquistFraction    = 0.5  // Half the sample rate (Nyquist)
	transitionBWFactor = 0.05 // Transition bandwidth relative to Nyquist

	// Polyphase filter design constants.
	downsamplingTransitionScale = 0.15  // Transition band scale factor for downsampling
	transitionBandwidthHalf     = 0.5   // Half factor for transition bandwidth calculation
	invFRespThreshold           = 0.999 // Guard against division by zero in rolloff compensation

	// Buffer sizing constants.
	historyBufferMultiplier = 2 // Extra capacity for history buffers

	// Cache optimization constants.
	// Process in chunks that fit in L2 cache (~256KB) for better cache efficiency.
	// Chunk size chosen so signal + kernel + output all fit in L2.
	l2CacheChunkSize = 4096 // Samples per chunk for L2 cache efficiency

	// Rational approximation constants.
	rationalApproxTolerance = 1e-10 // Tolerance for finding rational approximation

	// Loop unrolling constants.
	loopUnrollFactor = 4 // Process 4 taps at a time
	loopUnrollMask   = 3 // Mask for computing unrolled count (factor - 1)

	// Half-band optimization constant.
	// Half-band filters are used for 2× upsampling where Phase 0 is a passthrough.
	halfBandFactor = 2

	// soxr-derived filter design constants.
	// These values are from soxr's filter design algorithm.
	soxrDFTStageFc       = 0.4778321 // soxr's Fc for DFT stage (1.0 = Nyquist)
	passbandRolloffScale = 0.99      // Scale factor for passband edge (99% of Nyquist)
	imageRejectionFactor = 2.0       // Factor for image rejection frequency calculation
	soxrFcDenominator    = 2.0       // Denominator in soxr's Fc formula: Fc = (Fp+Fs)/(2*phases)
	soxrToOurNormScale   = 2.0       // Scale to convert soxr [0,1] to our [0,0.5] normalization

	// soxr downsampling filter design constants (from cr.c lines 426-431).
	// For downsampling with pre-stage: Fn = 2 * mult, Fs = 3 + |Fs1 - 1|
	// These produce dramatically different filter parameters than upsampling.
	soxrDownsamplingFnFactor = 2.0 // Fn = 2 * mult for downsampling
	soxrDownsamplingFsBase   = 3.0 // Fs = 3 + |Fs1 - 1| base value
	soxrUpsamplingFsCoeff    = 0.7 // Fs = 2 - (Fp1 + (Fs1 - Fp1) * 0.7) coefficient

	// lsxInvFResp function constants (from soxr's filter.c).
	// sinePhi polynomial coefficients: ((a3*a + a2)*a + a1)*a + a0
	sinePhiCoeffA3   = 2.0517e-07  // Cubic coefficient
	sinePhiCoeffA2   = -1.1303e-04 // Quadratic coefficient
	sinePhiCoeffA1   = 0.023154    // Linear coefficient
	sinePhiConstant  = 0.55924     // Constant term in sinePhi polynomial
	dbToLinearFactor = 0.05        // Multiplier for dB to linear conversion (ln(10) * 0.05)
	halfAmplitude    = 0.5         // Half amplitude threshold for lsxInvFResp

	// Input guard constants for lsxInvFResp to prevent NaN from bad inputs.
	minAttenuation = 1.0   // Minimum attenuation in dB (prevents polynomial issues)
	maxAttenuation = 300.0 // Maximum attenuation in dB (prevents x*0.5 > π)
	sineEpsilon    = 1e-10 // Minimum value for sin(x*0.5) to avoid log(0) or log(negative)

	// Cubic interpolation constants for Catmull-Rom style coefficient interpolation.
	// Used in NewPolyphaseStage to compute smooth interpolation between phases.
	// Formula: f(x) = a + b*x + c*x² + d*x³
	cubicPhaseOffset = 2   // Offset to get f2 (two phases ahead)
	cubicCenterCoeff = 0.5 // Coefficient for computing c: c = 0.5*(f1+fm1) - f0
	cubicDivisor     = 6.0 // Divisor for computing d: d = (1/6) * (...)
	cubicCMultiplier = 4.0 // Multiplier for c in d formula: d = (1/6)*(f2-f1+fm1-f0 - 4*c)
)

// qualityToAttenuation returns stopband attenuation for quality level.
func qualityToAttenuation(q Quality) float64 {
	switch q {
	case QualityQuick:
		return attenuationQuick
	case QualityLow:
		return attenuationLow
	case QualityMedium:
		return attenuationMedium
	case QualityHigh:
		return attenuationHigh
	case QualityVeryHigh:
		return attenuationVeryHigh
	case Quality16Bit:
		return (bits16Bit + 1) * dbPerBit
	case Quality20Bit:
		return (bits20Bit + 1) * dbPerBit
	case Quality24Bit:
		return (bits24Bit + 1) * dbPerBit
	case Quality28Bit:
		return (bits28Bit + 1) * dbPerBit
	case Quality32Bit:
		return (bits32Bit + 1) * dbPerBit
	default:
		return attenuationHigh // Default to high quality
	}
}

// =============================================================================
// DFT Stage - Simple FIR-based upsampling
// =============================================================================

// DFTStage implements integer-ratio upsampling using FIR filtering.
// Uses polyphase decomposition to avoid multiplying zeros.
//
// Type parameter F controls the precision of sample processing.
type DFTStage[F simdops.Float] struct {
	factor int // Upsampling factor (e.g., 2)

	// Polyphase filter bank - coeffs[phase][tap] in reversed order
	// Reversed order enables direct use with ConvolveValid
	polyCoeffs   [][]F
	tapsPerPhase int

	// Half-band optimization: For 2× upsampling, Phase 0 is often a pure
	// passthrough (single coefficient ≈ 1.0 at center). This skips ~50% of work.
	isHalfBand      bool // True if Phase 0 is a passthrough
	phase0TapOffset int  // Offset of non-zero tap in Phase 0 (for passthrough)
	phase0TapScale  F    // Scale factor for the passthrough tap

	// Input history buffer (non-upsampled)
	history []F

	// Pre-allocated output buffer for reduced allocations
	outputBuf []F

	// Temporary buffers for ConvolveValid per-phase output
	phaseBufs [][]F

	// SIMD operations for type F
	ops *simdops.Ops[F]
}

// NewDFTStage creates a DFT upsampling stage.
func NewDFTStage[F simdops.Float](factor int, quality Quality) (*DFTStage[F], error) {
	if factor < 1 {
		return nil, fmt.Errorf("upsampling factor must be >= 1: %d", factor)
	}

	ops := simdops.For[F]()

	if factor == 1 {
		// No upsampling needed, pass-through
		return &DFTStage[F]{factor: 1, ops: ops}, nil
	}

	// Design lowpass filter for upsampling (always in float64 for precision)
	// For L× upsampling, we want to pass the original signal (0 to original Nyquist)
	// and block images that appear above the original Nyquist.
	//
	// In normalized frequency (0.5 = output Nyquist):
	// - Original Nyquist = 0.5/L of output Nyquist
	// - For L=2: original content occupies 0 to 0.25 of output Nyquist
	//
	// soxr uses Fc=0.4778321 for the DFT stage with L=2, which in their
	// normalization (1.0 = Nyquist) means cutoff at 0.4778 of Nyquist.
	// In our normalization (0.5 = Nyquist), that's 0.4778 * 0.5 = 0.2389.
	//
	// For factor=2: cutoff = 0.4778 / 2 = 0.2389 (pass up to 0.2389 of sample rate)
	cutoff := soxrDFTStageFc / float64(factor) // soxr's Fc scaled for our normalization
	transitionBW := transitionBWFactor / float64(factor)
	attenuation := qualityToAttenuation(quality)

	coeffs, err := filter.DesignLowPassFilterAuto(cutoff, transitionBW, attenuation, 1.0)
	if err != nil {
		return nil, fmt.Errorf("failed to design DFT filter: %w", err)
	}

	// Decompose into polyphase filter bank
	// coeffs[phase][tap] = prototype[tap * factor + phase]
	// This avoids multiplying by zeros after zero-insertion
	// Coefficients stored in REVERSED order for ConvolveValid
	tapsPerPhase := (len(coeffs) + factor - 1) / factor

	// Convert coefficients to target precision F
	polyCoeffs := make([][]F, factor)
	for phase := range factor {
		polyCoeffs[phase] = make([]F, tapsPerPhase)
		for tap := range tapsPerPhase {
			protoIdx := tap*factor + phase
			if protoIdx < len(coeffs) {
				// Scale by factor and store reversed
				polyCoeffs[phase][tapsPerPhase-1-tap] = F(coeffs[protoIdx] * float64(factor))
			}
		}
	}

	// Pre-allocate phase buffers for ConvolveValid
	phaseBufs := make([][]F, factor)

	// Detect half-band structure for 2× upsampling
	// In a half-band filter, Phase 0 has only one significant coefficient (center tap ≈ 1.0)
	isHalfBand := false
	phase0TapOffset := 0
	var phase0TapScale F = 1.0

	if factor == halfBandFactor {
		// Check if Phase 0 is essentially a passthrough
		const halfBandThreshold = 1e-8
		significantTaps := 0
		significantIdx := 0
		var significantVal F

		for i, c := range polyCoeffs[0] {
			if math.Abs(float64(c)) > halfBandThreshold {
				significantTaps++
				significantIdx = i
				significantVal = c
			}
		}

		// If only one significant tap and it's close to 1.0, this is half-band
		if significantTaps == 1 && math.Abs(float64(significantVal)-1.0) < 0.01 {
			isHalfBand = true
			phase0TapOffset = significantIdx
			phase0TapScale = significantVal
		}
	}

	return &DFTStage[F]{
		factor:          factor,
		polyCoeffs:      polyCoeffs,
		tapsPerPhase:    tapsPerPhase,
		isHalfBand:      isHalfBand,
		phase0TapOffset: phase0TapOffset,
		phase0TapScale:  phase0TapScale,
		history:         make([]F, 0, tapsPerPhase*historyBufferMultiplier),
		phaseBufs:       phaseBufs,
		ops:             ops,
	}, nil
}

// Process upsamples the input using polyphase FIR filtering.
// Uses polyphase decomposition to avoid multiplying by zeros.
// Optimized for cache efficiency: processes in L2-sized chunks for large inputs.
func (s *DFTStage[F]) Process(input []F) ([]F, error) {
	if s.factor == 1 {
		// Pass-through
		return input, nil
	}

	inputLen := len(input)
	if inputLen == 0 {
		return []F{}, nil
	}

	// Append input to history (no zero-insertion needed!)
	s.history = append(s.history, input...)

	numAvailable := len(s.history)
	if numAvailable < s.tapsPerPhase {
		return []F{}, nil
	}

	// Number of input samples we can fully process
	numInputProcessable := numAvailable - s.tapsPerPhase + 1
	// Each input sample produces 'factor' output samples
	numOutput := numInputProcessable * s.factor

	// Reuse output buffer to reduce allocations
	if cap(s.outputBuf) < numOutput {
		s.outputBuf = make([]F, numOutput)
	} else {
		s.outputBuf = s.outputBuf[:numOutput]
	}

	factor := s.factor
	history := s.history
	output := s.outputBuf
	tapsPerPhase := s.tapsPerPhase

	// For small inputs, use simple non-chunked path (less overhead)
	if numInputProcessable <= l2CacheChunkSize {
		s.processChunk(history, output, numInputProcessable, factor, tapsPerPhase)
	} else {
		// Large input: process in L2-cache-friendly chunks
		// This keeps working set in L2 cache for better performance
		s.processChunked(history, output, numInputProcessable, factor, tapsPerPhase)
	}

	// Shift history - keep only unprocessed samples
	if numInputProcessable > 0 {
		copy(s.history, s.history[numInputProcessable:])
		s.history = s.history[:numAvailable-numInputProcessable]
	}

	// Return a copy to prevent caller's slice from being corrupted
	// if they call Process() or Flush() again (which reuses s.outputBuf)
	result := make([]F, len(output))
	copy(result, output)
	return result, nil
}

// processChunk processes a single chunk of input data.
// This is the simple path for small inputs.
// Optimized with half-band detection, ConvolveValidMulti and Interleave2.
func (s *DFTStage[F]) processChunk(history, output []F, numInputProcessable, factor, tapsPerPhase int) {
	// Ensure phase buffers are large enough
	for phase := range factor {
		if cap(s.phaseBufs[phase]) < numInputProcessable {
			s.phaseBufs[phase] = make([]F, numInputProcessable)
		} else {
			s.phaseBufs[phase] = s.phaseBufs[phase][:numInputProcessable]
		}
	}

	historySlice := history[:numInputProcessable+tapsPerPhase-1]

	// Half-band optimization: Phase 0 is a passthrough (single tap ≈ 1.0)
	// Skip convolution for Phase 0, just copy with offset - saves ~50% compute
	if s.isHalfBand && factor == halfBandFactor {
		// Phase 0: direct copy from history (passthrough)
		offset := s.phase0TapOffset
		scale := s.phase0TapScale
		phase0Buf := s.phaseBufs[0]
		for i := range numInputProcessable {
			phase0Buf[i] = historySlice[i+offset] * scale
		}

		// Phase 1: still needs convolution
		s.ops.ConvolveValid(s.phaseBufs[1], historySlice, s.polyCoeffs[1])

		// Interleave using SIMD
		s.ops.Interleave2(output, s.phaseBufs[0], s.phaseBufs[1])
	} else {
		// Standard path: ConvolveValidMulti for all phases
		s.ops.ConvolveValidMulti(s.phaseBufs, historySlice, s.polyCoeffs)

		// Interleave phase outputs
		if factor == halfBandFactor {
			s.ops.Interleave2(output, s.phaseBufs[0], s.phaseBufs[1])
		} else {
			for i := range numInputProcessable {
				outBase := i * factor
				for phase := range factor {
					output[outBase+phase] = s.phaseBufs[phase][i]
				}
			}
		}
	}
}

// processChunked processes input in L2-cache-friendly chunks.
// For large inputs, this provides better cache utilization than processing all at once.
// Optimized with half-band detection, ConvolveValidMulti and Interleave2.
func (s *DFTStage[F]) processChunked(history, output []F, numInputProcessable, factor, tapsPerPhase int) {
	chunkSize := l2CacheChunkSize

	// Ensure phase buffers are large enough for chunk processing
	for phase := range factor {
		if cap(s.phaseBufs[phase]) < chunkSize {
			s.phaseBufs[phase] = make([]F, chunkSize)
		}
	}

	// Cache half-band parameters for the loop
	isHalfBand := s.isHalfBand && factor == halfBandFactor
	offset := s.phase0TapOffset
	scale := s.phase0TapScale

	// Process in chunks
	for chunkStart := 0; chunkStart < numInputProcessable; chunkStart += chunkSize {
		chunkEnd := min(chunkStart+chunkSize, numInputProcessable)
		chunkLen := chunkEnd - chunkStart

		// Slice of history for this chunk (includes tapsPerPhase-1 extra for convolution)
		historySlice := history[chunkStart : chunkEnd+tapsPerPhase-1]

		// Resize phase buffers for this chunk
		for phase := range factor {
			s.phaseBufs[phase] = s.phaseBufs[phase][:chunkLen]
		}

		// Half-band optimization: Phase 0 passthrough saves ~50% compute
		if isHalfBand {
			// Phase 0: direct copy from history (passthrough)
			phase0Buf := s.phaseBufs[0]
			for i := range chunkLen {
				phase0Buf[i] = historySlice[i+offset] * scale
			}

			// Phase 1: still needs convolution
			s.ops.ConvolveValid(s.phaseBufs[1], historySlice, s.polyCoeffs[1])

			// Interleave using SIMD
			outOffset := chunkStart * halfBandFactor
			s.ops.Interleave2(output[outOffset:outOffset+chunkLen*halfBandFactor], s.phaseBufs[0], s.phaseBufs[1])
		} else {
			// Standard path: ConvolveValidMulti for all phases
			s.ops.ConvolveValidMulti(s.phaseBufs, historySlice, s.polyCoeffs)

			// Interleave results directly into output
			outOffset := chunkStart * factor
			if factor == halfBandFactor {
				s.ops.Interleave2(output[outOffset:outOffset+chunkLen*halfBandFactor], s.phaseBufs[0], s.phaseBufs[1])
			} else {
				for i := range chunkLen {
					outBase := outOffset + i*factor
					for phase := range factor {
						output[outBase+phase] = s.phaseBufs[phase][i]
					}
				}
			}
		}
	}
}

// Flush returns any remaining buffered samples.
func (s *DFTStage[F]) Flush() ([]F, error) {
	if s.factor == 1 || len(s.history) == 0 {
		return []F{}, nil
	}

	// Pad with zeros to flush pipeline
	zeros := make([]F, s.tapsPerPhase)
	return s.Process(zeros)
}

// Reset clears internal state.
func (s *DFTStage[F]) Reset() {
	s.history = s.history[:0]
}

// =============================================================================
// Polyphase Stage - soxr-style integer division/modulo algorithm
// =============================================================================

// PolyphaseStage implements polyphase FIR resampling using soxr's algorithm.
//
// Type parameter F controls the precision of sample processing.
//
// This uses sub-phase coefficient interpolation matching soxr's poly-fir.h:
// - Fixed-point phase accumulator with 32-bit fractional precision
// - Cubic polynomial interpolation between phases: coef(x) = a + x*(b + x*(c + x*d))
// - This provides smooth coefficient transitions and excellent THD at high frequencies
type PolyphaseStage[F simdops.Float] struct {
	// Filter coefficients with cubic interpolation support
	// polyCoeffs[phase][tap] - base coefficient (a)
	// polyCoeffsB/C/D[phase][tap] - cubic interpolation coefficients
	polyCoeffs  [][]F
	polyCoeffsB [][]F // Linear coefficient (b)
	polyCoeffsC [][]F // Quadratic coefficient (c)
	polyCoeffsD [][]F // Cubic coefficient (d)

	numPhases    int // L in soxr
	tapsPerPhase int // Number of taps per phase

	// Precomputed loop bounds (avoid recalculating each iteration)
	tapsPerPhase4 int // tapsPerPhase &^ 3, for loop unrolling

	// Phase accumulator - fixed-point with 32-bit fractional precision (like soxr)
	// at = integer_part * (1 << phaseFracBits) + fractional_part
	at   int64 // Current position in fixed-point
	step int64 // Step per output sample in fixed-point

	// Phase extraction constants
	phaseFracBits int   // Number of bits for sub-phase interpolation
	phaseFracMask int64 // Mask for extracting fractional bits

	// Input history buffer
	history []F

	// Pre-allocated output buffer for reduced allocations
	outputBuf []F

	// SIMD operations for type F
	ops *simdops.Ops[F]

	// Statistics
	samplesIn  int64
	samplesOut int64
}

// NewPolyphaseStage creates a polyphase resampling stage.
//
// Parameters:
//   - ratio: Output/input ratio for this stage (e.g., 1.0884 for 88.2→96 kHz)
//   - totalIORatio: Total input/output ratio (e.g., 0.459 for 44.1→96 kHz)
//     This is used to correctly set Fp1 for anti-imaging filter design.
//   - hasPreStage: Whether this polyphase stage is preceded by a DFT pre-stage.
//     This affects the Fn/Fs calculation (soxr uses different formulas).
//   - quality: Quality level
func NewPolyphaseStage[F simdops.Float](ratio, totalIORatio float64, hasPreStage bool, quality Quality) (*PolyphaseStage[F], error) {
	if ratio <= 0 {
		return nil, fmt.Errorf("ratio must be positive: %f", ratio)
	}

	ops := simdops.For[F]()

	// Find rational approximation for the ratio
	// We want: ratio ≈ L / step (output samples per input sample)
	// So: step / L ≈ 1 / ratio
	numPhases, stepInt := findRationalApprox(ratio)

	// Design polyphase filter bank (always in float64 for precision)
	// Pass totalIORatio and hasPreStage for correct Fp1/Fn calculation (soxr uses total ratio)
	filterBank, err := designPolyphaseFilter(numPhases, ratio, totalIORatio, hasPreStage, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to design polyphase filter: %w", err)
	}

	tapsPerPhase := filterBank.tapsPerPhase

	// Sub-phase interpolation configuration (matching soxr's poly-fir.h)
	// We use 16 bits for sub-phase precision (65536 sub-phases per phase)
	// This provides excellent THD at high frequencies while keeping integer math fast
	const phaseFracBits = 16
	const phaseFracMask = (1 << phaseFracBits) - 1

	// Compute step as a true fixed-point number with full fractional precision
	// step = (1/ratio) * numPhases * (1 << phaseFracBits)
	// This is CRITICAL: using stepInt would lose fractional precision and make
	// sub-phase interpolation useless (frac would always be 0)
	_ = stepInt // Keep for rational approximation validation (used in filter design)
	phaseFracScale := float64(int64(1) << phaseFracBits)
	step := int64(math.Round((1.0 / ratio) * float64(numPhases) * phaseFracScale))

	// Helper function to get prototype coefficient with wrap-around for interpolation
	getCoeff := func(phase, tap int) float64 {
		// Wrap phase around for interpolation at boundaries
		wrappedPhase := phase % numPhases
		if wrappedPhase < 0 {
			wrappedPhase += numPhases
		}
		idx := tap*numPhases + wrappedPhase
		if idx < 0 || idx >= len(filterBank.coeffs) {
			return 0.0
		}
		return filterBank.coeffs[idx]
	}

	// Allocate coefficient arrays with cubic interpolation support
	// polyCoeffs = a (base), polyCoeffsB = b (linear), polyCoeffsC = c (quadratic), polyCoeffsD = d (cubic)
	// Interpolation formula: coef(x) = a + x*(b + x*(c + x*d)) where x ∈ [0, 1)
	polyCoeffs := make([][]F, numPhases)
	polyCoeffsB := make([][]F, numPhases)
	polyCoeffsC := make([][]F, numPhases)
	polyCoeffsD := make([][]F, numPhases)

	for phase := range numPhases {
		polyCoeffs[phase] = make([]F, tapsPerPhase)
		polyCoeffsB[phase] = make([]F, tapsPerPhase)
		polyCoeffsC[phase] = make([]F, tapsPerPhase)
		polyCoeffsD[phase] = make([]F, tapsPerPhase)

		for tap := range tapsPerPhase {
			// Get coefficients from adjacent phases for cubic interpolation
			// f0 = current phase, f1 = next phase, fm1 = previous phase, f2 = next-next phase
			f0 := getCoeff(phase, tap)
			f1 := getCoeff(phase+1, tap)
			fm1 := getCoeff(phase-1, tap)
			f2 := getCoeff(phase+cubicPhaseOffset, tap)

			// Compute cubic interpolation coefficients (Catmull-Rom style)
			// These allow smooth interpolation: f(x) = a + b*x + c*x² + d*x³
			a := f0
			c := cubicCenterCoeff*(f1+fm1) - f0
			d := (1.0 / cubicDivisor) * (f2 - f1 + fm1 - f0 - cubicCMultiplier*c)
			b := f1 - f0 - d - c

			// Store in REVERSED order for correct convolution direction
			revTap := tapsPerPhase - 1 - tap
			polyCoeffs[phase][revTap] = F(a)
			polyCoeffsB[phase][revTap] = F(b)
			polyCoeffsC[phase][revTap] = F(c)
			polyCoeffsD[phase][revTap] = F(d)
		}
	}

	return &PolyphaseStage[F]{
		polyCoeffs:    polyCoeffs,
		polyCoeffsB:   polyCoeffsB,
		polyCoeffsC:   polyCoeffsC,
		polyCoeffsD:   polyCoeffsD,
		numPhases:     numPhases,
		tapsPerPhase:  tapsPerPhase,
		tapsPerPhase4: tapsPerPhase &^ loopUnrollMask,
		at:            0,
		step:          step,
		phaseFracBits: phaseFracBits,
		phaseFracMask: phaseFracMask,
		history:       make([]F, 0, tapsPerPhase*historyBufferMultiplier),
		ops:           ops,
	}, nil
}

// Process resamples input using soxr's polyphase algorithm with cubic coefficient interpolation.
//
// This implements the core loop from soxr's poly-fir.h with sub-phase interpolation:
//
//	for i = 0; at < num_in * L * (1<<fracBits); i++, at += step {
//	    div := at >> (fracBits + phaseBits)      // Input sample index
//	    phase := (at >> fracBits) & phaseMask    // Integer phase index
//	    x := (at & fracMask) / (1<<fracBits)     // Fractional phase [0, 1)
//	    output[i] = convolve_interpolated(input[div:], coeffs[phase], x)
//	}
//
// The cubic interpolation formula per coefficient: coef(x) = a + x*(b + x*(c + x*d))
func (s *PolyphaseStage[F]) Process(input []F) ([]F, error) {
	if len(input) == 0 {
		return []F{}, nil
	}

	s.samplesIn += int64(len(input))

	// Append input to history
	s.history = append(s.history, input...)

	numIn := len(s.history) - s.tapsPerPhase + 1
	if numIn <= 0 {
		return []F{}, nil
	}

	// Calculate number of output samples
	// at is in fixed-point: integer_phase * (1 << phaseFracBits) + fractional
	// limit = numIn * numPhases * (1 << phaseFracBits)
	numPhases64 := int64(s.numPhases)
	phaseFracBits := s.phaseFracBits
	limit := int64(numIn) * numPhases64 << phaseFracBits
	numOut := int((limit - s.at + s.step - 1) / s.step)
	if numOut <= 0 {
		return []F{}, nil
	}

	// Reuse output buffer to reduce allocations
	if cap(s.outputBuf) < numOut {
		s.outputBuf = make([]F, numOut)
	} else {
		s.outputBuf = s.outputBuf[:numOut]
	}

	// Hoist invariants out of the loop for better optimization
	polyCoeffs := s.polyCoeffs
	polyCoeffsB := s.polyCoeffsB
	polyCoeffsC := s.polyCoeffsC
	polyCoeffsD := s.polyCoeffsD
	history := s.history
	numPhases := s.numPhases
	tapsPerPhase := s.tapsPerPhase
	step := s.step
	histLen := len(history)
	phaseFracMask := s.phaseFracMask

	// Precompute scale factor for converting fractional bits to [0, 1)
	fracScale := F(1.0 / float64(int64(1)<<phaseFracBits))

	// Main resampling loop with cubic coefficient interpolation
	at := s.at
	outIdx := 0
	for at < limit {
		// Extract integer phase and fractional sub-phase from fixed-point accumulator
		// at = (input_sample * numPhases + integer_phase) << phaseFracBits + frac
		fullPhase := at >> phaseFracBits      // input_sample * numPhases + integer_phase
		div := int(fullPhase / numPhases64)   // Input sample index
		phase := int(fullPhase % numPhases64) // Integer phase index (0 to numPhases-1)
		frac := at & phaseFracMask            // Fractional phase (0 to phaseFracMask)
		x := F(frac) * fracScale              // Fractional phase normalized to [0, 1)

		// Boundary check
		if div+tapsPerPhase > histLen {
			break
		}

		// Convolve with cubic coefficient interpolation using SIMD
		// Computes: sum = Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
		coeffsA := polyCoeffs[phase]
		coeffsB := polyCoeffsB[phase]
		coeffsC := polyCoeffsC[phase]
		coeffsD := polyCoeffsD[phase]
		hist := history[div : div+tapsPerPhase]

		sum := s.ops.CubicInterpDot(hist, coeffsA, coeffsB, coeffsC, coeffsD, x)

		s.outputBuf[outIdx] = sum
		outIdx++
		at += step
	}

	// Trim output to actual size produced
	output := s.outputBuf[:outIdx]

	// Consume processed samples from history
	consumed := int(at>>phaseFracBits) / numPhases
	if consumed > 0 && consumed <= histLen {
		copy(s.history, s.history[consumed:])
		s.history = s.history[:histLen-consumed]
	}

	// Save remainder for next call
	// Keep the fractional part within one input sample
	s.at = at - int64(consumed*numPhases)<<phaseFracBits

	s.samplesOut += int64(len(output))

	// Return a copy to prevent caller's slice from being corrupted
	result := make([]F, len(output))
	copy(result, output)
	return result, nil
}

// Flush returns any remaining buffered samples.
func (s *PolyphaseStage[F]) Flush() ([]F, error) {
	zeros := make([]F, s.tapsPerPhase*historyBufferMultiplier)
	return s.Process(zeros)
}

// Reset clears internal state.
func (s *PolyphaseStage[F]) Reset() {
	s.at = 0
	s.history = s.history[:0]
	s.samplesIn = 0
	s.samplesOut = 0
}

// GetStatistics returns processing statistics.
func (s *PolyphaseStage[F]) GetStatistics() map[string]int64 {
	return map[string]int64{
		"samplesIn":  s.samplesIn,
		"samplesOut": s.samplesOut,
	}
}

// =============================================================================
// Polyphase Filter Design
// =============================================================================

// polyphaseFilter holds filter coefficients for polyphase resampling.
type polyphaseFilter struct {
	coeffs       []float64
	numPhases    int
	tapsPerPhase int
}

// designPolyphaseFilter creates a polyphase filter bank matching soxr.
//
// Key design points (from soxr analysis):
//   - Number of taps calculated dynamically using Kaiser formula: ceil(att/tr_bw + 1)
//   - Prototype filter DC gain = numPhases
//   - Each phase independently has DC gain ≈ 1.0
//   - Cutoff frequency calculated using soxr's Fp/Fs methodology
//
// The polyphase stage may or may not be preceded by a DFT pre-stage.
// This implementation follows soxr's cr.c and filter.c filter design exactly:
//   - For downsampling WITH pre-stage: Fn = 2 * mult, Fs = 3 + |Fs1 - 1|
//   - For upsampling OR no pre-stage: Fn = 1, Fs = 2 - (Fp1 + (Fs1 - Fp1) * 0.7)
//   - Fp and Fs normalized by Fn before calculating Fc
//   - Fp adjusted using lsx_inv_f_resp for rolloff compensation
//
// Parameters:
//   - numPhases: number of polyphase filter phases
//   - ratio: polyphase stage ratio (output/input for this stage)
//   - totalIORatio: total input/output ratio for the full resampler (used for Fp1 calculation)
//   - hasPreStage: whether this stage is preceded by a DFT pre-stage
//   - quality: quality level
func designPolyphaseFilter(numPhases int, ratio, totalIORatio float64, hasPreStage bool, quality Quality) (*polyphaseFilter, error) {
	attenuation := qualityToAttenuation(quality)

	// Use the new testable parameter computation function
	params := ComputePolyphaseFilterParams(numPhases, ratio, totalIORatio, hasPreStage, attenuation)

	// Convert Fc to our filter design normalization ([0, 0.5] where 0.5 = Nyquist)
	// params.Fc is already normalized by Fn and phases, but in soxr's [0, 1] scale
	// Our filter design uses [0, 0.5], so divide by 2
	cutoff := params.Fc / soxrToOurNormScale

	// Ensure cutoff is valid (within 0 to 0.5 normalized range for lowpass)
	if cutoff <= 0 {
		cutoff = 0.001 // Minimum valid cutoff
	}
	if cutoff >= nyquistFraction {
		cutoff = 0.499 // Maximum valid cutoff (just below Nyquist)
	}

	// Design prototype using Kaiser window
	prototype, err := filter.DesignLowPassFilter(filter.FilterParams{
		NumTaps:     params.TotalTaps,
		CutoffFreq:  cutoff,
		Attenuation: attenuation,
		Gain:        1.0,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to design prototype filter: %w", err)
	}

	// Normalize prototype so that DC gain = numPhases
	// This ensures each phase has DC gain ≈ 1.0
	// Uses SIMD-accelerated sum and scale operations
	sum := f64.Sum(prototype)
	if sum != 0 {
		scale := float64(numPhases) / sum
		f64.Scale(prototype, prototype, scale)
	}

	// Decompose into polyphase branches
	// Coefficient layout: coeffs[tap * numPhases + phase]
	coeffs := make([]float64, params.TapsPerPhase*numPhases)
	for tap := range params.TapsPerPhase {
		for phase := range numPhases {
			protoIdx := tap*numPhases + phase
			if protoIdx < len(prototype) {
				coeffs[tap*numPhases+phase] = prototype[protoIdx]
			}
		}
	}

	return &polyphaseFilter{
		coeffs:       coeffs,
		numPhases:    numPhases,
		tapsPerPhase: params.TapsPerPhase,
	}, nil
}

// findRationalApprox finds a rational approximation for the ratio.
//
// We want: ratio ≈ L / step
// For soxr-style, we find L (numPhases) and step such that step/L ≈ 1/ratio
//
// Returns: numPhases (L), step (arbM)
func findRationalApprox(ratio float64) (numPhases, step int) {
	// For ratios close to 1, use 80 phases (matching soxr's choice for CD→DAT)
	// For other ratios, find a good approximation

	// Default: 80 phases (soxr's choice)
	const defaultPhases = 80
	const maxPhases = 256

	// Try to find exact rational representation
	invRatio := 1.0 / ratio

	// Search for best L and step
	bestL := defaultPhases
	bestStep := int(math.Round(invRatio * float64(defaultPhases)))
	bestError := math.Abs(float64(bestStep)/float64(bestL) - invRatio)

	// Try other phase counts
	for L := 64; L <= maxPhases; L++ {
		candidateStep := int(math.Round(invRatio * float64(L)))
		if candidateStep <= 0 {
			continue
		}
		err := math.Abs(float64(candidateStep)/float64(L) - invRatio)
		if err < bestError {
			bestL = L
			bestStep = candidateStep
			bestError = err
		}
		// If we found an exact match, stop
		if bestError < rationalApproxTolerance {
			break
		}
	}

	return bestL, bestStep
}

// =============================================================================
// soxr Filter Response Functions
// =============================================================================

// lsxInvFResp computes the inverse frequency response, matching soxr's lsx_inv_f_resp.
//
// This function returns the normalized frequency at which the filter response
// has dropped by 'drop' dB, given a target stopband attenuation of 'a' dB.
//
// From soxr's filter.c:
//
//	double lsx_inv_f_resp(double drop, double a) {
//	  double x = sinePhi(a), s;
//	  drop = dB_to_linear(drop);
//	  s = drop > .5 ? 1 - drop : drop;
//	  x = asin(pow(s, 1/sinePow(x))) / x;
//	  return drop > .5? x : 1 -x;
//	}
//
// Parameters:
//   - drop: dB level to find (e.g., -0.01 for the rolloff point)
//   - a: stopband attenuation in dB (e.g., 180 for high quality)
//
// Returns: normalized frequency (0 to 1) where response equals 'drop' dB
func lsxInvFResp(drop, a float64) float64 {
	// Guard: clamp attenuation to valid range to prevent NaN
	// Very low values cause polynomial issues; very high values make x*0.5 > π
	if a < minAttenuation {
		a = minAttenuation
	} else if a > maxAttenuation {
		a = maxAttenuation
	}

	// sinePhi(a) = ((a3*a + a2)*a + a1)*a + a0
	x := ((sinePhiCoeffA3*a+sinePhiCoeffA2)*a+sinePhiCoeffA1)*a + sinePhiConstant

	// dB_to_linear(drop) = exp(drop * ln(10) * 0.05)
	dropLinear := math.Exp(drop * math.Ln10 * dbToLinearFactor)

	// s = drop > 0.5 ? 1 - drop : drop
	var s float64
	if dropLinear > halfAmplitude {
		s = 1 - dropLinear
	} else {
		s = dropLinear
	}

	// sinePow(x) = log(0.5) / log(sin(x * 0.5))
	// Guard: ensure sin(x*0.5) > 0 to avoid log(0) or log(negative) -> NaN
	sinVal := math.Sin(x * halfAmplitude)
	if sinVal <= sineEpsilon {
		sinVal = sineEpsilon
	}
	sinePow := math.Log(halfAmplitude) / math.Log(sinVal)

	// x = asin(pow(s, 1/sinePow(x))) / x
	x = math.Asin(math.Pow(s, 1.0/sinePow)) / x

	// return drop > 0.5 ? x : 1 - x
	if dropLinear > halfAmplitude {
		return x
	}
	return 1 - x
}

// =============================================================================
// Filter Parameter Computation (Testable)
// =============================================================================

// PolyphaseFilterParams holds the computed filter design parameters.
// This struct is exported for testing purposes.
type PolyphaseFilterParams struct {
	// Input parameters
	NumPhases    int     // Number of polyphase phases
	Ratio        float64 // Polyphase stage ratio (output/input for this stage)
	TotalIORatio float64 // Total input/output ratio for the full resampler
	HasPreStage  bool    // Whether preceded by a DFT pre-stage
	Attenuation  float64 // Stopband attenuation in dB

	// Computed intermediate values (for debugging/testing)
	IsUpsampling bool    // true if overall upsampling operation
	Mult         float64 // Decimation multiplier (1.0 for upsampling, >1 for downsampling)
	Fn           float64 // Nyquist normalization factor
	Fp1          float64 // Initial passband edge (before Fn normalization)
	Fs1          float64 // Stopband reference (Nyquist = 0.5)
	FpRaw        float64 // Passband after rolloff adjustment, before Fn normalization
	FsRaw        float64 // Stopband before Fn normalization

	// Normalized parameters (after Fn normalization)
	Fp   float64 // Normalized passband edge
	Fs   float64 // Normalized stopband edge
	TrBw float64 // Transition bandwidth (after phase division)
	Fc   float64 // Final cutoff frequency for filter design

	// Filter sizing
	TotalTaps    int // Total filter taps
	TapsPerPhase int // Taps per polyphase phase
}

// ComputePolyphaseFilterParams computes filter design parameters following soxr's methodology.
//
// This function implements the critical Fn normalization from soxr's cr.c and filter.c:
//   - For downsampling WITH pre-stage: Fn = 2 * mult, Fs = 3 + |Fs1 - 1|
//   - For upsampling OR no pre-stage: Fn = 1, Fs = 2 - (Fp1 + (Fs1 - Fp1) * 0.7)
//   - Fp and Fs are normalized by dividing by Fn before computing Fc
//
// Parameters:
//   - numPhases: number of polyphase filter phases
//   - ratio: polyphase stage ratio (output/input for this stage)
//   - totalIORatio: total input/output ratio (input_rate/output_rate)
//   - hasPreStage: whether this polyphase stage is preceded by a DFT pre-stage
//   - attenuation: desired stopband attenuation in dB
//
// Returns computed filter parameters for filter design.
func ComputePolyphaseFilterParams(numPhases int, ratio, totalIORatio float64, hasPreStage bool, attenuation float64) PolyphaseFilterParams {
	params := PolyphaseFilterParams{
		NumPhases:    numPhases,
		Ratio:        ratio,
		TotalIORatio: totalIORatio,
		HasPreStage:  hasPreStage,
		Attenuation:  attenuation,
		Fs1:          nyquistFraction, // 0.5
	}

	phases := float64(numPhases)
	params.IsUpsampling = totalIORatio < 1.0

	// Compute mult: the decimation multiplier
	// For upsampling: mult = 1
	// For downsampling: mult = totalIORatio (input/output ratio)
	// This matches soxr's: mult = upsample? 1 : arbM / arbL
	if params.IsUpsampling {
		params.Mult = 1.0
	} else {
		params.Mult = totalIORatio
	}

	// Compute initial passband edge Fp1
	// From soxr: Fp1 = p->io_ratio * (rolloff < 0 ? 1 : (1 - .05 * pow(5., rolloff)))
	// With rolloff=0: Fp1 = io_ratio * 0.95 (we use 0.99 for high quality)
	if params.IsUpsampling {
		// Upsampling: passband at original Nyquist scaled by io_ratio
		params.Fp1 = totalIORatio * passbandRolloffScale
	} else {
		// Downsampling: passband at output Nyquist
		// output_Nyquist = 0.5 * (output_rate/input_rate) = 0.5 / totalIORatio
		// But in normalized form where input Nyquist = 0.5:
		// Fp1 = 0.5 * ratio * 0.99 (ratio = output/input for polyphase stage)
		params.Fp1 = nyquistFraction * ratio * passbandRolloffScale
	}

	// Compute Fn and Fs based on upsampling/downsampling AND presence of pre-stage
	// This is the CRITICAL logic from soxr cr.c lines 429-431:
	//
	//   if (!upsample && preM)
	//     Fn = 2 * mult, Fs = 3 + fabs(Fs1 - 1);
	//   else
	//     Fn = 1, Fs = 2 - (mode? Fp1 + (Fs1 - Fp1) * .7 : Fs1);
	//
	// Key insight: The Fn=2*mult formula is ONLY used when:
	//   1. Downsampling (!upsample), AND
	//   2. There IS a pre-stage (preM != 0)
	//
	// For downsampling WITHOUT pre-stage, we need ANTI-ALIASING filter parameters
	// that cut off at the OUTPUT Nyquist frequency, not use the upsampling formula.
	if !params.IsUpsampling && hasPreStage {
		// Downsampling WITH pre-stage: Fn = 2 * mult, Fs = 3 + |Fs1 - 1|
		params.Fn = soxrDownsamplingFnFactor * params.Mult
		// Fs = 3 + |0.5 - 1| = 3.5
		params.FsRaw = soxrDownsamplingFsBase + math.Abs(params.Fs1-1.0)
		params.FpRaw = params.Fp1
	} else {
		// Upsampling: anti-imaging formula
		params.Fn = 1.0
		// Fs = 2 - (Fp1 + (Fs1 - Fp1) * 0.7) for mode > 0
		params.FsRaw = imageRejectionFactor - (params.Fp1 + (params.Fs1-params.Fp1)*soxrUpsamplingFsCoeff)
		params.FpRaw = params.Fp1
	}

	// Apply rolloff compensation using lsxInvFResp
	// From soxr: Fp = Fs - (Fs - Fp) / (1 - lsx_inv_f_resp(rolloffs[rolloff], attArb))
	// rolloffs[0] = -0.01 for high quality
	//
	// NOTE: This adjustment is primarily designed for upsampling where Fp1 and Fs
	// are relatively close. For downsampling with Fs = 3.5 and small Fp1, this
	// formula can produce negative values, which is invalid.
	// We only apply the adjustment when it produces a valid (positive) result.
	invFResp := lsxInvFResp(-0.01, attenuation)
	if invFResp < invFRespThreshold { // Guard against division by zero
		adjustedFp := params.FsRaw - (params.FsRaw-params.FpRaw)/(1.0-invFResp)
		// Only apply if result is positive and less than FsRaw
		if adjustedFp > 0 && adjustedFp < params.FsRaw {
			params.FpRaw = adjustedFp
		}
		// For downsampling with large Fs, keep original Fp1 as FpRaw
	}

	// Normalize by Fn (CRITICAL: this was missing in the old implementation!)
	// From soxr filter.c: Fp /= fabs(Fn), Fs /= fabs(Fn)
	params.Fp = params.FpRaw / math.Abs(params.Fn)
	params.Fs = params.FsRaw / math.Abs(params.Fn)

	// Calculate transition bandwidth following soxr's filter.c:
	//   tr_bw = 0.5 * (Fs - Fp)
	//   tr_bw /= phases
	//   tr_bw = min(tr_bw, 0.5 * Fs / phases)
	//   Fc = Fs - tr_bw
	params.TrBw = transitionBandwidthHalf * (params.Fs - params.Fp)
	params.TrBw /= phases

	trBwLimit := transitionBandwidthHalf * params.Fs / phases
	if params.TrBw > trBwLimit {
		params.TrBw = trBwLimit
	}

	// Enforce minimum transition bandwidth for practical filter lengths
	const minTrBw = 0.001 // More permissive than before since Fn normalization fixes the main issue
	if params.TrBw < minTrBw {
		params.TrBw = minTrBw
	}

	// Fc = Fs / phases - tr_bw (following soxr's filter.c)
	// After phase division: Fs_phase = Fs / phases, then Fc = Fs_phase - tr_bw
	fsPhase := params.Fs / phases
	params.Fc = fsPhase - params.TrBw

	// Ensure Fc is positive and reasonable
	if params.Fc < minTrBw {
		params.Fc = minTrBw
	}

	// Calculate filter taps using Kaiser formula
	// From soxr: num_taps = ceil(att/tr_bw + 1)
	const (
		minTapsPerPhase = 8
		filterLibLimit  = 8191 - 1 // Hard limit from filter library (minus 1 for safety)
		minAtten        = 80.0

		// Scale max taps per phase with attenuation to respect quality differences.
		// These thresholds allow higher quality to use more taps.
		lowQualityAttenuation      = 110.0 // ~16-bit, ~102 dB
		highQualityAttenuation     = 130.0 // ~20-bit, ~126 dB
		veryHighQualityAttenuation = 160.0 // ~28-bit, ~175 dB

		maxTapsLowQuality      = 32  // Max taps per phase for low quality
		maxTapsHighQuality     = 64  // Max taps per phase for high quality
		maxTapsVeryHighQuality = 100 // Max taps per phase for very high quality
	)

	// Determine max taps per phase based on attenuation (quality level)
	var maxTapsPerPhase int
	switch {
	case attenuation < lowQualityAttenuation:
		maxTapsPerPhase = maxTapsLowQuality
	case attenuation < highQualityAttenuation:
		maxTapsPerPhase = maxTapsHighQuality
	case attenuation < veryHighQualityAttenuation:
		maxTapsPerPhase = maxTapsVeryHighQuality
	default:
		// For very high quality (>160 dB), allow even more taps
		// but still respect the filter library limit
		maxTapsPerPhase = (filterLibLimit + 1) / numPhases
	}

	// Calculate ideal taps from Kaiser formula
	idealTaps := int(math.Ceil(attenuation/params.TrBw + 1))

	// Compute taps per phase, respecting the quality-dependent limit
	params.TotalTaps = idealTaps
	params.TapsPerPhase = (params.TotalTaps + numPhases - 1) / numPhases

	if params.TapsPerPhase < minTapsPerPhase {
		params.TapsPerPhase = minTapsPerPhase
	} else if params.TapsPerPhase > maxTapsPerPhase {
		params.TapsPerPhase = maxTapsPerPhase
	}

	// Calculate final total taps and enforce the filter library's hard limit
	params.TotalTaps = numPhases*params.TapsPerPhase - 1
	if params.TotalTaps > filterLibLimit {
		// Reduce taps per phase to stay within limit
		// filterLibLimit >= numPhases * tapsPerPhase - 1
		// tapsPerPhase <= (filterLibLimit + 1) / numPhases
		params.TapsPerPhase = max((filterLibLimit+1)/numPhases, minTapsPerPhase)
		params.TotalTaps = numPhases*params.TapsPerPhase - 1
	}

	return params
}
