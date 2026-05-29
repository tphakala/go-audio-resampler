package engine

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/filter"
	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

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

// processZeroCopy upsamples the input using polyphase FIR filtering.
// It uses polyphase decomposition to avoid multiplying by zeros.
// For large inputs it processes in L2-sized chunks for cache efficiency.
// The returned slice
// aliases s.outputBuf and is only valid until the next call to processZeroCopy,
// Process, or Flush. Pipeline-internal callers that consume the output
// immediately (e.g. writing it into the next stage's buffer) should use this
// to avoid a defensive copy.
func (s *DFTStage[F]) processZeroCopy(input []F) ([]F, error) { //nolint:unparam // error kept for symmetry with Process
	if s.factor == 1 {
		// Pass-through
		return input, nil
	}

	if len(input) == 0 {
		return []F{}, nil
	}

	// Append input to history (no zero-insertion needed!).
	// appendStable grows with headroom so steady-state streaming does not
	// reallocate when the run length jitters by a sample between calls.
	s.history = appendStable(s.history, input)

	numAvailable := len(s.history)
	if numAvailable < s.tapsPerPhase {
		return []F{}, nil
	}

	// Number of input samples we can fully process
	numInputProcessable := numAvailable - s.tapsPerPhase + 1
	// Each input sample produces 'factor' output samples
	numOutput := numInputProcessable * s.factor

	// Reuse output buffer to reduce allocations. growStableLen adds headroom
	// when it must grow so that numOutput jitter between calls does not
	// reallocate in the steady state.
	s.outputBuf = growStableLen(s.outputBuf, numOutput)

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

	return s.outputBuf[:numOutput], nil
}

// Process resamples input through the DFT upsampling stage. The returned
// slice is owned by the caller and remains valid across subsequent calls.
func (s *DFTStage[F]) Process(input []F) ([]F, error) {
	output, err := s.processZeroCopy(input)
	if err != nil || len(output) == 0 {
		return output, err
	}
	if s.factor == 1 {
		return output, nil
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
// DFT Decimation Stage - soxr-style integer ratio downsampling
// =============================================================================

// DFTDecimationStage implements integer-ratio downsampling using FIR filtering
// followed by decimation. This matches soxr's approach for integer downsample
// ratios like 96kHz→48kHz (2:1) or 192kHz→48kHz (4:1).
//
// The key insight from soxr is that for integer ratio downsampling:
// - Use Fn = decimation_factor to normalize filter frequencies
// - Filter cutoff is at output Nyquist (not input Nyquist)
// - This achieves 150+ dB anti-aliasing attenuation
//
// Type parameter F controls the precision of sample processing.
type DFTDecimationStage[F simdops.Float] struct {
	factor int // Decimation factor (e.g., 2 for 96→48)

	// FIR filter coefficients (stored in reversed order for convolution)
	coeffs  []F
	numTaps int

	// Input history buffer
	history []F

	// Decimation state - which input sample to output next
	decimPhase int

	// Pre-allocated output buffer
	outputBuf []F

	// SIMD operations for type F
	ops *simdops.Ops[F]
}

// NewDFTDecimationStage creates a DFT decimation stage for integer ratio downsampling.
//
// This implements soxr's approach for integer downsampling ratios:
//   - Design filter with Fn = factor (normalization to output Nyquist)
//   - Fp = 0.913 (passband at 91.3% of output Nyquist)
//   - Fs = 1.0 (stopband starts at output Nyquist)
//   - After filtering, decimate by taking every 'factor'-th sample
//
// For 96kHz→48kHz (factor=2):
//   - Filter cutoff at ~22 kHz (just below output Nyquist of 24 kHz)
//   - Achieves 150+ dB anti-aliasing attenuation
func NewDFTDecimationStage[F simdops.Float](factor int, quality Quality) (*DFTDecimationStage[F], error) {
	if factor < 1 {
		return nil, fmt.Errorf("decimation factor must be >= 1: %d", factor)
	}

	ops := simdops.For[F]()

	if factor == 1 {
		// No decimation needed, pass-through
		return &DFTDecimationStage[F]{factor: 1, ops: ops}, nil
	}

	// Design lowpass filter for decimation (always in float64 for precision)
	//
	// soxr filter design for decimation (from SOXR_FILTER_ANALYSIS.md):
	//   Fn = max(preL, preM) = factor (for pure decimation, preL=1, preM=factor)
	//   Fp = 0.913 (VHQ passband, relative to input Nyquist before normalization)
	//   Fs = 1.0 (stopband at output Nyquist, relative to input Nyquist)
	//
	// After normalization by Fn:
	//   Fp_norm = 0.913 / factor
	//   Fs_norm = 1.0 / factor
	//
	// For factor=2: Fp_norm=0.4565, Fs_norm=0.5
	// This places cutoff just below output Nyquist
	//
	// In our filter design (where 0.5 = Nyquist of current sample rate):
	//   cutoff = Fs_norm * 0.5 = 0.5 / factor = 0.25 for factor=2
	//   This means cutoff at 25% of INPUT sample rate = 50% of OUTPUT Nyquist

	// Get quality-specific passband end and stopband start
	// soxrFp: Passband end (relative to input Nyquist = 1.0)
	// soxrFs: Stopband start (at input Nyquist = output Nyquist after decimation)
	soxrFp := qualityToPassbandEnd(quality)
	soxrFs := 1.0

	// Normalize by Fn (decimation factor)
	FpNorm := soxrFp / float64(factor)
	FsNorm := soxrFs / float64(factor)

	// Transition bandwidth
	trBW := transitionBandwidthHalf * (FsNorm - FpNorm)

	// Cutoff frequency (in our normalization where 0.5 = Nyquist)
	// soxr: Fc = Fs_norm - tr_bw
	Fc := FsNorm - trBW

	// Convert to our filter design convention (0.5 = Nyquist)
	cutoff := Fc * nyquistFraction // Scale to 0-0.5 range

	attenuation := qualityToAttenuation(quality)

	// Use auto filter design with transition bandwidth scaled to our convention
	transitionBW := trBW * nyquistFraction // Scale to 0-0.5 range
	coeffs, err := filter.DesignLowPassFilterAuto(cutoff, transitionBW, attenuation, 1.0)
	if err != nil {
		return nil, fmt.Errorf("failed to design decimation filter: %w", err)
	}

	// Convert coefficients to target precision F and reverse for convolution
	numTaps := len(coeffs)
	reversedCoeffs := make([]F, numTaps)
	for i, c := range coeffs {
		reversedCoeffs[numTaps-1-i] = F(c)
	}

	return &DFTDecimationStage[F]{
		factor:     factor,
		coeffs:     reversedCoeffs,
		numTaps:    numTaps,
		history:    make([]F, 0, numTaps*historyBufferMultiplier),
		decimPhase: 0,
		ops:        ops,
	}, nil
}

// Process filters and decimates the input samples.
//
// The algorithm:
// 1. Append input to history buffer
// 2. For each position where we have enough samples:
//   - Compute filtered output using FIR convolution
//   - If this position aligns with decimation phase, output the sample
//
// 3. Advance decimation phase
// processZeroCopy is the allocation-free internal path. The returned slice
// aliases s.outputBuf and is only valid until the next call.
func (s *DFTDecimationStage[F]) processZeroCopy(input []F) ([]F, error) { //nolint:unparam // error kept for symmetry with Process
	if s.factor == 1 {
		// Pass-through
		return input, nil
	}

	if len(input) == 0 {
		return []F{}, nil
	}

	// Append input to history (stable growth avoids steady-state realloc on input jitter)
	s.history = appendStable(s.history, input)

	numAvailable := len(s.history)
	if numAvailable < s.numTaps {
		return []F{}, nil
	}

	// Number of input positions we can compute filtered output for
	numFilterable := numAvailable - s.numTaps + 1

	// Calculate how many decimated outputs we'll produce
	// Starting from decimPhase, we output every 'factor'-th filtered sample
	numOutput := 0
	for i := s.decimPhase; i < numFilterable; i += s.factor {
		numOutput++
	}

	if numOutput == 0 {
		return []F{}, nil
	}

	// Allocate output buffer (stable growth avoids steady-state realloc on jitter)
	s.outputBuf = growStableLen(s.outputBuf, numOutput)

	// Filter and decimate
	outIdx := 0
	for pos := s.decimPhase; pos < numFilterable && outIdx < numOutput; pos += s.factor {
		// Compute FIR filter output at this position
		// Coefficients are reversed, so we can use direct dot product
		var sum F
		histSlice := s.history[pos : pos+s.numTaps]
		// Use SIMD dot product (always available)
		sum = s.ops.DotProductUnsafe(histSlice, s.coeffs)
		s.outputBuf[outIdx] = sum
		outIdx++
	}

	// Update decimation phase for next call
	// The phase represents the offset into the decimation cycle. After consuming
	// numFilterable samples, the buffer shifts and the new phase becomes
	// (oldPhase - consumed) mod factor. We use (x%f+f)%f to handle Go's
	// negative modulo behavior (Go returns negative for negative dividend).
	s.decimPhase = ((s.decimPhase-numFilterable)%s.factor + s.factor) % s.factor

	consumed := numFilterable
	// Shift history - keep only what we need for next call
	if consumed > 0 {
		remaining := len(s.history) - consumed
		if remaining > 0 {
			copy(s.history[:remaining], s.history[consumed:])
		}
		s.history = s.history[:remaining]
	}

	return s.outputBuf[:outIdx], nil
}

// Process resamples input through the DFT decimation stage. The returned
// slice is owned by the caller and remains valid across subsequent calls.
func (s *DFTDecimationStage[F]) Process(input []F) ([]F, error) {
	output, err := s.processZeroCopy(input)
	if err != nil || len(output) == 0 {
		return output, err
	}
	if s.factor == 1 {
		return output, nil
	}
	// IMPORTANT: Return a COPY of the output, not a slice of the internal buffer.
	// Returning s.outputBuf directly would cause buffer corruption on the next
	// Process() call, as the caller's slice would share the same backing array.
	// This was the cause of TestResampler_BufferIntegrity failures for 96→48.
	result := make([]F, len(output))
	copy(result, output)
	return result, nil
}

// Flush returns any remaining buffered samples.
func (s *DFTDecimationStage[F]) Flush() ([]F, error) {
	if s.factor == 1 || len(s.history) == 0 {
		return []F{}, nil
	}

	// Pad with zeros to flush pipeline
	zeros := make([]F, s.numTaps)
	return s.Process(zeros)
}

// Reset clears internal state.
func (s *DFTDecimationStage[F]) Reset() {
	s.history = s.history[:0]
	s.decimPhase = 0
}
