package engine

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

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
	numPhases, _ := findRationalApprox(ratio)

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
	// This is CRITICAL: the integer step from findRationalApprox would lose
	// fractional precision and make sub-phase interpolation useless (frac would
	// always be 0), so it is intentionally discarded above.
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
// processZeroCopy is the allocation-free internal path. The returned slice
// aliases s.outputBuf and is only valid until the next call.
func (s *PolyphaseStage[F]) processZeroCopy(input []F) ([]F, error) { //nolint:unparam // error kept for symmetry with Process
	if len(input) == 0 {
		return []F{}, nil
	}

	s.samplesIn += int64(len(input))

	// Append input to history.
	// appendStable grows with headroom so steady-state streaming does not
	// reallocate when the run length jitters by a sample between calls.
	s.history = appendStable(s.history, input)

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

	// Reuse output buffer to reduce allocations. growStableLen adds headroom
	// when it must grow so that numOut jitter between calls (driven by the
	// fixed-point at/step accumulator) does not reallocate in the steady state.
	s.outputBuf = growStableLen(s.outputBuf, numOut)

	// Hoist invariants out of the loop for better optimization.
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

	// Establish once that all four coefficient banks are at least numPhases long.
	// They are allocated together with identical length (numPhases) in
	// NewPolyphaseStage, so this never fires; it is a BCE hint. Combined with the
	// in-loop phase guard (0 <= phase < numPhases), the compiler can chain
	// phase < numPhases <= len(bank) and drop the per-bank, per-sample bounds
	// checks on polyCoeffs[phase] .. polyCoeffsD[phase] without reslicing. If the
	// invariant were ever violated we produce no output rather than panic.
	if len(polyCoeffs) < numPhases || len(polyCoeffsB) < numPhases ||
		len(polyCoeffsC) < numPhases || len(polyCoeffsD) < numPhases {
		return []F{}, nil
	}

	// Write outputs through a local slice bounded to numOut (numOut == len(out)).
	// The loop produces exactly numOut samples so out[outIdx] is always in range.
	// The compiler still keeps a per-sample bounds check on the write because the
	// loop is driven by the fixed-point accumulator (at < limit), not by outIdx,
	// so it cannot relate outIdx to len(out). Leaving it; forcing it out would
	// need a contorted loop shape for no measurable gain.
	out := s.outputBuf[:numOut]

	// Precompute scale factor for converting fractional bits to [0, 1)
	fracScale := F(1.0 / float64(int64(1)<<phaseFracBits))

	// Main resampling loop with cubic coefficient interpolation.
	at := s.at
	outIdx := 0
	for at < limit {
		// Extract integer phase and fractional sub-phase from fixed-point accumulator
		// at = (input_sample * numPhases + integer_phase) << phaseFracBits + frac
		fullPhase := at >> phaseFracBits
		div := int(fullPhase / numPhases64)
		phase := int(fullPhase % numPhases64)
		frac := at & phaseFracMask
		x := F(frac) * fracScale

		// Boundary check
		if div+tapsPerPhase > histLen {
			break
		}

		// Establish phase is in [0, numPhases). Combined with the bank-length
		// guard above (len(bank) >= numPhases), this lets the compiler remove the
		// per-bank bounds checks on all four coefficient indexings below. phase is
		// fullPhase % numPhases so this never fires for valid input; it is purely a
		// BCE hint plus safety net.
		if phase < 0 || phase >= numPhases {
			break
		}

		coeffsA := polyCoeffs[phase]
		coeffsB := polyCoeffsB[phase]
		coeffsC := polyCoeffsC[phase]
		coeffsD := polyCoeffsD[phase]
		hist := history[div : div+tapsPerPhase]

		// Convolve with cubic coefficient interpolation using SIMD
		// Computes: sum = Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
		sum := s.ops.CubicInterpDot(hist, coeffsA, coeffsB, coeffsC, coeffsD, x)

		out[outIdx] = sum
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

	return output, nil
}

// Process resamples input through the polyphase stage. The returned
// slice is owned by the caller and remains valid across subsequent calls.
func (s *PolyphaseStage[F]) Process(input []F) ([]F, error) {
	output, err := s.processZeroCopy(input)
	if err != nil || len(output) == 0 {
		return output, err
	}
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
