package engine

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/filter"
	"github.com/tphakala/simd/f64"
)

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

	// Passband end frequency constants (Fp0) as fraction of Nyquist.
	// Values derived from soxr's quality-spec.h and lsx_to_3dB calculations.
	passbandLow      = 0.67625 // soxr's lq_bw0 = 1385/2048 (FP exact)
	passbandMedium   = 0.91    // soxr's MQ from lsx_to_3dB
	passbandHigh     = 0.912   // soxr's HQ passband
	passbandVeryHigh = 0.913   // soxr's VHQ passband

	// Filter cutoff frequency constants.
	nyquistFraction    = 0.5  // Half the sample rate (Nyquist)
	transitionBWFactor = 0.05 // Transition bandwidth relative to Nyquist

	// Polyphase filter design constants.
	transitionBandwidthHalf = 0.5   // Half factor for transition bandwidth calculation
	invFRespThreshold       = 0.999 // Guard against division by zero in rolloff compensation

	// Buffer sizing constants.
	historyBufferMultiplier = 2 // Extra capacity for history buffers

	// Steady-state buffer growth slack.
	//
	// When the warm streaming buffers (history, outputBuf) need to grow, the
	// required size jitters by a sample or two between calls because the
	// fixed-point at/step accumulator drifts how much input each call consumes
	// and how many output samples it produces. Allocating exactly the required
	// size leaves no headroom, so the very next call's one-sample jitter trips
	// another allocation. Growing with a fixed extra margin (and rounding the
	// new capacity up) lets the buffer settle at a stable maximum after warmup,
	// so steady-state ProcessInto reaches 0 allocs/op. The slack is a handful of
	// samples; it does not change any output value, only the buffer capacity.
	bufferGrowthSlack = 16

	// Cache optimization constants.
	// Process in chunks that fit in L2 cache (~256KB) for better cache efficiency.
	// Chunk size chosen so signal + kernel + output all fit in L2.
	l2CacheChunkSize = 4096 // Samples per chunk for L2 cache efficiency

	// Rational approximation constants.
	rationalApproxTolerance = 1e-10 // Tolerance for finding rational approximation

	// Half-band optimization constant.
	// Half-band filters are used for 2× upsampling where Phase 0 is a passthrough.
	halfBandFactor = 2

	// soxr-derived filter design constants.
	// These values are from soxr's filter design algorithm.
	soxrDFTStageFc       = 0.4778321 // soxr's Fc for DFT stage (1.0 = Nyquist)
	imageRejectionFactor = 2.0       // Factor for image rejection frequency calculation
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

// qualityToPassbandEnd returns the passband end frequency (Fp0) for quality level.
// This is the -3dB frequency as a fraction of Nyquist (0-1 where 1 = Nyquist).
// Values are derived from soxr's quality-spec.h and lsx_to_3dB calculations.
func qualityToPassbandEnd(q Quality) float64 {
	switch q {
	case QualityQuick, QualityLow:
		return passbandLow
	case QualityMedium:
		return passbandMedium
	case QualityHigh, Quality20Bit:
		return passbandHigh
	case QualityVeryHigh, Quality24Bit, Quality28Bit, Quality32Bit:
		return passbandVeryHigh
	case Quality16Bit:
		return passbandLow
	default:
		return passbandHigh
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
	passbandEnd := qualityToPassbandEnd(quality)

	// Use the new testable parameter computation function
	params := ComputePolyphaseFilterParams(numPhases, ratio, totalIORatio, hasPreStage, attenuation, passbandEnd)

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
//   - passbandEnd: quality-based passband end frequency (Fp0), e.g., 0.913 for VHQ
//
// Returns computed filter parameters for filter design.
func ComputePolyphaseFilterParams(numPhases int, ratio, totalIORatio float64, hasPreStage bool, attenuation, passbandEnd float64) PolyphaseFilterParams {
	params := PolyphaseFilterParams{
		NumPhases:    numPhases,
		Ratio:        ratio,
		TotalIORatio: totalIORatio,
		HasPreStage:  hasPreStage,
		Attenuation:  attenuation,
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

	// Compute initial passband and stopband edges (Fp1, Fs1)
	// These are scaled by the polyphase ratio, matching soxr's approach where
	// Fp1 = Fp0 / Fn1 after the pre-stage (Fn1 = 1/ratio for our case).
	//
	// For downsampling 48→44.1 with 2x pre-stage (96 kHz intermediate):
	//   ratio = 44.1/96 = 0.459375
	//   Fp1 = 0.913 * 0.459375 = 0.4197 (matches soxr trace)
	//   Fs1 = 1.0 * 0.459375 = 0.4594 (matches soxr trace)
	if params.IsUpsampling {
		// Upsampling: passband at original Nyquist scaled by io_ratio
		params.Fp1 = totalIORatio * passbandEnd
		params.Fs1 = totalIORatio * 1.0 // Fs0 = 1.0 in soxr
	} else {
		// Downsampling: scale by polyphase ratio (output/intermediate rate)
		// This matches soxr's Fp1/Fn1 where Fn1 is the cumulative factor
		params.Fp1 = passbandEnd * ratio
		params.Fs1 = ratio // Fs0 = 1.0, so Fs1 = 1.0 * ratio = ratio
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
	// For downsampling WITHOUT pre-stage (hasPreStage=false), soxr intentionally
	// uses the same formula as upsampling (Fn=1). This case occurs when there's
	// a 2x upsampling pre-stage (preM=0 in soxr terms), so the polyphase stage
	// sees Fn=1 normalization and uses the anti-imaging formula.
	if !params.IsUpsampling && hasPreStage {
		// Downsampling WITH pre-stage: Fn = 2 * mult, Fs = 3 + |Fs1 - 1|
		params.Fn = soxrDownsamplingFnFactor * params.Mult
		// Fs = 3 + |0.5 - 1| = 3.5
		params.FsRaw = soxrDownsamplingFsBase + math.Abs(params.Fs1-1.0)
		params.FpRaw = params.Fp1
	} else {
		// Upsampling OR Downsampling WITHOUT pre-stage:
		// Both cases use the anti-imaging formula with Fn=1.
		// For downsampling without pre-stage, soxr uses the same formula as upsampling
		// (this occurs when we have a 2x upsampling pre-stage, preM=0 in soxr terms).
		//
		// soxr code: else Fn = 1, Fs = 2 - (mode? Fp1 + (Fs1 - Fp1) * .7 : Fs1);
		params.Fn = 1.0
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
