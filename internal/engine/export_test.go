package engine

import "math"

// Export internal functions for testing.
// This file uses the _test.go suffix so it's only included in test builds.

// PolyphaseFilterParams holds computed filter design parameters for testing
type PolyphaseFilterParams struct {
	NumPhases      int
	TapsPerPhase   int
	TotalTaps      int
	Ratio          float64
	TotalIORatio   float64
	Fp1            float64 // Initial passband edge
	Fs1            float64 // Initial stopband edge
	Fp             float64 // Final passband edge
	Fs             float64 // Final stopband edge
	TransitionBW   float64 // Transition bandwidth (Fs - Fp)
	Cutoff         float64 // Final cutoff frequency
	Attenuation    float64 // Target attenuation in dB
	RequiredTaps   float64 // Estimated taps needed for target attenuation
	TapsAdequate   bool    // Whether actual taps >= required taps
}

// ComputePolyphaseFilterParams computes and returns the filter design parameters
// without actually creating the filter. This allows testing the parameter computation.
// NOTE: This function mirrors the logic in designPolyphaseFilter() for testing purposes.
func ComputePolyphaseFilterParams(numPhases int, ratio, totalIORatio float64, quality Quality) PolyphaseFilterParams {
	attenuation := qualityToAttenuation(quality)

	// Passband edge calculation (matching designPolyphaseFilter)
	var Fp1 float64
	if ratio > 1.0 {
		Fp1 = totalIORatio * passbandRolloffScale
	} else {
		Fp1 = nyquistFraction * ratio * passbandRolloffScale
	}
	Fs1 := nyquistFraction

	var Fp, Fs float64
	if ratio > 1.0 {
		// Upsampling formula
		Fs = imageRejectionFactor - (Fp1 + (Fs1-Fp1)*0.7)
		invFResp := lsxInvFResp(-0.01, attenuation)
		Fp = Fs - (Fs-Fp1)/(1.0-invFResp)
	} else {
		// Downsampling formula (matching polyphase.go)
		Fp = Fp1
		transitionScale := 0.15 * ratio
		Fs = Fp1 + transitionScale
		if Fs > Fs1 {
			Fs = Fs1
		}
	}

	// Calculate transition bandwidth (matching designPolyphaseFilter)
	phases := float64(numPhases)
	trBw := 0.5 * (Fs - Fp)
	trBw /= phases

	// Minimum transition bandwidth
	const minTrBw = 0.02
	trBwLimit := 0.5 * Fs / phases
	if trBw > trBwLimit {
		trBw = trBwLimit
	}
	if trBw < minTrBw {
		trBw = minTrBw
	}

	// Dynamic tap calculation (matching designPolyphaseFilter)
	const minTapsPerPhase = 8
	const maxTapsPerPhase = 64
	const maxTotalTaps = 8000
	const minAttenForRatio = 80.0

	effectiveAttenuation := attenuation
	idealTotalTaps := int(math.Ceil(attenuation/trBw + 1))

	if idealTotalTaps > maxTotalTaps {
		effectiveAttenuation = float64(maxTotalTaps-1) * trBw
		if effectiveAttenuation < minAttenForRatio {
			effectiveAttenuation = minAttenForRatio
		}
	}

	totalTaps := int(math.Ceil(effectiveAttenuation/trBw + 1))
	tapsPerPhase := (totalTaps + numPhases - 1) / numPhases

	if tapsPerPhase < minTapsPerPhase {
		tapsPerPhase = minTapsPerPhase
	} else if tapsPerPhase > maxTapsPerPhase {
		tapsPerPhase = maxTapsPerPhase
	}

	totalTaps = numPhases*tapsPerPhase - 1

	// Cutoff calculation
	var cutoff float64
	if ratio > 1.0 {
		soxrFc := (Fp + Fs) / (soxrFcDenominator * phases)
		cutoff = soxrFc / soxrToOurNormScale
	} else {
		cutoff = Fp + trBw*0.5
	}

	if cutoff <= 0 {
		cutoff = 0.001
	}
	if cutoff >= nyquistFraction {
		cutoff = 0.499
	}

	transitionBW := Fs - Fp

	// Required taps calculation using simple Kaiser formula
	var requiredTaps float64
	if transitionBW > 0 {
		requiredTaps = attenuation/trBw + 1
	}

	return PolyphaseFilterParams{
		NumPhases:     numPhases,
		TapsPerPhase:  tapsPerPhase,
		TotalTaps:     totalTaps,
		Ratio:         ratio,
		TotalIORatio:  totalIORatio,
		Fp1:           Fp1,
		Fs1:           Fs1,
		Fp:            Fp,
		Fs:            Fs,
		TransitionBW:  transitionBW,
		Cutoff:        cutoff,
		Attenuation:   attenuation,
		RequiredTaps:  requiredTaps,
		TapsAdequate:  float64(totalTaps) >= requiredTaps,
	}
}

// ExportedDesignPolyphaseFilter wraps designPolyphaseFilter for testing
func ExportedDesignPolyphaseFilter(numPhases int, ratio, totalIORatio float64, quality Quality) (*PolyphaseFilterResult, error) {
	result, err := designPolyphaseFilter(numPhases, ratio, totalIORatio, quality)
	if err != nil {
		return nil, err
	}
	return &PolyphaseFilterResult{
		Coeffs:       result.coeffs,
		NumPhases:    result.numPhases,
		TapsPerPhase: result.tapsPerPhase,
	}, nil
}

// PolyphaseFilterResult is an exported version of polyphaseFilter for testing
type PolyphaseFilterResult struct {
	Coeffs       []float64
	NumPhases    int
	TapsPerPhase int
}

// ExportedFindRationalApprox wraps findRationalApprox for testing
func ExportedFindRationalApprox(ratio float64) (numPhases, step int) {
	return findRationalApprox(ratio)
}

// ExportedLsxInvFResp wraps lsxInvFResp for testing
func ExportedLsxInvFResp(drop, attenuation float64) float64 {
	return lsxInvFResp(drop, attenuation)
}

// ExportedIsIntegerRatio wraps isIntegerRatio for testing
func ExportedIsIntegerRatio(ratio float64) bool {
	return isIntegerRatio(ratio)
}

// ExportedQualityToAttenuation wraps qualityToAttenuation for testing
func ExportedQualityToAttenuation(q Quality) float64 {
	return qualityToAttenuation(q)
}

// GetPolyphaseStageInternals returns internal state for testing
func (s *PolyphaseStage[F]) GetPolyphaseStageInternals() (numPhases, tapsPerPhase, step int, polyCoeffs [][]F) {
	return s.numPhases, s.tapsPerPhase, s.step, s.polyCoeffs
}

// GetDFTStageInternals returns internal state for testing
func (s *DFTStage[F]) GetDFTStageInternals() (factor, tapsPerPhase int, polyCoeffs [][]F, isHalfBand bool) {
	return s.factor, s.tapsPerPhase, s.polyCoeffs, s.isHalfBand
}

// GetResamplerInternals returns internal stages for testing
func (r *Resampler[F]) GetResamplerInternals() (preStage *DFTStage[F], polyphaseStage *PolyphaseStage[F]) {
	return r.preStage, r.polyphaseStage
}
