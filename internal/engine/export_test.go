package engine

// Export internal functions for testing.
// This file uses the _test.go suffix so it's only included in test builds.

// Note: PolyphaseFilterParams and ComputePolyphaseFilterParams are now
// exported directly from polyphase.go for better testability.

// ExportedDesignPolyphaseFilter wraps designPolyphaseFilter for testing
func ExportedDesignPolyphaseFilter(numPhases int, ratio, totalIORatio float64, hasPreStage bool, quality Quality) (*PolyphaseFilterResult, error) {
	result, err := designPolyphaseFilter(numPhases, ratio, totalIORatio, hasPreStage, quality)
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

// ExportedQualityToPassbandEnd wraps qualityToPassbandEnd for testing
func ExportedQualityToPassbandEnd(q Quality) float64 {
	return qualityToPassbandEnd(q)
}

// GetPolyphaseStageInternals returns internal state for testing
func (s *PolyphaseStage[F]) GetPolyphaseStageInternals() (numPhases, tapsPerPhase int, step int64, polyCoeffs [][]F) {
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
