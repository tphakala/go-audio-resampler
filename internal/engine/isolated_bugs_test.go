package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// BUG 1: designPolyphaseFilter produces inadequate taps for heavy downsampling
// =============================================================================

// TestBug_DesignPolyphaseFilter_InsufficientTaps tests that the filter design
// produces enough taps for the transition bandwidth and target attenuation.
//
// ROOT CAUSE: For 96k→48k, the polyphase ratio is 0.25 (after 2x DFT pre-stage).
// This creates a transition band of only ~0.00125, but with fixed 20 taps/phase
// and 80 phases = 1599 total taps, the filter cannot achieve 180dB attenuation.
func TestBug_DesignPolyphaseFilter_InsufficientTaps(t *testing.T) {
	testCases := []struct {
		ratio        float64
		totalIORatio float64
		numPhases    int
		desc         string
	}{
		// 96k→48k: After 2x DFT pre-stage, polyphase does 0.25x
		{0.25, 2.0, 80, "96k_to_48k_polyphase_0.25x"},
		// 192k→48k: After 2x DFT pre-stage, polyphase does 0.125x
		{0.125, 4.0, 80, "192k_to_48k_polyphase_0.125x"},
		// Milder downsampling
		{0.5, 1.5, 80, "96k_to_64k_polyphase_0.5x"},
		// 48k→44.1k: polyphase does 0.459x
		{0.459375, 0.91875, 147, "48k_to_44.1k_polyphase"},
		// 44.1k→48k: polyphase does 1.088x (upsampling)
		{1.088435374, 0.459375, 160, "44.1k_to_48k_polyphase"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			params := ComputePolyphaseFilterParams(tc.numPhases, tc.ratio, tc.totalIORatio, QualityHigh)

			t.Logf("=== Filter Design Parameters ===")
			t.Logf("Ratio: %.6f, TotalIORatio: %.6f", tc.ratio, tc.totalIORatio)
			t.Logf("NumPhases: %d, TapsPerPhase: %d, TotalTaps: %d",
				params.NumPhases, params.TapsPerPhase, params.TotalTaps)
			t.Logf("Fp1: %.6f, Fs1: %.6f", params.Fp1, params.Fs1)
			t.Logf("Fp: %.6f, Fs: %.6f", params.Fp, params.Fs)
			t.Logf("TransitionBW: %.6f", params.TransitionBW)
			t.Logf("Cutoff: %.6f", params.Cutoff)
			t.Logf("Target Attenuation: %.1f dB", params.Attenuation)
			t.Logf("Required Taps: %.0f, Actual Taps: %d", params.RequiredTaps, params.TotalTaps)
			t.Logf("Taps Adequate: %v", params.TapsAdequate)

			// BUG: Check if transition band is too narrow
			if params.TransitionBW < 0.01 && params.Ratio < 1.0 {
				t.Logf("BUG INDICATOR: Transition band %.6f is extremely narrow for downsampling",
					params.TransitionBW)
			}

			// BUG: Check if taps are insufficient
			if !params.TapsAdequate {
				tapDeficit := params.RequiredTaps - float64(params.TotalTaps)
				t.Errorf("BUG FOUND: Filter has %d taps but needs %.0f (deficit: %.0f) "+
					"for %.1f dB attenuation with TBW=%.6f",
					params.TotalTaps, params.RequiredTaps, tapDeficit,
					params.Attenuation, params.TransitionBW)
			}
		})
	}
}

// TestBug_DesignPolyphaseFilter_DownsamplingFormula tests the specific
// formula used for downsampling filter design.
func TestBug_DesignPolyphaseFilter_DownsamplingFormula(t *testing.T) {
	// For downsampling (ratio < 1.0), the code uses:
	//   Fp = Fp1 = nyquistFraction * ratio * passbandRolloffScale
	//   Fs = Fs1 * ratio = nyquistFraction * ratio
	//
	// This creates: TransitionBW = Fs - Fp = 0.5*ratio - 0.5*ratio*0.99 = 0.5*ratio*0.01
	// For ratio=0.25: TBW = 0.5 * 0.25 * 0.01 = 0.00125 (TOO NARROW!)

	testCases := []struct {
		ratio             float64
		expectedTBWApprox float64
		desc              string
	}{
		{0.25, 0.00125, "4x_downsample"},
		{0.5, 0.0025, "2x_downsample"},
		{0.333333, 0.00167, "3x_downsample"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			// Manually compute the transition bandwidth using the buggy formula
			Fp1 := nyquistFraction * tc.ratio * passbandRolloffScale
			Fs := nyquistFraction * tc.ratio
			actualTBW := Fs - Fp1

			t.Logf("Ratio: %.6f", tc.ratio)
			t.Logf("Fp1 = 0.5 * %.6f * 0.99 = %.6f", tc.ratio, Fp1)
			t.Logf("Fs = 0.5 * %.6f = %.6f", tc.ratio, Fs)
			t.Logf("TransitionBW = Fs - Fp1 = %.6f", actualTBW)

			assert.InDelta(t, tc.expectedTBWApprox, actualTBW, 0.0001,
				"Transition bandwidth calculation mismatch")

			// The problem: TBW is proportional to ratio, so aggressive downsampling
			// creates impossibly narrow transition bands
			if actualTBW < 0.01 {
				t.Logf("BUG: TransitionBW %.6f is too narrow for practical filter design", actualTBW)

				// Calculate what attenuation is achievable with fixed 1599 taps
				const fixedTaps = 1599
				achievableAtten := 7.95 + 2.285*2.0*math.Pi*actualTBW*float64(fixedTaps)
				t.Logf("With %d taps, achievable attenuation ≈ %.1f dB (target: 180 dB)",
					fixedTaps, achievableAtten)

				if achievableAtten < 180.0 {
					t.Errorf("BUG CONFIRMED: Can only achieve %.1f dB with current design (need 180 dB)",
						achievableAtten)
				}
			}
		})
	}
}

// =============================================================================
// BUG 2: Passband edge calculation for downsampling
// =============================================================================

// TestBug_PassbandEdgeCalculation tests the passband edge (Fp) calculation
// for downsampling scenarios.
func TestBug_PassbandEdgeCalculation(t *testing.T) {
	// The current code for downsampling:
	//   Fp1 = nyquistFraction * ratio * passbandRolloffScale
	//
	// For 96k→48k with polyphaseRatio=0.25:
	//   Fp1 = 0.5 * 0.25 * 0.99 = 0.12375
	//
	// This represents 12.375% of the polyphase input sample rate (192kHz)
	// = 23.76 kHz, which is correct for output Nyquist of 24kHz.
	//
	// The issue is not Fp, but Fs being too close to Fp.

	testCases := []struct {
		polyphaseRatio float64
		totalIORatio   float64
		expectedFp     float64
		expectedFs     float64
		desc           string
	}{
		// 96k→48k: polyphaseRatio = 48000/(96000*2) = 0.25
		{0.25, 2.0, 0.12375, 0.125, "96k_to_48k"},
		// 48k→44.1k: polyphaseRatio = 44100/(48000*2) = 0.459375
		{0.459375, 0.91875, 0.22740, 0.2297, "48k_to_44.1k"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			params := ComputePolyphaseFilterParams(80, tc.polyphaseRatio, tc.totalIORatio, QualityHigh)

			t.Logf("Polyphase ratio: %.6f", tc.polyphaseRatio)
			t.Logf("Computed Fp: %.6f (expected: %.6f)", params.Fp, tc.expectedFp)
			t.Logf("Computed Fs: %.6f (expected: %.6f)", params.Fs, tc.expectedFs)
			t.Logf("Transition BW: %.6f", params.TransitionBW)

			// Verify formula is applied correctly
			assert.InDelta(t, tc.expectedFp, params.Fp, 0.001, "Fp mismatch")
			assert.InDelta(t, tc.expectedFs, params.Fs, 0.001, "Fs mismatch")

			// Identify the bug: Fs - Fp is too small
			if params.TransitionBW < 0.01 {
				t.Errorf("BUG: Transition BW %.6f is too narrow (Fp=%.6f, Fs=%.6f)",
					params.TransitionBW, params.Fp, params.Fs)
			}
		})
	}
}

// =============================================================================
// BUG 3: Fixed taps-per-phase regardless of ratio
// =============================================================================

// TestBug_FixedTapsPerPhase tests that the fixed 20 taps per phase is inadequate
// for some conversion scenarios.
func TestBug_FixedTapsPerPhase(t *testing.T) {
	const fixedTapsPerPhase = 20

	testCases := []struct {
		polyphaseRatio float64
		totalIORatio   float64
		numPhases      int
		desc           string
	}{
		{0.25, 2.0, 80, "96k_to_48k_aggressive"},
		{0.125, 4.0, 80, "192k_to_48k_very_aggressive"},
		{1.088435374, 0.459375, 160, "44.1k_to_48k_mild"},
		{0.459375, 0.91875, 147, "48k_to_44.1k_mild"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			params := ComputePolyphaseFilterParams(tc.numPhases, tc.polyphaseRatio, tc.totalIORatio, QualityHigh)

			actualTaps := tc.numPhases * fixedTapsPerPhase
			requiredTapsPerPhase := params.RequiredTaps / float64(tc.numPhases)

			t.Logf("Ratio: %.6f, NumPhases: %d", tc.polyphaseRatio, tc.numPhases)
			t.Logf("Fixed taps/phase: %d, Required taps/phase: %.1f",
				fixedTapsPerPhase, requiredTapsPerPhase)
			t.Logf("Total taps: %d, Required: %.0f", actualTaps, params.RequiredTaps)
			t.Logf("Transition BW: %.6f", params.TransitionBW)

			if requiredTapsPerPhase > float64(fixedTapsPerPhase) {
				deficit := requiredTapsPerPhase - float64(fixedTapsPerPhase)
				t.Errorf("BUG: Need %.1f taps/phase but only have %d (deficit: %.1f per phase)",
					requiredTapsPerPhase, fixedTapsPerPhase, deficit)
			}
		})
	}
}

// =============================================================================
// BUG 4: Cutoff frequency too narrow for heavy downsampling
// =============================================================================

// TestBug_CutoffFrequencyComputation tests the cutoff frequency calculation
func TestBug_CutoffFrequencyComputation(t *testing.T) {
	// Cutoff formula: cutoff = (Fp + Fs) / (4 * phases)
	// For 96k→48k with ratio=0.25, phases=80:
	//   Fp = 0.12375, Fs = 0.125
	//   cutoff = (0.12375 + 0.125) / (4 * 80) = 0.24875 / 320 = 0.000777
	//
	// This cutoff is EXTREMELY narrow!

	testCases := []struct {
		polyphaseRatio float64
		totalIORatio   float64
		numPhases      int
		desc           string
	}{
		{0.25, 2.0, 80, "96k_to_48k"},
		{0.125, 4.0, 80, "192k_to_48k"},
		{1.088435374, 0.459375, 160, "44.1k_to_48k"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			params := ComputePolyphaseFilterParams(tc.numPhases, tc.polyphaseRatio, tc.totalIORatio, QualityHigh)

			t.Logf("Ratio: %.6f, Phases: %d", tc.polyphaseRatio, tc.numPhases)
			t.Logf("Fp: %.6f, Fs: %.6f", params.Fp, params.Fs)
			t.Logf("Cutoff: %.6f (in [0, 0.5] normalization)", params.Cutoff)

			// Cutoff of less than 0.01 indicates a very narrow passband
			if params.Cutoff < 0.01 && tc.polyphaseRatio < 1.0 {
				t.Logf("BUG INDICATOR: Cutoff %.6f is extremely narrow", params.Cutoff)

				// Calculate what percentage of Nyquist this represents
				pctNyquist := params.Cutoff / 0.5 * 100
				t.Logf("This cutoff passes only %.2f%% of Nyquist", pctNyquist)
			}
		})
	}
}

// =============================================================================
// BUG 5: Architecture issue - DFT upsample before heavy downsample
// =============================================================================

// TestBug_ResamplerArchitecture tests that the resampler architecture
// creates problematic polyphase ratios for certain conversions.
func TestBug_ResamplerArchitecture(t *testing.T) {
	testCases := []struct {
		inputRate              float64
		outputRate             float64
		expectedPolyphaseRatio float64
		desc                   string
	}{
		// 96k→48k: ratio=0.5, not integer, so uses DFT(2x) + polyphase(0.25x)
		{96000, 48000, 0.25, "96k_to_48k_creates_0.25x_polyphase"},
		// 192k→48k: ratio=0.25, uses DFT(2x) + polyphase(0.125x)
		{192000, 48000, 0.125, "192k_to_48k_creates_0.125x_polyphase"},
		// 48k→96k: ratio=2.0, integer, uses only DFT(2x)
		{48000, 96000, 0.0, "48k_to_96k_integer_no_polyphase"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ratio := tc.outputRate / tc.inputRate
			isInt := ExportedIsIntegerRatio(ratio)

			t.Logf("%.0f → %.0f, ratio=%.6f, isInteger=%v", tc.inputRate, tc.outputRate, ratio, isInt)

			if !isInt {
				// Calculate polyphase ratio
				intermediateRate := tc.inputRate * 2.0 // DFT 2x upsample
				polyphaseRatio := tc.outputRate / intermediateRate
				t.Logf("Intermediate rate: %.0f, Polyphase ratio: %.6f", intermediateRate, polyphaseRatio)

				assert.InDelta(t, tc.expectedPolyphaseRatio, polyphaseRatio, 0.001)

				// Check if this is a problematic ratio
				if polyphaseRatio < 0.3 {
					t.Logf("BUG: Polyphase ratio %.6f requires >3x downsampling in polyphase stage", polyphaseRatio)

					// Verify filter params are inadequate
					numPhases, _ := ExportedFindRationalApprox(polyphaseRatio)
					totalIORatio := tc.inputRate / tc.outputRate
					params := ComputePolyphaseFilterParams(numPhases, polyphaseRatio, totalIORatio, QualityHigh)

					if !params.TapsAdequate {
						t.Errorf("ARCHITECTURE BUG CONFIRMED: %.0f→%.0f creates polyphase ratio %.6f "+
							"which needs %.0f taps but only has %d",
							tc.inputRate, tc.outputRate, polyphaseRatio,
							params.RequiredTaps, params.TotalTaps)
					}
				}
			}
		})
	}
}

// =============================================================================
// BUG 6: lsxInvFResp edge cases
// =============================================================================

// TestBug_LsxInvFResp_EdgeCases tests lsxInvFResp for potential numerical issues
func TestBug_LsxInvFResp_EdgeCases(t *testing.T) {
	testCases := []struct {
		drop        float64
		attenuation float64
		desc        string
	}{
		{-0.01, 180.0, "normal_high_quality"},
		{-0.01, 300.0, "extreme_attenuation"},
		{-0.01, 1.0, "minimal_attenuation"},
		{0.0, 180.0, "zero_drop"},
		{-20.0, 180.0, "large_drop"},
		{-0.001, 180.0, "tiny_drop"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			result := ExportedLsxInvFResp(tc.drop, tc.attenuation)

			// Check for invalid results
			require.False(t, math.IsNaN(result), "Result is NaN for drop=%.4f, atten=%.1f", tc.drop, tc.attenuation)
			require.False(t, math.IsInf(result, 0), "Result is Inf for drop=%.4f, atten=%.1f", tc.drop, tc.attenuation)

			// Result should be in [0, 1]
			assert.GreaterOrEqual(t, result, 0.0, "Result should be >= 0")
			assert.LessOrEqual(t, result, 1.0, "Result should be <= 1")

			t.Logf("lsxInvFResp(%.4f, %.1f) = %.6f", tc.drop, tc.attenuation, result)
		})
	}
}

// =============================================================================
// Integration test: Verify actual filter frequency response
// =============================================================================

// TestBug_ActualFilterResponse tests the actual filter created by designPolyphaseFilter
func TestBug_ActualFilterResponse(t *testing.T) {
	testCases := []struct {
		ratio        float64
		totalIORatio float64
		numPhases    int
		desc         string
	}{
		{0.25, 2.0, 80, "96k_to_48k"},
		{1.088435374, 0.459375, 160, "44.1k_to_48k"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			filter, err := ExportedDesignPolyphaseFilter(tc.numPhases, tc.ratio, tc.totalIORatio, QualityHigh)
			require.NoError(t, err)

			t.Logf("Filter created: %d phases, %d taps/phase, %d total coefficients",
				filter.NumPhases, filter.TapsPerPhase, len(filter.Coeffs))

			// Verify coefficient count
			expectedCoeffs := filter.NumPhases * filter.TapsPerPhase
			assert.Len(t, filter.Coeffs, expectedCoeffs, "Coefficient count mismatch")

			// Check DC gain per phase
			for phase := 0; phase < min(5, filter.NumPhases); phase++ {
				phaseSum := 0.0
				for tap := 0; tap < filter.TapsPerPhase; tap++ {
					idx := tap*filter.NumPhases + phase
					if idx < len(filter.Coeffs) {
						phaseSum += filter.Coeffs[idx]
					}
				}
				t.Logf("Phase %d DC gain: %.6f", phase, phaseSum)

				// Each phase should have DC gain ≈ 1.0
				if math.Abs(phaseSum-1.0) > 0.1 {
					t.Logf("WARNING: Phase %d has unusual DC gain: %.6f", phase, phaseSum)
				}
			}

			// Calculate total DC gain
			totalSum := 0.0
			for _, c := range filter.Coeffs {
				totalSum += c
			}
			t.Logf("Total DC gain: %.6f (expected: %d)", totalSum, filter.NumPhases)
		})
	}
}
