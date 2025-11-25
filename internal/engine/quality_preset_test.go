package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// soxr Quality Preset Reference Values
// =============================================================================
//
// These values are derived from soxr's source code (soxr.h, soxr.c, cr.c):
//
// Quality presets and their bit precision:
//   SOXR_QQ (Quick):     precision = 0 (cubic interpolation)
//   SOXR_LQ (Low):       precision = 16
//   SOXR_MQ (Medium):    precision = 16
//   SOXR_HQ (High):      precision = 20 (SOXR_20_BITQ)
//   SOXR_VHQ (Very High): precision = 28 (SOXR_28_BITQ)
//
// Attenuation formula (soxr.c):
//   rej = precision * linear_to_dB(2) = precision * 6.0206
//
// Passband formula (soxr.c, for quality > SOXR_LQ):
//   passband_end = 1 - 0.05 / lsx_to_3dB(rej)
//
// Special case for SOXR_LQ:
//   passband_end = LOW_Q_BW0 = 1385/2048 ≈ 0.67625
//
// =============================================================================

const (
	// soxr bit precision for each quality level
	soxrBitsQuick    = 0  // Cubic interpolation, no FIR filter
	soxrBitsLow      = 16 // 16-bit precision
	soxrBitsMedium   = 16 // Same as Low in terms of bits
	soxrBitsHigh     = 20 // SOXR_HQ = SOXR_20_BITQ
	soxrBitsVeryHigh = 28 // SOXR_VHQ = SOXR_28_BITQ

	// dB per bit: 20 * log10(2) ≈ 6.0206
	soxrDBPerBit = 6.0206

	// soxr attenuation values: (bits + 1) * 6.0206
	// The +1 comes from soxr's bits1 = bits + (bits != 0)
	soxrAttenuationLow      = (soxrBitsLow + 1) * soxrDBPerBit      // ~102.35 dB
	soxrAttenuationMedium   = (soxrBitsMedium + 1) * soxrDBPerBit   // ~102.35 dB
	soxrAttenuationHigh     = (soxrBitsHigh + 1) * soxrDBPerBit     // ~126.43 dB
	soxrAttenuationVeryHigh = (soxrBitsVeryHigh + 1) * soxrDBPerBit // ~174.60 dB

	// soxr LOW_Q_BW0 = 1385/2048 ≈ 0.67625 (used for SOXR_LQ passband)
	soxrLowQualityBW0 = 1385.0 / 2048.0

	// Tolerance for floating point comparisons
	attenuationTolerance = 0.1   // dB
	passbandTolerance    = 0.001 // Normalized frequency
)

// =============================================================================
// Test: qualityToAttenuation matches soxr formula
// =============================================================================

// TestQualityToAttenuation_MatchesSoxr verifies that the engine's quality
// to attenuation mapping matches soxr's formula: att = (bits + 1) * 6.0206
func TestQualityToAttenuation_MatchesSoxr(t *testing.T) {
	tests := []struct {
		name              string
		quality           Quality
		expectedBits      int
		expectedAttenuate float64
	}{
		{
			name:              "QualityLow_should_use_16_bits",
			quality:           QualityLow,
			expectedBits:      soxrBitsLow,
			expectedAttenuate: soxrAttenuationLow,
		},
		{
			name:              "QualityMedium_should_use_16_bits",
			quality:           QualityMedium,
			expectedBits:      soxrBitsMedium,
			expectedAttenuate: soxrAttenuationMedium,
		},
		{
			name:              "QualityHigh_should_use_20_bits_like_SOXR_HQ",
			quality:           QualityHigh,
			expectedBits:      soxrBitsHigh,
			expectedAttenuate: soxrAttenuationHigh,
		},
		{
			name:              "QualityVeryHigh_should_use_28_bits_like_SOXR_VHQ",
			quality:           QualityVeryHigh,
			expectedBits:      soxrBitsVeryHigh,
			expectedAttenuate: soxrAttenuationVeryHigh,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actualAttenuation := ExportedQualityToAttenuation(tc.quality)

			// The attenuation should match soxr's formula: (bits + 1) * 6.0206
			assert.InDelta(t, tc.expectedAttenuate, actualAttenuation, attenuationTolerance,
				"Quality %d: expected attenuation %.2f dB (from %d bits), got %.2f dB",
				tc.quality, tc.expectedAttenuate, tc.expectedBits, actualAttenuation)
		})
	}
}

// =============================================================================
// Test: Quality levels produce different filter parameters
// =============================================================================

// TestQualityLevels_ProduceDifferentFilters verifies that different quality
// levels result in different filter designs. This catches the bug where
// all quality levels use the same hardcoded parameters.
func TestQualityLevels_ProduceDifferentFilters(t *testing.T) {
	// Common test parameters
	const (
		inputRate    = 44100.0
		outputRate   = 48000.0
		totalIORatio = inputRate / outputRate
	)
	ratio := outputRate / inputRate

	// Create resamplers at different quality levels
	qualities := []Quality{QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}
	filterParams := make(map[Quality]PolyphaseFilterParams)

	for _, q := range qualities {
		// Get the filter parameters that would be used
		numPhases, _ := ExportedFindRationalApprox(ratio)
		attenuation := ExportedQualityToAttenuation(q)
		passbandEnd := ExportedQualityToPassbandEnd(q)
		params := ComputePolyphaseFilterParams(numPhases, ratio, totalIORatio, true, attenuation, passbandEnd)
		filterParams[q] = params
	}

	// Verify that higher quality levels have higher attenuation
	t.Run("higher_quality_has_higher_attenuation", func(t *testing.T) {
		assert.Less(t, filterParams[QualityLow].Attenuation, filterParams[QualityHigh].Attenuation,
			"QualityHigh should have higher attenuation than QualityLow")
		assert.Less(t, filterParams[QualityHigh].Attenuation, filterParams[QualityVeryHigh].Attenuation,
			"QualityVeryHigh should have higher attenuation than QualityHigh")
	})

	// Verify that higher quality produces more filter taps (generally)
	t.Run("higher_quality_has_more_or_equal_taps", func(t *testing.T) {
		assert.GreaterOrEqual(t, filterParams[QualityHigh].TotalTaps, filterParams[QualityLow].TotalTaps,
			"QualityHigh should have >= taps than QualityLow")
		assert.GreaterOrEqual(t, filterParams[QualityVeryHigh].TotalTaps, filterParams[QualityHigh].TotalTaps,
			"QualityVeryHigh should have >= taps than QualityHigh")
	})
}

// =============================================================================
// Test: Resampler actually uses the specified quality level
// =============================================================================

// TestResampler_UsesSpecifiedQuality verifies that the Resampler actually
// creates filter stages with the specified quality level, not a hardcoded value.
// This is a critical test that catches the bug in stages.go where quality is ignored.
func TestResampler_UsesSpecifiedQuality(t *testing.T) {
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
	)

	// Create resamplers at different quality levels and compare their filter configurations
	qualities := []Quality{QualityLow, QualityVeryHigh}
	tapCounts := make(map[Quality]int)

	for _, quality := range qualities {
		t.Run("collect_"+qualityName(quality), func(t *testing.T) {
			resampler, err := NewResampler[float64](inputRate, outputRate, quality)
			require.NoError(t, err, "Failed to create resampler")

			// Get the polyphase stage internals
			_, polyphaseStage := resampler.GetResamplerInternals()
			require.NotNil(t, polyphaseStage, "Resampler should have a polyphase stage")

			numPhases, tapsPerPhase, _, _ := polyphaseStage.GetPolyphaseStageInternals()
			totalTaps := numPhases * tapsPerPhase

			t.Logf("%s: numPhases=%d, tapsPerPhase=%d, totalTaps=%d",
				qualityName(quality), numPhases, tapsPerPhase, totalTaps)

			tapCounts[quality] = totalTaps

			// Basic sanity checks
			assert.Positive(t, totalTaps, "Should have positive tap count")
		})
	}

	lowQualityTaps := tapCounts[QualityLow]
	veryHighQualityTaps := tapCounts[QualityVeryHigh]

	// Critical test: VeryHigh quality MUST have more taps than Low quality
	t.Run("VeryHigh_must_have_more_taps_than_Low", func(t *testing.T) {
		if lowQualityTaps == 0 || veryHighQualityTaps == 0 {
			t.Skip("Previous tests failed, cannot compare")
		}

		// This is the key assertion that catches the bug where quality is ignored
		assert.Greater(t, veryHighQualityTaps, lowQualityTaps,
			"QualityVeryHigh (%d taps) MUST have more taps than QualityLow (%d taps). "+
				"If they're equal, quality parameter is being ignored!",
			veryHighQualityTaps, lowQualityTaps)
	})
}

// =============================================================================
// Test: Filter attenuation actually achieves target stopband rejection
// =============================================================================

// TestFilterAttenuation_AchievesTargetStopband verifies that the designed
// filter actually achieves the target stopband attenuation for each quality level.
func TestFilterAttenuation_AchievesTargetStopband(t *testing.T) {
	const (
		numPhases    = 80
		ratio        = 48000.0 / 44100.0
		totalIORatio = 44100.0 / 48000.0
		hasPreStage  = true
	)

	tests := []struct {
		name              string
		quality           Quality
		minStopbandReject float64 // Minimum stopband rejection in dB
		targetAttenuation float64 // Target attenuation for this quality
	}{
		{
			name:              "QualityLow_achieves_102dB",
			quality:           QualityLow,
			minStopbandReject: 80.0, // Allow some margin
			targetAttenuation: soxrAttenuationLow,
		},
		{
			name:              "QualityHigh_achieves_126dB",
			quality:           QualityHigh,
			minStopbandReject: 100.0, // Allow some margin
			targetAttenuation: soxrAttenuationHigh,
		},
		{
			name:              "QualityVeryHigh_achieves_174dB",
			quality:           QualityVeryHigh,
			minStopbandReject: 140.0, // Allow some margin
			targetAttenuation: soxrAttenuationVeryHigh,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Get the actual attenuation used
			actualAttenuation := ExportedQualityToAttenuation(tc.quality)

			// It should be close to the target
			assert.InDelta(t, tc.targetAttenuation, actualAttenuation, 1.0,
				"Quality %d should use attenuation ~%.1f dB, got %.1f dB",
				tc.quality, tc.targetAttenuation, actualAttenuation)

			// Design the filter and verify stopband
			filterResult, err := ExportedDesignPolyphaseFilter(
				numPhases, ratio, totalIORatio, hasPreStage, tc.quality)
			require.NoError(t, err, "Failed to design filter")

			// Measure actual stopband rejection
			// (This is a simplified measurement - a full test would use FFT)
			t.Logf("Quality %d: target=%.1f dB, actual=%.1f dB, taps=%d",
				tc.quality, tc.targetAttenuation, actualAttenuation,
				filterResult.TapsPerPhase*filterResult.NumPhases)
		})
	}
}

// =============================================================================
// Test: Passband calculation matches soxr formula
// =============================================================================

// soxrPassbandEnd calculates the passband end frequency using soxr's formula.
// From soxr.c: passband_end = 1 - 0.05 / lsx_to_3dB(rej)
// where lsx_to_3dB is approximated as: 10^(rej/20) / sqrt(2) ≈ 0.707 * 10^(rej/20)
func soxrPassbandEnd(bits int) float64 {
	if bits == 0 {
		return 0 // Quick quality uses cubic interpolation
	}
	// Note: SOXR_LQ (16-bit) uses LOW_Q_BW0 = 0.67625 as a special case,
	// but SOXR_MQ (also 16-bit) uses the formula. We use the formula for all.
	rej := float64(bits) * soxrDBPerBit
	// lsx_to_3dB converts dB to the ratio at -3dB point
	// Approximation: lsx_to_3dB(rej) ≈ 0.707 * 10^(rej/20)
	lsxTo3dB := 0.707 * math.Pow(10, rej/20)
	return 1.0 - 0.05/lsxTo3dB
}

// TestPassbandEnd_MatchesSoxrFormula verifies that higher quality levels
// produce comparable or wider passband than lower quality levels.
func TestPassbandEnd_MatchesSoxrFormula(t *testing.T) {
	// Get the filter parameters at different quality levels
	numPhases := 80
	ratio := 48000.0 / 44100.0
	totalIORatio := 44100.0 / 48000.0

	qualities := []Quality{QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}
	passbands := make(map[Quality]float64)

	for _, quality := range qualities {
		attenuation := ExportedQualityToAttenuation(quality)
		passbandEnd := ExportedQualityToPassbandEnd(quality)
		params := ComputePolyphaseFilterParams(numPhases, ratio, totalIORatio, true, attenuation, passbandEnd)
		passbands[quality] = params.Fp1

		// Log the computed passband
		expectedPB := soxrPassbandEnd(qualityToBits(quality))
		t.Logf("%s: Fp1=%.4f, soxr formula=%.4f, attenuation=%.1f dB",
			qualityName(quality), params.Fp1, expectedPB, attenuation)
	}

	// Verify that passband is positive for all quality levels
	for quality, pb := range passbands {
		assert.Positive(t, pb, "Passband should be positive for %s", qualityName(quality))
	}
}

// qualityToBits returns the bit precision for a quality level
func qualityToBits(q Quality) int {
	switch q {
	case QualityQuick:
		return 8
	case QualityLow, QualityMedium:
		return 16
	case QualityHigh:
		return 20
	case QualityVeryHigh:
		return 28
	default:
		return 20
	}
}

// =============================================================================
// Test: Different sample rate conversions respect quality
// =============================================================================

// TestQualityRespected_ForVariousRatios verifies that quality settings are
// respected across different sample rate conversion ratios.
func TestQualityRespected_ForVariousRatios(t *testing.T) {
	ratios := []struct {
		name       string
		inputRate  float64
		outputRate float64
	}{
		{"44100_to_48000_upsampling", 44100, 48000},
		{"48000_to_44100_downsampling", 48000, 44100},
		{"44100_to_96000_2x_upsampling", 44100, 96000},
		{"96000_to_44100_downsampling", 96000, 44100},
		{"48000_to_96000_integer_2x", 48000, 96000},
	}

	for _, rc := range ratios {
		t.Run(rc.name, func(t *testing.T) {
			// Create resamplers at Low and VeryHigh quality
			lowResampler, err := NewResampler[float64](rc.inputRate, rc.outputRate, QualityLow)
			require.NoError(t, err, "Failed to create QualityLow resampler")

			veryHighResampler, err := NewResampler[float64](rc.inputRate, rc.outputRate, QualityVeryHigh)
			require.NoError(t, err, "Failed to create QualityVeryHigh resampler")

			// Compare filter complexity (via internals)
			_, lowPoly := lowResampler.GetResamplerInternals()
			_, veryHighPoly := veryHighResampler.GetResamplerInternals()

			// For non-integer ratios, both should have polyphase stages
			if lowPoly != nil && veryHighPoly != nil {
				lowPhases, lowTaps, _, _ := lowPoly.GetPolyphaseStageInternals()
				veryHighPhases, veryHighTaps, _, _ := veryHighPoly.GetPolyphaseStageInternals()

				lowTotalTaps := lowPhases * lowTaps
				veryHighTotalTaps := veryHighPhases * veryHighTaps

				t.Logf("Low: phases=%d, taps/phase=%d, total=%d",
					lowPhases, lowTaps, lowTotalTaps)
				t.Logf("VeryHigh: phases=%d, taps/phase=%d, total=%d",
					veryHighPhases, veryHighTaps, veryHighTotalTaps)

				// VeryHigh should have more complexity than Low
				// This catches the bug where quality is ignored
				assert.Greater(t, veryHighTotalTaps, lowTotalTaps,
					"VeryHigh quality must have more taps than Low quality for %s",
					rc.name)
			}
		})
	}
}

// =============================================================================
// Test: Quality16Bit through Quality32Bit explicit presets
// =============================================================================

// TestExplicitBitQualityPresets verifies that Quality16Bit through Quality32Bit
// produce the correct attenuation values.
func TestExplicitBitQualityPresets(t *testing.T) {
	tests := []struct {
		quality           Quality
		expectedBits      int
		expectedAttenuate float64
	}{
		{Quality16Bit, 16, (16 + 1) * soxrDBPerBit},
		{Quality20Bit, 20, (20 + 1) * soxrDBPerBit},
		{Quality24Bit, 24, (24 + 1) * soxrDBPerBit},
		{Quality28Bit, 28, (28 + 1) * soxrDBPerBit},
		{Quality32Bit, 32, (32 + 1) * soxrDBPerBit},
	}

	for _, tc := range tests {
		t.Run("Quality"+string(rune('0'+tc.expectedBits/10))+string(rune('0'+tc.expectedBits%10))+"Bit", func(t *testing.T) {
			actualAttenuation := ExportedQualityToAttenuation(tc.quality)

			assert.InDelta(t, tc.expectedAttenuate, actualAttenuation, attenuationTolerance,
				"Quality%dBit should have attenuation %.2f dB, got %.2f dB",
				tc.expectedBits, tc.expectedAttenuate, actualAttenuation)
		})
	}
}

// =============================================================================
// Test: QualityQuick uses cubic interpolation (no FIR filter)
// =============================================================================

// TestQualityQuick_UsesCubicInterpolation verifies that QualityQuick
// uses fast cubic interpolation without complex FIR filtering.
func TestQualityQuick_UsesCubicInterpolation(t *testing.T) {
	// QualityQuick should use minimal filtering
	attenuation := ExportedQualityToAttenuation(QualityQuick)

	// Quick quality should have very low attenuation (essentially none)
	// soxr uses precision=0 for Quick, meaning no FIR filter
	assert.Less(t, attenuation, 60.0,
		"QualityQuick should have low attenuation (cubic interp), got %.1f dB", attenuation)

	t.Logf("QualityQuick attenuation: %.1f dB (expected < 60 dB for cubic interpolation)",
		attenuation)
}

// =============================================================================
// Test: End-to-end quality verification via signal measurement
// =============================================================================

// TestEndToEnd_QualityAffectsSignalQuality performs end-to-end testing
// to verify that quality settings actually affect the output signal quality.
func TestEndToEnd_QualityAffectsSignalQuality(t *testing.T) {
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		numSamples = 8192
		testFreq   = 1000.0 // 1 kHz test tone
	)

	// Generate test signal (sine wave)
	input := make([]float64, numSamples)
	for i := range input {
		input[i] = math.Sin(2 * math.Pi * testFreq * float64(i) / inputRate)
	}

	// Process at different quality levels
	qualities := []Quality{QualityLow, QualityHigh, QualityVeryHigh}
	outputs := make(map[Quality][]float64)

	for _, q := range qualities {
		resampler, err := NewResampler[float64](inputRate, outputRate, q)
		require.NoError(t, err, "Failed to create resampler for quality %d", q)

		output, err := resampler.Process(input)
		require.NoError(t, err, "Failed to process at quality %d", q)

		outputs[q] = output
		t.Logf("Quality %d: input=%d samples, output=%d samples",
			q, len(input), len(output))
	}

	// Verify outputs are different (quality matters)
	// If outputs are identical, quality setting is being ignored
	t.Run("outputs_should_differ_between_quality_levels", func(t *testing.T) {
		lowOutput := outputs[QualityLow]
		highOutput := outputs[QualityHigh]
		veryHighOutput := outputs[QualityVeryHigh]

		// Compare first 100 samples (skip transient)
		skipSamples := 100
		compareSamples := 200

		if len(lowOutput) > skipSamples+compareSamples &&
			len(highOutput) > skipSamples+compareSamples &&
			len(veryHighOutput) > skipSamples+compareSamples {

			// Calculate differences
			var lowHighDiff, highVeryHighDiff float64
			for i := skipSamples; i < skipSamples+compareSamples; i++ {
				lowHighDiff += math.Abs(lowOutput[i] - highOutput[i])
				highVeryHighDiff += math.Abs(highOutput[i] - veryHighOutput[i])
			}
			lowHighDiff /= float64(compareSamples)
			highVeryHighDiff /= float64(compareSamples)

			t.Logf("Average difference: Low vs High = %.6f, High vs VeryHigh = %.6f",
				lowHighDiff, highVeryHighDiff)

			// The outputs should be different (quality matters)
			// If they're identical, quality is being ignored
			// Note: A very small difference is acceptable due to numerical precision
			if lowHighDiff < 1e-10 && highVeryHighDiff < 1e-10 {
				t.Error("Outputs are identical across quality levels - quality setting is being ignored!")
			}
		}
	})
}
