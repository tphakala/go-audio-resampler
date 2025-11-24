package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Unit Tests for isIntegerRatio
// =============================================================================

func TestIsIntegerRatio(t *testing.T) {
	testCases := []struct {
		ratio    float64
		expected bool
		desc     string
	}{
		// Integer upsampling ratios - should be detected
		{1.0, true, "1:1_passthrough"},
		{2.0, true, "2x_upsample"},
		{3.0, true, "3x_upsample"},
		{4.0, true, "4x_upsample"},
		{2.0000000001, true, "2x_with_tiny_error"},

		// Non-integer ratios - should NOT be detected as integer
		{0.5, false, "0.5_downsample"},
		{0.333333, false, "0.33_downsample"},
		{1.5, false, "1.5_ratio"},
		{1.088435374, false, "44.1k_to_48k"},
		{0.91875, false, "48k_to_44.1k"},
		{2.1768707, false, "44.1k_to_96k"},

		// Edge cases
		{0.0, false, "zero"},
		{0.999999, false, "almost_1_far"},   // Far enough from 1 to not round
		{0.9999999999, true, "almost_1"},    // Rounds to 1 within tolerance
		{1.0000000001, true, "just_over_1"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			result := isIntegerRatio(tc.ratio)
			assert.Equal(t, tc.expected, result,
				"isIntegerRatio(%f) = %v, expected %v", tc.ratio, result, tc.expected)
		})
	}
}

// =============================================================================
// Unit Tests for findRationalApprox
// =============================================================================

func TestFindRationalApprox(t *testing.T) {
	testCases := []struct {
		ratio     float64
		minPhases int
		maxPhases int
		desc      string
	}{
		// Common sample rate ratios
		{1.088435374, 64, 256, "44.1k_to_48k"}, // 48000/44100 = 160/147
		{0.91875, 64, 256, "48k_to_44.1k"},     // 44100/48000 = 147/160
		{1.0, 64, 256, "1:1"},
		{0.5, 64, 256, "2x_downsample"},
		{0.25, 64, 256, "4x_downsample"},
		{2.0, 64, 256, "2x_upsample"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			numPhases, step := findRationalApprox(tc.ratio)

			// Validate outputs are positive
			assert.Positive(t, numPhases, "numPhases should be positive")
			assert.Positive(t, step, "step should be positive")

			// Validate phase count is within expected range
			assert.GreaterOrEqual(t, numPhases, tc.minPhases, "numPhases should be >= %d", tc.minPhases)
			assert.LessOrEqual(t, numPhases, tc.maxPhases, "numPhases should be <= %d", tc.maxPhases)

			// Validate the approximation is accurate
			// ratio ≈ L / step, so step / L ≈ 1 / ratio
			approxInvRatio := float64(step) / float64(numPhases)
			actualInvRatio := 1.0 / tc.ratio
			relError := math.Abs(approxInvRatio-actualInvRatio) / actualInvRatio
			assert.Less(t, relError, 0.01, "Relative error should be < 1%%")

			t.Logf("ratio=%.6f -> numPhases=%d, step=%d (approx inv ratio=%.6f, actual=%.6f, err=%.6f%%)",
				tc.ratio, numPhases, step, approxInvRatio, actualInvRatio, relError*100)
		})
	}
}

// =============================================================================
// Unit Tests for lsxInvFResp
// =============================================================================

func TestLsxInvFResp(t *testing.T) {
	testCases := []struct {
		drop        float64
		attenuation float64
		expectValid bool
		desc        string
	}{
		// Normal operating ranges
		{-0.01, 180.0, true, "high_quality_small_rolloff"},
		{-0.01, 140.0, true, "medium_quality_small_rolloff"},
		{-0.01, 100.0, true, "low_quality_small_rolloff"},
		{-0.1, 180.0, true, "high_quality_medium_rolloff"},
		{-1.0, 180.0, true, "high_quality_large_rolloff"},
		{-3.0, 180.0, true, "high_quality_very_large_rolloff"},
		{-6.0, 180.0, true, "half_power_point"},

		// Edge cases that previously caused NaN
		{-0.01, 1.0, true, "minimum_attenuation"},
		{-0.01, 300.0, true, "maximum_attenuation"},
		{0.0, 180.0, true, "zero_drop"},
		{-20.0, 180.0, true, "very_large_drop"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			result := lsxInvFResp(tc.drop, tc.attenuation)

			// Check for NaN
			assert.False(t, math.IsNaN(result), "Result should not be NaN")

			// Check for Inf
			assert.False(t, math.IsInf(result, 0), "Result should not be Inf")

			// For valid inputs, result should be in [0, 1]
			if tc.expectValid {
				assert.GreaterOrEqual(t, result, 0.0, "Result should be >= 0")
				assert.LessOrEqual(t, result, 1.0, "Result should be <= 1")
			}

			t.Logf("lsxInvFResp(%.4f, %.1f) = %.6f", tc.drop, tc.attenuation, result)
		})
	}
}

// TestLsxInvFResp_MonotonicBehavior tests that the function behaves monotonically
// as expected when varying parameters.
func TestLsxInvFResp_MonotonicBehavior(t *testing.T) {
	// For fixed attenuation, as drop decreases (more negative), result should increase
	attenuation := 180.0
	drops := []float64{-0.001, -0.01, -0.1, -1.0, -3.0, -6.0}
	var prevResult float64 = -1

	for _, drop := range drops {
		result := lsxInvFResp(drop, attenuation)
		if prevResult >= 0 {
			assert.Greater(t, result, prevResult,
				"Result should increase as drop decreases: drop=%.3f, prev=%.6f, curr=%.6f",
				drop, prevResult, result)
		}
		prevResult = result
	}
}

// =============================================================================
// Unit Tests for DFTStage Core Functionality
// =============================================================================

func TestDFTStage_Factor1_Passthrough(t *testing.T) {
	stage, err := NewDFTStage[float64](1, QualityHigh)
	require.NoError(t, err)

	// Verify it's a passthrough
	assert.Equal(t, 1, stage.factor)

	// Test that input equals output
	input := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	output, err := stage.Process(input)
	require.NoError(t, err)

	assert.Len(t, output, len(input))
	for i := range input {
		assert.InDelta(t, input[i], output[i], 1e-15)
	}
}

func TestDFTStage_Factor2_OutputLength(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	// Generate test signal
	numSamples := 1000
	input := make([]float64, numSamples)
	for i := range input {
		input[i] = 1.0 // DC signal
	}

	output, err := stage.Process(input)
	require.NoError(t, err)

	flush, err := stage.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	// Should produce approximately 2x samples (minus filter delay)
	ratio := float64(len(output)) / float64(numSamples)
	assert.InDelta(t, 2.0, ratio, 0.2, "Factor=2 should produce ~2x output")
}

func TestDFTStage_FilterCoefficients(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	// Verify polyphase structure
	assert.Len(t, stage.polyCoeffs, 2, "Should have 2 phases for factor=2")
	assert.Len(t, stage.polyCoeffs[0], stage.tapsPerPhase, "Phase 0 should have tapsPerPhase coefficients")
	assert.Len(t, stage.polyCoeffs[1], stage.tapsPerPhase, "Phase 1 should have tapsPerPhase coefficients")

	// Verify coefficient sums are reasonable
	sum0 := 0.0
	sum1 := 0.0
	for _, c := range stage.polyCoeffs[0] {
		sum0 += c
	}
	for _, c := range stage.polyCoeffs[1] {
		sum1 += c
	}

	// Each phase should have DC gain ≈ 1.0 for proper 2x upsampling
	assert.InDelta(t, 1.0, sum0, 0.01, "Phase 0 sum should be ~1.0")
	assert.InDelta(t, 1.0, sum1, 0.01, "Phase 1 sum should be ~1.0")
}

// =============================================================================
// Unit Tests for PolyphaseStage Core Functionality
// =============================================================================

func TestPolyphaseStage_Constructor(t *testing.T) {
	testCases := []struct {
		ratio        float64
		totalIORatio float64
		expectError  bool
		desc         string
	}{
		{1.088435374, 0.459375, false, "valid_upsample"},
		{0.91875, 0.91875, false, "valid_downsample"},
		{0.5, 1.0, false, "half_ratio"},
		{2.0, 0.5, false, "double_ratio"},
		{0.0, 1.0, true, "zero_ratio"},
		{-1.0, 1.0, true, "negative_ratio"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			stage, err := NewPolyphaseStage[float64](tc.ratio, tc.totalIORatio, QualityHigh)
			if tc.expectError {
				require.Error(t, err)
				assert.Nil(t, stage)
			} else {
				require.NoError(t, err)
				require.NotNil(t, stage)
				assert.Positive(t, stage.numPhases)
				assert.Positive(t, stage.tapsPerPhase)
				assert.Positive(t, stage.step)
			}
		})
	}
}

func TestPolyphaseStage_DCGain(t *testing.T) {
	// Test that DC signals pass through with gain ≈ 1.0
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, QualityHigh)
	require.NoError(t, err)

	// Generate DC signal
	numSamples := 5000
	input := make([]float64, numSamples)
	for i := range input {
		input[i] = 1.0
	}

	output, err := stage.Process(input)
	require.NoError(t, err)
	flush, err := stage.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	// Check DC level in stable region
	if len(output) < 100 {
		t.Skip("Not enough output for DC test")
	}
	startIdx := len(output) / 4
	endIdx := 3 * len(output) / 4

	sum := 0.0
	for i := startIdx; i < endIdx; i++ {
		sum += output[i]
	}
	dcGain := sum / float64(endIdx-startIdx)

	assert.InDelta(t, 1.0, dcGain, 0.01, "DC gain should be ~1.0")
}

func TestPolyphaseStage_OutputRatio(t *testing.T) {
	testCases := []struct {
		ratio        float64
		totalIORatio float64
		desc         string
	}{
		{1.088435374, 0.459375, "slightly_upsample"},
		{0.91875, 0.91875, "slightly_downsample"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			stage, err := NewPolyphaseStage[float64](tc.ratio, tc.totalIORatio, QualityHigh)
			require.NoError(t, err)

			numSamples := 10000
			input := make([]float64, numSamples)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
			}

			output, err := stage.Process(input)
			require.NoError(t, err)
			flush, err := stage.Flush()
			require.NoError(t, err)
			output = append(output, flush...)

			actualRatio := float64(len(output)) / float64(numSamples)
			assert.InDelta(t, tc.ratio, actualRatio, 0.1,
				"Output ratio should be ~%.3f, got %.3f", tc.ratio, actualRatio)
		})
	}
}

// =============================================================================
// Unit Tests for Resampler Core Functionality
// =============================================================================

func TestResampler_Constructor(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		expectErr  bool
		desc       string
	}{
		{44100, 48000, false, "valid_cd_to_dat"},
		{48000, 44100, false, "valid_dat_to_cd"},
		{48000, 96000, false, "valid_2x_upsample"},
		{96000, 48000, false, "valid_2x_downsample"},
		{0, 48000, true, "invalid_zero_input"},
		{44100, 0, true, "invalid_zero_output"},
		{-44100, 48000, true, "invalid_negative_input"},
		{44100, -48000, true, "invalid_negative_output"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			if tc.expectErr {
				require.Error(t, err)
				assert.Nil(t, resampler)
			} else {
				require.NoError(t, err)
				require.NotNil(t, resampler)
				assert.InDelta(t, tc.outputRate/tc.inputRate, resampler.GetRatio(), 1e-10)
			}
		})
	}
}

func TestResampler_QualityLevels(t *testing.T) {
	qualities := []Quality{QualityLow, QualityMedium, QualityHigh}

	for _, q := range qualities {
		t.Run(qualityName(q), func(t *testing.T) {
			resampler, err := NewResampler[float64](44100, 48000, q)
			require.NoError(t, err)

			input := make([]float64, 1000)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * float64(i) / 44.1)
			}

			output, err := resampler.Process(input)
			require.NoError(t, err)

			// Verify output is valid
			for i, v := range output {
				assert.False(t, math.IsNaN(v), "output[%d] is NaN", i)
				assert.False(t, math.IsInf(v, 0), "output[%d] is Inf", i)
			}
		})
	}
}

func TestResampler_Statistics(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	input := make([]float64, 1000)
	for i := range input {
		input[i] = 1.0
	}

	_, err = resampler.Process(input)
	require.NoError(t, err)

	stats := resampler.GetStatistics()
	assert.Equal(t, int64(1000), stats["samplesIn"])
	assert.Positive(t, stats["samplesOut"])
}

func TestResampler_ResetClearStats(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	// Process some data
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 1.0
	}
	_, err = resampler.Process(input)
	require.NoError(t, err)

	// Reset
	resampler.Reset()

	// Verify statistics are cleared
	stats := resampler.GetStatistics()
	assert.Equal(t, int64(0), stats["samplesIn"])
	assert.Equal(t, int64(0), stats["samplesOut"])
}

// =============================================================================
// Unit Tests for qualityToAttenuation
// =============================================================================

func TestQualityToAttenuation(t *testing.T) {
	testCases := []struct {
		quality       Quality
		expectedAtten float64
	}{
		{QualityLow, 100.0},
		{QualityMedium, 140.0},
		{QualityHigh, 180.0},
		{Quality(999), 140.0}, // Unknown defaults to medium
	}

	for _, tc := range testCases {
		t.Run(qualityName(tc.quality), func(t *testing.T) {
			result := qualityToAttenuation(tc.quality)
			assert.InDelta(t, tc.expectedAtten, result, 1e-10)
		})
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

func qualityName(q Quality) string {
	switch q {
	case QualityLow:
		return "low"
	case QualityMedium:
		return "medium"
	case QualityHigh:
		return "high"
	default:
		return "unknown"
	}
}
