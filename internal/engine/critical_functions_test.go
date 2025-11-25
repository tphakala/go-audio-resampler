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
		{0.999999, false, "almost_1_far"}, // Far enough from 1 to not round
		{0.9999999999, true, "almost_1"},  // Rounds to 1 within tolerance
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
// Unit Tests for ComputePolyphaseFilterParams (Fn Normalization)
// =============================================================================

// TestComputePolyphaseFilterParams_FnNormalization tests that the Fn normalization
// is correctly applied based on upsampling/downsampling AND presence of pre-stage.
//
// From soxr cr.c lines 429-431:
//
//	if (!upsample && preM)
//	  Fn = 2 * mult, Fs = 3 + fabs(Fs1 - 1);
//	else
//	  Fn = 1, Fs = 2 - (mode? Fp1 + (Fs1 - Fp1) * .7 : Fs1);
//
// Key insight: Fn=2*mult is ONLY used when downsampling WITH a pre-stage.
// For downsampling WITHOUT pre-stage (our Go architecture), Fn=1.
func TestComputePolyphaseFilterParams_FnNormalization(t *testing.T) {
	const attenuation = 126.0 // QualityHigh

	testCases := []struct {
		name         string
		numPhases    int
		ratio        float64
		totalIORatio float64
		hasPreStage  bool
		expectFn     float64 // Expected Fn value (approx)
		isUpsampling bool
	}{
		// Upsampling cases with pre-stage: Fn should be 1.0
		{
			name:         "44.1kHz_to_48kHz_upsampling_with_prestage",
			numPhases:    147,
			ratio:        48000.0 / 44100.0, // ~1.088
			totalIORatio: 44100.0 / 48000.0, // ~0.919
			hasPreStage:  true,
			expectFn:     1.0,
			isUpsampling: true,
		},
		{
			name:         "44.1kHz_to_96kHz_upsampling_with_prestage",
			numPhases:    147,
			ratio:        96000.0 / 44100.0, // ~2.177
			totalIORatio: 44100.0 / 96000.0, // ~0.459
			hasPreStage:  true,
			expectFn:     1.0,
			isUpsampling: true,
		},
		// Downsampling WITHOUT pre-stage: Fn should be 1.0 (our Go architecture)
		{
			name:         "48kHz_to_44.1kHz_downsampling_no_prestage",
			numPhases:    160,
			ratio:        44100.0 / 48000.0, // ~0.919
			totalIORatio: 48000.0 / 44100.0, // ~1.088
			hasPreStage:  false,
			expectFn:     1.0, // Fn=1 when no pre-stage
			isUpsampling: false,
		},
		{
			name:         "96kHz_to_48kHz_downsampling_no_prestage",
			numPhases:    1,
			ratio:        48000.0 / 96000.0, // 0.5
			totalIORatio: 96000.0 / 48000.0, // 2.0
			hasPreStage:  false,
			expectFn:     1.0, // Fn=1 when no pre-stage
			isUpsampling: false,
		},
		{
			name:         "48kHz_to_32kHz_downsampling_no_prestage",
			numPhases:    2,
			ratio:        32000.0 / 48000.0, // ~0.667
			totalIORatio: 48000.0 / 32000.0, // 1.5
			hasPreStage:  false,
			expectFn:     1.0, // Fn=1 when no pre-stage
			isUpsampling: false,
		},
		// Downsampling WITH pre-stage: Fn should be 2 * mult (soxr formula)
		{
			name:         "48kHz_to_44.1kHz_downsampling_with_prestage",
			numPhases:    160,
			ratio:        44100.0 / 48000.0, // ~0.919
			totalIORatio: 48000.0 / 44100.0, // ~1.088
			hasPreStage:  true,
			expectFn:     2.0 * 1.088, // ~2.176
			isUpsampling: false,
		},
		{
			name:         "96kHz_to_48kHz_downsampling_with_prestage",
			numPhases:    1,
			ratio:        48000.0 / 96000.0, // 0.5
			totalIORatio: 96000.0 / 48000.0, // 2.0
			hasPreStage:  true,
			expectFn:     2.0 * 2.0, // 4.0
			isUpsampling: false,
		},
	}

	passbandEnd := 0.912 // QualityHigh passband end

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := ComputePolyphaseFilterParams(tc.numPhases, tc.ratio, tc.totalIORatio, tc.hasPreStage, attenuation, passbandEnd)

			// Verify upsampling detection
			assert.Equal(t, tc.isUpsampling, params.IsUpsampling,
				"IsUpsampling should be %v", tc.isUpsampling)

			// Verify hasPreStage is stored
			assert.Equal(t, tc.hasPreStage, params.HasPreStage,
				"HasPreStage should be %v", tc.hasPreStage)

			// Verify Fn value (within 1% tolerance)
			assert.InDelta(t, tc.expectFn, params.Fn, tc.expectFn*0.01,
				"Fn should be approximately %.3f", tc.expectFn)

			// For downsampling WITH pre-stage, verify Fs formula: Fs = 3 + |Fs1 - 1|
			// where Fs1 = ratio (for downsampling)
			if !tc.isUpsampling && tc.hasPreStage {
				expectedFsRaw := 3.0 + math.Abs(tc.ratio-1.0)
				assert.InDelta(t, expectedFsRaw, params.FsRaw, 0.01,
					"FsRaw for downsampling with pre-stage should be 3 + |ratio - 1| = %.4f", expectedFsRaw)
			}

			// Verify Fp and Fs are normalized by Fn
			assert.InDelta(t, params.FpRaw/params.Fn, params.Fp, 0.0001,
				"Fp should equal FpRaw / Fn")
			assert.InDelta(t, params.FsRaw/params.Fn, params.Fs, 0.0001,
				"Fs should equal FsRaw / Fn")

			// Verify Fc is reasonable (positive and less than 1)
			assert.Greater(t, params.Fc, 0.0, "Fc should be positive")
			assert.Less(t, params.Fc, 1.0, "Fc should be less than 1")

			// Log detailed parameters for debugging
			t.Logf("Params for %s:", tc.name)
			t.Logf("  IsUpsampling: %v, HasPreStage: %v", params.IsUpsampling, params.HasPreStage)
			t.Logf("  Mult: %.4f", params.Mult)
			t.Logf("  Fn: %.4f (expected: %.4f)", params.Fn, tc.expectFn)
			t.Logf("  Fp1: %.4f, FsRaw: %.4f, FpRaw: %.4f", params.Fp1, params.FsRaw, params.FpRaw)
			t.Logf("  Fp (normalized): %.4f, Fs (normalized): %.4f", params.Fp, params.Fs)
			t.Logf("  TrBw: %.6f, Fc: %.6f", params.TrBw, params.Fc)
			t.Logf("  TotalTaps: %d, TapsPerPhase: %d", params.TotalTaps, params.TapsPerPhase)
		})
	}
}

// TestComputePolyphaseFilterParams_DownsamplingVsUpsampling verifies that
// the hasPreStage parameter correctly changes the filter design parameters.
func TestComputePolyphaseFilterParams_DownsamplingVsUpsampling(t *testing.T) {
	const attenuation = 126.0 // QualityHigh
	const passbandEnd = 0.912 // QualityHigh passband end

	// Compare upsampling with pre-stage (our Go architecture)
	// vs downsampling without pre-stage (our Go architecture)
	// vs downsampling with pre-stage (hypothetical)
	upParams := ComputePolyphaseFilterParams(147, 48000.0/44100.0, 44100.0/48000.0, true, attenuation, passbandEnd)
	downNoPreParams := ComputePolyphaseFilterParams(160, 44100.0/48000.0, 48000.0/44100.0, false, attenuation, passbandEnd)
	downWithPreParams := ComputePolyphaseFilterParams(160, 44100.0/48000.0, 48000.0/44100.0, true, attenuation, passbandEnd)

	// Upsampling should have Fn = 1
	assert.InEpsilon(t, 1.0, upParams.Fn, 1e-9, "Upsampling Fn should be 1.0")

	// Downsampling WITHOUT pre-stage should also have Fn = 1 (matches soxr logic)
	assert.InEpsilon(t, 1.0, downNoPreParams.Fn, 1e-9, "Downsampling without pre-stage Fn should be 1.0")

	// Downsampling WITH pre-stage should have Fn = 2 * mult ≈ 2.176
	assert.Greater(t, downWithPreParams.Fn, 1.5, "Downsampling with pre-stage Fn should be > 1.5")

	// Downsampling WITH pre-stage FsRaw should be 3 + |ratio - 1|
	// For 48→44.1: ratio = 44100/48000 = 0.91875, so FsRaw = 3.08125
	expectedFsRaw := 3.0 + math.Abs(44100.0/48000.0-1.0)
	assert.InDelta(t, expectedFsRaw, downWithPreParams.FsRaw, 0.01,
		"Downsampling with pre-stage FsRaw should be 3 + |ratio - 1| = %.4f", expectedFsRaw)

	// Downsampling WITHOUT pre-stage should use the image rejection formula for FsRaw
	// FsRaw = 2 - (Fp1 + (Fs1 - Fp1) * 0.7)
	assert.Less(t, downNoPreParams.FsRaw, 2.0, "Downsampling without pre-stage FsRaw should be < 2.0")

	t.Logf("Upsampling (44.1k→48k): Fn=%.3f, FsRaw=%.3f, Fs=%.3f, Fc=%.6f",
		upParams.Fn, upParams.FsRaw, upParams.Fs, upParams.Fc)
	t.Logf("Downsampling no pre-stage (48k→44.1k): Fn=%.3f, FsRaw=%.3f, Fs=%.3f, Fc=%.6f",
		downNoPreParams.Fn, downNoPreParams.FsRaw, downNoPreParams.Fs, downNoPreParams.Fc)
	t.Logf("Downsampling with pre-stage (48k→44.1k): Fn=%.3f, FsRaw=%.3f, Fs=%.3f, Fc=%.6f",
		downWithPreParams.Fn, downWithPreParams.FsRaw, downWithPreParams.Fs, downWithPreParams.Fc)
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
			stage, err := NewPolyphaseStage[float64](tc.ratio, tc.totalIORatio, true, QualityHigh)
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
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
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
			stage, err := NewPolyphaseStage[float64](tc.ratio, tc.totalIORatio, true, QualityHigh)
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
		// Attenuation = (bits + 1) * 6.0206 dB
		{QualityLow, (16 + 1) * 6.0206},    // 102.35 dB
		{QualityMedium, (16 + 1) * 6.0206}, // 102.35 dB (same bits as Low)
		{QualityHigh, (20 + 1) * 6.0206},   // 126.43 dB
		{Quality(999), (20 + 1) * 6.0206},  // Unknown defaults to High
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
	case QualityQuick:
		return "quick"
	case QualityLow:
		return "low"
	case QualityMedium:
		return "medium"
	case QualityHigh:
		return "high"
	case QualityVeryHigh:
		return "veryHigh"
	case Quality16Bit:
		return "16bit"
	case Quality20Bit:
		return "20bit"
	case Quality24Bit:
		return "24bit"
	case Quality28Bit:
		return "28bit"
	case Quality32Bit:
		return "32bit"
	default:
		return "unknown"
	}
}
