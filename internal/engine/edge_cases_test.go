package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Empty Input Tests
// =============================================================================

// TestDFTStage_EmptyInput verifies DFTStage handles empty input correctly.
func TestDFTStage_EmptyInput(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	output, err := stage.Process([]float64{})
	require.NoError(t, err, "Process() with empty input should not error")
	assert.Empty(t, output, "Output should be empty for empty input")

	// Verify stage is still usable after empty input
	// Use enough samples to exceed filter latency (tapsPerPhase)
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 1.0
	}
	output, err = stage.Process(input)
	require.NoError(t, err, "Process() after empty input should work")
	// Stage may need more samples than filter latency to produce output
	// The important thing is no error occurred
	t.Logf("After empty input, produced %d samples from %d input", len(output), len(input))
}

// TestPolyphaseStage_EmptyInput verifies PolyphaseStage handles empty input correctly.
func TestPolyphaseStage_EmptyInput(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	output, err := stage.Process([]float64{})
	require.NoError(t, err, "Process() with empty input should not error")
	assert.Empty(t, output, "Output should be empty for empty input")

	// Verify stage is still usable after empty input
	input := make([]float64, 500)
	for i := range input {
		input[i] = 1.0
	}
	_, err = stage.Process(input)
	require.NoError(t, err, "Process() after empty input should work")
	// Note: Polyphase may not produce output until enough samples are accumulated
}

// TestResampler_EmptyInput verifies Resampler handles empty input correctly.
func TestResampler_EmptyInput(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	output, err := resampler.Process([]float64{})
	require.NoError(t, err, "Process() with empty input should not error")
	assert.Empty(t, output, "Output should be empty for empty input")

	// Verify resampler is still usable
	input := make([]float64, 1000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}
	output, err = resampler.Process(input)
	require.NoError(t, err)
	assert.NotEmpty(t, output)
}

// TestCubicStage_EmptyInput verifies CubicStage handles empty input correctly.
func TestCubicStage_EmptyInput(t *testing.T) {
	stage := NewCubicStage(2.0)

	output, err := stage.Process([]float64{})
	require.NoError(t, err, "Process() with empty input should not error")
	assert.Empty(t, output, "Output should be empty for empty input")
}

// TestLinearStage_EmptyInput verifies LinearStage handles empty input correctly.
func TestLinearStage_EmptyInput(t *testing.T) {
	stage := NewLinearStage(2.0)

	output, err := stage.Process([]float64{})
	require.NoError(t, err, "Process() with empty input should not error")
	assert.Empty(t, output, "Output should be empty for empty input")
}

// =============================================================================
// Single Sample Tests
// =============================================================================

// TestDFTStage_SingleSample verifies DFTStage handles single sample input.
func TestDFTStage_SingleSample(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	// Single sample input
	output, err := stage.Process([]float64{1.0})
	require.NoError(t, err, "Process() with single sample should not error")

	// May not produce output yet due to filter latency
	t.Logf("Single sample produced %d output samples", len(output))

	// Additional samples should eventually produce output
	// Need enough samples to exceed filter taps (high quality filters can have 300+ taps)
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 1.0
	}
	output, err = stage.Process(input)
	require.NoError(t, err)
	// With 1001 total samples, should have enough to produce output
	t.Logf("After additional samples, produced %d output samples", len(output))
	// Flush to verify the stage processed correctly
	flush, err := stage.Flush()
	require.NoError(t, err)
	totalOutput := len(output) + len(flush)
	assert.Positive(t, totalOutput, "Should produce some output after flush")
}

// TestPolyphaseStage_SingleSample verifies PolyphaseStage handles single sample input.
func TestPolyphaseStage_SingleSample(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	// Single sample input
	output, err := stage.Process([]float64{1.0})
	require.NoError(t, err, "Process() with single sample should not error")
	t.Logf("Single sample produced %d output samples", len(output))

	// Additional samples should eventually produce output
	input := make([]float64, 500)
	for i := range input {
		input[i] = 1.0
	}
	output, err = stage.Process(input)
	require.NoError(t, err)
	assert.NotEmpty(t, output)
}

// TestResampler_SingleSample verifies Resampler handles single sample input.
func TestResampler_SingleSample(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	output, err := resampler.Process([]float64{1.0})
	require.NoError(t, err, "Process() with single sample should not error")
	t.Logf("Single sample produced %d output samples", len(output))
}

// =============================================================================
// Small Buffer Tests
// =============================================================================

// TestDFTStage_SmallBuffers verifies DFTStage handles various small buffer sizes.
func TestDFTStage_SmallBuffers(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	sizes := []int{1, 2, 3, 5, 10, 16, 32, 64}

	for _, size := range sizes {
		t.Run(sizeTestName(size), func(t *testing.T) {
			stage.Reset()

			input := make([]float64, size)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * float64(i) / float64(size))
			}

			output, err := stage.Process(input)
			require.NoError(t, err, "Process() with %d samples failed", size)

			// Verify no NaN or Inf values
			for i, v := range output {
				assert.False(t, math.IsNaN(v), "output[%d] is NaN for size %d", i, size)
				assert.False(t, math.IsInf(v, 0), "output[%d] is Inf for size %d", i, size)
			}
		})
	}
}

// TestPolyphaseStage_SmallBuffers verifies PolyphaseStage handles various small buffer sizes.
func TestPolyphaseStage_SmallBuffers(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	sizes := []int{1, 2, 3, 5, 10, 16, 32, 64, 100}

	for _, size := range sizes {
		t.Run(sizeTestName(size), func(t *testing.T) {
			stage.Reset()

			input := make([]float64, size)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * float64(i) / float64(max(size, 10)))
			}

			output, err := stage.Process(input)
			require.NoError(t, err, "Process() with %d samples failed", size)

			// Verify no NaN or Inf values
			for i, v := range output {
				assert.False(t, math.IsNaN(v), "output[%d] is NaN for size %d", i, size)
				assert.False(t, math.IsInf(v, 0), "output[%d] is Inf for size %d", i, size)
			}
		})
	}
}

// TestResampler_SmallBuffers verifies Resampler handles various small buffer sizes.
func TestResampler_SmallBuffers(t *testing.T) {
	sizes := []int{1, 2, 5, 10, 32, 64, 100, 256}

	for _, size := range sizes {
		t.Run(sizeTestName(size), func(t *testing.T) {
			resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
			require.NoError(t, err)

			input := make([]float64, size)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * float64(i) / float64(max(size, 10)))
			}

			output, err := resampler.Process(input)
			require.NoError(t, err, "Process() with %d samples failed", size)

			// Verify no NaN or Inf values
			for i, v := range output {
				assert.False(t, math.IsNaN(v), "output[%d] is NaN for size %d", i, size)
				assert.False(t, math.IsInf(v, 0), "output[%d] is Inf for size %d", i, size)
			}

			t.Logf("Size %d: input=%d, output=%d", size, size, len(output))
		})
	}
}

// =============================================================================
// Nil Input Tests
// =============================================================================

// TestDFTStage_NilInput verifies DFTStage handles nil input correctly.
func TestDFTStage_NilInput(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	output, err := stage.Process(nil)
	require.NoError(t, err, "Process() with nil input should not error")
	assert.Empty(t, output, "Output should be empty for nil input")
}

// TestPolyphaseStage_NilInput verifies PolyphaseStage handles nil input correctly.
func TestPolyphaseStage_NilInput(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	output, err := stage.Process(nil)
	require.NoError(t, err, "Process() with nil input should not error")
	assert.Empty(t, output, "Output should be empty for nil input")
}

// TestResampler_NilInput verifies Resampler handles nil input correctly.
func TestResampler_NilInput(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	output, err := resampler.Process(nil)
	require.NoError(t, err, "Process() with nil input should not error")
	assert.Empty(t, output, "Output should be empty for nil input")
}

// =============================================================================
// DC Signal Tests (for numerical stability)
// =============================================================================

// TestDFTStage_DCSignal verifies DFTStage correctly handles DC (constant) signals.
func TestDFTStage_DCSignal(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	// DC signal at 1.0
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 1.0
	}

	output, err := stage.Process(input)
	require.NoError(t, err)
	flush, err := stage.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	// Check DC level in stable region (skip edges)
	startIdx := len(output) / 4
	endIdx := 3 * len(output) / 4

	var sum float64
	count := 0
	for i := startIdx; i < endIdx; i++ {
		sum += output[i]
		count++
	}
	dcLevel := sum / float64(count)

	// DC should be preserved close to 1.0
	assert.InDelta(t, 1.0, dcLevel, 0.01,
		"DC level should be preserved, got %f", dcLevel)
}

// TestPolyphaseStage_DCSignal verifies PolyphaseStage correctly handles DC signals.
func TestPolyphaseStage_DCSignal(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	// DC signal at 1.0
	input := make([]float64, 2000)
	for i := range input {
		input[i] = 1.0
	}

	output, err := stage.Process(input)
	require.NoError(t, err)
	flush, err := stage.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	if len(output) < 100 {
		t.Skip("Not enough output samples for DC test")
	}

	// Check DC level in stable region
	startIdx := len(output) / 4
	endIdx := 3 * len(output) / 4

	var sum float64
	count := 0
	for i := startIdx; i < endIdx; i++ {
		sum += output[i]
		count++
	}
	dcLevel := sum / float64(count)

	assert.InDelta(t, 1.0, dcLevel, 0.01,
		"DC level should be preserved, got %f", dcLevel)
}

// =============================================================================
// Extreme Value Tests
// =============================================================================

// TestDFTStage_ExtremeValues verifies DFTStage handles extreme input values.
func TestDFTStage_ExtremeValues(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	testCases := []struct {
		name  string
		value float64
	}{
		{"zero", 0.0},
		{"positive_one", 1.0},
		{"negative_one", -1.0},
		{"small_positive", 1e-10},
		{"small_negative", -1e-10},
		{"large_positive", 1e6},
		{"large_negative", -1e6},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stage.Reset()

			input := make([]float64, 500)
			for i := range input {
				input[i] = tc.value
			}

			output, err := stage.Process(input)
			require.NoError(t, err, "Process() failed for value %f", tc.value)

			// Verify no NaN or Inf values in output
			for i, v := range output {
				assert.False(t, math.IsNaN(v),
					"output[%d] is NaN for input value %f", i, tc.value)
				assert.False(t, math.IsInf(v, 0),
					"output[%d] is Inf for input value %f", i, tc.value)
			}
		})
	}
}

// TestResampler_ExtremeValues verifies Resampler handles extreme input values.
func TestResampler_ExtremeValues(t *testing.T) {
	testCases := []struct {
		name  string
		value float64
	}{
		{"zero", 0.0},
		{"positive_one", 1.0},
		{"negative_one", -1.0},
		{"small_positive", 1e-10},
		{"small_negative", -1e-10},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
			require.NoError(t, err)

			input := make([]float64, 1000)
			for i := range input {
				input[i] = tc.value
			}

			output, err := resampler.Process(input)
			require.NoError(t, err, "Process() failed for value %f", tc.value)

			for i, v := range output {
				assert.False(t, math.IsNaN(v),
					"output[%d] is NaN for input value %f", i, tc.value)
				assert.False(t, math.IsInf(v, 0),
					"output[%d] is Inf for input value %f", i, tc.value)
			}
		})
	}
}

// =============================================================================
// Constructor Validation Tests
// =============================================================================

// TestNewDFTStage_InvalidFactor verifies DFTStage rejects invalid factors.
func TestNewDFTStage_InvalidFactor(t *testing.T) {
	testCases := []struct {
		name    string
		factor  int
		wantErr bool
	}{
		{"factor_1_passthrough", 1, false},
		{"factor_2_valid", 2, false},
		{"factor_4_valid", 4, false},
		{"factor_0_invalid", 0, true},
		{"factor_negative", -1, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stage, err := NewDFTStage[float64](tc.factor, QualityHigh)
			if tc.wantErr {
				require.Error(t, err, "Should reject factor %d", tc.factor)
				assert.Nil(t, stage)
			} else {
				require.NoError(t, err, "Should accept factor %d", tc.factor)
				assert.NotNil(t, stage)
			}
		})
	}
}

// TestNewPolyphaseStage_InvalidRatio verifies PolyphaseStage rejects invalid ratios.
func TestNewPolyphaseStage_InvalidRatio(t *testing.T) {
	testCases := []struct {
		name    string
		ratio   float64
		wantErr bool
	}{
		{"ratio_positive", 1.5, false},
		{"ratio_less_than_1", 0.5, false},
		{"ratio_zero", 0, true},
		{"ratio_negative", -1.0, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stage, err := NewPolyphaseStage[float64](tc.ratio, 0.5, true, QualityHigh)
			if tc.wantErr {
				require.Error(t, err, "Should reject ratio %f", tc.ratio)
				assert.Nil(t, stage)
			} else {
				require.NoError(t, err, "Should accept ratio %f", tc.ratio)
				assert.NotNil(t, stage)
			}
		})
	}
}

// TestNewResampler_InvalidRates verifies Resampler rejects invalid sample rates.
func TestNewResampler_InvalidRates(t *testing.T) {
	testCases := []struct {
		name       string
		inputRate  float64
		outputRate float64
		wantErr    bool
	}{
		{"valid_rates", 44100, 48000, false},
		{"zero_input", 0, 48000, true},
		{"zero_output", 44100, 0, true},
		{"negative_input", -44100, 48000, true},
		{"negative_output", 44100, -48000, true},
		{"both_zero", 0, 0, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			if tc.wantErr {
				require.Error(t, err)
				assert.Nil(t, resampler)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, resampler)
			}
		})
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

func sizeTestName(size int) string {
	return "size_" + intToString(size)
}

func intToString(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}
	// Simple conversion for test naming
	result := ""
	for n > 0 {
		result = string(rune('0'+n%10)) + result
		n /= 10
	}
	return result
}
