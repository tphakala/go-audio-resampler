package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Reset State Tests - Verify Reset() properly clears internal state
// =============================================================================

// TestDFTStage_Reset verifies Reset() properly clears DFTStage state.
func TestDFTStage_Reset(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	// Process some data to populate internal state
	input1 := make([]float64, 1000)
	for i := range input1 {
		input1[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}
	output1, err := stage.Process(input1)
	require.NoError(t, err)
	require.NotEmpty(t, output1)

	// Reset the stage
	stage.Reset()

	// Process the same data again - should produce identical output
	stage2, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	output2, err := stage.Process(input1)
	require.NoError(t, err)

	outputFresh, err := stage2.Process(input1)
	require.NoError(t, err)

	// After reset, output should match a fresh stage
	assert.Len(t, output2, len(outputFresh),
		"Output length after reset should match fresh stage")

	for i := 0; i < min(len(output2), len(outputFresh)); i++ {
		assert.InDelta(t, outputFresh[i], output2[i], 1e-15,
			"Output[%d] after reset differs from fresh stage", i)
	}

	t.Log("DFT stage Reset() verified: state properly cleared")
}

// TestPolyphaseStage_Reset verifies Reset() properly clears PolyphaseStage state.
func TestPolyphaseStage_Reset(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	// Process some data to populate internal state
	input1 := make([]float64, 2000)
	for i := range input1 {
		input1[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}
	output1, err := stage.Process(input1)
	require.NoError(t, err)
	require.NotEmpty(t, output1)

	// Reset the stage
	stage.Reset()

	// Process the same data again
	stage2, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	output2, err := stage.Process(input1)
	require.NoError(t, err)

	outputFresh, err := stage2.Process(input1)
	require.NoError(t, err)

	// After reset, output should match a fresh stage
	assert.Len(t, output2, len(outputFresh),
		"Output length after reset should match fresh stage")

	for i := 0; i < min(len(output2), len(outputFresh)); i++ {
		assert.InDelta(t, outputFresh[i], output2[i], 1e-15,
			"Output[%d] after reset differs from fresh stage", i)
	}

	t.Log("Polyphase stage Reset() verified: state properly cleared")
}

// TestResampler_Reset verifies Reset() properly clears Resampler state.
func TestResampler_Reset(t *testing.T) {
	testCases := []struct {
		name       string
		inputRate  float64
		outputRate float64
	}{
		{"44100_to_48000", 44100, 48000},
		{"44100_to_96000", 44100, 96000},
		{"44100_to_88200", 44100, 88200},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err)

			// Process some data
			input := make([]float64, 4000)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * 1000 * float64(i) / tc.inputRate)
			}
			output1, err := resampler.Process(input)
			require.NoError(t, err)
			require.NotEmpty(t, output1)

			// Reset
			resampler.Reset()

			// Process same data again
			output2, err := resampler.Process(input)
			require.NoError(t, err)

			// Create fresh resampler for comparison
			freshResampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err)
			outputFresh, err := freshResampler.Process(input)
			require.NoError(t, err)

			// Should match fresh resampler
			assert.Len(t, output2, len(outputFresh),
				"Output length after reset should match fresh resampler")

			for i := 0; i < min(len(output2), len(outputFresh)); i++ {
				assert.InDelta(t, outputFresh[i], output2[i], 1e-15,
					"Output[%d] after reset differs from fresh resampler", i)
			}
		})
	}
}

// interpolationStage is an interface for testing cubic and linear stages.
type interpolationStage interface {
	Process([]float64) ([]float64, error)
	Reset()
}

// TestInterpolationStages_Reset verifies Reset() properly clears CubicStage and LinearStage state.
func TestInterpolationStages_Reset(t *testing.T) {
	testCases := []struct {
		name     string
		newStage func() interpolationStage
		newFresh func() interpolationStage
	}{
		{
			name:     "CubicStage",
			newStage: func() interpolationStage { return NewCubicStage(2.0) },
			newFresh: func() interpolationStage { return NewCubicStage(2.0) },
		},
		{
			name:     "LinearStage",
			newStage: func() interpolationStage { return NewLinearStage(2.0) },
			newFresh: func() interpolationStage { return NewLinearStage(2.0) },
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stage := tc.newStage()

			// Process some data
			input := make([]float64, 1000)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
			}
			output1, err := stage.Process(input)
			require.NoError(t, err)
			require.NotEmpty(t, output1)

			// Reset
			stage.Reset()

			// Process same data again
			output2, err := stage.Process(input)
			require.NoError(t, err)

			// Create fresh stage for comparison
			freshStage := tc.newFresh()
			outputFresh, err := freshStage.Process(input)
			require.NoError(t, err)

			// Should match fresh stage
			assert.Len(t, output2, len(outputFresh),
				"Output length after reset should match fresh stage")

			for i := 0; i < min(len(output2), len(outputFresh)); i++ {
				assert.InDelta(t, outputFresh[i], output2[i], 1e-15,
					"Output[%d] after reset differs from fresh stage", i)
			}

			t.Logf("%s Reset() verified", tc.name)
		})
	}
}

// =============================================================================
// Multiple Reset Tests - Verify Reset() can be called multiple times
// =============================================================================

// TestDFTStage_MultipleResets verifies Reset() can be called repeatedly.
func TestDFTStage_MultipleResets(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	input := make([]float64, 500)
	for i := range input {
		input[i] = 1.0
	}

	for round := range 5 {
		// Process
		output, err := stage.Process(input)
		require.NoError(t, err, "Round %d Process() failed", round)

		// Verify output is valid
		for i, v := range output {
			assert.False(t, math.IsNaN(v), "Round %d: output[%d] is NaN", round, i)
		}

		// Reset
		stage.Reset()
	}

	t.Log("DFT stage: 5 Reset() cycles completed successfully")
}

// TestResampler_MultipleResets verifies Resampler Reset() can be called repeatedly.
func TestResampler_MultipleResets(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	input := make([]float64, 1000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	for round := range 5 {
		// Process
		output, err := resampler.Process(input)
		require.NoError(t, err, "Round %d Process() failed", round)

		// Verify output
		for i, v := range output {
			assert.False(t, math.IsNaN(v), "Round %d: output[%d] is NaN", round, i)
		}

		// Reset
		resampler.Reset()
	}

	t.Log("Resampler: 5 Reset() cycles completed successfully")
}

// =============================================================================
// Reset During Processing Tests
// =============================================================================

// TestDFTStage_ResetMidStream verifies Reset() properly handles mid-stream reset.
func TestDFTStage_ResetMidStream(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)

	// Process partial data (not enough to flush all)
	input1 := make([]float64, 100)
	for i := range input1 {
		input1[i] = math.Sin(2.0 * math.Pi * float64(i) / 50)
	}
	_, err = stage.Process(input1)
	require.NoError(t, err)

	// Reset mid-stream (there may be buffered data)
	stage.Reset()

	// Process new data - should not be contaminated by old data
	input2 := make([]float64, 500)
	for i := range input2 {
		input2[i] = 1.0 // DC signal
	}

	output, err := stage.Process(input2)
	require.NoError(t, err)
	flush, err := stage.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	// Output should be a DC signal (not contaminated by sine)
	if len(output) > 100 {
		startIdx := len(output) / 4
		endIdx := 3 * len(output) / 4

		var sum float64
		for i := startIdx; i < endIdx; i++ {
			sum += output[i]
		}
		dcLevel := sum / float64(endIdx-startIdx)

		// Should be close to 1.0 (not contaminated by earlier sine)
		assert.InDelta(t, 1.0, dcLevel, 0.1,
			"DC level after reset should be ~1.0, got %f", dcLevel)
	}

	t.Log("DFT stage: mid-stream reset verified")
}

// TestResampler_ResetMidStream verifies Reset() properly handles mid-stream reset.
func TestResampler_ResetMidStream(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	// Process partial data
	input1 := make([]float64, 500)
	for i := range input1 {
		input1[i] = math.Sin(2.0 * math.Pi * float64(i) / 50)
	}
	_, err = resampler.Process(input1)
	require.NoError(t, err)

	// Reset mid-stream
	resampler.Reset()

	// Process new data - should not be contaminated
	input2 := make([]float64, 2000)
	for i := range input2 {
		input2[i] = 1.0 // DC signal
	}

	output, err := resampler.Process(input2)
	require.NoError(t, err)
	flush, err := resampler.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	// Output should be DC (not contaminated)
	if len(output) > 100 {
		startIdx := len(output) / 4
		endIdx := 3 * len(output) / 4

		var sum float64
		for i := startIdx; i < endIdx; i++ {
			sum += output[i]
		}
		dcLevel := sum / float64(endIdx-startIdx)

		assert.InDelta(t, 1.0, dcLevel, 0.1,
			"DC level after reset should be ~1.0, got %f", dcLevel)
	}

	t.Log("Resampler: mid-stream reset verified")
}

// =============================================================================
// Statistics Reset Tests
// =============================================================================

// TestResampler_StatisticsReset verifies statistics are cleared on Reset().
func TestResampler_StatisticsReset(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	// Process some data
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 1.0
	}
	_, err = resampler.Process(input)
	require.NoError(t, err)

	// Check statistics are non-zero
	stats := resampler.GetStatistics()
	assert.Positive(t, stats["samplesIn"], "samplesIn should be non-zero before reset")

	// Reset
	resampler.Reset()

	// Check statistics are cleared
	statsAfter := resampler.GetStatistics()
	assert.Equal(t, int64(0), statsAfter["samplesIn"], "samplesIn should be zero after reset")
	assert.Equal(t, int64(0), statsAfter["samplesOut"], "samplesOut should be zero after reset")

	t.Log("Statistics reset verified")
}

// TestPolyphaseStage_StatisticsReset verifies statistics are cleared on Reset().
func TestPolyphaseStage_StatisticsReset(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err)

	// Process some data
	input := make([]float64, 2000)
	for i := range input {
		input[i] = 1.0
	}
	_, err = stage.Process(input)
	require.NoError(t, err)

	// Check statistics are non-zero
	stats := stage.GetStatistics()
	assert.Positive(t, stats["samplesIn"], "samplesIn should be non-zero before reset")

	// Reset
	stage.Reset()

	// Check statistics are cleared
	statsAfter := stage.GetStatistics()
	assert.Equal(t, int64(0), statsAfter["samplesIn"], "samplesIn should be zero after reset")
	assert.Equal(t, int64(0), statsAfter["samplesOut"], "samplesOut should be zero after reset")

	t.Log("Polyphase statistics reset verified")
}
