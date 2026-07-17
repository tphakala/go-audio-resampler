// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDFTStage_BufferIntegrity verifies that output buffers remain valid
// after subsequent calls to Process() or Flush().
// This is a regression test for the buffer reuse bug that corrupted output
// when callers stored results and then called Process/Flush again.
func TestDFTStage_BufferIntegrity(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err, "Failed to create DFT stage")

	// Generate test signal
	input := make([]float64, 1000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	// First process call
	output1, err := stage.Process(input)
	require.NoError(t, err, "First Process() failed")
	require.NotEmpty(t, output1, "First output should not be empty")

	// Save first few values for comparison
	savedValues := make([]float64, min(10, len(output1)))
	copy(savedValues, output1)

	// Second process call - this should NOT corrupt output1
	input2 := make([]float64, 500)
	for i := range input2 {
		input2[i] = math.Cos(2.0 * math.Pi * float64(i) / 50)
	}
	_, err = stage.Process(input2)
	require.NoError(t, err, "Second Process() failed")

	// Verify output1 was not corrupted
	for i, expected := range savedValues {
		assert.InDelta(t, expected, output1[i], 1e-15,
			"output1[%d] was corrupted after second Process() call: expected %f, got %f",
			i, expected, output1[i])
	}

	t.Log("DFT stage buffer integrity verified: output survives subsequent Process() calls")
}

// TestDFTStage_FlushDoesNotCorruptOutput verifies that calling Flush()
// after Process() does not corrupt the previously returned output.
func TestDFTStage_FlushDoesNotCorruptOutput(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err, "Failed to create DFT stage")

	// Generate test signal
	input := make([]float64, 1000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	// Process the signal
	output, err := stage.Process(input)
	require.NoError(t, err, "Process() failed")
	require.NotEmpty(t, output, "Output should not be empty")

	// Save all values for comparison
	savedOutput := make([]float64, len(output))
	copy(savedOutput, output)

	// Flush the stage - this should NOT corrupt the previous output
	flush, err := stage.Flush()
	require.NoError(t, err, "Flush() failed")

	// Verify output was not corrupted
	for i, expected := range savedOutput {
		assert.InDelta(t, expected, output[i], 1e-15,
			"output[%d] was corrupted after Flush(): expected %f, got %f",
			i, expected, output[i])
	}

	// Also verify flush output is valid
	for i, v := range flush {
		assert.False(t, math.IsNaN(v), "flush[%d] is NaN", i)
		assert.False(t, math.IsInf(v, 0), "flush[%d] is Inf", i)
	}

	t.Log("DFT stage: output survives Flush() call")
}

// TestPolyphaseStage_BufferIntegrity verifies that output buffers remain valid
// after subsequent calls to Process() or Flush().
func TestPolyphaseStage_BufferIntegrity(t *testing.T) {
	// Create polyphase stage for non-integer ratio
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err, "Failed to create polyphase stage")

	// Generate test signal
	input := make([]float64, 2000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	// First process call
	output1, err := stage.Process(input)
	require.NoError(t, err, "First Process() failed")
	require.NotEmpty(t, output1, "First output should not be empty")

	// Save first few values for comparison
	savedValues := make([]float64, min(10, len(output1)))
	copy(savedValues, output1)

	// Second process call - this should NOT corrupt output1
	input2 := make([]float64, 1000)
	for i := range input2 {
		input2[i] = math.Cos(2.0 * math.Pi * float64(i) / 50)
	}
	_, err = stage.Process(input2)
	require.NoError(t, err, "Second Process() failed")

	// Verify output1 was not corrupted
	for i, expected := range savedValues {
		assert.InDelta(t, expected, output1[i], 1e-15,
			"output1[%d] was corrupted after second Process() call: expected %f, got %f",
			i, expected, output1[i])
	}

	t.Log("Polyphase stage buffer integrity verified: output survives subsequent Process() calls")
}

// TestPolyphaseStage_FlushDoesNotCorruptOutput verifies that calling Flush()
// after Process() does not corrupt the previously returned output.
func TestPolyphaseStage_FlushDoesNotCorruptOutput(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err, "Failed to create polyphase stage")

	// Generate test signal
	input := make([]float64, 2000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	// Process the signal
	output, err := stage.Process(input)
	require.NoError(t, err, "Process() failed")
	require.NotEmpty(t, output, "Output should not be empty")

	// Save all values for comparison
	savedOutput := make([]float64, len(output))
	copy(savedOutput, output)

	// Flush the stage - this should NOT corrupt the previous output
	flush, err := stage.Flush()
	require.NoError(t, err, "Flush() failed")

	// Verify output was not corrupted
	for i, expected := range savedOutput {
		assert.InDelta(t, expected, output[i], 1e-15,
			"output[%d] was corrupted after Flush(): expected %f, got %f",
			i, expected, output[i])
	}

	// Also verify flush output is valid
	for i, v := range flush {
		assert.False(t, math.IsNaN(v), "flush[%d] is NaN", i)
		assert.False(t, math.IsInf(v, 0), "flush[%d] is Inf", i)
	}

	t.Log("Polyphase stage: output survives Flush() call")
}

// TestResampler_BufferIntegrity verifies buffer integrity for full resampler.
func TestResampler_BufferIntegrity(t *testing.T) {
	testCases := []struct {
		name       string
		inputRate  float64
		outputRate float64
	}{
		{"44100_to_48000", 44100, 48000},
		{"48000_to_44100", 48000, 44100},
		{"44100_to_96000", 44100, 96000},
		{"96000_to_48000", 96000, 48000},
		{"44100_to_88200", 44100, 88200}, // 2x integer ratio
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err, "Failed to create resampler")

			// Generate test signal
			input := make([]float64, 4000)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * 1000 * float64(i) / tc.inputRate)
			}

			// First process call
			output1, err := resampler.Process(input)
			require.NoError(t, err, "First Process() failed")

			// Save values for comparison
			savedOutput := make([]float64, len(output1))
			copy(savedOutput, output1)

			// Second process call
			input2 := make([]float64, 2000)
			for i := range input2 {
				input2[i] = math.Cos(2.0 * math.Pi * 500 * float64(i) / tc.inputRate)
			}
			_, err = resampler.Process(input2)
			require.NoError(t, err, "Second Process() failed")

			// Verify output1 was not corrupted
			for i, expected := range savedOutput {
				assert.InDelta(t, expected, output1[i], 1e-15,
					"output1[%d] was corrupted: expected %f, got %f", i, expected, output1[i])
			}

			t.Logf("%s: buffer integrity verified", tc.name)
		})
	}
}

// TestResampler_ProcessAndFlushSequence tests typical usage pattern of
// Process() followed by Flush() to ensure output is not corrupted.
func TestResampler_ProcessAndFlushSequence(t *testing.T) {
	testCases := []struct {
		name       string
		inputRate  float64
		outputRate float64
	}{
		{"44100_to_48000", 44100, 48000},
		{"44100_to_96000", 44100, 96000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err, "Failed to create resampler")

			// Generate test signal
			input := make([]float64, 4000)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * 1000 * float64(i) / tc.inputRate)
			}

			// Process the signal
			output, err := resampler.Process(input)
			require.NoError(t, err, "Process() failed")

			// Save for comparison
			savedOutput := make([]float64, len(output))
			copy(savedOutput, output)

			// Flush - this should NOT corrupt output
			flush, err := resampler.Flush()
			require.NoError(t, err, "Flush() failed")

			// Verify output was not corrupted
			for i, expected := range savedOutput {
				assert.InDelta(t, expected, output[i], 1e-15,
					"output[%d] was corrupted after Flush()", i)
			}

			// Append flush and verify both are valid
			combined := make([]float64, len(output)+len(flush))
			copy(combined, output)
			copy(combined[len(output):], flush)
			for i, v := range combined {
				assert.False(t, math.IsNaN(v), "combined[%d] is NaN", i)
				assert.False(t, math.IsInf(v, 0), "combined[%d] is Inf", i)
			}

			t.Logf("%s: Process+Flush sequence verified, output=%d, flush=%d",
				tc.name, len(output), len(flush))
		})
	}
}

// TestDFTStage_MultipleProcessCalls tests that multiple consecutive Process()
// calls work correctly, don't corrupt earlier returned outputs, and are
// deterministic: an identical call sequence on a second fresh instance must
// reproduce the same outputs bit-exactly.
func TestDFTStage_MultipleProcessCalls(t *testing.T) {
	const numCalls = 10
	const samplesPerCall = 500

	// makeInput builds the same deterministic per-call input for both
	// instances below, so the two runs are directly comparable.
	makeInput := func(callIdx int) []float64 {
		input := make([]float64, samplesPerCall)
		for j := range input {
			// Different phase for each call to detect any cross-contamination
			input[j] = math.Sin(2.0*math.Pi*float64(j)/100 + float64(callIdx)*math.Pi/5)
		}
		return input
	}

	runSequence := func(stage *DFTStage[float64]) (raw, saved [][]float64) {
		raw = make([][]float64, numCalls)
		saved = make([][]float64, numCalls)
		for i := range numCalls {
			output, err := stage.Process(makeInput(i))
			require.NoError(t, err, "Process() call %d failed", i)
			raw[i] = output
			saved[i] = make([]float64, len(output))
			copy(saved[i], output)
		}
		return raw, saved
	}

	stageA, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err, "Failed to create DFT stage")
	rawA, savedA := runSequence(stageA)

	// Every earlier call's returned slice must still hold the values it held
	// right after that call: a later Process() reusing the same backing
	// array without properly copying out would silently corrupt it.
	for i := range numCalls {
		require.Len(t, rawA[i], len(savedA[i]), "call %d output length changed", i)
		for j := range rawA[i] {
			assert.Equal(t, savedA[i][j], rawA[i][j],
				"call %d output[%d] was corrupted by a later Process() call", i, j)
		}
	}

	// Determinism: an identical call sequence on a second, fresh instance
	// must produce bit-identical output per call.
	stageB, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err, "Failed to create second DFT stage")
	_, savedB := runSequence(stageB)

	for i := range numCalls {
		require.Len(t, savedB[i], len(savedA[i]), "call %d length differs between two fresh instances", i)
		for j := range savedA[i] {
			assert.Equal(t, savedA[i][j], savedB[i][j],
				"call %d output[%d] differs between two fresh instances given identical input", i, j)
		}
	}

	t.Logf("DFT stage: %d consecutive Process() calls verified bit-identical and uncorrupted", numCalls)
}

// TestPolyphaseStage_MultipleProcessCalls tests that multiple consecutive
// Process() calls work correctly, don't corrupt earlier returned outputs,
// and are deterministic across a second fresh instance.
func TestPolyphaseStage_MultipleProcessCalls(t *testing.T) {
	const numCalls = 10
	const samplesPerCall = 1000

	makeInput := func(callIdx int) []float64 {
		input := make([]float64, samplesPerCall)
		for j := range input {
			input[j] = math.Sin(2.0*math.Pi*float64(j)/100 + float64(callIdx)*math.Pi/5)
		}
		return input
	}

	runSequence := func(stage *PolyphaseStage[float64]) (raw, saved [][]float64) {
		raw = make([][]float64, numCalls)
		saved = make([][]float64, numCalls)
		for i := range numCalls {
			output, err := stage.Process(makeInput(i))
			require.NoError(t, err, "Process() call %d failed", i)
			raw[i] = output
			saved[i] = make([]float64, len(output))
			copy(saved[i], output)
		}
		return raw, saved
	}

	stageA, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err, "Failed to create polyphase stage")
	rawA, savedA := runSequence(stageA)

	for i := range numCalls {
		require.Len(t, rawA[i], len(savedA[i]), "call %d output length changed", i)
		for j := range rawA[i] {
			assert.Equal(t, savedA[i][j], rawA[i][j],
				"call %d output[%d] was corrupted by a later Process() call", i, j)
		}
	}

	stageB, err := NewPolyphaseStage[float64](1.088435374, 0.459375, true, QualityHigh)
	require.NoError(t, err, "Failed to create second polyphase stage")
	_, savedB := runSequence(stageB)

	for i := range numCalls {
		require.Len(t, savedB[i], len(savedA[i]), "call %d length differs between two fresh instances", i)
		for j := range savedA[i] {
			assert.Equal(t, savedA[i][j], savedB[i][j],
				"call %d output[%d] differs between two fresh instances given identical input", i, j)
		}
	}

	t.Logf("Polyphase stage: %d consecutive Process() calls verified bit-identical and uncorrupted", numCalls)
}

// TestCubicStage_BufferIntegrity verifies CubicStage doesn't have buffer issues.
func TestCubicStage_BufferIntegrity(t *testing.T) {
	stage := NewCubicStage[float64](2.0)

	// Generate test signal
	input := make([]float64, 1000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	// First process call
	output1, err := stage.Process(input)
	require.NoError(t, err, "First Process() failed")

	// Save values
	savedOutput := make([]float64, len(output1))
	copy(savedOutput, output1)

	// Second process call
	input2 := make([]float64, 500)
	for i := range input2 {
		input2[i] = math.Cos(2.0 * math.Pi * float64(i) / 50)
	}
	_, err = stage.Process(input2)
	require.NoError(t, err, "Second Process() failed")

	// Verify output1 was not corrupted
	for i, expected := range savedOutput {
		assert.InDelta(t, expected, output1[i], 1e-15,
			"output1[%d] was corrupted", i)
	}

	t.Log("Cubic stage buffer integrity verified")
}

// TestLinearStage_BufferIntegrity verifies LinearStage doesn't have buffer issues.
func TestLinearStage_BufferIntegrity(t *testing.T) {
	stage := NewLinearStage(2.0)

	// Generate test signal
	input := make([]float64, 1000)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}

	// First process call
	output1, err := stage.Process(input)
	require.NoError(t, err, "First Process() failed")

	// Save values
	savedOutput := make([]float64, len(output1))
	copy(savedOutput, output1)

	// Second process call
	input2 := make([]float64, 500)
	for i := range input2 {
		input2[i] = math.Cos(2.0 * math.Pi * float64(i) / 50)
	}
	_, err = stage.Process(input2)
	require.NoError(t, err, "Second Process() failed")

	// Verify output1 was not corrupted
	for i, expected := range savedOutput {
		assert.InDelta(t, expected, output1[i], 1e-15,
			"output1[%d] was corrupted", i)
	}

	t.Log("Linear stage buffer integrity verified")
}
