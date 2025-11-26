package resampler

import (
	"math"
	"testing"
)

// TestNewEngineFloat32 verifies that NewEngineFloat32 creates a working resampler.
func TestNewEngineFloat32(t *testing.T) {
	tests := []struct {
		name       string
		inputRate  float64
		outputRate float64
		quality    QualityPreset
	}{
		{"CD_to_DAT_High", 44100, 48000, QualityHigh},
		{"DAT_to_CD_High", 48000, 44100, QualityHigh},
		{"2x_Upsample_Medium", 44100, 88200, QualityMedium},
		{"2x_Downsample_Low", 96000, 48000, QualityLow},
		{"VeryHigh_Quality", 44100, 48000, QualityVeryHigh},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := NewEngineFloat32(tt.inputRate, tt.outputRate, tt.quality)
			if err != nil {
				t.Fatalf("NewEngineFloat32 failed: %v", err)
			}

			// Verify ratio
			expectedRatio := tt.outputRate / tt.inputRate
			if math.Abs(r.GetRatio()-expectedRatio) > 1e-9 {
				t.Errorf("GetRatio = %v, want %v", r.GetRatio(), expectedRatio)
			}

			// Test Reset doesn't panic
			r.Reset()
		})
	}
}

// TestSimpleResamplerFloat32_Process verifies that Process returns float32.
func TestSimpleResamplerFloat32_Process(t *testing.T) {
	r, err := NewEngineFloat32(44100, 48000, QualityHigh)
	if err != nil {
		t.Fatalf("NewEngineFloat32 failed: %v", err)
	}

	// Generate a 1kHz sine wave
	const numSamples = 4410 // 0.1 seconds at 44.1kHz
	input := make([]float32, numSamples)
	for i := range input {
		input[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / 44100))
	}

	output, err := r.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	// Check output is not empty
	if len(output) == 0 {
		t.Error("Process returned empty output")
	}

	// Verify output length is approximately correct
	expectedLen := float64(numSamples) * (48000.0 / 44100.0)
	if math.Abs(float64(len(output))-expectedLen) > expectedLen*0.1 {
		t.Errorf("Output length %d is too far from expected %v", len(output), expectedLen)
	}
}

// TestSimpleResamplerFloat32_Flush verifies that Flush returns float32.
func TestSimpleResamplerFloat32_Flush(t *testing.T) {
	r, err := NewEngineFloat32(44100, 48000, QualityHigh)
	if err != nil {
		t.Fatalf("NewEngineFloat32 failed: %v", err)
	}

	// Process some samples first
	input := make([]float32, 1000)
	for i := range input {
		input[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / 44100))
	}
	_, err = r.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	// Flush should return float32
	flushed, err := r.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	// Flushed samples should exist (filter has latency)
	if len(flushed) == 0 {
		t.Log("Flush returned empty (may be valid depending on filter design)")
	}
}

// TestResampleMonoFloat32 verifies one-shot mono resampling.
func TestResampleMonoFloat32(t *testing.T) {
	// Generate a 1kHz sine wave
	const inputRate = 44100
	const outputRate = 48000
	const numSamples = 4410 // 0.1 seconds

	input := make([]float32, numSamples)
	for i := range input {
		input[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / inputRate))
	}

	output, err := ResampleMonoFloat32(input, inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("ResampleMonoFloat32 failed: %v", err)
	}

	// Verify output length is approximately correct
	expectedLen := float64(numSamples) * (float64(outputRate) / float64(inputRate))
	if math.Abs(float64(len(output))-expectedLen) > expectedLen*0.1 {
		t.Errorf("Output length %d is too far from expected %v", len(output), expectedLen)
	}

	// Verify output is valid (not all zeros, not NaN)
	hasNonZero := false
	for _, v := range output {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("Output contains NaN or Inf")
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		t.Error("Output is all zeros")
	}
}

// TestResampleStereoFloat32 verifies one-shot stereo resampling.
func TestResampleStereoFloat32(t *testing.T) {
	const inputRate = 44100
	const outputRate = 48000
	const numSamples = 4410 // 0.1 seconds

	// Generate different frequencies for left and right
	left := make([]float32, numSamples)
	right := make([]float32, numSamples)
	for i := range left {
		left[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / inputRate))  // A4
		right[i] = float32(math.Sin(2 * math.Pi * 880 * float64(i) / inputRate)) // A5
	}

	leftOut, rightOut, err := ResampleStereoFloat32(left, right, inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("ResampleStereoFloat32 failed: %v", err)
	}

	// Both channels should have the same length
	if len(leftOut) != len(rightOut) {
		t.Errorf("Channel lengths differ: left=%d, right=%d", len(leftOut), len(rightOut))
	}

	// Verify output length is approximately correct
	expectedLen := float64(numSamples) * (float64(outputRate) / float64(inputRate))
	if math.Abs(float64(len(leftOut))-expectedLen) > expectedLen*0.1 {
		t.Errorf("Output length %d is too far from expected %v", len(leftOut), expectedLen)
	}
}

// TestInterleaveDeinterleaveFloat32 verifies interleave/deinterleave roundtrip.
func TestInterleaveDeinterleaveFloat32(t *testing.T) {
	const numSamples = 100

	left := make([]float32, numSamples)
	right := make([]float32, numSamples)
	for i := range left {
		left[i] = float32(i)
		right[i] = float32(i + 1000)
	}

	// Interleave
	interleaved := InterleaveToStereoFloat32(left, right)
	if len(interleaved) != numSamples*2 {
		t.Fatalf("Interleaved length = %d, want %d", len(interleaved), numSamples*2)
	}

	// Verify interleaved format
	for i := range numSamples {
		if interleaved[i*2] != left[i] {
			t.Errorf("interleaved[%d] = %v, want %v (left)", i*2, interleaved[i*2], left[i])
		}
		if interleaved[i*2+1] != right[i] {
			t.Errorf("interleaved[%d] = %v, want %v (right)", i*2+1, interleaved[i*2+1], right[i])
		}
	}

	// Deinterleave
	leftOut, rightOut := DeinterleaveFromStereoFloat32(interleaved)
	if len(leftOut) != numSamples || len(rightOut) != numSamples {
		t.Fatalf("Deinterleaved lengths: left=%d, right=%d, want %d", len(leftOut), len(rightOut), numSamples)
	}

	// Verify roundtrip
	for i := range numSamples {
		if leftOut[i] != left[i] {
			t.Errorf("leftOut[%d] = %v, want %v", i, leftOut[i], left[i])
		}
		if rightOut[i] != right[i] {
			t.Errorf("rightOut[%d] = %v, want %v", i, rightOut[i], right[i])
		}
	}
}

// TestFloat32_vs_Float64_Consistency verifies float32 and float64 produce similar results.
func TestFloat32_vs_Float64_Consistency(t *testing.T) {
	const inputRate = 44100
	const outputRate = 48000
	const numSamples = 4410

	// Generate input
	inputF64 := make([]float64, numSamples)
	inputF32 := make([]float32, numSamples)
	for i := range inputF64 {
		v := math.Sin(2 * math.Pi * 1000 * float64(i) / inputRate)
		inputF64[i] = v
		inputF32[i] = float32(v)
	}

	// Resample with both precisions
	outputF64, err := ResampleMono(inputF64, inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("ResampleMono failed: %v", err)
	}

	outputF32, err := ResampleMonoFloat32(inputF32, inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("ResampleMonoFloat32 failed: %v", err)
	}

	// Lengths should be the same
	if len(outputF64) != len(outputF32) {
		t.Fatalf("Output lengths differ: float64=%d, float32=%d", len(outputF64), len(outputF32))
	}

	// Results should be similar (within float32 precision)
	const tolerance = 1e-5 // float32 has ~7 decimal digits of precision
	maxDiff := 0.0
	for i := range outputF64 {
		diff := math.Abs(outputF64[i] - float64(outputF32[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("Max difference between float64 and float32 output: %e", maxDiff)
	if maxDiff > tolerance {
		t.Errorf("Float32 and float64 outputs differ too much: max diff = %e, tolerance = %e", maxDiff, tolerance)
	}
}

// TestSimpleResamplerFloat32_GetStatistics verifies statistics are tracked.
func TestSimpleResamplerFloat32_GetStatistics(t *testing.T) {
	r, err := NewEngineFloat32(44100, 48000, QualityHigh)
	if err != nil {
		t.Fatalf("NewEngineFloat32 failed: %v", err)
	}

	// Initial stats should be zero
	stats := r.GetStatistics()
	if stats["samplesIn"] != 0 || stats["samplesOut"] != 0 {
		t.Errorf("Initial stats not zero: %v", stats)
	}

	// Process some samples
	input := make([]float32, 1000)
	_, err = r.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	// Stats should be updated
	stats = r.GetStatistics()
	if stats["samplesIn"] == 0 {
		t.Error("samplesIn not updated after Process")
	}

	// Flush
	_, err = r.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	// Stats should be further updated
	finalStats := r.GetStatistics()
	if finalStats["samplesOut"] == 0 {
		t.Error("samplesOut is zero after Flush")
	}
}
