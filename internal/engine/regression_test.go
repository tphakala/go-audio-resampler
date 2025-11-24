package engine

import (
	"math"
	"testing"
)

// TestRegressionDCGain verifies DC signals pass through with gain ≈ 1.0
func TestRegressionDCGain(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
	}{
		{44100, 48000},
		{48000, 44100},
		{44100, 96000},
		{96000, 48000},
	}

	for _, tc := range testCases {
		resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
		if err != nil {
			t.Fatalf("%.0f→%.0f: NewResampler failed: %v", tc.inputRate, tc.outputRate, err)
		}

		// Generate DC signal
		numSamples := 10000
		input := make([]float64, numSamples)
		for i := range input {
			input[i] = 1.0
		}

		output, _ := resampler.Process(input)
		flush, _ := resampler.Flush()
		output = append(output, flush...)

		// Skip initial latency, check stable region
		startIdx := len(output) / 4
		endIdx := 3 * len(output) / 4

		// Calculate mean DC level
		sum := 0.0
		count := 0
		for i := startIdx; i < endIdx; i++ {
			sum += output[i]
			count++
		}
		dcGain := sum / float64(count)

		// DC gain should be very close to 1.0
		if math.Abs(dcGain-1.0) > 0.001 {
			t.Errorf("%.0f→%.0f: DC gain = %.6f, want ~1.0", tc.inputRate, tc.outputRate, dcGain)
		} else {
			t.Logf("%.0f→%.0f: DC gain = %.6f ✓", tc.inputRate, tc.outputRate, dcGain)
		}
	}
}

// TestRegressionSineAmplitude verifies sine waves preserve amplitude
func TestRegressionSineAmplitude(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		frequency  float64
	}{
		{44100, 48000, 1000},
		{44100, 48000, 5000},
		{48000, 44100, 1000},
		{44100, 96000, 1000},
	}

	for _, tc := range testCases {
		resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
		if err != nil {
			t.Fatalf("%.0f→%.0f @%.0fHz: NewResampler failed: %v",
				tc.inputRate, tc.outputRate, tc.frequency, err)
		}

		// Generate sine wave
		numSamples := 10000
		input := make([]float64, numSamples)
		for i := range input {
			phase := 2.0 * math.Pi * tc.frequency * float64(i) / tc.inputRate
			input[i] = math.Sin(phase)
		}

		output, _ := resampler.Process(input)
		flush, _ := resampler.Flush()
		output = append(output, flush...)

		// Skip initial latency, find max amplitude in stable region
		startIdx := len(output) / 4
		endIdx := 3 * len(output) / 4

		maxVal := 0.0
		for i := startIdx; i < endIdx; i++ {
			if math.Abs(output[i]) > maxVal {
				maxVal = math.Abs(output[i])
			}
		}

		// Amplitude should be close to 1.0 (input amplitude)
		// Allow 5% deviation for filter passband ripple
		if math.Abs(maxVal-1.0) > 0.05 {
			t.Errorf("%.0f→%.0f @%.0fHz: amplitude = %.3f, want ~1.0",
				tc.inputRate, tc.outputRate, tc.frequency, maxVal)
		} else {
			t.Logf("%.0f→%.0f @%.0fHz: amplitude = %.3f ✓",
				tc.inputRate, tc.outputRate, tc.frequency, maxVal)
		}
	}
}

// TestRegressionNoClipping verifies no clipping/overflow occurs
func TestRegressionNoClipping(t *testing.T) {
	resampler, _ := NewResampler[float64](44100, 48000, QualityHigh)

	// Generate maximum amplitude signal
	numSamples := 10000
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * 1000 * float64(i) / 44100
		input[i] = math.Sin(phase)
	}

	output, _ := resampler.Process(input)

	// Check no values exceed ±1.1 (allow small overshoot from filter ringing)
	for i, v := range output {
		if math.Abs(v) > 1.1 {
			t.Errorf("Clipping at index %d: value = %.6f", i, v)
			return
		}
	}
	t.Log("No clipping detected ✓")
}

// TestRegressionZeroInput verifies zero input produces zero output
func TestRegressionZeroInput(t *testing.T) {
	resampler, _ := NewResampler[float64](44100, 48000, QualityHigh)

	input := make([]float64, 10000)
	output, _ := resampler.Process(input)

	// All output should be zero (or very close)
	maxVal := 0.0
	for _, v := range output {
		if math.Abs(v) > maxVal {
			maxVal = math.Abs(v)
		}
	}

	if maxVal > 1e-10 {
		t.Errorf("Zero input produced non-zero output: max = %.2e", maxVal)
	} else {
		t.Log("Zero input → zero output ✓")
	}
}
