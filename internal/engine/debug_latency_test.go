package engine

import (
	"fmt"
	"testing"
)

func TestDebugLatency(t *testing.T) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}

	// Check stage parameters
	fmt.Printf("\n=== Resampler Configuration ===\n")
	fmt.Printf("Ratio: %.6f\n", resampler.ratio)

	if resampler.preStage != nil {
		fmt.Printf("\nDFT Pre-Stage:\n")
		fmt.Printf("  Factor: %d\n", resampler.preStage.factor)
		fmt.Printf("  Taps per phase: %d\n", resampler.preStage.tapsPerPhase)
		fmt.Printf("  Filter latency: %d samples (input domain)\n", resampler.preStage.tapsPerPhase/2)
	}

	if resampler.polyphaseStage != nil {
		fmt.Printf("\nPolyphase Stage:\n")
		fmt.Printf("  Num phases: %d\n", resampler.polyphaseStage.numPhases)
		fmt.Printf("  Taps per phase: %d\n", resampler.polyphaseStage.tapsPerPhase)
		fmt.Printf("  Step: %d\n", resampler.polyphaseStage.step)
		fmt.Printf("  Filter latency: %d samples (intermediate domain)\n", resampler.polyphaseStage.tapsPerPhase/2)
	}

	// Calculate total latency
	// DFT stage latency (in input samples) + polyphase latency (in intermediate samples, converted to output)
	dftLatency := 0
	if resampler.preStage != nil && resampler.preStage.factor > 1 {
		dftLatency = resampler.preStage.tapsPerPhase / 2
	}

	polyLatency := 0
	if resampler.polyphaseStage != nil {
		// Polyphase latency in intermediate samples
		polyLatencyIntermediate := resampler.polyphaseStage.tapsPerPhase / 2
		// Convert to output samples (approximately)
		polyLatency = int(int64(polyLatencyIntermediate) * int64(resampler.polyphaseStage.numPhases) / resampler.polyphaseStage.step)
	}

	fmt.Printf("\nEstimated total latency: %d + %d ≈ %d output samples\n",
		dftLatency*2, polyLatency, dftLatency*2+polyLatency)

	// Test with small input to see when output appears
	fmt.Printf("\n=== Testing Output Timing ===\n")
	input := make([]float64, 100)
	for i := range input {
		input[i] = 1.0 // DC signal
	}

	output, _ := resampler.Process(input)

	// Count leading zeros
	leadingZeros := 0
	for _, v := range output {
		if v == 0 {
			leadingZeros++
		} else {
			break
		}
	}

	fmt.Printf("Input: 100 samples\n")
	fmt.Printf("Output: %d samples\n", len(output))
	fmt.Printf("Leading zeros: %d\n", leadingZeros)

	// Show first non-zero values
	fmt.Printf("\nFirst 10 output values:\n")
	for i := 0; i < 10 && i < len(output); i++ {
		fmt.Printf("  [%d] %.10f\n", i, output[i])
	}
}
