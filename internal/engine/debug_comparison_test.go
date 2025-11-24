package engine

import (
	"fmt"
	"testing"
)

func TestDebugSoxrComparison(t *testing.T) {
	inputRate := 44100.0
	outputRate := 48000.0
	frequency := 1000.0

	// Get soxr reference
	soxrOut, err := getSoxrReference(inputRate, outputRate, "sine", frequency)
	if err != nil {
		t.Skipf("soxr reference not available: %v", err)
	}

	// Run our resampler
	resampler, err := NewResampler[float64](inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("NewResampler failed: %v", err)
	}

	input := generateTestSignal("sine", inputRate, frequency)
	output, err := resampler.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}
	flush, _ := resampler.Flush()
	output = append(output, flush...)

	fmt.Printf("\n=== Sample Comparison ===\n")
	fmt.Printf("soxr samples: %d, our samples: %d\n\n", len(soxrOut), len(output))

	fmt.Printf("First 20 samples:\n")
	fmt.Printf("%5s  %15s  %15s  %10s\n", "idx", "soxr", "ours", "diff")
	for i := 0; i < 20 && i < len(soxrOut) && i < len(output); i++ {
		diff := output[i] - soxrOut[i]
		fmt.Printf("%5d  %15.10f  %15.10f  %10.6f\n", i, soxrOut[i], output[i], diff)
	}

	// Find where soxr output is near peak (1.0)
	peakIdx := -1
	for i, v := range soxrOut {
		if v > 0.99 {
			peakIdx = i
			break
		}
	}

	// Find where our output is near peak
	ourPeakIdx := -1
	for i, v := range output {
		if v > 0.99 {
			ourPeakIdx = i
			break
		}
	}

	fmt.Printf("\nFirst peak locations:\n")
	fmt.Printf("  soxr peak (>0.99): index %d\n", peakIdx)
	fmt.Printf("  our peak (>0.99):  index %d\n", ourPeakIdx)
	if peakIdx >= 0 && ourPeakIdx >= 0 {
		fmt.Printf("  phase offset: %d samples\n", ourPeakIdx-peakIdx)
	}

	// Try to find best alignment
	fmt.Printf("\n=== Finding Best Alignment ===\n")
	bestOffset := 0
	bestCorr := -2.0

	for offset := -500; offset <= 500; offset++ {
		corr := computeCorrelationWithSkip(output, soxrOut, offset, 100)
		if corr > bestCorr {
			bestCorr = corr
			bestOffset = offset
		}
	}

	fmt.Printf("Best offset: %d samples (our output shifted by %d relative to soxr)\n", bestOffset, bestOffset)
	fmt.Printf("Correlation at best offset: %.6f\n", bestCorr)

	// Show aligned comparison
	if bestOffset != 0 {
		fmt.Printf("\nAligned comparison (offset=%d):\n", bestOffset)
		fmt.Printf("%5s  %15s  %15s  %10s\n", "idx", "soxr", "ours", "diff")
		for i := 100; i < 120; i++ {
			ourIdx := i + bestOffset
			if ourIdx >= 0 && ourIdx < len(output) && i < len(soxrOut) {
				diff := output[ourIdx] - soxrOut[i]
				fmt.Printf("%5d  %15.10f  %15.10f  %10.6f\n", i, soxrOut[i], output[ourIdx], diff)
			}
		}
	}
}
