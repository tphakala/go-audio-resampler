package engine

import (
	"fmt"
	"testing"
)

func TestDebug96kTo48kDC(t *testing.T) {
	// Get soxr reference
	soxrOut, err := getSoxrReference(96000, 48000, "dc", 0)
	if err != nil {
		t.Skip(err)
	}

	// Run our resampler
	resampler, err := NewResampler[float64](96000, 48000, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}

	input := generateTestSignal("dc", 96000, 0)
	output, err := resampler.Process(input)
	if err != nil {
		t.Fatal(err)
	}
	flush, _ := resampler.Flush()
	output = append(output, flush...)

	fmt.Printf("\n=== 96000→48000 DC Comparison ===\n")
	fmt.Printf("soxr samples: %d, our samples: %d\n\n", len(soxrOut), len(output))

	// Check soxr output values
	fmt.Printf("soxr first 10 values:\n")
	for i := 0; i < 10 && i < len(soxrOut); i++ {
		fmt.Printf("  [%d] %.10f\n", i, soxrOut[i])
	}

	fmt.Printf("\nOur first 10 values:\n")
	for i := 0; i < 10 && i < len(output); i++ {
		fmt.Printf("  [%d] %.10f\n", i, output[i])
	}

	// Check middle values
	mid := len(output) / 2
	fmt.Printf("\nOur middle 10 values (around index %d):\n", mid)
	for i := mid - 5; i < mid+5 && i < len(output); i++ {
		fmt.Printf("  [%d] %.10f\n", i, output[i])
	}

	// Check for any values significantly different from 1.0
	fmt.Printf("\nValues deviating >0.01 from 1.0:\n")
	count := 0
	for i, v := range output {
		if v < 0.99 || v > 1.01 {
			if count < 20 {
				fmt.Printf("  [%d] %.10f\n", i, v)
			}
			count++
		}
	}
	fmt.Printf("Total deviating values: %d / %d\n", count, len(output))
}
