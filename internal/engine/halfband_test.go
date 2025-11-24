package engine

import (
	"fmt"
	"testing"
)

func TestAnalyzeHalfbandStructure(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("\n=== Half-band Filter Analysis ===")
	fmt.Printf("Factor: %d, TapsPerPhase: %d\n", stage.factor, stage.tapsPerPhase)

	// For a half-band filter in 2x upsampling:
	// - Phase 0: center tap = 0.5, others = 0 (ideally)
	// - Phase 1: the actual lowpass filter

	// Count near-zero coefficients
	const threshold = 1e-10
	for phase := 0; phase < stage.factor; phase++ {
		coeffs := stage.polyCoeffs[phase]
		zeroCount := 0
		nonZeroSum := 0.0
		for _, c := range coeffs {
			if c < threshold && c > -threshold {
				zeroCount++
			} else {
				nonZeroSum += c
			}
		}
		fmt.Printf("Phase %d: %d/%d near-zero coeffs (%.1f%%), sum=%.4f\n",
			phase, zeroCount, len(coeffs),
			float64(zeroCount)/float64(len(coeffs))*100, nonZeroSum)
	}

	// Check if this is a true half-band structure
	// In a perfect half-band, every other tap (except center) would be zero
	fmt.Println("\nSample coefficients (first 10):")
	for phase := 0; phase < stage.factor; phase++ {
		fmt.Printf("Phase %d: ", phase)
		for i := 0; i < 10 && i < len(stage.polyCoeffs[phase]); i++ {
			fmt.Printf("%.4f ", stage.polyCoeffs[phase][i])
		}
		fmt.Println("...")
	}
}
