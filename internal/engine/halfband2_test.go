package engine

import (
	"fmt"
	"math"
	"testing"
)

func TestFindNonZeroCoeffs(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println("\n=== Phase 0 Non-Zero Coefficients ===")
	const threshold = 1e-8
	for i, c := range stage.polyCoeffs[0] {
		if math.Abs(c) > threshold {
			fmt.Printf("  Index %d: %.10f\n", i, c)
		}
	}

	// Calculate what Phase 0 actually does
	fmt.Println("\n=== Phase 0 Behavior Analysis ===")
	// The center tap should be close to 1.0 (or factor=2.0 before scaling)
	centerIdx := len(stage.polyCoeffs[0]) / 2
	fmt.Printf("Center index: %d\n", centerIdx)
	fmt.Printf("Center coefficient: %.10f\n", stage.polyCoeffs[0][centerIdx])

	// Sum all Phase 0 coeffs
	sum0 := 0.0
	for _, c := range stage.polyCoeffs[0] {
		sum0 += c
	}
	fmt.Printf("Sum of Phase 0 coeffs: %.10f\n", sum0)

	// Check if Phase 0 is a delayed delta (pass-through)
	maxCoeff := 0.0
	maxIdx := 0
	for i, c := range stage.polyCoeffs[0] {
		if math.Abs(c) > maxCoeff {
			maxCoeff = math.Abs(c)
			maxIdx = i
		}
	}
	fmt.Printf("Max coefficient: index=%d, value=%.10f\n", maxIdx, stage.polyCoeffs[0][maxIdx])

	// Verify: is Phase 0 essentially input[i-delay] * scale?
	// If so, we can skip the convolution and just copy!
	if maxCoeff > 0.99 {
		fmt.Println("\n*** Phase 0 IS a pass-through filter! ***")
		fmt.Println("Optimization: Skip convolution, use direct copy with offset")
	}
}
