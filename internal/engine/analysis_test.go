package engine

import (
	"fmt"
	"testing"
)

func TestAnalyzeFilterSizes(t *testing.T) {
	qualities := []struct {
		name    string
		quality Quality
	}{
		{"Low", QualityLow},
		{"Medium", QualityMedium},
		{"High", QualityHigh},
	}

	fmt.Println("\n=== Filter Size Analysis ===")
	fmt.Println("\nDFT Stage (2x upsample):")
	for _, q := range qualities {
		stage, err := NewDFTStage[float64](2, q.quality)
		if err != nil {
			t.Fatal(err)
		}
		totalTaps := stage.tapsPerPhase * stage.factor
		fmt.Printf("  %s: %d taps/phase × %d phases = %d total taps\n",
			q.name, stage.tapsPerPhase, stage.factor, totalTaps)
	}

	fmt.Println("\nPolyphase Stage (88200 -> 48000):")
	ratio := 48000.0 / 88200.0
	totalIORatio := 44100.0 / 48000.0 // For a 44100 → 48000 conversion
	for _, q := range qualities {
		stage, err := NewPolyphaseStage[float64](ratio, totalIORatio, true, q.quality)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("  %s: %d taps/phase × %d phases\n",
			q.name, stage.tapsPerPhase, stage.numPhases)
	}

	// Calculate multiply-adds per second of audio
	fmt.Println("\n=== Compute Intensity (per 1 sec audio @ 44100 Hz) ===")
	for _, q := range qualities {
		dftStage, _ := NewDFTStage[float64](2, q.quality)
		polyStage, _ := NewPolyphaseStage[float64](ratio, totalIORatio, true, q.quality)

		// DFT: 44100 input samples × tapsPerPhase × factor (phases)
		dftOps := 44100 * dftStage.tapsPerPhase * dftStage.factor

		// Polyphase: ~48000 output samples × tapsPerPhase
		polyOps := 48000 * polyStage.tapsPerPhase

		totalOps := dftOps + polyOps
		fmt.Printf("  %s: DFT=%dM + Poly=%dK = %dM multiply-adds\n",
			q.name, dftOps/1000000, polyOps/1000, totalOps/1000000)
	}
}
