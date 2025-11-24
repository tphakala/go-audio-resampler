package engine

import (
	"fmt"
	"testing"
)

func TestAnalyzePolyphase(t *testing.T) {
	rates := []struct{ in, out float64 }{
		{44100, 48000}, // CD to DAT
		{48000, 44100}, // DAT to CD
		{44100, 96000}, // 2x upsample
		{96000, 44100}, // downsample
	}

	fmt.Println("\n=== PolyphaseStage Analysis ===")
	for _, r := range rates {
		resampler, _ := NewResampler[float64](r.in, r.out, QualityHigh)

		fmt.Printf("\n%.0f → %.0f Hz:\n", r.in, r.out)
		fmt.Printf("  numPhases: %d\n", resampler.polyphaseStage.numPhases)
		fmt.Printf("  tapsPerPhase: %d\n", resampler.polyphaseStage.tapsPerPhase)
		fmt.Printf("  step: %d\n", resampler.polyphaseStage.step)

		// Calculate iterations per 1 second of audio
		inputSamples := int(r.in)
		numPhases := resampler.polyphaseStage.numPhases
		step := resampler.polyphaseStage.step
		taps := resampler.polyphaseStage.tapsPerPhase

		// After DFT 2x upsample, we have 2x samples
		dftOutput := inputSamples * 2

		// Polyphase iterations = (dftOutput * numPhases) / step approximately
		polyphaseIterations := (int64(dftOutput) * int64(numPhases)) / step

		fmt.Printf("  DFT output samples: %d\n", dftOutput)
		fmt.Printf("  Polyphase iterations (output samples): ~%d\n", polyphaseIterations)
		fmt.Printf("  Total multiply-adds: %d × %d taps = %dK\n",
			polyphaseIterations, taps, polyphaseIterations*int64(taps)/1000)

		// Power of 2?
		isPow2 := numPhases > 0 && (numPhases&(numPhases-1)) == 0
		fmt.Printf("  numPhases isPowerOf2: %v\n", isPow2)
	}
}
