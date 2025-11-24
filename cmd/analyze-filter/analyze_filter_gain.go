package main

import (
	"fmt"

	"github.com/tphakala/go-audio-resampler/internal/filter"
)

const (
	// Filter design parameters (matching soxr defaults)
	defaultNumPhases    = 80    // Number of polyphase filter phases
	defaultCutoff       = 0.45  // Cutoff frequency for CD→DAT
	defaultTransitionBW = 0.05  // Transition bandwidth
	defaultAttenuation  = 100.0 // Stopband attenuation in dB

	// Phase calculation constants
	phaseShiftBits = 8  // log2(256) = 8 bits for phase indexing
	fracBits       = 32 // 32-bit fractional precision

	// Display limits
	maxPhasesToShow = 5    // Maximum phases to display in detail
	testIterations  = 1000 // Number of iterations for phase usage test
)

func main() {
	// Test 2x upsampling case
	fmt.Println("=== Analyzing Filter DC Gain ===")

	params := filter.PolyphaseParams{
		NumPhases:    defaultNumPhases,
		Cutoff:       defaultCutoff,
		TransitionBW: defaultTransitionBW,
		Attenuation:  defaultAttenuation,
		InterpOrder:  filter.InterpLinear,
		Gain:         1.0,
	}

	pfb, err := filter.DesignPolyphaseFilterBank(params)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Filter bank info:\n")
	fmt.Printf("  NumPhases: %d\n", pfb.NumPhases)
	fmt.Printf("  TapsPerPhase: %d\n", pfb.TapsPerPhase)
	fmt.Printf("  TotalTaps: %d\n", pfb.TotalTaps)
	fmt.Printf("  InterpOrder: %d\n\n", pfb.InterpOrder)

	// Calculate DC gain of each phase
	fmt.Println("DC gain per phase:")
	var totalDC float64
	phasesToShow := []int{0, 1, 2, 3, 4, 5, 6, 7, 31, 32, 33}
	phaseGains := make(map[int]float64)

	for phase := 0; phase < pfb.NumPhases; phase++ {
		var phaseDC float64
		for tap := 0; tap < pfb.TapsPerPhase; tap++ {
			coef := pfb.GetCoefficient(tap, phase, 0.0)
			phaseDC += coef
		}
		phaseGains[phase] = phaseDC
		totalDC += phaseDC
	}

	for _, phase := range phasesToShow {
		if phase < pfb.NumPhases {
			fmt.Printf("  Phase %2d: %.10f\n", phase, phaseGains[phase])
		}
	}
	if pfb.NumPhases > len(phasesToShow) {
		fmt.Printf("  ... (%d more phases)\n", pfb.NumPhases-len(phasesToShow))
	}

	fmt.Printf("\nTotal DC gain (sum of all phases): %.10f\n", totalDC)
	fmt.Printf("Average DC gain per phase: %.10f\n", totalDC/float64(pfb.NumPhases))

	// Test multiple ratios
	testRatios := []struct {
		ratio float64
		name  string
	}{
		{2.0, "2x upsampling"},
		{0.5, "2x downsampling"},
		{48000.0 / 44100.0, "CD→DAT"},
		{44100.0 / 48000.0, "DAT→CD"},
		{1.5, "3:2 upsampling"},
	}

	phaseShift := uint(phaseShiftBits)
	fracScale := float64(uint64(1) << fracBits)

	for _, test := range testRatios {
		fmt.Printf("\n=== %s (ratio = %.6f) ===\n", test.name, test.ratio)

		// Calculate phase step
		phaseStep := uint32(fracScale / test.ratio)
		fmt.Printf("  Phase step: 0x%08x\n", phaseStep)

		// Simulate which phases are used
		phaseFrac := uint32(0)
		usedPhases := make(map[int]bool)
		var sumUsedPhaseDC float64

		// Test enough outputs to find the pattern
		for range testIterations {
			phaseIndex := int(phaseFrac >> (fracBits - phaseShift))
			if phaseIndex >= pfb.NumPhases {
				phaseIndex = pfb.NumPhases - 1
			}
			if !usedPhases[phaseIndex] {
				usedPhases[phaseIndex] = true
				sumUsedPhaseDC += phaseGains[phaseIndex]
				if len(usedPhases) <= maxPhasesToShow {
					fmt.Printf("    Phase %d: DC gain = %.10f\n",
						phaseIndex, phaseGains[phaseIndex])
				}
			}
			phaseFrac += phaseStep
		}

		numUsedPhases := len(usedPhases)
		avgUsedPhaseDC := sumUsedPhaseDC / float64(numUsedPhases)
		fmt.Printf("  Used %d unique phases (out of %d)\n", numUsedPhases, pfb.NumPhases)
		fmt.Printf("  Average DC gain of used phases: %.10f\n", avgUsedPhaseDC)
		fmt.Printf("  Correct outputGain: 1.0 / %.10f = %.10f\n",
			avgUsedPhaseDC, 1.0/avgUsedPhaseDC)
	}
}
