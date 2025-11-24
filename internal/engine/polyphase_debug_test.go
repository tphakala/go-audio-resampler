package engine

import (
	"math"
	"testing"
)

// TestPolyphaseFilterResponse checks the actual frequency response of the polyphase filter
func TestPolyphaseFilterResponse(t *testing.T) {
	// Create polyphase stage for 88200 -> 96000 (upsampling ratio = 1.0884)
	ratio := 96000.0 / 88200.0
	totalIORatio := 44100.0 / 96000.0

	stage, err := NewPolyphaseStage[float64](ratio, totalIORatio, true, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create polyphase stage: %v", err)
	}

	t.Logf("Polyphase stage: numPhases=%d, tapsPerPhase=%d, step=%d",
		stage.numPhases, stage.tapsPerPhase, stage.step)

	// Compute frequency response of phase 0 (as a simple check)
	phase0Coeffs := stage.polyCoeffs[0]

	// Calculate DC gain of phase 0
	var dcGain float64
	for _, c := range phase0Coeffs {
		dcGain += float64(c)
	}
	t.Logf("Phase 0 DC gain: %.6f (should be ~1.0)", dcGain)

	// Calculate total DC gain (sum of all phases)
	var totalDC float64
	for phase := 0; phase < stage.numPhases; phase++ {
		for _, c := range stage.polyCoeffs[phase] {
			totalDC += float64(c)
		}
	}
	t.Logf("Total DC gain: %.6f (should be ~%d)", totalDC, stage.numPhases)

	// Reconstruct prototype filter from polyphase branches
	// Prototype layout: coeffs[tap * numPhases + phase]
	prototypeLen := stage.tapsPerPhase * stage.numPhases
	prototype := make([]float64, prototypeLen)
	for phase := 0; phase < stage.numPhases; phase++ {
		for tap := 0; tap < stage.tapsPerPhase; tap++ {
			prototype[tap*stage.numPhases+phase] = float64(stage.polyCoeffs[phase][tap])
		}
	}

	// Compute frequency response of PROTOTYPE filter at key frequencies
	// The prototype should show lowpass behavior
	t.Logf("\nPrototype filter (len=%d) frequency response:", prototypeLen)
	testFreqs := []float64{0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}
	for _, freq := range testFreqs {
		omega := 2 * math.Pi * freq
		var realPart, imagPart float64
		for n, c := range prototype {
			angle := omega * float64(n)
			realPart += c * math.Cos(angle)
			imagPart -= c * math.Sin(angle)
		}
		mag := math.Sqrt(realPart*realPart + imagPart*imagPart)
		magDB := 20 * math.Log10(max(mag/float64(stage.numPhases), 1e-20))
		t.Logf("  freq=%.4f: %.2f dB (cutoff ~0.0027)", freq, magDB)
	}
}
