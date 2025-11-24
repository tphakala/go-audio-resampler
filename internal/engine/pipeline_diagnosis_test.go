package engine

import (
	"fmt"
	"math"
	"testing"
)

// TestPipelineDiagnosis analyzes pipeline construction for various ratios.
// NOTE: This is a diagnostic test for manual inspection - it logs results
// but does not assert on specific thresholds. Use TestAntiAliasing_Upsampling
// for regression testing with assertions.
func TestPipelineDiagnosis(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
	}{
		{48000, 96000}, // Exact 2× - integer ratio
		{44100, 88200}, // Exact 2× - integer ratio
		{44100, 96000}, // Non-integer: 96000/44100 ≈ 2.177
		{44100, 48000}, // Common CD→DAT: 48000/44100 ≈ 1.088
		{32000, 48000}, // 1.5× ratio
	}

	for _, tc := range testCases {
		ratio := tc.outputRate / tc.inputRate
		name := fmt.Sprintf("%.0f_to_%.0f_ratio_%.4f", tc.inputRate, tc.outputRate, ratio)

		t.Run(name, func(t *testing.T) {
			t.Logf("Input: %.0f Hz, Output: %.0f Hz, Ratio: %.6f", tc.inputRate, tc.outputRate, ratio)

			// Create resampler and test anti-aliasing
			result, err := measureAntiAliasing(tc.inputRate, tc.outputRate, SignalNoise, QualityHigh)
			if err != nil {
				t.Fatalf("Failed to measure anti-aliasing: %v", err)
			}

			t.Logf("Passband energy: %.2f dB", result.PassbandEnergy)
			t.Logf("Stopband energy: %.2f dB", result.StopbandEnergy)
			t.Logf("Stopband attenuation: %.2f dB", result.StopbandAttenuation)
		})
	}
}

// TestStageByStageFiltering tests each stage's filtering independently.
// NOTE: Diagnostic test for manual inspection of stage-by-stage behavior.
func TestStageByStageFiltering(t *testing.T) {
	// Test 44100 → 96000 (historically showed poor anti-imaging before polyphase fix)
	inputRate := 44100.0
	outputRate := 96000.0

	t.Logf("Testing %.0f → %.0f", inputRate, outputRate)

	// Generate input noise
	input := generateAntialiasTestSignal(SignalNoise, inputRate, antialiasTestSamples)

	// Compute PSD of input
	freqsIn, psdIn := computePSD(input, inputRate, fftWindowSize)

	// Find energy in 0-20kHz band (below Nyquist)
	inBandEnergy := measureBandEnergy(freqsIn, psdIn, 100, 20000)
	t.Logf("Input in-band energy (0-20kHz): %.2f dB", inBandEnergy)

	// Create resampler
	resampler, err := NewResampler[float64](inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	// Process
	output, err := resampler.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}
	flush, err := resampler.Flush()
	if err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Input samples: %d", len(input))
	t.Logf("Output samples: %d", len(output))

	// Compute PSD of output
	freqsOut, psdOut := computePSD(output, outputRate, fftWindowSize)

	// Measure energy in various bands
	origNyquist := inputRate / 2.0 // 22050 Hz
	newNyquist := outputRate / 2.0 // 48000 Hz

	// Below original Nyquist (should preserve energy)
	passbandEnergy := measureBandEnergy(freqsOut, psdOut, 100, origNyquist*0.9)

	// Transition band (around original Nyquist)
	transitionEnergy := measureBandEnergy(freqsOut, psdOut, origNyquist*0.9, origNyquist*1.1)

	// Stopband (above original Nyquist - images should be attenuated)
	stopbandEnergy := measureBandEnergy(freqsOut, psdOut, origNyquist*1.1, newNyquist*0.95)

	t.Logf("")
	t.Logf("=== Energy Analysis ===")
	t.Logf("Passband (0 - %.0f Hz):     %.2f dB", origNyquist*0.9, passbandEnergy)
	t.Logf("Transition (%.0f - %.0f Hz): %.2f dB", origNyquist*0.9, origNyquist*1.1, transitionEnergy)
	t.Logf("Stopband (%.0f - %.0f Hz):  %.2f dB", origNyquist*1.1, newNyquist*0.95, stopbandEnergy)
	t.Logf("")
	t.Logf("Stopband attenuation: %.2f dB", passbandEnergy-stopbandEnergy)

	// Print spectrum at key frequencies
	t.Logf("")
	t.Logf("=== Spectrum at key frequencies ===")
	keyFreqs := []float64{
		1000, 5000, 10000, 15000, 20000, 21000, // Passband
		22050, 23000, 24000, 25000, // Around original Nyquist
		30000, 35000, 40000, 45000, // Stopband
	}

	for _, target := range keyFreqs {
		if target >= newNyquist {
			continue
		}
		closestIdx := findClosestIndex(freqsOut, target)
		t.Logf("  %.0f Hz: %.2f dB", freqsOut[closestIdx], psdOut[closestIdx])
	}
}

func findClosestIndex(freqs []float64, target float64) int {
	closestIdx := 0
	closestDist := math.Abs(freqs[0] - target)
	for i, f := range freqs {
		dist := math.Abs(f - target)
		if dist < closestDist {
			closestDist = dist
			closestIdx = i
		}
	}
	return closestIdx
}

// TestIntermediateStages tests with intermediate upsampling
func TestIntermediateStages(t *testing.T) {
	// For 44100 → 96000, check if there's a 2× stage that should filter
	// Let's manually upsample in stages:
	// 44100 → 88200 (2×) → 96000 (fine adjustment ~1.088×)

	t.Run("Stage1_44100_to_88200", func(t *testing.T) {
		result, err := measureAntiAliasing(44100, 88200, SignalNoise, QualityHigh)
		if err != nil {
			t.Fatalf("measureAntiAliasing failed: %v", err)
		}
		t.Logf("44100→88200 (2×): Stopband attenuation: %.2f dB", result.StopbandAttenuation)
	})

	t.Run("Stage2_88200_to_96000", func(t *testing.T) {
		result, err := measureAntiAliasing(88200, 96000, SignalNoise, QualityHigh)
		if err != nil {
			t.Fatalf("measureAntiAliasing failed: %v", err)
		}
		// For this ratio (1.088×), the stopband is above 44.1kHz
		// We need to measure energy in the new frequency range
		t.Logf("88200→96000 (1.088×): Stopband attenuation: %.2f dB", result.StopbandAttenuation)
	})

	t.Run("Combined_44100_to_96000", func(t *testing.T) {
		result, err := measureAntiAliasing(44100, 96000, SignalNoise, QualityHigh)
		if err != nil {
			t.Fatalf("measureAntiAliasing failed: %v", err)
		}
		t.Logf("44100→96000 (combined): Stopband attenuation: %.2f dB", result.StopbandAttenuation)
	})
}

// TestHalfBandFilterResponse tests the half-band filter frequency response
func TestHalfBandFilterResponse(t *testing.T) {
	// Create a 2× upsampling resampler with half-band
	inputRate := 48000.0
	outputRate := 96000.0

	resampler, err := NewResampler[float64](inputRate, outputRate, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	// Generate an impulse to measure impulse response
	impulseSamples := 4096
	impulse := make([]float64, impulseSamples)
	impulse[impulseSamples/2] = 1.0 // Impulse at center

	output, err := resampler.Process(impulse)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}
	flush, err := resampler.Flush()
	if err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Input samples: %d", len(impulse))
	t.Logf("Output samples: %d", len(output))

	// Compute frequency response from impulse response
	freqs, psdDB := computePSD(output, outputRate, fftWindowSize)

	// Find attenuation at key frequencies
	origNyquist := inputRate / 2.0

	t.Logf("Frequency response from impulse:")
	keyFreqs := []float64{1000, 10000, 20000, 24000, 30000, 40000, 45000}
	for _, target := range keyFreqs {
		idx := findClosestIndex(freqs, target)
		t.Logf("  %.0f Hz: %.2f dB", freqs[idx], psdDB[idx])
	}

	// Measure attenuation above original Nyquist
	passbandMax := -200.0
	stopbandMax := -200.0

	for i, f := range freqs {
		if f > 1000 && f < origNyquist*0.9 {
			if psdDB[i] > passbandMax {
				passbandMax = psdDB[i]
			}
		}
		if f > origNyquist*1.1 && f < outputRate/2*0.95 {
			if psdDB[i] > stopbandMax {
				stopbandMax = psdDB[i]
			}
		}
	}

	t.Logf("")
	t.Logf("Max passband level: %.2f dB", passbandMax)
	t.Logf("Max stopband level: %.2f dB", stopbandMax)
	t.Logf("Min attenuation: %.2f dB", passbandMax-stopbandMax)
}
