package engine

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestMultitoneDetailedPSD(t *testing.T) {
	inputRate := 48000.0
	outputRate := 96000.0

	// Generate multitone signal
	input := generateAntialiasTestSignal(SignalMultitone, inputRate, antialiasTestSamples)

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

	t.Logf("Input samples: %d, Output samples: %d", len(input), len(output))

	// Compute PSD of output
	freqs, psdDB := computePSD(output, outputRate, fftWindowSize)

	origNyquist := inputRate / 2.0 // 24000
	newNyquist := outputRate / 2.0 // 48000

	// Print at key frequencies
	t.Logf("\n=== OUTPUT Signal PSD at Key Frequencies ===")
	keyFreqs := []float64{
		1000, 2000, 4000, 8000, 12000, 16000, 20000, 22000, // Passband tones
		23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, // Around Nyquist
		35000, 40000, 45000,
	}

	for _, target := range keyFreqs {
		if target >= newNyquist {
			continue
		}
		closestIdx := findClosestBin(freqs, target)
		label := ""
		switch target {
		case 24000:
			label = " <- Original Nyquist"
		case 26000:
			label = " <- 22kHz image"
		}
		t.Logf("  %5.0f Hz: %7.2f dB%s", freqs[closestIdx], psdDB[closestIdx], label)
	}

	// Measure bands
	passbandEnd := origNyquist * 0.9    // 21600
	stopbandStart := origNyquist + 1000 // 25000
	stopbandEnd := newNyquist - 1000    // 47000

	passbandEnergy := measureBandEnergy(freqs, psdDB, 100, passbandEnd)
	stopbandEnergy := measureBandEnergy(freqs, psdDB, stopbandStart, stopbandEnd)

	t.Logf("\nPassband (100 - %.0f Hz): %.2f dB", passbandEnd, passbandEnergy)
	t.Logf("Stopband (%.0f - %.0f Hz): %.2f dB", stopbandStart, stopbandEnd, stopbandEnergy)
	t.Logf("STOPBAND ATTENUATION: %.2f dB", passbandEnergy-stopbandEnergy)

	// Find peaks in stopband
	t.Logf("\nStopband peaks (above -100 dB):")
	foundPeaks := false
	for i, f := range freqs {
		if f >= stopbandStart && f <= stopbandEnd && psdDB[i] > -100 {
			t.Logf("  %.0f Hz: %.2f dB", f, psdDB[i])
			foundPeaks = true
		}
	}
	if !foundPeaks {
		t.Logf("  None found")
	}
}

func findClosestBin(freqs []float64, target float64) int {
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

// Simpler FFT-based freq response check for DFT stage
func TestDFTStageFreqResponse(t *testing.T) {
	// Create DFT stage for 2x upsample
	stage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Generate impulse and get impulse response
	impulse := make([]float64, 8192)
	impulse[0] = 1.0

	resp, err := stage.Process(impulse)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}

	// The impulse response is 2x the length (upsampled)
	t.Logf("Impulse response length: %d", len(resp))

	// Compute frequency response via FFT
	fftSize := 16384
	fftIn := make([]complex128, fftSize)
	for i, v := range resp {
		if i < fftSize {
			fftIn[i] = complex(v, 0)
		}
	}

	fftOut := fft(fftIn)

	// Output rate is 96 kHz
	outputRate := 96000.0
	t.Logf("\nDFT Stage Frequency Response (measured):")

	testFreqs := []float64{1000, 10000, 20000, 22000, 23000, 24000, 25000, 26000, 30000, 40000}
	for _, freq := range testFreqs {
		bin := int(freq / outputRate * float64(fftSize))
		if bin < len(fftOut)/2 {
			mag := cmplx.Abs(fftOut[bin])
			magDB := 20 * math.Log10(mag+1e-20)
			label := ""
			if freq == 24000 {
				label = " <- Original Nyquist"
			}
			t.Logf("  %5.0f Hz: %7.2f dB%s", freq, magDB, label)
		}
	}
}

// TestMultitonePeakBased measures using peak energy instead of average
func TestMultitonePeakBased(t *testing.T) {
	inputRate := 48000.0
	outputRate := 96000.0

	// Generate multitone signal
	input := generateAntialiasTestSignal(SignalMultitone, inputRate, antialiasTestSamples)

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

	// Compute PSD of output
	freqs, psdDB := computePSD(output, outputRate, fftWindowSize)

	origNyquist := inputRate / 2.0
	newNyquist := outputRate / 2.0
	passbandEnd := origNyquist * 0.9
	stopbandStart := origNyquist + 1000
	stopbandEnd := newNyquist - 1000

	// Find PEAK energy in passband (strongest tone)
	passbandPeak := -200.0
	for i, f := range freqs {
		if f >= 100 && f < passbandEnd && psdDB[i] > passbandPeak {
			passbandPeak = psdDB[i]
		}
	}

	// Find PEAK energy in stopband (strongest image)
	stopbandPeak := -200.0
	stopbandPeakFreq := 0.0
	for i, f := range freqs {
		if f >= stopbandStart && f <= stopbandEnd && psdDB[i] > stopbandPeak {
			stopbandPeak = psdDB[i]
			stopbandPeakFreq = f
		}
	}

	// Also get average for comparison
	passbandAvg := measureBandEnergy(freqs, psdDB, 100, passbandEnd)
	stopbandAvg := measureBandEnergy(freqs, psdDB, stopbandStart, stopbandEnd)

	t.Logf("Passband peak: %.2f dB", passbandPeak)
	t.Logf("Passband average: %.2f dB", passbandAvg)
	t.Logf("Stopband peak: %.2f dB @ %.0f Hz", stopbandPeak, stopbandPeakFreq)
	t.Logf("Stopband average: %.2f dB", stopbandAvg)
	t.Logf("")
	t.Logf("PEAK-BASED ATTENUATION: %.2f dB", passbandPeak-stopbandPeak)
	t.Logf("AVERAGE-BASED ATTENUATION: %.2f dB", passbandAvg-stopbandAvg)

	// Peak-based should give ~90+ dB attenuation
	peakAttenuation := passbandPeak - stopbandPeak
	if peakAttenuation < 80.0 {
		t.Errorf("Peak-based attenuation %.2f dB below 80 dB threshold", peakAttenuation)
	}
}
