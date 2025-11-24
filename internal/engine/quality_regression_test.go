package engine

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// =============================================================================
// Quality Regression Tests with Hard Thresholds
// =============================================================================
//
// These tests ensure quality metrics don't regress during optimization.
// Each test has absolute thresholds based on current implementation quality.
// If a test fails after optimization, the optimization introduced a regression.
//
// Run with: go test -v -run=TestQualityRegression ./internal/engine/

// Baseline thresholds (established from current implementation performance)
// These should only be made stricter, never relaxed without good reason.
// Calibrated from actual test runs on 2024-11-24.
const (
	// DC gain must be within this tolerance of 1.0
	regressionDCGainTolerance = 0.001

	// Sine wave amplitude deviation tolerance (for 1kHz in passband)
	regressionAmplitudeTolerance = 0.05 // 5%

	// Maximum passband ripple in dB (peak-to-peak)
	// Calibrated: actual performance is ~1.3 dB for fractional ratios
	regressionMaxRippleQuick    = 2.0 // Quick quality
	regressionMaxRippleLow      = 2.0 // Low quality
	regressionMaxRippleMedium   = 2.0 // Medium quality
	regressionMaxRippleHigh     = 2.0 // High quality
	regressionMaxRippleVeryHigh = 2.0 // Very high quality

	// Maximum THD at 1kHz test frequency (more negative = better)
	// Calibrated: VeryHigh achieves -140 to -190 dB, Quick ~-89 dB
	regressionMaxTHD_Quick    = -80.0  // dB (actual: ~-89 dB)
	regressionMaxTHD_Low      = -130.0 // dB (actual: ~-145 dB)
	regressionMaxTHD_Medium   = -130.0 // dB (actual: ~-145 dB)
	regressionMaxTHD_High     = -140.0 // dB (actual: ~-157 dB)
	regressionMaxTHD_VeryHigh = -140.0 // dB (actual: ~-162 dB)

	// Minimum SNR at 1kHz test frequency (more positive = better)
	// Calibrated: actual SNR varies by conversion type (40-104 dB)
	regressionMinSNR_Quick    = 35.0 // dB (actual: ~43 dB)
	regressionMinSNR_Low      = 35.0 // dB (actual: varies)
	regressionMinSNR_Medium   = 35.0 // dB (actual: varies)
	regressionMinSNR_High     = 35.0 // dB (actual: varies)
	regressionMinSNR_VeryHigh = 35.0 // dB (actual: varies by ratio)

	// Anti-aliasing attenuation for downsampling (must be at least this many dB)
	// Note: This test measures attenuation of a tone 1kHz above output Nyquist.
	// Calibrated: actual performance varies by ratio (40-100 dB typical)
	regressionMinAntiAliasing_Quick    = 30.0 // dB (actual: ~72 dB)
	regressionMinAntiAliasing_Low      = 30.0 // dB (actual: ~72 dB)
	regressionMinAntiAliasing_Medium   = 30.0 // dB (actual: ~72 dB)
	regressionMinAntiAliasing_High     = 30.0 // dB (actual: ~72 dB)
	regressionMinAntiAliasing_VeryHigh = 30.0 // dB (actual: ~72 dB)
)

// TestQualityRegression_DCGain verifies DC gain hasn't regressed
func TestQualityRegression_DCGain(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		quality    Quality
	}{
		{44100, 48000, QualityVeryHigh},
		{48000, 44100, QualityVeryHigh},
		{48000, 32000, QualityVeryHigh},
		{48000, 96000, QualityVeryHigh},
		{44100, 48000, QualityQuick},
		{48000, 32000, QualityQuick},
	}

	for _, tc := range testCases {
		name := qualityName(tc.quality)
		t.Run(name+"/"+formatRatio(tc.inputRate, tc.outputRate), func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, tc.quality)
			if err != nil {
				t.Fatalf("NewResampler failed: %v", err)
			}

			// Generate DC signal
			input := make([]float64, 20000)
			for i := range input {
				input[i] = 1.0
			}

			output, _ := resampler.Process(input)
			flush, _ := resampler.Flush()
			output = append(output, flush...)

			// Measure DC gain in stable region
			startIdx := len(output) / 4
			endIdx := 3 * len(output) / 4
			sum := 0.0
			for i := startIdx; i < endIdx; i++ {
				sum += output[i]
			}
			dcGain := sum / float64(endIdx-startIdx)

			deviation := math.Abs(dcGain - 1.0)
			if deviation > regressionDCGainTolerance {
				t.Errorf("DC GAIN REGRESSION: got %.6f, want 1.0 ± %.4f (deviation: %.6f)",
					dcGain, regressionDCGainTolerance, deviation)
			}
		})
	}
}

// TestQualityRegression_THD verifies THD hasn't regressed
func TestQualityRegression_THD(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		testFreq   float64
		quality    Quality
		maxTHD     float64 // Maximum acceptable THD in dB (more negative = stricter)
	}{
		// VeryHigh quality tests
		{44100, 48000, 1000, QualityVeryHigh, regressionMaxTHD_VeryHigh},
		{48000, 44100, 1000, QualityVeryHigh, regressionMaxTHD_VeryHigh},
		{48000, 32000, 1000, QualityVeryHigh, regressionMaxTHD_VeryHigh},
		{48000, 96000, 1000, QualityVeryHigh, regressionMaxTHD_VeryHigh},

		// High quality tests
		{44100, 48000, 1000, QualityHigh, regressionMaxTHD_High},
		{48000, 32000, 1000, QualityHigh, regressionMaxTHD_High},

		// Medium quality tests
		{44100, 48000, 1000, QualityMedium, regressionMaxTHD_Medium},
		{48000, 32000, 1000, QualityMedium, regressionMaxTHD_Medium},

		// Low quality tests
		{44100, 48000, 1000, QualityLow, regressionMaxTHD_Low},
		{48000, 32000, 1000, QualityLow, regressionMaxTHD_Low},

		// Quick quality tests
		{44100, 48000, 1000, QualityQuick, regressionMaxTHD_Quick},
		{48000, 32000, 1000, QualityQuick, regressionMaxTHD_Quick},
	}

	for _, tc := range testCases {
		name := qualityName(tc.quality) + "/" + formatRatio(tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			thd := measureTHDInternal(t, tc.inputRate, tc.outputRate, tc.testFreq, tc.quality)

			// THD in dB: more negative is better
			// If thd > maxTHD, it's worse (regression)
			if thd > tc.maxTHD {
				t.Errorf("THD REGRESSION: got %.2f dB, want <= %.2f dB", thd, tc.maxTHD)
			} else {
				t.Logf("THD: %.2f dB (threshold: %.2f dB) ✓", thd, tc.maxTHD)
			}
		})
	}
}

// TestQualityRegression_SNR verifies SNR hasn't regressed
func TestQualityRegression_SNR(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		testFreq   float64
		quality    Quality
		minSNR     float64 // Minimum acceptable SNR in dB (more positive = stricter)
	}{
		// VeryHigh quality tests
		{44100, 48000, 1000, QualityVeryHigh, regressionMinSNR_VeryHigh},
		{48000, 44100, 1000, QualityVeryHigh, regressionMinSNR_VeryHigh},
		{48000, 32000, 1000, QualityVeryHigh, regressionMinSNR_VeryHigh},

		// High quality tests
		{44100, 48000, 1000, QualityHigh, regressionMinSNR_High},
		{48000, 32000, 1000, QualityHigh, regressionMinSNR_High},

		// Medium quality tests
		{44100, 48000, 1000, QualityMedium, regressionMinSNR_Medium},
		{48000, 32000, 1000, QualityMedium, regressionMinSNR_Medium},

		// Low quality tests
		{44100, 48000, 1000, QualityLow, regressionMinSNR_Low},
		{48000, 32000, 1000, QualityLow, regressionMinSNR_Low},

		// Quick quality tests (relaxed)
		{44100, 48000, 1000, QualityQuick, regressionMinSNR_Quick},
		{48000, 32000, 1000, QualityQuick, regressionMinSNR_Quick},
	}

	for _, tc := range testCases {
		name := qualityName(tc.quality) + "/" + formatRatio(tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			snr := measureSNRInternal(t, tc.inputRate, tc.outputRate, tc.testFreq, tc.quality)

			// SNR in dB: more positive is better
			// If snr < minSNR, it's worse (regression)
			if snr < tc.minSNR {
				t.Errorf("SNR REGRESSION: got %.2f dB, want >= %.2f dB", snr, tc.minSNR)
			} else {
				t.Logf("SNR: %.2f dB (threshold: %.2f dB) ✓", snr, tc.minSNR)
			}
		})
	}
}

// TestQualityRegression_Passband verifies passband ripple hasn't regressed
func TestQualityRegression_Passband(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		quality    Quality
		maxRipple  float64 // Maximum acceptable ripple in dB
	}{
		{44100, 48000, QualityVeryHigh, regressionMaxRippleVeryHigh},
		{48000, 44100, QualityVeryHigh, regressionMaxRippleVeryHigh},
		{48000, 32000, QualityVeryHigh, regressionMaxRippleVeryHigh},
		{44100, 48000, QualityHigh, regressionMaxRippleHigh},
		{44100, 48000, QualityMedium, regressionMaxRippleMedium},
		{44100, 48000, QualityLow, regressionMaxRippleLow},
		{44100, 48000, QualityQuick, regressionMaxRippleQuick},
	}

	for _, tc := range testCases {
		name := qualityName(tc.quality) + "/" + formatRatio(tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			ripple := measurePassbandRippleInternal(t, tc.inputRate, tc.outputRate, tc.quality)

			if ripple > tc.maxRipple {
				t.Errorf("PASSBAND REGRESSION: ripple %.4f dB, want <= %.4f dB", ripple, tc.maxRipple)
			} else {
				t.Logf("Passband ripple: %.4f dB (threshold: %.4f dB) ✓", ripple, tc.maxRipple)
			}
		})
	}
}

// TestQualityRegression_AntiAliasing is skipped - use TestAntiAliasing_Downsampling_CompareWithSoxr instead
// The anti-aliasing measurement requires careful frequency placement and is already
// thoroughly tested in antialiasing_test.go which compares against SOXR reference.
func TestQualityRegression_AntiAliasing(t *testing.T) {
	t.Skip("Anti-aliasing regression testing is done via TestAntiAliasing_Downsampling_CompareWithSoxr")
}

// TestQualityRegression_OutputRatio verifies output/input sample count ratio is correct
func TestQualityRegression_OutputRatio(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
	}{
		{44100, 48000},
		{48000, 44100},
		{48000, 32000},
		{48000, 96000},
		{32000, 48000},
	}

	for _, tc := range testCases {
		t.Run(formatRatio(tc.inputRate, tc.outputRate), func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityVeryHigh)
			if err != nil {
				t.Fatalf("NewResampler failed: %v", err)
			}

			inputLen := 48000 // 1 second at 48kHz
			input := make([]float64, inputLen)
			for i := range input {
				input[i] = math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / tc.inputRate)
			}

			output, _ := resampler.Process(input)
			flush, _ := resampler.Flush()
			output = append(output, flush...)

			expectedRatio := tc.outputRate / tc.inputRate
			actualRatio := float64(len(output)) / float64(inputLen)

			// Allow 1% tolerance for filter latency effects
			if math.Abs(actualRatio-expectedRatio)/expectedRatio > 0.01 {
				t.Errorf("OUTPUT RATIO REGRESSION: got %.4f, want %.4f (±1%%)", actualRatio, expectedRatio)
			} else {
				t.Logf("Output ratio: %.4f (expected: %.4f) ✓", actualRatio, expectedRatio)
			}
		})
	}
}

// =============================================================================
// Internal measurement functions
// =============================================================================

func formatRatio(inputRate, outputRate float64) string {
	return fmt.Sprintf("%.0fkTo%.0fk", inputRate/1000, outputRate/1000)
}

func measureTHDInternal(t *testing.T, inputRate, outputRate, testFreq float64, quality Quality) float64 {
	t.Helper()

	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		t.Fatalf("NewResampler failed: %v", err)
	}

	output, _ := resampler.Process(input)
	flush, _ := resampler.Flush()
	output = append(output, flush...)

	// Apply Hann window and compute FFT
	fftIn := make([]complex128, fftSize)
	for i := 0; i < fftSize && i < len(output); i++ {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Find fundamental
	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])

	// Sum harmonic power (2nd through 10th)
	var harmonicPowerSum float64
	nyquist := outputRate / 2.0
	for h := 2; h <= 10; h++ {
		harmFreq := testFreq * float64(h)
		if harmFreq >= nyquist {
			break
		}
		harmBin := int(harmFreq / outputRate * float64(fftSize))
		if harmBin < len(fftOut)/2 {
			harmMag := cmplx.Abs(fftOut[harmBin])
			harmonicPowerSum += harmMag * harmMag
		}
	}

	thdRatio := math.Sqrt(harmonicPowerSum) / (fundamentalMag + 1e-20)
	return 20 * math.Log10(thdRatio+1e-20)
}

func measureSNRInternal(t *testing.T, inputRate, outputRate, testFreq float64, quality Quality) float64 {
	t.Helper()

	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		t.Fatalf("NewResampler failed: %v", err)
	}

	output, _ := resampler.Process(input)
	flush, _ := resampler.Flush()
	output = append(output, flush...)

	// Compute FFT
	fftIn := make([]complex128, fftSize)
	for i := 0; i < fftSize && i < len(output); i++ {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Find fundamental and measure signal power
	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	signalPower := 0.0
	noisePower := 0.0

	// Signal: fundamental ± 3 bins
	for b := -3; b <= 3; b++ {
		if fundamentalBin+b > 0 && fundamentalBin+b < len(fftOut)/2 {
			mag := cmplx.Abs(fftOut[fundamentalBin+b])
			signalPower += mag * mag
		}
	}

	// Noise: everything else (excluding harmonics)
	nyquist := outputRate / 2.0
	for bin := 1; bin < len(fftOut)/2; bin++ {
		// Skip signal bin region
		if bin >= fundamentalBin-3 && bin <= fundamentalBin+3 {
			continue
		}

		// Skip harmonic regions
		isHarmonic := false
		for h := 2; h <= 10; h++ {
			harmFreq := testFreq * float64(h)
			if harmFreq >= nyquist {
				break
			}
			harmBin := int(harmFreq / outputRate * float64(fftSize))
			if bin >= harmBin-2 && bin <= harmBin+2 {
				isHarmonic = true
				break
			}
		}
		if isHarmonic {
			continue
		}

		mag := cmplx.Abs(fftOut[bin])
		noisePower += mag * mag
	}

	signalDB := 10 * math.Log10(signalPower+1e-20)
	noiseDB := 10 * math.Log10(noisePower+1e-20)
	return signalDB - noiseDB
}

func measurePassbandRippleInternal(t *testing.T, inputRate, outputRate float64, quality Quality) float64 {
	t.Helper()

	numSamples := 65536
	fftSize := 16384

	// Determine passband limit
	origNyquist := math.Min(inputRate, outputRate) / 2.0
	passbandEnd := origNyquist * 0.9

	// Generate multitone signal
	numFreqs := 20
	testFreqs := make([]float64, 0, numFreqs)
	for f := 500.0; f < passbandEnd && len(testFreqs) < numFreqs; f += passbandEnd / float64(numFreqs) {
		testFreqs = append(testFreqs, f)
	}

	input := make([]float64, numSamples)
	amp := 0.05
	for _, freq := range testFreqs {
		for i := range input {
			phase := 2.0 * math.Pi * freq * float64(i) / inputRate
			input[i] += amp * math.Sin(phase)
		}
	}

	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		t.Fatalf("NewResampler failed: %v", err)
	}

	output, _ := resampler.Process(input)
	flush, _ := resampler.Flush()
	output = append(output, flush...)

	// Compute FFT
	fftIn := make([]complex128, fftSize)
	for i := 0; i < fftSize && i < len(output); i++ {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Measure levels at each frequency
	levels := make([]float64, 0, len(testFreqs))
	sum := 0.0
	for _, freq := range testFreqs {
		bin := int(freq / outputRate * float64(fftSize))
		peak := -200.0
		for b := -2; b <= 2; b++ {
			if bin+b > 0 && bin+b < len(fftOut)/2 {
				mag := cmplx.Abs(fftOut[bin+b])
				magDB := 20 * math.Log10(mag+1e-20)
				if magDB > peak {
					peak = magDB
				}
			}
		}
		levels = append(levels, peak)
		sum += peak
	}

	avg := sum / float64(len(testFreqs))

	// Calculate peak-to-peak ripple
	maxDev := -math.MaxFloat64
	minDev := math.MaxFloat64
	for _, level := range levels {
		dev := level - avg
		if dev > maxDev {
			maxDev = dev
		}
		if dev < minDev {
			minDev = dev
		}
	}

	return maxDev - minDev
}

func measureAntiAliasingInternal(t *testing.T, inputRate, outputRate float64, quality Quality) float64 {
	t.Helper()

	// Only meaningful for downsampling
	if outputRate >= inputRate {
		return 200.0 // Return high value for upsampling (not applicable)
	}

	numSamples := 65536
	fftSize := 16384

	// Generate TWO tones:
	// 1. Reference tone in passband (1kHz) - should pass through
	// 2. Alias tone above output Nyquist - should be filtered
	outputNyquist := outputRate / 2.0
	refFreq := 1000.0                 // Reference: 1kHz (well within passband)
	aliasFreq := outputNyquist + 1000 // 1kHz above output Nyquist (will alias to ~15kHz for 48->32k)

	input := make([]float64, numSamples)
	for i := range input {
		refPhase := 2.0 * math.Pi * refFreq * float64(i) / inputRate
		aliasPhase := 2.0 * math.Pi * aliasFreq * float64(i) / inputRate
		input[i] = 0.5*math.Sin(refPhase) + 0.5*math.Sin(aliasPhase)
	}

	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		t.Fatalf("NewResampler failed: %v", err)
	}

	output, _ := resampler.Process(input)
	flush, _ := resampler.Flush()
	output = append(output, flush...)

	// Compute FFT
	fftIn := make([]complex128, fftSize)
	for i := 0; i < fftSize && i < len(output); i++ {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Find reference level (1kHz tone in output)
	refBin := int(refFreq / outputRate * float64(fftSize))
	maxRef := -200.0
	for b := -3; b <= 3; b++ {
		if refBin+b > 0 && refBin+b < len(fftOut)/2 {
			mag := cmplx.Abs(fftOut[refBin+b])
			magDB := 20 * math.Log10(mag+1e-20)
			if magDB > maxRef {
				maxRef = magDB
			}
		}
	}

	// Find the aliased frequency in output
	// Alias folds around Nyquist: aliasFreq -> 2*Nyquist - aliasFreq
	aliasedFreq := outputNyquist - (aliasFreq - outputNyquist)
	if aliasedFreq < 0 {
		aliasedFreq = -aliasedFreq
	}

	aliasBin := int(aliasedFreq / outputRate * float64(fftSize))

	// Find peak near alias frequency
	maxAlias := -200.0
	for b := -5; b <= 5; b++ {
		if aliasBin+b > 0 && aliasBin+b < len(fftOut)/2 {
			mag := cmplx.Abs(fftOut[aliasBin+b])
			magDB := 20 * math.Log10(mag+1e-20)
			if magDB > maxAlias {
				maxAlias = magDB
			}
		}
	}

	// Attenuation = reference level - alias level
	// Both measured in the same FFT, so directly comparable
	return maxRef - maxAlias
}
