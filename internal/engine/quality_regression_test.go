// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

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

	// Maximum passband ripple in dB (peak-to-peak gain spread of 20 tones over
	// the preset's own passband, see measurePassbandRipple).
	// Measured 2026-10-19 with that definition: 44.1k to 48k VeryHigh 0.00016,
	// High 0.00139, Medium 0.00269, Low 0.00004, Quick 1.927 dB (cubic droop
	// at 0.676 of Nyquist); 48k to 44.1k VeryHigh 0.00000, Low 0.00008.
	regressionMaxRippleQuick    = 2.0  // Quick quality (cubic interpolation)
	regressionMaxRippleLow      = 0.01 // Low quality
	regressionMaxRippleMedium   = 0.01 // Medium quality
	regressionMaxRippleHigh     = 0.01 // High quality
	regressionMaxRippleVeryHigh = 0.01 // Very high quality

	// Maximum THD at 1kHz test frequency (more negative = better)
	// Calibrated: VeryHigh achieves -140 to -190 dB, Quick ~-89 dB
	regressionMaxTHD_Quick    = -80.0  // dB (actual: ~-89 dB)
	regressionMaxTHD_Low      = -130.0 // dB (actual: ~-145 dB)
	regressionMaxTHD_Medium   = -129.0 // dB (actual: ~-130 dB, after dynamic Fs1 change)
	regressionMaxTHD_High     = -140.0 // dB (actual: ~-157 dB)
	regressionMaxTHD_VeryHigh = -140.0 // dB (actual: ~-162 dB)

	// Active-interpolation ratio THD limits (#54). Measured on 2026-07-17:
	// 32000->44100 High: -162.05 dB, Medium: -134.95 dB, Low: -144.22 dB;
	// 44100->64000 High: -151.63 dB, Medium: -135.03 dB, Low: -139.47 dB.
	// High gets a dedicated limit 6.6 dB above its worst measurement. Medium
	// and Low reuse the regressionMaxTHD_Medium and regressionMaxTHD_Low
	// constants above, which sit 6.0 and 9.5 dB above the worst measurements
	// here, so no dedicated constants are needed for them.
	regressionMaxTHDInterp_High = -145.0 // dB (worst measured: -151.63 dB)

	// Minimum SNR at 1kHz test frequency (more positive = better)
	// Calibrated: actual SNR varies by conversion type (40-104 dB)
	regressionMinSNR_Quick    = 35.0 // dB (actual: ~43 dB)
	regressionMinSNR_Low      = 35.0 // dB (actual: varies)
	regressionMinSNR_Medium   = 35.0 // dB (actual: varies)
	regressionMinSNR_High     = 35.0 // dB (actual: varies)
	regressionMinSNR_VeryHigh = 35.0 // dB (actual: varies by ratio)

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

// Active-interpolation THD regression (#54): unlike the exact-rational
// ratios above, these ratios consult the sub-phase Catmull-Rom
// interpolation banks, so this pins the whole pipeline against the
// phase-boundary indexing defect from #51, fixed in PR #56 (stage-level
// guard: phase_wrap_measure_test.go). Thresholds are measured values plus
// margin; see constants.
func TestQualityRegression_THD_ActiveInterpolation(t *testing.T) {
	tests := []struct {
		inputRate, outputRate float64
		testFreq              float64
		quality               Quality
		maxTHD                float64
	}{
		{32000, 44100, 1000, QualityHigh, regressionMaxTHDInterp_High},
		{32000, 44100, 1000, QualityMedium, regressionMaxTHD_Medium},
		{32000, 44100, 1000, QualityLow, regressionMaxTHD_Low},
		{44100, 64000, 1000, QualityHigh, regressionMaxTHDInterp_High},
		{44100, 64000, 1000, QualityMedium, regressionMaxTHD_Medium},
		{44100, 64000, 1000, QualityLow, regressionMaxTHD_Low},
	}

	for _, tc := range tests {
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

// TestQualityRegression_Passband verifies passband ripple hasn't regressed,
// over each preset's own passband (qualityToPassbandEnd).
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
		// Low's anti-alias path: TestPassbandEdge_NonIntegerDownsampling_MatchesDecimationPath
		// skips its flatness loop for Low, so this row is the only flatness pin there.
		{48000, 44100, QualityLow, regressionMaxRippleLow},
		{44100, 48000, QualityQuick, regressionMaxRippleQuick},
	}

	for _, tc := range testCases {
		name := qualityName(tc.quality) + "/" + formatRatio(tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			res, err := measurePassbandRipple(tc.inputRate, tc.outputRate, tc.quality, qualityToPassbandEnd(tc.quality))
			if err != nil {
				t.Fatalf("measurePassbandRipple failed: %v", err)
			}
			ripple := res.RipplePeakPeak

			if ripple > tc.maxRipple {
				t.Errorf("PASSBAND REGRESSION: ripple %.4f dB, want <= %.4f dB", ripple, tc.maxRipple)
			} else {
				t.Logf("Passband ripple: %.4f dB (threshold: %.4f dB) ✓", ripple, tc.maxRipple)
			}
		})
	}
}

// TestQualityRegression_AntiAliasing pins the alias rejection of the High
// preset on the non-integer downsampling ratios the library is used for (the
// tone oracle and its thresholds are shared with
// TestAntiAliasing_NonIntegerDownsampling_MeetsStopband). The comparison with
// libsoxr lives in TestAntiAliasing_Downsampling_CompareWithSoxr.
func TestQualityRegression_AntiAliasing(t *testing.T) {
	for _, rr := range [][2]float64{{48000, 32000}, {48000, 44100}} {
		t.Run(fmt.Sprintf("%g_to_%g", rr[0], rr[1]), func(t *testing.T) {
			t.Parallel()
			worst := aaWorstAlias(t, rr[0], rr[1], QualityHigh)
			if want := -qualityToAttenuation(QualityHigh); worst > want {
				t.Errorf("alias level %.1f dB, want <= %.1f dB", worst, want)
			}
		})
	}
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
