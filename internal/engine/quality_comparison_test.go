package engine

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// =============================================================================
// Static Reference Data (loaded from JSON file for CI without C tools)
// =============================================================================

// soxrReferenceData holds pre-computed soxr reference values
type soxrReferenceData struct {
	AntiAliasing map[string]float64         `json:"antialiasing"`
	Quality      map[string]json.RawMessage `json:"quality"`
}

var (
	staticRefData     *soxrReferenceData
	staticRefDataOnce sync.Once
	errStaticRefData  error
)

// loadStaticReferenceData loads pre-computed soxr reference data from JSON
func loadStaticReferenceData() (*soxrReferenceData, error) {
	staticRefDataOnce.Do(func() {
		_, currentFile, _, ok := runtime.Caller(0)
		if !ok {
			errStaticRefData = fmt.Errorf("failed to get current file path")
			return
		}

		dataPath := filepath.Join(filepath.Dir(currentFile), "testdata", "soxr_reference_data.json")
		data, err := os.ReadFile(dataPath)
		if err != nil {
			errStaticRefData = fmt.Errorf("failed to read reference data: %w", err)
			return
		}

		staticRefData = &soxrReferenceData{}
		if err := json.Unmarshal(data, staticRefData); err != nil {
			errStaticRefData = fmt.Errorf("failed to parse reference data: %w", err)
			return
		}
	})

	return staticRefData, errStaticRefData
}

// Quality comparison tests between Go resampler and soxr reference
// Goal: Identify where Go performs worse than soxr, NOT to make tests pass

const (
	qualityTestSamples = 65536 // Larger sample for better frequency resolution
)

// =============================================================================
// TEST 1: Passband Ripple / Frequency Response Flatness
// =============================================================================

// PassbandRippleResult holds passband flatness measurement
type PassbandRippleResult struct {
	InputRate      float64
	OutputRate     float64
	MaxDeviation   float64 // Maximum deviation from 0 dB in passband
	MinDeviation   float64 // Minimum deviation from 0 dB in passband
	RipplePeakPeak float64 // Peak-to-peak ripple in dB
	Frequencies    []float64
	Response       []float64 // Frequency response in dB
}

// measurePassbandRipple measures frequency response flatness in passband
// Uses multitone test signal for accurate frequency response measurement
func measurePassbandRipple(inputRate, outputRate float64, quality Quality) (*PassbandRippleResult, error) {
	numSamples := qualityTestSamples
	fftSize := 16384

	// Determine passband limits
	origNyquist := math.Min(inputRate, outputRate) / 2.0
	passbandEnd := origNyquist * 0.9 // 90% of lower Nyquist

	// Generate test frequencies (20 tones across passband)
	numFreqs := 20
	testFreqs := make([]float64, 0, numFreqs)
	for f := 500.0; f < passbandEnd && len(testFreqs) < numFreqs; f += passbandEnd / float64(numFreqs) {
		testFreqs = append(testFreqs, f)
	}

	// Generate multitone input signal
	input := make([]float64, numSamples)
	amp := 0.05 // Low amplitude to avoid clipping
	for _, freq := range testFreqs {
		for i := range input {
			phase := 2.0 * math.Pi * freq * float64(i) / inputRate
			input[i] += amp * math.Sin(phase)
		}
	}

	// Resample
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	output, err := resampler.Process(input)
	if err != nil {
		return nil, fmt.Errorf("failed to process: %w", err)
	}

	flush, err := resampler.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush: %w", err)
	}
	output = append(output, flush...)

	// Compute FFT of output
	fftIn := make([]complex128, fftSize)
	for i := 0; i < len(output) && i < fftSize; i++ {
		// Apply Hann window
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Measure level at each test frequency
	levels := make([]float64, len(testFreqs))
	sum := 0.0
	for i, freq := range testFreqs {
		bin := int(freq / outputRate * float64(fftSize))
		// Find peak in ±2 bins around expected
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
		levels[i] = peak
		sum += peak
	}

	avg := sum / float64(len(testFreqs))

	// Measure deviation from average
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

	return &PassbandRippleResult{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		MaxDeviation:   maxDev,
		MinDeviation:   minDev,
		RipplePeakPeak: maxDev - minDev,
		Frequencies:    testFreqs,
		Response:       levels,
	}, nil
}

// TestPassbandRipple_CompareWithSoxr compares passband flatness
func TestPassbandRipple_CompareWithSoxr(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz_CD_to_DAT"},
		{48000, 44100, "48kHz_to_44.1kHz_DAT_to_CD"},
		{48000, 96000, "48kHz_to_96kHz_2x_upsample"},
		{96000, 48000, "96kHz_to_48kHz_2x_downsample"},
		{48000, 32000, "48kHz_to_32kHz_BirdNET_v3"},
		{16000, 48000, "16kHz_to_48kHz_BirdNET_v2.4"},
		{32000, 48000, "32kHz_to_48kHz_BirdNET_v2.4"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Get soxr reference
			soxrResult, err := getSoxrPassbandRipple(tc.inputRate, tc.outputRate)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Get Go result
			goResult, err := measurePassbandRipple(tc.inputRate, tc.outputRate, QualityVeryHigh)
			if err != nil {
				t.Fatalf("measurePassbandRipple failed: %v", err)
			}

			t.Logf("=== Passband Ripple Test ===")
			t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
			t.Logf("")
			t.Logf("SOXR Reference:")
			t.Logf("  Peak-to-peak ripple: %.4f dB", soxrResult.RipplePeakPeak)
			t.Logf("  Max deviation: %.4f dB", soxrResult.MaxDeviation)
			t.Logf("  Min deviation: %.4f dB", soxrResult.MinDeviation)
			t.Logf("")
			t.Logf("Go Resampler:")
			t.Logf("  Peak-to-peak ripple: %.4f dB", goResult.RipplePeakPeak)
			t.Logf("  Max deviation: %.4f dB", goResult.MaxDeviation)
			t.Logf("  Min deviation: %.4f dB", goResult.MinDeviation)
			t.Logf("")
			t.Logf("=== Comparison ===")
			rippleDiff := goResult.RipplePeakPeak - soxrResult.RipplePeakPeak
			t.Logf("Ripple difference: %.4f dB (go - soxr)", rippleDiff)

			if rippleDiff > 0.5 {
				t.Logf("WARNING: Go has %.4f dB more passband ripple than soxr", rippleDiff)
			}

			// Fail if Go ripple is significantly worse (> 1 dB more than soxr)
			if rippleDiff > 1.0 {
				t.Errorf("PASSBAND RIPPLE GAP: Go has %.4f dB more ripple than soxr", rippleDiff)
			}
		})
	}
}

// =============================================================================
// TEST 2: THD (Total Harmonic Distortion)
// =============================================================================

// THDResult holds THD measurement
type THDResult struct {
	InputRate     float64
	OutputRate    float64
	TestFrequency float64
	FundamentalDB float64
	THD_DB        float64   // THD in dB (negative is better)
	THD_Percent   float64   // THD as percentage
	Harmonics     []float64 // Harmonic levels in dB
}

// measureTHD measures total harmonic distortion
func measureTHD(inputRate, outputRate, testFreq float64, quality Quality) (*THDResult, error) {
	numSamples := qualityTestSamples
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase) // Leave headroom
	}

	// Resample
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	output, err := resampler.Process(input)
	if err != nil {
		return nil, fmt.Errorf("failed to process: %w", err)
	}

	flush, err := resampler.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush: %w", err)
	}
	output = append(output, flush...)

	// Apply Hann window and compute FFT
	if len(output) < fftSize {
		padded := make([]float64, fftSize)
		copy(padded, output)
		output = padded
	}

	fftIn := make([]complex128, fftSize)
	for i := range fftSize {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Find fundamental
	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])
	fundamentalDB := 20 * math.Log10(fundamentalMag+1e-20)

	// Find harmonics (2nd through 10th)
	var harmonics []float64
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
			harmDB := 20 * math.Log10(harmMag+1e-20)
			harmonics = append(harmonics, harmDB)

			// Sum power for THD calculation
			harmonicPowerSum += harmMag * harmMag
		}
	}

	// THD = sqrt(sum of harmonic powers) / fundamental
	thdRatio := math.Sqrt(harmonicPowerSum) / (fundamentalMag + 1e-20)
	thdDB := 20 * math.Log10(thdRatio+1e-20)
	thdPercent := thdRatio * 100

	return &THDResult{
		InputRate:     inputRate,
		OutputRate:    outputRate,
		TestFrequency: testFreq,
		FundamentalDB: fundamentalDB,
		THD_DB:        thdDB,
		THD_Percent:   thdPercent,
		Harmonics:     harmonics,
	}, nil
}

// TestTHD_CompareWithSoxr compares harmonic distortion
func TestTHD_CompareWithSoxr(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		testFreq   float64
		name       string
	}{
		{44100, 48000, 1000, "44.1kHz_to_48kHz_1kHz"},
		{48000, 44100, 1000, "48kHz_to_44.1kHz_1kHz"},
		{48000, 96000, 1000, "48kHz_to_96kHz_1kHz"},
		{96000, 48000, 1000, "96kHz_to_48kHz_1kHz"},
		{48000, 32000, 1000, "48kHz_to_32kHz_BirdNET_v3_1kHz"},
		{16000, 48000, 1000, "16kHz_to_48kHz_BirdNET_v2.4_1kHz"},
		{32000, 48000, 1000, "32kHz_to_48kHz_BirdNET_v2.4_1kHz"},
		{44100, 48000, 10000, "44.1kHz_to_48kHz_10kHz"},
		{48000, 44100, 10000, "48kHz_to_44.1kHz_10kHz"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Get soxr reference
			soxrResult, err := getSoxrTHD(tc.inputRate, tc.outputRate, tc.testFreq)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Get Go result
			goResult, err := measureTHD(tc.inputRate, tc.outputRate, tc.testFreq, QualityVeryHigh)
			if err != nil {
				t.Fatalf("measureTHD failed: %v", err)
			}

			t.Logf("=== THD Test @ %.0f Hz ===", tc.testFreq)
			t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
			t.Logf("")
			t.Logf("SOXR Reference:")
			t.Logf("  THD: %.2f dB (%.6f%%)", soxrResult.THD_DB, soxrResult.THD_Percent)
			t.Logf("")
			t.Logf("Go Resampler:")
			t.Logf("  THD: %.2f dB (%.6f%%)", goResult.THD_DB, goResult.THD_Percent)
			if len(goResult.Harmonics) > 0 {
				t.Logf("  2nd harmonic: %.2f dB", goResult.Harmonics[0])
			}
			if len(goResult.Harmonics) > 1 {
				t.Logf("  3rd harmonic: %.2f dB", goResult.Harmonics[1])
			}
			t.Logf("")
			t.Logf("=== Comparison ===")
			thdDiff := goResult.THD_DB - soxrResult.THD_DB
			t.Logf("THD difference: %.2f dB (go - soxr, positive = go is worse)", thdDiff)

			// THD in dB: more negative is better
			// Positive diff means Go has worse (higher) THD
			if thdDiff > 20 {
				t.Errorf("THD GAP: Go THD is %.2f dB worse than soxr", thdDiff)
			}
		})
	}
}

// =============================================================================
// TEST 3: SNR (Signal-to-Noise Ratio)
// =============================================================================

// SNRResult holds SNR measurement
type SNRResult struct {
	InputRate  float64
	OutputRate float64
	SignalDB   float64
	NoiseDB    float64
	SNR_DB     float64
}

// measureSNR measures signal-to-noise ratio
func measureSNR(inputRate, outputRate, testFreq float64, quality Quality) (*SNRResult, error) {
	numSamples := qualityTestSamples
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Resample
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	output, err := resampler.Process(input)
	if err != nil {
		return nil, fmt.Errorf("failed to process: %w", err)
	}

	flush, err := resampler.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush: %w", err)
	}
	output = append(output, flush...)

	// Compute FFT
	if len(output) < fftSize {
		padded := make([]float64, fftSize)
		copy(padded, output)
		output = padded
	}

	fftIn := make([]complex128, fftSize)
	for i := range fftSize {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	// Find signal power (fundamental + a few bins around it)
	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	var signalPower float64
	for b := fundamentalBin - 2; b <= fundamentalBin+2; b++ {
		if b > 0 && b < len(fftOut)/2 {
			mag := cmplx.Abs(fftOut[b])
			signalPower += mag * mag
		}
	}

	// Find noise power (everything else below Nyquist)
	var noisePower float64
	nyquistBin := len(fftOut) / 2
	for b := 1; b < nyquistBin; b++ {
		// Skip signal bins and their harmonics
		isSignal := false
		for h := 1; h <= 10; h++ {
			harmBin := fundamentalBin * h
			if b >= harmBin-2 && b <= harmBin+2 {
				isSignal = true
				break
			}
		}
		if !isSignal {
			mag := cmplx.Abs(fftOut[b])
			noisePower += mag * mag
		}
	}

	signalDB := 10 * math.Log10(signalPower+1e-20)
	noiseDB := 10 * math.Log10(noisePower+1e-20)
	snrDB := signalDB - noiseDB

	return &SNRResult{
		InputRate:  inputRate,
		OutputRate: outputRate,
		SignalDB:   signalDB,
		NoiseDB:    noiseDB,
		SNR_DB:     snrDB,
	}, nil
}

// TestSNR_CompareWithSoxr compares signal-to-noise ratio
func TestSNR_CompareWithSoxr(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz"},
		{48000, 44100, "48kHz_to_44.1kHz"},
		{48000, 96000, "48kHz_to_96kHz"},
		{96000, 48000, "96kHz_to_48kHz"},
		{48000, 32000, "48kHz_to_32kHz_BirdNET_v3"},
		{16000, 48000, "16kHz_to_48kHz_BirdNET_v2.4"},
		{32000, 48000, "32kHz_to_48kHz_BirdNET_v2.4"},
	}

	testFreq := 1000.0

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Get soxr reference
			soxrResult, err := getSoxrSNR(tc.inputRate, tc.outputRate, testFreq)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Get Go result
			goResult, err := measureSNR(tc.inputRate, tc.outputRate, testFreq, QualityVeryHigh)
			if err != nil {
				t.Fatalf("measureSNR failed: %v", err)
			}

			t.Logf("=== SNR Test ===")
			t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
			t.Logf("")
			t.Logf("SOXR Reference:")
			t.Logf("  SNR: %.2f dB", soxrResult.SNR_DB)
			t.Logf("")
			t.Logf("Go Resampler:")
			t.Logf("  Signal: %.2f dB", goResult.SignalDB)
			t.Logf("  Noise:  %.2f dB", goResult.NoiseDB)
			t.Logf("  SNR:    %.2f dB", goResult.SNR_DB)
			t.Logf("")
			t.Logf("=== Comparison ===")
			snrDiff := soxrResult.SNR_DB - goResult.SNR_DB
			t.Logf("SNR difference: %.2f dB (soxr - go, positive = soxr is better)", snrDiff)

			if snrDiff > 20 {
				t.Errorf("SNR GAP: soxr has %.2f dB better SNR than Go", snrDiff)
			}
		})
	}
}

// =============================================================================
// TEST 4: Impulse Response Characteristics
// =============================================================================

// ImpulseResult holds impulse response analysis
type ImpulseResult struct {
	InputRate       float64
	OutputRate      float64
	PreRingingDB    float64 // Peak level before main impulse
	PostRingingDB   float64 // Peak level after main impulse
	MainPeakSamples int     // Position of main peak
	RingoutSamples  int     // Samples until ringing decays below -60dB
}

// measureImpulseResponse analyzes impulse response characteristics
func measureImpulseResponse(inputRate, outputRate float64, quality Quality) (*ImpulseResult, error) {
	// Create impulse at center of buffer
	numSamples := 8192
	impulse := make([]float64, numSamples)
	impulsePos := numSamples / 2
	impulse[impulsePos] = 1.0

	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	output, err := resampler.Process(impulse)
	if err != nil {
		return nil, fmt.Errorf("failed to process: %w", err)
	}

	flush, err := resampler.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush: %w", err)
	}
	output = append(output, flush...)

	// Find main peak
	mainPeakIdx := 0
	mainPeakVal := 0.0
	for i, v := range output {
		if math.Abs(v) > mainPeakVal {
			mainPeakVal = math.Abs(v)
			mainPeakIdx = i
		}
	}

	// Measure pre-ringing (peak before main impulse)
	preRingingPeak := 0.0
	for i := 0; i < mainPeakIdx; i++ {
		if math.Abs(output[i]) > preRingingPeak {
			preRingingPeak = math.Abs(output[i])
		}
	}
	preRingingDB := 20 * math.Log10(preRingingPeak/mainPeakVal+1e-20)

	// Measure post-ringing (peak after main impulse, excluding it)
	postRingingPeak := 0.0
	for i := mainPeakIdx + 10; i < len(output); i++ {
		if math.Abs(output[i]) > postRingingPeak {
			postRingingPeak = math.Abs(output[i])
		}
	}
	postRingingDB := 20 * math.Log10(postRingingPeak/mainPeakVal+1e-20)

	// Find ringout time (samples until below -60dB)
	threshold := mainPeakVal * math.Pow(10, -60.0/20.0)
	ringoutSamples := 0
	for i := mainPeakIdx; i < len(output); i++ {
		if math.Abs(output[i]) > threshold {
			ringoutSamples = i - mainPeakIdx
		}
	}

	return &ImpulseResult{
		InputRate:       inputRate,
		OutputRate:      outputRate,
		PreRingingDB:    preRingingDB,
		PostRingingDB:   postRingingDB,
		MainPeakSamples: mainPeakIdx,
		RingoutSamples:  ringoutSamples,
	}, nil
}

// TestImpulseResponse_CompareWithSoxr compares impulse response characteristics
func TestImpulseResponse_CompareWithSoxr(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz"},
		{48000, 44100, "48kHz_to_44.1kHz"},
		{48000, 96000, "48kHz_to_96kHz"},
		{96000, 48000, "96kHz_to_48kHz"},
		{48000, 32000, "48kHz_to_32kHz_BirdNET_v3"},
		{16000, 48000, "16kHz_to_48kHz_BirdNET_v2.4"},
		{32000, 48000, "32kHz_to_48kHz_BirdNET_v2.4"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Get soxr reference
			soxrResult, err := getSoxrImpulse(tc.inputRate, tc.outputRate)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Get Go result
			goResult, err := measureImpulseResponse(tc.inputRate, tc.outputRate, QualityVeryHigh)
			if err != nil {
				t.Fatalf("measureImpulseResponse failed: %v", err)
			}

			t.Logf("=== Impulse Response Test ===")
			t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
			t.Logf("")
			t.Logf("SOXR Reference:")
			t.Logf("  Pre-ringing:  %.2f dB", soxrResult.PreRingingDB)
			t.Logf("  Post-ringing: %.2f dB", soxrResult.PostRingingDB)
			t.Logf("  Ringout samples: %d", soxrResult.RingoutSamples)
			t.Logf("")
			t.Logf("Go Resampler:")
			t.Logf("  Pre-ringing:  %.2f dB", goResult.PreRingingDB)
			t.Logf("  Post-ringing: %.2f dB", goResult.PostRingingDB)
			t.Logf("  Ringout samples: %d", goResult.RingoutSamples)
			t.Logf("")
			t.Logf("=== Comparison ===")

			// Pre-ringing: less negative = more pre-ringing (worse)
			preRingDiff := goResult.PreRingingDB - soxrResult.PreRingingDB
			t.Logf("Pre-ringing difference: %.2f dB (go - soxr, positive = go has more pre-ringing)", preRingDiff)

			if preRingDiff > 6 {
				t.Logf("WARNING: Go has significantly more pre-ringing than soxr")
			}
		})
	}
}

// =============================================================================
// TEST 5: Rational Ratio Quality (Non-integer ratios)
// =============================================================================

// TestRationalRatio_Quality specifically tests problematic non-integer ratios
func TestRationalRatio_Quality(t *testing.T) {
	// These ratios are challenging because they require rational resampling
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1k_to_48k_147:160"},
		{48000, 44100, "48k_to_44.1k_160:147"},
		{44100, 96000, "44.1k_to_96k_147:320"},
		{96000, 44100, "96k_to_44.1k_320:147"},
		{22050, 48000, "22.05k_to_48k"},
		{48000, 22050, "48k_to_22.05k"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test anti-aliasing/anti-imaging
			var result *AntiAliasingResult
			var err error

			if tc.outputRate > tc.inputRate {
				// Upsampling
				result, err = measureAntiAliasing(tc.inputRate, tc.outputRate, SignalNoise, QualityVeryHigh)
			} else {
				// Downsampling - use downsampling-specific measurement
				dsResult, dsErr := measureDownsamplingAntiAliasing(tc.inputRate, tc.outputRate, QualityVeryHigh)
				if dsErr != nil {
					t.Fatalf("measurement failed: %v", dsErr)
				}
				result = &AntiAliasingResult{
					InputRate:           tc.inputRate,
					OutputRate:          tc.outputRate,
					StopbandAttenuation: dsResult.Attenuation,
				}
				err = nil
			}

			if err != nil {
				t.Fatalf("measurement failed: %v", err)
			}

			t.Logf("=== Rational Ratio Quality Test ===")
			t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
			t.Logf("Stopband attenuation: %.2f dB", result.StopbandAttenuation)

			// Get THD
			thdResult, err := measureTHD(tc.inputRate, tc.outputRate, 1000, QualityVeryHigh)
			if err != nil {
				t.Logf("THD measurement failed: %v", err)
			} else {
				t.Logf("THD @ 1kHz: %.2f dB (%.6f%%)", thdResult.THD_DB, thdResult.THD_Percent)
			}

			// Get SNR
			snrResult, err := measureSNR(tc.inputRate, tc.outputRate, 1000, QualityVeryHigh)
			if err != nil {
				t.Logf("SNR measurement failed: %v", err)
			} else {
				t.Logf("SNR: %.2f dB", snrResult.SNR_DB)
			}

			// Minimum quality thresholds for any resampler
			if result.StopbandAttenuation < 60 {
				t.Errorf("QUALITY ISSUE: Stopband attenuation %.2f dB is too low for ratio %.0f:%.0f",
					result.StopbandAttenuation, tc.inputRate, tc.outputRate)
			}
		})
	}
}

// =============================================================================
// SOXR Reference Functions (try static data first, then C tool)
// =============================================================================

func getSoxrPassbandRipple(inputRate, outputRate float64) (*PassbandRippleResult, error) {
	// Try static data first
	if refData, err := loadStaticReferenceData(); err == nil {
		key := fmt.Sprintf("ripple_%.0f_%.0f", inputRate, outputRate)
		if raw, ok := refData.Quality[key]; ok {
			var data struct {
				Ripple float64 `json:"ripple"`
				MaxDev float64 `json:"max_dev"`
				MinDev float64 `json:"min_dev"`
			}
			if json.Unmarshal(raw, &data) == nil {
				return &PassbandRippleResult{
					InputRate:      inputRate,
					OutputRate:     outputRate,
					MaxDeviation:   data.MaxDev,
					MinDeviation:   data.MinDev,
					RipplePeakPeak: data.Ripple,
				}, nil
			}
		}
	}

	// Fall back to C tool
	result, err := runSoxrQualityTest(inputRate, outputRate, "ripple")
	if err != nil {
		return nil, err
	}

	return &PassbandRippleResult{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		MaxDeviation:   result["max_deviation"],
		MinDeviation:   result["min_deviation"],
		RipplePeakPeak: result["ripple"],
	}, nil
}

func getSoxrTHD(inputRate, outputRate, testFreq float64) (*THDResult, error) {
	// Try static data first
	if refData, err := loadStaticReferenceData(); err == nil {
		key := fmt.Sprintf("thd_%.0f_%.0f_%.0f", inputRate, outputRate, testFreq)
		if raw, ok := refData.Quality[key]; ok {
			var data struct {
				THD_DB      float64 `json:"thd_db"`
				THD_Percent float64 `json:"thd_percent"`
			}
			if json.Unmarshal(raw, &data) == nil {
				return &THDResult{
					InputRate:     inputRate,
					OutputRate:    outputRate,
					TestFrequency: testFreq,
					THD_DB:        data.THD_DB,
					THD_Percent:   data.THD_Percent,
				}, nil
			}
		}
	}

	// Fall back to C tool
	result, err := runSoxrQualityTest(inputRate, outputRate, fmt.Sprintf("thd:%.0f", testFreq))
	if err != nil {
		return nil, err
	}

	return &THDResult{
		InputRate:     inputRate,
		OutputRate:    outputRate,
		TestFrequency: testFreq,
		THD_DB:        result["thd_db"],
		THD_Percent:   result["thd_percent"],
	}, nil
}

func getSoxrSNR(inputRate, outputRate, testFreq float64) (*SNRResult, error) {
	// Try static data first
	if refData, err := loadStaticReferenceData(); err == nil {
		key := fmt.Sprintf("snr_%.0f_%.0f", inputRate, outputRate)
		if raw, ok := refData.Quality[key]; ok {
			var snr float64
			if json.Unmarshal(raw, &snr) == nil {
				return &SNRResult{
					InputRate:  inputRate,
					OutputRate: outputRate,
					SNR_DB:     snr,
				}, nil
			}
		}
	}

	// Fall back to C tool
	result, err := runSoxrQualityTest(inputRate, outputRate, fmt.Sprintf("snr:%.0f", testFreq))
	if err != nil {
		return nil, err
	}

	return &SNRResult{
		InputRate:  inputRate,
		OutputRate: outputRate,
		SNR_DB:     result["snr_db"],
	}, nil
}

func getSoxrImpulse(inputRate, outputRate float64) (*ImpulseResult, error) {
	// Try static data first
	if refData, err := loadStaticReferenceData(); err == nil {
		key := fmt.Sprintf("impulse_%.0f_%.0f", inputRate, outputRate)
		if raw, ok := refData.Quality[key]; ok {
			var data struct {
				PreRingingDB   float64 `json:"pre_ringing_db"`
				PostRingingDB  float64 `json:"post_ringing_db"`
				RingoutSamples int     `json:"ringout_samples"`
			}
			if json.Unmarshal(raw, &data) == nil {
				return &ImpulseResult{
					InputRate:      inputRate,
					OutputRate:     outputRate,
					PreRingingDB:   data.PreRingingDB,
					PostRingingDB:  data.PostRingingDB,
					RingoutSamples: data.RingoutSamples,
				}, nil
			}
		}
	}

	// Fall back to C tool
	result, err := runSoxrQualityTest(inputRate, outputRate, "impulse")
	if err != nil {
		return nil, err
	}

	return &ImpulseResult{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		PreRingingDB:   result["pre_ringing_db"],
		PostRingingDB:  result["post_ringing_db"],
		RingoutSamples: int(result["ringout_samples"]),
	}, nil
}

// runSoxrQualityTest runs the soxr quality test tool and parses output
func runSoxrQualityTest(inputRate, outputRate float64, testType string) (map[string]float64, error) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return nil, fmt.Errorf("failed to get current file path")
	}

	projectRoot := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))
	toolPath := filepath.Join(projectRoot, "test-reference", "test_quality")

	if _, err := os.Stat(toolPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("soxr quality tool not found at %s (run 'make test_quality' in test-reference/)", toolPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*1000*1000*1000) // 30 seconds
	defer cancel()

	cmd := exec.CommandContext(ctx, toolPath,
		fmt.Sprintf("%.0f", inputRate),
		fmt.Sprintf("%.0f", outputRate),
		testType)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run soxr quality tool: %w", err)
	}

	// Parse output (format: key=value lines starting with #)
	result := make(map[string]float64)
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "# ") && strings.Contains(line, "=") {
			line = strings.TrimPrefix(line, "# ")
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				valStr := strings.TrimSpace(parts[1])
				// Remove any units suffix
				valStr = strings.Fields(valStr)[0]
				val, err := strconv.ParseFloat(valStr, 64)
				if err == nil {
					result[key] = val
				}
			}
		}
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("no results parsed from soxr quality tool output")
	}

	return result, nil
}
