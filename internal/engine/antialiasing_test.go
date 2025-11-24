package engine

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

const (
	// Test signal parameters
	antialiasTestSamples = 32768 // Power of 2 for efficient FFT
	fftWindowSize        = 8192  // FFT window size

	// Minimum acceptable stopband attenuation
	minStopbandAttenuation = 80.0 // dB - should be much higher in practice
)

// SignalType for anti-aliasing tests
type SignalType int

const (
	SignalNoise SignalType = iota
	SignalMultitone
	SignalSweep
	SignalAliasTones // Tones in aliasing region for downsampling tests
)

func (s SignalType) String() string {
	switch s {
	case SignalNoise:
		return "noise"
	case SignalMultitone:
		return "multitone"
	case SignalSweep:
		return "sweep"
	case SignalAliasTones:
		return "alias_tones"
	default:
		return "unknown"
	}
}

// generateAntialiasTestSignal creates test signals for anti-aliasing measurement
//
//nolint:unparam // numSamples kept as parameter for test flexibility
func generateAntialiasTestSignal(signalType SignalType, sampleRate float64, numSamples int) []float64 {
	signal := make([]float64, numSamples)

	switch signalType {
	case SignalNoise:
		// Reproducible pseudo-random noise using LCG
		state := uint32(12345)
		for i := range signal {
			state = state*1103515245 + 12345
			signal[i] = float64(int32(state&0x7FFFFFFF))/float64(0x7FFFFFFF)*2.0 - 1.0
			signal[i] *= 0.5 // Scale down to avoid clipping
		}

	case SignalMultitone:
		// Multiple tones at various frequencies up to Nyquist
		freqs := []float64{1000, 2000, 4000, 8000, 12000, 16000, 20000, 22000, 23000}
		nyquist := sampleRate / 2.0

		for _, freq := range freqs {
			if freq < nyquist*0.95 { // Stay below Nyquist
				amplitude := 0.1
				for i := range signal {
					phase := 2.0 * math.Pi * freq * float64(i) / sampleRate
					signal[i] += amplitude * math.Sin(phase)
				}
			}
		}

	case SignalSweep:
		// Linear frequency sweep from 100 Hz to near-Nyquist
		fStart := 100.0
		fEnd := sampleRate * 0.45 // 90% of Nyquist
		duration := float64(numSamples) / sampleRate
		sweepRate := (fEnd - fStart) / duration

		for i := range signal {
			t := float64(i) / sampleRate
			phase := 2.0 * math.Pi * (fStart*t + sweepRate*t*t/2.0)
			signal[i] = 0.7 * math.Sin(phase)
		}

	case SignalAliasTones:
		// Tones specifically in the aliasing region for downsampling tests.
		// For 48kHz -> 32kHz: output Nyquist = 16kHz
		// Place tones at 17, 18, 19, 20, 21, 22, 23 kHz
		// These would alias to 1, 2, 3, 4, 5, 6, 7 kHz if not filtered
		outputNyquistEst := sampleRate / 3.0 // Estimate for 48->32
		nyquist := sampleRate / 2.0

		// Generate tones from output_nyquist to input_nyquist
		for freq := outputNyquistEst + 1000; freq < nyquist-500; freq += 1000 {
			amplitude := 0.1
			for i := range signal {
				phase := 2.0 * math.Pi * freq * float64(i) / sampleRate
				signal[i] += amplitude * math.Sin(phase)
			}
		}
	}

	return signal
}

// computePSD computes power spectral density using FFT (Welch's method)
//
//nolint:unparam // fftSize kept as parameter for test flexibility
func computePSD(signal []float64, sampleRate float64, fftSize int) (freqs, psdDB []float64) {
	numBins := fftSize/2 + 1
	freqs = make([]float64, numBins)
	psdAccum := make([]float64, numBins)

	for k := range numBins {
		freqs[k] = float64(k) * sampleRate / float64(fftSize)
	}

	// Hann window
	window := make([]float64, fftSize)
	for n := range fftSize {
		window[n] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(n)/float64(fftSize-1)))
	}

	// 50% overlap
	hopSize := fftSize / 2
	numWindows := 0

	for start := 0; start+fftSize <= len(signal); start += hopSize {
		// Apply window and compute FFT
		fftIn := make([]complex128, fftSize)
		for n := range fftSize {
			fftIn[n] = complex(signal[start+n]*window[n], 0)
		}

		fftOut := fft(fftIn)

		// Accumulate power spectrum
		for k := range numBins {
			power := cmplx.Abs(fftOut[k])
			psdAccum[k] += power * power
		}
		numWindows++
	}

	// Compute window power
	windowPower := 0.0
	for n := range fftSize {
		windowPower += window[n] * window[n]
	}

	// Average and convert to dB
	psdDB = make([]float64, numBins)
	for k := range numBins {
		power := psdAccum[k] / (float64(numWindows) * float64(fftSize) * windowPower)
		if power > 1e-20 {
			psdDB[k] = 10.0 * math.Log10(power)
		} else {
			psdDB[k] = -200.0
		}
	}

	return freqs, psdDB
}

// fft computes the FFT using the Cooley-Tukey algorithm
func fft(x []complex128) []complex128 {
	n := len(x)
	if n <= 1 {
		return x
	}

	// Check if n is power of 2
	if n&(n-1) != 0 {
		// Pad to next power of 2
		nextPow2 := 1
		for nextPow2 < n {
			nextPow2 <<= 1
		}
		padded := make([]complex128, nextPow2)
		copy(padded, x)
		x = padded
		n = nextPow2
	}

	// Bit-reversal permutation
	result := make([]complex128, n)
	bits := int(math.Log2(float64(n)))
	for i := range n {
		rev := 0
		for j := range bits {
			if i&(1<<j) != 0 {
				rev |= 1 << (bits - 1 - j)
			}
		}
		result[rev] = x[i]
	}

	// Cooley-Tukey iterative FFT
	for s := 1; s <= bits; s++ {
		m := 1 << s
		wm := cmplx.Exp(complex(0, -2*math.Pi/float64(m)))
		for k := 0; k < n; k += m {
			w := complex(1, 0)
			for j := 0; j < m/2; j++ {
				t := w * result[k+j+m/2]
				u := result[k+j]
				result[k+j] = u + t
				result[k+j+m/2] = u - t
				w *= wm
			}
		}
	}

	return result
}

// measureBandEnergy measures average energy in a frequency band
func measureBandEnergy(freqs, psdDB []float64, fLow, fHigh float64) float64 {
	totalPower := 0.0
	count := 0

	for k := range freqs {
		if freqs[k] >= fLow && freqs[k] < fHigh {
			totalPower += math.Pow(10.0, psdDB[k]/10.0)
			count++
		}
	}

	if count > 0 {
		return 10.0 * math.Log10(totalPower/float64(count))
	}
	return -200.0
}

// measurePeakEnergy measures peak energy in a frequency band (for discrete tones)
func measurePeakEnergy(freqs, psdDB []float64, fLow, fHigh float64) float64 {
	peak := -200.0

	for k := range freqs {
		if freqs[k] >= fLow && freqs[k] < fHigh {
			if psdDB[k] > peak {
				peak = psdDB[k]
			}
		}
	}

	return peak
}

// AntiAliasingResult holds test results
type AntiAliasingResult struct {
	InputRate           float64
	OutputRate          float64
	SignalType          SignalType
	PassbandEnergy      float64
	StopbandEnergy      float64
	StopbandAttenuation float64
	PSD                 []float64
	Frequencies         []float64
}

// measureAntiAliasing tests the anti-imaging filter effectiveness
func measureAntiAliasing(inputRate, outputRate float64, signalType SignalType, quality Quality) (*AntiAliasingResult, error) {
	// Generate test signal
	input := generateAntialiasTestSignal(signalType, inputRate, antialiasTestSamples)

	// Create resampler
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	// Process signal
	output, err := resampler.Process(input)
	if err != nil {
		return nil, fmt.Errorf("failed to process: %w", err)
	}

	// Flush remaining samples
	flush, err := resampler.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush: %w", err)
	}
	output = append(output, flush...)

	// Compute PSD of output
	freqs, psdDB := computePSD(output, outputRate, fftWindowSize)

	// Define frequency bands
	origNyquist := inputRate / 2.0
	newNyquist := outputRate / 2.0

	// Passband: DC to 90% of original Nyquist
	passbandEnd := origNyquist * 0.9

	// Stopband: above original Nyquist (the "imaging" region)
	// Use some margin to avoid transition band effects
	stopbandStart := origNyquist + 1000
	stopbandEnd := newNyquist - 1000

	// Measure energies based on signal type
	// For multitone: use peak-based measurement (discrete tones)
	// For noise/sweep: use average-based measurement (broadband)
	var passbandEnergy, stopbandEnergy float64
	if signalType == SignalMultitone {
		// Peak-based for discrete tones - avoids averaging dilution
		passbandEnergy = measurePeakEnergy(freqs, psdDB, 100, passbandEnd)
		stopbandEnergy = measurePeakEnergy(freqs, psdDB, stopbandStart, stopbandEnd)
	} else {
		// Average-based for broadband signals
		passbandEnergy = measureBandEnergy(freqs, psdDB, 100, passbandEnd)
		stopbandEnergy = measureBandEnergy(freqs, psdDB, stopbandStart, stopbandEnd)
	}

	attenuation := passbandEnergy - stopbandEnergy

	return &AntiAliasingResult{
		InputRate:           inputRate,
		OutputRate:          outputRate,
		SignalType:          signalType,
		PassbandEnergy:      passbandEnergy,
		StopbandEnergy:      stopbandEnergy,
		StopbandAttenuation: attenuation,
		PSD:                 psdDB,
		Frequencies:         freqs,
	}, nil
}

// getSoxrAntiAliasingResult runs the soxr reference tool.
// The tool path can be set via SOXR_ANTIALIAS_TOOL environment variable,
// otherwise it defaults to test-reference/test_antialiasing relative to the source file.
func getSoxrAntiAliasingResult(inputRate, outputRate float64, signalType SignalType) (*AntiAliasingResult, error) {
	toolPath := os.Getenv("SOXR_ANTIALIAS_TOOL")
	if toolPath == "" {
		// Default to relative path from this source file
		_, thisFile, _, ok := runtime.Caller(0)
		if !ok {
			return nil, fmt.Errorf("cannot determine source file path for soxr tool")
		}
		// Navigate from internal/engine/ to test-reference/
		toolPath = filepath.Join(filepath.Dir(thisFile), "..", "..", "test-reference", "test_antialiasing")
	}

	cmd := exec.CommandContext(context.Background(), toolPath,
		fmt.Sprintf("%.0f", inputRate),
		fmt.Sprintf("%.0f", outputRate),
		signalType.String())

	outputBytes, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("soxr reference failed: %w", err)
	}

	// Parse output
	result := &AntiAliasingResult{
		InputRate:  inputRate,
		OutputRate: outputRate,
		SignalType: signalType,
	}

	scanner := bufio.NewScanner(strings.NewReader(string(outputBytes)))
	inPSD := false
	var psdData []float64
	var freqData []float64

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "OUTPUT_PSD_START" {
			inPSD = true
			continue
		}
		if line == "OUTPUT_PSD_END" {
			inPSD = false
			continue
		}

		if inPSD {
			parts := strings.Split(line, ",")
			if len(parts) == 2 {
				freq, _ := strconv.ParseFloat(parts[0], 64)
				psd, _ := strconv.ParseFloat(parts[1], 64)
				freqData = append(freqData, freq)
				psdData = append(psdData, psd)
			}
			continue
		}

		// Parse summary lines
		if strings.HasPrefix(line, "#   Output passband energy:") {
			parts := strings.Fields(line)
			if len(parts) >= 5 {
				result.PassbandEnergy, _ = strconv.ParseFloat(parts[4], 64)
			}
		}
		if strings.HasPrefix(line, "#   Output stopband energy:") {
			parts := strings.Fields(line)
			if len(parts) >= 5 {
				result.StopbandEnergy, _ = strconv.ParseFloat(parts[4], 64)
			}
		}
		if strings.HasPrefix(line, "# STOPBAND ATTENUATION:") {
			parts := strings.Fields(line)
			if len(parts) >= 4 {
				result.StopbandAttenuation, _ = strconv.ParseFloat(parts[3], 64)
			}
		}
	}

	result.PSD = psdData
	result.Frequencies = freqData

	return result, nil
}

func TestAntiAliasing_Upsampling(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		signalType SignalType
	}{
		{48000, 96000, SignalNoise},
		{48000, 96000, SignalMultitone},
		{48000, 96000, SignalSweep},
		{44100, 88200, SignalNoise},
		{44100, 96000, SignalNoise},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%.0fto%.0f_%s", tc.inputRate, tc.outputRate, tc.signalType)
		t.Run(name, func(t *testing.T) {
			// Test our resampler
			result, err := measureAntiAliasing(tc.inputRate, tc.outputRate, tc.signalType, QualityHigh)
			if err != nil {
				t.Fatalf("measureAntiAliasing failed: %v", err)
			}

			t.Logf("Input rate:  %.0f Hz", tc.inputRate)
			t.Logf("Output rate: %.0f Hz", tc.outputRate)
			t.Logf("Signal type: %s", tc.signalType)
			t.Logf("Passband energy:  %.2f dB", result.PassbandEnergy)
			t.Logf("Stopband energy:  %.2f dB", result.StopbandEnergy)
			t.Logf("STOPBAND ATTENUATION: %.2f dB", result.StopbandAttenuation)

			if result.StopbandAttenuation < minStopbandAttenuation {
				t.Errorf("Stopband attenuation %.2f dB is below minimum %.2f dB",
					result.StopbandAttenuation, minStopbandAttenuation)
			}
		})
	}
}

func TestAntiAliasing_CompareWithSoxr(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		signalType SignalType
	}{
		{48000, 96000, SignalNoise},
		{48000, 96000, SignalMultitone},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%.0fto%.0f_%s", tc.inputRate, tc.outputRate, tc.signalType)
		t.Run(name, func(t *testing.T) {
			// Get soxr reference result
			soxrResult, err := getSoxrAntiAliasingResult(tc.inputRate, tc.outputRate, tc.signalType)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Test our resampler
			goResult, err := measureAntiAliasing(tc.inputRate, tc.outputRate, tc.signalType, QualityHigh)
			if err != nil {
				t.Fatalf("measureAntiAliasing failed: %v", err)
			}

			t.Logf("=== SOXR Reference ===")
			t.Logf("Passband energy:  %.2f dB", soxrResult.PassbandEnergy)
			t.Logf("Stopband energy:  %.2f dB", soxrResult.StopbandEnergy)
			t.Logf("Stopband attenuation: %.2f dB", soxrResult.StopbandAttenuation)

			t.Logf("")
			t.Logf("=== Go Resampler ===")
			t.Logf("Passband energy:  %.2f dB", goResult.PassbandEnergy)
			t.Logf("Stopband energy:  %.2f dB", goResult.StopbandEnergy)
			t.Logf("Stopband attenuation: %.2f dB", goResult.StopbandAttenuation)

			t.Logf("")
			t.Logf("=== Comparison ===")
			diff := soxrResult.StopbandAttenuation - goResult.StopbandAttenuation
			ratio := goResult.StopbandAttenuation / soxrResult.StopbandAttenuation * 100
			t.Logf("Attenuation difference: %.2f dB (soxr - go)", diff)
			t.Logf("Go achieves %.1f%% of soxr's attenuation", ratio)

			// Fail if Go is significantly worse than soxr (more than 60 dB gap)
			// This identifies real performance issues, not minor differences
			if diff > 60 {
				t.Errorf("PERFORMANCE GAP: Go is %.2f dB worse than soxr (%.2f vs %.2f)",
					diff, goResult.StopbandAttenuation, soxrResult.StopbandAttenuation)
			}

			// Also fail if Go doesn't meet absolute minimum
			if goResult.StopbandAttenuation < minStopbandAttenuation {
				t.Errorf("MINIMUM NOT MET: Go stopband attenuation %.2f dB is below minimum %.2f dB",
					goResult.StopbandAttenuation, minStopbandAttenuation)
			}
		})
	}
}

// TestAntiAliasing_DetailedPSD outputs detailed PSD data for analysis
func TestAntiAliasing_DetailedPSD(t *testing.T) {
	inputRate := 48000.0
	outputRate := 96000.0
	signalType := SignalNoise

	result, err := measureAntiAliasing(inputRate, outputRate, signalType, QualityHigh)
	if err != nil {
		t.Fatalf("measureAntiAliasing failed: %v", err)
	}

	// Print PSD at key frequencies
	origNyquist := inputRate / 2.0
	newNyquist := outputRate / 2.0

	t.Logf("Detailed PSD at key frequencies:")
	t.Logf("Original Nyquist: %.0f Hz", origNyquist)
	t.Logf("New Nyquist: %.0f Hz", newNyquist)
	t.Logf("")

	// Find PSD values near key frequencies
	keyFreqs := []float64{
		1000, 5000, 10000, 15000, 20000, // In passband
		origNyquist - 1000, origNyquist, origNyquist + 1000, // Around original Nyquist
		30000, 35000, 40000, 45000, // In stopband
	}

	for _, target := range keyFreqs {
		if target >= newNyquist {
			continue
		}
		// Find closest frequency bin
		closestIdx := 0
		closestDist := math.Abs(result.Frequencies[0] - target)
		for i, f := range result.Frequencies {
			dist := math.Abs(f - target)
			if dist < closestDist {
				closestDist = dist
				closestIdx = i
			}
		}
		t.Logf("  %.0f Hz: %.2f dB", result.Frequencies[closestIdx], result.PSD[closestIdx])
	}
}

// TestAntiAliasing_QualityLevels compares different quality settings
func TestAntiAliasing_QualityLevels(t *testing.T) {
	inputRate := 48000.0
	outputRate := 96000.0
	signalType := SignalNoise

	qualities := []struct {
		q    Quality
		name string
	}{
		{QualityLow, "Low"},
		{QualityMedium, "Medium"},
		{QualityHigh, "High"},
	}

	for _, qc := range qualities {
		t.Run(qc.name, func(t *testing.T) {
			result, err := measureAntiAliasing(inputRate, outputRate, signalType, qc.q)
			if err != nil {
				t.Fatalf("measureAntiAliasing failed: %v", err)
			}

			t.Logf("Quality: %s", qc.name)
			t.Logf("Passband energy:  %.2f dB", result.PassbandEnergy)
			t.Logf("Stopband energy:  %.2f dB", result.StopbandEnergy)
			t.Logf("STOPBAND ATTENUATION: %.2f dB", result.StopbandAttenuation)
		})
	}
}

// DownsamplingAntiAliasingResult holds downsampling test results
type DownsamplingAntiAliasingResult struct {
	InputRate             float64
	OutputRate            float64
	SignalType            SignalType
	InputAliasingPeak     float64 // Peak energy in aliasing region of input
	OutputAliasTargetPeak float64 // Peak energy where aliases would appear
	Attenuation           float64 // How much aliasing is suppressed
	InputPSD              []float64
	InputFreqs            []float64
	OutputPSD             []float64
	OutputFreqs           []float64
}

// generateAliasTones creates tones specifically in the aliasing region for downsampling tests
// The tones are placed between outputNyquist and inputNyquist
func generateAliasTones(inputRate, outputRate float64, numSamples int) []float64 {
	signal := make([]float64, numSamples)
	inputNyquist := inputRate / 2.0
	outputNyquist := outputRate / 2.0

	// Generate tones from output_nyquist+1000 to input_nyquist-500
	// These are the frequencies that would alias if not filtered
	for freq := outputNyquist + 1000; freq < inputNyquist-500; freq += 1000 {
		amplitude := 0.1
		for i := range signal {
			phase := 2.0 * math.Pi * freq * float64(i) / inputRate
			signal[i] += amplitude * math.Sin(phase)
		}
	}

	return signal
}

// measureDownsamplingAntiAliasing tests anti-aliasing filter effectiveness for downsampling
func measureDownsamplingAntiAliasing(inputRate, outputRate float64, quality Quality) (*DownsamplingAntiAliasingResult, error) {
	// Generate alias_tones signal - tones ONLY in the aliasing region
	// Use the proper function that knows the output rate
	input := generateAliasTones(inputRate, outputRate, antialiasTestSamples)

	// Create resampler
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	// Process signal
	output, err := resampler.Process(input)
	if err != nil {
		return nil, fmt.Errorf("failed to process: %w", err)
	}

	// Flush remaining samples
	flush, err := resampler.Flush()
	if err != nil {
		return nil, fmt.Errorf("failed to flush: %w", err)
	}
	output = append(output, flush...)

	// Compute PSD of input and output
	inputFreqs, inputPSD := computePSD(input, inputRate, fftWindowSize)
	outputFreqs, outputPSD := computePSD(output, outputRate, fftWindowSize)

	// Define frequency bands for downsampling
	origNyquist := inputRate / 2.0
	newNyquist := outputRate / 2.0

	// Aliasing region in input: output_nyquist to input_nyquist
	stopbandStart := newNyquist + 500
	stopbandEnd := origNyquist - 500

	// Measure peak energy in aliasing region of INPUT
	inputAliasingPeak := measurePeakEnergy(inputFreqs, inputPSD, stopbandStart, stopbandEnd)

	// Alias target region in output: 0 to (orig_nyquist - new_nyquist)
	// For 48->32: 16-24kHz aliases to 0-8kHz
	aliasRegionEnd := origNyquist - newNyquist
	outputAliasTargetPeak := measurePeakEnergy(outputFreqs, outputPSD, 100, aliasRegionEnd)

	// Attenuation = input peak - output peak
	attenuation := inputAliasingPeak - outputAliasTargetPeak

	return &DownsamplingAntiAliasingResult{
		InputRate:             inputRate,
		OutputRate:            outputRate,
		SignalType:            SignalAliasTones,
		InputAliasingPeak:     inputAliasingPeak,
		OutputAliasTargetPeak: outputAliasTargetPeak,
		Attenuation:           attenuation,
		InputPSD:              inputPSD,
		InputFreqs:            inputFreqs,
		OutputPSD:             outputPSD,
		OutputFreqs:           outputFreqs,
	}, nil
}

// TestAntiAliasing_Downsampling tests anti-aliasing for downsampling use cases
// This is critical for BirdNET v3 and Google Perch v2 which use 32kHz audio
func TestAntiAliasing_Downsampling(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{48000, 32000, "48kHz_to_32kHz_BirdNET_Perch"},
		{48000, 44100, "48kHz_to_44.1kHz_DAT_to_CD"},
		{96000, 48000, "96kHz_to_48kHz_2x_downsample"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := measureDownsamplingAntiAliasing(tc.inputRate, tc.outputRate, QualityVeryHigh)
			if err != nil {
				t.Fatalf("measureDownsamplingAntiAliasing failed: %v", err)
			}

			t.Logf("=== Downsampling Anti-Aliasing Test ===")
			t.Logf("Input rate:  %.0f Hz", tc.inputRate)
			t.Logf("Output rate: %.0f Hz", tc.outputRate)
			t.Logf("")
			t.Logf("Input aliasing region peak:   %.2f dB", result.InputAliasingPeak)
			t.Logf("Output alias target peak:     %.2f dB", result.OutputAliasTargetPeak)
			t.Logf("ANTI-ALIASING ATTENUATION:    %.2f dB", result.Attenuation)

			// Should have at least 80 dB attenuation
			if result.Attenuation < minStopbandAttenuation {
				t.Errorf("Anti-aliasing attenuation %.2f dB is below minimum %.2f dB",
					result.Attenuation, minStopbandAttenuation)
			}
		})
	}
}

// TestAntiAliasing_Downsampling_CompareWithSoxr validates Go resampler against soxr reference
func TestAntiAliasing_Downsampling_CompareWithSoxr(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{48000, 32000, "48kHz_to_32kHz_BirdNET_Perch"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Get soxr reference result
			soxrResult, err := getSoxrAntiAliasingResult(tc.inputRate, tc.outputRate, SignalAliasTones)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Test our resampler with VeryHigh quality to match soxr's SOXR_VHQ
			goResult, err := measureDownsamplingAntiAliasing(tc.inputRate, tc.outputRate, QualityVeryHigh)
			if err != nil {
				t.Fatalf("measureDownsamplingAntiAliasing failed: %v", err)
			}

			t.Logf("=== SOXR Reference ===")
			t.Logf("Stopband attenuation: %.2f dB", soxrResult.StopbandAttenuation)

			t.Logf("")
			t.Logf("=== Go Resampler ===")
			t.Logf("Input aliasing region peak:   %.2f dB", goResult.InputAliasingPeak)
			t.Logf("Output alias target peak:     %.2f dB", goResult.OutputAliasTargetPeak)
			t.Logf("Anti-aliasing attenuation:    %.2f dB", goResult.Attenuation)

			t.Logf("")
			t.Logf("=== Comparison ===")
			diff := soxrResult.StopbandAttenuation - goResult.Attenuation
			ratio := goResult.Attenuation / soxrResult.StopbandAttenuation * 100
			t.Logf("Attenuation difference: %.2f dB (soxr - go)", diff)
			t.Logf("Go achieves %.1f%% of soxr's attenuation", ratio)

			// Fail if Go is significantly worse than soxr (more than 60 dB gap)
			if diff > 60 {
				t.Errorf("PERFORMANCE GAP: Go is %.2f dB worse than soxr (%.2f vs %.2f)",
					diff, goResult.Attenuation, soxrResult.StopbandAttenuation)
			}

			// Also fail if Go doesn't meet absolute minimum
			if goResult.Attenuation < minStopbandAttenuation {
				t.Errorf("MINIMUM NOT MET: Go anti-aliasing attenuation %.2f dB is below minimum %.2f dB",
					goResult.Attenuation, minStopbandAttenuation)
			}
		})
	}
}
