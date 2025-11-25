package engine

import (
	"math"
	"testing"

	"github.com/tphakala/go-audio-resampler/internal/filter"
)

// TestDFTDecimationStage_FilterAnalysis analyzes the DFT decimation filter for 96→48
func TestDFTDecimationStage_FilterAnalysis(t *testing.T) {
	factor := 2
	quality := QualityVeryHigh

	// Create the decimation stage
	stage, err := NewDFTDecimationStage[float64](factor, quality)
	if err != nil {
		t.Fatalf("Failed to create DFT decimation stage: %v", err)
	}

	t.Logf("=== DFT Decimation Stage Analysis (factor=%d, quality=%d) ===", factor, quality)
	t.Logf("Number of taps: %d", stage.numTaps)

	// Get coefficients (they're stored reversed, so we need to un-reverse them)
	coeffs := make([]float64, stage.numTaps)
	for i, c := range stage.coeffs {
		coeffs[stage.numTaps-1-i] = float64(c)
	}

	// Compute frequency response
	resp := filter.ComputeFrequencyResponse(coeffs, 8192)

	// Calculate expected parameters
	const (
		soxrFp = 0.913
		soxrFs = 1.0
	)
	FpNorm := soxrFp / float64(factor)
	FsNorm := soxrFs / float64(factor)
	trBW := 0.5 * (FsNorm - FpNorm)
	Fc := FsNorm - trBW
	cutoff := Fc * 0.5 // In 0-0.5 range

	t.Logf("Design parameters:")
	t.Logf("  FpNorm=%.4f, FsNorm=%.4f, trBW=%.5f, Fc=%.5f, cutoff=%.5f",
		FpNorm, FsNorm, trBW, Fc, cutoff)

	// Find key metrics
	passbandEnd := cutoff - 0.005 // A bit before cutoff
	stopbandStart := cutoff + 0.02 // After transition band

	maxPassband := 0.0
	minPassband := 1000.0
	maxStopband := -300.0
	maxStopbandFreq := 0.0

	for i, f := range resp.Frequencies {
		mag := resp.Magnitude[i]
		magDB := 20 * math.Log10(mag + 1e-15)

		if f <= passbandEnd {
			if mag > maxPassband {
				maxPassband = mag
			}
			if mag < minPassband && mag > 0 {
				minPassband = mag
			}
		} else if f >= stopbandStart {
			if magDB > maxStopband {
				maxStopband = magDB
				maxStopbandFreq = f
			}
		}
	}

	rippleDB := 20 * math.Log10(maxPassband / minPassband)
	effectiveAtten := -maxStopband

	t.Logf("Filter response:")
	t.Logf("  Passband (f < %.3f): ripple = %.2f dB", passbandEnd, rippleDB)
	t.Logf("  Stopband (f > %.3f): max level = %.2f dB at f=%.4f", stopbandStart, maxStopband, maxStopbandFreq)
	t.Logf("  EFFECTIVE ATTENUATION: %.2f dB", effectiveAtten)

	// Also log some specific frequencies
	t.Logf("Frequency response at key points:")
	keyFreqs := []float64{0.1, 0.2, 0.23, 0.239, 0.245, 0.25, 0.26, 0.3, 0.4, 0.5}
	for _, f := range keyFreqs {
		// Find closest frequency in response
		idx := int(f / 0.5 * float64(len(resp.Frequencies)-1))
		if idx >= len(resp.Frequencies) {
			idx = len(resp.Frequencies) - 1
		}
		magDB := 20 * math.Log10(resp.Magnitude[idx] + 1e-15)
		freqHz := f * 96000 // At 96kHz input
		t.Logf("  f=%.3f (%.0f Hz @ 96kHz): %.2f dB", f, freqHz, magDB)
	}

	// Check requested attenuation
	attenuation := qualityToAttenuation(quality)
	t.Logf("\nRequested attenuation: %.2f dB", attenuation)
	t.Logf("Achieved attenuation: %.2f dB", effectiveAtten)
	if effectiveAtten < attenuation-10 {
		t.Errorf("Attenuation shortfall: requested %.2f dB, achieved %.2f dB", attenuation, effectiveAtten)
	}
}

// TestDFTDecimationStage_StopbandProcessing tests actual signal processing through the decimation stage
func TestDFTDecimationStage_StopbandProcessing(t *testing.T) {
	factor := 2
	inputRate := 96000.0

	stage, err := NewDFTDecimationStage[float64](factor, QualityVeryHigh)
	if err != nil {
		t.Fatalf("Failed to create stage: %v", err)
	}

	t.Logf("DFT stage: taps=%d", stage.numTaps)

	// Test with a stopband sinusoid at 30 kHz (above output Nyquist of 24 kHz)
	testFreq := 30000.0 // Hz
	duration := 0.1     // seconds
	numSamples := int(duration * inputRate)

	// Generate input signal
	input := make([]float64, numSamples)
	for i := range input {
		ti := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * testFreq * ti)
	}

	// Input amplitude
	inputRMS := 0.0
	for _, v := range input {
		inputRMS += v * v
	}
	inputRMS = math.Sqrt(inputRMS / float64(len(input)))
	inputDB := 20 * math.Log10(inputRMS)

	// Process through decimation stage
	output, err := stage.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	// Flush remaining samples
	flush, err := stage.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Input: %d samples, Output: %d samples", len(input), len(output))

	// Skip transient region (first and last numTaps/factor samples)
	skip := stage.numTaps/factor + 100
	if skip*2 >= len(output) {
		t.Fatalf("Output too short for analysis")
	}

	// Measure output RMS in steady state
	steadyOutput := output[skip : len(output)-skip]
	outputRMS := 0.0
	maxOutput := 0.0
	for _, v := range steadyOutput {
		outputRMS += v * v
		if math.Abs(v) > maxOutput {
			maxOutput = math.Abs(v)
		}
	}
	outputRMS = math.Sqrt(outputRMS / float64(len(steadyOutput)))
	outputDB := 20 * math.Log10(outputRMS + 1e-15)
	maxDB := 20 * math.Log10(maxOutput + 1e-15)

	attenuation := inputDB - outputDB

	t.Logf("Stopband sinusoid at %.0f Hz:", testFreq)
	t.Logf("  Input RMS:  %.6f (%.2f dB)", inputRMS, inputDB)
	t.Logf("  Output RMS: %.15f (%.2f dB)", outputRMS, outputDB)
	t.Logf("  Output max: %.15f (%.2f dB)", maxOutput, maxDB)
	t.Logf("  ATTENUATION: %.2f dB", attenuation)

	// Also test a passband signal at 10 kHz
	t.Logf("\nPassband test at 10 kHz:")
	stage.Reset()
	passbandFreq := 10000.0
	for i := range input {
		ti := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * passbandFreq * ti)
	}

	output, err = stage.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}
	flush, err = stage.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	output = append(output, flush...)

	steadyOutput = output[skip : len(output)-skip]
	outputRMS = 0.0
	maxOutput = 0.0
	for _, v := range steadyOutput {
		outputRMS += v * v
		if math.Abs(v) > maxOutput {
			maxOutput = math.Abs(v)
		}
	}
	outputRMS = math.Sqrt(outputRMS / float64(len(steadyOutput)))
	outputDB = 20 * math.Log10(outputRMS + 1e-15)
	maxDB = 20 * math.Log10(maxOutput + 1e-15)

	passbandGain := 20*math.Log10(outputRMS) - inputDB
	t.Logf("  Input RMS:  %.6f (%.2f dB)", inputRMS, inputDB)
	t.Logf("  Output RMS: %.6f (%.2f dB)", outputRMS, outputDB)
	t.Logf("  Output max: %.6f (%.2f dB)", maxOutput, maxDB)
	t.Logf("  PASSBAND GAIN: %.2f dB (should be ~0 dB)", passbandGain)
}

// TestFullResampler_96to48_Attenuation tests the full resampler for 96→48
func TestFullResampler_96to48_Attenuation(t *testing.T) {
	inputRate := 96000.0
	outputRate := 48000.0

	resampler, err := NewResampler[float64](inputRate, outputRate, QualityVeryHigh)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	t.Logf("Resampler created for %.0f → %.0f Hz", inputRate, outputRate)

	// Test with a stopband sinusoid at 30 kHz (above output Nyquist of 24 kHz)
	testFreq := 30000.0 // Hz
	duration := 0.1     // seconds
	numSamples := int(duration * inputRate)

	// Generate input signal
	input := make([]float64, numSamples)
	for i := range input {
		ti := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * testFreq * ti)
	}

	// Input amplitude
	inputRMS := 0.0
	for _, v := range input {
		inputRMS += v * v
	}
	inputRMS = math.Sqrt(inputRMS / float64(len(input)))
	inputDB := 20 * math.Log10(inputRMS)

	// Process through full resampler
	output, err := resampler.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	// Flush remaining samples
	flush, err := resampler.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Input: %d samples, Output: %d samples", len(input), len(output))

	// Skip transient region
	skip := 1000
	if skip*2 >= len(output) {
		skip = len(output) / 4
	}

	// Measure output RMS in steady state
	steadyOutput := output[skip : len(output)-skip]
	outputRMS := 0.0
	maxOutput := 0.0
	for _, v := range steadyOutput {
		outputRMS += v * v
		if math.Abs(v) > maxOutput {
			maxOutput = math.Abs(v)
		}
	}
	outputRMS = math.Sqrt(outputRMS / float64(len(steadyOutput)))
	outputDB := 20 * math.Log10(outputRMS + 1e-15)
	maxDB := 20 * math.Log10(maxOutput + 1e-15)

	attenuation := inputDB - outputDB

	t.Logf("Stopband sinusoid at %.0f Hz:", testFreq)
	t.Logf("  Input RMS:  %.6f (%.2f dB)", inputRMS, inputDB)
	t.Logf("  Output RMS: %.15f (%.2f dB)", outputRMS, outputDB)
	t.Logf("  Output max: %.15f (%.2f dB)", maxOutput, maxDB)
	t.Logf("  FULL RESAMPLER ATTENUATION: %.2f dB", attenuation)

	// Also test passband
	t.Logf("\nPassband test at 10 kHz:")
	resampler, _ = NewResampler[float64](inputRate, outputRate, QualityVeryHigh)
	passbandFreq := 10000.0
	for i := range input {
		ti := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * passbandFreq * ti)
	}

	output, _ = resampler.Process(input)
	flush, _ = resampler.Flush()
	output = append(output, flush...)

	steadyOutput = output[skip : len(output)-skip]
	outputRMS = 0.0
	for _, v := range steadyOutput {
		outputRMS += v * v
	}
	outputRMS = math.Sqrt(outputRMS / float64(len(steadyOutput)))
	outputDB = 20 * math.Log10(outputRMS + 1e-15)

	passbandGain := outputDB - inputDB
	t.Logf("  Passband gain: %.2f dB (should be ~0 dB)", passbandGain)
}

// TestFullResampler_96to48_MultiTone tests multi-tone attenuation like anti-aliasing test
func TestFullResampler_96to48_MultiTone(t *testing.T) {
	inputRate := 96000.0
	outputRate := 48000.0
	numSamples := 32768

	resampler, err := NewResampler[float64](inputRate, outputRate, QualityVeryHigh)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	// Generate multi-tone signal like generateAliasTones
	input := make([]float64, numSamples)
	inputNyquist := inputRate / 2.0
	outputNyquist := outputRate / 2.0

	// Same as generateAliasTones: tones from outputNyquist+1000 to inputNyquist-500
	numTones := 0
	for freq := outputNyquist + 1000; freq < inputNyquist-500; freq += 1000 {
		amplitude := 0.1
		for i := range input {
			phase := 2.0 * math.Pi * freq * float64(i) / inputRate
			input[i] += amplitude * math.Sin(phase)
		}
		numTones++
	}
	t.Logf("Generated %d alias tones from %.0f Hz to %.0f Hz", numTones, outputNyquist+1000, inputNyquist-500)

	// Measure input RMS
	inputRMS := 0.0
	for _, v := range input {
		inputRMS += v * v
	}
	inputRMS = math.Sqrt(inputRMS / float64(len(input)))
	inputDB := 20 * math.Log10(inputRMS)
	t.Logf("Input RMS: %.6f (%.2f dB)", inputRMS, inputDB)

	// Process
	output, err := resampler.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}
	flush, err := resampler.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	output = append(output, flush...)

	// Skip transient region
	skip := 1000
	if skip*2 >= len(output) {
		skip = len(output) / 4
	}
	steadyOutput := output[skip : len(output)-skip]

	// Measure output RMS
	outputRMS := 0.0
	maxOutput := 0.0
	for _, v := range steadyOutput {
		outputRMS += v * v
		if math.Abs(v) > maxOutput {
			maxOutput = math.Abs(v)
		}
	}
	outputRMS = math.Sqrt(outputRMS / float64(len(steadyOutput)))
	outputDB := 20 * math.Log10(outputRMS + 1e-15)
	maxDB := 20 * math.Log10(maxOutput + 1e-15)

	attenuation := inputDB - outputDB

	t.Logf("Multi-tone (alias tones) attenuation:")
	t.Logf("  Input RMS:  %.6f (%.2f dB)", inputRMS, inputDB)
	t.Logf("  Output RMS: %.15f (%.2f dB)", outputRMS, outputDB)
	t.Logf("  Output max: %.15f (%.2f dB)", maxOutput, maxDB)
	t.Logf("  MULTI-TONE ATTENUATION: %.2f dB", attenuation)

	// Check if there's energy at specific frequencies
	t.Logf("\nSample output values around max:")
	for i, v := range steadyOutput {
		if math.Abs(v) > maxOutput*0.9 {
			t.Logf("  output[%d] = %.15e", i+skip, v)
		}
	}

	// Measure in different frequency bands using simple DFT at specific frequencies
	t.Logf("\nChecking specific frequency bands in output:")
	for testFreq := 1000.0; testFreq <= 23000.0; testFreq += 2000 {
		// Measure energy at this frequency using correlation
		sumCos := 0.0
		sumSin := 0.0
		for i, v := range steadyOutput {
			phase := 2.0 * math.Pi * testFreq * float64(i) / outputRate
			sumCos += v * math.Cos(phase)
			sumSin += v * math.Sin(phase)
		}
		amp := math.Sqrt(sumCos*sumCos+sumSin*sumSin) / float64(len(steadyOutput)) * 2
		ampDB := 20 * math.Log10(amp + 1e-15)
		t.Logf("  f=%.0f Hz: amplitude %.2e (%.2f dB)", testFreq, amp, ampDB)
	}
}

// TestPolyphaseFilterParams_48to44_Diagnostic shows filter parameters for 48→44.1
func TestPolyphaseFilterParams_48to44_Diagnostic(t *testing.T) {
	// Parameters for 48kHz → 44.1kHz with 2x pre-stage (matching soxr architecture)
	// Architecture: 48 → 96 (2x DFT upsample) → 44.1 (polyphase)
	inputRate := 48000.0
	outputRate := 44100.0
	intermediateRate := inputRate * 2.0 // 96 kHz after 2x pre-stage
	attenuation := 174.58 // VeryHigh

	// Polyphase ratio is from intermediate to output
	polyphaseRatio := outputRate / intermediateRate // 44.1/96 = 0.459375
	totalIORatio := inputRate / outputRate // ~1.088 (original input/output)
	passbandEnd := 0.913 // VeryHigh quality passband end

	// GCD-based numPhases for polyphase stage
	// 44100/96000 reduces to 147/320, so numPhases = 147
	numPhases := 147

	// hasPreStage = false because soxr uses preM=0 for upsampling pre-stage
	// (the polyphase doesn't "see" the pre-stage in soxr's filter calculation)
	params := ComputePolyphaseFilterParams(numPhases, polyphaseRatio, totalIORatio, false, attenuation, passbandEnd)

	t.Logf("=== Filter Parameters for 48kHz → 44.1kHz (VeryHigh) with 2x Pre-Stage ===")
	t.Logf("Input rate: %.0f Hz", inputRate)
	t.Logf("Intermediate rate: %.0f Hz (after 2x pre-stage)", intermediateRate)
	t.Logf("Output rate: %.0f Hz", outputRate)
	t.Logf("Output Nyquist: %.0f Hz", outputRate/2)
	t.Logf("")
	t.Logf("NumPhases: %d", params.NumPhases)
	t.Logf("Polyphase Ratio: %.6f (output/intermediate)", params.Ratio)
	t.Logf("TotalIORatio: %.6f (original input/output)", params.TotalIORatio)
	t.Logf("IsUpsampling: %v", params.IsUpsampling)
	t.Logf("HasPreStage: %v", params.HasPreStage)
	t.Logf("")
	t.Logf("Mult: %.6f", params.Mult)
	t.Logf("Fn: %.6f", params.Fn)
	t.Logf("Fp1: %.6f (should be ~0.419 like soxr)", params.Fp1)
	t.Logf("Fs1: %.6f (should be ~0.459 like soxr)", params.Fs1)
	t.Logf("FpRaw: %.6f", params.FpRaw)
	t.Logf("FsRaw: %.6f", params.FsRaw)
	t.Logf("Fp (normalized): %.6f", params.Fp)
	t.Logf("Fs (normalized): %.6f", params.Fs)
	t.Logf("")
	t.Logf("TrBw: %.6f", params.TrBw)
	t.Logf("Fc: %.6f", params.Fc)
	t.Logf("TapsPerPhase: %d", params.TapsPerPhase)
	t.Logf("TotalTaps: %d", params.TotalTaps)
	t.Logf("")

	// Convert Fc to actual frequency
	// For polyphase stage operating at intermediate rate, cutoff is relative to that rate
	actualCutoffHz := params.Fc * intermediateRate
	t.Logf("Actual filter cutoff: %.0f Hz (relative to intermediate rate)", actualCutoffHz)
	t.Logf("Expected cutoff (output Nyquist * 0.9): %.0f Hz", outputRate/2*0.9)

	// soxr trace shows: Fp1=0.4197509907 Fs1=0.4593750000
	// Check if our Fp1 matches soxr
	soxrFp1 := 0.4197509907
	if math.Abs(params.Fp1-soxrFp1) > 0.01 {
		t.Logf("WARNING: Fp1 differs from soxr (got %.4f, soxr=%.4f)", params.Fp1, soxrFp1)
	} else {
		t.Logf("GOOD: Fp1 matches soxr trace (%.4f ≈ %.4f)", params.Fp1, soxrFp1)
	}

	// Check if cutoff is reasonable (should be around 20 kHz for 22.05 kHz output Nyquist)
	expectedCutoff := outputRate / 2 * 0.9 // 90% of output Nyquist = 19845 Hz
	if actualCutoffHz < expectedCutoff*0.5 || actualCutoffHz > expectedCutoff*1.5 {
		t.Errorf("Filter cutoff %.0f Hz is outside expected range (~%.0f Hz)", actualCutoffHz, expectedCutoff)
	}
}

// TestPassbandRipple_48to44_Diagnostic diagnoses the passband ripple issue for 48→44.1
func TestPassbandRipple_48to44_Diagnostic(t *testing.T) {
	inputRate := 48000.0
	outputRate := 44100.0

	resampler, err := NewResampler[float64](inputRate, outputRate, QualityVeryHigh)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	// Test passband frequencies (should all have ~0 dB gain)
	// Output Nyquist is 22.05 kHz, so test up to ~20 kHz
	testFreqs := []float64{1000, 2000, 5000, 8000, 10000, 12000, 15000, 18000, 20000}

	t.Logf("=== Passband Gain Analysis for 48kHz → 44.1kHz ===")
	t.Logf("Output Nyquist: %.0f Hz", outputRate/2)
	t.Logf("")

	maxRipple := 0.0
	minGain := 0.0
	maxGain := 0.0

	for _, testFreq := range testFreqs {
		// Create fresh resampler for each test
		resampler, _ = NewResampler[float64](inputRate, outputRate, QualityVeryHigh)

		// Generate test sine wave
		duration := 0.1
		numSamples := int(duration * inputRate)
		input := make([]float64, numSamples)
		for i := range input {
			ti := float64(i) / inputRate
			input[i] = math.Sin(2 * math.Pi * testFreq * ti)
		}

		// Process
		output, err := resampler.Process(input)
		if err != nil {
			t.Fatalf("Process failed: %v", err)
		}
		flush, _ := resampler.Flush()
		output = append(output, flush...)

		// Skip transients
		skip := 500
		if skip*2 >= len(output) {
			skip = len(output) / 4
		}
		steadyOutput := output[skip : len(output)-skip]

		// Measure amplitude
		maxAmp := 0.0
		for _, v := range steadyOutput {
			if math.Abs(v) > maxAmp {
				maxAmp = math.Abs(v)
			}
		}

		gainDB := 20 * math.Log10(maxAmp)
		t.Logf("  f=%5.0f Hz: amplitude=%.4f, gain=%.2f dB", testFreq, maxAmp, gainDB)

		if gainDB > maxGain {
			maxGain = gainDB
		}
		if gainDB < minGain {
			minGain = gainDB
		}
	}

	ripple := maxGain - minGain
	if ripple > maxRipple {
		maxRipple = ripple
	}

	t.Logf("")
	t.Logf("Min gain: %.2f dB, Max gain: %.2f dB", minGain, maxGain)
	t.Logf("PASSBAND RIPPLE: %.2f dB", ripple)

	if ripple > 2.0 {
		t.Errorf("Passband ripple %.2f dB exceeds 2.0 dB threshold", ripple)
	}
}

// TestDFTDecimationStage_BufferIntegrity tests that Process returns independent buffers
func TestDFTDecimationStage_BufferIntegrity(t *testing.T) {
	factor := 2
	inputRate := 96000.0

	stage, err := NewDFTDecimationStage[float64](factor, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create stage: %v", err)
	}

	// Generate first input signal (1kHz sine)
	input1 := make([]float64, 4000)
	for i := range input1 {
		ti := float64(i) / inputRate
		input1[i] = math.Sin(2 * math.Pi * 1000 * ti)
	}

	// Process first input
	output1, err := stage.Process(input1)
	if err != nil {
		t.Fatalf("First Process failed: %v", err)
	}
	t.Logf("First output: %d samples", len(output1))

	// Save first few values
	saved := make([]float64, min(10, len(output1)))
	copy(saved, output1)
	t.Logf("Saved output1[0:10]: %v", saved)

	// Generate second input signal (different - 500Hz cosine)
	input2 := make([]float64, 2000)
	for i := range input2 {
		ti := float64(i) / inputRate
		input2[i] = math.Cos(2 * math.Pi * 500 * ti)
	}

	// Process second input
	output2, err := stage.Process(input2)
	if err != nil {
		t.Fatalf("Second Process failed: %v", err)
	}
	t.Logf("Second output: %d samples", len(output2))
	t.Logf("output2[0:10]: %v", output2[:min(10, len(output2))])

	// Check if output1 was corrupted
	t.Logf("output1[0:10] after second Process: %v", output1[:min(10, len(output1))])

	corrupted := false
	for i := range saved {
		if i < len(output1) && saved[i] != output1[i] {
			t.Errorf("output1[%d] corrupted: saved=%v, now=%v", i, saved[i], output1[i])
			corrupted = true
		}
	}

	if corrupted {
		t.Error("BUFFER INTEGRITY ISSUE: Process() returns slice of internal buffer, not a copy")
	} else {
		t.Log("Buffer integrity OK")
	}
}

// TestDFTDecimationStage_CompareQualities compares filter responses across quality levels
func TestDFTDecimationStage_CompareQualities(t *testing.T) {
	factor := 2
	qualities := []struct {
		q    Quality
		name string
	}{
		{QualityLow, "Low"},
		{QualityMedium, "Medium"},
		{QualityHigh, "High"},
		{QualityVeryHigh, "VeryHigh"},
	}

	for _, qc := range qualities {
		t.Run(qc.name, func(t *testing.T) {
			stage, err := NewDFTDecimationStage[float64](factor, qc.q)
			if err != nil {
				t.Fatalf("Failed to create stage: %v", err)
			}

			attenuation := qualityToAttenuation(qc.q)
			t.Logf("%s: taps=%d, requested_attenuation=%.2f dB",
				qc.name, stage.numTaps, attenuation)

			// Quick attenuation check
			coeffs := make([]float64, stage.numTaps)
			for i, c := range stage.coeffs {
				coeffs[stage.numTaps-1-i] = float64(c)
			}
			resp := filter.ComputeFrequencyResponse(coeffs, 4096)

			// Find max stopband level
			maxStopband := -300.0
			for i, f := range resp.Frequencies {
				if f > 0.26 { // After transition band
					magDB := 20 * math.Log10(resp.Magnitude[i] + 1e-15)
					if magDB > maxStopband {
						maxStopband = magDB
					}
				}
			}

			t.Logf("%s: achieved_attenuation=%.2f dB", qc.name, -maxStopband)
		})
	}
}

// TestLsxInvFResp_Debug shows what lsxInvFResp returns for different attenuations
func TestLsxInvFResp_Debug(t *testing.T) {
	drop := -0.01 // rolloff setting
	
	attenuations := []float64{50, 80, 102, 126, 160, 175}
	for _, att := range attenuations {
		result := lsxInvFResp(drop, att)
		t.Logf("lsxInvFResp(%.2f, %.1f dB) = %.6f", drop, att, result)
	}
	
	// Show what happens to FpRaw adjustment for 48→44.1
	t.Log("\n=== FpRaw adjustment for 48→44.1 ===")
	fpRaw := 0.4547
	fsRaw := 1.5136
	
	for _, att := range []float64{102, 126, 175} {
		invF := lsxInvFResp(drop, att)
		adjustedFp := fsRaw - (fsRaw-fpRaw)/(1.0-invF)
		t.Logf("att=%.0f dB: invFResp=%.4f, adjustedFp=%.4f (original=%.4f, reduction=%.1f%%)", 
			att, invF, adjustedFp, fpRaw, (1-adjustedFp/fpRaw)*100)
	}
}
