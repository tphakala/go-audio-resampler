package engine

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestDFTStage_THD_Isolation tests the DFT stage in isolation for THD
// This helps determine if THD issues come from DFT stage or polyphase stage
func TestDFTStage_THD_Isolation(t *testing.T) {
	testFreq := 1000.0
	inputRate := 48000.0
	outputRate := 96000.0 // 2x upsample, DFT stage only
	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase) // 0.9 amplitude, plenty of headroom
	}

	// Test DFT stage directly
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	output, err := dftStage.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}

	flush, err := dftStage.Flush()
	if err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Input samples: %d, Output samples: %d", len(input), len(output))
	t.Logf("Expected ratio: 2.0, Actual: %.4f", float64(len(output))/float64(len(input)))

	// Check for clipping/saturation
	maxAbs := 0.0
	minVal := 0.0
	maxVal := 0.0
	for _, v := range output {
		if v > maxVal {
			maxVal = v
		}
		if v < minVal {
			minVal = v
		}
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	t.Logf("Output range: [%.6f, %.6f], max abs: %.6f", minVal, maxVal, maxAbs)

	if maxAbs > 1.0 {
		t.Logf("WARNING: Output exceeds ±1.0 - potential clipping issue")
	}

	// Compute THD using FFT
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

	t.Logf("Fundamental @ %.0f Hz (bin %d): %.2f dB", testFreq, fundamentalBin, fundamentalDB)

	// Find harmonics
	nyquist := outputRate / 2.0
	var harmonicPowerSum float64

	t.Logf("Harmonic analysis:")
	for h := 2; h <= 10; h++ {
		harmFreq := testFreq * float64(h)
		if harmFreq >= nyquist {
			break
		}

		harmBin := int(harmFreq / outputRate * float64(fftSize))
		if harmBin < len(fftOut)/2 {
			harmMag := cmplx.Abs(fftOut[harmBin])
			harmDB := 20 * math.Log10(harmMag+1e-20)
			relDB := harmDB - fundamentalDB

			t.Logf("  %dth harmonic @ %.0f Hz (bin %d): %.2f dB (%.2f dB relative)",
				h, harmFreq, harmBin, harmDB, relDB)

			harmonicPowerSum += harmMag * harmMag
		}
	}

	// THD calculation
	thdRatio := math.Sqrt(harmonicPowerSum) / (fundamentalMag + 1e-20)
	thdDB := 20 * math.Log10(thdRatio+1e-20)
	t.Logf("")
	t.Logf("DFT Stage THD: %.2f dB (%.6f%%)", thdDB, thdRatio*100)

	// Compare with soxr reference for this ratio
	t.Logf("soxr reference: -142.80 dB")
	t.Logf("Gap: %.2f dB", thdDB-(-142.80))
}

// TestDFTStage_CoefficientAnalysis analyzes the DFT filter coefficients
func TestDFTStage_CoefficientAnalysis(t *testing.T) {
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	t.Logf("=== DFT Stage Coefficient Analysis ===")
	t.Logf("Factor: %d", dftStage.factor)
	t.Logf("Taps per phase: %d", dftStage.tapsPerPhase)
	t.Logf("Is half-band: %v", dftStage.isHalfBand)
	t.Logf("Phase 0 tap offset: %d", dftStage.phase0TapOffset)
	t.Logf("Phase 0 tap scale: %.10f", dftStage.phase0TapScale)

	// Analyze each phase's DC gain (sum of coefficients)
	for phase := 0; phase < dftStage.factor; phase++ {
		sum := 0.0
		for _, c := range dftStage.polyCoeffs[phase] {
			sum += c
		}
		t.Logf("Phase %d DC gain (sum): %.10f", phase, sum)

		// Find min/max coefficient
		minC, maxC := dftStage.polyCoeffs[phase][0], dftStage.polyCoeffs[phase][0]
		for _, c := range dftStage.polyCoeffs[phase] {
			if c < minC {
				minC = c
			}
			if c > maxC {
				maxC = c
			}
		}
		t.Logf("Phase %d coefficient range: [%.6f, %.6f]", phase, minC, maxC)

		// Count significant coefficients
		significant := 0
		for _, c := range dftStage.polyCoeffs[phase] {
			if math.Abs(c) > 1e-8 {
				significant++
			}
		}
		t.Logf("Phase %d significant coefficients: %d/%d", phase, significant, dftStage.tapsPerPhase)
	}

	// For half-band, Phase 0 should have only 1 significant coefficient ≈ 1.0
	// Phase 1 should have sum ≈ 1.0
	t.Logf("")
	t.Logf("Expected for half-band 2x upsample:")
	t.Logf("  Phase 0: 1 significant tap ≈ 1.0 (passthrough)")
	t.Logf("  Phase 1: sum ≈ 1.0 (interpolation)")
}

// TestDFTStage_HalfBandDisabled tests with half-band optimization disabled
func TestDFTStage_HalfBandDisabled(t *testing.T) {
	testFreq := 1000.0
	inputRate := 48000.0
	outputRate := 96000.0
	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Create DFT stage and force disable half-band
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Force disable half-band optimization to test if it's the cause
	wasHalfBand := dftStage.isHalfBand
	dftStage.isHalfBand = false

	output, err := dftStage.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}

	flush, err := dftStage.Flush()
	if err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Half-band was: %v, now disabled", wasHalfBand)

	// Compute THD
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

	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])

	nyquist := outputRate / 2.0
	var harmonicPowerSum float64

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
	thdDB := 20 * math.Log10(thdRatio+1e-20)

	t.Logf("THD with half-band DISABLED: %.2f dB", thdDB)
	t.Logf("If this is significantly different, half-band optimization is the issue")
}

// TestPolyphaseStage_THD_Isolation tests polyphase stage in isolation
func TestPolyphaseStage_THD_Isolation(t *testing.T) {
	// Test a simple polyphase ratio without DFT pre-stage
	testFreq := 1000.0
	inputRate := 48000.0
	ratio := 1.0884 // Approximate ratio for 48k→52.2k (testing polyphase only)
	outputRate := inputRate * ratio
	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Create polyphase stage directly (normally used after DFT pre-stage)
	totalIORatio := 1.0 / ratio
	polyStage, err := NewPolyphaseStage[float64](ratio, totalIORatio, true, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create polyphase stage: %v", err)
	}

	output, err := polyStage.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}

	flush, err := polyStage.Flush()
	if err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}
	output = append(output, flush...)

	t.Logf("Polyphase stage ratio: %.4f", ratio)
	t.Logf("Input samples: %d, Output samples: %d", len(input), len(output))
	t.Logf("Actual ratio: %.4f", float64(len(output))/float64(len(input)))

	// Check amplitude
	maxAbs := 0.0
	for _, v := range output {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	t.Logf("Max output amplitude: %.6f", maxAbs)

	// Compute THD
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

	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])
	fundamentalDB := 20 * math.Log10(fundamentalMag+1e-20)

	t.Logf("Fundamental: %.2f dB", fundamentalDB)

	nyquist := outputRate / 2.0
	var harmonicPowerSum float64

	for h := 2; h <= 10; h++ {
		harmFreq := testFreq * float64(h)
		if harmFreq >= nyquist {
			break
		}

		harmBin := int(harmFreq / outputRate * float64(fftSize))
		if harmBin < len(fftOut)/2 {
			harmMag := cmplx.Abs(fftOut[harmBin])
			harmDB := 20 * math.Log10(harmMag+1e-20)
			relDB := harmDB - fundamentalDB
			harmonicPowerSum += harmMag * harmMag

			if h <= 4 {
				t.Logf("  %dth harmonic: %.2f dB relative", h, relDB)
			}
		}
	}

	thdRatio := math.Sqrt(harmonicPowerSum) / (fundamentalMag + 1e-20)
	thdDB := 20 * math.Log10(thdRatio+1e-20)

	t.Logf("Polyphase Stage THD: %.2f dB", thdDB)
}

// TestDFTStage_FilterResponse tests the DFT stage filter frequency response
func TestDFTStage_FilterResponse(t *testing.T) {
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	t.Logf("=== DFT Stage Filter Response Analysis ===")
	t.Logf("Factor: %d, Taps per phase: %d", dftStage.factor, dftStage.tapsPerPhase)

	// Reconstruct prototype filter by interleaving phases
	totalTaps := dftStage.tapsPerPhase * dftStage.factor
	prototype := make([]float64, totalTaps)

	// Coefficients are stored reversed, so we need to un-reverse them
	for phase := 0; phase < dftStage.factor; phase++ {
		for tap := 0; tap < dftStage.tapsPerPhase; tap++ {
			// polyCoeffs[phase][tapsPerPhase-1-tap] = prototype[tap*factor + phase] * factor
			// So: prototype[tap*factor + phase] = polyCoeffs[phase][tapsPerPhase-1-tap] / factor
			protoIdx := tap*dftStage.factor + phase
			if protoIdx < totalTaps {
				// Un-reverse: coeffs were stored as polyCoeffs[phase][tapsPerPhase-1-tap]
				prototype[protoIdx] = dftStage.polyCoeffs[phase][dftStage.tapsPerPhase-1-tap] / float64(dftStage.factor)
			}
		}
	}

	// Compute frequency response at key points
	testFreqs := []float64{0.0, 0.1, 0.2, 0.24, 0.25, 0.26, 0.3, 0.4, 0.49}
	t.Logf("\nFrequency Response (0.5 = output Nyquist, 0.25 = original Nyquist):")
	t.Logf("Freq(norm)\tFreq(Hz@96k)\tMag(dB)")

	for _, freq := range testFreqs {
		// Compute H(e^jω) at this frequency
		omega := 2.0 * math.Pi * freq
		var realPart, imagPart float64

		for n, h := range prototype {
			angle := omega * float64(n)
			realPart += h * math.Cos(angle)
			imagPart -= h * math.Sin(angle)
		}

		mag := math.Sqrt(realPart*realPart + imagPart*imagPart)
		magDB := 20 * math.Log10(mag+1e-20)

		freqHz := freq * 96000 // For 48k→96k
		t.Logf("%.4f\t\t%.0f Hz\t\t%.2f dB", freq, freqHz, magDB)
	}

	// Check passband flatness (should be 0 dB from DC to 0.25)
	t.Logf("\nPassband check (DC to original Nyquist = 0.25):")
	passbandDeviation := 0.0
	for f := 0.01; f <= 0.24; f += 0.01 {
		omega := 2.0 * math.Pi * f
		var realPart, imagPart float64
		for n, h := range prototype {
			angle := omega * float64(n)
			realPart += h * math.Cos(angle)
			imagPart -= h * math.Sin(angle)
		}
		mag := math.Sqrt(realPart*realPart + imagPart*imagPart)
		magDB := 20 * math.Log10(mag+1e-20)
		if math.Abs(magDB) > math.Abs(passbandDeviation) {
			passbandDeviation = magDB
		}
	}
	t.Logf("Max passband deviation from 0 dB: %.4f dB", passbandDeviation)

	// Check DC gain
	dcSum := 0.0
	for _, h := range prototype {
		dcSum += h
	}
	t.Logf("Prototype DC gain: %.6f (should be ~1.0)", dcSum)

	// Check phase gains
	t.Logf("\nPer-phase DC gain (should each be ~1.0):")
	for phase := 0; phase < dftStage.factor; phase++ {
		phaseSum := 0.0
		for _, c := range dftStage.polyCoeffs[phase] {
			phaseSum += c
		}
		t.Logf("  Phase %d: %.6f", phase, phaseSum)
	}
}

// TestDFTStage_SimpleVerification verifies DFT stage with DC and low-freq signals
func TestDFTStage_SimpleVerification(t *testing.T) {
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Test 1: DC signal (all 1.0) - should remain all 1.0
	t.Log("Test 1: DC signal")
	dcInput := make([]float64, 1000)
	for i := range dcInput {
		dcInput[i] = 1.0
	}

	dcOutput, _ := dftStage.Process(dcInput)
	dcFlush, _ := dftStage.Flush()
	dcOutput = append(dcOutput, dcFlush...)

	// Skip transients, check middle
	dcMean := 0.0
	dcMin, dcMax := dcOutput[500], dcOutput[500]
	for i := 500; i < len(dcOutput)-500; i++ {
		dcMean += dcOutput[i]
		if dcOutput[i] < dcMin {
			dcMin = dcOutput[i]
		}
		if dcOutput[i] > dcMax {
			dcMax = dcOutput[i]
		}
	}
	dcMean /= float64(len(dcOutput) - 1000)
	t.Logf("  Input: 1.0, Output mean: %.6f, range: [%.6f, %.6f]", dcMean, dcMin, dcMax)

	// Reset for next test
	dftStage.Reset()

	// Test 2: Very low frequency sine (should pass through unchanged)
	t.Log("Test 2: Low frequency sine (100 Hz @ 48k input)")
	testFreq := 100.0
	inputRate := 48000.0
	lowFreqInput := make([]float64, 2000)
	for i := range lowFreqInput {
		lowFreqInput[i] = math.Sin(2.0 * math.Pi * testFreq * float64(i) / inputRate)
	}

	lowFreqOutput, _ := dftStage.Process(lowFreqInput)
	lowFreqFlush, _ := dftStage.Flush()
	lowFreqOutput = append(lowFreqOutput, lowFreqFlush...)

	// Check amplitude
	maxAbs := 0.0
	for i := 500; i < len(lowFreqOutput)-500; i++ {
		if math.Abs(lowFreqOutput[i]) > maxAbs {
			maxAbs = math.Abs(lowFreqOutput[i])
		}
	}
	t.Logf("  Input amplitude: 1.0, Output max amplitude: %.6f", maxAbs)
	t.Logf("  (Should be ~1.0 for a good lowpass filter)")
}

// TestDFTStage_OutputPattern examines output pattern sample-by-sample
func TestDFTStage_OutputPattern(t *testing.T) {
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Generate a short sine wave to examine the output pattern
	testFreq := 1000.0
	inputRate := 48000.0
	numSamples := 500

	input := make([]float64, numSamples)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * testFreq * float64(i) / inputRate)
	}

	output, _ := dftStage.Process(input)
	flush, _ := dftStage.Flush()
	output = append(output, flush...)

	// Skip initial transient, examine middle samples
	startIdx := 400 // Well past filter transient
	endIdx := startIdx + 100

	t.Log("Examining output samples (should be 2x upsampled sine):")
	t.Log("Index\tOutput\t\tPhase\tExpected(approx)")

	// For 1kHz at 96kHz output rate, period is 96 samples
	outputRate := 96000.0
	for i := startIdx; i < endIdx && i < len(output); i++ {
		phase := i % 2 // Which polyphase this sample came from

		// Calculate expected value (accounting for filter delay)
		// For 2x upsample, output[i] corresponds to time i/(2*inputRate)
		expectedTime := float64(i) / outputRate
		expected := math.Sin(2.0 * math.Pi * testFreq * expectedTime)

		diff := output[i] - expected

		if i < startIdx+20 || math.Abs(diff) > 0.01 {
			t.Logf("%d\t%.6f\t%d\t%.6f (diff: %.6f)", i, output[i], phase, expected, diff)
		}
	}

	// Separate analysis by phase
	t.Log("\nPhase-separated amplitude analysis (middle 200 samples):")
	phase0Max, phase0Min := -math.MaxFloat64, math.MaxFloat64
	phase1Max, phase1Min := -math.MaxFloat64, math.MaxFloat64

	for i := 400; i < 600 && i < len(output); i++ {
		if i%2 == 0 {
			if output[i] > phase0Max {
				phase0Max = output[i]
			}
			if output[i] < phase0Min {
				phase0Min = output[i]
			}
		} else {
			if output[i] > phase1Max {
				phase1Max = output[i]
			}
			if output[i] < phase1Min {
				phase1Min = output[i]
			}
		}
	}

	t.Logf("Phase 0 range: [%.6f, %.6f]", phase0Min, phase0Max)
	t.Logf("Phase 1 range: [%.6f, %.6f]", phase1Min, phase1Max)
	t.Logf("Phase 0 amplitude: %.6f", (phase0Max-phase0Min)/2)
	t.Logf("Phase 1 amplitude: %.6f", (phase1Max-phase1Min)/2)
}

// TestDFTStage_ConvolutionOrder checks if coefficient reversal is correct
func TestDFTStage_ConvolutionOrder(t *testing.T) {
	// Create a simple impulse test to verify convolution direction
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Create impulse in the middle of buffer
	input := make([]float64, 1000)
	impulsePos := 500
	input[impulsePos] = 1.0

	output, _ := dftStage.Process(input)
	flush, _ := dftStage.Flush()
	output = append(output, flush...)

	// Find the output impulse peak
	maxVal := 0.0
	maxIdx := 0
	for i, v := range output {
		if math.Abs(v) > math.Abs(maxVal) {
			maxVal = v
			maxIdx = i
		}
	}

	t.Logf("Input impulse at sample %d", impulsePos)
	t.Logf("Output peak at sample %d, value %.6f", maxIdx, maxVal)
	t.Logf("Expected output position: ~%d (2x input)", impulsePos*2)

	// The peak should be positive (not inverted)
	if maxVal < 0 {
		t.Logf("WARNING: Impulse response is inverted!")
	}

	// Check symmetry of impulse response around peak
	t.Log("\nImpulse response around peak (should be symmetric for linear phase):")
	for offset := -5; offset <= 5; offset++ {
		idx := maxIdx + offset
		if idx >= 0 && idx < len(output) {
			t.Logf("  [%+d] = %.6f", offset, output[idx])
		}
	}

	// Check if response alternates (interleaved phases)
	t.Log("\nChecking interleave pattern around peak:")
	for i := maxIdx - 10; i <= maxIdx+10 && i >= 0 && i < len(output); i++ {
		phase := i % 2
		t.Logf("  [%d] phase=%d value=%.6f", i, phase, output[i])
	}
}

// TestDFTStage_ManualConvolution bypasses SIMD to verify algorithm correctness
func TestDFTStage_ManualConvolution(t *testing.T) {
	testFreq := 1000.0
	inputRate := 48000.0
	outputRate := 96000.0
	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Create DFT stage to get its coefficients
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	t.Logf("Taps per phase: %d", dftStage.tapsPerPhase)

	// Manual polyphase convolution (no SIMD)
	tapsPerPhase := dftStage.tapsPerPhase
	factor := 2

	// Create history buffer
	history := make([]float64, len(input)+tapsPerPhase)
	copy(history[tapsPerPhase-1:], input) // Pre-pad with zeros

	// Process each input sample
	numOutput := (len(input) - tapsPerPhase + 1) * factor
	output := make([]float64, numOutput)

	outIdx := 0
	for i := tapsPerPhase - 1; i < len(input); i++ {
		// For each input sample, produce 'factor' output samples
		for phase := range factor {
			// Manual convolution with pre-reversed coefficients
			sum := 0.0
			coeffs := dftStage.polyCoeffs[phase]
			histStart := i - (tapsPerPhase - 1)

			for j := range tapsPerPhase {
				sum += coeffs[j] * history[histStart+j+tapsPerPhase-1]
			}

			if outIdx < len(output) {
				output[outIdx] = sum
				outIdx++
			}
		}
	}

	t.Logf("Generated %d output samples", outIdx)

	// Measure THD
	if outIdx < fftSize {
		padded := make([]float64, fftSize)
		copy(padded, output[:outIdx])
		output = padded
	}

	fftIn := make([]complex128, fftSize)
	for i := range fftSize {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(output[i]*window, 0)
	}

	fftOut := fft(fftIn)

	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])

	nyquist := outputRate / 2.0
	var harmonicPowerSum float64

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
	thdDB := 20 * math.Log10(thdRatio+1e-20)

	t.Logf("Manual convolution THD: %.2f dB", thdDB)
	t.Logf("If this matches DFT stage THD (~-95 dB), algorithm is the issue")
	t.Logf("If this is much better, SIMD implementation has a bug")
}

// TestDFTStage_DirectUpsample tests true zero-stuffing and filter approach
func TestDFTStage_DirectUpsample(t *testing.T) {
	testFreq := 1000.0
	inputRate := 48000.0
	outputRate := 96000.0
	numSamples := 10000
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Get DFT stage filter (reconstruct prototype from polyphase)
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Reconstruct prototype filter
	factor := 2
	tapsPerPhase := dftStage.tapsPerPhase
	totalTaps := tapsPerPhase * factor
	prototype := make([]float64, totalTaps)

	for phase := range factor {
		for tap := range tapsPerPhase {
			protoIdx := tap*factor + phase
			if protoIdx < totalTaps {
				// Un-reverse and un-scale
				prototype[protoIdx] = dftStage.polyCoeffs[phase][tapsPerPhase-1-tap] / float64(factor)
			}
		}
	}

	// True upsampling: zero-stuff then filter
	stuffed := make([]float64, len(input)*factor)
	for i, v := range input {
		stuffed[i*factor] = v // Put input at even positions, zeros at odd
	}

	// Direct convolution with prototype filter
	filterLen := len(prototype)
	output := make([]float64, len(stuffed)-filterLen+1)

	for i := 0; i < len(output); i++ {
		sum := 0.0
		for j := range filterLen {
			sum += stuffed[i+j] * prototype[filterLen-1-j] // Note: prototype is not reversed for proper conv
		}
		output[i] = sum * float64(factor) // Scale for upsampling
	}

	t.Logf("Direct upsample: input %d, stuffed %d, output %d", len(input), len(stuffed), len(output))

	// Measure THD
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

	fundamentalBin := int(testFreq / outputRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])

	nyquist := outputRate / 2.0
	var harmonicPowerSum float64

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
	thdDB := 20 * math.Log10(thdRatio+1e-20)

	t.Logf("Direct zero-stuff+filter THD: %.2f dB", thdDB)
	t.Logf("This is the theoretical best for this filter design")
}

// TestIdentityResample tests 1:1 "resampling" - should have near-zero THD
func TestIdentityResample(t *testing.T) {
	testFreq := 1000.0
	sampleRate := 48000.0
	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / sampleRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Compute THD of input (should be limited only by FFT precision)
	fftIn := make([]complex128, fftSize)
	for i := range fftSize {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
		fftIn[i] = complex(input[i]*window, 0)
	}

	fftOut := fft(fftIn)

	fundamentalBin := int(testFreq / sampleRate * float64(fftSize))
	fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])

	nyquist := sampleRate / 2.0
	var harmonicPowerSum float64

	for h := 2; h <= 10; h++ {
		harmFreq := testFreq * float64(h)
		if harmFreq >= nyquist {
			break
		}

		harmBin := int(harmFreq / sampleRate * float64(fftSize))
		if harmBin < len(fftOut)/2 {
			harmMag := cmplx.Abs(fftOut[harmBin])
			harmonicPowerSum += harmMag * harmMag
		}
	}

	thdRatio := math.Sqrt(harmonicPowerSum) / (fundamentalMag + 1e-20)
	thdDB := 20 * math.Log10(thdRatio+1e-20)

	t.Logf("Input signal (pure sine) THD: %.2f dB", thdDB)
	t.Logf("This is the baseline - any resampler THD should be close to this")

	if thdDB > -150 {
		t.Logf("WARNING: Input THD is higher than expected - check test methodology")
	}
}

// TestDFTStage_THDComparison computes THD for both DFT stage and pure Go in same test
func TestDFTStage_THDComparison(t *testing.T) {
	testFreq := 1000.0
	inputRate := 48000.0
	outputRate := 96000.0
	numSamples := 65536
	fftSize := 16384

	// Generate pure sine wave
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Create DFT stage
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	tapsPerPhase := dftStage.tapsPerPhase
	factor := 2

	t.Logf("Taps per phase: %d", tapsPerPhase)
	t.Logf("Is half-band: %v", dftStage.isHalfBand)

	// === Method 1: DFT Stage (SIMD path) ===
	simdOutput, err := dftStage.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}
	flush, _ := dftStage.Flush()
	simdOutput = append(simdOutput, flush...)

	// === Method 2: Pure Go convolution (same logic as DFT stage) ===
	numInputProcessable := len(input) - tapsPerPhase + 1
	goOutput := make([]float64, numInputProcessable*factor)

	outIdx := 0
	for i := range numInputProcessable {
		for phase := range factor {
			sum := 0.0
			coeffs := dftStage.polyCoeffs[phase]
			for j := range tapsPerPhase {
				sum += input[i+j] * float64(coeffs[j])
			}
			goOutput[outIdx] = sum
			outIdx++
		}
	}

	t.Logf("SIMD output length: %d", len(simdOutput))
	t.Logf("Go output length: %d", len(goOutput))

	// Compare first few samples to verify they match
	for i := range 10 {
		diff := simdOutput[i] - goOutput[i]
		if math.Abs(diff) > 1e-10 {
			t.Logf("WARNING: Sample %d differs: SIMD=%.10f, Go=%.10f, diff=%.2e",
				i, simdOutput[i], goOutput[i], diff)
		}
	}

	// Helper function to compute THD
	computeTHD := func(output []float64) float64 {
		fftIn := make([]complex128, fftSize)
		for i := range fftSize {
			window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(fftSize-1)))
			fftIn[i] = complex(output[i]*window, 0)
		}

		fftOut := fft(fftIn)

		fundamentalBin := int(testFreq / outputRate * float64(fftSize))
		fundamentalMag := cmplx.Abs(fftOut[fundamentalBin])

		nyquist := outputRate / 2.0
		var harmonicPowerSum float64

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
		thdDB := 20 * math.Log10(thdRatio+1e-20)
		return thdDB
	}

	simdTHD := computeTHD(simdOutput)
	goTHD := computeTHD(goOutput)

	t.Logf("SIMD (DFT stage) THD: %.2f dB", simdTHD)
	t.Logf("Pure Go THD: %.2f dB", goTHD)
	t.Logf("Difference: %.2f dB", simdTHD-goTHD)

	if math.Abs(simdTHD-goTHD) > 1.0 {
		t.Logf("SIGNIFICANT THD DIFFERENCE - investigating samples...")

		// Check if outputs are numerically different
		maxDiff := 0.0
		maxDiffIdx := 0
		for i := 0; i < min(len(simdOutput), len(goOutput)); i++ {
			diff := math.Abs(simdOutput[i] - goOutput[i])
			if diff > maxDiff {
				maxDiff = diff
				maxDiffIdx = i
			}
		}
		t.Logf("Max sample difference: %.6e at index %d", maxDiff, maxDiffIdx)

		// Check DC gain of each phase
		t.Log("\nPhase DC gains:")
		for phase := range factor {
			sum := 0.0
			for _, c := range dftStage.polyCoeffs[phase] {
				sum += float64(c)
			}
			t.Logf("  Phase %d DC gain: %.6f", phase, sum)
		}

		// Manual convolution of first position for debugging
		t.Log("\nManual check of first output:")
		for phase := range factor {
			sum := 0.0
			for j := range tapsPerPhase {
				sum += input[j] * float64(dftStage.polyCoeffs[phase][j])
			}
			t.Logf("  Phase %d, pos 0: manual=%.6f, SIMD output[%d]=%.6f, Go output[%d]=%.6f",
				phase, sum, phase, simdOutput[phase], phase, goOutput[phase])
		}

		// Check first few input samples
		t.Log("\nFirst 5 input samples:")
		for i := range 5 {
			t.Logf("  input[%d] = %.6f", i, input[i])
		}

		// Check first few phase 0 coefficients
		t.Log("\nFirst 5 phase 0 coefficients:")
		for i := range 5 {
			t.Logf("  polyCoeffs[0][%d] = %.10f", i, dftStage.polyCoeffs[0][i])
		}
	}
}

// TestDFTStage_ProcessChunkedBug tests for bugs in the chunked processing path
func TestDFTStage_ProcessChunkedBug(t *testing.T) {
	// Use enough samples to trigger processChunked (> l2CacheChunkSize = 4096)
	numSamples := 10000
	testFreq := 1000.0
	inputRate := 48000.0

	// Use same sine wave as THDComparison test
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * testFreq * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	tapsPerPhase := dftStage.tapsPerPhase
	factor := 2

	t.Logf("Taps per phase: %d", tapsPerPhase)
	t.Logf("Input samples: %d", numSamples)
	t.Logf("numInputProcessable: %d", numSamples-tapsPerPhase+1)
	t.Logf("l2CacheChunkSize: 4096")
	t.Logf("Using processChunked: %v", numSamples-tapsPerPhase >= 4096)

	// Process with DFT stage (including Flush, just like THDComparison)
	simdOutput, err := dftStage.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}
	flush, _ := dftStage.Flush()
	simdOutput = append(simdOutput, flush...)

	// Pure Go convolution
	numInputProcessable := numSamples - tapsPerPhase + 1
	goOutput := make([]float64, numInputProcessable*factor)

	outIdx := 0
	for i := range numInputProcessable {
		for phase := range factor {
			sum := 0.0
			coeffs := dftStage.polyCoeffs[phase]
			for j := range tapsPerPhase {
				sum += input[i+j] * float64(coeffs[j])
			}
			goOutput[outIdx] = sum
			outIdx++
		}
	}

	t.Logf("SIMD output length: %d", len(simdOutput))
	t.Logf("Go output length: %d", len(goOutput))

	// Compare first 20 samples
	t.Log("\nFirst 20 samples comparison:")
	for i := 0; i < 20 && i < len(simdOutput) && i < len(goOutput); i++ {
		diff := simdOutput[i] - goOutput[i]
		if math.Abs(diff) > 1e-10 {
			t.Logf("MISMATCH at %d: SIMD=%.10f, Go=%.10f, diff=%.2e", i, simdOutput[i], goOutput[i], diff)
		} else {
			t.Logf("OK at %d: SIMD=%.10f, Go=%.10f", i, simdOutput[i], goOutput[i])
		}
	}

	// Find first mismatch
	for i := 0; i < min(len(simdOutput), len(goOutput)); i++ {
		diff := math.Abs(simdOutput[i] - goOutput[i])
		if diff > 1e-10 {
			t.Logf("\nFirst mismatch at sample %d", i)
			break
		}
	}

	// Also check around chunk boundary (4096)
	chunkBoundary := 4096 * 2 // Output position of first sample in second chunk
	t.Logf("\nSamples around chunk boundary (output index %d):", chunkBoundary)
	for i := chunkBoundary - 5; i < chunkBoundary+5 && i >= 0 && i < len(simdOutput) && i < len(goOutput); i++ {
		diff := simdOutput[i] - goOutput[i]
		if math.Abs(diff) > 1e-10 {
			t.Logf("MISMATCH at %d: SIMD=%.10f, Go=%.10f, diff=%.2e", i, simdOutput[i], goOutput[i], diff)
		} else {
			t.Logf("OK at %d: SIMD=%.10f, Go=%.10f", i, simdOutput[i], goOutput[i])
		}
	}
}

// TestDFTStage_DirectComparison compares DFT stage output with pure Go convolution
// sample by sample to find where the divergence occurs
func TestDFTStage_DirectComparison(t *testing.T) {
	numSamples := 1000

	// Simple ramp input to make debugging easier
	input := make([]float64, numSamples)
	for i := range input {
		input[i] = float64(i) / float64(numSamples)
	}

	// Create DFT stage
	dftStage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatalf("Failed to create DFT stage: %v", err)
	}

	// Get coefficients
	tapsPerPhase := dftStage.tapsPerPhase
	factor := 2

	t.Logf("Taps per phase: %d", tapsPerPhase)
	t.Logf("Is half-band: %v", dftStage.isHalfBand)

	// Process with DFT stage (SIMD path)
	simdOutput, err := dftStage.Process(input)
	if err != nil {
		t.Fatalf("Failed to process: %v", err)
	}

	// Manual pure Go convolution with same coefficients
	numInputProcessable := numSamples - tapsPerPhase + 1
	if numInputProcessable <= 0 {
		t.Fatalf("Not enough input samples")
	}
	goOutput := make([]float64, numInputProcessable*factor)

	outIdx := 0
	for i := range numInputProcessable {
		// For each input position, produce 'factor' output samples
		for phase := range factor {
			sum := 0.0
			coeffs := dftStage.polyCoeffs[phase]
			// ConvolveValid does: dst[i] = sum(signal[i+j] * kernel[j])
			// So for output i, we need signal[i:i+tapsPerPhase]
			for j := range tapsPerPhase {
				sum += input[i+j] * float64(coeffs[j])
			}
			goOutput[outIdx] = sum
			outIdx++
		}
	}

	t.Logf("SIMD output length: %d", len(simdOutput))
	t.Logf("Go output length: %d", len(goOutput))

	// Compare first few samples
	compareLen := min(len(simdOutput), len(goOutput), 20)
	t.Log("\nFirst 20 samples comparison:")
	t.Log("Index\tSIMD Output\tGo Output\tDiff")

	maxDiff := 0.0
	maxDiffIdx := 0
	for i := range compareLen {
		diff := simdOutput[i] - goOutput[i]
		if math.Abs(diff) > math.Abs(maxDiff) {
			maxDiff = diff
			maxDiffIdx = i
		}
		t.Logf("%d\t%.10f\t%.10f\t%.2e", i, simdOutput[i], goOutput[i], diff)
	}

	// Find overall max diff
	for i := 0; i < min(len(simdOutput), len(goOutput)); i++ {
		diff := simdOutput[i] - goOutput[i]
		if math.Abs(diff) > math.Abs(maxDiff) {
			maxDiff = diff
			maxDiffIdx = i
		}
	}

	t.Logf("\nMax difference: %.6e at index %d", maxDiff, maxDiffIdx)

	// Check if Phase 0 and Phase 1 have different error patterns
	phase0Diff := 0.0
	phase1Diff := 0.0
	phase0Count := 0
	phase1Count := 0
	for i := 0; i < min(len(simdOutput), len(goOutput)); i++ {
		diff := math.Abs(simdOutput[i] - goOutput[i])
		if i%2 == 0 {
			phase0Diff += diff
			phase0Count++
		} else {
			phase1Diff += diff
			phase1Count++
		}
	}
	if phase0Count > 0 {
		t.Logf("Phase 0 average abs diff: %.6e", phase0Diff/float64(phase0Count))
	}
	if phase1Count > 0 {
		t.Logf("Phase 1 average abs diff: %.6e", phase1Diff/float64(phase1Count))
	}
}
