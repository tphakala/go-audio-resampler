package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Tests to Find 96k->48k Downsampling Bug (12.46 dB attenuation)
// =============================================================================

// TestDownsampling_96kTo48k_FilterDesign tests that the filter design
// parameters are correct for 96k->48k downsampling.
//
// Bug hypothesis: The polyphase filter for 96k->48k has an extremely narrow
// transition band (Fs - Fp ≈ 0.00125) with only 1599 taps, which is insufficient
// to achieve the target 180 dB attenuation.
func TestDownsampling_96kTo48k_FilterDesign(t *testing.T) {
	// For 96k->48k:
	// - ratio = 0.5 (outputRate/inputRate)
	// - isIntegerRatio(0.5) = false (0.5 rounds to 0 or 1, not equal to 0.5)
	// - Uses DFT pre-stage (2x) + polyphase stage
	// - polyphaseRatio = 48000 / (96000*2) = 0.25
	// - totalIORatio = 96000 / 48000 = 2.0

	inputRate := 96000.0
	outputRate := 48000.0
	ratio := outputRate / inputRate

	// Verify ratio detection
	isInt := isIntegerRatio(ratio)
	t.Logf("Ratio %.6f isIntegerRatio: %v", ratio, isInt)

	// BUG: ratio=0.5 should ideally be treated as 1/2 integer downsampling
	// but it's being treated as a non-integer ratio requiring polyphase
	if isInt {
		t.Log("WARNING: 0.5 ratio is being treated as integer - check isIntegerRatio logic")
	}

	// Create resampler to examine internal structure
	resampler, err := NewResampler[float64](inputRate, outputRate, QualityHigh)
	require.NoError(t, err)

	// Check what stages were created
	hasPreStage := resampler.preStage != nil
	hasPolyphase := resampler.polyphaseStage != nil
	t.Logf("Has pre-stage: %v, Has polyphase: %v", hasPreStage, hasPolyphase)

	if hasPolyphase {
		t.Logf("Polyphase numPhases: %d, tapsPerPhase: %d",
			resampler.polyphaseStage.numPhases,
			resampler.polyphaseStage.tapsPerPhase)

		// Calculate the effective transition bandwidth for polyphase filter
		// polyphaseRatio = 0.25 (48000/192000)
		polyphaseRatio := outputRate / (inputRate * 2.0)
		t.Logf("Polyphase ratio: %.6f", polyphaseRatio)

		// The transition band for downsampling is effectively:
		// Fp ≈ 0.5 * ratio * 0.99 = 0.12375
		// Fs ≈ 0.5 * ratio = 0.125
		// Transition BW = Fs - Fp = 0.00125 (EXTREMELY NARROW!)
		fp := 0.5 * polyphaseRatio * 0.99
		fs := 0.5 * polyphaseRatio
		transitionBW := fs - fp
		t.Logf("Estimated Fp: %.6f, Fs: %.6f, Transition BW: %.6f", fp, fs, transitionBW)

		// Required taps for 180dB attenuation with narrow transition band
		// N ≈ (A - 7.95) / (2.285 * 2π * transitionBW)
		requiredTaps := (180.0 - 7.95) / (2.285 * 2.0 * math.Pi * transitionBW)
		actualTaps := resampler.polyphaseStage.numPhases * resampler.polyphaseStage.tapsPerPhase
		t.Logf("Required taps for 180dB: %.0f, Actual taps: %d", requiredTaps, actualTaps)

		// BUG DETECTION: If required taps >> actual taps, we found the bug
		if requiredTaps > float64(actualTaps)*2 {
			t.Errorf("BUG FOUND: Filter has %d taps but needs ~%.0f for 180dB attenuation with TBW=%.6f",
				actualTaps, requiredTaps, transitionBW)
		}
	}
}

// TestDownsampling_96kTo48k_SignalQuality directly tests the signal quality
// for 96k->48k downsampling with a known input.
func TestDownsampling_96kTo48k_SignalQuality(t *testing.T) {
	inputRate := 96000.0
	outputRate := 48000.0

	resampler, err := NewResampler[float64](inputRate, outputRate, QualityHigh)
	require.NoError(t, err)

	// Generate a signal with tones that should be filtered out
	// Output Nyquist = 24kHz, so frequencies above 24kHz should be attenuated
	numSamples := 32768
	input := make([]float64, numSamples)

	// Add a tone at 10kHz (passband - should be preserved)
	// Add a tone at 30kHz (stopband - should be filtered)
	for i := range input {
		t := float64(i) / inputRate
		input[i] = 0.5*math.Sin(2*math.Pi*10000*t) + 0.5*math.Sin(2*math.Pi*30000*t)
	}

	output, err := resampler.Process(input)
	require.NoError(t, err)
	flush, err := resampler.Flush()
	require.NoError(t, err)
	output = append(output, flush...)

	// Analyze output: the 30kHz tone aliases to 30-48=18kHz in the output
	// We need to measure how much it's attenuated

	// Measure passband energy (around 10kHz which stays at 10kHz)
	passbandEnergy := measureFrequencyEnergy(output, outputRate, 9500, 10500)

	// Measure where the alias would appear (30kHz -> 30-48=-18kHz -> 18kHz due to aliasing)
	// Actually for 30kHz with 24kHz Nyquist: aliases to 48-30=18kHz in output
	aliasEnergy := measureFrequencyEnergy(output, outputRate, 17500, 18500)

	attenuation := passbandEnergy - aliasEnergy
	t.Logf("Passband energy (10kHz): %.2f dB", passbandEnergy)
	t.Logf("Alias energy (18kHz): %.2f dB", aliasEnergy)
	t.Logf("Attenuation: %.2f dB", attenuation)

	// Expected: >100 dB attenuation for high quality
	// Bug symptom: only ~12 dB attenuation
	if attenuation < 80 {
		t.Errorf("BUG: 96k->48k downsampling has only %.2f dB alias attenuation (expected >80 dB)", attenuation)
	}
}

// TestDownsampling_IntegerRatioDetection tests that integer downsampling ratios
// are properly detected and handled.
func TestDownsampling_IntegerRatioDetection(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		ratio      float64
		expectInt  bool
		desc       string
	}{
		{48000, 96000, 2.0, true, "2x upsample - should be integer"},
		{44100, 88200, 2.0, true, "2x upsample - should be integer"},
		{96000, 48000, 0.5, false, "0.5x downsample - NOT detected as integer"},
		{88200, 44100, 0.5, false, "0.5x downsample - NOT detected as integer"},
		{96000, 32000, 1.0 / 3.0, false, "3x downsample - not integer"},
		{48000, 44100, 0.91875, false, "rational - not integer"},
		{44100, 48000, 1.0884, false, "rational - not integer"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ratio := tc.outputRate / tc.inputRate
			isInt := isIntegerRatio(ratio)
			t.Logf("%.0f -> %.0f, ratio=%.6f, isIntegerRatio=%v", tc.inputRate, tc.outputRate, ratio, isInt)

			// Note: Current implementation only detects ratios >= 1 as integer
			// This means 2x DOWNsampling (ratio=0.5) is NOT treated as integer
			// This could be intentional or a bug depending on design goals

			// Check if we have a potential architecture issue
			if ratio < 1.0 && ratio > 0 && math.Abs(1.0/ratio-math.Round(1.0/ratio)) < 1e-9 {
				// This is an integer DOWNsampling ratio (e.g., 0.5 = 1/2)
				t.Logf("NOTE: ratio %.6f is 1/%d integer downsampling but isIntegerRatio=%v",
					ratio, int(math.Round(1.0/ratio)), isInt)
			}
		})
	}
}

// TestDownsampling_HeavyRatios tests heavy downsampling scenarios
// that stress the polyphase filter design.
func TestDownsampling_HeavyRatios(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{96000, 48000, "96k_to_48k_2x_down"},
		{96000, 32000, "96k_to_32k_3x_down"},
		{96000, 24000, "96k_to_24k_4x_down"},
		{192000, 48000, "192k_to_48k_4x_down"},
		{192000, 44100, "192k_to_44.1k_heavy_rational"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err)

			// Generate signal with content in aliasing region
			numSamples := 32768
			input := make([]float64, numSamples)
			outputNyquist := tc.outputRate / 2.0

			// Add tones: one in passband, one in aliasing region
			passbandFreq := outputNyquist * 0.5 // Middle of passband
			aliasFreq := outputNyquist * 1.5    // Would alias to 0.5*outputNyquist

			if aliasFreq < tc.inputRate/2.0 { // Only if within input Nyquist
				for i := range input {
					t := float64(i) / tc.inputRate
					input[i] = 0.5*math.Sin(2*math.Pi*passbandFreq*t) +
						0.5*math.Sin(2*math.Pi*aliasFreq*t)
				}

				output, err := resampler.Process(input)
				require.NoError(t, err)
				flush, err := resampler.Flush()
				require.NoError(t, err)
				output = append(output, flush...)

				// Measure attenuation
				passbandE := measureFrequencyEnergy(output, tc.outputRate, passbandFreq*0.95, passbandFreq*1.05)
				// Alias appears at: aliasFreq - outputNyquist or outputRate - aliasFreq (depending on fold)
				aliasAppears := outputNyquist - (aliasFreq - outputNyquist)
				if aliasAppears < 0 {
					aliasAppears = -aliasAppears
				}
				aliasE := measureFrequencyEnergy(output, tc.outputRate, aliasAppears*0.95, aliasAppears*1.05)

				attenuation := passbandE - aliasE
				t.Logf("Passband (%.0fHz): %.2f dB, Alias region: %.2f dB, Attenuation: %.2f dB",
					passbandFreq, passbandE, aliasE, attenuation)

				if attenuation < 60 {
					t.Errorf("POOR ANTI-ALIASING: %.0f->%.0f only achieves %.2f dB attenuation",
						tc.inputRate, tc.outputRate, attenuation)
				}
			} else {
				t.Skipf("Alias frequency %.0f exceeds input Nyquist", aliasFreq)
			}
		})
	}
}

// =============================================================================
// Tests for Passband Ripple Issues in Rational Ratios
// =============================================================================

// TestPassbandRipple_RationalRatios tests passband flatness for non-integer ratios
// Bug: Rational ratios have 7-17 dB more ripple than soxr
func TestPassbandRipple_RationalRatios(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1k_to_48k"},
		{48000, 44100, "48k_to_44.1k"},
		{44100, 96000, "44.1k_to_96k"},
		{96000, 44100, "96k_to_44.1k"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err)

			// Test passband flatness with multitone signal
			numTones := 20
			numSamples := 65536
			lowerNyquist := math.Min(tc.inputRate, tc.outputRate) / 2.0
			maxFreq := lowerNyquist * 0.8 // Stay well within passband

			// Generate equal amplitude tones
			input := make([]float64, numSamples)
			testFreqs := make([]float64, numTones)
			for i := range numTones {
				testFreqs[i] = 500.0 + float64(i)*(maxFreq-500.0)/float64(numTones-1)
				for j := range input {
					t := float64(j) / tc.inputRate
					input[j] += (1.0 / float64(numTones)) * math.Sin(2*math.Pi*testFreqs[i]*t)
				}
			}

			output, err := resampler.Process(input)
			require.NoError(t, err)
			flush, err := resampler.Flush()
			require.NoError(t, err)
			output = append(output, flush...)

			// Measure level at each test frequency
			levels := make([]float64, numTones)
			for i, freq := range testFreqs {
				levels[i] = measureFrequencyEnergy(output, tc.outputRate, freq*0.98, freq*1.02)
			}

			// Calculate ripple (max - min level)
			maxLevel, minLevel := levels[0], levels[0]
			for _, l := range levels {
				if l > maxLevel {
					maxLevel = l
				}
				if l < minLevel {
					minLevel = l
				}
			}
			ripple := maxLevel - minLevel

			t.Logf("Frequency response levels: max=%.2f dB, min=%.2f dB, ripple=%.2f dB",
				maxLevel, minLevel, ripple)

			// soxr achieves ~1.3 dB ripple, we should be within ~2 dB
			// Bug symptom: 7-17 dB more ripple than expected
			if ripple > 3.0 {
				t.Errorf("EXCESSIVE PASSBAND RIPPLE: %.2f dB (expected < 3 dB)", ripple)
			}
		})
	}
}

// TestPassbandRipple_FilterDesignParams tests if the filter design parameters
// are correctly computed for rational ratios.
func TestPassbandRipple_FilterDesignParams(t *testing.T) {
	// Test the lsxInvFResp function that affects filter rolloff
	testCases := []struct {
		drop        float64
		attenuation float64
		desc        string
	}{
		{-0.01, 180.0, "high_quality_rolloff"},
		{-0.01, 140.0, "medium_quality_rolloff"},
		{-0.01, 100.0, "low_quality_rolloff"},
		{-0.1, 180.0, "larger_rolloff"},
		{-1.0, 180.0, "very_large_rolloff"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			result := lsxInvFResp(tc.drop, tc.attenuation)
			t.Logf("lsxInvFResp(%.4f, %.0f) = %.6f", tc.drop, tc.attenuation, result)

			// Result should be in valid range [0, 1]
			if result < 0 || result > 1 {
				t.Errorf("lsxInvFResp returned invalid value: %.6f (expected 0-1)", result)
			}

			// Check for NaN
			if math.IsNaN(result) {
				t.Errorf("lsxInvFResp returned NaN for drop=%.4f, atten=%.0f", tc.drop, tc.attenuation)
			}
		})
	}
}

// =============================================================================
// Tests for Pre-ringing Issues in Rational Ratios
// =============================================================================

// TestPreRinging_RationalRatios tests impulse response pre-ringing
// Bug: Rational ratios have 13-19 dB more pre-ringing than soxr
func TestPreRinging_RationalRatios(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1k_to_48k"},
		{48000, 44100, "48k_to_44.1k"},
		{96000, 48000, "96k_to_48k"},
		{48000, 32000, "48k_to_32k"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err)

			// Create impulse at center of buffer
			numSamples := 16384
			impulse := make([]float64, numSamples)
			impulsePos := numSamples / 2
			impulse[impulsePos] = 1.0

			output, err := resampler.Process(impulse)
			require.NoError(t, err)
			flush, err := resampler.Flush()
			require.NoError(t, err)
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

			// Measure pre-ringing (max before main peak)
			preRingPeak := 0.0
			for i := 0; i < mainPeakIdx; i++ {
				if math.Abs(output[i]) > preRingPeak {
					preRingPeak = math.Abs(output[i])
				}
			}
			preRingDB := 20 * math.Log10(preRingPeak/mainPeakVal+1e-20)

			// Measure post-ringing
			postRingPeak := 0.0
			for i := mainPeakIdx + 10; i < len(output); i++ {
				if math.Abs(output[i]) > postRingPeak {
					postRingPeak = math.Abs(output[i])
				}
			}
			postRingDB := 20 * math.Log10(postRingPeak/mainPeakVal+1e-20)

			t.Logf("Main peak at index %d, value %.6f", mainPeakIdx, mainPeakVal)
			t.Logf("Pre-ringing: %.2f dB, Post-ringing: %.2f dB", preRingDB, postRingDB)

			// soxr achieves -17 to -26 dB pre-ringing for rational ratios
			// Bug symptom: only -2 to -4.6 dB (13-19 dB worse)
			if preRingDB > -10 {
				t.Errorf("EXCESSIVE PRE-RINGING: %.2f dB (expected < -10 dB)", preRingDB)
			}
		})
	}
}

// =============================================================================
// Tests for Polyphase Filter Coefficient Quality
// =============================================================================

// TestPolyphaseStage_CoefficientSum tests that polyphase filter coefficients
// sum correctly for DC preservation.
func TestPolyphaseStage_CoefficientSum(t *testing.T) {
	testCases := []struct {
		ratio        float64
		totalIORatio float64
		desc         string
	}{
		{1.088435374, 0.459375, "44.1k_to_96k_polyphase"},
		{0.91875, 0.91875, "48k_to_44.1k_polyphase"},
		{0.25, 2.0, "96k_to_48k_polyphase"},
		{0.5, 1.5, "96k_to_32k_polyphase"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			stage, err := NewPolyphaseStage[float64](tc.ratio, tc.totalIORatio, QualityHigh)
			require.NoError(t, err)

			// Sum coefficients for each phase
			totalSum := 0.0
			for phase := 0; phase < stage.numPhases; phase++ {
				phaseSum := 0.0
				for _, coeff := range stage.polyCoeffs[phase] {
					phaseSum += coeff
				}
				totalSum += phaseSum

				// Each phase should have DC gain ≈ 1.0
				if math.Abs(phaseSum-1.0) > 0.1 {
					t.Logf("Phase %d sum: %.6f (expected ~1.0)", phase, phaseSum)
				}
			}

			// Total DC gain should equal numPhases
			expectedTotal := float64(stage.numPhases)
			if math.Abs(totalSum-expectedTotal) > 1.0 {
				t.Errorf("Total coefficient sum %.6f differs from expected %.6f",
					totalSum, expectedTotal)
			}

			t.Logf("Polyphase stage: %d phases, %d taps/phase, total sum: %.6f",
				stage.numPhases, stage.tapsPerPhase, totalSum)
		})
	}
}

// TestPolyphaseStage_PhaseInterpolation tests that adjacent phases
// produce smooth interpolation.
func TestPolyphaseStage_PhaseInterpolation(t *testing.T) {
	stage, err := NewPolyphaseStage[float64](1.088435374, 0.459375, QualityHigh)
	require.NoError(t, err)

	// Check that coefficient changes smoothly between adjacent phases
	maxPhaseDiff := 0.0
	for phase := 0; phase < stage.numPhases-1; phase++ {
		phaseDiff := 0.0
		for tap := 0; tap < stage.tapsPerPhase; tap++ {
			diff := math.Abs(stage.polyCoeffs[phase+1][tap] - stage.polyCoeffs[phase][tap])
			phaseDiff += diff * diff
		}
		phaseDiff = math.Sqrt(phaseDiff)
		if phaseDiff > maxPhaseDiff {
			maxPhaseDiff = phaseDiff
		}
	}

	t.Logf("Max phase-to-phase coefficient RMS difference: %.6f", maxPhaseDiff)

	// If phases differ too much, interpolation will be poor
	if maxPhaseDiff > 0.5 {
		t.Errorf("Phases differ too much: max RMS diff = %.6f", maxPhaseDiff)
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

// measureFrequencyEnergy measures energy at a specific frequency band using simple DFT
func measureFrequencyEnergy(signal []float64, sampleRate, fLow, fHigh float64) float64 {
	if len(signal) == 0 {
		return -200.0
	}

	// Use a chunk of the signal
	fftSize := min(8192, len(signal))
	if fftSize < 256 {
		return -200.0
	}

	// Apply window and compute DFT at target frequencies
	freqResolution := sampleRate / float64(fftSize)
	startBin := int(fLow / freqResolution)
	endBin := int(fHigh / freqResolution)
	if startBin < 1 {
		startBin = 1
	}
	if endBin >= fftSize/2 {
		endBin = fftSize/2 - 1
	}

	// Simple DFT for target frequency bins
	maxPower := 0.0
	for bin := startBin; bin <= endBin; bin++ {
		freq := float64(bin) * freqResolution
		omega := 2.0 * math.Pi * freq / sampleRate

		// Compute DFT at this frequency
		realPart := 0.0
		imagPart := 0.0
		for n := range fftSize {
			// Apply Hann window
			window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(n)/float64(fftSize-1)))
			sample := signal[n] * window
			realPart += sample * math.Cos(omega*float64(n))
			imagPart -= sample * math.Sin(omega*float64(n))
		}
		power := realPart*realPart + imagPart*imagPart
		if power > maxPower {
			maxPower = power
		}
	}

	if maxPower < 1e-20 {
		return -200.0
	}
	return 10.0 * math.Log10(maxPower)
}

// =============================================================================
// Tests for DFT Stage Downsampling Behavior
// =============================================================================

// TestDFTStage_NoDownsampleSupport verifies DFT stage behavior
// DFT stage is designed for upsampling only (factor >= 1)
func TestDFTStage_NoDownsampleSupport(t *testing.T) {
	// DFT stage with factor 1 should be pass-through
	stage1, err := NewDFTStage[float64](1, QualityHigh)
	require.NoError(t, err)
	assert.Equal(t, 1, stage1.factor)

	// DFT stage with factor 2 should upsample
	stage2, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err)
	assert.Equal(t, 2, stage2.factor)

	// Test that factor=1 is truly pass-through
	input := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	output, err := stage1.Process(input)
	require.NoError(t, err)

	// Factor=1 should return input directly
	assert.Len(t, output, len(input))
	for i := range input {
		assert.InDelta(t, input[i], output[i], 1e-10, "Factor=1 should be pass-through")
	}

	// Test factor=2 upsampling produces 2x output
	input2 := make([]float64, 1000)
	for i := range input2 {
		input2[i] = math.Sin(2.0 * math.Pi * float64(i) / 100)
	}
	output2, err := stage2.Process(input2)
	require.NoError(t, err)
	flush2, err := stage2.Flush()
	require.NoError(t, err)
	output2 = append(output2, flush2...)

	// Should produce approximately 2x samples (minus filter delay)
	t.Logf("Factor=2: input=%d, output=%d (expected ~%d)", len(input2), len(output2), len(input2)*2)
	ratio := float64(len(output2)) / float64(len(input2))
	assert.InDelta(t, 2.0, ratio, 0.1, "Factor=2 should produce ~2x output")
}

// TestResampler_DownsampleArchitecture verifies the downsampling architecture
// for various ratios.
func TestResampler_DownsampleArchitecture(t *testing.T) {
	testCases := []struct {
		inputRate       float64
		outputRate      float64
		expectPreStage  bool
		expectPolyphase bool
		desc            string
	}{
		{48000, 96000, true, false, "2x_upsample_int"},
		{44100, 88200, true, false, "2x_upsample_int"},
		{96000, 48000, true, true, "2x_downsample_uses_polyphase"},
		{48000, 44100, true, true, "rational_downsample"},
		{44100, 48000, true, true, "rational_upsample"},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			require.NoError(t, err)

			hasPreStage := resampler.preStage != nil
			hasPolyphase := resampler.polyphaseStage != nil

			t.Logf("%.0f -> %.0f: preStage=%v, polyphase=%v",
				tc.inputRate, tc.outputRate, hasPreStage, hasPolyphase)

			if hasPreStage != tc.expectPreStage {
				t.Errorf("Expected preStage=%v but got %v", tc.expectPreStage, hasPreStage)
			}
			if hasPolyphase != tc.expectPolyphase {
				t.Errorf("Expected polyphase=%v but got %v", tc.expectPolyphase, hasPolyphase)
			}

			// For downsampling with polyphase, verify the architecture makes sense
			if hasPolyphase && tc.outputRate < tc.inputRate {
				polyphaseRatio := tc.outputRate / (tc.inputRate * 2.0)
				t.Logf("Polyphase stage ratio: %.6f (should handle %.1fx downsampling)",
					polyphaseRatio, 1.0/polyphaseRatio)

				// Check if this ratio is very aggressive
				if polyphaseRatio < 0.3 {
					t.Logf("WARNING: Polyphase ratio %.3f is aggressive (>3x downsampling in polyphase alone)",
						polyphaseRatio)
				}
			}
		})
	}
}
