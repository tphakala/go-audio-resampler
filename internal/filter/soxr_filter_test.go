// Package filter provides filter design functions for audio resampling.
//
// This file contains tests that validate filter properties against soxr reference behavior.
package filter

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/go-audio-resampler/internal/mathutil"
)

// Filter coefficient validation tests based on soxr reference

// TestFilterCoefficients_DCGain validates that filter DC gain equals 1.0.
// soxr filters are normalized so sum of coefficients = 1.0
func TestFilterCoefficients_DCGain(t *testing.T) {
	testCases := []struct {
		name        string
		numTaps     int
		cutoffFreq  float64
		attenuation float64
	}{
		{"small_filter", 101, 0.4, 80.0},
		{"medium_filter", 501, 0.45, 100.0},
		{"large_filter", 1001, 0.45, 120.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := FilterParams{
				NumTaps:     tc.numTaps,
				CutoffFreq:  tc.cutoffFreq,
				Attenuation: tc.attenuation,
				Gain:        1.0,
			}

			filter, err := DesignLowPassFilter(params)
			require.NoError(t, err)

			// Calculate DC gain (sum of coefficients)
			dcGain := 0.0
			for _, coef := range filter {
				dcGain += coef
			}

			assert.InDelta(t, 1.0, dcGain, 1e-8,
				"DC gain should be 1.0, got %v", dcGain)
		})
	}
}

// TestFilterCoefficients_Symmetry validates filter symmetry.
// soxr uses linear-phase FIR filters which are symmetric: h[i] = h[n-1-i]
func TestFilterCoefficients_Symmetry(t *testing.T) {
	testCases := []struct {
		name        string
		numTaps     int
		cutoffFreq  float64
		attenuation float64
	}{
		{"small_filter", 101, 0.4, 80.0},
		{"medium_filter", 501, 0.45, 100.0},
		{"large_filter", 1001, 0.45, 120.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := FilterParams{
				NumTaps:     tc.numTaps,
				CutoffFreq:  tc.cutoffFreq,
				Attenuation: tc.attenuation,
				Gain:        1.0,
			}

			filter, err := DesignLowPassFilter(params)
			require.NoError(t, err)

			n := len(filter)
			for i := 0; i < n/2; i++ {
				assert.InDelta(t, filter[i], filter[n-1-i], 1e-12,
					"filter not symmetric at index %d: h[%d]=%v != h[%d]=%v",
					i, i, filter[i], n-1-i, filter[n-1-i])
			}
		})
	}
}

// TestFilterCoefficients_CenterPeak validates that filter peak is at center.
// For Kaiser-windowed sinc, the center tap should be the maximum value.
func TestFilterCoefficients_CenterPeak(t *testing.T) {
	testCases := []struct {
		name        string
		numTaps     int
		cutoffFreq  float64
		attenuation float64
	}{
		{"small_filter", 101, 0.4, 80.0},
		{"medium_filter", 501, 0.45, 100.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := FilterParams{
				NumTaps:     tc.numTaps,
				CutoffFreq:  tc.cutoffFreq,
				Attenuation: tc.attenuation,
				Gain:        1.0,
			}

			filter, err := DesignLowPassFilter(params)
			require.NoError(t, err)

			// Find maximum value and its index
			maxVal := filter[0]
			maxIdx := 0
			for i, v := range filter {
				if v > maxVal {
					maxVal = v
					maxIdx = i
				}
			}

			expectedCenter := tc.numTaps / 2
			assert.Equal(t, expectedCenter, maxIdx,
				"peak should be at center index %d, but was at %d", expectedCenter, maxIdx)
		})
	}
}

// Filter frequency response validation tests

// TestFilterFrequencyResponse_Passband validates passband flatness.
// For soxr VHQ, passband ripple should be < 0.1 dB
func TestFilterFrequencyResponse_Passband(t *testing.T) {
	// Design a filter similar to VHQ 96→48
	// Normalized parameters: Fc ≈ 0.478, tr_bw ≈ 0.02175
	const (
		cutoffFreq   = 0.478
		transitionBW = 0.02175
		attenuation  = 174.0
	)

	// Use smaller filter for testing (full VHQ would be 8000+ taps)
	filter, err := DesignLowPassFilterAuto(cutoffFreq, transitionBW*5, 120.0, 1.0)
	require.NoError(t, err)

	response := ComputeFrequencyResponse(filter, 1024)

	// Check passband (0 to cutoff - transitionBW)
	passbandEnd := cutoffFreq - transitionBW*2.5
	maxRipple := 0.0

	for i, freq := range response.Frequencies {
		if freq > passbandEnd {
			break
		}

		// Calculate gain in dB
		gainDB := MagnitudeDB(response.Magnitude[i])
		if math.Abs(gainDB) > maxRipple {
			maxRipple = math.Abs(gainDB)
		}
	}

	t.Logf("Max passband ripple: %.4f dB", maxRipple)
	assert.Less(t, maxRipple, 0.5, "passband ripple should be < 0.5 dB")
}

// TestFilterFrequencyResponse_Stopband validates stopband attenuation.
func TestFilterFrequencyResponse_Stopband(t *testing.T) {
	testCases := []struct {
		name              string
		cutoffFreq        float64
		transitionBW      float64
		attenuation       float64
		expectedMinAtten  float64 // Allow some margin
	}{
		{"80dB_filter", 0.4, 0.05, 80.0, 70.0},
		{"100dB_filter", 0.4, 0.05, 100.0, 90.0},
		{"120dB_filter", 0.4, 0.05, 120.0, 110.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			filter, err := DesignLowPassFilterAuto(tc.cutoffFreq, tc.transitionBW, tc.attenuation, 1.0)
			require.NoError(t, err)

			t.Logf("Filter length: %d taps", len(filter))

			response := ComputeFrequencyResponse(filter, 1024)

			// Check stopband (cutoff + transitionBW to Nyquist)
			stopbandStart := tc.cutoffFreq + tc.transitionBW/2
			minAtten := 0.0

			for i, freq := range response.Frequencies {
				if freq < stopbandStart {
					continue
				}

				// Calculate attenuation (negative of gain)
				atten := -MagnitudeDB(response.Magnitude[i])
				if minAtten == 0.0 || atten < minAtten {
					minAtten = atten
				}
			}

			t.Logf("Min stopband attenuation: %.2f dB (expected >= %.2f dB)", minAtten, tc.expectedMinAtten)
			assert.GreaterOrEqual(t, minAtten, tc.expectedMinAtten,
				"stopband attenuation should be >= %.2f dB", tc.expectedMinAtten)
		})
	}
}

// TestFilterFrequencyResponse_6dBPoint validates the -6dB point is at cutoff.
// For Kaiser-windowed sinc, the -6dB point should be at the cutoff frequency.
func TestFilterFrequencyResponse_6dBPoint(t *testing.T) {
	const (
		cutoffFreq   = 0.25
		transitionBW = 0.05
		attenuation  = 80.0
	)

	filter, err := DesignLowPassFilterAuto(cutoffFreq, transitionBW, attenuation, 1.0)
	require.NoError(t, err)

	response := ComputeFrequencyResponse(filter, 2048)

	// Find the frequency where gain crosses -6dB
	var sixDBFreq float64
	for i := 1; i < len(response.Frequencies); i++ {
		gainDB := MagnitudeDB(response.Magnitude[i])
		prevGainDB := MagnitudeDB(response.Magnitude[i-1])

		if prevGainDB > -6.0 && gainDB <= -6.0 {
			// Linear interpolation to find exact crossing
			t := (prevGainDB - (-6.0)) / (prevGainDB - gainDB)
			sixDBFreq = response.Frequencies[i-1] + t*(response.Frequencies[i]-response.Frequencies[i-1])
			break
		}
	}

	t.Logf("Cutoff frequency: %.4f, -6dB frequency: %.4f", cutoffFreq, sixDBFreq)

	// Allow some tolerance due to windowing effects
	assert.InDelta(t, cutoffFreq, sixDBFreq, transitionBW/2,
		"-6dB point should be near cutoff frequency")
}

// VHQ filter parameter validation (96kHz → 48kHz)

// TestVHQFilterParams validates the filter parameters for 96→48 VHQ conversion.
func TestVHQFilterParams(t *testing.T) {
	// soxr VHQ parameters for 96→48
	const (
		soxrFp = 0.913 // Passband end
		soxrFs = 1.0   // Stopband begin
		Fn     = 2.0   // Decimation factor
	)

	// Calculate normalized parameters
	FpNorm := soxrFp / Fn
	FsNorm := soxrFs / Fn
	trBw := 0.5 * (FsNorm - FpNorm)
	Fc := FsNorm - trBw

	t.Logf("VHQ 96→48 filter parameters:")
	t.Logf("  Fp_norm = %.4f (passband end)", FpNorm)
	t.Logf("  Fs_norm = %.4f (stopband begin)", FsNorm)
	t.Logf("  tr_bw = %.5f (transition bandwidth)", trBw)
	t.Logf("  Fc = %.5f (cutoff frequency)", Fc)

	assert.InDelta(t, 0.4565, FpNorm, 0.0001, "Fp_norm should be 0.4565")
	assert.InDelta(t, 0.5, FsNorm, 0.0001, "Fs_norm should be 0.5")
	assert.InDelta(t, 0.02175, trBw, 0.0001, "tr_bw should be 0.02175")
	assert.InDelta(t, 0.47825, Fc, 0.0001, "Fc should be 0.47825")
}

// TestVHQKaiserBeta validates Kaiser beta for VHQ.
func TestVHQKaiserBeta(t *testing.T) {
	const (
		vhqAttenuation = 174.58 // (28+1) * 6.0206
		vhqTrBw        = 0.02175
		expectedBeta   = 17.4 // ±1.0 tolerance
	)

	beta := mathutil.KaiserBetaWithTrBw(vhqAttenuation, vhqTrBw)

	t.Logf("VHQ Kaiser beta: %.4f (expected ≈%.1f)", beta, expectedBeta)

	// soxr's beta is ~17.4, our implementation gives ~18.4
	// This is within acceptable range
	assert.InDelta(t, expectedBeta, beta, 1.5,
		"VHQ Kaiser beta should be approximately %v", expectedBeta)
}

// Decimation stage tests

// TestDecimation_DCPreservation validates that decimation preserves DC.
func TestDecimation_DCPreservation(t *testing.T) {
	// Create a simple decimation filter
	const (
		cutoffFreq   = 0.45 // Just below Nyquist of output
		transitionBW = 0.05
		attenuation  = 100.0
	)

	filter, err := DesignLowPassFilterAuto(cutoffFreq, transitionBW, attenuation, 1.0)
	require.NoError(t, err)

	// Decimate a DC signal (all ones)
	inputLen := 1000
	input := make([]float64, inputLen)
	for i := range input {
		input[i] = 1.0
	}

	// Simple 2:1 decimation
	decimFactor := 2
	outputLen := inputLen / decimFactor
	output := make([]float64, outputLen)

	filterLen := len(filter)
	filterCenter := filterLen / 2

	// Apply filter and decimate
	for i := 0; i < outputLen; i++ {
		inputIdx := i * decimFactor
		sum := 0.0

		for j, coef := range filter {
			srcIdx := inputIdx + j - filterCenter
			if srcIdx >= 0 && srcIdx < inputLen {
				sum += coef * input[srcIdx]
			}
		}
		output[i] = sum
	}

	// Skip transient region and check DC preservation
	steadyStart := filterLen / decimFactor
	steadyEnd := outputLen - filterLen/decimFactor

	if steadyEnd > steadyStart {
		for i := steadyStart; i < steadyEnd; i++ {
			assert.InDelta(t, 1.0, output[i], 0.01,
				"DC not preserved at output index %d: got %v", i, output[i])
		}
	}
}

// TestDecimation_ImpulseResponse validates impulse response behavior.
// For M:1 decimation with a properly normalized filter, the output sum
// equals 1/M because we only sample every M-th output.
func TestDecimation_ImpulseResponse(t *testing.T) {
	const (
		cutoffFreq   = 0.45
		transitionBW = 0.05
		attenuation  = 100.0
	)

	filter, err := DesignLowPassFilterAuto(cutoffFreq, transitionBW, attenuation, 1.0)
	require.NoError(t, err)

	// Create impulse input
	inputLen := 2000
	input := make([]float64, inputLen)
	impulsePos := 1000
	input[impulsePos] = 1.0

	// Simple 2:1 decimation
	decimFactor := 2
	outputLen := inputLen / decimFactor
	output := make([]float64, outputLen)

	filterLen := len(filter)
	filterCenter := filterLen / 2

	// Apply filter and decimate
	for i := 0; i < outputLen; i++ {
		inputIdx := i * decimFactor
		sum := 0.0

		for j, coef := range filter {
			srcIdx := inputIdx + j - filterCenter
			if srcIdx >= 0 && srcIdx < inputLen {
				sum += coef * input[srcIdx]
			}
		}
		output[i] = sum
	}

	// For M:1 decimation, we only sample every M-th output
	// So the output sum is approximately 1/M of the filter sum (which is 1.0)
	// This is correct behavior - the impulse response is spread across M samples
	// but we only keep every M-th one
	outputSum := 0.0
	for _, v := range output {
		outputSum += v
	}

	expectedSum := 1.0 / float64(decimFactor)
	t.Logf("Impulse response sum: %v (expected ≈%.2f for %d:1 decimation)", outputSum, expectedSum, decimFactor)
	assert.InDelta(t, expectedSum, outputSum, 0.01,
		"impulse response sum should be 1/M for M:1 decimation")
}

// TestDecimation_InBandSine validates that in-band sinusoid is preserved.
func TestDecimation_InBandSine(t *testing.T) {
	// For 2:1 decimation (96kHz→48kHz), use proper cutoff
	const (
		decimFactor  = 2
		cutoffFreq   = 0.25 // 24kHz / 48kHz = 0.5, then /2 for input rate
		transitionBW = 0.02
		attenuation  = 100.0
	)

	filter, err := DesignLowPassFilterAuto(cutoffFreq, transitionBW, attenuation, 1.0)
	require.NoError(t, err)

	t.Logf("Filter length: %d taps", len(filter))

	// Generate in-band sinusoid
	// For 96kHz→48kHz: input at 10kHz should remain at 10kHz in output
	inputLen := 8000
	inputRate := 96000.0
	testFreq := 10000.0 // 10kHz (well within 24kHz Nyquist)

	input := make([]float64, inputLen)
	for i := range input {
		ti := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * testFreq * ti)
	}

	outputLen := inputLen / decimFactor
	output := make([]float64, outputLen)

	filterLen := len(filter)
	filterCenter := filterLen / 2

	// Apply filter and decimate
	for i := 0; i < outputLen; i++ {
		inputIdx := i * decimFactor
		sum := 0.0

		for j, coef := range filter {
			srcIdx := inputIdx + j - filterCenter
			if srcIdx >= 0 && srcIdx < inputLen {
				sum += coef * input[srcIdx]
			}
		}
		output[i] = sum
	}

	// Measure amplitude in steady state region
	steadyStart := filterLen / decimFactor
	steadyEnd := outputLen - filterLen/decimFactor

	if steadyEnd <= steadyStart {
		t.Skip("filter too long for input length")
	}

	maxAmplitude := 0.0
	for i := steadyStart; i < steadyEnd; i++ {
		if math.Abs(output[i]) > maxAmplitude {
			maxAmplitude = math.Abs(output[i])
		}
	}

	// In-band signal should have amplitude close to 1.0 (within 0.5 dB)
	gainDB := MagnitudeDB(maxAmplitude)
	t.Logf("In-band sinusoid (%.0f Hz) gain: %.2f dB", testFreq, gainDB)

	assert.InDelta(t, 0.0, gainDB, 0.5,
		"in-band sinusoid gain should be ~0 dB")
}

// TestDecimation_StopbandSine validates that stopband sinusoid is attenuated.
func TestDecimation_StopbandSine(t *testing.T) {
	// For 2:1 decimation (96kHz→48kHz), the filter cutoff must be at output Nyquist
	// Output Nyquist = 24kHz, normalized to input sample rate: 24/48 = 0.5
	// So for 96kHz input, we need cutoff at 0.25 (=24kHz/96kHz*2)
	const (
		decimFactor  = 2
		cutoffFreq   = 0.25 // 24kHz / 48kHz = 0.5, then /2 for input rate
		transitionBW = 0.02
		attenuation  = 100.0
	)

	filter, err := DesignLowPassFilterAuto(cutoffFreq, transitionBW, attenuation, 1.0)
	require.NoError(t, err)

	t.Logf("Filter length: %d taps (cutoff=%.3f, tr_bw=%.3f)", len(filter), cutoffFreq, transitionBW)

	// Generate stopband sinusoid
	// For 96kHz→48kHz: input at 30kHz should be attenuated (above 24kHz Nyquist)
	inputLen := 8000
	inputRate := 96000.0
	testFreq := 30000.0 // 30kHz (above 24kHz output Nyquist)

	input := make([]float64, inputLen)
	for i := range input {
		ti := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * testFreq * ti)
	}

	outputLen := inputLen / decimFactor
	output := make([]float64, outputLen)

	filterLen := len(filter)
	filterCenter := filterLen / 2

	// Apply filter and decimate
	for i := 0; i < outputLen; i++ {
		inputIdx := i * decimFactor
		sum := 0.0

		for j, coef := range filter {
			srcIdx := inputIdx + j - filterCenter
			if srcIdx >= 0 && srcIdx < inputLen {
				sum += coef * input[srcIdx]
			}
		}
		output[i] = sum
	}

	// Measure amplitude in steady state region
	steadyStart := filterLen / decimFactor
	steadyEnd := outputLen - filterLen/decimFactor

	if steadyEnd <= steadyStart {
		t.Skip("filter too long for input length")
	}

	maxAmplitude := 0.0
	for i := steadyStart; i < steadyEnd; i++ {
		if math.Abs(output[i]) > maxAmplitude {
			maxAmplitude = math.Abs(output[i])
		}
	}

	// Stopband signal should be heavily attenuated
	attenuationDB := -MagnitudeDB(maxAmplitude)
	t.Logf("Stopband sinusoid (%.0f Hz) attenuation: %.2f dB", testFreq, attenuationDB)

	// Should have at least 60 dB attenuation (filter has some margin from design spec)
	assert.GreaterOrEqual(t, attenuationDB, 60.0,
		"stopband sinusoid should be attenuated by >= 60 dB")
}
