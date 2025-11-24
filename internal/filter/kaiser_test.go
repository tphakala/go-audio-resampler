package filter

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/go-audio-resampler/internal/mathutil"
	"github.com/tphakala/go-audio-resampler/internal/testutil"
)

const (
	// Test tolerances
	defaultTolerance   = 1e-10
	magnitudeTolerance = 1e-2
	windowTolerance    = 1e-10

	// Test window parameters
	testWindowLength11 = 11
	testWindowLength21 = 21
	testWindowLength51 = 51
	testBeta5          = 5.0
	testBeta8          = 8.653728
	testBeta10         = 10.0

	// Test filter parameters
	testAttenuation80  = 80.0
	testAttenuation100 = 100.0
	testAttenuation120 = 120.0
	testCutoff0_25     = 0.25
	testCutoff0_4      = 0.4
	testTransitionBW   = 0.05
	testGainUnity      = 1.0

	// Frequency response test parameters
	testNumPoints512  = 512
	testNumPoints1024 = 1024
	testPassbandFreq  = 0.2
	testStopbandFreq  = 0.3

	// dB thresholds
	passbandRippleDB = 0.1
	stopbandFloorDB  = -100.0
)

// TestKaiserWindow_Symmetry verifies that Kaiser window is symmetric.
func TestKaiserWindow_Symmetry(t *testing.T) {
	tests := []struct {
		name   string
		length int
		beta   float64
	}{
		{"length_11_beta_5", testWindowLength11, testBeta5},
		{"length_21_beta_8", testWindowLength21, testBeta8},
		{"length_51_beta_10", testWindowLength51, testBeta10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			window := KaiserWindow(tt.length, tt.beta)

			assert.Len(t, window, tt.length, "window length mismatch")
			testutil.AssertSymmetric(t, window, windowTolerance)
		})
	}
}

// TestKaiserWindow_CenterTap verifies that center tap is maximum.
func TestKaiserWindow_CenterTap(t *testing.T) {
	window := KaiserWindow(testWindowLength21, testBeta8)

	testutil.AssertCenterIsMax(t, window)

	// Center value should be close to 1.0 (I₀(β)/I₀(β) = 1)
	centerIdx := testWindowLength21 / 2
	assert.InDelta(t, 1.0, window[centerIdx], windowTolerance,
		"center value should be ~1.0")
}

// TestKaiserWindow_EdgeCases tests edge cases.
func TestKaiserWindow_EdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		length int
		beta   float64
		want   int
	}{
		{"zero_length", 0, testBeta5, 0},
		{"negative_length", -1, testBeta5, 0},
		{"length_one", 1, testBeta5, 1},
		{"length_two", 2, testBeta5, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			window := KaiserWindow(tt.length, tt.beta)
			assert.Len(t, window, tt.want, "window length mismatch")

			if tt.length == 1 && len(window) == 1 {
				// Single tap should be 1.0
				assert.InDelta(t, 1.0, window[0], windowTolerance,
					"single tap value should be 1.0")
			}
		})
	}
}

// TestFilterParams_Validate tests parameter validation.
func TestFilterParams_Validate(t *testing.T) {
	tests := []struct {
		name    string
		params  FilterParams
		wantErr bool
	}{
		{
			name: "valid_params",
			params: FilterParams{
				NumTaps:     101,
				CutoffFreq:  testCutoff0_25,
				Attenuation: testAttenuation80,
				Gain:        testGainUnity,
			},
			wantErr: false,
		},
		{
			name: "too_few_taps",
			params: FilterParams{
				NumTaps:     1,
				CutoffFreq:  testCutoff0_25,
				Attenuation: testAttenuation80,
				Gain:        testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "too_many_taps",
			params: FilterParams{
				NumTaps:     10000,
				CutoffFreq:  testCutoff0_25,
				Attenuation: testAttenuation80,
				Gain:        testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "cutoff_too_low",
			params: FilterParams{
				NumTaps:     101,
				CutoffFreq:  0.0,
				Attenuation: testAttenuation80,
				Gain:        testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "cutoff_too_high",
			params: FilterParams{
				NumTaps:     101,
				CutoffFreq:  0.5,
				Attenuation: testAttenuation80,
				Gain:        testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "negative_attenuation",
			params: FilterParams{
				NumTaps:     101,
				CutoffFreq:  testCutoff0_25,
				Attenuation: -10.0,
				Gain:        testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "zero_gain",
			params: FilterParams{
				NumTaps:     101,
				CutoffFreq:  testCutoff0_25,
				Attenuation: testAttenuation80,
				Gain:        0.0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.params.Validate()
			if tt.wantErr {
				assert.Error(t, err, "expected validation error")
			} else {
				assert.NoError(t, err, "unexpected validation error")
			}
		})
	}
}

// TestDesignLowPassFilter_Symmetry verifies filter symmetry.
func TestDesignLowPassFilter_Symmetry(t *testing.T) {
	params := FilterParams{
		NumTaps:     101,
		CutoffFreq:  testCutoff0_25,
		Attenuation: testAttenuation80,
		Gain:        testGainUnity,
	}

	filter, err := DesignLowPassFilter(params)
	require.NoError(t, err, "DesignLowPassFilter failed")

	assert.Len(t, filter, params.NumTaps, "filter length mismatch")
	testutil.AssertSymmetric(t, filter, defaultTolerance)
}

// TestDesignLowPassFilter_DCGain verifies that DC gain equals target gain.
func TestDesignLowPassFilter_DCGain(t *testing.T) {
	tests := []struct {
		name string
		gain float64
	}{
		{"gain_1", testGainUnity},
		{"gain_2", 2.0},
		{"gain_0_5", 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := FilterParams{
				NumTaps:     101,
				CutoffFreq:  testCutoff0_25,
				Attenuation: testAttenuation80,
				Gain:        tt.gain,
			}

			filter, err := DesignLowPassFilter(params)
			require.NoError(t, err, "DesignLowPassFilter failed")

			testutil.AssertDCGain(t, filter, tt.gain, defaultTolerance)
		})
	}
}

// TestDesignLowPassFilter_FrequencyResponse verifies frequency response characteristics.
func TestDesignLowPassFilter_FrequencyResponse(t *testing.T) {
	// Use automatic filter design to get proper relationship between
	// attenuation, transition bandwidth, and filter length
	const (
		testTransition = 0.05
		testCutoff     = 0.25
		testAtten      = 80.0
	)

	filter, err := DesignLowPassFilterAuto(testCutoff, testTransition, testAtten, testGainUnity)
	require.NoError(t, err, "DesignLowPassFilterAuto failed")

	// Compute frequency response
	response := ComputeFrequencyResponse(filter, testNumPoints512)

	assert.Len(t, response.Frequencies, testNumPoints512, "response length mismatch")

	// Check passband (0 to cutoff - transition/2)
	passbandEnd := testCutoff - testTransition/2.0
	for i, freq := range response.Frequencies {
		if freq > passbandEnd {
			break
		}

		// Magnitude should be close to 1.0 in passband
		mag := response.Magnitude[i]
		magDB := MagnitudeDB(mag)

		assert.LessOrEqual(t, math.Abs(magDB), passbandRippleDB,
			"passband ripple at freq=%f: %f dB exceeds %f dB",
			freq, magDB, passbandRippleDB)
	}

	// Check stopband (cutoff + transition/2 to Nyquist)
	// The actual attenuation should be close to the design spec (80 dB)
	const stopbandTolerance = 10.0 // Allow 10 dB margin
	stopbandStart := testCutoff + testTransition/2.0
	stopbandTarget := -testAtten + stopbandTolerance

	for i, freq := range response.Frequencies {
		if freq < stopbandStart {
			continue
		}

		// Magnitude should be well attenuated in stopband
		mag := response.Magnitude[i]
		magDB := MagnitudeDB(mag)

		assert.LessOrEqual(t, magDB, stopbandTarget,
			"insufficient stopband attenuation at freq=%f: %f dB exceeds %f dB",
			freq, magDB, stopbandTarget)
	}
}

// TestDesignLowPassFilter_HigherAttenuation verifies that higher attenuation specs work.
func TestDesignLowPassFilter_HigherAttenuation(t *testing.T) {
	tests := []struct {
		name        string
		attenuation float64
		numTaps     int
	}{
		{"attenuation_80dB", testAttenuation80, 101},
		{"attenuation_100dB", testAttenuation100, 151},
		{"attenuation_120dB", testAttenuation120, 201},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := FilterParams{
				NumTaps:     tt.numTaps,
				CutoffFreq:  testCutoff0_25,
				Attenuation: tt.attenuation,
				Gain:        testGainUnity,
			}

			filter, err := DesignLowPassFilter(params)
			require.NoError(t, err, "DesignLowPassFilter failed")

			assert.Len(t, filter, tt.numTaps, "filter length mismatch")

			// Verify beta parameter is reasonable
			beta := mathutil.KaiserBeta(tt.attenuation)
			assert.Positive(t, beta, "beta should be > 0")
		})
	}
}

// TestDesignLowPassFilterAuto tests automatic filter length calculation.
func TestDesignLowPassFilterAuto(t *testing.T) {
	tests := []struct {
		name         string
		cutoffFreq   float64
		transitionBW float64
		attenuation  float64
		gain         float64
	}{
		{"standard", testCutoff0_25, testTransitionBW, testAttenuation80, testGainUnity},
		{"narrow_transition", testCutoff0_4, 0.02, testAttenuation100, testGainUnity},
		{"wide_transition", testCutoff0_25, 0.1, testAttenuation80, testGainUnity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter, err := DesignLowPassFilterAuto(tt.cutoffFreq, tt.transitionBW, tt.attenuation, tt.gain)
			require.NoError(t, err, "DesignLowPassFilterAuto failed")

			// Filter should have odd length (symmetric FIR)
			testutil.AssertOddLength(t, filter)

			// Verify DC gain
			testutil.AssertDCGain(t, filter, tt.gain, defaultTolerance)

			// Verify symmetry
			testutil.AssertSymmetric(t, filter, defaultTolerance)
		})
	}
}

// TestComputeFrequencyResponse tests frequency response calculation.
func TestComputeFrequencyResponse(t *testing.T) {
	// Simple 3-tap averaging filter: [0.25, 0.5, 0.25]
	const (
		tap0 = 0.25
		tap1 = 0.5
		tap2 = 0.25
	)
	coeffs := []float64{tap0, tap1, tap2}

	response := ComputeFrequencyResponse(coeffs, testNumPoints512)

	assert.Len(t, response.Frequencies, testNumPoints512, "frequencies length mismatch")
	assert.Len(t, response.Magnitude, testNumPoints512, "magnitude length mismatch")
	assert.Len(t, response.Phase, testNumPoints512, "phase length mismatch")

	// DC response (freq=0) should equal sum of coefficients
	expectedDC := tap0 + tap1 + tap2
	assert.InDelta(t, expectedDC, response.Magnitude[0], magnitudeTolerance,
		"DC magnitude mismatch")

	// Nyquist response (freq=0.5) for this filter should be zero
	// because [0.25, 0.5, 0.25] alternating signs = 0.25 - 0.5 + 0.25 = 0
	nyquistIdx := testNumPoints512 - 1
	nyquistMag := response.Magnitude[nyquistIdx]
	assert.LessOrEqual(t, nyquistMag, magnitudeTolerance,
		"Nyquist magnitude should be ~0")
}

// TestMagnitudeDB tests linear to dB conversion.
func TestMagnitudeDB(t *testing.T) {
	const (
		mag1    = 1.0
		mag0_5  = 0.5
		mag0_1  = 0.1
		mag0_01 = 0.01

		db1    = 0.0
		db0_5  = -6.0206
		db0_1  = -20.0
		db0_01 = -40.0

		dbTolerance = 0.01
	)

	tests := []struct {
		name string
		mag  float64
		want float64
	}{
		{"magnitude_1", mag1, db1},
		{"magnitude_0_5", mag0_5, db0_5},
		{"magnitude_0_1", mag0_1, db0_1},
		{"magnitude_0_01", mag0_01, db0_01},
		{"magnitude_zero", 0.0, -200.0}, // Should clip to minimum
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MagnitudeDB(tt.mag)
			assert.InDelta(t, tt.want, got, dbTolerance,
				"MagnitudeDB(%f) = %f dB, want %f dB", tt.mag, got, tt.want)
		})
	}
}

// BenchmarkKaiserWindow benchmarks window generation.
func BenchmarkKaiserWindow(b *testing.B) {
	benchmarks := []struct {
		name   string
		length int
		beta   float64
	}{
		{"length_51", testWindowLength51, testBeta8},
		{"length_101", 101, testBeta8},
		{"length_201", 201, testBeta10},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			for b.Loop() {
				_ = KaiserWindow(bm.length, bm.beta)
			}
		})
	}
}

// BenchmarkDesignLowPassFilter benchmarks filter design.
func BenchmarkDesignLowPassFilter(b *testing.B) {
	params := FilterParams{
		NumTaps:     201,
		CutoffFreq:  testCutoff0_25,
		Attenuation: testAttenuation100,
		Gain:        testGainUnity,
	}

	b.ResetTimer()
	for b.Loop() {
		_, _ = DesignLowPassFilter(params)
	}
}

// BenchmarkComputeFrequencyResponse benchmarks frequency response calculation.
func BenchmarkComputeFrequencyResponse(b *testing.B) {
	params := FilterParams{
		NumTaps:     201,
		CutoffFreq:  testCutoff0_25,
		Attenuation: testAttenuation100,
		Gain:        testGainUnity,
	}

	filter, _ := DesignLowPassFilter(params)

	b.ResetTimer()
	for b.Loop() {
		_ = ComputeFrequencyResponse(filter, testNumPoints1024)
	}
}
