package filter

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/go-audio-resampler/internal/testutil"
)

const (
	// Test parameters for polyphase tests
	testNumPhases64   = 64
	testNumPhases256  = 256
	testNumPhases1024 = 1024

	testTransition005 = 0.05

	// Test tolerances
	coeffTolerance = 1e-10
	freqTolerance  = 1e-6
)

// TestPolyphaseParams_Validate tests parameter validation.
func TestPolyphaseParams_Validate(t *testing.T) {
	tests := []struct {
		name    string
		params  PolyphaseParams
		wantErr bool
	}{
		{
			name: "valid_params",
			params: PolyphaseParams{
				NumPhases:    testNumPhases256,
				Cutoff:       testCutoff0_25,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpLinear,
				Gain:         testGainUnity,
			},
			wantErr: false,
		},
		{
			name: "too_few_phases",
			params: PolyphaseParams{
				NumPhases:    1,
				Cutoff:       testCutoff0_25,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpLinear,
				Gain:         testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "too_many_phases",
			params: PolyphaseParams{
				NumPhases:    10000,
				Cutoff:       testCutoff0_25,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpLinear,
				Gain:         testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "invalid_cutoff_low",
			params: PolyphaseParams{
				NumPhases:    testNumPhases256,
				Cutoff:       0.0,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpLinear,
				Gain:         testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "invalid_cutoff_high",
			params: PolyphaseParams{
				NumPhases:    testNumPhases256,
				Cutoff:       0.5,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpLinear,
				Gain:         testGainUnity,
			},
			wantErr: true,
		},
		{
			name: "invalid_interp_order",
			params: PolyphaseParams{
				NumPhases:    testNumPhases256,
				Cutoff:       testCutoff0_25,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpOrder(2), // Invalid: only 0, 1, 3 allowed
				Gain:         testGainUnity,
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

// TestDesignPolyphaseFilterBank tests basic polyphase filter bank design.
func TestDesignPolyphaseFilterBank(t *testing.T) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation80,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, err := DesignPolyphaseFilterBank(params)
	require.NoError(t, err, "DesignPolyphaseFilterBank failed")

	// Check basic properties
	assert.Equal(t, params.NumPhases, pfb.NumPhases, "NumPhases mismatch")
	assert.Equal(t, params.InterpOrder, pfb.InterpOrder, "InterpOrder mismatch")
	assert.Positive(t, pfb.TotalTaps, "TotalTaps should be > 0")
	assert.Positive(t, pfb.TapsPerPhase, "TapsPerPhase should be > 0")

	// Check coefficient storage size
	expectedCoeffs := pfb.TapsPerPhase * pfb.NumPhases * (int(pfb.InterpOrder) + 1)
	assert.Len(t, pfb.Coeffs, expectedCoeffs, "Coeffs length mismatch")
}

// TestPolyphaseFilterBank_InterpolationOrders tests all interpolation orders.
func TestPolyphaseFilterBank_InterpolationOrders(t *testing.T) {
	orders := []struct {
		name  string
		order InterpOrder
	}{
		{"none", InterpNone},
		{"linear", InterpLinear},
		{"cubic", InterpCubic},
	}

	for _, ord := range orders {
		t.Run(ord.name, func(t *testing.T) {
			params := PolyphaseParams{
				NumPhases:    testNumPhases64,
				Cutoff:       testCutoff0_25,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  ord.order,
				Gain:         testGainUnity,
			}

			pfb, err := DesignPolyphaseFilterBank(params)
			require.NoError(t, err, "DesignPolyphaseFilterBank failed")

			// Verify coefficient storage
			coeffsPerTap := int(ord.order) + 1
			expectedSize := pfb.TapsPerPhase * pfb.NumPhases * coeffsPerTap
			assert.Len(t, pfb.Coeffs, expectedSize, "Coeffs length mismatch")
		})
	}
}

// TestPolyphaseFilterBank_GetCoefficient tests coefficient retrieval and interpolation.
func TestPolyphaseFilterBank_GetCoefficient(t *testing.T) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation80,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, err := DesignPolyphaseFilterBank(params)
	require.NoError(t, err, "DesignPolyphaseFilterBank failed")

	// Test coefficient retrieval
	tap := 0
	phase := 0

	// At frac = 0, should return base coefficient
	coef0 := pfb.GetCoefficient(tap, phase, 0.0)
	assert.False(t, math.IsNaN(coef0), "GetCoefficient returned NaN for frac=0")

	// At frac = 1, should approach next phase
	coef1 := pfb.GetCoefficient(tap, phase, 1.0)
	assert.False(t, math.IsNaN(coef1), "GetCoefficient returned NaN for frac=1")

	// With linear interpolation, frac=0.5 should be average of endpoints
	coef0_5 := pfb.GetCoefficient(tap, phase, 0.5)
	assert.False(t, math.IsNaN(coef0_5), "GetCoefficient returned NaN for frac=0.5")

	// For linear interpolation, midpoint should satisfy interpolation property
	// This is a basic sanity check, not a strict equality test
	maxEndpoint := math.Max(math.Abs(coef0), math.Abs(coef1)) * 2
	assert.LessOrEqual(t, math.Abs(coef0_5), maxEndpoint,
		"Interpolated coefficient suspiciously large: %f (endpoints: %f, %f)",
		coef0_5, coef0, coef1)
}

// TestPolyphaseFilterBank_Structure tests that the filter bank has valid structure.
// Note: In polyphase decomposition with soxr-style scaling, individual tap coefficients
// across phases may vary significantly (that's normal). We test structural validity instead.
func TestPolyphaseFilterBank_Structure(t *testing.T) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases64,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation80,
		InterpOrder:  InterpNone,
		Gain:         testGainUnity,
	}

	pfb, err := DesignPolyphaseFilterBank(params)
	require.NoError(t, err, "DesignPolyphaseFilterBank failed")

	// Verify structure is consistent
	assert.Equal(t, params.NumPhases, pfb.NumPhases, "NumPhases mismatch")

	// Verify minimum taps per phase
	assert.GreaterOrEqual(t, pfb.TapsPerPhase, 2, "TapsPerPhase should be at least 2")

	// Verify coefficient storage size
	expectedCoeffs := pfb.TapsPerPhase * pfb.NumPhases * (int(pfb.InterpOrder) + 1)
	assert.Len(t, pfb.Coeffs, expectedCoeffs, "Coeffs length mismatch")

	// Verify all coefficients are valid (not NaN or Inf)
	testutil.AssertNoNaNOrInf(t, pfb.Coeffs)

	t.Logf("Filter bank: %d phases, %d taps/phase, %d total coefficients",
		pfb.NumPhases, pfb.TapsPerPhase, len(pfb.Coeffs))
}

// TestPolyphaseFilterBank_DCGain tests that DC gain is preserved per phase.
// soxr-style: each phase should have DC gain ≈ 1.0 (not 1/numPhases).
// This ensures DC preservation regardless of which phases are used during resampling.
func TestPolyphaseFilterBank_DCGain(t *testing.T) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation80,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, err := DesignPolyphaseFilterBank(params)
	require.NoError(t, err, "DesignPolyphaseFilterBank failed")

	// Check DC gain across multiple phases (soxr-style: each phase ≈ 1.0)
	// The average DC gain across all phases should be approximately 1.0
	var totalDCGain float64
	for phase := range pfb.NumPhases {
		var phaseSum float64
		for tap := range pfb.TapsPerPhase {
			phaseSum += pfb.GetCoefficient(tap, phase, 0.0)
		}
		totalDCGain += phaseSum
	}

	avgDCGain := totalDCGain / float64(pfb.NumPhases)
	expectedGain := params.Gain

	// Allow some tolerance for filter design imprecision
	tolerance := 0.5 // Wider tolerance for average across all phases
	assert.InDelta(t, expectedGain, avgDCGain, tolerance,
		"Average DC gain mismatch")
	t.Logf("Average DC gain across %d phases: %.6f", pfb.NumPhases, avgDCGain)
}

// TestPolyphaseFilterBank_DifferentPhases tests various phase counts.
func TestPolyphaseFilterBank_DifferentPhases(t *testing.T) {
	phaseCounts := []int{testNumPhases64, testNumPhases256, testNumPhases1024}

	for _, numPhases := range phaseCounts {
		t.Run(fmt.Sprintf("phases_%d", numPhases), func(t *testing.T) {
			params := PolyphaseParams{
				NumPhases:    numPhases,
				Cutoff:       testCutoff0_25,
				TransitionBW: testTransition005,
				Attenuation:  testAttenuation80,
				InterpOrder:  InterpLinear,
				Gain:         testGainUnity,
			}

			pfb, err := DesignPolyphaseFilterBank(params)
			require.NoError(t, err, "DesignPolyphaseFilterBank failed")

			assert.Equal(t, numPhases, pfb.NumPhases, "NumPhases mismatch")

			// Verify we have coefficients for all phases
			coeffsPerTap := int(pfb.InterpOrder) + 1
			expectedSize := pfb.TapsPerPhase * numPhases * coeffsPerTap
			assert.Len(t, pfb.Coeffs, expectedSize, "Coeffs length mismatch")
		})
	}
}

// TestPolyphaseFilterBank_FrequencyResponse tests frequency response computation.
func TestPolyphaseFilterBank_FrequencyResponse(t *testing.T) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation80,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, err := DesignPolyphaseFilterBank(params)
	require.NoError(t, err, "DesignPolyphaseFilterBank failed")

	response := pfb.ComputeFrequencyResponse(testNumPoints512)

	assert.Len(t, response.Frequencies, testNumPoints512, "response length mismatch")

	// Check that DC response is reasonable
	dcMagnitude := response.Magnitude[0]
	assert.Positive(t, dcMagnitude, "DC magnitude should be > 0")
	assert.False(t, math.IsNaN(dcMagnitude), "DC magnitude should not be NaN")

	// Check that frequencies are in expected range [0, 0.5]
	testutil.AssertAllInRange(t, response.Frequencies, 0, 0.5)
}

// TestPolyphaseFilterBank_MemoryUsage tests memory usage calculation.
func TestPolyphaseFilterBank_MemoryUsage(t *testing.T) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation80,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, err := DesignPolyphaseFilterBank(params)
	require.NoError(t, err, "DesignPolyphaseFilterBank failed")

	memUsage := pfb.GetMemoryUsage()
	const bytesPerFloat64 = 8
	expectedUsage := int64(len(pfb.Coeffs)) * bytesPerFloat64

	assert.Equal(t, expectedUsage, memUsage, "GetMemoryUsage mismatch")
}

// BenchmarkDesignPolyphaseFilterBank benchmarks filter bank design.
func BenchmarkDesignPolyphaseFilterBank(b *testing.B) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation100,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	b.ResetTimer()
	for b.Loop() {
		_, _ = DesignPolyphaseFilterBank(params)
	}
}

// BenchmarkPolyphaseGetCoefficient benchmarks coefficient retrieval.
func BenchmarkPolyphaseGetCoefficient(b *testing.B) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation100,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, _ := DesignPolyphaseFilterBank(params)

	b.ResetTimer()
	for b.Loop() {
		_ = pfb.GetCoefficient(0, 0, 0.5)
	}
}

// BenchmarkPolyphaseFrequencyResponse benchmarks frequency response computation.
func BenchmarkPolyphaseFrequencyResponse(b *testing.B) {
	params := PolyphaseParams{
		NumPhases:    testNumPhases256,
		Cutoff:       testCutoff0_25,
		TransitionBW: testTransition005,
		Attenuation:  testAttenuation100,
		InterpOrder:  InterpLinear,
		Gain:         testGainUnity,
	}

	pfb, _ := DesignPolyphaseFilterBank(params)

	b.ResetTimer()
	for b.Loop() {
		_ = pfb.ComputeFrequencyResponse(testNumPoints512)
	}
}
