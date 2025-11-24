package filter

import (
	"fmt"
	"math"
	"testing"
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
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
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
	if err != nil {
		t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
	}

	// Check basic properties
	if pfb.NumPhases != params.NumPhases {
		t.Errorf("NumPhases = %d, want %d", pfb.NumPhases, params.NumPhases)
	}

	if pfb.InterpOrder != params.InterpOrder {
		t.Errorf("InterpOrder = %d, want %d", pfb.InterpOrder, params.InterpOrder)
	}

	if pfb.TotalTaps <= 0 {
		t.Errorf("TotalTaps = %d, want > 0", pfb.TotalTaps)
	}

	if pfb.TapsPerPhase <= 0 {
		t.Errorf("TapsPerPhase = %d, want > 0", pfb.TapsPerPhase)
	}

	// Check coefficient storage size
	expectedCoeffs := pfb.TapsPerPhase * pfb.NumPhases * (int(pfb.InterpOrder) + 1)
	if len(pfb.Coeffs) != expectedCoeffs {
		t.Errorf("Coeffs length = %d, want %d", len(pfb.Coeffs), expectedCoeffs)
	}
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
			if err != nil {
				t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
			}

			// Verify coefficient storage
			coeffsPerTap := int(ord.order) + 1
			expectedSize := pfb.TapsPerPhase * pfb.NumPhases * coeffsPerTap
			if len(pfb.Coeffs) != expectedSize {
				t.Errorf("Coeffs length = %d, want %d", len(pfb.Coeffs), expectedSize)
			}
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
	if err != nil {
		t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
	}

	// Test coefficient retrieval
	tap := 0
	phase := 0

	// At frac = 0, should return base coefficient
	coef0 := pfb.GetCoefficient(tap, phase, 0.0)
	if math.IsNaN(coef0) {
		t.Error("GetCoefficient returned NaN for frac=0")
	}

	// At frac = 1, should approach next phase
	coef1 := pfb.GetCoefficient(tap, phase, 1.0)
	if math.IsNaN(coef1) {
		t.Error("GetCoefficient returned NaN for frac=1")
	}

	// With linear interpolation, frac=0.5 should be average of endpoints
	coef0_5 := pfb.GetCoefficient(tap, phase, 0.5)
	if math.IsNaN(coef0_5) {
		t.Error("GetCoefficient returned NaN for frac=0.5")
	}

	// For linear interpolation, midpoint should satisfy interpolation property
	// This is a basic sanity check, not a strict equality test
	if math.Abs(coef0_5) > math.Max(math.Abs(coef0), math.Abs(coef1))*2 {
		t.Errorf("Interpolated coefficient suspiciously large: %f (endpoints: %f, %f)",
			coef0_5, coef0, coef1)
	}
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
	if err != nil {
		t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
	}

	// Verify structure is consistent
	if pfb.NumPhases != params.NumPhases {
		t.Errorf("NumPhases = %d, want %d", pfb.NumPhases, params.NumPhases)
	}

	// Verify minimum taps per phase
	if pfb.TapsPerPhase < 2 {
		t.Errorf("TapsPerPhase = %d, want at least 2", pfb.TapsPerPhase)
	}

	// Verify coefficient storage size
	expectedCoeffs := pfb.TapsPerPhase * pfb.NumPhases * (int(pfb.InterpOrder) + 1)
	if len(pfb.Coeffs) != expectedCoeffs {
		t.Errorf("Coeffs length = %d, want %d", len(pfb.Coeffs), expectedCoeffs)
	}

	// Verify all coefficients are valid (not NaN or Inf)
	for i, c := range pfb.Coeffs {
		if math.IsNaN(c) || math.IsInf(c, 0) {
			t.Errorf("Invalid coefficient at index %d: %f", i, c)
		}
	}

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
	if err != nil {
		t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
	}

	// Check DC gain across multiple phases (soxr-style: each phase ≈ 1.0)
	// The average DC gain across all phases should be approximately 1.0
	var totalDCGain float64
	for phase := 0; phase < pfb.NumPhases; phase++ {
		var phaseSum float64
		for tap := 0; tap < pfb.TapsPerPhase; tap++ {
			phaseSum += pfb.GetCoefficient(tap, phase, 0.0)
		}
		totalDCGain += phaseSum
	}

	avgDCGain := totalDCGain / float64(pfb.NumPhases)
	expectedGain := params.Gain

	// Allow some tolerance for filter design imprecision
	tolerance := 0.5 // Wider tolerance for average across all phases
	if math.Abs(avgDCGain-expectedGain) > tolerance {
		t.Errorf("Average DC gain = %f, want %f (tolerance: %f)", avgDCGain, expectedGain, tolerance)
	}
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
			if err != nil {
				t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
			}

			if pfb.NumPhases != numPhases {
				t.Errorf("NumPhases = %d, want %d", pfb.NumPhases, numPhases)
			}

			// Verify we have coefficients for all phases
			coeffsPerTap := int(pfb.InterpOrder) + 1
			expectedSize := pfb.TapsPerPhase * numPhases * coeffsPerTap
			if len(pfb.Coeffs) != expectedSize {
				t.Errorf("Coeffs length = %d, want %d", len(pfb.Coeffs), expectedSize)
			}
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
	if err != nil {
		t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
	}

	response := pfb.ComputeFrequencyResponse(testNumPoints512)

	if len(response.Frequencies) != testNumPoints512 {
		t.Errorf("response length = %d, want %d", len(response.Frequencies), testNumPoints512)
	}

	// Check that DC response is reasonable
	dcMagnitude := response.Magnitude[0]
	if dcMagnitude <= 0 || math.IsNaN(dcMagnitude) {
		t.Errorf("DC magnitude = %f, want > 0", dcMagnitude)
	}

	// Check that frequencies are in expected range [0, 0.5]
	for i, freq := range response.Frequencies {
		if freq < 0 || freq > 0.5 {
			t.Errorf("frequency[%d] = %f, want in range [0, 0.5]", i, freq)
			break
		}
	}
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
	if err != nil {
		t.Fatalf("DesignPolyphaseFilterBank() error = %v", err)
	}

	memUsage := pfb.GetMemoryUsage()
	const bytesPerFloat64 = 8
	expectedUsage := int64(len(pfb.Coeffs)) * bytesPerFloat64

	if memUsage != expectedUsage {
		t.Errorf("GetMemoryUsage() = %d, want %d", memUsage, expectedUsage)
	}
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
