package mathutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tphakala/go-audio-resampler/internal/testutil"
)

// TestBesselI0 tests BesselI0 against known values.
func TestBesselI0(t *testing.T) {
	tests := []struct {
		name      string
		x         float64
		expected  float64
		tolerance float64
	}{
		{"Zero", 0.0, 1.0, 1e-15},
		{"Small positive", 0.5, 1.063483344, 1e-7},
		{"One", 1.0, 1.266065848, 1e-7},
		{"Two", 2.0, 2.279585307, 1e-7},
		{"Three", 3.0, 4.880792565, 1e-7},
		{"Boundary 3.75", 3.75, 9.118945994, 1e-7},
		{"Four", 4.0, 11.30192217, 1e-7},
		{"Five", 5.0, 27.23987183, 1e-7},
		{"Ten", 10.0, 2815.716628, 1e-6},
		{"Twenty", 20.0, 4.355826e7, 1e-1},
		{"Small negative", -0.5, 1.063483344, 1e-7},
		{"Negative one", -1.0, 1.266065848, 1e-7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := BesselI0(tt.x)
			testutil.AssertRelativeError(t, tt.expected, result, tt.tolerance)
		})
	}
}

// TestBesselI0_Symmetry tests I₀(x) = I₀(-x) (even function property).
func TestBesselI0_Symmetry(t *testing.T) {
	testValues := []float64{0.1, 1.0, 2.5, 5.0, 10.0}

	for _, x := range testValues {
		pos := BesselI0(x)
		neg := BesselI0(-x)
		assert.InDelta(t, pos, neg, 1e-10,
			"BesselI0 not symmetric: I₀(%v)=%v, I₀(%v)=%v", x, pos, -x, neg)
	}
}

// TestBesselI0_AtZero tests I₀(0) = 1.
func TestBesselI0_AtZero(t *testing.T) {
	result := BesselI0(0)
	assert.InDelta(t, 1.0, result, 1e-15, "BesselI0(0) should be 1.0")
}

// TestBesselI0_Monotonic tests I₀(x) is monotonically increasing for x > 0.
func TestBesselI0_Monotonic(t *testing.T) {
	prev := BesselI0(0)
	for x := 0.1; x < 10.0; x += 0.1 {
		curr := BesselI0(x)
		assert.Greater(t, curr, prev,
			"BesselI0 not monotonically increasing at x=%v: %v <= %v", x, curr, prev)
		prev = curr
	}
}

// BenchmarkBesselI0_Small benchmarks BesselI0 for small values.
func BenchmarkBesselI0_Small(b *testing.B) {
	x := 1.5
	for b.Loop() {
		_ = BesselI0(x)
	}
}

// BenchmarkBesselI0_Large benchmarks BesselI0 for large values.
func BenchmarkBesselI0_Large(b *testing.B) {
	x := 10.0
	for b.Loop() {
		_ = BesselI0(x)
	}
}

// TestKaiserBeta tests Kaiser beta calculation.
func TestKaiserBeta(t *testing.T) {
	tests := []struct {
		name        string
		attenuation float64
		expectedMin float64
		expectedMax float64
	}{
		{"20dB", 20.0, 0.0, 0.1},
		{"50dB", 50.0, 4.5, 4.6},
		{"60dB", 60.0, 5.6, 5.7},
		{"80dB", 80.0, 7.8, 7.9},
		{"100dB", 100.0, 10.0, 10.1},
		{"120dB", 120.0, 12.2, 12.3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			beta := KaiserBeta(tt.attenuation)
			testutil.AssertInRange(t, beta, tt.expectedMin, tt.expectedMax)
		})
	}
}

// TestKaiserBeta_Monotonic tests KaiserBeta is monotonically increasing.
func TestKaiserBeta_Monotonic(t *testing.T) {
	prevBeta := KaiserBeta(20.0)
	for att := 25.0; att <= 150.0; att += 5.0 {
		beta := KaiserBeta(att)
		assert.GreaterOrEqual(t, beta, prevBeta,
			"KaiserBeta not monotonic at att=%v: %v < %v", att, beta, prevBeta)
		prevBeta = beta
	}
}

// TestKaiserAttenuation_Inverse tests KaiserAttenuation is approximately inverse of KaiserBeta.
func TestKaiserAttenuation_Inverse(t *testing.T) {
	attenuations := []float64{60.0, 80.0, 100.0, 120.0}

	for _, origAtt := range attenuations {
		beta := KaiserBeta(origAtt)
		recoveredAtt := KaiserAttenuation(beta)

		// Should be within 5% for high attenuations
		testutil.AssertRelativeError(t, origAtt, recoveredAtt, 0.05)
	}
}

// TestEstimateFilterLength tests that EstimateFilterLength produces reasonable results.
func TestEstimateFilterLength(t *testing.T) {
	tests := []struct {
		name         string
		attenuation  float64
		transitionBW float64
		minTaps      int
		maxTaps      int
	}{
		{"CD quality", 96.0, 0.1, 60, 80},
		{"High quality", 120.0, 0.05, 150, 200},
		{"Very high", 150.0, 0.02, 450, 550},
		{"Wide transition", 80.0, 0.2, 20, 35},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			taps := EstimateFilterLength(tt.attenuation, tt.transitionBW)

			// Should be odd
			assert.Equal(t, 1, taps%2, "Filter length should be odd: %d", taps)

			// Should be in reasonable range
			assert.GreaterOrEqual(t, taps, tt.minTaps,
				"Filter length %d below minimum %d", taps, tt.minTaps)
			assert.LessOrEqual(t, taps, tt.maxTaps,
				"Filter length %d exceeds maximum %d", taps, tt.maxTaps)
		})
	}
}

// TestEstimateFilterLength_EdgeCases tests EstimateFilterLength edge cases.
func TestEstimateFilterLength_EdgeCases(t *testing.T) {
	// Zero transition bandwidth should not crash
	taps := EstimateFilterLength(100.0, 0.0)
	assert.GreaterOrEqual(t, taps, 3, "Filter length too small for edge case: %d", taps)

	// Very small attenuation
	taps = EstimateFilterLength(10.0, 0.1)
	assert.GreaterOrEqual(t, taps, 3, "Filter length too small for low attenuation: %d", taps)

	// Very high attenuation with narrow transition
	taps = EstimateFilterLength(200.0, 0.001)
	assert.LessOrEqual(t, taps, 8191, "Filter length exceeds maximum: %d", taps)
}

// BenchmarkKaiserBeta benchmarks KaiserBeta.
func BenchmarkKaiserBeta(b *testing.B) {
	for b.Loop() {
		_ = KaiserBeta(100.0)
	}
}

// BenchmarkEstimateFilterLength benchmarks EstimateFilterLength.
func BenchmarkEstimateFilterLength(b *testing.B) {
	for b.Loop() {
		_ = EstimateFilterLength(120.0, 0.05)
	}
}

// TestBesselI1 tests BesselI1 (internal function) against known values.
func TestBesselI1(t *testing.T) {
	tests := []struct {
		x         float64
		expected  float64
		tolerance float64
	}{
		{0.0, 0.0, 1e-15},
		{0.5, 0.257894303, 1e-7},
		{1.0, 0.565159098, 1e-7},
		{2.0, 1.590636857, 1e-7},
		{5.0, 24.33564185, 1e-7},
	}

	for _, tt := range tests {
		result := besselI1(tt.x)
		assert.InDelta(t, tt.expected, result, tt.tolerance,
			"besselI1(%v) = %v, want %v", tt.x, result, tt.expected)
	}
}

// TestBesselI1_Antisymmetry tests I₁(-x) = -I₁(x) (odd function property).
func TestBesselI1_Antisymmetry(t *testing.T) {
	testValues := []float64{0.1, 1.0, 2.5, 5.0}

	for _, x := range testValues {
		pos := besselI1(x)
		neg := besselI1(-x)
		assert.InDelta(t, -pos, neg, 1e-10,
			"besselI1 not antisymmetric: I₁(%v)=%v, I₁(%v)=%v", x, pos, -x, neg)
	}
}

// TestBesselI0Ratio tests BesselI0Ratio.
func TestBesselI0Ratio(t *testing.T) {
	tests := []struct {
		name      string
		x         float64
		tolerance float64
	}{
		{"Near zero", 1e-12, 1e-10},
		{"Small", 0.5, 1e-9},
		{"One", 1.0, 1e-9},
		{"Five", 5.0, 1e-9},
		{"Large", 60.0, 1e-6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ratio := BesselI0Ratio(tt.x)

			// Should be in range (0, 1)
			assert.Greater(t, ratio, 0.0, "BesselI0Ratio(%v) should be > 0", tt.x)
			assert.Less(t, ratio, 1.0, "BesselI0Ratio(%v) should be < 1", tt.x)

			// Verify by computing I₁/I₀ directly (except for very large x)
			if tt.x < 50 {
				i0 := BesselI0(tt.x)
				i1 := besselI1(tt.x)
				expected := i1 / i0

				testutil.AssertRelativeError(t, expected, ratio, tt.tolerance)
			}
		})
	}
}
