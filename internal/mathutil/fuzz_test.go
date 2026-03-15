package mathutil

import (
	"math"
	"testing"
)

func FuzzBesselI0(f *testing.F) {
	f.Add(0.0)
	f.Add(1.0)
	f.Add(-1.0)
	f.Add(3.75) // boundary between small/large arg
	f.Add(10.0)
	f.Add(100.0)
	f.Add(700.0) // near exp overflow
	f.Add(math.SmallestNonzeroFloat64)

	f.Fuzz(func(t *testing.T, x float64) {
		result := BesselI0(x)

		// Must not panic (implicit by reaching here)

		// I₀(x) is always positive for finite x
		if !math.IsNaN(x) && !math.IsInf(x, 0) {
			if math.IsNaN(result) {
				t.Errorf("BesselI0(%v) = NaN, want finite positive", x)
			}
			if result < 0 {
				t.Errorf("BesselI0(%v) = %v, want positive", x, result)
			}
		}

		// I₀(x) = I₀(-x) (even function)
		if !math.IsNaN(x) && !math.IsInf(x, 0) {
			pos := BesselI0(math.Abs(x))
			neg := BesselI0(-math.Abs(x))
			if pos != neg {
				t.Errorf("BesselI0 not symmetric: I₀(%v)=%v, I₀(%v)=%v",
					math.Abs(x), pos, -math.Abs(x), neg)
			}
		}

		// I₀(0) = 1
		if x == 0 {
			if result != 1.0 {
				t.Errorf("BesselI0(0) = %v, want 1.0", result)
			}
		}
	})
}

func FuzzKaiserBeta(f *testing.F) {
	f.Add(0.0)
	f.Add(10.0)
	f.Add(21.0) // boundary
	f.Add(50.0) // boundary
	f.Add(100.0)
	f.Add(150.0)
	f.Add(-1.0)

	f.Fuzz(func(t *testing.T, attenuation float64) {
		result := KaiserBeta(attenuation)

		if math.IsNaN(result) {
			t.Errorf("KaiserBeta(%v) = NaN", attenuation)
		}

		// β should be non-negative for non-negative attenuation
		if attenuation >= 0 && result < 0 {
			t.Errorf("KaiserBeta(%v) = %v, want non-negative", attenuation, result)
		}

		// β = 0 for attenuation <= 21
		if attenuation <= 21 && attenuation >= 0 && result != 0.0 {
			t.Errorf("KaiserBeta(%v) = %v, want 0", attenuation, result)
		}
	})
}

func FuzzKaiserBetaWithTrBw(f *testing.F) {
	f.Add(60.0, 0.01)
	f.Add(100.0, 0.1)
	f.Add(150.0, 0.001)
	f.Add(50.0, 0.0)
	f.Add(0.0, 0.05)
	f.Add(200.0, 0.5)

	f.Fuzz(func(t *testing.T, attenuation, trBw float64) {
		result := KaiserBetaWithTrBw(attenuation, trBw)

		if math.IsNaN(result) && !math.IsNaN(attenuation) && !math.IsNaN(trBw) {
			t.Errorf("KaiserBetaWithTrBw(%v, %v) = NaN", attenuation, trBw)
		}
	})
}

func FuzzEstimateFilterLength(f *testing.F) {
	f.Add(100.0, 0.1)
	f.Add(50.0, 0.01)
	f.Add(150.0, 0.001)
	f.Add(0.0, 0.0)
	f.Add(-10.0, 0.5)

	f.Fuzz(func(t *testing.T, attenuation, transitionBW float64) {
		result := EstimateFilterLength(attenuation, transitionBW)

		// Result must always be within sanity bounds
		if result < minFilterLength {
			t.Errorf("EstimateFilterLength(%v, %v) = %d, below minimum %d",
				attenuation, transitionBW, result, minFilterLength)
		}
		if result > maxFilterLength {
			t.Errorf("EstimateFilterLength(%v, %v) = %d, above maximum %d",
				attenuation, transitionBW, result, maxFilterLength)
		}

		// Result must be odd (symmetric FIR)
		if result%2 == 0 {
			t.Errorf("EstimateFilterLength(%v, %v) = %d, want odd number",
				attenuation, transitionBW, result)
		}
	})
}
