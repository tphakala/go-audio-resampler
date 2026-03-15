package filter

import (
	"math"
	"testing"
)

func FuzzDesignLowPassFilter(f *testing.F) {
	// Valid seeds
	f.Add(31, 0.25, 60.0, 1.0)
	f.Add(63, 0.1, 100.0, 1.0)
	f.Add(127, 0.4, 150.0, 2.0)
	f.Add(3, 0.01, 0.0, 1.0)     // minimum taps, zero attenuation
	f.Add(8191, 0.49, 200.0, 1.0) // maximum taps, high attenuation

	// Boundary/invalid seeds
	f.Add(0, 0.25, 60.0, 1.0)    // too few taps
	f.Add(31, 0.0, 60.0, 1.0)    // zero cutoff
	f.Add(31, 0.5, 60.0, 1.0)    // cutoff at Nyquist
	f.Add(31, 0.25, -1.0, 1.0)   // negative attenuation
	f.Add(31, 0.25, 60.0, 0.0)   // zero gain
	f.Add(31, 0.25, 60.0, -1.0)  // negative gain

	f.Fuzz(func(t *testing.T, numTaps int, cutoffFreq, attenuation, gain float64) {
		coeffs, err := DesignLowPassFilter(FilterParams{
			NumTaps:     numTaps,
			CutoffFreq:  cutoffFreq,
			Attenuation: attenuation,
			Gain:        gain,
		})

		if err != nil {
			// Validation rejected it — that's fine
			return
		}

		// If successful, verify output properties
		if len(coeffs) != numTaps {
			t.Errorf("got %d coefficients, want %d", len(coeffs), numTaps)
		}

		// No NaN or Inf in coefficients
		for i, c := range coeffs {
			if math.IsNaN(c) {
				t.Errorf("coefficient[%d] is NaN", i)
			}
			if math.IsInf(c, 0) {
				t.Errorf("coefficient[%d] is Inf", i)
			}
		}
	})
}

func FuzzKaiserWindow(f *testing.F) {
	f.Add(1, 0.0)
	f.Add(31, 5.0)
	f.Add(63, 10.0)
	f.Add(127, 15.0)
	f.Add(0, 5.0)  // empty
	f.Add(2, -1.0) // negative beta

	f.Fuzz(func(t *testing.T, length int, beta float64) {
		// Limit length to prevent OOM
		if length > 10000 || length < 0 {
			return
		}

		window := KaiserWindow(length, beta)

		if len(window) != max(length, 0) {
			if length < 1 && len(window) != 0 {
				t.Errorf("KaiserWindow(%d, %v) returned %d elements, want 0",
					length, beta, len(window))
			} else if length >= 1 && len(window) != length {
				t.Errorf("KaiserWindow(%d, %v) returned %d elements, want %d",
					length, beta, len(window), length)
			}
		}

		// No NaN or Inf in window
		for i, w := range window {
			if math.IsNaN(w) {
				t.Errorf("window[%d] is NaN for length=%d, beta=%v", i, length, beta)
			}
			if math.IsInf(w, 0) {
				t.Errorf("window[%d] is Inf for length=%d, beta=%v", i, length, beta)
			}
		}

		// Window should be symmetric: w[i] = w[length-1-i]
		if !math.IsNaN(beta) && !math.IsInf(beta, 0) {
			for i := range len(window) / 2 {
				j := len(window) - 1 - i
				if window[i] != window[j] {
					t.Errorf("window not symmetric: w[%d]=%v != w[%d]=%v",
						i, window[i], j, window[j])
					break
				}
			}
		}
	})
}
