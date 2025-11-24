// Package mathutil provides mathematical functions for audio resampling.
package mathutil

import (
	"math"
)

// BesselI0 computes the modified Bessel function of the first kind, order zero: I₀(x).
// This function is used in Kaiser window calculation for filter design.
//
// The implementation uses Chebyshev polynomial approximations for numerical stability:
//   - For |x| ≤ 3.75: Direct polynomial series expansion
//   - For |x| > 3.75: Asymptotic expansion with exponential scaling
//
// Accuracy: ~15 digits of precision (sufficient for audio DSP)
//
// Reference: Abramowitz & Stegun, "Handbook of Mathematical Functions"
// Also based on soxr's dbesi0.c implementation.
func BesselI0(x float64) float64 {
	// Use absolute value since I₀(x) = I₀(-x)
	ax := math.Abs(x)

	// For small arguments, use polynomial approximation
	if ax < besselSmallArgThreshold {
		// I₀(x) ≈ 1 + (x/2)² * P(t) where t = (x/3.75)²
		t := x / besselSmallArgThreshold
		t *= t

		// Polynomial coefficients (Chebyshev approximation)
		return 1.0 + t*(besselI0Coeff1+t*(besselI0Coeff2+t*(besselI0Coeff3+
			t*(besselI0Coeff4+t*(besselI0Coeff5+t*besselI0Coeff6)))))
	}

	// For larger arguments, use asymptotic expansion
	// I₀(x) ≈ (eˣ / √(2πx)) * P(t) where t = 3.75/x
	t := besselSmallArgThreshold / ax

	// Polynomial approximation for the scaled function
	// Result = exp(x) * P(t) / sqrt(x)
	result := besselI0AsympCoeff0 + t*(besselI0AsympCoeff1+t*(besselI0AsympCoeff2+
		t*(besselI0AsympCoeff3+t*(besselI0AsympCoeff4+t*(besselI0AsympCoeff5+
			t*(besselI0AsympCoeff6+t*(besselI0AsympCoeff7+t*besselI0AsympCoeff8)))))))

	// Scale by exp(x) / sqrt(x)
	return math.Exp(ax) * result / math.Sqrt(ax)
}

// BesselI0Ratio computes I₁(x) / I₀(x), which is useful for Kaiser window calculations.
// This ratio is more numerically stable than computing I₁ and I₀ separately for large x.
func BesselI0Ratio(x float64) float64 {
	// For small x, use series expansion: I₁(x)/I₀(x) ≈ x/2 for x near 0
	if math.Abs(x) < besselTinyArgThreshold {
		return x / halfDivisor
	}

	// For larger x, compute directly
	// I₁(x)/I₀(x) = (I₀'(x)) / I₀(x)
	// We can use the fact that for large x: I₁(x)/I₀(x) → 1 - 1/(2x)

	ax := math.Abs(x)
	if ax > besselLargeArgThreshold {
		// Asymptotic approximation for large arguments
		return 1.0 - 1.0/(halfDivisor*ax)
	}

	// For moderate values, compute I₁ using series and divide by I₀
	return besselI1(x) / BesselI0(x)
}

// besselI1 computes the modified Bessel function of the first kind, order one: I₁(x).
// This is used internally for I₁/I₀ ratio calculations.
func besselI1(x float64) float64 {
	ax := math.Abs(x)

	if ax < besselSmallArgThreshold {
		// For small arguments, use polynomial approximation
		t := x / besselSmallArgThreshold
		t *= t

		result := ax * (besselI1Coeff0 + t*(besselI1Coeff1+t*(besselI1Coeff2+
			t*(besselI1Coeff3+t*(besselI1Coeff4+t*(besselI1Coeff5+
				t*besselI1Coeff6))))))

		if x < 0 {
			return -result
		}
		return result
	}

	// For larger arguments, use asymptotic expansion
	t := besselSmallArgThreshold / ax

	result := besselI1AsympCoeff0 + t*(besselI1AsympCoeff1+t*(besselI1AsympCoeff2+
		t*(besselI1AsympCoeff3+t*(besselI1AsympCoeff4+t*(besselI1AsympCoeff5+
			t*(besselI1AsympCoeff6+t*(besselI1AsympCoeff7+t*besselI1AsympCoeff8)))))))

	result = math.Exp(ax) * result / math.Sqrt(ax)

	if x < 0 {
		return -result
	}
	return result
}

// KaiserBeta computes the Kaiser window β parameter from the desired
// stopband attenuation in decibels.
//
// The β parameter controls the trade-off between main lobe width and
// sidelobe level in the Kaiser window.
//
// Formula from Kaiser & Schafer:
//   - For att > 50 dB: β = 0.1102 * (att - 8.7)
//   - For 21 dB < att ≤ 50 dB: β = 0.5842 * (att - 21)^0.4 + 0.07886 * (att - 21)
//   - For att ≤ 21 dB: β = 0
//
// Parameters:
//
//	attenuation: Desired stopband attenuation in dB (typically 50-150 dB)
//
// Returns:
//
//	β parameter for Kaiser window (typically 0-15)
func KaiserBeta(attenuation float64) float64 {
	if attenuation > kaiserAttHigh {
		return kaiserBetaHighCoeff1 * (attenuation - kaiserBetaHighOffset)
	} else if attenuation >= kaiserAttMedium {
		delta := attenuation - kaiserAttMedium
		return kaiserBetaMediumCoeff1*math.Pow(delta, kaiserBetaMediumPower) + kaiserBetaMediumCoeff2*delta
	}
	return 0.0
}

// KaiserBetaWithTrBw computes the Kaiser window β parameter using soxr's
// polynomial approximation which takes transition bandwidth into account.
//
// This is more accurate than KaiserBeta for high attenuations (>= 60 dB)
// because it uses polynomial coefficients that are interpolated based on
// the transition bandwidth, matching soxr's lsx_kaiser_beta function.
//
// Parameters:
//
//	attenuation: Desired stopband attenuation in dB
//	trBw: Transition bandwidth (normalized)
//
// Returns:
//
//	β parameter for Kaiser window
func KaiserBetaWithTrBw(attenuation, trBw float64) float64 {
	if attenuation >= kaiserAttPolynomial {
		// soxr's polynomial coefficients for different transition bandwidths
		// Each row: {a3, a2, a1, a0} for polynomial ((a3*att + a2)*att + a1)*att + a0
		coefs := [][4]float64{
			{-6.784957e-10, 1.02856e-05, 0.1087556, -0.8988365 + .001},
			{-6.897885e-10, 1.027433e-05, 0.10876, -0.8994658 + .002},
			{-1.000683e-09, 1.030092e-05, 0.1087677, -0.9007898 + .003},
			{-3.654474e-10, 1.040631e-05, 0.1087085, -0.8977766 + .006},
			{8.106988e-09, 6.983091e-06, 0.1091387, -0.9172048 + .015},
			{9.519571e-09, 7.272678e-06, 0.1090068, -0.9140768 + .025},
			{-5.626821e-09, 1.342186e-05, 0.1083999, -0.9065452 + .05},
			{-9.965946e-08, 5.073548e-05, 0.1040967, -0.7672778 + .085},
			{1.604808e-07, -5.856462e-05, 0.1185998, -1.34824 + .1},
			{-1.511964e-07, 6.363034e-05, 0.1064627, -0.9876665 + .18},
		}

		// Select polynomial coefficients based on transition bandwidth
		// realm = log2(trBw / 0.0005)
		if trBw < kaiserMinTrBw {
			trBw = kaiserMinTrBw // Prevent log(0)
		}
		realm := math.Log(trBw/kaiserTrBwRealmBase) / math.Ln2

		// Clamp indices to valid range
		idx0 := max(int(realm), 0)
		if idx0 >= len(coefs) {
			idx0 = len(coefs) - 1
		}
		idx1 := idx0 + 1
		if idx1 >= len(coefs) {
			idx1 = len(coefs) - 1
		}

		c0 := coefs[idx0]
		c1 := coefs[idx1]

		// Evaluate polynomials
		b0 := ((c0[0]*attenuation+c0[1])*attenuation+c0[2])*attenuation + c0[3]
		b1 := ((c1[0]*attenuation+c1[1])*attenuation+c1[2])*attenuation + c1[3]

		// Interpolate between b0 and b1
		frac := realm - float64(int(realm))
		if frac < 0 {
			frac = 0
		}
		return b0 + (b1-b0)*frac
	}
	if attenuation > kaiserAttHigh {
		return kaiserBetaHighCoeff1 * (attenuation - kaiserBetaHighOffset)
	} else if attenuation >= kaiserAttMedium {
		delta := attenuation - kaiserAttMedium
		return kaiserBetaMediumCoeff1*math.Pow(delta, kaiserBetaMediumPower) + kaiserBetaMediumCoeff2*delta
	}
	return 0.0
}

// KaiserAttenuation estimates the stopband attenuation achieved by a
// Kaiser window with the given β parameter.
//
// This is the inverse of KaiserBeta, useful for verifying filter design.
//
// Approximate formula:
//
//	att ≈ 8.7 + β / 0.1102
func KaiserAttenuation(beta float64) float64 {
	if beta < kaiserBetaMinThreshold {
		return 0.0
	}
	// Approximate inverse
	return kaiserBetaHighOffset + beta/kaiserBetaHighCoeff1
}

// EstimateFilterLength estimates the required FIR filter length to achieve
// the specified attenuation with the given transition bandwidth.
//
// Based on the Kaiser formula:
//
//	N ≈ (att - 8) / (2.285 * Δω * π)
//
// where:
//
//	N = filter length (number of taps)
//	att = stopband attenuation in dB
//	Δω = transition bandwidth in normalized frequency (0-1)
//
// Parameters:
//
//	attenuation: Desired stopband attenuation in dB
//	transitionBW: Transition bandwidth as fraction of sample rate (0-1)
//
// Returns:
//
//	Estimated number of filter taps (odd number)
func EstimateFilterLength(attenuation, transitionBW float64) int {
	if transitionBW <= 0 {
		transitionBW = defaultTransitionBW
	}

	// Kaiser's formula
	numTaps := (attenuation - kaiserFilterLengthOffset) / (kaiserFilterLengthMultiplier * kaiserFilterLengthPiFactor * math.Pi * transitionBW)

	// Round up to nearest odd integer (symmetric FIR filters are odd-length)
	taps := int(math.Ceil(numTaps))
	if taps%2 == 0 {
		taps++
	}

	// Sanity bounds
	if taps < minFilterLength {
		taps = minFilterLength
	}
	if taps > maxFilterLength {
		taps = maxFilterLength
	}

	return taps
}
