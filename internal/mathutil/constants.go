package mathutil

// Bessel function approximation constants
// These constants are Chebyshev polynomial coefficients from
// Abramowitz & Stegun, "Handbook of Mathematical Functions"

const (
	// Threshold for switching between polynomial and asymptotic approximations
	besselSmallArgThreshold = 3.75 // |x| threshold for I₀ and I₁
	besselLargeArgThreshold = 50.0 // Threshold for asymptotic approximation in ratio

	// Numerical stability thresholds
	besselTinyArgThreshold = 1e-10 // Threshold for series expansion in ratio
	kaiserBetaMinThreshold = 0.1   // Minimum β for attenuation calculation
)

// Chebyshev coefficients for I₀(x) small argument approximation
const (
	besselI0Coeff1 = 3.5156229
	besselI0Coeff2 = 3.0899424
	besselI0Coeff3 = 1.2067492
	besselI0Coeff4 = 0.2659732
	besselI0Coeff5 = 0.360768e-1
	besselI0Coeff6 = 0.45813e-2
)

// Chebyshev coefficients for I₀(x) large argument approximation
const (
	besselI0AsympCoeff0 = 0.39894228
	besselI0AsympCoeff1 = 0.1328592e-1
	besselI0AsympCoeff2 = 0.225319e-2
	besselI0AsympCoeff3 = -0.157565e-2
	besselI0AsympCoeff4 = 0.916281e-2
	besselI0AsympCoeff5 = -0.2057706e-1
	besselI0AsympCoeff6 = 0.2635537e-1
	besselI0AsympCoeff7 = -0.1647633e-1
	besselI0AsympCoeff8 = 0.392377e-2
)

// Chebyshev coefficients for I₁(x) small argument approximation
const (
	besselI1Coeff0 = 0.5
	besselI1Coeff1 = 0.87890594
	besselI1Coeff2 = 0.51498869
	besselI1Coeff3 = 0.15084934
	besselI1Coeff4 = 0.2658733e-1
	besselI1Coeff5 = 0.301532e-2
	besselI1Coeff6 = 0.32411e-3
)

// Chebyshev coefficients for I₁(x) large argument approximation
const (
	besselI1AsympCoeff0 = 0.39894228
	besselI1AsympCoeff1 = -0.3988024e-1
	besselI1AsympCoeff2 = -0.362018e-2
	besselI1AsympCoeff3 = 0.163801e-2
	besselI1AsympCoeff4 = -0.1031555e-1
	besselI1AsympCoeff5 = 0.2282967e-1
	besselI1AsympCoeff6 = -0.2895312e-1
	besselI1AsympCoeff7 = 0.1787654e-1
	besselI1AsympCoeff8 = -0.420059e-2
)

// Kaiser window formula constants
// From Kaiser & Schafer's empirical formulas
const (
	// Attenuation thresholds for β calculation
	kaiserAttHigh       = 50.0   // High attenuation threshold (dB)
	kaiserAttMedium     = 21.0   // Medium attenuation threshold (dB)
	kaiserAttPolynomial = 60.0   // Threshold for polynomial approximation in KaiserBetaWithTrBw
	kaiserMinTrBw       = 0.0001 // Minimum transition bandwidth to prevent log(0)
	kaiserTrBwRealmBase = 0.0005 // Base value for realm calculation in polynomial selection

	// Kaiser β formula coefficients
	kaiserBetaHighCoeff1 = 0.1102 // Coefficient for high attenuation
	kaiserBetaHighOffset = 8.7    // Offset for high attenuation

	kaiserBetaMediumCoeff1 = 0.5842  // Primary coefficient for medium attenuation
	kaiserBetaMediumPower  = 0.4     // Power for medium attenuation formula
	kaiserBetaMediumCoeff2 = 0.07886 // Secondary coefficient for medium attenuation
)

// Filter length estimation constants
const (
	// Kaiser's filter length formula: N ≈ (att - 8) / (2.285 * Δω * π)
	kaiserFilterLengthOffset     = 8.0   // Attenuation offset in Kaiser formula
	kaiserFilterLengthMultiplier = 2.285 // Multiplier in Kaiser formula
	kaiserFilterLengthPiFactor   = 2.0   // Factor for 2π in formula

	// Filter length bounds
	minFilterLength = 3    // Minimum filter length (taps)
	maxFilterLength = 8191 // Maximum filter length (taps)

	// Default transition bandwidth for safety
	defaultTransitionBW = 0.01 // Prevent division by zero
)

// Common division constants
const (
	halfDivisor = 2.0 // Division by 2
)
