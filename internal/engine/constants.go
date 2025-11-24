package engine

// Cubic (Hermite) interpolation constants
const (
	// Cubic interpolation uses 4-point window
	cubicInterpolationPoints = 4

	// Cubic interpolation latency (centered around middle points)
	cubicLatencySamples = 2

	// Hermite interpolation coefficients for smooth C1 continuity
	// Formula: y = ((a*x + b)*x + c)*x + d
	// These constants appear in the interpolation formula:
	// coefA := -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
	hermiteCoeff0_5 = 0.5
	hermiteCoeff1_5 = 1.5
	hermiteCoeff2_5 = 2.5

	// Memory usage estimate for cubic stage (bytes)
	cubicMemoryUsage = 64
)

// Linear interpolation constants
const (
	// Linear interpolation uses 2-point window
	linearInterpolationPoints = 2

	// Linear interpolation latency
	linearLatencySamples = 1

	// Memory usage estimate for linear stage (bytes)
	linearMemoryUsage = 32
)
