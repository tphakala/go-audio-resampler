package engine

// Cubic (Hermite) interpolation constants
const (
	// Cubic interpolation uses 4-point window
	cubicInterpolationPoints = 4

	// Cubic interpolation latency (centered around middle points)
	cubicLatencySamples = 2

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
