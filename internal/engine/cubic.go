// Package engine implements various resampling algorithms.
package engine

import (
	"math"
)

// CubicStage implements cubic (4-point, 3rd order) Hermite interpolation.
// This is the fastest but lowest quality resampling method.
type CubicStage struct {
	ratio   float64
	phase   float64
	history [4]float64 // 4-point window for interpolation
	histPos int
	latency int
}

// NewCubicStage creates a new cubic interpolation stage.
func NewCubicStage(ratio float64) *CubicStage {
	return &CubicStage{
		ratio:   ratio,
		phase:   0,
		latency: cubicLatencySamples,
	}
}

// Process resamples input using cubic interpolation.
func (c *CubicStage) Process(input []float64) ([]float64, error) {
	if len(input) == 0 {
		return []float64{}, nil
	}

	// Estimate output size
	outputSize := int(math.Ceil(float64(len(input)) * c.ratio))
	output := make([]float64, 0, outputSize)

	for _, sample := range input {
		// Shift history window
		c.history[3] = c.history[2]
		c.history[2] = c.history[1]
		c.history[1] = c.history[0]
		c.history[0] = sample

		// Generate output samples
		for c.phase < 1.0 {
			// Cubic Hermite interpolation
			y := c.interpolate(c.phase)
			output = append(output, y)

			// Advance phase
			c.phase += 1.0 / c.ratio
		}

		// Wrap phase
		c.phase -= 1.0
	}

	return output, nil
}

// interpolate performs cubic Hermite interpolation.
// Uses the formula: y = ((a*x + b)*x + c)*x + d
// where x is the fractional position between samples.
func (c *CubicStage) interpolate(x float64) float64 {
	// Get the 4 points for interpolation
	y0 := c.history[3] // oldest
	y1 := c.history[2]
	y2 := c.history[1]
	y3 := c.history[0] // newest

	// Hermite basis functions
	// These coefficients provide smooth interpolation with continuous first derivative
	coefA := -hermiteCoeff0_5*y0 + hermiteCoeff1_5*y1 - hermiteCoeff1_5*y2 + hermiteCoeff0_5*y3
	coefB := y0 - hermiteCoeff2_5*y1 + 2*y2 - hermiteCoeff0_5*y3
	coefC := -hermiteCoeff0_5*y0 + hermiteCoeff0_5*y2
	coefD := y1

	// Evaluate polynomial
	return ((coefA*x+coefB)*x+coefC)*x + coefD
}

// Flush returns any remaining samples.
func (c *CubicStage) Flush() ([]float64, error) {
	// Cubic interpolation doesn't buffer samples
	return []float64{}, nil
}

// Reset clears internal state.
func (c *CubicStage) Reset() {
	c.phase = 0
	c.history = [4]float64{}
}

// GetRatio returns the stage's resampling ratio.
func (c *CubicStage) GetRatio() float64 {
	return c.ratio
}

// GetLatency returns the stage latency in samples.
func (c *CubicStage) GetLatency() int {
	return c.latency
}

// GetMinInput returns the minimum input size for processing.
func (c *CubicStage) GetMinInput() int {
	return 1 // Can process sample by sample
}

// GetMemoryUsage returns approximate memory usage in bytes.
func (c *CubicStage) GetMemoryUsage() int64 {
	return cubicMemoryUsage
}

// GetFilterLength returns 0 as cubic doesn't use a filter.
func (c *CubicStage) GetFilterLength() int {
	return cubicInterpolationPoints
}

// GetPhases returns 0 as cubic doesn't use phases.
func (c *CubicStage) GetPhases() int {
	return 0
}

// GetSIMDInfo returns empty as cubic doesn't use SIMD.
func (c *CubicStage) GetSIMDInfo() string {
	return ""
}

// LinearStage implements linear (2-point, 1st order) interpolation.
// Even faster than cubic but lower quality.
type LinearStage struct {
	ratio   float64
	phase   float64
	prev    float64
	latency int
}

// NewLinearStage creates a new linear interpolation stage.
func NewLinearStage(ratio float64) *LinearStage {
	return &LinearStage{
		ratio:   ratio,
		phase:   0,
		latency: linearLatencySamples,
	}
}

// Process resamples input using linear interpolation.
func (l *LinearStage) Process(input []float64) ([]float64, error) {
	if len(input) == 0 {
		return []float64{}, nil
	}

	outputSize := int(math.Ceil(float64(len(input)) * l.ratio))
	output := make([]float64, 0, outputSize)

	for _, sample := range input {
		// Generate output samples between previous and current
		for l.phase < 1.0 {
			// Linear interpolation: y = (1-x)*prev + x*current
			y := (1-l.phase)*l.prev + l.phase*sample
			output = append(output, y)

			// Advance phase
			l.phase += 1.0 / l.ratio
		}

		// Update state
		l.prev = sample
		l.phase -= 1.0
	}

	return output, nil
}

// Flush returns any remaining samples.
func (l *LinearStage) Flush() ([]float64, error) {
	return []float64{}, nil
}

// Reset clears internal state.
func (l *LinearStage) Reset() {
	l.phase = 0
	l.prev = 0
}

// GetRatio returns the stage's resampling ratio.
func (l *LinearStage) GetRatio() float64 {
	return l.ratio
}

// GetLatency returns the stage latency in samples.
func (l *LinearStage) GetLatency() int {
	return l.latency
}

// GetMinInput returns the minimum input size for processing.
func (l *LinearStage) GetMinInput() int {
	return 1
}

// GetMemoryUsage returns approximate memory usage in bytes.
func (l *LinearStage) GetMemoryUsage() int64 {
	return linearMemoryUsage
}

// GetFilterLength returns 0 as linear doesn't use a filter.
func (l *LinearStage) GetFilterLength() int {
	return linearInterpolationPoints
}

// GetPhases returns 0 as linear doesn't use phases.
func (l *LinearStage) GetPhases() int {
	return 0
}

// GetSIMDInfo returns empty as linear doesn't use SIMD.
func (l *LinearStage) GetSIMDInfo() string {
	return ""
}
