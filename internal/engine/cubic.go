// Package engine implements various resampling algorithms.
package engine

import (
	"math"

	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

// CubicStage implements cubic (4-point, 3rd order) interpolation matching SOXR.
// This is the fastest resampling method, used for QualityQuick preset.
type CubicStage[F simdops.Float] struct {
	ratio   float64
	phase   float64
	history [4]F // 4-point window for interpolation
	histPos int
	latency int
}

// NewCubicStage creates a new cubic interpolation stage.
func NewCubicStage[F simdops.Float](ratio float64) *CubicStage[F] {
	return &CubicStage[F]{
		ratio:   ratio,
		phase:   0,
		latency: cubicLatencySamples,
	}
}

// Process resamples input using cubic interpolation.
func (c *CubicStage[F]) Process(input []F) ([]F, error) {
	if len(input) == 0 {
		return []F{}, nil
	}

	// Estimate output size
	outputSize := int(math.Ceil(float64(len(input)) * c.ratio))
	output := make([]F, 0, outputSize)

	for _, sample := range input {
		// Shift history window
		c.history[3] = c.history[2]
		c.history[2] = c.history[1]
		c.history[1] = c.history[0]
		c.history[0] = sample

		// Generate output samples
		for c.phase < 1.0 {
			// Cubic interpolation matching SOXR
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

// interpolate performs cubic interpolation matching SOXR's implementation.
// Uses the formula: y = ((a*x + b)*x + coefC)*x + s[0]
// where x is the fractional position between samples.
//
// SOXR formula from cr-core.c:59-61:
//   b = 0.5*(s[1]+s[-1]) - s[0]
//   a = (1/6)*(s[2]-s[1]+s[-1]-s[0] - 4*b)
//   coefC = s[1] - s[0] - a - b
func (c *CubicStage[F]) interpolate(x float64) F {
	// Get the 4 points for interpolation
	// Map to SOXR's convention: s[-1], s[0], s[1], s[2]
	sMinus1 := float64(c.history[3]) // oldest  = s[-1]
	s0 := float64(c.history[2])      // center  = s[0]
	s1 := float64(c.history[1])      // next    = s[1]
	s2 := float64(c.history[0])      // newest  = s[2]

	// SOXR's cubic formula (from cr-core.c:59-61)
	b := 0.5*(s1+sMinus1) - s0                                 //nolint:mnd // Mathematical constant from SOXR formula
	a := (1.0 / 6.0) * (s2 - s1 + sMinus1 - s0 - 4*b) //nolint:mnd // Mathematical constant from SOXR formula
	coefC := s1 - s0 - a - b

	// Evaluate polynomial: (((a*x + b)*x + coefC)*x + s[0])
	return F(((a*x+b)*x+coefC)*x + s0)
}

// Flush returns any remaining samples.
func (c *CubicStage[F]) Flush() ([]F, error) {
	// Cubic interpolation doesn't buffer samples
	return []F{}, nil
}

// Reset clears internal state.
func (c *CubicStage[F]) Reset() {
	c.phase = 0
	c.history = [4]F{}
}

// GetRatio returns the stage's resampling ratio.
func (c *CubicStage[F]) GetRatio() float64 {
	return c.ratio
}

// GetLatency returns the stage latency in samples.
func (c *CubicStage[F]) GetLatency() int {
	return c.latency
}

// GetMinInput returns the minimum input size for processing.
func (c *CubicStage[F]) GetMinInput() int {
	return 1 // Can process sample by sample
}

// GetMemoryUsage returns approximate memory usage in bytes.
func (c *CubicStage[F]) GetMemoryUsage() int64 {
	return cubicMemoryUsage
}

// GetFilterLength returns 0 as cubic doesn't use a filter.
func (c *CubicStage[F]) GetFilterLength() int {
	return cubicInterpolationPoints
}

// GetPhases returns 0 as cubic doesn't use phases.
func (c *CubicStage[F]) GetPhases() int {
	return 0
}

// GetSIMDInfo returns empty as cubic doesn't use SIMD.
func (c *CubicStage[F]) GetSIMDInfo() string {
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
