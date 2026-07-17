// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

// Package engine implements various resampling algorithms.
package engine

import (
	"math"

	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

// CubicStage implements cubic (4-point, 3rd order) interpolation matching SOXR.
// This is the fastest resampling method, used for QualityQuick preset.
type CubicStage[F simdops.Float] struct {
	ratio     float64
	phase     float64
	history   [4]F // 4-point window for interpolation
	primed    int  // real samples pushed so far, capped at cubicLatencySamples
	latency   int
	outputBuf []F // reused across calls so the warm path allocates nothing
}

// NewCubicStage creates a new cubic interpolation stage.
func NewCubicStage[F simdops.Float](ratio float64) *CubicStage[F] {
	return &CubicStage[F]{
		ratio:   ratio,
		phase:   0,
		latency: cubicLatencySamples,
	}
}

// Process resamples input using cubic interpolation. The returned slice is
// owned by the caller and remains valid across subsequent calls.
func (c *CubicStage[F]) Process(input []F) ([]F, error) {
	out, err := c.processZeroCopy(input)
	if err != nil || len(out) == 0 {
		return out, err
	}
	// Return a copy so the caller's slice is not corrupted when the next call
	// reuses the internal output buffer.
	result := make([]F, len(out))
	copy(result, out)
	return result, nil
}

// processZeroCopy is the allocation-free internal path. The returned slice
// aliases c.outputBuf and is only valid until the next Process,
// processZeroCopy, Flush, or Reset call.
func (c *CubicStage[F]) processZeroCopy(input []F) ([]F, error) { //nolint:unparam // error kept for symmetry with the FIR stages' Process signature
	if len(input) == 0 {
		return []F{}, nil
	}

	// Upper bound on the outputs this call can emit: the phase accumulator
	// advances one input unit per sample and emits at most one output per
	// 1/ratio of that advance, so the count never exceeds ceil(len*ratio)
	// plus one boundary-carry sample. Pre-size the reused buffer to that
	// bound via growStableLen (which adds headroom when it must grow) so
	// steady-state constant-chunk streaming reuses it without allocating.
	// append then fills it, so an off-by-one in the bound can never write out
	// of range; on the warm path the capacity already covers the count and
	// append allocates nothing.
	maxOut := int(math.Ceil(float64(len(input))*c.ratio)) + 1
	out := growStableLen(c.outputBuf, maxOut)[:0]

	for _, sample := range input {
		// Shift history window
		c.history[3] = c.history[2]
		c.history[2] = c.history[1]
		c.history[1] = c.history[0]
		c.history[0] = sample

		// The center point used below (history[2]) needs cubicLatencySamples
		// real pushes past it before it holds real signal instead of the
		// zero value left by construction or Reset. Withhold emission during
		// that priming window so Process never fabricates output from
		// implicit pre-silence; Flush drains the true tail this reserves.
		if c.primed < cubicLatencySamples {
			c.primed++
			continue
		}

		// Generate output samples
		for c.phase < 1.0 {
			// Cubic interpolation matching SOXR
			out = append(out, c.interpolate(c.phase))

			// Advance phase
			c.phase += 1.0 / c.ratio
		}

		// Wrap phase
		c.phase -= 1.0
	}

	c.outputBuf = out
	return out, nil
}

// interpolate performs cubic interpolation matching SOXR's implementation.
// Uses the formula: y = ((a*x + b)*x + coefC)*x + s[0]
// where x is the fractional position between samples.
//
// SOXR formula from cr-core.c:59-61:
//
//	b = 0.5*(s[1]+s[-1]) - s[0]
//	a = (1/6)*(s[2]-s[1]+s[-1]-s[0] - 4*b)
//	coefC = s[1] - s[0] - a - b
func (c *CubicStage[F]) interpolate(x float64) F {
	// Get the 4 points for interpolation
	// Map to SOXR's convention: s[-1], s[0], s[1], s[2]
	sMinus1 := float64(c.history[3]) // oldest  = s[-1]
	s0 := float64(c.history[2])      // center  = s[0]
	s1 := float64(c.history[1])      // next    = s[1]
	s2 := float64(c.history[0])      // newest  = s[2]

	// SOXR's cubic formula (from cr-core.c:59-61)
	b := 0.5*(s1+sMinus1) - s0                        //nolint:mnd // Mathematical constant from SOXR formula
	a := (1.0 / 6.0) * (s2 - s1 + sMinus1 - s0 - 4*b) //nolint:mnd // Mathematical constant from SOXR formula
	coefC := s1 - s0 - a - b

	// Evaluate polynomial: (((a*x + b)*x + coefC)*x + s[0])
	return F(((a*x+b)*x+coefC)*x + s0)
}

// Flush drains the interpolator's held tail.
//
// The 4-point history window holds cubicLatencySamples of latency: the
// final real samples pushed into Process never reach the withheld-emission
// center position (see Process) and stay trapped internally when the caller
// stops feeding input. Flush pads that many zeros through the same Process
// path to release them, then resets to fresh state so a second Flush
// returns nothing and a subsequent Process behaves like a new instance.
//
// The zero padding is a hard silence assumption at the true end of signal.
// For a signal with a large sample-to-sample swing right at the boundary,
// cubic's polynomial fit can overshoot past the padding transition before
// settling; unlike an FIR filter's convolution, which decays smoothly by
// construction, point interpolation has no equivalent damping. This is a
// known, minor characteristic, not something Flush can avoid while still
// delivering the real tail.
func (c *CubicStage[F]) Flush() ([]F, error) {
	if c.primed == 0 {
		// Never fed: no held tail to drain.
		return []F{}, nil
	}
	zeros := make([]F, cubicLatencySamples)
	out, err := c.Process(zeros)
	c.Reset()
	return out, err
}

// Reset clears internal state.
func (c *CubicStage[F]) Reset() {
	c.phase = 0
	c.history = [4]F{}
	c.primed = 0
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
