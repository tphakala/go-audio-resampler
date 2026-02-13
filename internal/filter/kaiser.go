// Package filter provides filter design functions for audio resampling.
package filter

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/mathutil"
	"github.com/tphakala/simd/f64"
)

const (
	// Filter design constants
	minFilterTaps = 3
	maxFilterTaps = 8191

	// Window normalization
	windowNormalizationFactor = 2.0

	// Sinc function constants
	sincCenterTap     = 1.0
	sincPiMultiplier  = math.Pi
	sincZeroThreshold = 1e-10

	// Filter normalization
	filterGainTarget = 1.0
)

// KaiserWindow generates a Kaiser window of the specified length and β parameter.
//
// The Kaiser window provides excellent control over the trade-off between
// main lobe width and sidelobe level in frequency domain.
//
// Parameters:
//
//	length: Number of samples in the window (should be odd for symmetric FIR)
//	beta: Kaiser β parameter (controls sidelobe attenuation)
//	      Typically 0-15, where higher values = more attenuation but wider main lobe
//
// Returns:
//
//	Window coefficients normalized so that sum = 1.0
//
// The window is symmetric: w[i] = w[length-1-i]
func KaiserWindow(length int, beta float64) []float64 {
	if length < 1 {
		return []float64{}
	}

	window := make([]float64, length)

	// Special case for length 1
	if length == 1 {
		window[0] = sincCenterTap
		return window
	}

	// Calculate window using Kaiser formula:
	// w[n] = I₀(β * sqrt(1 - ((n - α)/α)²)) / I₀(β)
	// where α = (N-1)/2 and N is the window length

	alpha := float64(length-1) / windowNormalizationFactor
	i0Beta := mathutil.BesselI0(beta)

	for n := range length {
		// Calculate position relative to center: [-1, 1]
		x := (float64(n) - alpha) / alpha

		// Kaiser window formula
		arg := beta * math.Sqrt(1.0-x*x)
		window[n] = mathutil.BesselI0(arg) / i0Beta
	}

	return window
}

// FilterParams holds parameters for filter design.
type FilterParams struct {
	// NumTaps is the filter length (number of coefficients)
	// Should be odd for symmetric linear-phase FIR
	NumTaps int

	// CutoffFreq is the normalized cutoff frequency (0 to 0.5)
	// 0.5 represents Nyquist frequency (half the sample rate)
	CutoffFreq float64

	// Attenuation is the desired stopband attenuation in dB
	// Typical values: 60-150 dB
	Attenuation float64

	// Gain is the passband gain (typically 1.0)
	Gain float64
}

// Validate checks if filter parameters are valid.
func (fp *FilterParams) Validate() error {
	if fp.NumTaps < minFilterTaps {
		return fmt.Errorf("filter too short: %d taps (minimum %d)", fp.NumTaps, minFilterTaps)
	}

	if fp.NumTaps > maxFilterTaps {
		return fmt.Errorf("filter too long: %d taps (maximum %d)", fp.NumTaps, maxFilterTaps)
	}

	if fp.CutoffFreq <= 0 || fp.CutoffFreq >= 0.5 {
		return fmt.Errorf("invalid cutoff frequency: %f (must be in (0, 0.5))", fp.CutoffFreq)
	}

	if fp.Attenuation < 0 {
		return fmt.Errorf("invalid attenuation: %f dB (must be positive)", fp.Attenuation)
	}

	if fp.Gain <= 0 {
		return fmt.Errorf("invalid gain: %f (must be positive)", fp.Gain)
	}

	return nil
}

// DesignLowPassFilter designs a windowed-sinc lowpass FIR filter.
//
// This uses the Kaiser window method:
// 1. Generate ideal sinc function (infinite impulse response)
// 2. Truncate to finite length
// 3. Apply Kaiser window to reduce Gibbs phenomenon
// 4. Normalize for desired gain
//
// The resulting filter has linear phase (symmetric impulse response)
// and excellent stopband attenuation.
//
// Parameters:
//
//	params: Filter design parameters
//
// Returns:
//
//	Filter coefficients (length = params.NumTaps)
//	Error if parameters are invalid
func DesignLowPassFilter(params FilterParams) ([]float64, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}

	// Calculate Kaiser β from desired attenuation
	beta := mathutil.KaiserBeta(params.Attenuation)

	// Generate Kaiser window
	window := KaiserWindow(params.NumTaps, beta)

	// Generate windowed sinc function
	filter := make([]float64, params.NumTaps)
	center := float64(params.NumTaps-1) / windowNormalizationFactor

	for n := range params.NumTaps {
		// Calculate position relative to center
		x := float64(n) - center

		// Generate sinc function: sin(2πfc·x) / (πx)
		// At x=0: limit is 2*fc (by L'Hôpital's rule)
		var sincValue float64
		if math.Abs(x) < sincZeroThreshold {
			// Center tap value: 2 * cutoff frequency
			sincValue = windowNormalizationFactor * params.CutoffFreq
		} else {
			arg := windowNormalizationFactor * sincPiMultiplier * params.CutoffFreq * x
			sincValue = math.Sin(arg) / (sincPiMultiplier * x)
		}

		// Apply Kaiser window
		filter[n] = sincValue * window[n]
	}

	// Normalize filter for desired gain at DC
	// Uses SIMD-accelerated sum and scale operations
	sum := f64.Sum(filter)

	if math.Abs(sum) > sincZeroThreshold {
		scale := params.Gain / sum
		f64.Scale(filter, filter, scale)
	}

	return filter, nil
}

// DesignLowPassFilterAuto designs a lowpass filter with automatic length calculation.
//
// This is a convenience function that automatically calculates the required
// filter length based on the attenuation and transition bandwidth requirements.
//
// Parameters:
//
//	cutoffFreq: Normalized cutoff frequency (0 to 0.5)
//	transitionBW: Normalized transition bandwidth (0 to 0.5)
//	attenuation: Desired stopband attenuation in dB
//	gain: Passband gain (typically 1.0)
//
// Returns:
//
//	Filter coefficients
//	Error if parameters are invalid
func DesignLowPassFilterAuto(cutoffFreq, transitionBW, attenuation, gain float64) ([]float64, error) {
	// Calculate required filter length
	numTaps := mathutil.EstimateFilterLength(attenuation, transitionBW)

	params := FilterParams{
		NumTaps:     numTaps,
		CutoffFreq:  cutoffFreq,
		Attenuation: attenuation,
		Gain:        gain,
	}

	return DesignLowPassFilter(params)
}

// FilterResponse holds the frequency response of a filter.
type FilterResponse struct {
	// Frequencies at which response was calculated (normalized, 0 to 0.5)
	Frequencies []float64

	// Magnitude response at each frequency (linear scale)
	Magnitude []float64

	// Phase response at each frequency (radians)
	Phase []float64
}

// ComputeFrequencyResponse calculates the frequency response of a FIR filter.
//
// Uses the discrete-time Fourier transform (DTFT) to evaluate the filter's
// frequency response at the specified number of points.
//
// Parameters:
//
//	coeffs: Filter coefficients
//	numPoints: Number of frequency points to evaluate (default: 512)
//
// Returns:
//
//	Frequency response data
func ComputeFrequencyResponse(coeffs []float64, numPoints int) FilterResponse {
	if numPoints <= 0 {
		numPoints = 512
	}

	response := FilterResponse{
		Frequencies: make([]float64, numPoints),
		Magnitude:   make([]float64, numPoints),
		Phase:       make([]float64, numPoints),
	}

	// Evaluate frequency response at numPoints frequencies from 0 to Nyquist
	for k := range numPoints {
		// Normalized frequency (0 to 0.5)
		freq := float64(k) / float64(windowNormalizationFactor*numPoints)
		response.Frequencies[k] = freq

		// Compute H(e^jω) = Σ h[n]·e^(-jωn)
		// Split into real and imaginary parts
		var realPart, imagPart float64
		omega := windowNormalizationFactor * sincPiMultiplier * freq

		for n, h := range coeffs {
			angle := omega * float64(n)
			realPart += h * math.Cos(angle)
			imagPart -= h * math.Sin(angle)
		}

		// Calculate magnitude and phase
		response.Magnitude[k] = math.Sqrt(realPart*realPart + imagPart*imagPart)
		response.Phase[k] = math.Atan2(imagPart, realPart)
	}

	return response
}

// MagnitudeDB converts linear magnitude to decibels.
func MagnitudeDB(magnitude float64) float64 {
	const (
		minMagnitude = 1e-10 // Avoid log(0)
		dbMultiplier = 20.0  // 20*log10 for magnitude
	)

	if magnitude < minMagnitude {
		magnitude = minMagnitude
	}
	return dbMultiplier * math.Log10(magnitude)
}
