package filter

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/mathutil"
)

const (
	// Default number of phases for polyphase filter bank
	defaultNumPhases = 256

	// Interpolation orders
	interpOrderNone   = 0 // No coefficient interpolation
	interpOrderLinear = 1 // Linear interpolation between phases
	interpOrderCubic  = 3 // Cubic interpolation between phases

	// Coefficient storage multiplier
	minNumPhases = 2
	maxNumPhases = 8192

	// Minimum taps per phase for effective filtering
	// With too few taps, the filter cannot provide good stopband attenuation
	// and DC gain varies significantly between phases
	// Increased from 4 to 16 to match soxr's approach and ensure uniform DC gain
	minTapsPerPhase = 16

	// Phase precision (bits for fractional phase index)
	phasePrecisionBits = 20
	phasePrecisionMask = (1 << phasePrecisionBits) - 1

	// Polyphase decomposition constants
	nextPhaseOffset       = 1
	prevPhaseLookback     = 1
	secondNextPhaseOffset = 2

	// Interpolation polynomial coefficients
	cubicCenterCoeff = 0.5
	cubicDCoeff      = 1.0 / 6.0
	cubicCMultiplier = 4.0

	// Frequency response calculation
	frequencyNyquistDivisor = 2
)

// InterpOrder represents the coefficient interpolation order.
type InterpOrder int

const (
	// InterpNone means no interpolation (nearest phase)
	InterpNone InterpOrder = interpOrderNone
	// InterpLinear means linear interpolation between adjacent phases
	InterpLinear InterpOrder = interpOrderLinear
	// InterpCubic means cubic interpolation between phases
	InterpCubic InterpOrder = interpOrderCubic
)

// PolyphaseFilterBank represents a polyphase decomposition of an FIR filter.
//
// The filter is decomposed into multiple phases, where each phase contains
// a decimated version of the original filter. This allows efficient arbitrary
// ratio resampling using fixed-point phase accumulation.
//
// Coefficient storage format (per tap, per phase):
//   - InterpNone:   [coef]
//   - InterpLinear: [coef, delta]  where value = coef + delta*x
//   - InterpCubic:  [coef, b, c, d] where value = coef + (b + (c + d*x)*x)*x
type PolyphaseFilterBank struct {
	// Coeffs stores all filter coefficients in a flat array.
	// Layout: [phase0_tap0_coefs...][phase0_tap1_coefs...]...[phaseN_tapM_coefs...]
	// Each tap stores (InterpOrder+1) coefficients for polynomial evaluation.
	Coeffs []float64

	// NumPhases is the number of phases (polyphase branches)
	NumPhases int

	// TapsPerPhase is the number of taps in each phase
	TapsPerPhase int

	// TotalTaps is the original filter length before decomposition
	TotalTaps int

	// InterpOrder is the coefficient interpolation order (0, 1, or 3)
	InterpOrder InterpOrder

	// Cutoff is the normalized cutoff frequency used in design
	Cutoff float64

	// Attenuation is the stopband attenuation in dB
	Attenuation float64
}

// PolyphaseParams holds parameters for polyphase filter bank design.
type PolyphaseParams struct {
	// NumPhases is the number of polyphase branches (e.g., 64, 256, 1024)
	// Higher values allow finer phase resolution and better quality
	NumPhases int

	// Cutoff is the normalized cutoff frequency (0 to 0.5)
	// For upsampling: typically 0.5/upsampleRatio
	// For downsampling: typically 0.5
	Cutoff float64

	// TransitionBW is the transition bandwidth as fraction of sample rate
	TransitionBW float64

	// Attenuation is the desired stopband attenuation in dB
	Attenuation float64

	// InterpOrder specifies coefficient interpolation (0, 1, or 3)
	// 0: No interpolation (nearest phase)
	// 1: Linear interpolation (~6dB improvement)
	// 3: Cubic interpolation (~12dB improvement)
	InterpOrder InterpOrder

	// Gain is the passband gain (typically 1.0)
	Gain float64
}

// Validate checks if polyphase parameters are valid.
func (pp *PolyphaseParams) Validate() error {
	if pp.NumPhases < minNumPhases || pp.NumPhases > maxNumPhases {
		return fmt.Errorf("number of phases %d out of range [%d, %d]",
			pp.NumPhases, minNumPhases, maxNumPhases)
	}

	if pp.Cutoff <= 0 || pp.Cutoff >= 0.5 {
		return fmt.Errorf("cutoff frequency %f out of range (0, 0.5)", pp.Cutoff)
	}

	if pp.TransitionBW <= 0 || pp.TransitionBW >= 0.5 {
		return fmt.Errorf("transition bandwidth %f out of range (0, 0.5)", pp.TransitionBW)
	}

	if pp.Attenuation < 0 {
		return fmt.Errorf("attenuation %f dB must be positive", pp.Attenuation)
	}

	if pp.InterpOrder != InterpNone && pp.InterpOrder != InterpLinear && pp.InterpOrder != InterpCubic {
		return fmt.Errorf("invalid interpolation order %d (must be 0, 1, or 3)", pp.InterpOrder)
	}

	if pp.Gain <= 0 {
		return fmt.Errorf("gain %f must be positive", pp.Gain)
	}

	return nil
}

// DesignPolyphaseFilterBank creates a polyphase filter bank from the given parameters.
//
// The process:
// 1. Design a prototype lowpass filter using Kaiser window method
// 2. Decompose the filter into multiple phases
// 3. Compute interpolation coefficients for sub-phase precision
//
// Returns the polyphase filter bank ready for use in resampling.
func DesignPolyphaseFilterBank(params PolyphaseParams) (*PolyphaseFilterBank, error) {
	if err := params.Validate(); err != nil {
		return nil, fmt.Errorf("invalid polyphase parameters: %w", err)
	}

	// Calculate minimum filter length to ensure adequate taps per phase
	minTotalTaps := minTapsPerPhase * params.NumPhases

	// Design prototype lowpass filter with adequate length
	// Use the auto design first to get estimated length
	estimatedFilter, err := DesignLowPassFilterAuto(
		params.Cutoff,
		params.TransitionBW,
		params.Attenuation,
		params.Gain,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to design prototype filter: %w", err)
	}

	// If the estimated filter is too short, design a longer one by using the minimum length
	var prototypeFilter []float64
	if len(estimatedFilter) < minTotalTaps {
		// Design with explicit length
		beta := mathutil.KaiserBeta(params.Attenuation)
		window := KaiserWindow(minTotalTaps, beta)

		// Generate windowed sinc
		prototypeFilter = make([]float64, minTotalTaps)
		center := float64(minTotalTaps-1) / windowNormalizationFactor

		for n := range minTotalTaps {
			x := float64(n) - center
			var sincValue float64
			const sincZeroThreshold = 1e-10
			if math.Abs(x) < sincZeroThreshold {
				sincValue = windowNormalizationFactor * params.Cutoff
			} else {
				arg := windowNormalizationFactor * math.Pi * params.Cutoff * x
				sincValue = math.Sin(arg) / (math.Pi * x)
			}
			prototypeFilter[n] = sincValue * window[n]
		}

		// Normalize for desired gain
		// Scale by NumPhases so that the average DC gain per phase is 1.0
		// This matches soxr's approach: the prototype filter has total DC gain = NumPhases
		sum := 0.0
		for _, coeff := range prototypeFilter {
			sum += coeff
		}
		if math.Abs(sum) > sincZeroThreshold {
			scale := params.Gain * float64(params.NumPhases) / sum
			for i := range prototypeFilter {
				prototypeFilter[i] *= scale
			}
		}
	} else {
		prototypeFilter = estimatedFilter
	}

	// Create polyphase filter bank
	pfb := &PolyphaseFilterBank{
		NumPhases:   params.NumPhases,
		TotalTaps:   len(prototypeFilter),
		InterpOrder: params.InterpOrder,
		Cutoff:      params.Cutoff,
		Attenuation: params.Attenuation,
	}

	// Calculate taps per phase
	pfb.TapsPerPhase = (pfb.TotalTaps + pfb.NumPhases - 1) / pfb.NumPhases

	// Decompose prototype filter into phases with interpolation coefficients
	pfb.Coeffs = decomposePolyphase(prototypeFilter, pfb.NumPhases, pfb.TapsPerPhase, pfb.InterpOrder)

	return pfb, nil
}

// decomposePolyphase decomposes a prototype filter into polyphase branches
// and computes interpolation coefficients.
//
// The polyphase decomposition distributes the prototype filter coefficients
// across multiple phases. For coefficient interpolation, we compute polynomial
// coefficients that allow smooth interpolation between adjacent phases.
func decomposePolyphase(prototype []float64, numPhases, tapsPerPhase int, interpOrder InterpOrder) []float64 {
	// Allocate coefficient storage
	// Each tap in each phase stores (interpOrder+1) coefficients
	coeffsPerTap := int(interpOrder) + 1
	totalCoeffs := tapsPerPhase * numPhases * coeffsPerTap
	coeffs := make([]float64, totalCoeffs)

	// Helper function to get prototype coefficient with boundary handling
	getProtoCoeff := func(phase, tap int) float64 {
		idx := tap*numPhases + phase
		if idx < 0 || idx >= len(prototype) {
			return 0.0
		}
		return prototype[idx]
	}

	// For each tap position in each phase, compute interpolation coefficients
	for tap := range tapsPerPhase {
		for phase := range numPhases {
			// Get coefficients from adjacent phases for interpolation
			// f0 = current phase, f1 = next phase, etc.
			prevPhase := max(phase-1, 0)

			f0 := getProtoCoeff(phase, tap)
			f1 := getProtoCoeff(phase+nextPhaseOffset, tap)
			fm1 := getProtoCoeff(prevPhase, tap)
			f2 := getProtoCoeff(phase+secondNextPhaseOffset, tap)

			// Calculate base index for this tap/phase combination
			baseIdx := (tap*numPhases + phase) * coeffsPerTap

			// Store coefficients based on interpolation order
			switch interpOrder {
			case InterpNone:
				// No interpolation: just store the coefficient
				coeffs[baseIdx] = f0

			case InterpLinear:
				// Linear interpolation: f(x) = f0 + b*x
				// where b = f1 - f0
				coeffs[baseIdx] = f0
				coeffs[baseIdx+1] = f1 - f0

			case InterpCubic:
				// Cubic interpolation: f(x) = f0 + b*x + c*x^2 + d*x^3
				// Using centered finite differences for smooth interpolation
				c := cubicCenterCoeff*(f1+fm1) - f0
				d := cubicDCoeff * (f2 - f1 + fm1 - f0 - cubicCMultiplier*c)
				b := f1 - f0 - d - c

				coeffs[baseIdx] = f0
				coeffs[baseIdx+1] = b
				coeffs[baseIdx+2] = c
				coeffs[baseIdx+3] = d
			}
		}
	}

	return coeffs
}

// GetCoefficient returns the interpolated coefficient for a given tap and fractional phase.
//
// Parameters:
//   - tap: The tap index (0 to TapsPerPhase-1)
//   - phase: The integer phase index (0 to NumPhases-1)
//   - frac: The fractional phase position [0, 1) for sub-phase interpolation
func (pfb *PolyphaseFilterBank) GetCoefficient(tap, phase int, frac float64) float64 {
	coeffsPerTap := int(pfb.InterpOrder) + 1
	baseIdx := (tap*pfb.NumPhases + phase) * coeffsPerTap

	switch pfb.InterpOrder {
	case InterpNone:
		return pfb.Coeffs[baseIdx]

	case InterpLinear:
		// Linear: f0 + b*x
		f0 := pfb.Coeffs[baseIdx]
		b := pfb.Coeffs[baseIdx+1]
		return f0 + b*frac

	case InterpCubic:
		// Cubic: f0 + (b + (c + d*x)*x)*x
		// Horner's method for efficient evaluation
		f0 := pfb.Coeffs[baseIdx]
		b := pfb.Coeffs[baseIdx+1]
		c := pfb.Coeffs[baseIdx+2]
		d := pfb.Coeffs[baseIdx+3]
		return f0 + (b+(c+d*frac)*frac)*frac

	default:
		return pfb.Coeffs[baseIdx]
	}
}

// ComputeFrequencyResponse computes the frequency response of the polyphase filter bank.
// This evaluates the response of a single phase (phase 0) as a representative.
func (pfb *PolyphaseFilterBank) ComputeFrequencyResponse(numPoints int) FilterResponse {
	if numPoints <= 0 {
		numPoints = 512
	}

	response := FilterResponse{
		Frequencies: make([]float64, numPoints),
		Magnitude:   make([]float64, numPoints),
		Phase:       make([]float64, numPoints),
	}

	// Extract phase 0 coefficients for frequency response calculation
	phase0Coeffs := make([]float64, pfb.TapsPerPhase)
	coeffsPerTap := int(pfb.InterpOrder) + 1

	for tap := 0; tap < pfb.TapsPerPhase; tap++ {
		baseIdx := (tap*pfb.NumPhases + 0) * coeffsPerTap
		phase0Coeffs[tap] = pfb.Coeffs[baseIdx]
	}

	// Compute DTFT of phase 0
	const twoPi = 2.0 * math.Pi
	for k := 0; k < numPoints; k++ {
		freq := float64(k) / float64(frequencyNyquistDivisor*numPoints)
		response.Frequencies[k] = freq

		var realPart, imagPart float64
		omega := twoPi * freq

		for n, h := range phase0Coeffs {
			angle := omega * float64(n*pfb.NumPhases)
			realPart += h * math.Cos(angle)
			imagPart -= h * math.Sin(angle)
		}

		response.Magnitude[k] = math.Sqrt(realPart*realPart + imagPart*imagPart)
		response.Phase[k] = math.Atan2(imagPart, realPart)
	}

	return response
}

// GetMemoryUsage returns the approximate memory usage in bytes.
func (pfb *PolyphaseFilterBank) GetMemoryUsage() int64 {
	const bytesPerFloat64 = 8
	return int64(len(pfb.Coeffs)) * bytesPerFloat64
}
