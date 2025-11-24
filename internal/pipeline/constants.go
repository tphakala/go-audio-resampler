package pipeline

// DSP algorithm constants
const (
	// dB per bit of precision (20 * log10(2) ≈ 6.02)
	dbPerBit = 6.02

	// Attenuation calculation divisor
	attenuationDivisor = 6.0

	// Minimum and maximum filter taps
	minFilterTaps = 7
	maxFilterTaps = 127

	// Minimum polyphase taps
	minPolyphaseTaps = 4
	maxPolyphaseTaps = 2048

	// Kaiser formula constants
	kaiserOffset     = 8.0
	kaiserMultiplier = 2.285
	kaiserTwoPi      = 6.283185307179586 // 2 * π

	// SIMD alignment - round to multiple of 4
	simdAlignment     = 4
	simdAlignmentMask = 3 // Used with &^ for rounding
)

// Resampling ratio thresholds and factors
const (
	halfRatio         = 0.5    // Half-band decimation ratio
	doubleRatio       = 2.0    // Half-band interpolation ratio
	ratioTolerance    = 0.001  // Tolerance for ratio comparison
	ratioToleranceFFT = 0.0001 // Tighter tolerance for FFT selection
)

// Pipeline stage capacities and sizes
const (
	defaultStageCapacity = 4 // Initial capacity for stages slice
	bufferGrowthFactor   = 2 // Factor for buffer growth
)

// Polyphase filter phases
const (
	phasesBase  = 64   // Base number of phases
	phases24Bit = 256  // Phases for 24-bit precision
	phases32Bit = 1024 // Phases for 32-bit precision
)

// Interpolation orders
const (
	interpOrderNone   = 0 // No interpolation (table lookup)
	interpOrderLinear = 1 // Linear interpolation
	interpOrderCubic  = 3 // Cubic interpolation
)

// FFT sizes
const (
	fftSizeBase  = 1024 // Base FFT size
	fftSize24Bit = 4096 // FFT size for 24-bit precision
	fftSize32Bit = 8192 // FFT size for 32-bit precision
)

// Latency divisors for different stage types
const (
	latencyCubic     = 2 // Cubic needs 4 points, centered
	latencyHalfband  = 2 // Half-band latency divisor
	latencyPolyphase = 2 // Polyphase latency divisor
	latencyFFT       = 4 // Overlap-add latency divisor
)

// Quality precision levels in bits.
// NOTE: These are duplicated from the main resampler package because internal
// packages cannot import the main package (would create import cycle).
const (
	precision8Bit  = 8
	precision16Bit = 16
	precision24Bit = 24
	precision28Bit = 28
	precision32Bit = 32
)

// Common audio sample rate ratios (for FFT optimization)
var commonAudioRatios = []float64{
	44100.0 / 48000.0, // CD to 48kHz
	48000.0 / 44100.0, // 48kHz to CD
	44100.0 / 88200.0, // CD to 2x
	88200.0 / 44100.0, // 2x to CD
	48000.0 / 96000.0, // 48k to 96k
	96000.0 / 48000.0, // 96k to 48k
}
