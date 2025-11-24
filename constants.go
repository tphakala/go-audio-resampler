package resampler

// Channel constants
const (
	stereoChannels = 2   // Stereo channel count (used by interleave functions)
	maxChannels    = 256 // Maximum supported channel count
)

// Quality precision levels in bits
const (
	precision8Bit  = 8
	precision16Bit = 16
	precision24Bit = 24
	precision32Bit = 32
	precision33Bit = 33
)

// Quality preset parameters
const (
	// Phase response values
	linearPhaseResponse = 50.0 // Linear phase (symmetric impulse response)

	// Quick quality (8-bit)
	quickPassbandEnd   = 0.7
	quickStopbandBegin = 1.0

	// Low quality (16-bit)
	lowPassbandEnd   = 0.80
	lowStopbandBegin = 0.95

	// Medium quality (16-bit)
	mediumPassbandEnd   = 0.90
	mediumStopbandBegin = 0.98

	// High quality (24-bit)
	highPassbandEnd   = 0.95
	highStopbandBegin = 0.99

	// Very high quality (32-bit)
	veryHighPassbandEnd   = 0.99
	veryHighStopbandBegin = 0.995
)

// Resampling ratio limits
const (
	minRatioFactor = 1.0 / 256.0 // Minimum resampling ratio (1/256)
	maxRatioFactor = 256.0       // Maximum resampling ratio (256x)
)

// Buffer and memory constants
const (
	defaultBufferSize    = 8192 // Default ring buffer size in samples
	bytesPerFloat64      = 8    // Size of float64 in bytes
	bufferSizeMultiplier = 2    // Multiplier for buffer size based on input size
)

// Stage processing constants
const (
	latencyDivisor  = 2 // Divisor for calculating filter latency
	minInputSamples = 1 // Minimum input samples for processing
)
