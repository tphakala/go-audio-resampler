package engine

import (
	"github.com/tphakala/simd/c128"
	"github.com/tphakala/simd/f64"
	"gonum.org/v1/gonum/dsp/fourier"
)

// FFT convolution constants.
const (
	// Minimum kernel length to use FFT convolution (below this, direct is faster).
	// Benchmarking shows crossover around 400-500 taps with gonum FFT.
	// Direct SIMD convolution is faster for typical filter lengths (128-256 taps).
	minKernelForFFT = 400

	// Default FFT block size (power of 2 for efficiency)
	defaultFFTBlockSize = 512

	// fftHermitianDivisor is used to calculate unique frequency bins in real FFT.
	// Due to Hermitian symmetry, a real FFT of size N has N/2 + 1 unique complex coefficients.
	fftHermitianDivisor = 2
)

// FFTConvolver performs overlap-save FFT convolution for long filters.
// This is O(N log N) vs O(N×M) for direct convolution, beneficial for long kernels.
//
// Overlap-save method:
//  1. Process input in blocks of fftSize samples (with kernelLen-1 overlap)
//  2. Each block produces blockSize = fftSize - kernelLen + 1 valid output samples
//  3. The first kernelLen-1 output samples of each block are discarded (circular wrap)
type FFTConvolver struct {
	fft       *fourier.FFT
	fftSize   int
	blockSize int // Valid output samples per block = fftSize - kernelLen + 1

	// Precomputed kernel in frequency domain
	kernelFFT []complex128
	kernelLen int
	fftLen    int     // Length of FFT output = fftSize/2 + 1
	scale     float64 // 1/fftSize for IFFT normalization (gonum doesn't normalize)

	// Working buffers (pre-allocated for zero allocation during processing)
	signalBlock []float64
	signalFFT   []complex128
	productFFT  []complex128
	ifftResult  []float64
}

// NewFFTConvolver creates a new FFT convolver for the given kernel.
// The kernel is transformed once and reused for all convolutions.
func NewFFTConvolver(kernel []float64) *FFTConvolver {
	kernelLen := len(kernel)
	if kernelLen == 0 {
		return nil
	}

	// Choose FFT size: next power of 2 >= 2*kernelLen for good efficiency
	fftSize := defaultFFTBlockSize
	for fftSize < 2*kernelLen {
		fftSize *= 2
	}

	// Valid output samples per block (overlap-save method)
	blockSize := fftSize - kernelLen + 1

	// Create FFT instance
	fft := fourier.NewFFT(fftSize)

	// Precompute kernel FFT (zero-padded to fftSize)
	// IMPORTANT: Reverse the kernel for convolution (vs correlation)
	// FFT circular convolution computes: y[n] = sum(x[(n-k) mod N] * h[k])
	// We want: y[n] = sum(x[n+k] * h[k]) (the "valid" convolution)
	// Reversing h gives us the correct result
	kernelPadded := make([]float64, fftSize)
	for i := range kernelLen {
		kernelPadded[i] = kernel[kernelLen-1-i]
	}
	kernelFFT := fft.Coefficients(nil, kernelPadded)

	fftLen := fftSize/fftHermitianDivisor + 1

	return &FFTConvolver{
		fft:         fft,
		fftSize:     fftSize,
		blockSize:   blockSize,
		kernelFFT:   kernelFFT,
		kernelLen:   kernelLen,
		fftLen:      fftLen,
		scale:       1.0 / float64(fftSize),
		signalBlock: make([]float64, fftSize),
		signalFFT:   make([]complex128, fftLen),
		productFFT:  make([]complex128, fftLen),
		ifftResult:  make([]float64, fftSize),
	}
}

// Convolve performs overlap-save convolution.
// dst must have length >= len(signal) - kernelLen + 1
func (c *FFTConvolver) Convolve(dst, signal []float64) {
	signalLen := len(signal)
	outputLen := signalLen - c.kernelLen + 1
	if outputLen <= 0 || len(dst) < outputLen {
		return
	}

	// Overlap-save method with reversed kernel:
	// - Each FFT block produces blockSize valid output samples
	// - Block b reads signal[b*blockSize : b*blockSize + fftSize] (zero-padded at end if needed)
	// - Output y[kernelLen-1 + i] corresponds to convolution at position b*blockSize + i
	// - The first (kernelLen-1) outputs are discarded (circular wrap artifacts)

	outIdx := 0
	overlap := c.kernelLen - 1

	for outIdx < outputLen {
		// Clear the signal block
		for i := range c.signalBlock {
			c.signalBlock[i] = 0
		}

		// Copy signal starting at position outIdx (which is b*blockSize)
		// We need fftSize samples, but may have fewer if near end
		copyLen := c.fftSize
		if outIdx+copyLen > signalLen {
			copyLen = signalLen - outIdx
		}
		if copyLen > 0 {
			copy(c.signalBlock, signal[outIdx:outIdx+copyLen])
		}

		// FFT of signal block
		c.signalFFT = c.fft.Coefficients(c.signalFFT, c.signalBlock)

		// Multiply in frequency domain using SIMD
		c.multiplyFFT()

		// IFFT
		c.ifftResult = c.fft.Sequence(c.ifftResult, c.productFFT)

		// Scale by 1/N (gonum's IFFT doesn't normalize)
		f64.Scale(c.ifftResult, c.ifftResult, c.scale)

		// Valid output samples start at offset 'overlap' (= kernelLen - 1)
		validSamples := c.blockSize
		if outIdx+validSamples > outputLen {
			validSamples = outputLen - outIdx
		}

		// Copy valid samples to output
		copy(dst[outIdx:outIdx+validSamples], c.ifftResult[overlap:overlap+validSamples])

		outIdx += validSamples
	}
}

// multiplyFFT multiplies signalFFT by kernelFFT into productFFT using SIMD.
func (c *FFTConvolver) multiplyFFT() {
	// Use SIMD complex multiplication
	c128.Mul(c.productFFT, c.signalFFT, c.kernelFFT)
}

// ConvolveValidFFT is a convenience function that uses FFT convolution
// when beneficial, falling back to direct convolution for short kernels.
func ConvolveValidFFT(dst, signal, kernel []float64) {
	if len(kernel) < minKernelForFFT {
		// Use direct SIMD convolution for short kernels
		f64.ConvolveValid(dst, signal, kernel)
		return
	}

	// Use FFT convolution for long kernels
	conv := NewFFTConvolver(kernel)
	if conv != nil {
		conv.Convolve(dst, signal)
	}
}
