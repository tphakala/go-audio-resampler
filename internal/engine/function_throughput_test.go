package engine

import (
	"math"
	"testing"

	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

// =============================================================================
// Individual Function Throughput Benchmarks
// =============================================================================
//
// These benchmarks isolate individual functions to identify optimization targets.
// Run with: go test -bench=BenchmarkFunc -benchtime=3s ./internal/engine/

// -----------------------------------------------------------------------------
// SIMD Operations Benchmarks
// -----------------------------------------------------------------------------

// BenchmarkFunc_ConvolveValid benchmarks the core convolution operation.
// This is the most critical operation for DFT stage performance.
func BenchmarkFunc_ConvolveValid_Taps32(b *testing.B) {
	benchConvolveValid(b, 32, 4096)
}

func BenchmarkFunc_ConvolveValid_Taps64(b *testing.B) {
	benchConvolveValid(b, 64, 4096)
}

func BenchmarkFunc_ConvolveValid_Taps128(b *testing.B) {
	benchConvolveValid(b, 128, 4096)
}

func BenchmarkFunc_ConvolveValid_Taps256(b *testing.B) {
	benchConvolveValid(b, 256, 4096)
}

//nolint:unparam // signalLen kept as parameter for future test flexibility
func benchConvolveValid(b *testing.B, taps, signalLen int) {
	b.Helper()
	ops := simdops.Float64Ops()

	// Create test data
	signal := make([]float64, signalLen+taps-1)
	kernel := make([]float64, taps)
	dst := make([]float64, signalLen)

	for i := range signal {
		signal[i] = math.Sin(float64(i) * 0.1)
	}
	for i := range kernel {
		kernel[i] = 1.0 / float64(taps)
	}

	b.ResetTimer()
	b.SetBytes(int64(signalLen * 8)) // bytes processed

	for b.Loop() {
		ops.ConvolveValid(dst, signal, kernel)
	}

	// Report samples per second
	samplesPerOp := float64(signalLen)
	b.ReportMetric(samplesPerOp*float64(b.N)/b.Elapsed().Seconds()/1e6, "MS/s")
}

// BenchmarkFunc_ConvolveValidMulti benchmarks multi-kernel convolution (DFT polyphase).
func BenchmarkFunc_ConvolveValidMulti_2Phase_Taps64(b *testing.B) {
	benchConvolveValidMulti(b, 2, 64, 4096)
}

func BenchmarkFunc_ConvolveValidMulti_4Phase_Taps64(b *testing.B) {
	benchConvolveValidMulti(b, 4, 64, 4096)
}

func benchConvolveValidMulti(b *testing.B, phases, taps, signalLen int) {
	b.Helper()
	ops := simdops.Float64Ops()

	// Create test data
	signal := make([]float64, signalLen+taps-1)
	kernels := make([][]float64, phases)
	dsts := make([][]float64, phases)

	for i := range signal {
		signal[i] = math.Sin(float64(i) * 0.1)
	}
	for p := range phases {
		kernels[p] = make([]float64, taps)
		dsts[p] = make([]float64, signalLen)
		for i := range taps {
			kernels[p][i] = 1.0 / float64(taps*phases)
		}
	}

	b.ResetTimer()
	b.SetBytes(int64(signalLen * phases * 8))

	for b.Loop() {
		ops.ConvolveValidMulti(dsts, signal, kernels)
	}

	samplesPerOp := float64(signalLen * phases)
	b.ReportMetric(samplesPerOp*float64(b.N)/b.Elapsed().Seconds()/1e6, "MS/s")
}

// BenchmarkFunc_DotProduct benchmarks dot product for various sizes.
func BenchmarkFunc_DotProduct_20(b *testing.B) {
	benchDotProduct(b, 20)
}

func BenchmarkFunc_DotProduct_32(b *testing.B) {
	benchDotProduct(b, 32)
}

func BenchmarkFunc_DotProduct_64(b *testing.B) {
	benchDotProduct(b, 64)
}

func BenchmarkFunc_DotProduct_128(b *testing.B) {
	benchDotProduct(b, 128)
}

func benchDotProduct(b *testing.B, size int) {
	b.Helper()
	ops := simdops.Float64Ops()

	a := make([]float64, size)
	c := make([]float64, size)
	for i := range size {
		a[i] = float64(i) * 0.1
		c[i] = float64(size-i) * 0.1
	}

	var sum float64
	b.ResetTimer()

	for b.Loop() {
		sum = ops.DotProductUnsafe(a, c)
	}

	_ = sum
	b.ReportMetric(float64(size)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
}

// BenchmarkFunc_Interleave2 benchmarks 2-channel interleaving.
func BenchmarkFunc_Interleave2_4096(b *testing.B) {
	benchInterleave2(b, 4096)
}

func BenchmarkFunc_Interleave2_8192(b *testing.B) {
	benchInterleave2(b, 8192)
}

func benchInterleave2(b *testing.B, size int) {
	b.Helper()
	ops := simdops.Float64Ops()

	a := make([]float64, size)
	c := make([]float64, size)
	dst := make([]float64, size*2)

	for i := range size {
		a[i] = float64(i)
		c[i] = float64(i + size)
	}

	b.ResetTimer()
	b.SetBytes(int64(size * 2 * 8))

	for b.Loop() {
		ops.Interleave2(dst, a, c)
	}

	b.ReportMetric(float64(size*2)*float64(b.N)/b.Elapsed().Seconds()/1e6, "MS/s")
}

// -----------------------------------------------------------------------------
// Stage-Level Benchmarks
// -----------------------------------------------------------------------------

// BenchmarkFunc_DFTStage benchmarks the DFT upsampling stage.
func BenchmarkFunc_DFTStage_Quick(b *testing.B) {
	benchDFTStage(b, QualityQuick)
}

func BenchmarkFunc_DFTStage_Medium(b *testing.B) {
	benchDFTStage(b, QualityMedium)
}

func BenchmarkFunc_DFTStage_VeryHigh(b *testing.B) {
	benchDFTStage(b, QualityVeryHigh)
}

func benchDFTStage(b *testing.B, quality Quality) {
	b.Helper()
	stage, err := NewDFTStage[float64](2, quality)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of 48kHz audio
	inputLen := 48000
	input := make([]float64, inputLen)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / 48000.0)
	}

	b.ResetTimer()

	for b.Loop() {
		stage.Reset()
		output, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		_ = output
	}

	b.ReportMetric(float64(int64(b.N)*int64(len(input)))/b.Elapsed().Seconds()/1e6, "MS/s")
}

// BenchmarkFunc_PolyphaseStage benchmarks the polyphase resampling stage.
func BenchmarkFunc_PolyphaseStage_Quick_Down(b *testing.B) {
	benchPolyphaseStage(b, QualityQuick, 48000.0/32000.0, true)
}

func BenchmarkFunc_PolyphaseStage_Medium_Down(b *testing.B) {
	benchPolyphaseStage(b, QualityMedium, 48000.0/32000.0, true)
}

func BenchmarkFunc_PolyphaseStage_VeryHigh_Down(b *testing.B) {
	benchPolyphaseStage(b, QualityVeryHigh, 48000.0/32000.0, true)
}

func BenchmarkFunc_PolyphaseStage_VeryHigh_Up(b *testing.B) {
	// 44.1 -> 48 kHz with 2x pre-stage means polyphase ratio = 48/(44.1*2)
	benchPolyphaseStage(b, QualityVeryHigh, 48000.0/(44100.0*2), false)
}

func benchPolyphaseStage(b *testing.B, quality Quality, ratio float64, isDownsampling bool) {
	b.Helper()
	var totalIORatio float64
	if isDownsampling {
		totalIORatio = 1.0 / ratio // input/output for downsampling
	} else {
		totalIORatio = ratio // Already output/input
	}

	stage, err := NewPolyphaseStage[float64](ratio, totalIORatio, !isDownsampling, quality)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of 48kHz audio
	inputLen := 48000
	input := make([]float64, inputLen)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / 48000.0)
	}

	b.ResetTimer()

	for b.Loop() {
		stage.Reset()
		output, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		_ = output
	}

	b.ReportMetric(float64(int64(b.N)*int64(len(input)))/b.Elapsed().Seconds()/1e6, "MS/s")
}

// -----------------------------------------------------------------------------
// Inner Loop Benchmarks (Polyphase coefficient interpolation)
// -----------------------------------------------------------------------------

// BenchmarkFunc_CubicInterpolation benchmarks the cubic coefficient interpolation
// which is the hot inner loop of PolyphaseStage.Process.
func BenchmarkFunc_CubicInterpolation_20Taps(b *testing.B) {
	benchCubicInterpolation(b, 20)
}

func BenchmarkFunc_CubicInterpolation_32Taps(b *testing.B) {
	benchCubicInterpolation(b, 32)
}

func BenchmarkFunc_CubicInterpolation_64Taps(b *testing.B) {
	benchCubicInterpolation(b, 64)
}

func BenchmarkFunc_CubicInterpolation_100Taps(b *testing.B) {
	benchCubicInterpolation(b, 100)
}

func benchCubicInterpolation(b *testing.B, tapsPerPhase int) {
	b.Helper()
	// Simulate the polyphase inner loop with cubic coefficient interpolation
	coeffsA := make([]float64, tapsPerPhase)
	coeffsB := make([]float64, tapsPerPhase)
	coeffsC := make([]float64, tapsPerPhase)
	coeffsD := make([]float64, tapsPerPhase)
	hist := make([]float64, tapsPerPhase)

	// Initialize with realistic values
	for i := range tapsPerPhase {
		t := float64(i) / float64(tapsPerPhase)
		coeffsA[i] = math.Sin(t * math.Pi)
		coeffsB[i] = 0.1 * math.Cos(t*math.Pi)
		coeffsC[i] = 0.01 * math.Sin(2*t*math.Pi)
		coeffsD[i] = 0.001 * math.Cos(2*t*math.Pi)
		hist[i] = math.Sin(float64(i) * 0.1)
	}

	x := float64(0.3) // Fractional phase

	var sum float64
	b.ResetTimer()

	for b.Loop() {
		sum = 0
		for tap := range tapsPerPhase {
			// This is exactly the hot loop from PolyphaseStage.Process
			a := coeffsA[tap]
			bc := coeffsB[tap]
			c := coeffsC[tap]
			d := coeffsD[tap]
			interpolatedCoef := a + x*(bc+x*(c+x*d))
			sum += interpolatedCoef * hist[tap]
		}
	}

	_ = sum

	// Report operations per second (each tap = 7 FLOPs: 4 muls + 3 adds)
	flopsPerTap := 7.0
	b.ReportMetric(float64(tapsPerPhase)*flopsPerTap*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds()/1e6, "M_samples/s")
}

// BenchmarkFunc_CubicInterpolation_Unrolled benchmarks an unrolled version.
func BenchmarkFunc_CubicInterpolation_64Taps_Unrolled4(b *testing.B) {
	const tapsPerPhase = 64
	coeffsA := make([]float64, tapsPerPhase)
	coeffsB := make([]float64, tapsPerPhase)
	coeffsC := make([]float64, tapsPerPhase)
	coeffsD := make([]float64, tapsPerPhase)
	hist := make([]float64, tapsPerPhase)

	for i := range tapsPerPhase {
		t := float64(i) / float64(tapsPerPhase)
		coeffsA[i] = math.Sin(t * math.Pi)
		coeffsB[i] = 0.1 * math.Cos(t*math.Pi)
		coeffsC[i] = 0.01 * math.Sin(2*t*math.Pi)
		coeffsD[i] = 0.001 * math.Cos(2*t*math.Pi)
		hist[i] = math.Sin(float64(i) * 0.1)
	}

	x := float64(0.3)

	var sum float64
	b.ResetTimer()

	for b.Loop() {
		sum = 0
		// Unrolled by 4
		for tap := 0; tap < tapsPerPhase; tap += 4 {
			a0 := coeffsA[tap]
			b0 := coeffsB[tap]
			c0 := coeffsC[tap]
			d0 := coeffsD[tap]
			sum += (a0 + x*(b0+x*(c0+x*d0))) * hist[tap]

			a1 := coeffsA[tap+1]
			b1 := coeffsB[tap+1]
			c1 := coeffsC[tap+1]
			d1 := coeffsD[tap+1]
			sum += (a1 + x*(b1+x*(c1+x*d1))) * hist[tap+1]

			a2 := coeffsA[tap+2]
			b2 := coeffsB[tap+2]
			c2 := coeffsC[tap+2]
			d2 := coeffsD[tap+2]
			sum += (a2 + x*(b2+x*(c2+x*d2))) * hist[tap+2]

			a3 := coeffsA[tap+3]
			b3 := coeffsB[tap+3]
			c3 := coeffsC[tap+3]
			d3 := coeffsD[tap+3]
			sum += (a3 + x*(b3+x*(c3+x*d3))) * hist[tap+3]
		}
	}

	_ = sum
	b.ReportMetric(float64(tapsPerPhase)*7.0*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds()/1e6, "M_samples/s")
}

// -----------------------------------------------------------------------------
// Full Pipeline Benchmarks (for comparison)
// -----------------------------------------------------------------------------

// BenchmarkFunc_FullPipeline_Downsample benchmarks the complete resampler.
func BenchmarkFunc_FullPipeline_48kTo32k_Quick(b *testing.B) {
	benchFullPipeline(b, 48000, 32000, QualityQuick)
}

func BenchmarkFunc_FullPipeline_48kTo32k_Medium(b *testing.B) {
	benchFullPipeline(b, 48000, 32000, QualityMedium)
}

func BenchmarkFunc_FullPipeline_48kTo32k_VeryHigh(b *testing.B) {
	benchFullPipeline(b, 48000, 32000, QualityVeryHigh)
}

func BenchmarkFunc_FullPipeline_44kTo48k_VeryHigh(b *testing.B) {
	benchFullPipeline(b, 44100, 48000, QualityVeryHigh)
}

func benchFullPipeline(b *testing.B, inputRate, outputRate float64, quality Quality) {
	b.Helper()
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of input audio
	inputLen := int(inputRate)
	input := make([]float64, inputLen)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / inputRate)
	}

	b.ResetTimer()

	for b.Loop() {
		resampler.Reset()
		output, err := resampler.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		_ = output
	}

	b.ReportMetric(float64(int64(b.N)*int64(len(input)))/b.Elapsed().Seconds()/1e6, "MS/s")
}

// -----------------------------------------------------------------------------
// Test to print throughput breakdown
// -----------------------------------------------------------------------------

func TestFunctionThroughput_Summary(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping throughput summary in short mode")
	}

	t.Log("============================================================")
	t.Log("Function-Level Throughput Analysis")
	t.Log("============================================================")
	t.Log("")
	t.Log("Run benchmarks with:")
	t.Log("  GOAMD64=v3 go test -bench=BenchmarkFunc -benchtime=3s ./internal/engine/")
	t.Log("")
	t.Log("Key functions to optimize (in order of impact):")
	t.Log("")
	t.Log("1. PolyphaseStage inner loop (cubic interpolation)")
	t.Log("   - Runs tapsPerPhase times per output sample")
	t.Log("   - For VeryHigh: ~64-100 taps/sample = most CPU time")
	t.Log("   - Benchmark: BenchmarkFunc_CubicInterpolation_*")
	t.Log("")
	t.Log("2. ConvolveValid (DFT stage convolution)")
	t.Log("   - Used for each phase in DFT upsampling")
	t.Log("   - Already uses SIMD via github.com/tphakala/simd")
	t.Log("   - Benchmark: BenchmarkFunc_ConvolveValid_*")
	t.Log("")
	t.Log("3. DotProductUnsafe")
	t.Log("   - Core operation for convolution")
	t.Log("   - Benchmark: BenchmarkFunc_DotProduct_*")
	t.Log("")
	t.Log("4. Interleave2")
	t.Log("   - Used after DFT polyphase filtering")
	t.Log("   - Memory-bound operation")
	t.Log("   - Benchmark: BenchmarkFunc_Interleave2_*")
	t.Log("")
}
