package engine

import (
	"testing"

	"github.com/tphakala/simd/f64"
)

// BenchmarkConvolveValid_Separate benchmarks calling ConvolveValid twice
func BenchmarkConvolveValid_Separate(b *testing.B) {
	signalLen := 44100 + 241
	kernelLen := 241
	outputLen := 44100

	signal := make([]float64, signalLen)
	kernel0 := make([]float64, kernelLen)
	kernel1 := make([]float64, kernelLen)
	out0 := make([]float64, outputLen)
	out1 := make([]float64, outputLen)

	for i := range signal {
		signal[i] = float64(i) * 0.001
	}
	for i := range kernel0 {
		kernel0[i] = 0.004
		kernel1[i] = 0.004
	}

	b.ResetTimer()
	for b.Loop() {
		f64.ConvolveValid(out0, signal, kernel0)
		f64.ConvolveValid(out1, signal, kernel1)
	}
}

// BenchmarkConvolveValidMulti benchmarks ConvolveValidMulti
func BenchmarkConvolveValidMulti(b *testing.B) {
	signalLen := 44100 + 241
	kernelLen := 241
	outputLen := 44100

	signal := make([]float64, signalLen)
	kernels := [][]float64{
		make([]float64, kernelLen),
		make([]float64, kernelLen),
	}
	outputs := [][]float64{
		make([]float64, outputLen),
		make([]float64, outputLen),
	}

	for i := range signal {
		signal[i] = float64(i) * 0.001
	}
	for i := range kernels[0] {
		kernels[0][i] = 0.004
		kernels[1][i] = 0.004
	}

	b.ResetTimer()
	for b.Loop() {
		f64.ConvolveValidMulti(outputs, signal, kernels)
	}
}

// BenchmarkInterleave_Scalar benchmarks scalar interleaving
func BenchmarkInterleave_Scalar(b *testing.B) {
	n := 44100
	a := make([]float64, n)
	c := make([]float64, n)
	dst := make([]float64, n*2)

	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) + 0.5
	}

	b.ResetTimer()
	for b.Loop() {
		for i := range n {
			dst[i*2] = a[i]
			dst[i*2+1] = c[i]
		}
	}
}

// BenchmarkInterleave2 benchmarks SIMD Interleave2
func BenchmarkInterleave2(b *testing.B) {
	n := 44100
	a := make([]float64, n)
	c := make([]float64, n)
	dst := make([]float64, n*2)

	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) + 0.5
	}

	b.ResetTimer()
	for b.Loop() {
		f64.Interleave2(dst, a, c)
	}
}
