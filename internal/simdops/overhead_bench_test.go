package simdops

import (
	"testing"

	"github.com/tphakala/simd/f32"
	"github.com/tphakala/simd/f64"
)

// BenchmarkDirectF64DotProduct measures direct SIMD call overhead.
func BenchmarkDirectF64DotProduct(b *testing.B) {
	a := make([]float64, 64)
	c := make([]float64, 64)
	for i := range a {
		a[i] = float64(i) * 0.01
		c[i] = float64(i) * 0.02
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = f64.DotProductUnsafe(a, c)
	}
}

// BenchmarkIndirectF64DotProduct measures indirect call through Ops struct.
func BenchmarkIndirectF64DotProduct(b *testing.B) {
	ops := For[float64]()
	a := make([]float64, 64)
	c := make([]float64, 64)
	for i := range a {
		a[i] = float64(i) * 0.01
		c[i] = float64(i) * 0.02
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = ops.DotProductUnsafe(a, c)
	}
}

// BenchmarkDirectF32DotProduct measures direct SIMD call overhead.
func BenchmarkDirectF32DotProduct(b *testing.B) {
	a := make([]float32, 64)
	c := make([]float32, 64)
	for i := range a {
		a[i] = float32(i) * 0.01
		c[i] = float32(i) * 0.02
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = f32.DotProductUnsafe(a, c)
	}
}

// BenchmarkIndirectF32DotProduct measures indirect call through Ops struct.
func BenchmarkIndirectF32DotProduct(b *testing.B) {
	ops := For[float32]()
	a := make([]float32, 64)
	c := make([]float32, 64)
	for i := range a {
		a[i] = float32(i) * 0.01
		c[i] = float32(i) * 0.02
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = ops.DotProductUnsafe(a, c)
	}
}

// BenchmarkDirectF64ConvolveValid measures direct convolution.
func BenchmarkDirectF64ConvolveValid(b *testing.B) {
	signal := make([]float64, 128)
	kernel := make([]float64, 20)
	dst := make([]float64, 109) // 128 - 20 + 1
	for i := range signal {
		signal[i] = float64(i) * 0.01
	}
	for i := range kernel {
		kernel[i] = float64(i) * 0.05
	}

	b.ReportAllocs()
	for b.Loop() {
		f64.ConvolveValid(dst, signal, kernel)
	}
}

// BenchmarkIndirectF64ConvolveValid measures indirect convolution.
func BenchmarkIndirectF64ConvolveValid(b *testing.B) {
	ops := For[float64]()
	signal := make([]float64, 128)
	kernel := make([]float64, 20)
	dst := make([]float64, 109) // 128 - 20 + 1
	for i := range signal {
		signal[i] = float64(i) * 0.01
	}
	for i := range kernel {
		kernel[i] = float64(i) * 0.05
	}

	b.ReportAllocs()
	for b.Loop() {
		ops.ConvolveValid(dst, signal, kernel)
	}
}

// Larger sizes to measure if overhead becomes negligible
func BenchmarkDirectF64DotProduct_Large(b *testing.B) {
	a := make([]float64, 1024)
	c := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i) * 0.01
		c[i] = float64(i) * 0.02
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = f64.DotProductUnsafe(a, c)
	}
}

func BenchmarkIndirectF64DotProduct_Large(b *testing.B) {
	ops := For[float64]()
	a := make([]float64, 1024)
	c := make([]float64, 1024)
	for i := range a {
		a[i] = float64(i) * 0.01
		c[i] = float64(i) * 0.02
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = ops.DotProductUnsafe(a, c)
	}
}
