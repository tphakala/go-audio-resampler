package engine

import (
	"github.com/tphakala/simd/f64"
	"testing"
)

// BenchmarkDotProduct_20 benchmarks 20-tap dot product (polyphase filter size)
func BenchmarkDotProduct_20(b *testing.B) {
	a := make([]float64, 20)
	c := make([]float64, 20)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}

	for b.Loop() {
		_ = f64.DotProduct(a, c)
	}
}

// BenchmarkDotProduct_20_Manual benchmarks manual loop
func BenchmarkDotProduct_20_Manual(b *testing.B) {
	a := make([]float64, 20)
	c := make([]float64, 20)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}

	for b.Loop() {
		sum := 0.0
		for j := range 20 {
			sum += a[j] * c[j]
		}
		_ = sum
	}
}

// BenchmarkDotProduct_20_Unrolled4 benchmarks 4x unrolled loop
func BenchmarkDotProduct_20_Unrolled4(b *testing.B) {
	a := make([]float64, 20)
	c := make([]float64, 20)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}

	for b.Loop() {
		sum0, sum1, sum2, sum3 := 0.0, 0.0, 0.0, 0.0
		for j := 0; j < 20; j += 4 {
			sum0 += a[j+0] * c[j+0]
			sum1 += a[j+1] * c[j+1]
			sum2 += a[j+2] * c[j+2]
			sum3 += a[j+3] * c[j+3]
		}
		_ = sum0 + sum1 + sum2 + sum3
	}
}
