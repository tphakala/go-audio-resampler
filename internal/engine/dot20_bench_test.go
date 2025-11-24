package engine

import (
	"github.com/tphakala/simd/f64"
	"testing"
)

// Sink variable to prevent DCE
var benchSink float64

// BenchmarkDot20_SIMD benchmarks SIMD for 20 elements
func BenchmarkDot20_SIMD(b *testing.B) {
	a := make([]float64, 20)
	c := make([]float64, 20)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}
	var sum float64

	for b.Loop() {
		sum = f64.DotProduct(a, c)
	}
	benchSink = sum
}

// BenchmarkDot20_GoUnrolled4 benchmarks 4x unrolled Go loop
func BenchmarkDot20_GoUnrolled4(b *testing.B) {
	a := make([]float64, 20)
	c := make([]float64, 20)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}
	var sum float64

	for b.Loop() {
		sum0, sum1, sum2, sum3 := 0.0, 0.0, 0.0, 0.0
		for j := 0; j < 20; j += 4 {
			sum0 += a[j+0] * c[j+0]
			sum1 += a[j+1] * c[j+1]
			sum2 += a[j+2] * c[j+2]
			sum3 += a[j+3] * c[j+3]
		}
		sum = sum0 + sum1 + sum2 + sum3
	}
	benchSink = sum
}

// BenchmarkDot20_GoSimple benchmarks simple Go loop
func BenchmarkDot20_GoSimple(b *testing.B) {
	a := make([]float64, 20)
	c := make([]float64, 20)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}
	var sum float64

	for b.Loop() {
		sum = 0
		for j := range 20 {
			sum += a[j] * c[j]
		}
	}
	benchSink = sum
}

// Test full polyphase iteration (div + mod + dot product)
func BenchmarkPolyphaseIteration_SIMD(b *testing.B) {
	coeffs := make([][]float64, 80) // 80 phases
	for p := range coeffs {
		coeffs[p] = make([]float64, 20)
		for t := range coeffs[p] {
			coeffs[p][t] = float64(t) * 0.01
		}
	}
	history := make([]float64, 1000)
	for i := range history {
		history[i] = float64(i)
	}

	numPhases := 80
	step := 147
	var sum float64

	for i := 0; b.Loop(); i++ {
		at := (i * step) % (800 * numPhases) // Cycle through phases
		div := at / numPhases
		phase := at % numPhases
		if div+20 < len(history) {
			sum = f64.DotProduct(coeffs[phase], history[div:div+20])
		}
	}
	benchSink = sum
}

func BenchmarkPolyphaseIteration_GoUnrolled(b *testing.B) {
	coeffs := make([][]float64, 80) // 80 phases
	for p := range coeffs {
		coeffs[p] = make([]float64, 20)
		for t := range coeffs[p] {
			coeffs[p][t] = float64(t) * 0.01
		}
	}
	history := make([]float64, 1000)
	for i := range history {
		history[i] = float64(i)
	}

	numPhases := 80
	step := 147
	var sum float64

	for i := 0; b.Loop(); i++ {
		at := (i * step) % (800 * numPhases)
		div := at / numPhases
		phase := at % numPhases
		if div+20 < len(history) {
			c := coeffs[phase]
			h := history[div:]
			sum0, sum1, sum2, sum3 := 0.0, 0.0, 0.0, 0.0
			for j := 0; j < 20; j += 4 {
				sum0 += c[j+0] * h[j+0]
				sum1 += c[j+1] * h[j+1]
				sum2 += c[j+2] * h[j+2]
				sum3 += c[j+3] * h[j+3]
			}
			sum = sum0 + sum1 + sum2 + sum3
		}
	}
	benchSink = sum
}
