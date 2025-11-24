package engine

import (
	"github.com/tphakala/simd/f64"
	"testing"
)

func benchDotSIMD(b *testing.B, size int) {
	b.Helper()
	a := make([]float64, size)
	c := make([]float64, size)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f64.DotProduct(a, c)
	}
}

func benchDotUnrolled(b *testing.B, size int) {
	b.Helper()
	a := make([]float64, size)
	c := make([]float64, size)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum0, sum1, sum2, sum3 := 0.0, 0.0, 0.0, 0.0
		end4 := size &^ 3
		for j := 0; j < end4; j += 4 {
			sum0 += a[j+0] * c[j+0]
			sum1 += a[j+1] * c[j+1]
			sum2 += a[j+2] * c[j+2]
			sum3 += a[j+3] * c[j+3]
		}
		sum := sum0 + sum1 + sum2 + sum3
		for j := end4; j < size; j++ {
			sum += a[j] * c[j]
		}
		_ = sum
	}
}

func BenchmarkCrossover_16_SIMD(b *testing.B)      { benchDotSIMD(b, 16) }
func BenchmarkCrossover_16_Unrolled(b *testing.B)  { benchDotUnrolled(b, 16) }
func BenchmarkCrossover_20_SIMD(b *testing.B)      { benchDotSIMD(b, 20) }
func BenchmarkCrossover_20_Unrolled(b *testing.B)  { benchDotUnrolled(b, 20) }
func BenchmarkCrossover_32_SIMD(b *testing.B)      { benchDotSIMD(b, 32) }
func BenchmarkCrossover_32_Unrolled(b *testing.B)  { benchDotUnrolled(b, 32) }
func BenchmarkCrossover_48_SIMD(b *testing.B)      { benchDotSIMD(b, 48) }
func BenchmarkCrossover_48_Unrolled(b *testing.B)  { benchDotUnrolled(b, 48) }
func BenchmarkCrossover_64_SIMD(b *testing.B)      { benchDotSIMD(b, 64) }
func BenchmarkCrossover_64_Unrolled(b *testing.B)  { benchDotUnrolled(b, 64) }
func BenchmarkCrossover_128_SIMD(b *testing.B)     { benchDotSIMD(b, 128) }
func BenchmarkCrossover_128_Unrolled(b *testing.B) { benchDotUnrolled(b, 128) }
