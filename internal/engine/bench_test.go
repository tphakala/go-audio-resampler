package engine

import (
	"testing"
)

// BenchmarkResampler_CD2DAT benchmarks CD to DAT resampling (44100 -> 48000)
func BenchmarkResampler_CD2DAT(b *testing.B) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of input
	input := make([]float64, 44100)
	for i := range input {
		input[i] = float64(i) * 0.00001
	}

	b.ResetTimer()
	for b.Loop() {
		resampler.Reset()
		_, _ = resampler.Process(input)
	}
}

// BenchmarkResampler_Process benchmarks smaller chunk processing
func BenchmarkResampler_Process(b *testing.B) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// 1024 samples
	input := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i) * 0.001
	}

	b.ResetTimer()
	for b.Loop() {
		_, _ = resampler.Process(input)
	}
}
