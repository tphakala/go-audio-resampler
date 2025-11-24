package engine

import (
	"testing"
)

// BenchmarkDFTStage_Only benchmarks just the DFT stage
func BenchmarkDFTStage_Only(b *testing.B) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	input := make([]float64, 44100)
	for i := range input {
		input[i] = float64(i) * 0.00001
	}

	b.ResetTimer()
	for b.Loop() {
		stage.Reset()
		_, _ = stage.Process(input)
	}
}

// BenchmarkPolyphaseStage_Only benchmarks just the polyphase stage
func BenchmarkPolyphaseStage_Only(b *testing.B) {
	// Ratio for 88200 -> 48000 (after 2x upsample)
	ratio := 48000.0 / 88200.0
	// For standalone test, use ratio as totalIORatio
	stage, err := NewPolyphaseStage[float64](ratio, ratio, true, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// Input is 2x upsampled (88200 samples for 1 second)
	input := make([]float64, 88200)
	for i := range input {
		input[i] = float64(i) * 0.00001
	}

	b.ResetTimer()
	for b.Loop() {
		stage.Reset()
		_, _ = stage.Process(input)
	}
}

// BenchmarkResampler_QualityLow benchmarks with QualityLow
func BenchmarkResampler_QualityLow(b *testing.B) {
	resampler, err := NewResampler[float64](44100, 48000, QualityLow)
	if err != nil {
		b.Fatal(err)
	}
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

// BenchmarkResampler_QualityMedium benchmarks with QualityMedium
func BenchmarkResampler_QualityMedium(b *testing.B) {
	resampler, err := NewResampler[float64](44100, 48000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
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
