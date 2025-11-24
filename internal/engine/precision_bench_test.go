package engine

import (
	"testing"
)

// BenchmarkFloat64Resampler benchmarks float64 precision resampling.
func BenchmarkFloat64Resampler(b *testing.B) {
	resampler, err := NewResampler[float64](44100, 48000, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of audio at 44.1kHz
	input := make([]float64, 44100)
	for i := range input {
		input[i] = float64(i) / float64(len(input))
	}

	b.ReportAllocs()

	for b.Loop() {
		_, err := resampler.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		resampler.Reset()
	}
}

// BenchmarkFloat32Resampler benchmarks float32 precision resampling.
func BenchmarkFloat32Resampler(b *testing.B) {
	resampler, err := NewResampler[float32](44100, 48000, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of audio at 44.1kHz
	input := make([]float32, 44100)
	for i := range input {
		input[i] = float32(i) / float32(len(input))
	}

	b.ReportAllocs()

	for b.Loop() {
		_, err := resampler.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		resampler.Reset()
	}
}

// BenchmarkFloat64DFTStage benchmarks float64 DFT stage.
func BenchmarkFloat64DFTStage(b *testing.B) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	input := make([]float64, 44100)
	for i := range input {
		input[i] = float64(i) / float64(len(input))
	}

	b.ReportAllocs()

	for b.Loop() {
		_, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		stage.Reset()
	}
}

// BenchmarkFloat32DFTStage benchmarks float32 DFT stage.
func BenchmarkFloat32DFTStage(b *testing.B) {
	stage, err := NewDFTStage[float32](2, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	input := make([]float32, 44100)
	for i := range input {
		input[i] = float32(i) / float32(len(input))
	}

	b.ReportAllocs()

	for b.Loop() {
		_, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		stage.Reset()
	}
}

// BenchmarkFloat64Polyphase benchmarks float64 polyphase stage.
func BenchmarkFloat64Polyphase(b *testing.B) {
	// Ratio for 88200 -> 48000 (after 2x upsample)
	ratio := 48000.0 / 88200.0
	totalIORatio := 44100.0 / 48000.0 // For 44100 -> 48000 conversion
	stage, err := NewPolyphaseStage[float64](ratio, totalIORatio, true, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// Input is 2x upsampled (88200 samples for 1 second)
	input := make([]float64, 88200)
	for i := range input {
		input[i] = float64(i) / float64(len(input))
	}

	b.ReportAllocs()

	for b.Loop() {
		_, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		stage.Reset()
	}
}

// BenchmarkFloat32Polyphase benchmarks float32 polyphase stage.
func BenchmarkFloat32Polyphase(b *testing.B) {
	// Ratio for 88200 -> 48000 (after 2x upsample)
	ratio := 48000.0 / 88200.0
	totalIORatio := 44100.0 / 48000.0 // For 44100 -> 48000 conversion
	stage, err := NewPolyphaseStage[float32](ratio, totalIORatio, true, QualityHigh)
	if err != nil {
		b.Fatal(err)
	}

	// Input is 2x upsampled (88200 samples for 1 second)
	input := make([]float32, 88200)
	for i := range input {
		input[i] = float32(i) / float32(len(input))
	}

	b.ReportAllocs()

	for b.Loop() {
		_, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		stage.Reset()
	}
}
