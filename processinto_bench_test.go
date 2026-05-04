package resampler

import (
	"testing"
)

// BenchmarkProcess_Allocations benchmarks the allocating Process path
// for the most common birdnet-go rate pair (48kHz -> 16kHz, 3s chunks).
func BenchmarkProcess_Allocations(b *testing.B) {
	r, err := NewEngine(48000, 16000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*3) // 3 seconds
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.Reset()
		_, _ = r.Process(input)
	}
}

// BenchmarkProcessInto_Allocations benchmarks the zero-copy ProcessInto path
// for the same rate pair. Expected: 0 allocs/op after warm-up.
func BenchmarkProcessInto_Allocations(b *testing.B) {
	r, err := NewEngine(48000, 16000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*3)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	output := make([]float64, r.EstimateOutput(len(input)))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.Reset()
		_, _ = r.ProcessInto(input, output)
	}
}

// BenchmarkProcess_Allocations_48to32 benchmarks 48kHz -> 32kHz (3s).
func BenchmarkProcess_Allocations_48to32(b *testing.B) {
	r, err := NewEngine(48000, 32000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*3)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.Reset()
		_, _ = r.Process(input)
	}
}

// BenchmarkProcessInto_Allocations_48to32 benchmarks zero-copy 48kHz -> 32kHz (3s).
func BenchmarkProcessInto_Allocations_48to32(b *testing.B) {
	r, err := NewEngine(48000, 32000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*3)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	output := make([]float64, r.EstimateOutput(len(input)))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.Reset()
		_, _ = r.ProcessInto(input, output)
	}
}

// BenchmarkProcess_Allocations_48to32_5s benchmarks 48kHz -> 32kHz with 5s
// input — the BirdNET v3.0 production clip size (240K input samples).
func BenchmarkProcess_Allocations_48to32_5s(b *testing.B) {
	r, err := NewEngine(48000, 32000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*5)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.Reset()
		_, _ = r.Process(input)
	}
}

// BenchmarkProcessInto_Allocations_48to32_5s benchmarks zero-copy 48kHz -> 32kHz
// with 5s input — the BirdNET v3.0 production clip size.
func BenchmarkProcessInto_Allocations_48to32_5s(b *testing.B) {
	r, err := NewEngine(48000, 32000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*5)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	output := make([]float64, r.EstimateOutput(len(input)))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.Reset()
		_, _ = r.ProcessInto(input, output)
	}
}

// BenchmarkProcessInto_Warm_48to32 benchmarks steady-state ProcessInto without
// Reset between iterations. This matches the real birdnet-go usage pattern where
// a single resampler processes consecutive audio chunks continuously.
func BenchmarkProcessInto_Warm_48to32(b *testing.B) {
	r, err := NewEngine(48000, 32000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float64, 48000*5)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}
	output := make([]float64, r.EstimateOutput(len(input)))

	// Warm up so internal buffers reach steady-state size.
	_, _ = r.ProcessInto(input, output)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_, _ = r.ProcessInto(input, output)
	}
}
