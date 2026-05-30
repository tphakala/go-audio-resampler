// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

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
// input, the BirdNET v3.0 production clip size (240K input samples).
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
// with 5s input, the BirdNET v3.0 production clip size.
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

// BenchmarkProcessIntoFloat32_Warm_48to32 benchmarks steady-state float32
// ProcessInto on the engine path (SimpleResamplerFloat32) without Reset between
// iterations. The engine is float32-native, so there is no float64 round-trip.
func BenchmarkProcessIntoFloat32_Warm_48to32(b *testing.B) {
	r, err := NewEngineFloat32(48000, 32000, QualityMedium)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float32, 48000*5)
	for i := range input {
		input[i] = float32(i) * 1e-5
	}
	output := make([]float32, r.EstimateOutput(len(input)))

	// Warm up so internal buffers reach steady-state size.
	_, _ = r.ProcessInto(input, output)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_, _ = r.ProcessInto(input, output)
	}
}

// BenchmarkProcessFloat32Into_Warm_48to32 benchmarks steady-state float32
// ProcessFloat32Into on the New(...) batch path. This path runs through the
// float64 pipeline using grow-only scratch buffers, so it is zero-alloc once
// warm.
func BenchmarkProcessFloat32Into_Warm_48to32(b *testing.B) {
	raw, err := New(&Config{
		InputRate:  48000,
		OutputRate: 32000,
		Channels:   1,
		Quality:    QualitySpec{Preset: QualityMedium},
	})
	if err != nil {
		b.Fatal(err)
	}
	r, ok := raw.(processFloat32IntoResampler)
	if !ok {
		b.Fatal("New(...) result does not implement ProcessFloat32Into")
	}
	input := make([]float32, 48000*5)
	for i := range input {
		input[i] = float32(i) * 1e-5
	}
	output := make([]float32, r.EstimateOutput(len(input)))

	// Warm up so internal buffers and scratch slices reach steady-state size.
	_, _ = r.ProcessFloat32Into(input, output)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_, _ = r.ProcessFloat32Into(input, output)
	}
}
