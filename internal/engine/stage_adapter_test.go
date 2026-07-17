// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"testing"
)

// TestStageAdapterFloat32_ProcessZeroCopy verifies the float32 adapter fallback
// path handles []float64 input without panicking.
func TestStageAdapterFloat32_ProcessZeroCopy(t *testing.T) {
	r, err := NewResampler[float32](44100, 48000, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}

	adapter := NewStageAdapter(r)
	input := make([]float64, 4096)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * 440.0 * float64(i) / 44100.0)
	}

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("ProcessZeroCopy panicked for float32 adapter: %v", recovered)
		}
	}()

	if _, err := adapter.ProcessZeroCopy(input); err != nil {
		t.Fatalf("ProcessZeroCopy returned error: %v", err)
	}
}

// The public resample-package pipeline never wires QualityQuick through a
// StageAdapter, so GetLatency's cubic branch is only reachable by constructing
// the adapter directly. Pin it here.
func TestStageAdapter_GetLatency_CubicBranch(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityQuick)
	if err != nil {
		t.Fatal(err)
	}
	if r.cubicStage == nil {
		t.Fatal("QualityQuick did not create a cubic stage")
	}
	adapter := NewStageAdapter(r)
	if got := adapter.GetLatency(); got != cubicLatencySamples {
		t.Errorf("GetLatency() = %d, want %d (cubic branch)", got, cubicLatencySamples)
	}
}

// A QualityQuick resampler has only a cubic stage, so GetMemoryUsage must
// report exactly the cubic stage's own accounting.
func TestStageAdapter_GetMemoryUsage_CubicBranch(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityQuick)
	if err != nil {
		t.Fatal(err)
	}
	if r.cubicStage == nil {
		t.Fatal("QualityQuick did not create a cubic stage")
	}
	adapter := NewStageAdapter(r)
	want := r.cubicStage.GetMemoryUsage()
	if got := adapter.GetMemoryUsage(); got != want {
		t.Errorf("GetMemoryUsage() = %d, want %d (cubic branch)", got, want)
	}
}

// Integer 2:1 downsampling wires only the DFT decimation stage; exercise the
// decimation branch of GetMemoryUsage.
func TestStageAdapter_GetMemoryUsage_DecimationBranch(t *testing.T) {
	r, err := NewResampler[float64](96000, 48000, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}
	if r.decimationStage == nil {
		t.Fatal("expected a decimation stage for 96000->48000")
	}
	adapter := NewStageAdapter(r)
	const bytesPerElement = int64(bytesPerFloat64)
	want := int64(len(r.decimationStage.coeffs))*bytesPerElement +
		int64(cap(r.decimationStage.history))*bytesPerElement
	if got := adapter.GetMemoryUsage(); got != want {
		t.Errorf("GetMemoryUsage() = %d, want %d (decimation branch)", got, want)
	}
}

// GetMemoryUsage must count all four polyphase coefficient banks (a, b, c, d),
// not just the base bank; counting only polyCoeffs undercounts the
// coefficients by 4x.
func TestStageAdapter_GetMemoryUsage_CountsAllFourPolyphaseBanks(t *testing.T) {
	// Non-integer downsampling wires a DFT pre-stage plus a polyphase stage
	// whose four cubic-interpolation coefficient banks all consume memory.
	r, err := NewResampler[float64](48000, 44100, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}
	if r.polyphaseStage == nil {
		t.Fatal("expected a polyphase stage for 48000->44100")
	}
	adapter := NewStageAdapter(r)

	const bytesPerElement = int64(bytesPerFloat64)
	bankBytes := func(bank [][]float64) int64 {
		var n int64
		for _, phase := range bank {
			n += int64(len(phase)) * bytesPerElement
		}
		return n
	}

	ps := r.polyphaseStage
	oneBank := bankBytes(ps.polyCoeffs)
	if oneBank == 0 {
		t.Fatal("polyphase coefficient bank is empty; test cannot distinguish the undercount")
	}
	want := oneBank + bankBytes(ps.polyCoeffsB) + bankBytes(ps.polyCoeffsC) + bankBytes(ps.polyCoeffsD)
	want += int64(cap(ps.history)) * bytesPerElement
	if r.preStage != nil {
		want += bankBytes(r.preStage.polyCoeffs)
		want += int64(cap(r.preStage.history)) * bytesPerElement
	}

	if got := adapter.GetMemoryUsage(); got != want {
		t.Errorf("GetMemoryUsage() = %d, want %d (all four polyphase banks must be counted)", got, want)
	}
}
