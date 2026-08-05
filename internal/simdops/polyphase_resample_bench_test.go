// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package simdops

import "testing"

// resampleWorkload is a steady-state resampling workload (numPhases 80, a
// 1024-output block, 48k->44.1k upsample) shared by the fused and per-output
// benchmarks so they measure the same work at a given tap count.
type resampleWorkload struct {
	out, hist  []float32
	a, b, c, d [][]float32
	step       int64
	numPhases  int
	taps       int
}

func benchResampleSetup(taps int) resampleWorkload {
	const numPhases = 80
	w := resampleWorkload{numPhases: numPhases, taps: taps}
	w.step = stepFor(44100.0/48000.0, numPhases)
	w.out = make([]float32, 1024)
	w.hist = make([]float32, 4096)
	for i := range w.hist {
		w.hist[i] = float32(i%97)*0.01 - 0.5
	}
	w.a, w.b, w.c, w.d = makeBanks[float32](numPhases, taps)
	return w
}

func benchResampleFused(b *testing.B, taps int) {
	b.Helper()
	w := benchResampleSetup(taps)
	b.ReportAllocs()
	for b.Loop() {
		ResampleCubic32(w.out, w.hist, w.a, w.b, w.c, w.d, 0, w.step, w.numPhases, w.taps, testFracBits)
	}
}

func benchResamplePerOutput(b *testing.B, taps int) {
	b.Helper()
	w := benchResampleSetup(taps)
	ops := For[float32]()
	b.ReportAllocs()
	for b.Loop() {
		refResample(ops, w.out, w.hist, w.a, w.b, w.c, w.d, 0, w.step, w.numPhases, w.taps)
	}
}

func BenchmarkResampleCubic32_Fused_Taps20(b *testing.B)     { benchResampleFused(b, 20) }
func BenchmarkResampleCubic32_PerOutput_Taps20(b *testing.B) { benchResamplePerOutput(b, 20) }
func BenchmarkResampleCubic32_Fused_Taps32(b *testing.B)     { benchResampleFused(b, 32) }
func BenchmarkResampleCubic32_PerOutput_Taps32(b *testing.B) { benchResamplePerOutput(b, 32) }
func BenchmarkResampleCubic32_Fused_Taps64(b *testing.B)     { benchResampleFused(b, 64) }
func BenchmarkResampleCubic32_PerOutput_Taps64(b *testing.B) { benchResamplePerOutput(b, 64) }
