// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

//go:build amd64

package simdops

import "github.com/tphakala/simd/cpu"

// minAVXElements{32,64} is the smallest tapsPerPhase at which the AVX+FMA inner dot
// is worthwhile; it matches simd's CubicInterpDot tap threshold so the fused inner
// dot selects the same tier as a per-output CubicInterpDot.
const (
	minAVXElements32 = 8
	minAVXElements64 = 4
)

// resampleCubic32 dispatches the fused f32 polyphase cubic resampler. The AVX+FMA
// gate matches simd's CubicInterpDot (cpu.X86.AVX && cpu.X86.FMA && tapsPerPhase >=
// minAVXElements32), so the block result is bit-identical to the per-output form on
// every CPU; below the threshold it uses the pure-Go path.
func resampleCubic32(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	if cpu.X86.AVX && cpu.X86.FMA && tapsPerPhase >= minAVXElements32 {
		return resampleCubic32AVX(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	}
	return resampleCubicGo32(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
}

// resampleCubic64 dispatches the fused f64 polyphase cubic resampler under the same
// gate as [resampleCubic32] with the f64 tap threshold.
func resampleCubic64(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	if cpu.X86.AVX && cpu.X86.FMA && tapsPerPhase >= minAVXElements64 {
		return resampleCubic64AVX(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	}
	return resampleCubicGo64(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
}

// resampleCubic32AVX runs the whole output block in one fused AVX+FMA pass, inlining
// the cubic dot per output. Returns the number of outputs written. See
// polyphase_resample_amd64.s.
//
//go:noescape
func resampleCubic32AVX(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) int

// resampleCubic64AVX is the float64 counterpart of [resampleCubic32AVX].
//
//go:noescape
func resampleCubic64AVX(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) int
