// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

//go:build arm64

package simdops

import "github.com/tphakala/simd/cpu"

// hasNEON mirrors simd's dispatch flag so the fused inner dot selects the same tier
// as a per-output CubicInterpDot on ARM64.
var hasNEON = cpu.ARM64.NEON

// resampleCubic32 dispatches the fused f32 polyphase cubic resampler. The NEON gate
// matches simd's CubicInterpDot (hasNEON && tapsPerPhase >= 4), so the block result
// is bit-identical to the per-output form; below the threshold it uses pure Go.
func resampleCubic32(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	if hasNEON && tapsPerPhase >= 4 {
		return resampleCubic32NEON(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	}
	return resampleCubicGo32(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
}

// resampleCubic64 dispatches the fused f64 polyphase cubic resampler (NEON gate
// hasNEON && tapsPerPhase >= 2).
func resampleCubic64(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	if hasNEON && tapsPerPhase >= 2 {
		return resampleCubic64NEON(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	}
	return resampleCubicGo64(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
}

// resampleCubic32NEON runs the whole output block in one fused NEON pass, inlining
// the cubic dot per output. Returns the number of outputs written. See
// polyphase_resample_arm64.s.
//
//go:noescape
func resampleCubic32NEON(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) int

// resampleCubic64NEON is the float64 counterpart of [resampleCubic32NEON].
//
//go:noescape
func resampleCubic64NEON(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) int
