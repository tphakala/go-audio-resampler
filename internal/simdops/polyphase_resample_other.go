// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

//go:build !amd64 && !arm64

package simdops

// resampleCubic32 uses the pure-Go path on architectures without a hand-written
// kernel; it is bit-identical to the amd64/arm64 paths at the Go tier.
func resampleCubic32(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	return resampleCubicGo32(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
}

// resampleCubic64 uses the pure-Go path on architectures without a hand-written kernel.
func resampleCubic64(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	return resampleCubicGo64(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
}
