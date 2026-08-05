// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package simdops

import (
	"math"

	"github.com/tphakala/simd/f32"
	"github.com/tphakala/simd/f64"
)

// Fused polyphase cubic resampler.
//
// ResampleCubic32 and ResampleCubic64 run a whole block of soxr-style polyphase
// FIR resampling with cubic sub-phase coefficient interpolation in one call. This
// is the block form of the per-output CubicInterpDot loop the polyphase stage
// would otherwise run: for each output it derives the input window and the
// interpolation phase from a fixed-point accumulator, evaluates
//
//	out[k] = sum hist[div+i] * (a[phase][i] + x*(b[phase][i] + x*(c[phase][i] + x*d[phase][i])))
//
// over tapsPerPhase taps, then advances the accumulator by step. It returns n, the
// number of outputs written to out[:n], and atOut, the accumulator after them
// (atOut == at + n*step exactly), so a streaming caller carries atOut into the next
// call as the new at.
//
// The accumulator is fixed-point: at = (inputIndex*numPhases + phase) << fracBits +
// frac, and step is the per-output increment in the same units. x = float(frac) *
// 2^-fracBits is the sub-phase position in [0, 1). numPhases is the number of
// polyphase filters. Output k is produced only while k < len(out) and
// div+tapsPerPhase <= len(hist); the first output not satisfying the window bound
// ends the block. out must not overlap hist or any coefficient bank.
//
// This kernel lives in this repository, not in the simd library: the phase-stepping
// state machine (fixed-point accumulator, per-phase coefficient banks, sliding
// window bound, streaming rebase) is specific to this resampler. It is layered on
// simd's generic CubicInterpDot primitive. On amd64 (AVX+FMA) and arm64 (NEON) the
// inner cubic dot is inlined per output under the same CPU-feature gate as
// CubicInterpDot, so the block result is bit-identical to a per-output CubicInterpDot
// on every CPU, tier for tier; below the vector threshold and on other architectures
// it is a pure-Go loop calling CubicInterpDotUnsafe. It allocates nothing.

// polyphaseMaxFracBits{32,64} is the largest sub-phase fractional width for which
// float(frac) is exact for every frac in [0, 1<<fracBits): the significand width of
// the type. Above it, float(frac) would round and the sub-phase position would stop
// matching a per-output CubicInterpDot bit-for-bit, so ResampleCubic rejects it.
const (
	polyphaseMaxFracBits32 = 24
	polyphaseMaxFracBits64 = 53
)

// ResampleCubic32 is the float32 fused polyphase cubic resampler. It validates its
// inputs and is a no-op returning (0, at) (never a panic) if any of these do not
// hold: numPhases >= 1, tapsPerPhase >= 1, step >= 1, at >= 0, 0 <= fracBits <= 24,
// each of a, b, c, d has at least numPhases rows, each of the first numPhases rows
// has length >= tapsPerPhase, and at + len(out)*step does not overflow int64.
//nolint:dupl // f32/f64 validation bodies are intentionally parallel; only the element type and the fracBits bound differ.
func ResampleCubic32(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) (n int, atOut int64) {
	if numPhases < 1 || tapsPerPhase < 1 || step < 1 || at < 0 ||
		fracBits < 0 || fracBits > polyphaseMaxFracBits32 {
		return 0, at
	}
	if len(a) < numPhases || len(b) < numPhases || len(c) < numPhases || len(d) < numPhases {
		return 0, at
	}
	for p := range numPhases {
		if len(a[p]) < tapsPerPhase || len(b[p]) < tapsPerPhase ||
			len(c[p]) < tapsPerPhase || len(d[p]) < tapsPerPhase {
			return 0, at
		}
	}
	// Reject inputs whose block accumulator could overflow int64: the internal div
	// would wrap negative and defeat the window guard. Real resamplers keep step and
	// at far below this, so no legitimate input is rejected.
	if outLen := int64(len(out)); outLen > 0 && step > (math.MaxInt64-at)/outLen {
		return 0, at
	}
	n = resampleCubic32(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	return n, at + int64(n)*step
}

// ResampleCubic64 is the float64 fused polyphase cubic resampler. It applies the
// same validation as [ResampleCubic32] with fracBits bounded by 53.
//nolint:dupl // f32/f64 validation bodies are intentionally parallel; only the element type and the fracBits bound differ.
func ResampleCubic64(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) (n int, atOut int64) {
	if numPhases < 1 || tapsPerPhase < 1 || step < 1 || at < 0 ||
		fracBits < 0 || fracBits > polyphaseMaxFracBits64 {
		return 0, at
	}
	if len(a) < numPhases || len(b) < numPhases || len(c) < numPhases || len(d) < numPhases {
		return 0, at
	}
	for p := range numPhases {
		if len(a[p]) < tapsPerPhase || len(b[p]) < tapsPerPhase ||
			len(c[p]) < tapsPerPhase || len(d[p]) < tapsPerPhase {
			return 0, at
		}
	}
	if outLen := int64(len(out)); outLen > 0 && step > (math.MaxInt64-at)/outLen {
		return 0, at
	}
	n = resampleCubic64(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	return n, at + int64(n)*step
}

// resampleCubicGo32 is the pure-Go reference and fallback for [ResampleCubic32]. It
// runs the incremental phase-stepping state machine and evaluates every output with
// simd's CubicInterpDotUnsafe, so its result is bit-identical to a per-output loop
// that calls the same primitive at any tap count. The one-time divisions seed the
// incremental deltas; per output only adds and at most two conditional subtracts
// run. Callers guarantee the inputs are valid; validation is done by ResampleCubic32.
func resampleCubicGo32(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	numPhases64 := int64(numPhases)
	fracMask := int64(1)<<uint(fracBits) - 1
	fracScale := float32(1.0 / float64(int64(1)<<uint(fracBits)))

	full := at >> uint(fracBits)
	div := int(full / numPhases64)
	phase := int(full - int64(div)*numPhases64)
	frac := at & fracMask

	sFull := step >> uint(fracBits)
	sDiv := int(sFull / numPhases64)
	sPhase := int(sFull - int64(sDiv)*numPhases64)
	sFrac := step & fracMask

	histLen := len(hist)
	outLen := len(out)
	k := 0
	for k < outLen {
		if div+tapsPerPhase > histLen {
			break
		}
		x := float32(frac) * fracScale
		out[k] = f32.CubicInterpDotUnsafe(
			hist[div:div+tapsPerPhase],
			a[phase][:tapsPerPhase], b[phase][:tapsPerPhase],
			c[phase][:tapsPerPhase], d[phase][:tapsPerPhase], x)
		k++

		frac += sFrac
		if frac > fracMask {
			frac -= fracMask + 1
			phase++
		}
		phase += sPhase
		div += sDiv
		if phase >= numPhases {
			phase -= numPhases
			div++
		}
	}
	return k
}

// resampleCubicGo64 is the pure-Go reference and fallback for [ResampleCubic64].
func resampleCubicGo64(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) int {
	numPhases64 := int64(numPhases)
	fracMask := int64(1)<<uint(fracBits) - 1
	fracScale := 1.0 / float64(int64(1)<<uint(fracBits))

	full := at >> uint(fracBits)
	div := int(full / numPhases64)
	phase := int(full - int64(div)*numPhases64)
	frac := at & fracMask

	sFull := step >> uint(fracBits)
	sDiv := int(sFull / numPhases64)
	sPhase := int(sFull - int64(sDiv)*numPhases64)
	sFrac := step & fracMask

	histLen := len(hist)
	outLen := len(out)
	k := 0
	for k < outLen {
		if div+tapsPerPhase > histLen {
			break
		}
		x := float64(frac) * fracScale
		out[k] = f64.CubicInterpDotUnsafe(
			hist[div:div+tapsPerPhase],
			a[phase][:tapsPerPhase], b[phase][:tapsPerPhase],
			c[phase][:tapsPerPhase], d[phase][:tapsPerPhase], x)
		k++

		frac += sFrac
		if frac > fracMask {
			frac -= fracMask + 1
			phase++
		}
		phase += sPhase
		div += sDiv
		if phase >= numPhases {
			phase -= numPhases
			div++
		}
	}
	return k
}
