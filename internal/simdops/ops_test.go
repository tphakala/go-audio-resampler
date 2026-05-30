// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package simdops

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOpsFor(t *testing.T) {
	ops32 := For[float32]()
	assert.NotNil(t, ops32)
	assert.Same(t, Float32Ops(), ops32)

	ops64 := For[float64]()
	assert.NotNil(t, ops64)
	assert.Same(t, Float64Ops(), ops64)

	testOps(t, ops32)
	testOps(t, ops64)
}

func testOps[F Float](t *testing.T, ops *Ops[F]) {
	t.Helper()
	// DotProductUnsafe
	a := []F{1, 2, 3}
	b := []F{4, 5, 6}
	dp := ops.DotProductUnsafe(a, b)
	assert.InDelta(t, float64(1*4+2*5+3*6), float64(dp), 1e-5)

	// ConvolveValid
	// signal: [1, 2, 3, 4], kernel: [1, 0.5]
	sig := []F{1, 2, 3, 4}
	kern := []F{1, 0.5}
	dst := make([]F, 3)
	ops.ConvolveValid(dst, sig, kern)
	assert.InDeltaSlice(t, []float64{2, 3.5, 5}, toFloat64Slice(dst), 1e-5)

	// ConvolveValidMulti
	dsts := [][]F{make([]F, 3), make([]F, 3)}
	kernels := [][]F{{1, 0.5}, {0, 2}}
	ops.ConvolveValidMulti(dsts, sig, kernels)
	assert.InDeltaSlice(t, []float64{2, 3.5, 5}, toFloat64Slice(dsts[0]), 1e-5)
	assert.InDeltaSlice(t, []float64{4, 6, 8}, toFloat64Slice(dsts[1]), 1e-5)

	// Interleave2
	dstInter := make([]F, 6)
	ops.Interleave2(dstInter, a, b)
	assert.InDeltaSlice(t, []float64{1, 4, 2, 5, 3, 6}, toFloat64Slice(dstInter), 1e-5)

	// Sum
	s := ops.Sum(a)
	assert.InDelta(t, float64(6), float64(s), 1e-5)

	// Scale
	dstScale := make([]F, 3)
	ops.Scale(dstScale, a, 2)
	assert.InDeltaSlice(t, []float64{2, 4, 6}, toFloat64Slice(dstScale), 1e-5)

	// CubicInterpDot
	hist := []F{0.5, 0.5}
	ca := []F{1, 2}
	cb := []F{3, 4}
	cc := []F{5, 6}
	cd := []F{7, 8}
	cid := ops.CubicInterpDot(hist, ca, cb, cc, cd, 0.5)
	assert.InDelta(t, 5.5625, float64(cid), 1e-5)
}

func toFloat64Slice[F Float](s []F) []float64 {
	r := make([]float64, len(s))
	for i, v := range s {
		r[i] = float64(v)
	}
	return r
}
