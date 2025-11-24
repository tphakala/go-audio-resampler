// Package simdops provides generic SIMD operations for float32 and float64 types.
// This enables a single codebase to support both precision levels without duplication.
//
// With Profile-Guided Optimization (Go 1.22+), function pointer calls in hot paths
// can be devirtualized and inlined, achieving near-zero overhead.
package simdops

import (
	"github.com/tphakala/simd/f32"
	"github.com/tphakala/simd/f64"
)

// Float is the type constraint for supported floating-point types.
type Float interface {
	float32 | float64
}

// Ops provides SIMD-accelerated operations for type F.
// Function pointers allow type-safe generic code while delegating
// to optimized type-specific implementations.
//
// With PGO, these indirect calls can be devirtualized in hot paths.
type Ops[F Float] struct {
	// DotProductUnsafe computes the dot product without bounds checking.
	// Use only when slices are guaranteed to have equal length.
	DotProductUnsafe func(a, b []F) F

	// ConvolveValid computes valid convolution of signal with kernel.
	ConvolveValid func(dst, signal, kernel []F)

	// ConvolveValidMulti computes valid convolution for multiple kernels.
	ConvolveValidMulti func(dsts [][]F, signal []F, kernels [][]F)

	// Interleave2 interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], ...
	Interleave2 func(dst, a, b []F)

	// Sum returns the sum of all elements.
	Sum func(a []F) F

	// Scale multiplies each element by scalar s: dst[i] = a[i] * s
	Scale func(dst, a []F, s F)

	// CubicInterpDot computes the fused cubic interpolation dot product:
	//   Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
	// Used for polyphase resampling with cubic coefficient interpolation.
	CubicInterpDot func(hist, a, b, c, d []F, x F) F
}

// Pre-instantiated operations for each float type.
// These are package-level variables to avoid repeated allocation.
var (
	ops32 = Ops[float32]{
		DotProductUnsafe:   f32.DotProductUnsafe,
		ConvolveValid:      f32.ConvolveValid,
		ConvolveValidMulti: f32.ConvolveValidMulti,
		Interleave2:        f32.Interleave2,
		Sum:                f32.Sum,
		Scale:              f32.Scale,
		CubicInterpDot:     f32.CubicInterpDot,
	}
	ops64 = Ops[float64]{
		DotProductUnsafe:   f64.DotProductUnsafe,
		ConvolveValid:      f64.ConvolveValid,
		ConvolveValidMulti: f64.ConvolveValidMulti,
		Interleave2:        f64.Interleave2,
		Sum:                f64.Sum,
		Scale:              f64.Scale,
		CubicInterpDot:     f64.CubicInterpDot,
	}
)

// For returns the Ops instance for type F.
// The type switch happens at instantiation time, not in hot paths.
func For[F Float]() *Ops[F] {
	var zero F
	switch any(zero).(type) {
	case float32:
		ops, ok := any(&ops32).(*Ops[F])
		if !ok {
			panic("simdops: type assertion failed for float32")
		}
		return ops
	case float64:
		ops, ok := any(&ops64).(*Ops[F])
		if !ok {
			panic("simdops: type assertion failed for float64")
		}
		return ops
	default:
		panic("simdops: unsupported float type")
	}
}

// Type aliases for common configurations (Go 1.24 feature).
type (
	Ops32 = Ops[float32]
	Ops64 = Ops[float64]
)

// Float32Ops returns the float32 SIMD operations.
// Convenience function for non-generic code.
func Float32Ops() *Ops[float32] {
	return &ops32
}

// Float64Ops returns the float64 SIMD operations.
// Convenience function for non-generic code.
func Float64Ops() *Ops[float64] {
	return &ops64
}
