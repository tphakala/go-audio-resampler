// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

// =============================================================================
// Streaming buffer helpers
// =============================================================================

// growStableLen returns buf resliced to exactly n, reallocating only if the
// current capacity is too small. When a reallocation is required it adds
// bufferGrowthSlack headroom so that the inevitable one-sample jitter of the
// fixed-point accumulator on the next call reuses the buffer instead of
// reallocating. After warmup the capacity settles at a stable maximum and the
// steady-state path performs no allocations. The contents of the first n
// elements are not initialized by the caller's perspective beyond what Go
// guarantees (zeroed on a fresh allocation, otherwise reused), which is fine
// because both callers overwrite every element they read.
func growStableLen[F simdops.Float](buf []F, n int) []F {
	if cap(buf) < n {
		return make([]F, n, n+bufferGrowthSlack)
	}
	return buf[:n]
}

// appendStable appends src to dst the same way the builtin append does, but
// when a reallocation is needed it grows with bufferGrowthSlack headroom so a
// subsequent one-sample jitter in the input run length does not trigger another
// reallocation. The returned slice has the same length and element values as
// append(dst, src...); only the capacity policy differs. After warmup the
// capacity settles at a stable maximum so the streaming path stops allocating.
func appendStable[F simdops.Float](dst, src []F) []F {
	need := len(dst) + len(src)
	if cap(dst) < need {
		grown := make([]F, len(dst), need+bufferGrowthSlack)
		copy(grown, dst)
		dst = grown
	}
	return append(dst, src...)
}
