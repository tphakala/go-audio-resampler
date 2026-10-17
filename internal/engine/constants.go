// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

// Cubic (Hermite) interpolation constants
const (
	// Cubic interpolation uses 4-point window
	cubicInterpolationPoints = 4

	// Cubic interpolation latency (centered around middle points)
	cubicLatencySamples = 2

	// Memory usage estimate for cubic stage (bytes)
	cubicMemoryUsage = 64
)

// linearPhaseDelayDivisor halves the span of a symmetric (linear-phase) FIR:
// its group delay is (taps-1)/2 samples.
const linearPhaseDelayDivisor = 2

// groupDelaySamples returns the group delay of a linear-phase FIR with the
// given number of taps, in whole samples. An even tap count has a half-sample
// delay, which is rounded down.
func groupDelaySamples(taps int) int {
	return (taps - 1) / linearPhaseDelayDivisor
}
