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
