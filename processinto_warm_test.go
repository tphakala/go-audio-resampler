// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestProcessIntoFloat32_WarmNoReset_ZeroAllocs checks the steady-state
// ProcessInto path of the downsampling chain with its long anti-alias
// pre-stage, without Reset inside the measured loop (the Reset-based tests in
// processinto_test.go only cover the cold start path).
func TestProcessIntoFloat32_WarmNoReset_ZeroAllocs(t *testing.T) {
	// The pre-stage filter is at most 8191 taps, so its startup transient
	// spans at most half of that in input samples.
	const warmInputSamples = 2 * 8192
	for _, rr := range [][2]float64{{384000, 256000}, {48000, 32000}} {
		for _, chunk := range []int{256, 65536} {
			t.Run(fmt.Sprintf("%g_to_%g/chunk_%d", rr[0], rr[1], chunk), func(t *testing.T) {
				r, err := NewEngineFloat32(rr[0], rr[1], QualityHigh)
				require.NoError(t, err)
				input := make([]float32, chunk)
				for i := range input {
					input[i] = float32(i%97) * 1e-3
				}
				output := make([]float32, r.EstimateOutput(chunk))
				for range warmInputSamples/chunk + 2 {
					_, err = r.ProcessInto(input, output)
					require.NoError(t, err)
				}
				var runErr error
				allocs := testing.AllocsPerRun(50, func() {
					_, runErr = r.ProcessInto(input, output)
				})
				require.NoError(t, runErr)
				if allocs != 0 {
					t.Fatalf("warm ProcessInto allocated %.0f times per call; expected 0", allocs)
				}
			})
		}
	}
}
