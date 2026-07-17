// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"testing"
)

// Severe non-integer downsampling: the fixed-point accumulator can overshoot
// the per-call limit by up to one step. consumed must be capped at the number
// of fully available input positions or the history trim is silently skipped
// while the accumulator is rebased, corrupting output and leaking memory.
func TestPolyphase_SevereDownsampling_MonotonicAndBounded(t *testing.T) {
	cases := []struct{ in, out float64 }{
		{48000, 3001},
		{48000, 1000.5},
		{48000, 200.5},
	}
	for _, c := range cases {
		r, err := NewResampler[float64](c.in, c.out, QualityHigh)
		if err != nil {
			t.Fatalf("%v to %v: %v", c.in, c.out, err)
		}
		const (
			chunk = 4800
			calls = 200
			// transientSkip ignores the monotonic check over the initial
			// filter transient, where priming legitimately dips before the
			// delay line fills.
			transientSkip = 100
			// monotonicEps tolerates float rounding when comparing adjacent
			// ramp outputs for non-decreasing order.
			monotonicEps = 1e-6
			// countSlackHigh/Low bound how far the total output may exceed or
			// fall short of the ideal n*ratio: the high side absorbs
			// boundary-carry rounding across many chunks, the low side absorbs
			// the startup latency withheld at these severe ratios.
			countSlackHigh = 64
			countSlackLow  = 256
			// historyChunkSlack bounds the retained delay line: steady-state
			// history never exceeds tapsPerPhase-1 plus a few chunks of slack.
			historyChunkSlack = chunk * 4
		)
		x := 0.0
		last := -1.0
		total := 0
		for call := range calls {
			in := make([]float64, chunk)
			for i := range in {
				in[i] = x
				x += 1.0
			}
			out, err := r.Process(in)
			if err != nil {
				t.Fatalf("%v to %v call %d: %v", c.in, c.out, call, err)
			}
			total += len(out)
			for i, v := range out {
				// Ramp input must produce non-decreasing output away from
				// the initial filter transient.
				if total > transientSkip && v < last-monotonicEps {
					t.Fatalf("%v to %v call %d sample %d: non-monotonic %g after %g",
						c.in, c.out, call, i, v, last)
				}
				last = v
			}
		}
		ratio := c.out / c.in
		expected := float64(calls*chunk) * ratio
		if float64(total) > expected+countSlackHigh || float64(total) < expected-countSlackLow {
			t.Fatalf("%v to %v: total output %d, expected about %.0f", c.in, c.out, total, expected)
		}
		if r.polyphaseStage != nil {
			maxHist := r.polyphaseStage.tapsPerPhase - 1 + historyChunkSlack
			if len(r.polyphaseStage.history) > maxHist {
				t.Fatalf("%v to %v: history grew to %d (bound %d)", c.in, c.out,
					len(r.polyphaseStage.history), maxHist)
			}
		}
	}
}
