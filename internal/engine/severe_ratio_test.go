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
		const chunk = 4800
		const calls = 200
		x := 0.0
		last := -1.0
		total := 0
		for call := 0; call < calls; call++ {
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
				if total > 100 && v < last-1e-6 {
					t.Fatalf("%v to %v call %d sample %d: non-monotonic %g after %g",
						c.in, c.out, call, i, v, last)
				}
				last = v
			}
		}
		ratio := c.out / c.in
		expected := float64(calls*chunk) * ratio
		if float64(total) > expected+64 || float64(total) < expected-256 {
			t.Fatalf("%v to %v: total output %d, expected about %.0f", c.in, c.out, total, expected)
		}
		if r.polyphaseStage != nil {
			maxHist := r.polyphaseStage.tapsPerPhase - 1 + chunk*4
			if len(r.polyphaseStage.history) > maxHist {
				t.Fatalf("%v to %v: history grew to %d (bound %d)", c.in, c.out,
					len(r.polyphaseStage.history), maxHist)
			}
		}
	}
}
