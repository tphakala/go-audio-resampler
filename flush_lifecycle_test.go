// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"math"
	"slices"
	"testing"
)

func sineChunk(n int, rate float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/rate)
	}
	return out
}

// Flush ends the stream: a second Flush must return nothing, and a
// subsequent Process must behave exactly like a fresh instance instead of
// convolving new audio against leftover padding zeros.
func TestFlushLifecycle(t *testing.T) {
	for _, c := range []struct {
		in, out float64
		quality QualityPreset
	}{
		{44100, 48000, QualityHigh},
		{48000, 44100, QualityHigh},
		{48000, 16000, QualityHigh},
		// QualityQuick routes through CubicStage (reachable from NewEngine
		// since the QualityQuick mapping fix), which has its own held-tail
		// lifecycle distinct from the FIR stages' delay lines.
		{44100, 48000, QualityQuick},
	} {
		r, err := NewEngine(c.in, c.out, c.quality)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.Process(sineChunk(4410, c.in)); err != nil {
			t.Fatal(err)
		}
		if _, err := r.Flush(); err != nil {
			t.Fatal(err)
		}

		second, err := r.Flush()
		if err != nil {
			t.Fatal(err)
		}
		if len(second) != 0 {
			t.Errorf("%v to %v q=%v: second Flush returned %d samples, want 0", c.in, c.out, c.quality, len(second))
		}

		fresh, err := NewEngine(c.in, c.out, c.quality)
		if err != nil {
			t.Fatal(err)
		}
		chunk := sineChunk(4410, c.in)
		gotAfterFlush, err := r.Process(slices.Clone(chunk))
		if err != nil {
			t.Fatal(err)
		}
		gotFresh, err := fresh.Process(slices.Clone(chunk))
		if err != nil {
			t.Fatal(err)
		}
		if len(gotAfterFlush) != len(gotFresh) {
			t.Fatalf("%v to %v q=%v: post-flush Process length %d != fresh %d",
				c.in, c.out, c.quality, len(gotAfterFlush), len(gotFresh))
		}
		for i := range gotFresh {
			if gotAfterFlush[i] != gotFresh[i] {
				t.Fatalf("%v to %v q=%v: post-flush Process differs from fresh at %d", c.in, c.out, c.quality, i)
			}
		}
	}
}
