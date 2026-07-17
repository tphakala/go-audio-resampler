// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"slices"
	"testing"
)

// The cubic interpolator holds a 4-point window with 2 samples of latency;
// Flush must emit the tail those samples cover instead of dropping it.
//
// Total length alone is not a reliable regression signal for this stage:
// CubicStage.Process (unlike the FIR stages) never withholds output pending
// future context, so it emits immediately from a fictional zero-filled
// history. That keeps Process-alone output count close to n*ratio even when
// the true final samples are dropped, because emitting a few samples from
// fictional pre-silence at the head happens to offset the count. So this
// test also checks head content: the first few output samples must track
// the ramp continuously instead of dipping toward zero from that fictional
// history.
func TestCubicStage_FlushEmitsTail(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityQuick)
	if err != nil {
		t.Fatal(err)
	}
	const n = 4410
	in := make([]float64, n)
	for i := range in {
		in[i] = float64(i)
	}
	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := r.Flush()
	if err != nil {
		t.Fatal(err)
	}
	total := len(out) + len(tail)
	ideal := int(float64(n) * 48000.0 / 44100.0)
	if total < ideal-1 || total > ideal+1 {
		t.Errorf("cubic Process+Flush total %d, want %d +-1", total, ideal)
	}

	if len(out) < 4 {
		t.Fatalf("cubic Process produced only %d samples, too few to check head content", len(out))
	}
	const step = 44100.0 / 48000.0
	for i := range 4 {
		want := float64(i) * step
		got := out[i]
		if math.Abs(got-want) > 0.5 {
			t.Errorf("head sample %d = %v, want within 0.5 of %v (ramp continuity, not fictional zero history)", i, got, want)
		}
	}
}

// Flush must drain the true tail, not just leave the stage silently short:
// after real input has primed the interpolator, Flush must return samples.
func TestCubicStage_FlushIsNonEmptyAfterRealInput(t *testing.T) {
	c := NewCubicStage[float64](48000.0 / 44100.0)
	in := make([]float64, 100)
	for i := range in {
		in[i] = float64(i)
	}
	if _, err := c.Process(in); err != nil {
		t.Fatal(err)
	}
	tail, err := c.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(tail) == 0 {
		t.Fatal("Flush returned no samples after real input was processed; held tail was dropped")
	}
}

// Cubic chunked-vs-one-shot equivalence at engine level: the root pin test
// (streaming_equivalence_test.go) covers QualityLow/Medium/High only, not
// QualityQuick's cubic path. Feeding the same signal in small chunks versus
// one shot, with a single Flush at the end, must be bit-exact.
func TestCubicStage_ChunkedEquivalence(t *testing.T) {
	const n = 44100
	in := make([]float64, n)
	for i := range in {
		in[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/44100)
	}

	oneShot, err := NewResampler[float64](44100, 48000, QualityQuick)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := oneShot.Process(slices.Clone(in))
	if err != nil {
		t.Fatal(err)
	}
	refTail, err := oneShot.Flush()
	if err != nil {
		t.Fatal(err)
	}
	ref = append(ref, refTail...)

	plans := [][]int{
		{1, 7, 13, 470, 4096},
		{31, 331, 997},
	}
	for pi, plan := range plans {
		chunked, err := NewResampler[float64](44100, 48000, QualityQuick)
		if err != nil {
			t.Fatal(err)
		}
		var got []float64
		pos := 0
		planIdx := 0
		for pos < n {
			size := plan[planIdx%len(plan)]
			planIdx++
			if pos+size > n {
				size = n - pos
			}
			out, err := chunked.Process(slices.Clone(in[pos : pos+size]))
			if err != nil {
				t.Fatalf("plan %d: Process: %v", pi, err)
			}
			got = append(got, out...)
			pos += size
		}
		tail, err := chunked.Flush()
		if err != nil {
			t.Fatalf("plan %d: Flush: %v", pi, err)
		}
		got = append(got, tail...)

		if len(got) != len(ref) {
			t.Fatalf("plan %d: length %d != one-shot %d", pi, len(got), len(ref))
		}
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("plan %d: sample %d differs: %g != %g", pi, i, got[i], ref[i])
			}
		}
	}
}
