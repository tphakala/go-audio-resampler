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

// The flushed tail must contain the real trailing segments (centered on
// x[n-2] and x[n-1]), not garbage: its first sample should still track the
// ramp's final value, with later samples decaying toward the zero padding.
//
// This deliberately does not reuse TestCubicStage_FlushEmitsTail's unbounded
// 0..4409 ramp. That ramp is well suited to the head check (early values are
// widely spaced integers, so a fictional-zero head is obviously distinct
// from a correct one) but poorly suited to a tail check: its final value is
// thousands of units away from the zero Flush pads in, and cubic
// interpolation's polynomial fit overshoots substantially in the interior of
// a segment spanning that large a discontinuity (its endpoints stay exact;
// only the interior curve swings, by hundreds of units for that ramp's
// scale) even on the already-fixed implementation. This is the same known,
// minor characteristic documented on CubicStage.Flush, not a defect; a
// bounded ramp keeps the discontinuity small enough that tail values are
// checkable with a tight, meaningful tolerance instead of one loose enough
// to hide a real regression.
func TestCubicStage_FlushTailTracksRamp(t *testing.T) {
	c := NewCubicStage[float64](48000.0 / 44100.0)
	const n = 4410
	in := make([]float64, n)
	for i := range in {
		in[i] = float64(i) / float64(n-1) // bounded 0..1 ramp
	}
	if _, err := c.Process(in); err != nil {
		t.Fatal(err)
	}
	tail, err := c.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(tail) == 0 {
		t.Fatal("Flush returned no samples; nothing to check")
	}

	const tolerance = 0.1
	last := in[n-1]
	if math.Abs(tail[0]-last) > tolerance {
		t.Errorf("tail[0] = %v, want within %v of ramp end %v (real tail, not dropped)", tail[0], tolerance, last)
	}
	if math.Abs(tail[len(tail)-1]) > tolerance {
		t.Errorf("final tail sample = %v, want within %v of 0 (decayed toward the zero padding)", tail[len(tail)-1], tolerance)
	}
}

// Process exactly one real sample (0 < primed < cubicLatencySamples), then
// Flush. This is the partially-primed edge: the interpolator never reaches
// the priming threshold during Process, so Flush's own zero padding must
// both finish priming and drain the single real sample, without a panic and
// without breaking the terminal-flush lifecycle.
func TestCubicStage_FlushAfterPartialPriming(t *testing.T) {
	c := NewCubicStage[float64](48000.0 / 44100.0)

	out, err := c.Process([]float64{42.0})
	if err != nil {
		t.Fatal(err)
	}

	tail, err := c.Flush()
	if err != nil {
		t.Fatal(err)
	}

	total := len(out) + len(tail)
	n := 1 // runtime variable: 1*48000/44100 isn't an exact integer constant
	ideal := int(float64(n) * 48000.0 / 44100.0)
	if total < ideal-1 || total > ideal+1 {
		t.Errorf("partially primed Process+Flush total %d, want %d +-1", total, ideal)
	}

	// ideal is 1 here, so the +-1 count check above alone tolerates
	// total=0: a regression that silently drops the single pushed sample
	// would still pass it. Pin the drain directly: the tail must be
	// non-empty, and cubic interpolation evaluates exactly to the center
	// history point at x=0 (all polynomial terms multiply by x and vanish),
	// so tail[0] must equal the pushed sample.
	if len(tail) == 0 {
		t.Fatal("Flush returned no samples for a single partially-primed input; the pushed sample was dropped")
	}
	const pushedSample = 42.0
	const tolerance = 1e-9
	if math.Abs(tail[0]-pushedSample) > tolerance {
		t.Errorf("tail[0] = %v, want within %v of pushed sample %v", tail[0], tolerance, pushedSample)
	}

	second, err := c.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(second) != 0 {
		t.Errorf("second Flush returned %d samples, want 0", len(second))
	}

	fresh := NewCubicStage[float64](48000.0 / 44100.0)
	freshOut, err := fresh.Process([]float64{42.0})
	if err != nil {
		t.Fatal(err)
	}
	gotAfterFlush, err := c.Process([]float64{42.0})
	if err != nil {
		t.Fatal(err)
	}
	if len(gotAfterFlush) != len(freshOut) {
		t.Fatalf("post-flush Process length %d != fresh %d", len(gotAfterFlush), len(freshOut))
	}
	for i := range freshOut {
		if gotAfterFlush[i] != freshOut[i] {
			t.Fatalf("post-flush Process differs from fresh at %d", i)
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

// GetStatistics()["samplesOut"] must count every sample that Process and
// Flush return. On the FIR path the engine's Flush adds len(output) to
// samplesOut; the cubic (QualityQuick) branch early-returns the stage's
// flush tail and used to skip that accounting, so the statistic undercounted
// by the flush-tail length once cubic Flush began emitting a real tail. This
// pins the invariant: samplesOut == len(Process output) + len(Flush output).
func TestCubicStage_FlushUpdatesSamplesOut(t *testing.T) {
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

	// A non-empty tail is what makes the undercount observable; without it
	// the assertion below could pass even with the accounting bug present.
	if len(tail) == 0 {
		t.Fatal("Flush returned no tail; test cannot distinguish the samplesOut undercount")
	}

	returned := int64(len(out) + len(tail))
	got := r.GetStatistics()["samplesOut"]
	if got != returned {
		t.Errorf("samplesOut statistic = %d, want %d (Process %d + Flush %d): cubic Flush skips samplesOut accounting",
			got, returned, len(out), len(tail))
	}
}

// Process must return a caller-owned slice even when it emits nothing. During
// the priming window Process yields zero output, and the empty slice it returns
// must not alias the internal output buffer: a caller that appends to a
// zero-length-but-nonzero-capacity result would otherwise have its data
// silently overwritten by the next Process call reusing that buffer. This pins
// the owned-empty-slice contract that the sibling stages already satisfy by
// returning fresh []F{} literals (issue #51).
func TestCubicStage_ProcessEmptyResultIsOwned(t *testing.T) {
	c := NewCubicStage[float64](48000.0 / 44100.0)

	// One sample leaves the stage partially primed (primed=1 <
	// cubicLatencySamples=2), so Process emits nothing.
	empty, err := c.Process([]float64{1.0})
	if err != nil {
		t.Fatal(err)
	}
	if len(empty) != 0 {
		t.Fatalf("expected empty output during priming, got %d samples", len(empty))
	}

	// A caller reasonably appends its own data to the returned slice.
	const sentinel = 12345.0
	owned := append(empty, sentinel) //nolint:gocritic // deliberately appending to the returned slice to prove it is owned

	// A subsequent Process that emits output must not corrupt the caller's
	// slice. The input is small enough that Process reuses the same internal
	// buffer without reallocating, so an aliased empty result would be
	// overwritten here.
	if _, err := c.Process([]float64{2.0, 3.0, 4.0, 5.0}); err != nil {
		t.Fatal(err)
	}

	if owned[0] != sentinel {
		t.Errorf("caller's appended value was corrupted: got %v, want %v (Process returned an aliased empty slice)", owned[0], sentinel)
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
