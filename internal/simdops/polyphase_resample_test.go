// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package simdops

import (
	"math"
	"testing"
)

const testFracBits = 16

// stepFor returns the fixed-point per-output increment for a given rate ratio,
// matching the polyphase stage: step = round((1/ratio) * numPhases * 2^testFracBits).
func stepFor(ratio float64, numPhases int) int64 {
	return int64(math.Round((1.0 / ratio) * float64(numPhases) * float64(int64(1)<<uint(testFracBits))))
}

func makeBanks[F Float](numPhases, taps int) (a, b, c, d [][]F) {
	mk := func(base F) [][]F {
		s := make([][]F, numPhases)
		for p := range s {
			s[p] = make([]F, taps)
			for t := range s[p] {
				s[p][t] = base + F(p)*F(0.0007) - F(t)*F(0.0003)
			}
		}
		return s
	}
	return mk(0.5), mk(0.13), mk(-0.05), mk(0.021)
}

// refResample is the per-output CubicInterpDot loop the fused kernel replaces:
// division-based phase extraction, one CubicInterpDot per output. The fused
// ResampleCubic must be bit-identical to this on every CPU. It runs through the
// same Ops the production code uses, so it also covers the Ops wiring.
func refResample[F Float](ops *Ops[F], out, hist []F, a, b, c, d [][]F, at, step int64, numPhases, taps int) (n int, atOut int64) {
	numPhases64 := int64(numPhases)
	fracMask := int64(1)<<uint(testFracBits) - 1
	fracScale := F(1.0 / float64(int64(1)<<uint(testFracBits)))
	histLen := len(hist)
	k := 0
	for k < len(out) {
		full := at >> uint(testFracBits)
		div := int(full / numPhases64)
		phase := int(full % numPhases64)
		frac := at & fracMask
		if div+taps > histLen {
			break
		}
		x := F(frac) * fracScale
		out[k] = ops.CubicInterpDot(hist[div:div+taps], a[phase][:taps], b[phase][:taps], c[phase][:taps], d[phase][:taps], x)
		k++
		at += step
	}
	return k, at
}

var rateRegimes = []struct {
	name    string
	in, out float64
}{
	{"44100->48000", 44100, 48000},
	{"48000->44100", 48000, 44100},
	{"44100->64000", 44100, 64000},
	{"96000->48000", 96000, 48000}, // exact 2:1
	{"16000->48000", 16000, 48000}, // exact 1:3
}

// checkParity asserts the fused ResampleCubic is bit-identical to the per-output
// reference across numPhases x taps x rate regimes, for element type F.
func checkParity[F Float](t *testing.T) {
	t.Helper()
	ops := For[F]()
	hist := make([]F, 4096)
	for i := range hist {
		hist[i] = F(math.Sin(float64(i)*0.017) + 0.3*math.Sin(float64(i)*0.11))
	}
	for _, numPhases := range []int{64, 80, 128, 256} {
		for _, taps := range []int{16, 20, 32, 64, 100} {
			a, b, c, d := makeBanks[F](numPhases, taps)
			for _, r := range rateRegimes {
				step := stepFor(r.out/r.in, numPhases)
				got := make([]F, 600)
				want := make([]F, 600)
				gN, gAt := ops.ResampleCubic(got, hist, a, b, c, d, 0, step, numPhases, taps, testFracBits)
				wN, wAt := refResample(ops, want, hist, a, b, c, d, 0, step, numPhases, taps)
				if gN != wN || gAt != wAt {
					t.Fatalf("phases=%d taps=%d %s: (n,at)=(%d,%d) want (%d,%d)", numPhases, taps, r.name, gN, gAt, wN, wAt)
				}
				for i := range gN {
					if got[i] != want[i] {
						t.Fatalf("phases=%d taps=%d %s: out[%d]=%v want %v (not bit-identical)", numPhases, taps, r.name, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestResampleCubic32_MatchesPerOutputLoop(t *testing.T) { checkParity[float32](t) }
func TestResampleCubic64_MatchesPerOutputLoop(t *testing.T) { checkParity[float64](t) }

// TestResampleCubic32_StreamingContinuity checks that splitting a block into two
// calls, carrying the returned accumulator into the second, concatenates
// bit-identically to producing the whole block in one call.
func TestResampleCubic32_StreamingContinuity(t *testing.T) {
	const (
		numPhases = 80
		taps      = 32
	)
	hist := make([]float32, 4096)
	for i := range hist {
		hist[i] = float32(math.Cos(float64(i) * 0.023))
	}
	a, b, c, d := makeBanks[float32](numPhases, taps)
	step := stepFor(48000.0/44100.0, numPhases)

	oneShot := make([]float32, 400)
	n1, _ := ResampleCubic32(oneShot, hist, a, b, c, d, 0, step, numPhases, taps, testFracBits)

	partA := make([]float32, n1/2)
	nA, atA := ResampleCubic32(partA, hist, a, b, c, d, 0, step, numPhases, taps, testFracBits)
	partB := make([]float32, n1-nA)
	nB, _ := ResampleCubic32(partB, hist, a, b, c, d, atA, step, numPhases, taps, testFracBits)

	if nA+nB != n1 {
		t.Fatalf("chunked produced %d+%d=%d, one-shot produced %d", nA, nB, nA+nB, n1)
	}
	for i := range nA {
		if partA[i] != oneShot[i] {
			t.Fatalf("chunk A out[%d]=%v want %v", i, partA[i], oneShot[i])
		}
	}
	for i := range nB {
		if partB[i] != oneShot[nA+i] {
			t.Fatalf("chunk B out[%d]=%v want %v", i, partB[i], oneShot[nA+i])
		}
	}
}

func TestResampleCubic_ValidationNoOps(t *testing.T) {
	a, b, c, d := makeBanks[float32](8, 4)
	out := make([]float32, 16)
	hist := make([]float32, 64)
	cases := []struct {
		name                      string
		numPhases, taps, fracBits int
		at, step                  int64
	}{
		{"numPhases<1", 0, 4, 16, 0, 100},
		{"taps<1", 8, 0, 16, 0, 100},
		{"step<1", 8, 4, 16, 0, 0},
		{"at<0", 8, 4, 16, -1, 100},
		{"fracBits<0", 8, 4, -1, 0, 100},
		{"fracBits>24", 8, 4, 25, 0, 100},
		{"tooFewPhases", 16, 4, 16, 0, 100}, // banks only have 8 rows
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, atOut := ResampleCubic32(out, hist, a, b, c, d, tc.at, tc.step, tc.numPhases, tc.taps, tc.fracBits)
			if n != 0 || atOut != tc.at {
				t.Fatalf("expected no-op (0, %d), got (%d, %d)", tc.at, n, atOut)
			}
		})
	}
}

func TestResampleCubic32_ZeroAlloc(t *testing.T) {
	const (
		numPhases = 80
		taps      = 32
	)
	hist := make([]float32, 4096)
	for i := range hist {
		hist[i] = float32(math.Sin(float64(i) * 0.02))
	}
	a, b, c, d := makeBanks[float32](numPhases, taps)
	step := stepFor(48000.0/44100.0, numPhases)
	out := make([]float32, 512)
	if allocs := testing.AllocsPerRun(50, func() {
		ResampleCubic32(out, hist, a, b, c, d, 0, step, numPhases, taps, testFracBits)
	}); allocs != 0 {
		t.Fatalf("ResampleCubic32 allocated %v times per run, want 0", allocs)
	}
}
