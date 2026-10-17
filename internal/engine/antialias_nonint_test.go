// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/go-audio-resampler/internal/testutil"
)

// Non-integer downsampling must low-pass at the target Nyquist. The DFT
// pre-stage carries that filter (libsoxr cr.c:383-387); the polyphase stage
// after it only rejects images. These tests pin the alias rejection and the
// passband edge of that chain.

// aaOutSamples is the output length the tone measurements aim for; the oracle
// analyses the middle 16384 samples.
const aaOutSamples = 24000

var aaQualities = []struct {
	name string
	q    Quality
}{
	{"Low", QualityLow},
	{"Medium", QualityMedium},
	{"High", QualityHigh},
	{"VeryHigh", QualityVeryHigh},
	{"24bit", Quality24Bit},
	{"32bit", Quality32Bit},
}

// aaRun resamples a unit sine of freq Hz at inRate to outRate (one-shot
// Process plus Flush, float64) and returns the output.
func aaRun(t *testing.T, inRate, outRate float64, q Quality, freq float64) []float64 {
	t.Helper()
	r, err := NewResampler[float64](inRate, outRate, q)
	if err != nil {
		t.Fatalf("NewResampler(%g, %g): %v", inRate, outRate, err)
	}
	n := int(math.Ceil(aaOutSamples * inRate / outRate))
	y, err := r.Process(testutil.Sine(inRate, freq, n))
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	tail, err := r.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	return append(y, tail...)
}

// aaLevel is the level in dB of a tone at fr times the target Nyquist, read at
// the same frequency in the output (the passband gain).
func aaLevel(t *testing.T, inRate, outRate float64, q Quality, fr float64) float64 {
	t.Helper()
	f := fr * outRate / 2
	return testutil.ToneLevelDB(aaRun(t, inRate, outRate, q, f), outRate, f)
}

// aaWorstAlias returns the highest alias level over tones from just above the
// target Nyquist to 1.5 times it (capped at 0.985 of the input Nyquist).
func aaWorstAlias(t *testing.T, inRate, outRate float64, q Quality) float64 {
	t.Helper()
	worst := math.Inf(-1)
	for _, m := range []float64{1.001, 1.003, 1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5} {
		f := m * outRate / 2
		if f > 0.985*inRate/2 {
			break
		}
		y := aaRun(t, inRate, outRate, q, f)
		worst = math.Max(worst, testutil.ToneLevelDB(y, outRate, testutil.FoldFrequency(f, outRate)))
	}
	return worst
}

// aaEdge bisects the frequency, as a fraction of the target Nyquist, at which
// the passband gain crosses db.
func aaEdge(t *testing.T, inRate, outRate float64, q Quality, db float64) float64 {
	t.Helper()
	const bisectSteps = 14
	lo, hi := 0.5, 1.0
	for range bisectSteps {
		mid := (lo + hi) / 2
		if aaLevel(t, inRate, outRate, q, mid) > db {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo
}

// TestAntiAliasing_NonIntegerDownsampling_MeetsStopband checks that a tone
// above the target Nyquist is rejected by at least the quality's stopband
// attenuation after it folds into the output band.
func TestAntiAliasing_NonIntegerDownsampling_MeetsStopband(t *testing.T) {
	ratios := [][2]float64{
		{384000, 256000}, {48000, 32000}, {48000, 44100}, {96000, 64000},
		{384000, 288000}, {44100, 22000}, {44100, 16000}, {96000, 22050},
	}
	for _, rr := range ratios {
		for _, qq := range aaQualities {
			if testing.Short() && (qq.q != QualityHigh || rr[0] == 96000 || rr[0] == 44100) {
				continue
			}
			t.Run(fmt.Sprintf("%s/%g_to_%g", qq.name, rr[0], rr[1]), func(t *testing.T) {
				t.Parallel()
				worst := aaWorstAlias(t, rr[0], rr[1], qq.q)
				want := -qualityToAttenuation(qq.q)
				if worst > want {
					t.Errorf("alias level %.1f dB, want <= %.1f dB", worst, want)
				}
			})
		}
	}
}

// TestPassbandEdge_NonIntegerDownsampling_MatchesDecimationPath checks that
// the non-integer chain has the same passband edge as the integer decimation
// stage (it uses the same filter recipe) and is flat below it.
func TestPassbandEdge_NonIntegerDownsampling_MatchesDecimationPath(t *testing.T) {
	const (
		edgeTol     = 0.002 // fraction of target Nyquist
		flatTolDB   = 0.01
		flatLowFr   = 0.02
		flatHighFr  = 0.90
		flatStepFr  = 0.02
		intIn       = 48000
		intOut      = 24000
		minus3dB    = -3.0
		minusPt1dB  = -0.1
		minus100dB  = -100.0
		lowEdgeMin  = 0.81
		lowEdgeMax  = 0.84
		highEdgeMin = 0.945
		highEdgeMax = 0.960
	)
	ratios := [][2]float64{{384000, 256000}, {48000, 44100}}
	for _, qq := range aaQualities {
		if testing.Short() && qq.q != QualityHigh {
			continue
		}
		t.Run(qq.name, func(t *testing.T) {
			t.Parallel()
			want := map[float64]float64{}
			for _, db := range []float64{minusPt1dB, minus3dB, minus100dB} {
				want[db] = aaEdge(t, intIn, intOut, qq.q, db)
			}
			lo, hi := highEdgeMin, highEdgeMax
			if qq.q == QualityLow {
				lo, hi = lowEdgeMin, lowEdgeMax
			}
			for _, rr := range ratios {
				for _, db := range []float64{minusPt1dB, minus3dB, minus100dB} {
					got := aaEdge(t, rr[0], rr[1], qq.q, db)
					if math.Abs(got-want[db]) > edgeTol {
						t.Errorf("%g->%g %.1f dB edge %.4f, integer path %.4f (tolerance %.3f)",
							rr[0], rr[1], db, got, want[db], edgeTol)
					}
					if db == minus3dB && (got < lo || got > hi) {
						t.Errorf("%g->%g -3 dB edge %.4f outside [%.3f, %.3f]", rr[0], rr[1], got, lo, hi)
					}
				}
				if qq.q == QualityLow {
					continue
				}
				minDB, maxDB := math.Inf(1), math.Inf(-1)
				for fr := flatLowFr; fr <= flatHighFr+1e-9; fr += flatStepFr {
					l := aaLevel(t, rr[0], rr[1], qq.q, fr)
					minDB, maxDB = math.Min(minDB, l), math.Max(maxDB, l)
				}
				if maxDB-minDB > flatTolDB {
					t.Errorf("%g->%g passband ripple %.4f dB over %.2f-%.2f of target Nyquist, want <= %.2f",
						rr[0], rr[1], maxDB-minDB, flatLowFr, flatHighFr, flatTolDB)
				}
			}
		})
	}
}

// TestAntiAliasing_NonIntegerDownsampling_RatioLimit pins the ratio limit the
// README, doc.go and CHANGELOG quote: the pre-stage filter is capped at 8191
// taps, and the measured stopband holds up to 17.4:1 for High and Medium. The
// shortfall beyond the cap (34.8:1) is documented but deliberately not
// asserted, so cascading 2:1 stages can lift it without touching this test.
func TestAntiAliasing_NonIntegerDownsampling_RatioLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("large-ratio filters are long; skipped in -short mode")
	}
	for _, rr := range [][2]float64{{352800, 32000}, {384000, 22050}} {
		for _, q := range []Quality{QualityMedium, QualityHigh} {
			t.Run(fmt.Sprintf("%v/%g_to_%g", q, rr[0], rr[1]), func(t *testing.T) {
				t.Parallel()
				worst := aaWorstAlias(t, rr[0], rr[1], q)
				if want := -qualityToAttenuation(q); worst > want {
					t.Errorf("alias level %.1f dB at %.1f:1, want <= %.1f dB", worst, rr[0]/rr[1], want)
				}
			})
		}
	}
}
