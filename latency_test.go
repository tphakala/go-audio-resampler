// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"math"
	"testing"
)

// masterLatency holds Latency() (engine path) and GetLatency() (New path) as
// reported before output was time-aligned, when Latency() was twice the
// filters' group delay. Alignment halves them: a regression that restores
// the old padding shows up as a value above half the old one.
var masterLatency = []struct {
	in, out  float64
	q        QualityPreset
	eng, cfg int
}{
	{44100, 48000, QualityLow, 160, 160},
	{44100, 48000, QualityMedium, 160, 160},
	{44100, 48000, QualityHigh, 214, 271},
	{44100, 48000, QualityVeryHigh, 214, 345},
	{48000, 44100, QualityLow, 135, 135},
	{48000, 44100, QualityMedium, 135, 135},
	{48000, 44100, QualityHigh, 177, 208},
	{48000, 44100, QualityVeryHigh, 177, 270},
	{48000, 16000, QualityLow, 82, 152},
	{48000, 16000, QualityMedium, 292, 152},
	{48000, 16000, QualityHigh, 375, 470},
	{48000, 16000, QualityVeryHigh, 375, 619},
	{48000, 24000, QualityLow, 81, 81},
	{48000, 24000, QualityMedium, 292, 81},
	{48000, 24000, QualityHigh, 375, 456},
	{48000, 24000, QualityVeryHigh, 375, 611},
	{8000, 48000, QualityLow, 786, 1399},
	{8000, 48000, QualityMedium, 786, 1399},
	{8000, 48000, QualityHigh, 990, 2164},
	{8000, 48000, QualityVeryHigh, 990, 2886},
	{16000, 48000, QualityLow, 393, 613},
	{16000, 48000, QualityMedium, 393, 613},
	{16000, 48000, QualityHigh, 492, 970},
	{16000, 48000, QualityVeryHigh, 492, 1290},
	{384000, 256000, QualityLow, 98, 98},
	{384000, 256000, QualityMedium, 98, 98},
	{384000, 256000, QualityHigh, 131, 166},
	{384000, 256000, QualityVeryHigh, 131, 211},
	{22050, 48000, QualityLow, 319, 445},
	{22050, 48000, QualityMedium, 319, 445},
	{22050, 48000, QualityHigh, 414, 704},
	{22050, 48000, QualityVeryHigh, 414, 924},
	{44100, 96000, QualityLow, 319, 445},
	{44100, 96000, QualityMedium, 319, 445},
	{44100, 96000, QualityHigh, 414, 704},
	{44100, 96000, QualityVeryHigh, 414, 924},
}

// masterLatencyFor returns the pre-alignment latencies for a row, or ok false
// when the row was not recorded (Quick and unity were never affected).
func masterLatencyFor(in, out float64, q QualityPreset) (eng, cfg int, ok bool) {
	for _, m := range masterLatency {
		if m.in == in && m.out == out && m.q == q {
			return m.eng, m.cfg, true
		}
	}
	return 0, 0, false
}

// assertAtMostHalfMasterLatency fails when got exceeds half the pre-alignment
// value plus two samples of rounding slack.
func assertAtMostHalfMasterLatency(t *testing.T, label string, got, master int) {
	t.Helper()
	if got > master/2+2 {
		t.Errorf("%s: latency %d is more than half the pre-alignment value %d (+2)", label, got, master)
	}
}

// Latency() must predict the startup deficit: how many output samples the
// first Process call withholds while the filter primes. Verified against
// the measured deficit for a large first chunk.
func TestLatency_MatchesMeasuredDeficit(t *testing.T) {
	for _, c := range []struct{ in, out float64 }{
		{44100, 48000},
		{48000, 44100},
		{48000, 16000},
		{48000, 24000},
		{8000, 48000},
		{16000, 48000},
		{384000, 256000},
		{22050, 48000},
		{48000, 48000},
	} {
		for _, q := range []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh, QualityVeryHigh} {
			r, err := NewEngine(c.in, c.out, q)
			if err != nil {
				t.Fatal(err)
			}
			const n = 44100
			in := make([]float64, n)
			for i := range in {
				in[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/c.in)
			}
			out, err := r.Process(in)
			if err != nil {
				t.Fatal(err)
			}
			measured := int(float64(n)*c.out/c.in) - len(out)
			got := r.Latency()
			if got < measured-2 || got > measured+2 {
				t.Errorf("%v to %v q=%v: Latency()=%d, measured deficit %d",
					c.in, c.out, q, got, measured)
			}
			if eng, _, ok := masterLatencyFor(c.in, c.out, q); ok {
				assertAtMostHalfMasterLatency(t, fmt.Sprintf("%v to %v q=%v", c.in, c.out, q), got, eng)
			}
		}
	}
}

// Float32 mirror of one row of TestLatency_MatchesMeasuredDeficit: the float32
// engine (NewEngineFloat32) must report the same startup deficit as the
// float64 path.
func TestLatencyFloat32_MatchesMeasuredDeficit(t *testing.T) {
	const in, out = 44100.0, 48000.0
	for _, q := range []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh, QualityVeryHigh} {
		r, err := NewEngineFloat32(in, out, q)
		if err != nil {
			t.Fatal(err)
		}
		const n = 44100
		sig := make([]float32, n)
		for i := range sig {
			sig[i] = float32(0.5 * math.Sin(2*math.Pi*997*float64(i)/in))
		}
		outSamples, err := r.Process(sig)
		if err != nil {
			t.Fatal(err)
		}
		measured := int(float64(n)*out/in) - len(outSamples)
		got := r.Latency()
		if got < measured-2 || got > measured+2 {
			t.Errorf("q=%v: Latency()=%d, measured deficit %d", q, got, measured)
		}
		if eng, _, ok := masterLatencyFor(in, out, q); ok {
			assertAtMostHalfMasterLatency(t, fmt.Sprintf("float32 %v to %v q=%v", in, out, q), got, eng)
		}
	}
}

// GetLatency on the New(config) multi-stage path must report the startup
// deficit in output samples, like SimpleResampler.Latency does for the
// engine path. Guards against the domain-mixing error fixed in #52.
func TestGetLatency_ConfigPath_MatchesMeasuredDeficit(t *testing.T) {
	for _, c := range []struct{ in, out float64 }{
		{44100, 48000},
		{44100, 96000},
		{48000, 44100},
		{8000, 48000},
		{48000, 24000},
	} {
		for _, preset := range []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh, QualityVeryHigh} {
			cfg := Config{
				InputRate:  c.in,
				OutputRate: c.out,
				Channels:   1,
				Quality:    QualitySpec{Preset: preset},
			}
			r, err := New(&cfg)
			if err != nil {
				t.Fatal(err)
			}
			const n = 44100
			in := make([]float64, n)
			for i := range in {
				in[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/c.in)
			}
			out, err := r.Process(in)
			if err != nil {
				t.Fatal(err)
			}
			measured := int(float64(n)*c.out/c.in) - len(out)
			got := r.GetLatency()
			// The accumulated fractional deficit is rounded up once with
			// math.Ceil, so GetLatency lands one or two samples above the
			// measured deficit (observed max 2 across these rows). tol stays
			// well below the old domain-mixing error, which under-reported
			// the two-stage 44100->96000 case by 31 (672 versus 703).
			const tol = 4
			if got < measured-tol || got > measured+tol {
				t.Errorf("%v to %v %v: GetLatency()=%d, measured deficit %d",
					c.in, c.out, preset, got, measured)
			}
			if _, cfgLat, ok := masterLatencyFor(c.in, c.out, preset); ok {
				assertAtMostHalfMasterLatency(t, fmt.Sprintf("config %v to %v %v", c.in, c.out, preset), got, cfgLat)
			}
		}
	}
}

// The duck-typed StartupDeficit fast path covers every production stage type
// (enforced at compile time in stages.go); the fallback in GetLatency keeps
// working for stage types without deficit accounting. Pinned here with the
// test-only stubStage, including the downstream-ratio conversion.
func TestGetLatency_FallbackWithoutStartupDeficit(t *testing.T) {
	r := &constantRateResampler{
		pipeline: &Pipeline{},
		channels: []*channelResampler{{
			stages: []Stage{
				&stubStage{ratio: 2.0, filterLength: 32},
				&stubStage{ratio: 0.5, filterLength: 8},
			},
		}},
	}
	// Per-stage fallback deficits in each stage's own output domain:
	// 32/2*2.0 = 32 and 8/2*0.5 = 2. The first converts through the
	// downstream ratio 0.5 to 16, so the total is 18.
	if got := r.GetLatency(); got != 18 {
		t.Errorf("GetLatency()=%d, want 18", got)
	}
}
