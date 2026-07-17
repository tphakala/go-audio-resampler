// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"math"
	"testing"
)

// Latency() must predict the startup deficit: how many output samples the
// first Process call withholds while the filter primes. Verified against
// the measured deficit for a large first chunk.
func TestLatency_MatchesMeasuredDeficit(t *testing.T) {
	for _, c := range []struct{ in, out float64 }{
		{44100, 48000},
		{48000, 44100},
		{48000, 16000},
		{48000, 48000},
	} {
		for _, q := range []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh} {
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
		}
	}
}

// Float32 mirror of one row of TestLatency_MatchesMeasuredDeficit: the float32
// engine (NewEngineFloat32) must report the same startup deficit as the
// float64 path.
func TestLatencyFloat32_MatchesMeasuredDeficit(t *testing.T) {
	const in, out = 44100.0, 48000.0
	for _, q := range []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh} {
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
	}
}
