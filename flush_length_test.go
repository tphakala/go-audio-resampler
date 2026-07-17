// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"math"
	"testing"
)

// Process+Flush of a fresh instance must emit exactly the resampled length:
// no phantom padding samples. Issue #51: 470 samples at 44100 to 48000
// QualityHigh returned 514 where ceil(470*ratio)+1 = 513 is the maximum.
func TestFlushLength_Canonical(t *testing.T) {
	cases := []struct {
		in, out float64
		n       int
	}{
		{44100, 48000, 470},
		{44100, 48000, 4410},
		{44100, 48000, 44100},
		{48000, 44100, 480},
		{48000, 44100, 48000},
		{48000, 16000, 4800},
	}
	for _, q := range []QualityPreset{QualityMedium, QualityHigh} {
		for _, c := range cases {
			r, err := NewEngine(c.in, c.out, q)
			if err != nil {
				t.Fatal(err)
			}
			in := make([]float64, c.n)
			for i := range in {
				in[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/c.in)
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
			ideal := float64(c.n) * c.out / c.in
			lo := int(math.Floor(ideal))
			hi := int(math.Ceil(ideal)) + 1
			if total < lo || total > hi {
				t.Errorf("q=%v %v to %v n=%d: total %d outside [%d, %d]",
					q, c.in, c.out, c.n, total, lo, hi)
			}
		}
	}
}
