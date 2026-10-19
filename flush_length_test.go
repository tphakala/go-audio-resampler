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
// Issue #78: New dropped the final filter stage within 0.1% of 1:1 or of a
// power-of-two ratio, so the near-unity rows hold the length to the ratio on
// the New path as well as on NewEngine, at every preset.
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
		{48000, 48040, 48000}, // near unity, upsampling
		{48000, 47960, 48000}, // near unity, downsampling
		{48000, 96080, 48000}, // just past 2x
		{48000, 23990, 48000}, // just below 0.5x
	}
	paths := []struct {
		name string
		run  alignRunner
	}{
		{"NewEngine", runEngine64},
		{"New", runNewMono},
	}
	for _, p := range paths {
		for _, q := range alignPresets {
			for _, c := range cases {
				in := make([]float64, c.n)
				for i := range in {
					in[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/c.in)
				}
				out, err := p.run(in, c.in, c.out, q, 0)
				if err != nil {
					t.Fatal(err)
				}
				total := len(out)
				ideal := float64(c.n) * c.out / c.in
				lo := int(math.Floor(ideal))
				hi := int(math.Ceil(ideal)) + 1
				if total < lo || total > hi {
					t.Errorf("%s q=%v %v to %v n=%d: total %d outside [%d, %d]",
						p.name, q, c.in, c.out, c.n, total, lo, hi)
				}
			}
		}
	}
}
