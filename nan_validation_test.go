// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"math"
	"testing"
)

func TestNewEngine_RejectsNaNRates(t *testing.T) {
	nan := math.NaN()
	for _, c := range []struct{ in, out float64 }{
		{nan, 48000},
		{48000, nan},
		{nan, nan},
	} {
		if _, err := NewEngine(c.in, c.out, QualityHigh); err == nil {
			t.Errorf("NewEngine(%v, %v) accepted NaN", c.in, c.out)
		}
		if _, err := NewEngineFloat32(c.in, c.out, QualityHigh); err == nil {
			t.Errorf("NewEngineFloat32(%v, %v) accepted NaN", c.in, c.out)
		}
	}
}
