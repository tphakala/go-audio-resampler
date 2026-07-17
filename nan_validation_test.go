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

// Config.Validate (and thus the pipeline constructors New/NewMultiChannel/
// NewStereo/NewSimple and the preset helpers) must reject NaN rates. NaN-blind
// comparisons (c.InputRate <= 0, ratio < min) silently pass every branch, so a
// NaN rate would otherwise build a pipeline that produces unresampled
// passthrough garbage instead of returning an error.
func TestNew_RejectsNaNRates(t *testing.T) {
	nan := math.NaN()
	for _, c := range []struct{ in, out float64 }{
		{nan, 48000},
		{48000, nan},
		{nan, nan},
	} {
		cfg := &Config{
			InputRate:  c.in,
			OutputRate: c.out,
			Channels:   1,
			Quality:    QualitySpec{Preset: QualityHigh},
		}
		if _, err := New(cfg); err == nil {
			t.Errorf("New(InputRate=%v, OutputRate=%v) accepted NaN", c.in, c.out)
		}
		if _, err := NewMultiChannel(c.in, c.out, 2, QualityHigh); err == nil {
			t.Errorf("NewMultiChannel(%v, %v) accepted NaN", c.in, c.out)
		}
	}
}
