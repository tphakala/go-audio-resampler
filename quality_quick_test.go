// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import "testing"

// QualityQuick means cubic interpolation (resample.go documents this and
// New() honors it); the NewEngine path must agree instead of silently
// substituting a full FIR pipeline with different latency and flush
// behavior.
func TestNewEngine_QualityQuick_IsCubic(t *testing.T) {
	r, err := NewEngine(44100, 48000, QualityQuick)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.Latency(); got > 8 {
		t.Errorf("QualityQuick latency %d, want cubic-scale (<= 8): engine mapped to FIR pipeline", got)
	}
}
