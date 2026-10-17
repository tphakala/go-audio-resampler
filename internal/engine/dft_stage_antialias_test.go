// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"testing"
)

// TestNewDFTAntiAliasStage_RejectsBadArguments checks both guards: the factor
// must be at least 2, and fn (the pre-stage output Nyquist over the target
// Nyquist) must exceed the factor, NaN included.
func TestNewDFTAntiAliasStage_RejectsBadArguments(t *testing.T) {
	cases := []struct {
		name   string
		factor int
		fn     float64
	}{
		{"factor zero", 0, 3},
		{"factor one", 1, 3},
		{"negative factor", -2, 3},
		{"fn equals factor", 2, 2},
		{"fn below factor", 2, 1.5},
		{"fn zero", 2, 0},
		{"fn NaN", 2, math.NaN()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if st, err := NewDFTAntiAliasStage[float64](tc.factor, tc.fn, QualityHigh); err == nil {
				t.Fatalf("NewDFTAntiAliasStage(%d, %v) = %v, want an error", tc.factor, tc.fn, st)
			}
		})
	}
	if _, err := NewDFTAntiAliasStage[float64](2, 2.5, QualityHigh); err != nil {
		t.Fatalf("NewDFTAntiAliasStage(2, 2.5): %v", err)
	}
}
