// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import "testing"

// Process must return an owned buffer at every ratio. At 1:1 the DFT stage
// passthrough used to return the caller's own slice, so mutating the input
// buffer afterwards corrupted previously returned output.
func TestProcessOutputOwned_UnityRatio(t *testing.T) {
	r, err := NewEngine(48000, 48000, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]float64, 256)
	for i := range in {
		in[i] = float64(i)
	}
	out, err := r.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(in) {
		t.Fatalf("unity ratio length %d != %d", len(out), len(in))
	}
	for i := range in {
		in[i] = -1
	}
	for i, v := range out {
		if v != float64(i) {
			t.Fatalf("output aliases input: out[%d] = %g after caller mutation", i, v)
		}
	}
}
