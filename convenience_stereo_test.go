// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"math"
	"testing"
)

// assertSamplesEqual fails the test unless got and want are bit-identical.
func assertSamplesEqual[T float32 | float64](t *testing.T, label string, got, want []T) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s sample %d differs: got %v, want %v", label, i, got[i], want[i])
		}
	}
}

var stereoRatePairs = []struct{ in, out float64 }{
	{44100, 48000},
	{48000, 16000},
	{48000, 32000},
	{96000, 48000},
	{16000, 48000},
}

var stereoQualities = []QualityPreset{QualityLow, QualityMedium, QualityHigh}

// TestResampleStereo_MatchesMono pins the invariant that one-shot stereo
// resampling produces output bit-identical to resampling each channel
// independently with ResampleMono. ResampleStereo reuses a single engine across
// both channels (resetting between them) so the polyphase filter bank is only
// designed once; this guards that the reuse does not perturb either channel.
func TestResampleStereo_MatchesMono(t *testing.T) {
	const numSamples = 4096
	left := make([]float64, numSamples)
	right := make([]float64, numSamples)
	for i := range left {
		left[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 44100)
		right[i] = 0.5 * math.Sin(2*math.Pi*1760*float64(i)/44100)
	}

	for _, rp := range stereoRatePairs {
		for _, q := range stereoQualities {
			t.Run(fmt.Sprintf("%gto%g_q%d", rp.in, rp.out, q), func(t *testing.T) {
				wantLeft, err := ResampleMono(left, rp.in, rp.out, q)
				if err != nil {
					t.Fatalf("ResampleMono(left): %v", err)
				}
				wantRight, err := ResampleMono(right, rp.in, rp.out, q)
				if err != nil {
					t.Fatalf("ResampleMono(right): %v", err)
				}

				gotLeft, gotRight, err := ResampleStereo(left, right, rp.in, rp.out, q)
				if err != nil {
					t.Fatalf("ResampleStereo: %v", err)
				}

				assertSamplesEqual(t, "left", gotLeft, wantLeft)
				assertSamplesEqual(t, "right", gotRight, wantRight)
			})
		}
	}
}

// TestResampleStereoFloat32_MatchesMono is the float32 mirror of
// TestResampleStereo_MatchesMono.
func TestResampleStereoFloat32_MatchesMono(t *testing.T) {
	const numSamples = 4096
	left := make([]float32, numSamples)
	right := make([]float32, numSamples)
	for i := range left {
		left[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 44100))
		right[i] = float32(0.5 * math.Sin(2*math.Pi*1760*float64(i)/44100))
	}

	for _, rp := range stereoRatePairs {
		for _, q := range stereoQualities {
			t.Run(fmt.Sprintf("%gto%g_q%d", rp.in, rp.out, q), func(t *testing.T) {
				wantLeft, err := ResampleMonoFloat32(left, rp.in, rp.out, q)
				if err != nil {
					t.Fatalf("ResampleMonoFloat32(left): %v", err)
				}
				wantRight, err := ResampleMonoFloat32(right, rp.in, rp.out, q)
				if err != nil {
					t.Fatalf("ResampleMonoFloat32(right): %v", err)
				}

				gotLeft, gotRight, err := ResampleStereoFloat32(left, right, rp.in, rp.out, q)
				if err != nil {
					t.Fatalf("ResampleStereoFloat32: %v", err)
				}

				assertSamplesEqual(t, "left", gotLeft, wantLeft)
				assertSamplesEqual(t, "right", gotRight, wantRight)
			})
		}
	}
}
