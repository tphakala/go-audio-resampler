// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/tphakala/go-audio-resampler/internal/simdops"
)

// TestResamplerProcessEqualsZeroCopy pins the invariant that Process returns
// exactly what ProcessZeroCopy computes. Process is the copying convenience
// wrapper over the zero-copy pipeline, so for identical inputs and identical
// resampler state the two must be bit-for-bit equal across every stage
// configuration: rational upsampling and downsampling (pre-stage + polyphase),
// integer decimation (pre-stage-less DFT decimation), and the cubic-only quick
// path. This is the invariant that lets Process skip the intermediate
// owned-memory copy that the previous per-stage Process chaining performed.
func TestResamplerProcessEqualsZeroCopy(t *testing.T) {
	cases := []struct {
		name    string
		in, out float64
		quality Quality
	}{
		{"up_44100_48000_high", 44100, 48000, QualityHigh},   // pre-stage + polyphase
		{"down_48000_44100_high", 48000, 44100, QualityHigh}, // pre-stage + polyphase
		{"down_96000_48000_high", 96000, 48000, QualityHigh}, // integer decimation (2x)
		{"down_48000_16000_high", 48000, 16000, QualityHigh}, // integer decimation (3x)
		{"up_44100_48000_quick", 44100, 48000, QualityQuick}, // cubic only
	}
	for _, tc := range cases {
		t.Run(tc.name+"/f64", func(t *testing.T) {
			assertProcessEqualsZeroCopy[float64](t, tc.in, tc.out, tc.quality)
		})
		t.Run(tc.name+"/f32", func(t *testing.T) {
			assertProcessEqualsZeroCopy[float32](t, tc.in, tc.out, tc.quality)
		})
	}
}

func assertProcessEqualsZeroCopy[F simdops.Float](t *testing.T, inRate, outRate float64, q Quality) {
	t.Helper()

	rp, err := NewResampler[F](inRate, outRate, q)
	require.NoError(t, err)
	rz, err := NewResampler[F](inRate, outRate, q)
	require.NoError(t, err)

	// Several blocks so the multi-stage buffering and the intermediate hand-off
	// between stages are exercised across successive calls, not just once.
	const blocks = 4
	const blockLen = 1500
	for b := range blocks {
		in := make([]F, blockLen)
		for i := range in {
			in[i] = F(math.Sin(float64(b*blockLen+i) * 0.05))
		}

		got, err := rp.Process(in)
		require.NoError(t, err)

		// ProcessZeroCopy aliases internal buffers valid only until the next
		// call; snapshot before comparing.
		zc, err := rz.ProcessZeroCopy(in)
		require.NoError(t, err)
		want := make([]F, len(zc))
		copy(want, zc)

		require.Lenf(t, got, len(want), "block %d: length mismatch", b)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("block %d sample %d: Process=%v ProcessZeroCopy=%v", b, i, got[i], want[i])
			}
		}
	}
}
