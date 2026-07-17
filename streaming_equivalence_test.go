// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Chunked streaming Process calls plus one final Flush must be bit-exact
// with processing the whole signal in a single Process+Flush. This is the
// contract that makes real-time chunked use (issue #51) safe.
func TestStreamingEquivalence_Float64(t *testing.T) {
	ratios := []struct {
		name    string
		in, out float64
	}{
		{"44k1_to_48k", 44100, 48000},
		{"48k_to_44k1", 48000, 44100},
		{"48k_to_16k_integer", 48000, 16000},
		{"unity", 48000, 48000},
	}
	qualities := []QualityPreset{QualityLow, QualityMedium, QualityHigh}
	chunkPlans := [][]int{
		{1, 7, 13, 470, 4096},
		{31, 331, 997},
	}

	const n = 44100
	input := make([]float64, n)
	for i := range input {
		input[i] = 0.5 * math.Sin(2*math.Pi*997*float64(i)/44100)
	}

	for _, rr := range ratios {
		for _, q := range qualities {
			oneShot, err := NewEngine(rr.in, rr.out, q)
			if err != nil {
				t.Fatalf("%s: NewEngine: %v", rr.name, err)
			}
			ref, err := oneShot.Process(append([]float64(nil), input...))
			if err != nil {
				t.Fatalf("%s: Process: %v", rr.name, err)
			}
			refTail, err := oneShot.Flush()
			if err != nil {
				t.Fatalf("%s: Flush: %v", rr.name, err)
			}
			ref = append(ref, refTail...)

			for pi, plan := range chunkPlans {
				chunked, err := NewEngine(rr.in, rr.out, q)
				if err != nil {
					t.Fatalf("%s: NewEngine: %v", rr.name, err)
				}
				var got []float64
				rng := rand.New(rand.NewSource(int64(pi) + 1))
				pos := 0
				for pos < n {
					size := plan[rng.Intn(len(plan))]
					if pos+size > n {
						size = n - pos
					}
					out, err := chunked.Process(append([]float64(nil), input[pos:pos+size]...))
					if err != nil {
						t.Fatalf("%s plan %d: Process: %v", rr.name, pi, err)
					}
					got = append(got, out...)
					pos += size
				}
				tail, err := chunked.Flush()
				if err != nil {
					t.Fatalf("%s plan %d: Flush: %v", rr.name, pi, err)
				}
				got = append(got, tail...)

				if len(got) != len(ref) {
					t.Fatalf("%s q=%v plan %d: length %d != one-shot %d", rr.name, q, pi, len(got), len(ref))
				}
				for i := range got {
					if got[i] != ref[i] {
						t.Fatalf("%s q=%v plan %d: sample %d differs: %g != %g", rr.name, q, pi, i, got[i], ref[i])
					}
				}
			}
		}
	}
}

func TestStreamingEquivalence_Float32(t *testing.T) {
	// Broadened beyond the single issue #51 configuration: an upsample and a
	// downsample ratio, two qualities, and two fixed chunk sizes. Chunked
	// Process plus one Flush must stay bit-exact with the one-shot path.
	const n = 44100
	ratios := []struct {
		name    string
		in, out float64
	}{
		{"44k1_to_48k", 44100, 48000},
		{"48k_to_44k1", 48000, 44100},
	}
	qualities := []QualityPreset{QualityMedium, QualityHigh}
	chunkSizes := []int{470, 997}

	for _, rr := range ratios {
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(0.5 * math.Sin(2*math.Pi*997*float64(i)/rr.in))
		}
		for _, q := range qualities {
			for _, size := range chunkSizes {
				assertStreamingEquivalentFloat32(t, rr.name, rr.in, rr.out, q, size, input)
			}
		}
	}
}

// assertStreamingEquivalentFloat32 checks that feeding input in fixed-size
// chunks (with one final Flush) is bit-exact with a single one-shot
// Process+Flush for the float32 engine at the given ratio, quality, and chunk
// size.
func assertStreamingEquivalentFloat32(t *testing.T, name string, in, out float64, q QualityPreset, size int, input []float32) {
	t.Helper()

	oneShot, err := NewEngineFloat32(in, out, q)
	if err != nil {
		t.Fatalf("%s q=%v: NewEngineFloat32: %v", name, q, err)
	}
	ref, err := oneShot.Process(slices.Clone(input))
	if err != nil {
		t.Fatalf("%s q=%v: Process: %v", name, q, err)
	}
	refTail, err := oneShot.Flush()
	if err != nil {
		t.Fatalf("%s q=%v: Flush: %v", name, q, err)
	}
	ref = append(ref, refTail...)

	chunked, err := NewEngineFloat32(in, out, q)
	if err != nil {
		t.Fatalf("%s q=%v size=%d: NewEngineFloat32: %v", name, q, size, err)
	}
	var got []float32
	for pos := 0; pos < len(input); {
		step := size
		if pos+step > len(input) {
			step = len(input) - pos
		}
		outChunk, err := chunked.Process(slices.Clone(input[pos : pos+step]))
		if err != nil {
			t.Fatalf("%s q=%v size=%d: Process: %v", name, q, size, err)
		}
		got = append(got, outChunk...)
		pos += step
	}
	tail, err := chunked.Flush()
	if err != nil {
		t.Fatalf("%s q=%v size=%d: Flush: %v", name, q, size, err)
	}
	got = append(got, tail...)

	if len(got) != len(ref) {
		t.Fatalf("%s q=%v size=%d: length %d != one-shot %d", name, q, size, len(got), len(ref))
	}
	for i := range got {
		if got[i] != ref[i] {
			t.Fatalf("%s q=%v size=%d: sample %d differs: %g != %g", name, q, size, i, got[i], ref[i])
		}
	}
}
