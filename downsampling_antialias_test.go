// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/go-audio-resampler/internal/testutil"
)

const (
	// antiAliasOutSamples is the output length the tone measurements aim for.
	antiAliasOutSamples = 24000
	// antiAliasChunk is the chunk size of the BirdNET-Go analysis path.
	antiAliasChunk = 65536
	// antiAliasPublicChunk is the ProcessInto chunk size of the public-path
	// test. The tone inputs there hold 26122 to 36000 samples, so
	// antiAliasChunk would never split them; this size crosses several chunk
	// boundaries through the long pre-stage.
	antiAliasPublicChunk = 8192
	// antiAliasFloat32CapDB caps float32 thresholds: float32 floors near
	// -141 dB (engine VeryHigh float32 measures -140.8 dB).
	antiAliasFloat32CapDB = 135.0
)

// antiAliasStopbandDB is the stopband, in dB, each preset reaches on the
// engine path (NewEngine*, ResampleMono*) and on the New(&Config) path today.
// The two differ because the presets map to different engine qualities
// (convenience.go sends High and VeryHigh to engine.QualityHigh; New goes
// through precisionToEngineQuality in stages.go, where Medium reaches
// engine.QualityLow). Issue #76 tracks that mapping; when it changes, update
// this table with it. Values are the engine qualities' attenuations rounded
// down: Low and Medium 102.4, High 126.4, Quality24Bit 150.5, Quality32Bit
// 198.7.
var antiAliasStopbandDB = map[string]map[QualityPreset]float64{
	"engine": {QualityLow: 102, QualityMedium: 102, QualityHigh: 126, QualityVeryHigh: 126},
	"new":    {QualityLow: 102, QualityMedium: 102, QualityHigh: 150, QualityVeryHigh: 198},
}

func resampleViaMono(in []float64, inRate, outRate float64, q QualityPreset, _ int) ([]float64, error) {
	return ResampleMono(in, inRate, outRate, q)
}

// TestNonIntegerDownsampling_PublicPaths_MeetStopband checks that a tone above
// the target Nyquist is rejected on every public path, not only inside the
// engine: NewEngine, NewEngineFloat32 with ProcessInto in 8192-sample chunks
// (the BirdNET-Go analysis path uses 65536-sample chunks, see
// TestSoxSpectrogramParity_NonIntegerDownsampling), New(&Config) and
// ResampleMono.
func TestNonIntegerDownsampling_PublicPaths_MeetStopband(t *testing.T) {
	type pathCase struct {
		name    string
		table   string
		float32 bool
		run     alignRunner
	}
	paths := []pathCase{
		{"NewEngine", "engine", false, runEngine64},
		{"NewEngineFloat32_ProcessInto", "engine", true, func(in []float64, i, o float64, q QualityPreset, _ int) ([]float64, error) {
			return runEngine32Into(in, i, o, q, antiAliasPublicChunk)
		}},
		{"New", "new", false, runNewMono},
		{"ResampleMono", "engine", false, resampleViaMono},
	}
	ratios := []struct {
		in, out float64
		newOnly bool
	}{
		{384000, 256000, false},
		{48000, 32000, false},
		{48000, 44100, false},
		// New factors a 2:1 half-band stage first, so its 24000 to 16000
		// remainder reaches the non-integer engine branch; the engine paths
		// decimate 3:1 and were never affected.
		{48000, 16000, true},
	}
	presets := []QualityPreset{QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}
	for _, p := range paths {
		for _, rr := range ratios {
			if rr.newOnly && p.table != "new" {
				continue
			}
			for _, q := range presets {
				if testing.Short() && (q != QualityHigh || rr.in == 384000) {
					continue
				}
				t.Run(fmt.Sprintf("%s/%v_%g_to_%g", p.name, q, rr.in, rr.out), func(t *testing.T) {
					t.Parallel()
					want := antiAliasStopbandDB[p.table][q]
					if p.float32 {
						want = math.Min(want, antiAliasFloat32CapDB)
					}
					worst := math.Inf(-1)
					for _, m := range []float64{1.001, 1.003, 1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5} {
						f := m * rr.out / 2
						if f > 0.985*rr.in/2 {
							break
						}
						n := int(math.Ceil(antiAliasOutSamples * rr.in / rr.out))
						y, err := p.run(testutil.Sine(rr.in, f, n), rr.in, rr.out, q, 0)
						if err != nil {
							t.Fatal(err)
						}
						worst = math.Max(worst, testutil.ToneLevelDB(y, rr.out, testutil.FoldFrequency(f, rr.out)))
					}
					if worst > -want {
						t.Errorf("alias level %.1f dB, want <= %.1f dB", worst, -want)
					}
				})
			}
		}
	}
}

// TestNonIntegerDownsampling_Stereo_MeetsStopband checks the right channel of
// ResampleStereo and ResampleStereoFloat32, which reuse one engine after a
// Reset. The left channel carries a loud in-band tone and the right channel
// the out-of-band one: a bypassed pre-stage shows as missing rejection, and
// state left behind by the left channel shows as a right
// channel that differs from an independent ResampleMono of the same input.
func TestNonIntegerDownsampling_Stereo_MeetsStopband(t *testing.T) {
	const (
		inRate, outRate = 48000.0, 32000.0
		// 1.1 times the output Nyquist: inside the transition's far side,
		// where the pre-stage must already reject.
		tone = 1.1 * outRate / 2
		// The engine reaches the High filter (126 dB); float32 floors lower.
		wantDB = 126.0
	)
	n := int(math.Ceil(antiAliasOutSamples * inRate / outRate))
	right := testutil.Sine(inRate, tone, n)
	left := testutil.Sine(inRate, 0.3*outRate/2, n)
	fold := testutil.FoldFrequency(tone, outRate)

	_, r64, err := ResampleStereo(left, right, inRate, outRate, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	alone, err := ResampleMono(right, inRate, outRate, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	if len(alone) != len(r64) {
		t.Fatalf("ResampleStereo right channel has %d samples, ResampleMono %d", len(r64), len(alone))
	}
	for i := range alone {
		if alone[i] != r64[i] {
			t.Fatalf("ResampleStereo right channel differs from ResampleMono at sample %d: %v vs %v", i, r64[i], alone[i])
		}
	}
	if got := testutil.ToneLevelDB(r64, outRate, fold); got > -wantDB {
		t.Errorf("ResampleStereo right channel alias level %.1f dB, want <= %.1f dB", got, -wantDB)
	}

	left32, right32 := toFloat32(left), toFloat32(right)
	_, r32, err := ResampleStereoFloat32(left32, right32, inRate, outRate, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	if got := testutil.ToneLevelDB(toFloat64(r32), outRate, fold); got > -wantDB {
		t.Errorf("ResampleStereoFloat32 right channel alias level %.1f dB, want <= %.1f dB", got, -wantDB)
	}
}
