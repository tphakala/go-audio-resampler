// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/tphakala/go-audio-resampler/internal/pipeline"
)

// newHalfBandStage must never substitute the degraded stubStage fallback for
// a construction failure; it must return the error instead. This was
// previously silent: a failed newPolyphaseStage call fell back to a
// nearest-neighbor stub with no signal to the caller.
//
// pipeline.BuildPipeline only ever emits StageHalfBand specs with Ratio
// pinned to halfRatio (0.5) or doubleRatio (2.0); those are the only two
// ratios newHalfBandStage is ever invoked with through the public New()
// path. Both are well inside engine.NewResampler's valid ratio range
// [1/256, 256], so construction cannot fail for them today. These tests
// pin the signature contract (error-free construction still works, and the
// error return exists and would propagate if it were ever non-nil).
func TestNewHalfBandStage_RepresentativeRatios(t *testing.T) {
	ratios := []float64{0.5, 2.0} // exactly what pipeline.go emits for StageHalfBand
	precisions := []int{16, 20, 24, 28, 32, 33}

	for _, ratio := range ratios {
		for _, precision := range precisions {
			stage, err := newHalfBandStage(ratio, 0, precision)
			require.NoErrorf(t, err, "ratio=%v precision=%d", ratio, precision)
			require.NotNilf(t, stage, "ratio=%v precision=%d", ratio, precision)

			// Must not be the degraded nearest-neighbor stub.
			if _, isStub := stage.(*stubStage); isStub {
				t.Fatalf("ratio=%v precision=%d: newHalfBandStage returned stubStage instead of a real polyphase stage", ratio, precision)
			}

			assert.InDeltaf(t, ratio, stage.GetRatio(), 1e-9, "ratio=%v precision=%d", ratio, precision)
		}
	}
}

// createStage's StageHalfBand branch must propagate newHalfBandStage's
// (Stage, error) return directly, matching the pattern already used by the
// StagePolyphase and StageFFT branches, instead of discarding the error and
// always returning nil.
func TestCreateStage_HalfBand_PropagatesSignature(t *testing.T) {
	config := &Config{
		InputRate:  48000,
		OutputRate: 24000,
		Channels:   1,
		Quality:    GetPresetSpec(QualityHigh),
	}

	spec := StageSpec{
		StageSpec: pipeline.StageSpec{
			Type:         pipeline.StageHalfBand,
			Ratio:        0.5,
			FilterLength: 32,
		},
	}

	stage, err := createStage(spec, config)
	require.NoError(t, err)
	require.NotNil(t, stage)

	if _, isStub := stage.(*stubStage); isStub {
		t.Fatal("createStage(StageHalfBand) returned stubStage instead of a real polyphase stage")
	}
	assert.InDelta(t, 0.5, stage.GetRatio(), 1e-9)
}

// New() must construct a working pipeline (no silently degraded stub
// stages) for configurations whose ratio decomposition routes through one
// or more half-band stages, in both the upsampling and downsampling
// directions.
func TestNew_HalfBandPipeline_NoStubFallback(t *testing.T) {
	cases := []struct {
		name       string
		inputRate  float64
		outputRate float64
	}{
		{"upsample_needs_halfband", 8000, 48000},   // ratio 6.0: halfband stages factor out powers of 2
		{"downsample_needs_halfband", 48000, 8000}, // ratio 1/6: halfband stages factor out powers of 2
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := &Config{
				InputRate:  tc.inputRate,
				OutputRate: tc.outputRate,
				Channels:   1,
				Quality:    QualitySpec{Preset: QualityHigh},
			}

			r, err := New(config)
			require.NoError(t, err)
			require.NotNil(t, r)

			crr, ok := r.(*constantRateResampler)
			require.True(t, ok, "expected *constantRateResampler")

			sawHalfBand := false
			for _, spec := range crr.pipeline.stages {
				if spec.engine == "halfband" {
					sawHalfBand = true
				}
			}
			require.True(t, sawHalfBand, "test config expected to produce at least one halfband stage")

			for chIdx, ch := range crr.channels {
				for stIdx, stg := range ch.stages {
					if _, isStub := stg.(*stubStage); isStub {
						t.Fatalf("channel %d stage %d is a stubStage: half-band construction silently degraded", chIdx, stIdx)
					}
				}
			}

			// Confirm the pipeline actually processes audio end to end.
			// Input must clear filter startup latency to produce output from
			// Process alone, so also drain Flush to be independent of that.
			input := make([]float64, 8192)
			for i := range input {
				input[i] = 1.0
			}
			out, err := r.Process(input)
			require.NoError(t, err)

			tail, err := r.Flush()
			require.NoError(t, err)

			assert.NotEmpty(t, append(out, tail...))
		})
	}
}
