// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestNewPath_Flush_MultiStageTailNotDropped is a regression test for the bug
// where constantRateResampler.Flush() flushed each stage into the next stage's
// input buffer but never ran that buffer back through the next stage. For any
// pipeline with 2+ stages the last stage boundary's flushed tail was left in an
// intermediate buffer and dropped, so the New(config) path came up hundreds of
// samples short of the canonical resampled length (e.g. -269 for 48k->16k,
// -476 for 88.2k->16k). The engine path (NewEngine) routes inter-stage flush
// output correctly and lands within a few samples of ideal.
//
// The ratios below all decompose into multi-stage pipelines (halfband +
// polyphase, sometimes two halfband stages). The test asserts the New(config)
// Process+Flush total length is not short of the ideal resampled length, which
// fails by hundreds of samples when the inter-stage tail is dropped.
func TestNewPath_Flush_MultiStageTailNotDropped(t *testing.T) {
	pairs := []struct {
		inRate, outRate float64
	}{
		{48000, 16000},  // 2 stages: halfband + polyphase
		{96000, 16000},  // 3 stages: halfband + halfband + polyphase
		{48000, 8000},   // 3 stages
		{192000, 48000}, // 2 stages: halfband + halfband
		{88200, 16000},  // 3 stages
	}

	const durSeconds = 2

	for _, p := range pairs {
		t.Run(fmt.Sprintf("%gto%g", p.inRate, p.outRate), func(t *testing.T) {
			ratio := p.outRate / p.inRate
			n := int(p.inRate) * durSeconds
			ideal := int(math.Round(float64(n) * ratio))

			cfg := &Config{
				InputRate:  p.inRate,
				OutputRate: p.outRate,
				Channels:   1,
				Quality:    QualitySpec{Preset: QualityHigh},
			}

			r, err := New(cfg)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			// Guard the premise: this regression only exercises the inter-stage
			// flush routing when the pipeline actually has multiple stages.
			crr, ok := r.(*constantRateResampler)
			if !ok {
				t.Fatalf("New(...) did not return *constantRateResampler")
			}
			if got := len(crr.channels[0].stages); got < 2 {
				t.Fatalf("expected a multi-stage pipeline, got %d stage(s)", got)
			}

			rng := rand.New(rand.NewSource(4242))
			input := makeDeterministicInput(rng, n, p.inRate)

			proc, err := r.Process(input)
			if err != nil {
				t.Fatalf("Process: %v", err)
			}
			flush, err := r.Flush()
			if err != nil {
				t.Fatalf("Flush: %v", err)
			}
			total := len(proc) + len(flush)

			// A correctly flushed resampler emits approximately ideal samples;
			// the only legitimate shortfall is sub-sample rounding. The bug
			// dropped 200-480 samples, so a small lower margin catches it while
			// tolerating rounding. The upper bound guards against gross
			// over-emission (e.g. an over-padded flush).
			const lowMargin = 64
			const highMargin = 256
			if total < ideal-lowMargin {
				t.Errorf("Process+Flush total = %d, want >= %d (ideal %d); inter-stage flush tail dropped (%d samples short)",
					total, ideal-lowMargin, ideal, ideal-total)
			}
			if total > ideal+highMargin {
				t.Errorf("Process+Flush total = %d, want <= %d (ideal %d); flush over-emitted",
					total, ideal+highMargin, ideal)
			}
		})
	}
}
