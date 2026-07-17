// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"testing"
)

// TestLatencyReporting verifies Latency() (the startup deficit reported so
// callers can prime a fixed-size output FIFO) against a deficit measured
// independently of that accessor.
//
// The DFT pre-stage does valid-mode convolution (see processZeroCopy): it
// never zero-pads history, so it withholds output entirely (returns an
// empty slice) until enough real input has accumulated, rather than
// emitting spurious zero-valued samples. That means a single one-shot
// Process() call over a large chunk returns fewer output samples than an
// ideal, delay-free resampler would (n*ratio): the shortfall is exactly the
// output the filter is withholding while its delay lines prime, which
// Flush() would later release. That shortfall is this test's independent
// measurement of the startup deficit.
func TestLatencyReporting(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Resampler ratio: %.6f", r.ratio)
	if r.preStage != nil {
		t.Logf("DFT pre-stage: factor=%d tapsPerPhase=%d", r.preStage.factor, r.preStage.tapsPerPhase)
	}
	if r.polyphaseStage != nil {
		t.Logf("Polyphase stage: numPhases=%d tapsPerPhase=%d step=%d",
			r.polyphaseStage.numPhases, r.polyphaseStage.tapsPerPhase, r.polyphaseStage.step)
	}

	const n = 8192
	input := make([]float64, n)
	for i := range input {
		input[i] = math.Cos(2 * math.Pi * 997 * float64(i) / 44100)
	}

	output, err := r.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	ideal := float64(n) * r.ratio
	measuredDeficit := int(math.Round(ideal)) - len(output)

	latency := r.Latency()
	t.Logf("Input: %d samples, ideal output: %.1f, actual Process() output: %d, measured deficit: %d, Latency(): %d",
		n, ideal, len(output), measuredDeficit, latency)

	if diff := latency - measuredDeficit; diff < -2 || diff > 2 {
		t.Errorf("Latency() = %d, measured startup deficit = %d, want within 2 samples", latency, measuredDeficit)
	}
}
