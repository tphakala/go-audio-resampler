// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestFlushMulti_AllChannelsDrained(t *testing.T) {
	pairs := []struct {
		inRate, outRate float64
	}{
		{48000, 16000},
		{96000, 16000},
		{48000, 8000},
		{192000, 48000},
		{44100, 16000},
	}

	const durSeconds = 2
	const channels = 2

	for _, p := range pairs {
		t.Run(fmt.Sprintf("%gto%g", p.inRate, p.outRate), func(t *testing.T) {
			ratio := p.outRate / p.inRate
			n := int(p.inRate) * durSeconds
			ideal := int(math.Round(float64(n) * ratio))

			cfg := &Config{
				InputRate:  p.inRate,
				OutputRate: p.outRate,
				Channels:   channels,
				Quality:    QualitySpec{Preset: QualityHigh},
			}

			r, err := New(cfg)
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			rng := rand.New(rand.NewSource(4242))
			input := make([][]float64, channels)
			for ch := range input {
				input[ch] = makeDeterministicInput(rng, n, p.inRate)
			}

			proc, err := r.ProcessMulti(input)
			if err != nil {
				t.Fatalf("ProcessMulti: %v", err)
			}

			mf, ok := r.(MultiFlusher)
			if !ok {
				t.Fatal("resampler does not implement MultiFlusher")
			}

			flushed, err := mf.FlushMulti()
			if err != nil {
				t.Fatalf("FlushMulti: %v", err)
			}

			if len(flushed) != channels {
				t.Fatalf("FlushMulti returned %d channels, want %d", len(flushed), channels)
			}

			const lowMargin = 64
			const highMargin = 256

			for ch := range channels {
				total := len(proc[ch]) + len(flushed[ch])
				if total < ideal-lowMargin {
					t.Errorf("channel %d: Process+FlushMulti total = %d, want >= %d (ideal %d); %d samples short",
						ch, total, ideal-lowMargin, ideal, ideal-total)
				}
				if total > ideal+highMargin {
					t.Errorf("channel %d: Process+FlushMulti total = %d, want <= %d (ideal %d); flush over-emitted",
						ch, total, ideal+highMargin, ideal)
				}
			}
		})
	}
}

func TestFlushMulti_MatchesPerChannelFlush(t *testing.T) {
	const inRate = 48000.0
	const outRate = 16000.0
	const dur = 1
	const channels = 3
	n := int(inRate) * dur

	rng := rand.New(rand.NewSource(9999))
	channelInputs := make([][]float64, channels)
	for ch := range channelInputs {
		channelInputs[ch] = makeDeterministicInput(rng, n, inRate)
	}

	// Per-channel reference: process each channel independently with a mono resampler.
	monoTotals := make([]int, channels)
	for ch := range channels {
		cfg := &Config{
			InputRate:  inRate,
			OutputRate: outRate,
			Channels:   1,
			Quality:    QualitySpec{Preset: QualityHigh},
		}
		r, err := New(cfg)
		if err != nil {
			t.Fatalf("mono New: %v", err)
		}
		proc, err := r.Process(channelInputs[ch])
		if err != nil {
			t.Fatalf("mono Process ch%d: %v", ch, err)
		}
		fl, err := r.Flush()
		if err != nil {
			t.Fatalf("mono Flush ch%d: %v", ch, err)
		}
		monoTotals[ch] = len(proc) + len(fl)
	}

	// Multi-channel: process all channels together.
	cfg := &Config{
		InputRate:  inRate,
		OutputRate: outRate,
		Channels:   channels,
		Quality:    QualitySpec{Preset: QualityHigh},
	}
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("multi New: %v", err)
	}
	proc, err := r.ProcessMulti(channelInputs)
	if err != nil {
		t.Fatalf("ProcessMulti: %v", err)
	}
	mf, ok := r.(MultiFlusher)
	if !ok {
		t.Fatal("resampler does not implement MultiFlusher")
	}

	flushed, err := mf.FlushMulti()
	if err != nil {
		t.Fatalf("FlushMulti: %v", err)
	}

	for ch := range channels {
		multiTotal := len(proc[ch]) + len(flushed[ch])
		if multiTotal != monoTotals[ch] {
			t.Errorf("channel %d: multi total %d != mono total %d",
				ch, multiTotal, monoTotals[ch])
		}
	}
}

func TestFlushMulti_EmptyResampler(t *testing.T) {
	cfg := &Config{
		InputRate:  48000,
		OutputRate: 16000,
		Channels:   2,
		Quality:    QualitySpec{Preset: QualityHigh},
	}
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	mf, ok := r.(MultiFlusher)
	if !ok {
		t.Fatal("resampler does not implement MultiFlusher")
	}

	flushed, err := mf.FlushMulti()
	if err != nil {
		t.Fatalf("FlushMulti: %v", err)
	}
	if len(flushed) != 2 {
		t.Fatalf("FlushMulti returned %d channels, want 2", len(flushed))
	}
}
