// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

// Example: real-time chunked resampling into fixed-size output buffers.
//
// An audio callback (portaudio, miniaudio, ...) demands exactly N output
// frames per call, but a streaming resampler returns a varying number of
// samples per Process call (early calls withhold samples while the filter
// primes). The fix is a small FIFO between the resampler and the callback,
// primed with Latency() samples of silence. Never call Flush or Reset
// inside the stream: Flush is end-of-stream only, Reset starts a new
// stream (both destroy continuity and cause audible clicks, issue #51).
package main

import (
	"fmt"
	"math"

	resampler "github.com/tphakala/go-audio-resampler"
)

func main() {
	const (
		inRate    = 44100.0
		outRate   = 48000.0
		outFrames = 512
	)

	rs, err := resampler.NewEngineFloat32(inRate, outRate, resampler.QualityHigh)
	if err != nil {
		panic(err)
	}
	ratio := rs.GetRatio()

	// Prime the FIFO with the startup deficit so the first callbacks are
	// fed. This trades Latency() samples of leading silence for a steady
	// pipeline.
	fifo := make([]float32, rs.Latency())

	phase := 0.0
	firstCall := true
	for callback := 0; callback < 100; callback++ {
		// Size the input chunk from the FIFO's current deficit rather than
		// a fixed count. A fixed input size drifts against a fixed output
		// size whenever ratio does not divide outFrames evenly: truncating
		// the fixed size underfeeds the resampler and causes underruns
		// every few seconds, while rounding it up overfeeds and grows the
		// FIFO (and its latency) without bound over a long-running stream.
		// Pulling exactly enough input to cover the current shortfall self-
		// corrects both directions and keeps the FIFO bounded.
		//
		// The very first Process call is a special case: a fresh engine
		// pays its entire Latency() startup deficit on that one call,
		// regardless of how much input it receives, and the priming above
		// exists to cover exactly that. So the first request must target a
		// full outFrames, not outFrames minus the priming already sitting
		// in the FIFO; netting the priming against the first request would
		// count the same deficit twice and under-deliver on that call.
		need := outFrames
		if !firstCall {
			need = outFrames - len(fifo)
			if need < 0 {
				need = 0
			}
		}
		firstCall = false
		inFrames := int(math.Ceil(float64(need) / ratio))

		in := make([]float32, inFrames)
		for i := range in {
			in[i] = float32(0.5 * math.Sin(phase))
			phase += 2 * math.Pi * 997 / inRate
		}
		out, err := rs.Process(in)
		if err != nil {
			panic(err)
		}
		fifo = append(fifo, out...)

		if len(fifo) >= outFrames {
			deliver := fifo[:outFrames]
			_ = deliver // hand exactly outFrames samples to the audio API here
			fifo = fifo[outFrames:]
		} else {
			// Underrun (should not happen after priming): deliver silence.
			fmt.Printf("callback %d: FIFO underrun (%d < %d)\n", callback, len(fifo), outFrames)
		}
	}

	// End of stream: drain the filter tail exactly once.
	tail, err := rs.Flush()
	if err != nil {
		panic(err)
	}
	fifo = append(fifo, tail...)
	fmt.Printf("stream done, %d samples left to deliver\n", len(fifo))
}
