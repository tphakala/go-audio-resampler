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
		inRate        = 44100.0
		outRate       = 48000.0
		outFrames     = 512
		callbacks     = 100
		toneAmplitude = 0.5
		toneHz        = 997.0
	)

	rs, err := resampler.NewEngineFloat32(inRate, outRate, resampler.QualityHigh)
	if err != nil {
		panic(err)
	}
	ratio := rs.GetRatio()

	// Prime the FIFO with the startup deficit so the first callbacks are
	// fed. This trades Latency() samples of leading silence for a steady
	// pipeline. The FIFO starts empty with headroom, then the priming zeros
	// are appended so later appends grow a zero-length-origin slice.
	fifo := make([]float32, 0, rs.Latency()+2*outFrames)
	fifo = append(fifo, make([]float32, rs.Latency())...)

	phase := 0.0
	firstCall := true
	for callback := range callbacks {
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
		//
		// This holds when outFrames comfortably exceeds Latency(), as here.
		// With an output buffer smaller than the deficit, a single primed
		// Process cannot cover it and the shortfall instead spreads over the
		// first several callbacks until the FIFO fills; the underrun branch
		// below tolerates that warmup.
		need := outFrames
		if !firstCall {
			need = max(outFrames-len(fifo), 0)
		}
		firstCall = false
		inFrames := int(math.Ceil(float64(need) / ratio))

		// Allocated per callback for clarity; a production callback should
		// reuse a single scratch buffer instead of allocating each call.
		in := make([]float32, inFrames)
		for i := range in {
			in[i] = float32(toneAmplitude * math.Sin(phase))
			phase += 2 * math.Pi * toneHz / inRate
		}
		out, err := rs.Process(in)
		if err != nil {
			panic(err)
		}
		fifo = append(fifo, out...)

		if len(fifo) >= outFrames {
			deliver := fifo[:outFrames]
			_ = deliver // hand exactly outFrames samples to the audio API here
			// Resliced from the front for clarity; the drained head keeps the
			// backing array growing over a long stream, so production code
			// would use a ring buffer instead.
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
