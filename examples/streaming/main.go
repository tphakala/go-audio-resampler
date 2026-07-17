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
	// Round the input chunk size up rather than truncating: truncating would
	// underfeed the resampler by a fraction of a sample per callback, which
	// compounds into periodic FIFO underruns over a long-running stream.
	inFrames := int(math.Ceil(float64(outFrames) / ratio))

	// Prime the FIFO with the startup deficit so the first callbacks are
	// fed. This trades Latency() samples of leading silence for a steady
	// pipeline.
	fifo := make([]float32, rs.Latency())

	phase := 0.0
	for callback := 0; callback < 100; callback++ {
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
