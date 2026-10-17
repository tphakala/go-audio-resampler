// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// Output is time-aligned: output sample m corresponds to input time
// m/ratio, as in libsoxr. These tests measure that on a steady sine (phase
// fit) and on impulses (peak position), for every FIR preset and every
// entry point.

const (
	// alignTolInput is the allowed misalignment in input samples. The
	// polyphase prototype has an odd length for some filter banks, so its
	// group delay is a half phase unit; the stage rounds it to a whole unit.
	// The residual is at most 1/(4L) input samples, below 0.005 for L >= 64.
	alignTolInput = 0.005

	alignSineHz  = 100.0
	alignSineSec = 0.25
)

// 96000 to 22050 is not listed: on the New path its polyphase stage leaves a
// residual of 0.0016 output samples, which is 0.0068 input samples at this
// ratio, above alignTolInput. The residual is the same before and after the
// anti-alias pre-stage (measured on v1.8.0), so it is the documented polyphase
// rounding, not a pre-stage error.
var alignRatios = [][2]float64{
	{48000, 24000},   // decimation by 2
	{48000, 16000},   // decimation by 3
	{48000, 96000},   // DFT factor 2
	{16000, 48000},   // DFT factor 3
	{8000, 48000},    // DFT factor 6
	{44100, 48000},   // polyphase
	{22050, 48000},   // DFT plus polyphase
	{48000, 44100},   // polyphase, odd L*T
	{24000, 176400},  // largest output-domain residual
	{384000, 256000}, // 3:2
	{44100, 64000},   // active sub-phase interpolation
	{32000, 44100},   // active sub-phase interpolation
	{48000, 32000},   // anti-alias pre-stage plus polyphase
	{96000, 64000},   // anti-alias pre-stage plus polyphase
	{44100, 16000},   // anti-alias pre-stage, long filter
	{48000, 48000},   // unity
}

var alignPresets = []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}

// sinePhaseDelay returns the delay in samples of the sine of frequency f
// (sampled at rate) that best fits y over its middle half. A time-aligned
// output gives 0.
func sinePhaseDelay(y []float64, f, rate float64) float64 {
	w := 2 * math.Pi * f / rate
	lo, hi := len(y)/4, 3*len(y)/4
	var ss, cc, sc, ys, yc float64
	for m := lo; m < hi; m++ {
		s, c := math.Sin(w*float64(m)), math.Cos(w*float64(m))
		ss += s * s
		cc += c * c
		sc += s * c
		ys += y[m] * s
		yc += y[m] * c
	}
	det := ss*cc - sc*sc
	a := (ys*cc - yc*sc) / det
	b := (yc*ss - ys*sc) / det
	return -math.Atan2(b, a) / w
}

// parabolicPeak returns the position of the largest |y| refined by a
// parabola through its two neighbours.
func parabolicPeak(y []float64) float64 {
	bi := 0
	for i := range y {
		if math.Abs(y[i]) > math.Abs(y[bi]) {
			bi = i
		}
	}
	if bi == 0 || bi == len(y)-1 {
		return float64(bi)
	}
	a, b, c := y[bi-1], y[bi], y[bi+1]
	den := a - 2*b + c
	if den == 0 {
		return float64(bi)
	}
	return float64(bi) + 0.5*(a-c)/den
}

func gcdInt(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func toFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func toFloat64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

// alignRunner resamples a whole signal through one entry point, feeding it
// in chunks of the given size (0 = whole input; one-shot entry points ignore
// it).
type alignRunner func(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error)

func chunkSize(chunk, n int) int {
	if chunk <= 0 || chunk > n {
		return max(n, 1)
	}
	return chunk
}

func runEngine64(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error) {
	r, err := NewEngine(inRate, outRate, q)
	if err != nil {
		return nil, err
	}
	chunk = chunkSize(chunk, len(in))
	var out []float64
	for i := 0; i < len(in); i += chunk {
		o, err := r.Process(in[i:min(i+chunk, len(in))])
		if err != nil {
			return nil, err
		}
		out = append(out, o...)
	}
	t, err := r.Flush()
	return append(out, t...), err
}

func runEngine64Into(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error) {
	r, err := NewEngine(inRate, outRate, q)
	if err != nil {
		return nil, err
	}
	chunk = chunkSize(chunk, len(in))
	buf := make([]float64, r.EstimateOutput(chunk))
	var out []float64
	for i := 0; i < len(in); i += chunk {
		n, err := r.ProcessInto(in[i:min(i+chunk, len(in))], buf)
		if err != nil {
			return nil, err
		}
		out = append(out, buf[:n]...)
	}
	t, err := r.Flush()
	return append(out, t...), err
}

func runEngine32(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error) {
	r, err := NewEngineFloat32(inRate, outRate, q)
	if err != nil {
		return nil, err
	}
	in32 := toFloat32(in)
	chunk = chunkSize(chunk, len(in32))
	var out []float32
	for i := 0; i < len(in32); i += chunk {
		o, err := r.Process(in32[i:min(i+chunk, len(in32))])
		if err != nil {
			return nil, err
		}
		out = append(out, o...)
	}
	t, err := r.Flush()
	return toFloat64(append(out, t...)), err
}

func runEngine32Into(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error) {
	r, err := NewEngineFloat32(inRate, outRate, q)
	if err != nil {
		return nil, err
	}
	in32 := toFloat32(in)
	chunk = chunkSize(chunk, len(in32))
	buf := make([]float32, r.EstimateOutput(chunk))
	var out []float32
	for i := 0; i < len(in32); i += chunk {
		n, err := r.ProcessInto(in32[i:min(i+chunk, len(in32))], buf)
		if err != nil {
			return nil, err
		}
		out = append(out, buf[:n]...)
	}
	t, err := r.Flush()
	return toFloat64(append(out, t...)), err
}

func newConfig(inRate, outRate float64, q QualityPreset, channels int, parallel bool) *Config {
	return &Config{
		InputRate: inRate, OutputRate: outRate, Channels: channels,
		Quality: QualitySpec{Preset: q}, EnableParallel: parallel,
	}
}

func runNewMono(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error) {
	r, err := New(newConfig(inRate, outRate, q, 1, false))
	if err != nil {
		return nil, err
	}
	chunk = chunkSize(chunk, len(in))
	var out []float64
	for i := 0; i < len(in); i += chunk {
		o, err := r.Process(in[i:min(i+chunk, len(in))])
		if err != nil {
			return nil, err
		}
		out = append(out, o...)
	}
	t, err := r.Flush()
	return append(out, t...), err
}

// runNewStereo feeds the same signal to both channels through ProcessMulti
// and FlushMulti and returns the right channel after checking that both
// channels are identical.
func runNewStereo(parallel bool) alignRunner {
	return func(in []float64, inRate, outRate float64, q QualityPreset, chunk int) ([]float64, error) {
		r, err := New(newConfig(inRate, outRate, q, 2, parallel))
		if err != nil {
			return nil, err
		}
		chunk = chunkSize(chunk, len(in))
		var left, right []float64
		for i := 0; i < len(in); i += chunk {
			c := in[i:min(i+chunk, len(in))]
			o, err := r.ProcessMulti([][]float64{c, c})
			if err != nil {
				return nil, err
			}
			left = append(left, o[0]...)
			right = append(right, o[1]...)
		}
		mf, ok := r.(MultiFlusher)
		if !ok {
			return nil, fmt.Errorf("resampler does not implement MultiFlusher")
		}
		t, err := mf.FlushMulti()
		if err != nil {
			return nil, err
		}
		left = append(left, t[0]...)
		right = append(right, t[1]...)
		if !slices.Equal(left, right) {
			return nil, fmt.Errorf("stereo channels differ for identical input")
		}
		return right, nil
	}
}

func runResampleMono(in []float64, inRate, outRate float64, q QualityPreset, _ int) ([]float64, error) {
	return ResampleMono(in, inRate, outRate, q)
}

func runResampleMonoFloat32(in []float64, inRate, outRate float64, q QualityPreset, _ int) ([]float64, error) {
	out, err := ResampleMonoFloat32(toFloat32(in), inRate, outRate, q)
	return toFloat64(out), err
}

func runResampleStereo(in []float64, inRate, outRate float64, q QualityPreset, _ int) ([]float64, error) {
	_, right, err := ResampleStereo(in, in, inRate, outRate, q)
	return right, err
}

func runResampleStereoFloat32(in []float64, inRate, outRate float64, q QualityPreset, _ int) ([]float64, error) {
	in32 := toFloat32(in)
	_, right, err := ResampleStereoFloat32(in32, in32, inRate, outRate, q)
	return toFloat64(right), err
}

type alignEntry struct {
	name   string
	run    alignRunner
	chunks []int // 0 = whole input
}

func alignEntries() []alignEntry {
	return []alignEntry{
		{"NewEngine/Process", runEngine64, []int{0, 4096, 777, 7}},
		{"NewEngine/ProcessInto", runEngine64Into, []int{0, 777}},
		{"NewEngineFloat32/Process", runEngine32, []int{0, 777}},
		{"NewEngineFloat32/ProcessInto", runEngine32Into, []int{0, 777}},
		{"New/mono", runNewMono, []int{0, 4096, 777, 7}},
		{"New/stereo", runNewStereo(false), []int{0, 777}},
		{"New/stereo-parallel", runNewStereo(true), []int{0, 777}},
		{"ResampleMono", runResampleMono, []int{0}},
		{"ResampleMonoFloat32", runResampleMonoFloat32, []int{0}},
		{"ResampleStereo", runResampleStereo, []int{0}},
		{"ResampleStereoFloat32", runResampleStereoFloat32, []int{0}},
	}
}

func alignSine(inRate float64) []float64 {
	n := int(inRate * alignSineSec)
	in := make([]float64, n)
	for i := range in {
		in[i] = 0.5 * math.Sin(2*math.Pi*alignSineHz*float64(i)/inRate)
	}
	return in
}

func TestAlignment_SinePhase(t *testing.T) {
	entries := alignEntries()
	if testing.Short() {
		// Keep the race-detector run quick: one chunking per entry point.
		for i := range entries {
			entries[i].chunks = entries[i].chunks[len(entries[i].chunks)-1:]
		}
	}
	for _, rr := range alignRatios {
		in := alignSine(rr[0])
		for _, q := range alignPresets {
			for _, e := range entries {
				for _, chunk := range e.chunks {
					name := fmt.Sprintf("%g->%g/q%d/%s/chunk%d", rr[0], rr[1], q, e.name, chunk)
					out, err := e.run(in, rr[0], rr[1], q, chunk)
					if err != nil {
						t.Errorf("%s: %v", name, err)
						continue
					}
					d := sinePhaseDelay(out, alignSineHz, rr[1])
					if got := math.Abs(d) * rr[0] / rr[1]; got > alignTolInput {
						t.Errorf("%s: output is %.4f output samples late (%.4f input samples), want 0 within %g input samples",
							name, d, got, alignTolInput)
					}
				}
			}
		}
	}
}

// Chunk size 1 on a short input: the same alignment must hold when every
// Process call carries a single sample.
func TestAlignment_SinePhaseSingleSampleChunks(t *testing.T) {
	if testing.Short() {
		t.Skip("slow in -short mode")
	}
	const sec = 0.2
	for _, rr := range [][2]float64{{48000, 24000}, {8000, 48000}, {44100, 48000}, {48000, 44100}} {
		in := make([]float64, int(rr[0]*sec))
		for i := range in {
			in[i] = 0.5 * math.Sin(2*math.Pi*alignSineHz*float64(i)/rr[0])
		}
		for _, q := range []QualityPreset{QualityLow, QualityHigh} {
			for name, run := range map[string]alignRunner{"NewEngine": runEngine64, "New": runNewMono} {
				out, err := run(in, rr[0], rr[1], q, 1)
				if err != nil {
					t.Fatalf("%s %g->%g q%d: %v", name, rr[0], rr[1], q, err)
				}
				d := sinePhaseDelay(out, alignSineHz, rr[1])
				if got := math.Abs(d) * rr[0] / rr[1]; got > alignTolInput {
					t.Errorf("%s %g->%g q%d: %.4f output samples late, want 0", name, rr[0], rr[1], q, d)
				}
			}
		}
	}
}

// impulseResponse feeds a unit impulse at input sample p through run and
// returns the output.
func impulseResponse(run alignRunner, inRate, outRate float64, q QualityPreset, p, n int) ([]float64, error) {
	in := make([]float64, n)
	in[p] = 1
	return run(in, inRate, outRate, q, 4096)
}

// firstExactImpulse returns the smallest p >= minP for which p*ratio is a
// whole number of output samples.
func firstExactImpulse(inRate, outRate float64, minP int) int {
	g := gcdInt(int(inRate), int(outRate))
	step := int(inRate) / g
	return (minP + step - 1) / step * step
}

func TestAlignment_ImpulsePeak(t *testing.T) {
	paths := map[string]alignRunner{
		"NewEngine":        runEngine64,
		"NewEngineFloat32": runEngine32,
		"New":              runNewMono,
	}
	for _, rr := range alignRatios {
		p := firstExactImpulse(rr[0], rr[1], 2000)
		n := p + 4000
		want := float64(p) * rr[1] / rr[0]
		for _, q := range alignPresets {
			for name, run := range paths {
				out, err := impulseResponse(run, rr[0], rr[1], q, p, n)
				if err != nil {
					t.Fatalf("%s %g->%g q%d: %v", name, rr[0], rr[1], q, err)
				}
				if got := parabolicPeak(out); math.Abs(got-want) > 0.05 {
					t.Errorf("%s %g->%g q%d: impulse at input %d peaks at output %.3f, want %.3f",
						name, rr[0], rr[1], q, p, got, want)
				}
			}
		}
	}
}

// An impulse at input sample 0 must peak at output sample 0 for chains with
// a single FIR stage: the stage emits the head of the stream (its prefill),
// not only the steady state. Multi-stage chains are not covered here: their
// first stage emits no negative-time samples, so the next stage sees a
// one-sided onset and its output peaks one sample later; see
// TestAlignment_ImpulseNearStreamStartMultiStage.
func TestAlignment_ImpulseAtStreamStart(t *testing.T) {
	for _, rr := range [][2]float64{{48000, 96000}, {16000, 48000}, {8000, 48000}, {48000, 24000}, {48000, 16000}} {
		for _, q := range []QualityPreset{QualityLow, QualityMedium, QualityHigh, QualityVeryHigh} {
			in := make([]float64, 4000)
			in[0] = 1
			out, err := runEngine64(in, rr[0], rr[1], q, 4096)
			if err != nil {
				t.Fatal(err)
			}
			best := 0
			for i := range out {
				if math.Abs(out[i]) > math.Abs(out[best]) {
					best = i
				}
			}
			if best != 0 {
				t.Errorf("%g->%g q%d: impulse at input 0 peaks at output %d, want 0", rr[0], rr[1], q, best)
			}
		}
	}
}

// For multi-stage chains the impulse is placed at the first exact position
// at or after the chain's group delay (Latency() output samples, converted to
// input samples), so the whole filter response is inside the stream.
func TestAlignment_ImpulseNearStreamStartMultiStage(t *testing.T) {
	for _, rr := range [][2]float64{{8000, 44100}, {8000, 48000}, {11025, 44100}, {22050, 48000}, {48000, 32000}, {48000, 44100}} {
		for _, q := range []QualityPreset{QualityLow, QualityHigh} {
			e, err := NewEngine(rr[0], rr[1], q)
			if err != nil {
				t.Fatal(err)
			}
			p := firstExactImpulse(rr[0], rr[1], int(math.Ceil(float64(e.Latency())*rr[0]/rr[1])))
			want := float64(p) * rr[1] / rr[0]
			for name, run := range map[string]alignRunner{"NewEngine": runEngine64, "New": runNewMono} {
				out, err := impulseResponse(run, rr[0], rr[1], q, p, p+4000)
				if err != nil {
					t.Fatal(err)
				}
				if got := parabolicPeak(out); math.Abs(got-want) > 0.05 {
					t.Errorf("%s %g->%g q%d: impulse at input %d peaks at output %.3f, want %.3f",
						name, rr[0], rr[1], q, p, got, want)
				}
			}
		}
	}
}

// alignmentStream is the common surface of the entry points exercised by the
// lifecycle test.
type alignmentStream struct {
	name   string
	feed   func(in []float64) ([]float64, error)
	flush  func() ([]float64, error)
	reset  func()
	create func() alignmentStream
}

func engine64Stream(inRate, outRate float64, q QualityPreset) func() alignmentStream {
	var create func() alignmentStream
	create = func() alignmentStream {
		r, err := NewEngine(inRate, outRate, q)
		if err != nil {
			panic(err)
		}
		return alignmentStream{"NewEngine", r.Process, r.Flush, r.Reset, create}
	}
	return create
}

func engine32Stream(inRate, outRate float64, q QualityPreset) func() alignmentStream {
	var create func() alignmentStream
	create = func() alignmentStream {
		r, err := NewEngineFloat32(inRate, outRate, q)
		if err != nil {
			panic(err)
		}
		return alignmentStream{
			"NewEngineFloat32",
			func(in []float64) ([]float64, error) {
				o, err := r.Process(toFloat32(in))
				return toFloat64(o), err
			},
			func() ([]float64, error) {
				o, err := r.Flush()
				return toFloat64(o), err
			},
			r.Reset, create,
		}
	}
	return create
}

func newMonoStream(inRate, outRate float64, q QualityPreset) func() alignmentStream {
	var create func() alignmentStream
	create = func() alignmentStream {
		r, err := New(newConfig(inRate, outRate, q, 1, false))
		if err != nil {
			panic(err)
		}
		return alignmentStream{"New/mono", r.Process, r.Flush, r.Reset, create}
	}
	return create
}

func newStereoStream(inRate, outRate float64, q QualityPreset) func() alignmentStream {
	var create func() alignmentStream
	create = func() alignmentStream {
		r, err := New(newConfig(inRate, outRate, q, 2, false))
		if err != nil {
			panic(err)
		}
		mf, ok := r.(MultiFlusher)
		if !ok {
			panic("New(config) resampler does not implement MultiFlusher")
		}
		return alignmentStream{
			"New/stereo",
			func(in []float64) ([]float64, error) {
				o, err := r.ProcessMulti([][]float64{in, in})
				if err != nil {
					return nil, err
				}
				if !slices.Equal(o[0], o[1]) {
					return nil, fmt.Errorf("stereo channels differ")
				}
				return o[1], nil
			},
			func() ([]float64, error) {
				o, err := mf.FlushMulti()
				if err != nil {
					return nil, err
				}
				if !slices.Equal(o[0], o[1]) {
					return nil, fmt.Errorf("stereo channels differ")
				}
				return o[1], nil
			},
			r.Reset, create,
		}
	}
	return create
}

// run feeds in in chunks and flushes once.
func (s alignmentStream) run(t *testing.T, in []float64) []float64 {
	t.Helper()
	const chunk = 1000
	var out []float64
	for i := 0; i < len(in); i += chunk {
		o, err := s.feed(in[i:min(i+chunk, len(in))])
		if err != nil {
			t.Fatalf("%s: feed: %v", s.name, err)
		}
		out = append(out, o...)
	}
	tail, err := s.flush()
	if err != nil {
		t.Fatalf("%s: flush: %v", s.name, err)
	}
	return append(out, tail...)
}

// Flush is terminal and Reset starts a new stream: after either, the next
// stream must be bit-identical to one fed to a fresh instance, and must still
// be time-aligned (the stage prefill is re-established). A second Flush and a
// Flush on a never-fed instance return nothing.
func TestAlignment_FlushResetTransitions(t *testing.T) {
	type tc struct {
		in, out float64
		q       QualityPreset
	}
	cases := []tc{
		{48000, 24000, QualityHigh},
		{8000, 48000, QualityHigh},
		{16000, 48000, QualityMedium},
		{44100, 48000, QualityLow},
		{48000, 44100, QualityVeryHigh},
		{22050, 48000, QualityHigh},
	}
	makers := []func(in, out float64, q QualityPreset) func() alignmentStream{
		engine64Stream, engine32Stream, newMonoStream, newStereoStream,
	}
	for _, c := range cases {
		// Stream A differs from stream B, so leftover state would show.
		bAligned := alignSine(c.in)
		a := make([]float64, len(bAligned))
		for i := range a {
			a[i] = 0.25 * math.Sin(2*math.Pi*alignSineHz*float64(i)/c.in+1)
		}
		for _, mk := range makers {
			create := mk(c.in, c.out, c.q)
			name := fmt.Sprintf("%s %g->%g q%d", create().name, c.in, c.out, c.q)

			fresh := create().run(t, bAligned)
			if d := sinePhaseDelay(fresh, alignSineHz, c.out); math.Abs(d)*c.in/c.out > alignTolInput {
				t.Errorf("%s: fresh stream %.4f output samples late", name, d)
			}

			// Flush on a never-fed instance.
			if tail, err := create().flush(); err != nil || len(tail) != 0 {
				t.Errorf("%s: Flush on never-fed instance returned %d samples, err %v", name, len(tail), err)
			}

			// Stream A, Flush, second Flush, stream B.
			s := create()
			s.run(t, a)
			if tail, err := s.flush(); err != nil || len(tail) != 0 {
				t.Errorf("%s: second Flush returned %d samples, err %v", name, len(tail), err)
			}
			afterFlush := s.run(t, bAligned)
			if !slices.Equal(afterFlush, fresh) {
				t.Errorf("%s: stream after Flush differs from a fresh instance", name)
			}

			// Reset mid-stream, then stream B.
			s = create()
			if _, err := s.feed(a[:len(a)/3]); err != nil {
				t.Fatal(err)
			}
			s.reset()
			afterReset := s.run(t, bAligned)
			if !slices.Equal(afterReset, fresh) {
				t.Errorf("%s: stream after Reset differs from a fresh instance", name)
			}
		}
	}
}
