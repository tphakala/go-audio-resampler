// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"strconv"
	"testing"

	"github.com/tphakala/go-audio-resampler/internal/filter"
)

// Stage-level alignment: each FIR stage maps its output sample j to input
// time j/ratio on its own, so every chain built from them is aligned by
// composition. The root package tests (alignment_test.go) cover the public
// entry points; these pin each stage's contribution and its flush length.

var alignFIRQualities = []Quality{QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}

// alStage is the Process/Flush surface shared by the three FIR stages.
type alStage interface {
	Process(input []float64) ([]float64, error)
	Flush() ([]float64, error)
}

// alRun feeds in in chunks (0 = whole input) and flushes once.
func alRun(t *testing.T, s alStage, in []float64, chunk int) []float64 {
	t.Helper()
	if chunk <= 0 || chunk > len(in) {
		chunk = len(in)
	}
	var out []float64
	for i := 0; i < len(in); i += chunk {
		o, err := s.Process(in[i:min(i+chunk, len(in))])
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		out = append(out, o...)
	}
	tail, err := s.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	return append(out, tail...)
}

func alPeak(y []float64) (idx int, refined float64) {
	for i := range y {
		if math.Abs(y[i]) > math.Abs(y[idx]) {
			idx = i
		}
	}
	refined = float64(idx)
	if idx > 0 && idx < len(y)-1 {
		a, b, c := y[idx-1], y[idx], y[idx+1]
		if den := a - 2*b + c; den != 0 {
			refined += 0.5 * (a - c) / den
		}
	}
	return idx, refined
}

// alSineDelay returns the delay in samples of the sine of period `period`
// samples that best fits the middle half of y.
func alSineDelay(y []float64, period float64) float64 {
	w := 2 * math.Pi / period
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

func alImpulse(n, p int) []float64 {
	in := make([]float64, n)
	in[p] = 1
	return in
}

func alSine(n int, period float64) []float64 {
	in := make([]float64, n)
	for i := range in {
		in[i] = math.Sin(2 * math.Pi * float64(i) / period)
	}
	return in
}

// dftProtoLen returns the length of the prototype filter NewDFTStage designs
// for the given factor and quality (the stage keeps only the polyphase
// decomposition of it).
func dftProtoLen(t *testing.T, factor int, q Quality) int {
	t.Helper()
	coeffs, err := filter.DesignLowPassFilterAuto(
		soxrDFTStageFc/float64(factor), transitionBWFactor/float64(factor), qualityToAttenuation(q), 1.0)
	if err != nil {
		t.Fatal(err)
	}
	return len(coeffs)
}

// TestStageFilterLengthsOdd pins the assumption behind the stage alignment:
// DFT and decimation prototypes have odd length, so their group delay
// (N-1)/2 is a whole number of samples. An even length would leave half a
// sample of misalignment per stage.
func TestStageFilterLengthsOdd(t *testing.T) {
	for _, q := range alignFIRQualities {
		for factor := 2; factor <= 8; factor++ {
			if n := dftProtoLen(t, factor, q); n%2 == 0 {
				t.Errorf("DFT prototype factor=%d quality=%d has even length %d", factor, q, n)
			}
			d, err := NewDFTDecimationStage[float64](factor, q)
			if err != nil {
				t.Fatal(err)
			}
			if d.numTaps%2 == 0 {
				t.Errorf("decimation prototype factor=%d quality=%d has even length %d", factor, q, d.numTaps)
			}
		}
	}
}

func TestDFTStage_Aligned(t *testing.T) {
	const n, p = 900, 300
	skips := map[int]bool{}
	for _, q := range alignFIRQualities {
		for factor := 2; factor <= 8; factor++ {
			st, err := NewDFTStage[float64](factor, q)
			if err != nil {
				t.Fatal(err)
			}
			skips[st.skip] = true
			name := func(chunk int) string {
				return "factor=" + strconv.Itoa(factor) + " quality=" + strconv.Itoa(int(q)) + " chunk=" + strconv.Itoa(chunk)
			}
			for _, chunk := range []int{0, 1, 7, 333} {
				s, _ := NewDFTStage[float64](factor, q)
				out := alRun(t, s, alImpulse(n, p), chunk)
				if len(out) != n*factor {
					t.Errorf("%s: output length %d, want %d", name(chunk), len(out), n*factor)
				}
				if _, got := alPeak(out); math.Abs(got-float64(p*factor)) > 0.02 {
					t.Errorf("%s: impulse at %d peaks at %.3f, want %d", name(chunk), p, got, p*factor)
				}
				s, _ = NewDFTStage[float64](factor, q)
				out = alRun(t, s, alImpulse(n, 0), chunk)
				if idx, _ := alPeak(out); idx != 0 {
					t.Errorf("%s: impulse at input 0 peaks at output %d, want 0", name(chunk), idx)
				}
			}
			s, _ := NewDFTStage[float64](factor, q)
			out := alRun(t, s, alSine(4000, 400), 333)
			if d := alSineDelay(out, 400*float64(factor)); math.Abs(d) > 0.01 {
				t.Errorf("factor=%d quality=%d: sine is %.4f output samples late", factor, q, d)
			}
		}
	}
	// A future filter change must not silently drop either code path.
	if !skips[0] {
		t.Errorf("no DFT stage with skip 0 among the tested designs (skips seen: %v)", skips)
	}
	nonZero := false
	for k := range skips {
		nonZero = nonZero || k != 0
	}
	if !nonZero {
		t.Errorf("no DFT stage with a non-zero skip among the tested designs (skips seen: %v)", skips)
	}
	t.Logf("DFT stage skips covered: %v", skips)
}

func TestDFTDecimationStage_Aligned(t *testing.T) {
	const n = 3000
	for _, q := range alignFIRQualities {
		for _, factor := range []int{2, 3, 4} {
			p := 100 * factor
			for _, chunk := range []int{0, 1, 7, 333} {
				s, err := NewDFTDecimationStage[float64](factor, q)
				if err != nil {
					t.Fatal(err)
				}
				out := alRun(t, s, alImpulse(n, p), chunk)
				if want := (n + factor - 1) / factor; len(out) != want {
					t.Errorf("factor=%d quality=%d chunk=%d: output length %d, want %d", factor, q, chunk, len(out), want)
				}
				if _, got := alPeak(out); math.Abs(got-float64(p/factor)) > 0.02 {
					t.Errorf("factor=%d quality=%d chunk=%d: impulse at %d peaks at %.3f, want %d", factor, q, chunk, p, got, p/factor)
				}
				s, _ = NewDFTDecimationStage[float64](factor, q)
				out = alRun(t, s, alImpulse(n, 0), chunk)
				if idx, _ := alPeak(out); idx != 0 {
					t.Errorf("factor=%d quality=%d chunk=%d: impulse at input 0 peaks at output %d, want 0", factor, q, chunk, idx)
				}
			}
			s, _ := NewDFTDecimationStage[float64](factor, q)
			out := alRun(t, s, alSine(6000, 400), 333)
			if d := alSineDelay(out, 400/float64(factor)); math.Abs(d) > 0.01 {
				t.Errorf("factor=%d quality=%d: sine is %.4f output samples late", factor, q, d)
			}
		}
	}
}

func TestPolyphaseStage_Aligned(t *testing.T) {
	type row struct {
		ratio, totalIO float64
		hasPre         bool
	}
	rows := []row{
		{2.0 / 3.0, 2.0 / 3.0, false},
		{48000.0 / 44100.0, 44100.0 / 48000.0, false},
		{44100.0 / 48000.0, 48000.0 / 44100.0, false},
		{16000.0 / 22050.0, 22050.0 / 16000.0, false},
		{48000.0 / 88200.0, 44100.0 / 48000.0, true},
		{3.0 / 2.0, 2.0 / 3.0, false},
	}
	parities := map[bool]bool{}
	for _, q := range alignFIRQualities {
		for _, r := range rows {
			probe, err := NewPolyphaseStage[float64](r.ratio, r.totalIO, r.hasPre, q)
			if err != nil {
				t.Fatal(err)
			}
			L, T := probe.numPhases, probe.tapsPerPhase
			parities[(L*T)%2 == 1] = true
			// Residual misalignment: half a phase unit when L*T is odd.
			tol := 0.001
			if (L*T)%2 == 1 {
				tol += 1 / (2 * float64(L))
			}
			for _, chunk := range []int{0, 1, 7, 333} {
				const n = 4000
				s, _ := NewPolyphaseStage[float64](r.ratio, r.totalIO, r.hasPre, q)
				out := alRun(t, s, alSine(n, 400), chunk)
				// Exact output count: #{m : m*step < n*L<<fracBits}.
				lim := int64(n) * int64(L) << s.phaseFracBits
				if want := int((lim + s.step - 1) / s.step); len(out) != want {
					t.Errorf("ratio=%.4f quality=%d chunk=%d: output length %d, want %d", r.ratio, q, chunk, len(out), want)
				}
				if chunk != 0 {
					continue
				}
				// Delay in stage-input samples.
				d := alSineDelay(out, 400*r.ratio) / r.ratio
				if math.Abs(d) > tol {
					t.Errorf("ratio=%.4f quality=%d (L=%d T=%d): sine is %.5f input samples late, want within %.5f", r.ratio, q, L, T, d, tol)
				}
			}
		}
	}
	if !parities[true] || !parities[false] {
		t.Errorf("polyphase rows must cover both L*T parities, got %v", parities)
	}
}

func TestStageFlush_NeverFedReturnsNothing(t *testing.T) {
	dft, err := NewDFTStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := NewDFTDecimationStage[float64](2, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	poly, err := NewPolyphaseStage[float64](48000.0/44100.0, 44100.0/48000.0, false, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	for name, s := range map[string]alStage{"DFTStage": dft, "DFTDecimationStage": dec, "PolyphaseStage": poly} {
		tail, err := s.Flush()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(tail) != 0 {
			t.Errorf("%s: Flush on a never-fed stage returned %d samples, want 0", name, len(tail))
		}
		// Still true after a full stream: Flush ends the stream and a second
		// Flush returns nothing.
		alRun(t, s, alSine(500, 100), 0)
		if tail, _ := s.Flush(); len(tail) != 0 {
			t.Errorf("%s: second Flush returned %d samples, want 0", name, len(tail))
		}
	}
}

// TestStartupDeficit_EqualsGroupDelay checks StartupDeficit() against the
// stages' group delays computed from the prototype filter lengths, not from
// the fields the implementation derives it from.
func TestStartupDeficit_EqualsGroupDelay(t *testing.T) {
	for _, q := range alignFIRQualities {
		// DFT only: 8000 -> 48000 (factor 6). Group delay (N-1)/2 output samples.
		r, err := NewResampler[float64](8000, 48000, q)
		if err != nil {
			t.Fatal(err)
		}
		want := float64((dftProtoLen(t, 6, q)-1)/2) / 6 * r.ratio
		if got := r.StartupDeficit(); math.Abs(got-want) > 1e-9 {
			t.Errorf("8000->48000 quality=%d: StartupDeficit %.6f, want %.6f", q, got, want)
		}

		// Decimation only: 48000 -> 24000, group delay (numTaps-1)/2 input samples.
		r, err = NewResampler[float64](48000, 24000, q)
		if err != nil {
			t.Fatal(err)
		}
		want = float64((r.decimationStage.numTaps-1)/2) * r.ratio
		if got := r.StartupDeficit(); math.Abs(got-want) > 1e-9 {
			t.Errorf("48000->24000 quality=%d: StartupDeficit %.6f, want %.6f", q, got, want)
		}

		// DFT x2 plus polyphase, upsampling and downsampling. The polyphase
		// group delay uses the integer-rounded (protoLen-1)/2 in phase units.
		for _, c := range []struct {
			in, out float64
			hasPre  bool
		}{{44100, 48000, true}, {48000, 44100, false}} {
			r, err = NewResampler[float64](c.in, c.out, q)
			if err != nil {
				t.Fatal(err)
			}
			ps := r.polyphaseStage
			fb, err := designPolyphaseFilter(ps.numPhases, c.out/(2*c.in), c.in/c.out, c.hasPre, q)
			if err != nil {
				t.Fatal(err)
			}
			gdInt := (fb.protoLen - 1) / 2
			preGD := float64((dftProtoLen(t, 2, q)-1)/2) / 2
			polyGD := float64(gdInt) / float64(ps.numPhases) / 2
			want = (preGD + polyGD) * r.ratio
			if got := r.StartupDeficit(); math.Abs(got-want) > 1e-9 {
				t.Errorf("%g->%g quality=%d: StartupDeficit %.6f, want %.6f", c.in, c.out, q, got, want)
			}
		}
	}
}
