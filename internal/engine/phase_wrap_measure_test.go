// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"math"
	"math/cmplx"
	"testing"
)

// Phase-wrap coefficient indexing investigation (measurement first).
//
// NewPolyphaseStage builds the cubic sub-phase interpolation banks (B/C/D) by
// sampling the prototype at the current phase and its neighbours (getCoeff in
// polyphase_stage.go). Two reviews argued that the original code, which wrapped
// the neighbour phase WITHIN the same tap (phase % numPhases), picked a
// prototype sample numPhases-1 positions away at each phase boundary instead of
// the adjacent one. The mathematically adjacent sample is the neighbour in the
// FLAT prototype: the right neighbour of (tap t, phase L-1) is prototype[t*L +
// L] (phase 0 of tap t+1), and the left neighbour of (tap t, phase 0) is
// prototype[t*L - 1] (phase L-1 of tap t-1); out-of-range positions clamp to
// 0.0.
//
// This test measured both indexings on the canonical 997 Hz full-scale sine and
// drove the decision to switch getCoeff to flat indexing. It is retained as the
// record of that measurement and as a regression guard: if getCoeff is ever
// reverted to the wrap, the bank-equality check and the active-ratio THD+N
// assertion below both fail.
//
// KEY FINDING: the fixed-point sub-phase accumulator only produces a non-zero
// fractional part x (which is what the B/C/D banks interpolate over) when the
// resampling ratio is NOT an exact rational the phase count L divides cleanly.
// For 44100<->48000 the polyphase stage ratio is exactly 80/147, so the step is
// an integer number of phase units, x is identically 0 for every output sample,
// and the B/C/D banks (hence the disputed wrap) are never consulted. That is why
// the wrap error was invisible to the entire existing regression suite, whose
// ratios (44100<->48000, 48000->32000, 48000->96000) are all exact-rational or
// integer/DFT-only. The measurement therefore uses TWO probes: the degenerate
// exact ratio (to document the x==0 masking) and an active-interpolation ratio
// (to actually reach the code under test).
//
// The 44100 -> N QualityHigh pipeline is a 2x DFT pre-stage (44100 -> 88200)
// followed by the polyphase stage, so the measurement drives that stage in
// isolation with a clean full-scale sine at the intermediate 88200 Hz rate: a
// pure tone is the ideal probe for coefficient-interpolation distortion, and
// isolating the stage keeps everything identical between runs except the
// boundary indexing.

const (
	phaseWrapPreUpsample = 2
	phaseWrapTestFreq    = 997.0
	phaseWrapAmplitude   = 1.0    // full-scale (0 dBFS)
	phaseWrapNumSamples  = 131072 // input samples at the intermediate 88200 Hz rate
	phaseWrapFFTSize     = 32768
	phaseWrapFFTOffset   = 8192 // skip the filter start-up transient
	phaseWrapFundGuard   = 12   // bins around the fundamental treated as signal
	phaseWrapDCGuard     = 3    // low bins excluded as window DC leakage

	// Regression bounds on the active-interpolation probe (44100 -> 64000).
	// MEASURED on this machine (float64, values below): the production flat path
	// yields about -140.7 dB THD+N and the old wrap about -54.5 dB, an improvement
	// of about 86 dB. The bounds leave generous headroom so the guard fires only on
	// a genuine regression (e.g. reverting getCoeff to the wrap), not on minor
	// numeric drift.
	phaseWrapFlatCeilingDB    = -120.0 // production (flat) THD+N must be at least this clean
	phaseWrapImprovementFloor = 40.0   // flat must beat the old wrap by at least this much
	phaseWrapPlausibleFloorDB = -160.0 // below this, the harness is broken, not the library
	phaseWrapPlausibleCeilDB  = -40.0  // above this for flat, the harness is broken
)

// phaseWrapResult holds the outcome of a single ratio measurement.
type phaseWrapResult struct {
	flatTHDN   float64 // THD+N of the production (flat) path
	wrapTHDN   float64 // THD+N of the old wrap path
	sampleDiff int     // number of output samples that differ between the two
	total      int     // total compared samples
	numPhases  int
}

// buildPhaseInterpBanks reproduces the cubic interpolation bank construction
// from NewPolyphaseStage exactly, parameterised only by the boundary indexing.
// With flat=true it reproduces the production (fixed) indexing; with flat=false
// it reproduces the original wrap. Everything else, the Catmull-Rom cubic math
// and the reversed tap storage, is identical, so any output difference is
// attributable solely to the boundary indexing.
func buildPhaseInterpBanks(fb *polyphaseFilter, numPhases int, flat bool) (a, b, c, d [][]float64) {
	tapsPerPhase := fb.tapsPerPhase
	coeffs := fb.coeffs

	getCoeff := func(phase, tap int) float64 {
		var idx int
		if flat {
			// Flat-prototype-spill: neighbour of a phase boundary is the adjacent
			// prototype sample, crossing into the next/previous tap. Matches the
			// production getCoeff after the fix.
			idx = tap*numPhases + phase
		} else {
			// Original wrap: keep the neighbour within the same tap.
			wrappedPhase := phase % numPhases
			if wrappedPhase < 0 {
				wrappedPhase += numPhases
			}
			idx = tap*numPhases + wrappedPhase
		}
		if idx < 0 || idx >= len(coeffs) {
			return 0.0
		}
		return coeffs[idx]
	}

	a = make([][]float64, numPhases)
	b = make([][]float64, numPhases)
	c = make([][]float64, numPhases)
	d = make([][]float64, numPhases)

	for phase := range numPhases {
		a[phase] = make([]float64, tapsPerPhase)
		b[phase] = make([]float64, tapsPerPhase)
		c[phase] = make([]float64, tapsPerPhase)
		d[phase] = make([]float64, tapsPerPhase)

		for tap := range tapsPerPhase {
			f0 := getCoeff(phase, tap)
			f1 := getCoeff(phase+1, tap)
			fm1 := getCoeff(phase-1, tap)
			f2 := getCoeff(phase+cubicPhaseOffset, tap)

			av := f0
			cv := cubicCenterCoeff*(f1+fm1) - f0
			dv := (1.0 / cubicDivisor) * (f2 - f1 + fm1 - f0 - cubicCMultiplier*cv)
			bv := f1 - f0 - dv - cv

			revTap := tapsPerPhase - 1 - tap
			a[phase][revTap] = av
			b[phase][revTap] = bv
			c[phase][revTap] = cv
			d[phase][revTap] = dv
		}
	}
	return a, b, c, d
}

// processAllPhaseWrap runs Process followed by the terminal Flush and returns
// the full output stream.
func processAllPhaseWrap(t *testing.T, s *PolyphaseStage[float64], input []float64) []float64 {
	t.Helper()
	out, err := s.Process(input)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}
	flush, err := s.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	return append(out, flush...)
}

// blackmanHarris7 returns a 7-term Blackman-Harris window of length n. Its peak
// sidelobe is about -180 dB, far below the THD+N floor of interest, so spectral
// leakage from the non-bin-aligned 997 Hz tone does not mask the measurement.
func blackmanHarris7(n int) []float64 {
	const (
		a0 = 0.27105140069342
		a1 = 0.43329793923448
		a2 = 0.21812299954311
		a3 = 0.06592544638803
		a4 = 0.01081174209837
		a5 = 0.00077658482522
		a6 = 0.00001388721735
	)
	w := make([]float64, n)
	denom := float64(n - 1)
	for i := range w {
		t := 2.0 * math.Pi * float64(i) / denom
		w[i] = a0 - a1*math.Cos(t) + a2*math.Cos(2*t) - a3*math.Cos(3*t) +
			a4*math.Cos(4*t) - a5*math.Cos(5*t) + a6*math.Cos(6*t)
	}
	return w
}

// measureTHDNPhaseWrap computes THD+N in dB for a resampled tone. It notches the
// fundamental (and its window main lobe) plus the DC region, then ratios all
// remaining spectral power (harmonics, images, noise) against the fundamental
// power. A more negative figure is better.
func measureTHDNPhaseWrap(t *testing.T, output []float64, testFreq, sampleRate float64) float64 {
	t.Helper()
	if len(output) < phaseWrapFFTOffset+phaseWrapFFTSize {
		t.Fatalf("output too short for THD+N: have %d, need %d", len(output), phaseWrapFFTOffset+phaseWrapFFTSize)
	}

	win := blackmanHarris7(phaseWrapFFTSize)
	fftIn := make([]complex128, phaseWrapFFTSize)
	for i := range phaseWrapFFTSize {
		fftIn[i] = complex(output[phaseWrapFFTOffset+i]*win[i], 0)
	}
	fftOut := fft(fftIn)

	half := phaseWrapFFTSize / 2
	fundBin := int(math.Round(testFreq / sampleRate * float64(phaseWrapFFTSize)))

	var signalPower, totalPower float64
	for bin := phaseWrapDCGuard + 1; bin < half; bin++ {
		mag := cmplx.Abs(fftOut[bin])
		p := mag * mag
		totalPower += p
		if bin >= fundBin-phaseWrapFundGuard && bin <= fundBin+phaseWrapFundGuard {
			signalPower += p
		}
	}

	noiseDistPower := totalPower - signalPower
	if signalPower <= 0 {
		t.Fatalf("no signal power measured (fundBin=%d)", fundBin)
	}
	return 10.0 * math.Log10(noiseDistPower/signalPower+1e-30)
}

// measurePhaseWrap resamples a full-scale 997 Hz sine through the polyphase
// stage for inputRate -> outputRate (QualityHigh) with both the production flat
// indexing and the old wrap, and returns the THD+N of each plus how many output
// samples differ. It also asserts that the production stage's banks match the
// flat reconstruction, proving the fix is in place.
func measurePhaseWrap(t *testing.T, inputRate, outputRate float64) phaseWrapResult {
	t.Helper()

	intermediate := inputRate * phaseWrapPreUpsample // rate the polyphase stage sees
	polyphaseRatio := outputRate / intermediate
	totalIORatio := inputRate / outputRate
	const hasPreStage = true // 44100 -> N here is non-integer upsampling: 2x DFT pre-stage

	numPhases, _ := findRationalApprox(polyphaseRatio)

	fb, err := designPolyphaseFilter(numPhases, polyphaseRatio, totalIORatio, hasPreStage, QualityHigh)
	if err != nil {
		t.Fatalf("designPolyphaseFilter failed: %v", err)
	}

	flatA, flatB, flatC, flatD := buildPhaseInterpBanks(fb, numPhases, true)
	wrapA, wrapB, wrapC, wrapD := buildPhaseInterpBanks(fb, numPhases, false)

	// Production stage uses the fixed flat indexing; verify its banks match the
	// flat reconstruction. If getCoeff is ever reverted to the wrap this fails.
	flatStage, err := NewPolyphaseStage[float64](polyphaseRatio, totalIORatio, hasPreStage, QualityHigh)
	if err != nil {
		t.Fatalf("NewPolyphaseStage (flat) failed: %v", err)
	}
	assertBanksEqual(t, "polyCoeffs", flatStage.polyCoeffs, flatA)
	assertBanksEqual(t, "polyCoeffsB", flatStage.polyCoeffsB, flatB)
	assertBanksEqual(t, "polyCoeffsC", flatStage.polyCoeffsC, flatC)
	assertBanksEqual(t, "polyCoeffsD", flatStage.polyCoeffsD, flatD)

	// Second stage identical to the first except its banks are overwritten with
	// the old wrap variant.
	wrapStage, err := NewPolyphaseStage[float64](polyphaseRatio, totalIORatio, hasPreStage, QualityHigh)
	if err != nil {
		t.Fatalf("NewPolyphaseStage (wrap) failed: %v", err)
	}
	wrapStage.polyCoeffs = wrapA
	wrapStage.polyCoeffsB = wrapB
	wrapStage.polyCoeffsC = wrapC
	wrapStage.polyCoeffsD = wrapD

	// Full-scale 997 Hz sine sampled at the intermediate rate.
	input := make([]float64, phaseWrapNumSamples)
	for i := range input {
		input[i] = phaseWrapAmplitude * math.Sin(2.0*math.Pi*phaseWrapTestFreq*float64(i)/intermediate)
	}

	flatOut := processAllPhaseWrap(t, flatStage, input)
	wrapOut := processAllPhaseWrap(t, wrapStage, input)

	n := min(len(flatOut), len(wrapOut))
	sampleDiff := 0
	for i := range n {
		if flatOut[i] != wrapOut[i] {
			sampleDiff++
		}
	}

	return phaseWrapResult{
		flatTHDN:   measureTHDNPhaseWrap(t, flatOut, phaseWrapTestFreq, outputRate),
		wrapTHDN:   measureTHDNPhaseWrap(t, wrapOut, phaseWrapTestFreq, outputRate),
		sampleDiff: sampleDiff,
		total:      n,
		numPhases:  numPhases,
	}
}

// TestPhaseWrapCoeffIndexing_Measure records the measurement that drove the
// getCoeff wrap -> flat fix and guards against a regression.
//
// MEASURED RESULT (this machine, float64):
//
//	Probe 1, 44100->48000 (exact 80/147, sub-phase x == 0):
//	  wrap THD+N = -140.72 dB, flat THD+N = -140.72 dB, improvement = 0.00 dB
//	  0 / 71332 output samples differ (bit-identical: the B/C/D banks are never
//	  consulted, so the wrap error cannot manifest at this ratio).
//
//	Probe 2, 44100->64000 (active sub-phase interpolation, L = 201):
//	  wrap THD+N = -54.46 dB, flat THD+N = -140.72 dB, improvement = 86.26 dB
//	  1486 / 95109 output samples differ.
//
// The flat path (~-140.7 dB) matches soxr's HQ-class THD+N; the wrap path
// (-54 dB) is wildly worse, confirming the wrap was a genuine defect rather than
// a harness artifact. The decision to apply the fix follows: flat improves
// THD+N by far more than the 1 dB plan threshold wherever the disputed code is
// reachable. The 44100->48000 probe reads 0 dB only because that ratio's
// sub-phase interpolation is inert (x == 0), which is exactly why the bug hid.
func TestPhaseWrapCoeffIndexing_Measure(t *testing.T) {
	// Probe 1: degenerate exact ratio. Documents the x == 0 masking.
	deg := measurePhaseWrap(t, 44100, 48000)
	if deg.numPhases != 80 {
		t.Fatalf("expected 80 phases for 48000/88200, got %d", deg.numPhases)
	}
	t.Logf("44100->48000 (exact 80/147, sub-phase x==0): wrap THD+N=%.2f dB, flat THD+N=%.2f dB, improvement=%.2f dB; %d/%d samples differ",
		deg.wrapTHDN, deg.flatTHDN, deg.wrapTHDN-deg.flatTHDN, deg.sampleDiff, deg.total)
	if deg.sampleDiff != 0 {
		t.Errorf("expected bit-identical wrap/flat output at exact ratio 44100->48000 (x==0), got %d differing samples", deg.sampleDiff)
	}

	// Probe 2: active sub-phase interpolation. Reaches the disputed code.
	act := measurePhaseWrap(t, 44100, 64000)
	improvement := act.wrapTHDN - act.flatTHDN
	t.Logf("44100->64000 (active sub-phase interp, L=%d): wrap THD+N=%.2f dB, flat THD+N=%.2f dB, improvement=%.2f dB; %d/%d samples differ",
		act.numPhases, act.wrapTHDN, act.flatTHDN, improvement, act.sampleDiff, act.total)

	if act.sampleDiff == 0 {
		t.Fatalf("expected wrap/flat output to differ at active ratio 44100->64000; harness not exercising the code under test")
	}

	// Sanity: the production flat THD+N must be in a plausible HQ range. A
	// wildly-off figure means the harness is broken, not that the library is that
	// good or bad.
	if act.flatTHDN < phaseWrapPlausibleFloorDB || act.flatTHDN > phaseWrapPlausibleCeilDB {
		t.Fatalf("flat THD+N %.2f dB implausible; measurement harness likely broken", act.flatTHDN)
	}

	// Regression: the production (flat) path must stay clean.
	if act.flatTHDN > phaseWrapFlatCeilingDB {
		t.Errorf("flat THD+N regression: got %.2f dB, want <= %.2f dB", act.flatTHDN, phaseWrapFlatCeilingDB)
	}

	// Regression: the fix must keep beating the old wrap by a wide margin. If this
	// shrinks, getCoeff may have been reverted to the wrap.
	if improvement < phaseWrapImprovementFloor {
		t.Errorf("flat improves THD+N by only %.2f dB over the old wrap, want >= %.2f dB; "+
			"getCoeff may have regressed to the phase wrap", improvement, phaseWrapImprovementFloor)
	}
}

// assertBanksEqual fails the test if two coefficient banks differ, proving the
// production stage's banks match the flat reconstruction exactly.
func assertBanksEqual(t *testing.T, name string, got, want [][]float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: phase count mismatch: got %d want %d", name, len(got), len(want))
	}
	for phase := range want {
		if len(got[phase]) != len(want[phase]) {
			t.Fatalf("%s: phase %d tap count mismatch: got %d want %d", name, phase, len(got[phase]), len(want[phase]))
		}
		for tap := range want[phase] {
			if got[phase][tap] != want[phase][tap] {
				t.Fatalf("%s: mismatch at phase %d tap %d: got %v want %v",
					name, phase, tap, got[phase][tap], want[phase][tap])
			}
		}
	}
}
