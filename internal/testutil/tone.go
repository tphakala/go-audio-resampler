// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package testutil

import "math"

// Blackman-Harris 4-term window coefficients (minimum sidelobe, about -92 dB
// first sidelobe and fast decay, so a single tone's leakage stays far below
// the stopband levels this oracle is used to measure).
const (
	bhA0 = 0.35875
	bhA1 = 0.48829
	bhA2 = 0.14128
	bhA3 = 0.01168

	// toneMaxSegment is the longest segment analysed, in samples.
	toneMaxSegment = 16384
	// toneEdgeMargin is the number of samples skipped at each end of a short
	// output so start-up and tail transients stay out of the measurement.
	toneEdgeMargin = 1000

	// toneMinSegment is the shortest segment worth analysing.
	toneMinSegment = 2
	// toneAmpFloor keeps the logarithm finite for an exactly zero component.
	toneAmpFloor = 1e-300
	// dbPerDecade is the dB scale factor for an amplitude ratio.
	dbPerDecade = 20
	// toneBothEnds is the number of edges (start and end) trimmed from a short
	// output.
	toneBothEnds = 2
	// oneSidedGain converts a single-bin DFT magnitude to a sine amplitude.
	oneSidedGain = 2
	// bhSecondHarmonic and bhThirdHarmonic are the cosine orders of the window.
	bhSecondHarmonic = 2
	bhThirdHarmonic  = 3
	// halfSegment splits the unused samples evenly around the segment.
	halfSegment = 2
)

// Sine returns n samples of a unit-amplitude sine of frequency f at sample
// rate fs.
func Sine(fs, f float64, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2 * math.Pi * f * float64(i) / fs)
	}
	return x
}

// FoldFrequency returns the frequency a tone of f Hz appears at after being
// sampled at rate fs without prior band-limiting.
func FoldFrequency(f, fs float64) float64 {
	return math.Abs(fs*math.Round(f/fs) - f)
}

// ToneLevelDB returns the level in dB (re an amplitude of 1) of the component
// at frequency f in y, which is sampled at fs. It windows the middle segment
// of y (at most 16384 samples, fewer for short outputs) with a 4-term
// Blackman-Harris window and evaluates a single DFT bin (Goertzel style) at f.
// A pure tone elsewhere in the spectrum reads far below -200 dB in float64,
// so the result is the leakage-free level of whatever sits at f. y must hold
// more than 2*1000 samples.
func ToneLevelDB(y []float64, fs, f float64) float64 {
	n := toneMaxSegment
	if len(y) < n+toneBothEnds*toneEdgeMargin {
		n = len(y) - toneBothEnds*toneEdgeMargin
	}
	if n < toneMinSegment {
		return math.Inf(-1)
	}
	start := (len(y) - n) / halfSegment
	var re, im, ws float64
	for i := range n {
		x := 2 * math.Pi * float64(i) / float64(n-1)
		w := bhA0 - bhA1*math.Cos(x) + bhA2*math.Cos(bhSecondHarmonic*x) - bhA3*math.Cos(bhThirdHarmonic*x)
		ws += w
		ph := 2 * math.Pi * f * float64(i) / fs
		v := y[start+i] * w
		re += v * math.Cos(ph)
		im -= v * math.Sin(ph)
	}
	amp := oneSidedGain * math.Hypot(re, im) / ws
	return dbPerDecade * math.Log10(amp+toneAmpFloor)
}
