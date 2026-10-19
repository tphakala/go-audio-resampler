// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

// These tests compare the Go resampler with libsoxr on DC, sine and impulse
// inputs, using the C tool in test-reference/ (build it with `task test:soxr`).
// Each Go preset is compared with the libsoxr recipe of the same name, and the
// tool echoes the recipe it ran so a stale binary built from older source fails
// instead of comparing against the wrong recipe.
//
// They pin two things: time alignment against libsoxr (output sample n
// corresponds to input time n/ratio, AGENTS.md design rule 7), asserted as a
// sub-sample lag with no search over offsets, and waveform agreement for the
// same recipe. A search over offsets would hide a misaligned output, which is
// the failure these tests exist to catch.
//
// Tolerances were measured on linux/arm64 with libsoxr 0.1.3 (the Debian
// bookworm package) for High and VeryHigh, and are tighter than the thresholds
// this file used before the tool could run. Delaying the Go output by one
// sample made every sine and impulse lag check fail with a lag of about 1.0
// output samples.

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const (
	soxrRefToolName = "test_soxr_reference"

	// soxrSignalSamples is the input length; it matches INPUT_SAMPLES in
	// test-reference/test_soxr_reference.c.
	soxrSignalSamples = 4000

	// Default comparison window: skip the startup ramp, then compare a fixed
	// span of output samples. Impulse windows are centred on the libsoxr peak.
	soxrWindowStart = 300
	soxrWindowLen   = 2000

	// A window shorter than this is too little evidence to pass.
	soxrMinWindow = 500

	// Largest allowed lag versus libsoxr, in output samples. Measured worst
	// for High and VeryHigh: sine 0.00157 (phase fit), impulse peak 0.0042. A
	// one-sample delay gives about 1.0.
	soxrMaxLag = 0.01

	// DC: measured worst max error 0.00124 (High and VeryHigh). The previous
	// limit was 0.01.
	soxrDCMaxError = 0.005

	// Sine: measured worst max error 0.000223 and correlation 1.0000000 (High
	// and VeryHigh). The previous limits were 0.15 to 0.20 and 0.95 to 0.99.
	soxrSineMaxError = 0.002
	soxrSineMinCorr  = 0.9999

	// Impulse: measured worst max error 0.0029 and correlation 0.99972 (High
	// and VeryHigh). The previous correlation limit was 0.5.
	soxrImpulseMaxError = 0.01
	soxrImpulseMinCorr  = 0.999

	// Length contract: the Go output has the same length as libsoxr or one
	// more, inside the [floor, ceil+1] bound that flush_length_test.go:45-47
	// (root package) pins.
	soxrMaxExtraSamples = 1
)

// soxrWaveformPresets are the presets compared against libsoxr: the ones
// AGENTS.md names as the libsoxr HQ and VHQ match.
var soxrWaveformPresets = func() []qualityPresetPair {
	var presets []qualityPresetPair
	for _, p := range qualityPresets {
		if p.goQuality == QualityHigh || p.goQuality == QualityVeryHigh {
			presets = append(presets, p)
		}
	}
	return presets
}()

// soxrReferenceTool returns the path of the libsoxr reference tool, resolved
// from this source file. It skips the test when the tool has not been built;
// any other stat error fails, so a broken checkout is not hidden.
func soxrReferenceTool(t *testing.T) string {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine the path of soxr_comparison_test.go")
	}
	path := filepath.Join(filepath.Dir(thisFile), "..", "..", "test-reference", soxrRefToolName)

	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("libsoxr reference tool not built at %s; run `task test:soxr` (needs libsoxr and fftw3)", path)
		}
		t.Fatalf("stat %s: %v", path, err)
	}
	return path
}

// getSoxrReference runs the tool and returns its output samples. It fails when
// the tool cannot run, prints an unparsable sample line, or reports a recipe
// other than the requested one.
func getSoxrReference(ctx context.Context, tool string, inputRate, outputRate float64, signalType string, frequency float64, quality string) ([]float64, error) {
	cmd := exec.CommandContext(ctx, tool,
		strconv.FormatFloat(inputRate, 'f', 0, 64),
		strconv.FormatFloat(outputRate, 'f', 0, 64),
		signalType,
		strconv.FormatFloat(frequency, 'f', 0, 64),
		quality)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%s failed: %w: %s", tool, err, strings.TrimSpace(stderr.String()))
	}

	var samples []float64
	echoed := ""
	scanner := bufio.NewScanner(bytes.NewReader(output))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if rest, ok := strings.CutPrefix(line, "# quality:"); ok {
			echoed = strings.TrimSpace(rest)
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		val, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("unparsable sample line %q: %w", line, err)
		}
		samples = append(samples, val)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading %s output: %w", tool, err)
	}
	if echoed != quality {
		return nil, fmt.Errorf("%s reported quality %q, want %q; rebuild it with `make -C test-reference`", tool, echoed, quality)
	}
	return samples, nil
}

// generateSoxrTestSignal creates the same input as test_soxr_reference.c.
func generateSoxrTestSignal(signalType string, sampleRate, frequency float64) []float64 {
	signal := make([]float64, soxrSignalSamples)

	switch signalType {
	case "dc":
		for i := range signal {
			signal[i] = 1.0
		}
	case "sine":
		for i := range signal {
			signal[i] = math.Sin(2.0 * math.Pi * frequency * float64(i) / sampleRate)
		}
	case "impulse":
		signal[soxrSignalSamples/2] = 1.0
	}

	return signal
}

// resampleSoxrSignal runs the Go resampler on the test signal, then flushes.
func resampleSoxrSignal(t *testing.T, quality Quality, inputRate, outputRate float64, signalType string, frequency float64) []float64 {
	t.Helper()

	r, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		t.Fatalf("NewResampler failed: %v", err)
	}
	output, err := r.Process(generateSoxrTestSignal(signalType, inputRate, frequency))
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}
	tail, err := r.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	return append(output, tail...)
}

// soxrReference returns the libsoxr output for one case, failing the test when
// the tool misbehaves.
func soxrReference(t *testing.T, tool string, inputRate, outputRate float64, signalType string, frequency float64, quality string) []float64 {
	t.Helper()

	want, err := getSoxrReference(t.Context(), tool, inputRate, outputRate, signalType, frequency, quality)
	if err != nil {
		t.Fatal(err)
	}
	return want
}

// checkSoxrLength fails when the Go output is shorter than libsoxr's or more
// than soxrMaxExtraSamples longer.
func checkSoxrLength(t *testing.T, got, want []float64) {
	t.Helper()

	if extra := len(got) - len(want); extra < 0 || extra > soxrMaxExtraSamples {
		t.Errorf("output length %d, libsoxr %d: want between 0 and %d extra samples", len(got), len(want), soxrMaxExtraSamples)
	}
}

type soxrStats struct {
	maxErr float64
	rmsErr float64
	corr   float64 // Pearson correlation; 0 when either window is constant
}

// compareAligned compares got and want sample by sample at offset 0 over
// [start, start+soxrWindowLen), clipped to both lengths. It fails when fewer than
// soxrMinWindow samples remain, or when the window holds a NaN or infinite
// sample: the error limits compare with ">", which is false for NaN.
func compareAligned(got, want []float64, start int) (soxrStats, error) {
	n := min(soxrWindowLen, len(got)-start, len(want)-start)
	if start < 0 || n < soxrMinWindow {
		return soxrStats{}, fmt.Errorf("window of %d samples at %d is too short (got %d samples, libsoxr %d)", n, start, len(got), len(want))
	}
	for i, g := range got[start : start+n] {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			return soxrStats{}, fmt.Errorf("Go output sample %d is %v", start+i, g)
		}
	}

	var stats soxrStats
	var sumG, sumW, sumGW, sumGG, sumWW, sumSqErr float64
	for i := start; i < start+n; i++ {
		g, w := got[i], want[i]
		e := math.Abs(g - w)
		stats.maxErr = max(stats.maxErr, e)
		sumSqErr += e * e
		sumG += g
		sumW += w
		sumGW += g * w
		sumGG += g * g
		sumWW += w * w
	}

	fn := float64(n)
	stats.rmsErr = math.Sqrt(sumSqErr / fn)
	devG := math.Sqrt(sumGG - sumG*sumG/fn)
	devW := math.Sqrt(sumWW - sumW*sumW/fn)
	if devG > 0 && devW > 0 {
		stats.corr = (sumGW - sumG*sumW/fn) / (devG * devW)
	}
	return stats, nil
}

// sineLag returns how many output samples got trails want (positive means got
// is late), from the phase of a single-frequency fit of freq over [start,
// start+n), clipped to both lengths. freq and rate are in Hz; rate is the
// output rate.
func sineLag(got, want []float64, freq, rate float64, start, n int) float64 {
	n = min(n, len(got)-start, len(want)-start)
	phase := func(x []float64) float64 {
		var cosSum, sinSum float64
		for i := start; i < start+n; i++ {
			w := 2 * math.Pi * freq * float64(i) / rate
			cosSum += x[i] * math.Cos(w)
			sinSum += x[i] * math.Sin(w)
		}
		return math.Atan2(cosSum, sinSum)
	}
	d := math.Remainder(phase(got)-phase(want), 2*math.Pi)
	return -d / (2 * math.Pi * freq / rate)
}

// peakPos returns the position of the largest-magnitude sample, refined to a
// fraction of a sample by a parabola through its two neighbours.
func peakPos(x []float64) float64 {
	k := 0
	for i := range x {
		if math.Abs(x[i]) > math.Abs(x[k]) {
			k = i
		}
	}
	if k == 0 || k == len(x)-1 {
		return float64(k)
	}
	y0, y1, y2 := x[k-1], x[k], x[k+1]
	den := y0 - 2*y1 + y2
	if den == 0 {
		return float64(k)
	}
	return float64(k) + 0.5*(y0-y2)/den
}

func TestSoxrComparison_DC(t *testing.T) {
	tool := soxrReferenceTool(t)

	cases := []struct{ inputRate, outputRate float64 }{
		{44100, 48000}, // CD to DAT
		{48000, 44100}, // DAT to CD
		{44100, 96000}, // 2x upsample
		{96000, 48000}, // 2x downsample
	}

	for _, p := range soxrWaveformPresets {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/%.0f_to_%.0f", p.name, tc.inputRate, tc.outputRate), func(t *testing.T) {
				want := soxrReference(t, tool, tc.inputRate, tc.outputRate, "dc", 0, p.soxrQuality)
				got := resampleSoxrSignal(t, p.goQuality, tc.inputRate, tc.outputRate, "dc", 0)
				checkSoxrLength(t, got, want)

				stats, err := compareAligned(got, want, soxrWindowStart)
				if err != nil {
					t.Fatal(err)
				}
				// No correlation for DC: it is undefined for constant windows.
				t.Logf("samples %d/%d max error %.6f rms error %.6f", len(got), len(want), stats.maxErr, stats.rmsErr)

				if stats.maxErr > soxrDCMaxError {
					t.Errorf("DC max error %.6f exceeds %.6f", stats.maxErr, soxrDCMaxError)
				}
			})
		}
	}
}

func TestSoxrComparison_Sine(t *testing.T) {
	tool := soxrReferenceTool(t)

	cases := []struct{ inputRate, outputRate, frequency float64 }{
		{44100, 48000, 1000},
		{44100, 48000, 10000},
		{48000, 44100, 1000},
		{44100, 96000, 5000},
	}

	for _, p := range soxrWaveformPresets {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/%.0fHz_%.0f_to_%.0f", p.name, tc.frequency, tc.inputRate, tc.outputRate), func(t *testing.T) {
				want := soxrReference(t, tool, tc.inputRate, tc.outputRate, "sine", tc.frequency, p.soxrQuality)
				got := resampleSoxrSignal(t, p.goQuality, tc.inputRate, tc.outputRate, "sine", tc.frequency)
				checkSoxrLength(t, got, want)

				stats, err := compareAligned(got, want, soxrWindowStart)
				if err != nil {
					t.Fatal(err)
				}
				lag := sineLag(got, want, tc.frequency, tc.outputRate, soxrWindowStart, min(soxrWindowLen, len(want)-soxrWindowStart))
				t.Logf("samples %d/%d lag %+.5f max error %.6f rms error %.6f correlation %.7f",
					len(got), len(want), lag, stats.maxErr, stats.rmsErr, stats.corr)

				if math.Abs(lag) > soxrMaxLag {
					t.Errorf("lag %+.5f output samples, want |lag| <= %.3f", lag, soxrMaxLag)
				}
				if stats.maxErr > soxrSineMaxError {
					t.Errorf("max error %.6f exceeds %.6f", stats.maxErr, soxrSineMaxError)
				}
				if stats.corr < soxrSineMinCorr {
					t.Errorf("correlation %.7f below %.4f", stats.corr, soxrSineMinCorr)
				}
			})
		}
	}
}

// TestSoxrComparison_Impulse pins the alignment of the impulse peak against
// libsoxr, which is the non-periodic evidence for time alignment, and the shape
// of the impulse response around it. The other cases add a larger upsampling
// ratio, integer decimation and a 3:2 downsampling ratio.
func TestSoxrComparison_Impulse(t *testing.T) {
	tool := soxrReferenceTool(t)

	cases := []struct{ inputRate, outputRate float64 }{
		{44100, 48000},
		{48000, 44100},
		{44100, 96000},
		{96000, 48000},
		{48000, 32000},
	}

	for _, p := range soxrWaveformPresets {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/%.0f_to_%.0f", p.name, tc.inputRate, tc.outputRate), func(t *testing.T) {
				want := soxrReference(t, tool, tc.inputRate, tc.outputRate, "impulse", 0, p.soxrQuality)
				got := resampleSoxrSignal(t, p.goQuality, tc.inputRate, tc.outputRate, "impulse", 0)
				checkSoxrLength(t, got, want)

				gotPeak, wantPeak := peakPos(got), peakPos(want)
				// Centre the window on the libsoxr peak: its position depends on
				// the ratio, and a fixed window can miss it.
				start := max(0, int(wantPeak)-soxrWindowLen/2)
				stats, err := compareAligned(got, want, start)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("samples %d/%d peak %.4f/%.4f max error %.6f correlation %.7f",
					len(got), len(want), gotPeak, wantPeak, stats.maxErr, stats.corr)

				if lag := gotPeak - wantPeak; math.Abs(lag) > soxrMaxLag {
					t.Errorf("peak lag %+.5f output samples, want |lag| <= %.3f", lag, soxrMaxLag)
				}
				if stats.maxErr > soxrImpulseMaxError {
					t.Errorf("max error %.6f exceeds %.6f", stats.maxErr, soxrImpulseMaxError)
				}
				if stats.corr < soxrImpulseMinCorr {
					t.Errorf("correlation %.7f below %.4f", stats.corr, soxrImpulseMinCorr)
				}
			})
		}
	}
}

// TestSoxrCompareHelpers_RejectBadOutput pins that the comparison helpers fail
// on a broken Go output instead of passing or panicking, without needing the
// libsoxr tool.
func TestSoxrCompareHelpers_RejectBadOutput(t *testing.T) {
	const n = soxrWindowStart + soxrWindowLen
	want := make([]float64, n)
	for i := range want {
		want[i] = math.Sin(2 * math.Pi * 1000 * float64(i) / 48000)
	}

	t.Run("NonFinite", func(t *testing.T) {
		for _, bad := range []float64{math.NaN(), math.Inf(1)} {
			got := slices.Clone(want)
			got[soxrWindowStart+1] = bad
			if _, err := compareAligned(got, want, soxrWindowStart); err == nil {
				t.Errorf("compareAligned accepted a window containing %v", bad)
			}
		}
	})

	t.Run("ShortOutput", func(t *testing.T) {
		// A Go output shorter than libsoxr's but long enough for
		// compareAligned must not make sineLag index past its end.
		got := want[:soxrWindowStart+soxrMinWindow]
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("sineLag panicked on a short output: %v", r)
			}
		}()
		sineLag(got, want, 1000, 48000, soxrWindowStart, soxrWindowLen)
	})
}
