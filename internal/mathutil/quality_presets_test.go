package mathutil

// This file contains tests for quality preset parameters against soxr reference values.

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// soxr quality preset parameters
// Based on soxr source code analysis (cr.c, quality-spec.h)
//
// Quality presets are defined by:
// - Precision (bits): determines stopband attenuation = (bits + 1) * 6.0206 dB
// - Passband end (Fp): normalized frequency where passband ends
// - Stopband begin (Fs): normalized frequency where stopband begins
// - Transition bandwidth (tr_bw): 0.5 * (Fs - Fp)
//
// soxr quality levels correspond to:
// - SOXR_QQ (Quick): 8-bit, Fp=0.875, Fs=1.0
// - SOXR_LQ (Low): 16-bit, Fp=0.875, Fs=1.0
// - SOXR_MQ (Medium): 16-bit, Fp=0.891, Fs=1.0 (tighter transition)
// - SOXR_HQ (High): 20-bit, Fp=0.913, Fs=1.0 (default)
// - SOXR_VHQ (VeryHigh): 28-bit, Fp=0.913, Fs=1.0

// QualityPreset holds parameters for a quality level
type QualityPreset struct {
	Name        string
	Bits        int     // Precision bits
	Attenuation float64 // (bits + 1) * 6.0206 dB
	Fp          float64 // Passband end (normalized to Nyquist)
	Fs          float64 // Stopband begin (normalized to Nyquist)
	TrBw        float64 // Transition bandwidth = 0.5 * (Fs - Fp)
}

// soxr quality presets
// Values verified from soxr analysis session (quality-spec.h, lsx_to_3dB)
var soxrQualityPresets = []QualityPreset{
	{
		Name:        "Quick",
		Bits:        8,
		Attenuation: 9 * 6.0206, // 54.19 dB
		Fp:          0.67625,    // Quick uses same Fp as Low (from quality-spec.h)
		Fs:          1.0,
		TrBw:        0.161875, // 0.5 * (1.0 - 0.67625)
	},
	{
		Name:        "Low",
		Bits:        16,
		Attenuation: 17 * 6.0206, // 102.35 dB
		Fp:          0.67625,     // Fixed value from soxr quality-spec.h
		Fs:          1.0,
		TrBw:        0.161875, // 0.5 * (1.0 - 0.67625)
	},
	{
		Name:        "Medium",
		Bits:        16,
		Attenuation: 17 * 6.0206, // 102.35 dB
		Fp:          0.91,        // From lsx_to_3dB calculation
		Fs:          1.0,
		TrBw:        0.045, // 0.5 * (1.0 - 0.91)
	},
	{
		Name:        "High",
		Bits:        20,
		Attenuation: 21 * 6.0206, // 126.43 dB
		Fp:          0.912,       // From lsx_to_3dB calculation
		Fs:          1.0,
		TrBw:        0.044, // 0.5 * (1.0 - 0.912)
	},
	{
		Name:        "VeryHigh",
		Bits:        28,
		Attenuation: 29 * 6.0206, // 174.60 dB
		Fp:          0.913,       // From lsx_to_3dB calculation
		Fs:          1.0,
		TrBw:        0.0435, // 0.5 * (1.0 - 0.913)
	},
}

// TestQualityPreset_Attenuation validates attenuation formula for all presets.
func TestQualityPreset_Attenuation(t *testing.T) {
	const dbPerBit = 6.0206 // 20 * log10(2)

	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			expectedAtt := float64(preset.Bits+1) * dbPerBit

			assert.InDelta(t, expectedAtt, preset.Attenuation, 0.01,
				"%s: attenuation should be (bits+1)*6.0206", preset.Name)

			t.Logf("%s: %d bits -> %.2f dB attenuation", preset.Name, preset.Bits, expectedAtt)
		})
	}
}

// TestQualityPreset_TransitionBandwidth validates transition bandwidth calculation.
func TestQualityPreset_TransitionBandwidth(t *testing.T) {
	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			calculatedTrBw := 0.5 * (preset.Fs - preset.Fp)

			assert.InDelta(t, preset.TrBw, calculatedTrBw, 0.001,
				"%s: tr_bw should be 0.5*(Fs-Fp)", preset.Name)

			t.Logf("%s: Fp=%.3f, Fs=%.3f -> tr_bw=%.4f", preset.Name, preset.Fp, preset.Fs, calculatedTrBw)
		})
	}
}

// TestQualityPreset_KaiserBeta validates Kaiser beta for each preset.
func TestQualityPreset_KaiserBeta(t *testing.T) {
	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			beta := KaiserBetaWithTrBw(preset.Attenuation, preset.TrBw)

			t.Logf("%s: att=%.2f dB, tr_bw=%.4f -> beta=%.4f",
				preset.Name, preset.Attenuation, preset.TrBw, beta)

			// Beta should increase with attenuation
			if preset.Attenuation > 50 {
				assert.Greater(t, beta, 4.0, "beta should be > 4 for att > 50 dB")
			}
		})
	}
}

// TestQualityPreset_FilterLength validates filter length for each preset.
func TestQualityPreset_FilterLength(t *testing.T) {
	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			numTaps := EstimateFilterLength(preset.Attenuation, preset.TrBw)

			t.Logf("%s: att=%.2f dB, tr_bw=%.4f -> numTaps=%d",
				preset.Name, preset.Attenuation, preset.TrBw, numTaps)

			// Filter length should increase with attenuation and decrease with tr_bw
			assert.Greater(t, numTaps, 10, "filter should have at least 10 taps")
		})
	}
}

// Test cases for 96kHz -> 48kHz decimation with each quality preset
// This is a key use case where proper filter design is critical

// DecimationTestCase holds parameters for decimation testing
type DecimationTestCase struct {
	InputRate    float64
	OutputRate   float64
	Quality      QualityPreset
	Fn           float64 // Normalization factor = max(preL, preM)
	FpNorm       float64 // Normalized passband end
	FsNorm       float64 // Normalized stopband begin
	TrBwNorm     float64 // Normalized transition bandwidth
	ExpectedTaps int     // Expected filter length (approximate)
}

// TestDecimation96to48_AllPresets tests 96->48 decimation with all quality presets.
func TestDecimation96to48_AllPresets(t *testing.T) {
	const (
		inputRate  = 96000.0
		outputRate = 48000.0
		Fn         = 2.0 // Decimation factor
	)

	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			// Normalize parameters for decimation
			FpNorm := preset.Fp / Fn
			FsNorm := preset.Fs / Fn
			trBwNorm := 0.5 * (FsNorm - FpNorm)
			Fc := FsNorm - trBwNorm

			t.Logf("%s 96->48 decimation parameters:", preset.Name)
			t.Logf("  Fp=%.4f, Fs=%.4f, Fn=%.1f", preset.Fp, preset.Fs, Fn)
			t.Logf("  Fp_norm=%.4f, Fs_norm=%.4f", FpNorm, FsNorm)
			t.Logf("  tr_bw_norm=%.5f, Fc=%.5f", trBwNorm, Fc)

			// Calculate Kaiser beta
			beta := KaiserBetaWithTrBw(preset.Attenuation, trBwNorm)
			t.Logf("  attenuation=%.2f dB, beta=%.4f", preset.Attenuation, beta)

			// Calculate filter length (using our formula)
			numTaps := EstimateFilterLength(preset.Attenuation, trBwNorm)
			t.Logf("  our numTaps=%d", numTaps)

			// Calculate what soxr would use (VHQ uses att/tr_bw for narrow transitions)
			soxrTaps := SoxrFilterTapCount(preset.Attenuation, trBwNorm, beta)
			t.Logf("  soxr numTaps≈%d", soxrTaps)

			// Validate parameters are reasonable
			// Quick/Low have lower Fp (0.67625) giving Fp_norm ≈ 0.34
			// Medium/High/VHQ have higher Fp (0.91+) giving Fp_norm ≈ 0.45+
			assert.Greater(t, FpNorm, 0.3, "Fp_norm should be > 0.3")
			assert.LessOrEqual(t, FsNorm, 0.5, "Fs_norm should be <= 0.5 (Nyquist)")
			assert.Greater(t, trBwNorm, 0.0, "tr_bw should be positive")
		})
	}
}

// Test upsampling case: 44.1kHz -> 48kHz

// TestUpsampling44to48_AllPresets tests 44.1->48 upsampling with all quality presets.
func TestUpsampling44to48_AllPresets(t *testing.T) {
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
	)

	ratio := outputRate / inputRate // 1.088...
	t.Logf("Upsampling ratio: %.6f", ratio)

	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			// For upsampling, Fn = 1 (no decimation factor)
			// Filter designed to pass up to input Nyquist, block images
			Fn := 1.0
			FpNorm := preset.Fp / Fn
			FsNorm := preset.Fs / Fn
			trBw := 0.5 * (FsNorm - FpNorm)

			t.Logf("%s 44.1->48 upsampling parameters:", preset.Name)
			t.Logf("  Fp=%.4f, Fs=%.4f, tr_bw=%.4f", FpNorm, FsNorm, trBw)

			// Calculate filter length
			numTaps := EstimateFilterLength(preset.Attenuation, trBw)
			t.Logf("  attenuation=%.2f dB, numTaps=%d", preset.Attenuation, numTaps)
		})
	}
}

// Test downsampling case: 48kHz -> 32kHz (non-integer ratio)

// TestDownsampling48to32_AllPresets tests 48->32 downsampling with all quality presets.
func TestDownsampling48to32_AllPresets(t *testing.T) {
	const (
		inputRate  = 48000.0
		outputRate = 32000.0
	)

	ratio := outputRate / inputRate   // 0.666...
	ioRatio := inputRate / outputRate // 1.5
	t.Logf("Downsampling ratio: %.6f (io_ratio=%.2f)", ratio, ioRatio)

	for _, preset := range soxrQualityPresets {
		t.Run(preset.Name, func(t *testing.T) {
			// For non-integer downsampling, need to determine Fn
			// soxr uses Fn = mult for downsampling polyphase
			Fn := ioRatio // 1.5

			FpNorm := preset.Fp / Fn
			FsNorm := preset.Fs / Fn
			trBw := 0.5 * (FsNorm - FpNorm)

			t.Logf("%s 48->32 downsampling parameters:", preset.Name)
			t.Logf("  Fn=%.2f, Fp_norm=%.4f, Fs_norm=%.4f, tr_bw=%.4f", Fn, FpNorm, FsNorm, trBw)

			// For downsampling, filter cutoff must be below output Nyquist
			// Output Nyquist = 16kHz, normalized: 16/24 = 0.667 of input Nyquist
			outputNyquistNorm := (outputRate / 2.0) / (inputRate / 2.0)
			t.Logf("  output Nyquist (normalized): %.4f", outputNyquistNorm)

			assert.LessOrEqual(t, FsNorm, outputNyquistNorm,
				"Fs_norm should be <= output Nyquist to prevent aliasing")
		})
	}
}

// Expected soxr filter tap counts for various configurations
// These values should be verified against soxr analysis session

var expectedSoxrTapCounts = []struct {
	name        string
	quality     string
	inputRate   float64
	outputRate  float64
	expectedMin int // Minimum expected taps
	expectedMax int // Maximum expected taps
}{
	// VHQ 96->48 (from analysis)
	{"vhq_96_48", "VeryHigh", 96000, 48000, 7500, 8500},

	// HQ cases (need verification)
	{"hq_96_48", "High", 96000, 48000, 1000, 2000},
	{"hq_44_48", "High", 44100, 48000, 500, 1000},

	// MQ cases
	{"mq_96_48", "Medium", 96000, 48000, 500, 1000},
	{"mq_44_48", "Medium", 44100, 48000, 300, 600},

	// LQ cases
	{"lq_96_48", "Low", 96000, 48000, 300, 600},
	{"lq_44_48", "Low", 44100, 48000, 200, 400},
}

// TestSoxrTapCountRanges documents expected tap count ranges.
// These are placeholders that should be verified with soxr analysis.
func TestSoxrTapCountRanges(t *testing.T) {
	t.Log("Expected soxr tap count ranges (verify with soxr analysis):")
	for _, tc := range expectedSoxrTapCounts {
		t.Logf("  %s: %d-%d taps", tc.name, tc.expectedMin, tc.expectedMax)
		// Validate range is sensible
		assert.Less(t, tc.expectedMin, tc.expectedMax,
			"%s: expectedMin should be less than expectedMax", tc.name)
		assert.Positive(t, tc.expectedMin,
			"%s: expectedMin should be positive", tc.name)
	}
}

// Benchmark filter length calculation for all presets
func BenchmarkFilterLengthCalculation(b *testing.B) {
	for _, preset := range soxrQualityPresets {
		b.Run(preset.Name, func(b *testing.B) {
			for b.Loop() {
				_ = EstimateFilterLength(preset.Attenuation, preset.TrBw)
			}
		})
	}
}

// Benchmark Kaiser beta calculation for all presets
func BenchmarkKaiserBetaCalculation(b *testing.B) {
	for _, preset := range soxrQualityPresets {
		b.Run(preset.Name, func(b *testing.B) {
			for b.Loop() {
				_ = KaiserBetaWithTrBw(preset.Attenuation, preset.TrBw)
			}
		})
	}
}
