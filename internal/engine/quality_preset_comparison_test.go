package engine

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// =============================================================================
// Quality Preset Comparison Tests
// =============================================================================
//
// These tests compare each Go quality preset against its corresponding SOXR
// quality preset to verify that performance is aligned.
//
// Quality Preset Mapping:
//   Go QualityQuick    <-> SOXR_QQ  (quick)
//   Go QualityLow      <-> SOXR_LQ  (low)
//   Go QualityMedium   <-> SOXR_MQ  (medium)
//   Go QualityHigh     <-> SOXR_HQ  (high)
//   Go QualityVeryHigh <-> SOXR_VHQ (veryhigh)
// =============================================================================

// qualityPresetPair maps Go quality to SOXR quality string
type qualityPresetPair struct {
	goQuality   Quality
	soxrQuality string // "quick", "low", "medium", "high", "veryhigh"
	name        string
	bits        int // expected bit precision
}

var qualityPresets = []qualityPresetPair{
	{QualityQuick, "quick", "Quick", 8},
	{QualityLow, "low", "Low", 16},
	{QualityMedium, "medium", "Medium", 16},
	{QualityHigh, "high", "High", 20},
	{QualityVeryHigh, "veryhigh", "VeryHigh", 28},
}

// runSoxrQualityTestWithPreset runs the soxr quality test tool with a specific preset
func runSoxrQualityTestWithPreset(inputRate, outputRate float64, testType, quality string) (map[string]float64, error) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return nil, fmt.Errorf("failed to get current file path")
	}

	projectRoot := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))
	toolPath := filepath.Join(projectRoot, "test-reference", "test_quality")

	if _, err := os.Stat(toolPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("soxr quality tool not found at %s (run 'make test_quality' in test-reference/)", toolPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*1000*1000*1000) // 30 seconds
	defer cancel()

	cmd := exec.CommandContext(ctx, toolPath,
		fmt.Sprintf("%.0f", inputRate),
		fmt.Sprintf("%.0f", outputRate),
		testType,
		quality)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run soxr quality tool: %w", err)
	}

	// Parse output (format: key = value lines starting with #)
	result := make(map[string]float64)
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "# ") && strings.Contains(line, "=") {
			line = strings.TrimPrefix(line, "# ")
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				valStr := strings.TrimSpace(parts[1])
				// Remove any units suffix
				valStr = strings.Fields(valStr)[0]
				val, err := strconv.ParseFloat(valStr, 64)
				if err == nil {
					result[key] = val
				}
			}
		}
	}

	return result, nil
}

// =============================================================================
// TEST: THD across all quality presets
// =============================================================================

func TestTHD_AllQualityPresets(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		testFreq   float64
		name       string
	}{
		{44100, 48000, 1000, "44.1kHz_to_48kHz"},
		{48000, 44100, 1000, "48kHz_to_44.1kHz"},
		{48000, 96000, 1000, "48kHz_to_96kHz"},
		{48000, 32000, 1000, "48kHz_to_32kHz_BirdNET"},
	}

	for _, preset := range qualityPresets {
		for _, tc := range testCases {
			testName := fmt.Sprintf("%s/%s", preset.name, tc.name)
			t.Run(testName, func(t *testing.T) {
				// Get SOXR reference
				soxrResult, err := runSoxrQualityTestWithPreset(tc.inputRate, tc.outputRate,
					fmt.Sprintf("thd:%.0f", tc.testFreq), preset.soxrQuality)
				if err != nil {
					t.Skipf("soxr reference not available: %v", err)
				}

				// Get Go result
				goResult, err := measureTHD(tc.inputRate, tc.outputRate, tc.testFreq, preset.goQuality)
				if err != nil {
					t.Fatalf("measureTHD failed: %v", err)
				}

				soxrTHD := soxrResult["thd_db"]

				t.Logf("=== THD Comparison (%s) ===", preset.name)
				t.Logf("Conversion: %.0f Hz -> %.0f Hz @ %.0f Hz", tc.inputRate, tc.outputRate, tc.testFreq)
				t.Logf("SOXR %s: %.2f dB", preset.name, soxrTHD)
				t.Logf("Go %s:   %.2f dB", preset.name, goResult.THD_DB)

				diff := goResult.THD_DB - soxrTHD
				t.Logf("Difference: %.2f dB (positive = Go worse)", diff)

				// THD difference tolerance based on quality level
				// Lower quality levels can have more variation
				tolerance := 30.0 // dB - allow significant variation
				if preset.bits >= 20 {
					tolerance = 20.0
				}

				if diff > tolerance {
					t.Errorf("THD GAP: Go %s THD is %.2f dB worse than SOXR (tolerance: %.0f dB)",
						preset.name, diff, tolerance)
				}
			})
		}
	}
}

// =============================================================================
// TEST: SNR across all quality presets
// =============================================================================

func TestSNR_AllQualityPresets(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz"},
		{48000, 44100, "48kHz_to_44.1kHz"},
		{48000, 96000, "48kHz_to_96kHz"},
		{48000, 32000, "48kHz_to_32kHz_BirdNET"},
	}

	testFreq := 1000.0

	for _, preset := range qualityPresets {
		for _, tc := range testCases {
			testName := fmt.Sprintf("%s/%s", preset.name, tc.name)
			t.Run(testName, func(t *testing.T) {
				// Get SOXR reference
				soxrResult, err := runSoxrQualityTestWithPreset(tc.inputRate, tc.outputRate,
					fmt.Sprintf("snr:%.0f", testFreq), preset.soxrQuality)
				if err != nil {
					t.Skipf("soxr reference not available: %v", err)
				}

				// Get Go result
				goResult, err := measureSNR(tc.inputRate, tc.outputRate, testFreq, preset.goQuality)
				if err != nil {
					t.Fatalf("measureSNR failed: %v", err)
				}

				soxrSNR := soxrResult["snr_db"]

				t.Logf("=== SNR Comparison (%s) ===", preset.name)
				t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
				t.Logf("SOXR %s: %.2f dB", preset.name, soxrSNR)
				t.Logf("Go %s:   %.2f dB", preset.name, goResult.SNR_DB)

				diff := soxrSNR - goResult.SNR_DB
				t.Logf("Difference: %.2f dB (positive = SOXR better)", diff)

				// SNR tolerance - allow some variation
				tolerance := 20.0 // dB
				if diff > tolerance {
					t.Errorf("SNR GAP: SOXR %s has %.2f dB better SNR than Go (tolerance: %.0f dB)",
						preset.name, diff, tolerance)
				}
			})
		}
	}
}

// =============================================================================
// TEST: Passband Ripple across all quality presets
// =============================================================================

func TestPassbandRipple_AllQualityPresets(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz"},
		{48000, 44100, "48kHz_to_44.1kHz"},
		{48000, 96000, "48kHz_to_96kHz"},
	}

	for _, preset := range qualityPresets {
		for _, tc := range testCases {
			testName := fmt.Sprintf("%s/%s", preset.name, tc.name)
			t.Run(testName, func(t *testing.T) {
				// Get SOXR reference
				soxrResult, err := runSoxrQualityTestWithPreset(tc.inputRate, tc.outputRate,
					"ripple", preset.soxrQuality)
				if err != nil {
					t.Skipf("soxr reference not available: %v", err)
				}

				// Get Go result
				goResult, err := measurePassbandRipple(tc.inputRate, tc.outputRate, preset.goQuality)
				if err != nil {
					t.Fatalf("measurePassbandRipple failed: %v", err)
				}

				soxrRipple := soxrResult["ripple"]

				t.Logf("=== Passband Ripple Comparison (%s) ===", preset.name)
				t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
				t.Logf("SOXR %s: %.4f dB", preset.name, soxrRipple)
				t.Logf("Go %s:   %.4f dB", preset.name, goResult.RipplePeakPeak)

				diff := goResult.RipplePeakPeak - soxrRipple
				t.Logf("Difference: %.4f dB (positive = Go worse)", diff)

				// Ripple tolerance - should be relatively close
				tolerance := 2.0 // dB
				if diff > tolerance {
					t.Errorf("RIPPLE GAP: Go %s has %.4f dB more ripple than SOXR (tolerance: %.1f dB)",
						preset.name, diff, tolerance)
				}
			})
		}
	}
}

// =============================================================================
// TEST: Impulse Response across all quality presets
// =============================================================================

func TestImpulseResponse_AllQualityPresets(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz"},
		{48000, 44100, "48kHz_to_44.1kHz"},
		{48000, 96000, "48kHz_to_96kHz"},
	}

	for _, preset := range qualityPresets {
		for _, tc := range testCases {
			testName := fmt.Sprintf("%s/%s", preset.name, tc.name)
			t.Run(testName, func(t *testing.T) {
				// Get SOXR reference
				soxrResult, err := runSoxrQualityTestWithPreset(tc.inputRate, tc.outputRate,
					"impulse", preset.soxrQuality)
				if err != nil {
					t.Skipf("soxr reference not available: %v", err)
				}

				// Get Go result
				goResult, err := measureImpulseResponse(tc.inputRate, tc.outputRate, preset.goQuality)
				if err != nil {
					t.Fatalf("measureImpulseResponse failed: %v", err)
				}

				soxrPreRing := soxrResult["pre_ringing_db"]
				soxrPostRing := soxrResult["post_ringing_db"]

				t.Logf("=== Impulse Response Comparison (%s) ===", preset.name)
				t.Logf("Conversion: %.0f Hz -> %.0f Hz", tc.inputRate, tc.outputRate)
				t.Logf("SOXR %s: pre=%.2f dB, post=%.2f dB", preset.name, soxrPreRing, soxrPostRing)
				t.Logf("Go %s:   pre=%.2f dB, post=%.2f dB", preset.name, goResult.PreRingingDB, goResult.PostRingingDB)

				preRingDiff := goResult.PreRingingDB - soxrPreRing
				t.Logf("Pre-ringing diff: %.2f dB (positive = Go has more)", preRingDiff)

				// Pre-ringing tolerance - more negative is better, so positive diff means Go is worse
				tolerance := 10.0 // dB
				if preRingDiff > tolerance {
					t.Logf("WARNING: Go %s has %.2f dB more pre-ringing than SOXR", preset.name, preRingDiff)
				}
			})
		}
	}
}

// =============================================================================
// TEST: Quality levels produce expected relative performance
// =============================================================================

func TestQualityLevels_RelativePerformance(t *testing.T) {
	// Test that higher quality levels produce better metrics
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		testFreq   = 1000.0
	)

	// Collect THD for all quality levels
	thdResults := make(map[Quality]float64)
	for _, preset := range qualityPresets {
		result, err := measureTHD(inputRate, outputRate, testFreq, preset.goQuality)
		if err != nil {
			t.Fatalf("measureTHD failed for %s: %v", preset.name, err)
		}
		thdResults[preset.goQuality] = result.THD_DB
		t.Logf("%s THD: %.2f dB", preset.name, result.THD_DB)
	}

	// Verify that higher quality has equal or better (more negative) THD
	// Note: Quick uses cubic interpolation, so exclude it from strict comparison
	t.Run("Low_vs_High", func(t *testing.T) {
		if thdResults[QualityLow] < thdResults[QualityHigh]-10 {
			t.Errorf("QualityHigh (%.2f dB) should have equal or better THD than QualityLow (%.2f dB)",
				thdResults[QualityHigh], thdResults[QualityLow])
		}
	})

	t.Run("High_vs_VeryHigh", func(t *testing.T) {
		if thdResults[QualityHigh] < thdResults[QualityVeryHigh]-10 {
			t.Errorf("QualityVeryHigh (%.2f dB) should have equal or better THD than QualityHigh (%.2f dB)",
				thdResults[QualityVeryHigh], thdResults[QualityHigh])
		}
	})
}

// =============================================================================
// TEST: Comprehensive quality comparison summary
// =============================================================================

func TestQualityPresets_ComprehensiveSummary(t *testing.T) {
	// This test generates a comprehensive summary comparing all presets
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		testFreq   = 1000.0
	)

	t.Log("===========================================")
	t.Log("Quality Preset Comparison Summary")
	t.Log("Conversion: 44100 Hz -> 48000 Hz")
	t.Log("===========================================")
	t.Log("")

	type summaryRow struct {
		preset   string
		soxrTHD  float64
		goTHD    float64
		soxrSNR  float64
		goSNR    float64
		thdMatch bool
		snrMatch bool
	}

	summary := make([]summaryRow, 0, len(qualityPresets))

	for _, preset := range qualityPresets {
		row := summaryRow{preset: preset.name}

		// Get SOXR THD
		if soxrResult, err := runSoxrQualityTestWithPreset(inputRate, outputRate,
			fmt.Sprintf("thd:%.0f", testFreq), preset.soxrQuality); err == nil {
			row.soxrTHD = soxrResult["thd_db"]
		}

		// Get Go THD
		if goResult, err := measureTHD(inputRate, outputRate, testFreq, preset.goQuality); err == nil {
			row.goTHD = goResult.THD_DB
		}

		// Get SOXR SNR
		if soxrResult, err := runSoxrQualityTestWithPreset(inputRate, outputRate,
			fmt.Sprintf("snr:%.0f", testFreq), preset.soxrQuality); err == nil {
			row.soxrSNR = soxrResult["snr_db"]
		}

		// Get Go SNR
		if goResult, err := measureSNR(inputRate, outputRate, testFreq, preset.goQuality); err == nil {
			row.goSNR = goResult.SNR_DB
		}

		// Check if within tolerance
		row.thdMatch = math.Abs(row.goTHD-row.soxrTHD) < 30.0
		row.snrMatch = math.Abs(row.goSNR-row.soxrSNR) < 20.0

		summary = append(summary, row)
	}

	// Print summary table
	t.Log("Preset      | SOXR THD   | Go THD     | SOXR SNR  | Go SNR    | Match")
	t.Log("------------|------------|------------|-----------|-----------|------")
	for _, row := range summary {
		thdStatus := "✓"
		if !row.thdMatch {
			thdStatus = "✗"
		}
		snrStatus := "✓"
		if !row.snrMatch {
			snrStatus = "✗"
		}
		t.Logf("%-11s | %9.2f dB | %9.2f dB | %8.2f dB | %8.2f dB | THD:%s SNR:%s",
			row.preset, row.soxrTHD, row.goTHD, row.soxrSNR, row.goSNR, thdStatus, snrStatus)
	}

	// Fail if any preset doesn't match
	for _, row := range summary {
		if !row.thdMatch {
			t.Errorf("Preset %s: THD mismatch (SOXR: %.2f dB, Go: %.2f dB)",
				row.preset, row.soxrTHD, row.goTHD)
		}
		if !row.snrMatch {
			t.Errorf("Preset %s: SNR mismatch (SOXR: %.2f dB, Go: %.2f dB)",
				row.preset, row.soxrSNR, row.goSNR)
		}
	}
}
