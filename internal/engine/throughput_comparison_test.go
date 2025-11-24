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
	"time"
)

// =============================================================================
// Throughput Comparison Tests
// =============================================================================
//
// These tests compare throughput (samples/second) between Go resampler and SOXR
// at different quality presets.
// =============================================================================

const (
	// Number of samples per throughput test iteration (1 second at 48kHz)
	throughputTestSamples = 48000

	// Number of iterations for throughput measurement
	throughputIterations = 100
)

// ThroughputResult holds throughput measurement results
type ThroughputResult struct {
	Quality            string
	InputRate          float64
	OutputRate         float64
	TotalInputSamples  int64
	TotalOutputSamples int64
	ElapsedSec         float64
	InputSamplesPerSec float64
	MegasamplesPerSec  float64
}

// measureGoThroughput measures Go resampler throughput
func measureGoThroughput(inputRate, outputRate float64, quality Quality, numSamples, iterations int) (*ThroughputResult, error) {
	// Generate test signal
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * 1000.0 * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Warm up
	warmupResampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create warmup resampler: %w", err)
	}
	_, _ = warmupResampler.Process(input)

	// Time multiple iterations
	var totalInputSamples, totalOutputSamples int64

	start := time.Now()
	for range iterations {
		resampler, err := NewResampler[float64](inputRate, outputRate, quality)
		if err != nil {
			return nil, fmt.Errorf("failed to create resampler: %w", err)
		}

		output, err := resampler.Process(input)
		if err != nil {
			return nil, fmt.Errorf("failed to process: %w", err)
		}

		flush, _ := resampler.Flush()
		output = append(output, flush...)

		totalInputSamples += int64(numSamples)
		totalOutputSamples += int64(len(output))
	}
	elapsed := time.Since(start)

	elapsedSec := elapsed.Seconds()
	inputSamplesPerSec := float64(totalInputSamples) / elapsedSec
	megasamplesPerSec := inputSamplesPerSec / 1e6

	return &ThroughputResult{
		Quality:            qualityName(quality),
		InputRate:          inputRate,
		OutputRate:         outputRate,
		TotalInputSamples:  totalInputSamples,
		TotalOutputSamples: totalOutputSamples,
		ElapsedSec:         elapsedSec,
		InputSamplesPerSec: inputSamplesPerSec,
		MegasamplesPerSec:  megasamplesPerSec,
	}, nil
}

// measureGoThroughputReuse measures Go resampler throughput with resampler reuse
func measureGoThroughputReuse(inputRate, outputRate float64, quality Quality, numSamples, iterations int) (*ThroughputResult, error) {
	// Generate test signal
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * 1000.0 * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Create resampler once
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		return nil, fmt.Errorf("failed to create resampler: %w", err)
	}

	// Warm up
	_, _ = resampler.Process(input)
	resampler.Reset()

	// Time multiple iterations
	var totalInputSamples, totalOutputSamples int64

	start := time.Now()
	for range iterations {
		output, err := resampler.Process(input)
		if err != nil {
			return nil, fmt.Errorf("failed to process: %w", err)
		}

		totalInputSamples += int64(numSamples)
		totalOutputSamples += int64(len(output))
	}
	elapsed := time.Since(start)

	elapsedSec := elapsed.Seconds()
	inputSamplesPerSec := float64(totalInputSamples) / elapsedSec
	megasamplesPerSec := inputSamplesPerSec / 1e6

	return &ThroughputResult{
		Quality:            qualityName(quality) + "_reuse",
		InputRate:          inputRate,
		OutputRate:         outputRate,
		TotalInputSamples:  totalInputSamples,
		TotalOutputSamples: totalOutputSamples,
		ElapsedSec:         elapsedSec,
		InputSamplesPerSec: inputSamplesPerSec,
		MegasamplesPerSec:  megasamplesPerSec,
	}, nil
}

// getSoxrThroughput runs the soxr throughput test
func getSoxrThroughput(inputRate, outputRate float64, quality string) (*ThroughputResult, error) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return nil, fmt.Errorf("failed to get current file path")
	}

	projectRoot := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))
	toolPath := filepath.Join(projectRoot, "test-reference", "test_quality")

	if _, err := os.Stat(toolPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("soxr quality tool not found at %s", toolPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, toolPath,
		fmt.Sprintf("%.0f", inputRate),
		fmt.Sprintf("%.0f", outputRate),
		"throughput",
		quality)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run soxr throughput test: %w", err)
	}

	// Parse output
	result := &ThroughputResult{
		Quality:    quality,
		InputRate:  inputRate,
		OutputRate: outputRate,
	}

	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "# ") && strings.Contains(line, "=") {
			line = strings.TrimPrefix(line, "# ")
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				valStr := strings.TrimSpace(parts[1])

				switch key {
				case "input_samples_per_sec":
					result.InputSamplesPerSec, _ = strconv.ParseFloat(valStr, 64)
				case "megasamples_per_sec":
					result.MegasamplesPerSec, _ = strconv.ParseFloat(valStr, 64)
				case "elapsed_sec":
					result.ElapsedSec, _ = strconv.ParseFloat(valStr, 64)
				case "total_input_samples":
					val, _ := strconv.ParseInt(valStr, 10, 64)
					result.TotalInputSamples = val
				case "total_output_samples":
					val, _ := strconv.ParseInt(valStr, 10, 64)
					result.TotalOutputSamples = val
				}
			}
		}
	}

	return result, nil
}

// =============================================================================
// TEST: Throughput comparison across quality presets
// =============================================================================

func TestThroughput_AllQualityPresets(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		name       string
	}{
		{44100, 48000, "44.1kHz_to_48kHz"},
		{48000, 44100, "48kHz_to_44.1kHz"},
		{48000, 32000, "48kHz_to_32kHz_BirdNET"},
		{48000, 96000, "48kHz_to_96kHz"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("=== Throughput Comparison: %.0f Hz -> %.0f Hz ===", tc.inputRate, tc.outputRate)
			t.Logf("")
			t.Logf("%-12s | %12s | %12s | %8s", "Quality", "SOXR (MS/s)", "Go (MS/s)", "Ratio")
			t.Logf("-------------|--------------|--------------|----------")

			for _, preset := range qualityPresets {
				// Get SOXR throughput
				soxrResult, err := getSoxrThroughput(tc.inputRate, tc.outputRate, preset.soxrQuality)
				if err != nil {
					t.Logf("%-12s | %12s | %12s | %8s", preset.name, "N/A", "N/A", "N/A")
					continue
				}

				// Get Go throughput (with resampler reuse for fair comparison)
				goResult, err := measureGoThroughputReuse(tc.inputRate, tc.outputRate, preset.goQuality,
					int(tc.inputRate), throughputIterations)
				if err != nil {
					t.Logf("%-12s | %12.2f | %12s | %8s", preset.name, soxrResult.MegasamplesPerSec, "ERR", "N/A")
					continue
				}

				ratio := goResult.MegasamplesPerSec / soxrResult.MegasamplesPerSec
				t.Logf("%-12s | %12.2f | %12.2f | %7.1f%%",
					preset.name,
					soxrResult.MegasamplesPerSec,
					goResult.MegasamplesPerSec,
					ratio*100)
			}
		})
	}
}

// =============================================================================
// TEST: Detailed throughput comparison for specific conversion
// =============================================================================

func TestThroughput_DetailedComparison(t *testing.T) {
	const (
		inputRate  = 48000.0
		outputRate = 32000.0
	)

	t.Log("=== Detailed Throughput Comparison: 48kHz -> 32kHz (BirdNET) ===")
	t.Log("")

	for _, preset := range qualityPresets {
		t.Run(preset.name, func(t *testing.T) {
			// Get SOXR throughput
			soxrResult, err := getSoxrThroughput(inputRate, outputRate, preset.soxrQuality)
			if err != nil {
				t.Skipf("soxr throughput not available: %v", err)
			}

			// Get Go throughput (new resampler each time - fair comparison)
			goResult, err := measureGoThroughput(inputRate, outputRate, preset.goQuality,
				int(inputRate), throughputIterations)
			if err != nil {
				t.Fatalf("Go throughput measurement failed: %v", err)
			}

			// Get Go throughput with resampler reuse
			goReuseResult, err := measureGoThroughputReuse(inputRate, outputRate, preset.goQuality,
				int(inputRate), throughputIterations)
			if err != nil {
				t.Fatalf("Go throughput (reuse) measurement failed: %v", err)
			}

			t.Logf("=== %s Quality ===", preset.name)
			t.Logf("SOXR:           %.2f MS/s (%.0f samples/sec)", soxrResult.MegasamplesPerSec, soxrResult.InputSamplesPerSec)
			t.Logf("Go (new):       %.2f MS/s (%.0f samples/sec)", goResult.MegasamplesPerSec, goResult.InputSamplesPerSec)
			t.Logf("Go (reuse):     %.2f MS/s (%.0f samples/sec)", goReuseResult.MegasamplesPerSec, goReuseResult.InputSamplesPerSec)
			t.Logf("")

			ratioNew := goResult.MegasamplesPerSec / soxrResult.MegasamplesPerSec
			ratioReuse := goReuseResult.MegasamplesPerSec / soxrResult.MegasamplesPerSec

			t.Logf("Go/SOXR ratio (new):   %.1f%%", ratioNew*100)
			t.Logf("Go/SOXR ratio (reuse): %.1f%%", ratioReuse*100)

			// Log if Go is significantly slower
			if ratioReuse < 0.5 {
				t.Logf("WARNING: Go is less than 50%% of SOXR throughput")
			}
		})
	}
}

// =============================================================================
// TEST: Throughput summary
// =============================================================================

func TestThroughput_Summary(t *testing.T) {
	const (
		inputRate  = 48000.0
		outputRate = 32000.0
	)

	t.Log("=====================================================")
	t.Log("Throughput Summary: 48kHz -> 32kHz")
	t.Log("=====================================================")
	t.Log("")

	type summaryRow struct {
		quality    string
		soxrMS     float64
		goMS       float64
		goReuseMS  float64
		ratio      float64
		reuseRatio float64
	}

	summary := make([]summaryRow, 0, len(qualityPresets))

	for _, preset := range qualityPresets {
		row := summaryRow{quality: preset.name}

		// Get SOXR throughput
		if soxrResult, err := getSoxrThroughput(inputRate, outputRate, preset.soxrQuality); err == nil {
			row.soxrMS = soxrResult.MegasamplesPerSec
		}

		// Get Go throughput
		if goResult, err := measureGoThroughput(inputRate, outputRate, preset.goQuality,
			int(inputRate), throughputIterations); err == nil {
			row.goMS = goResult.MegasamplesPerSec
		}

		// Get Go throughput with reuse
		if goReuseResult, err := measureGoThroughputReuse(inputRate, outputRate, preset.goQuality,
			int(inputRate), throughputIterations); err == nil {
			row.goReuseMS = goReuseResult.MegasamplesPerSec
		}

		if row.soxrMS > 0 {
			row.ratio = row.goMS / row.soxrMS
			row.reuseRatio = row.goReuseMS / row.soxrMS
		}

		summary = append(summary, row)
	}

	// Print summary table
	t.Log("Quality     | SOXR (MS/s) | Go New (MS/s) | Go Reuse (MS/s) | Ratio (new) | Ratio (reuse)")
	t.Log("------------|-------------|---------------|-----------------|-------------|-------------")
	for _, row := range summary {
		t.Logf("%-11s | %11.2f | %13.2f | %15.2f | %10.1f%% | %11.1f%%",
			row.quality, row.soxrMS, row.goMS, row.goReuseMS, row.ratio*100, row.reuseRatio*100)
	}
}

// =============================================================================
// Benchmarks for Go throughput
// =============================================================================

func BenchmarkThroughput_Quick_48kTo32k(b *testing.B) {
	benchmarkThroughput(b, 48000, 32000, QualityQuick)
}

func BenchmarkThroughput_Low_48kTo32k(b *testing.B) {
	benchmarkThroughput(b, 48000, 32000, QualityLow)
}

func BenchmarkThroughput_Medium_48kTo32k(b *testing.B) {
	benchmarkThroughput(b, 48000, 32000, QualityMedium)
}

func BenchmarkThroughput_High_48kTo32k(b *testing.B) {
	benchmarkThroughput(b, 48000, 32000, QualityHigh)
}

func BenchmarkThroughput_VeryHigh_48kTo32k(b *testing.B) {
	benchmarkThroughput(b, 48000, 32000, QualityVeryHigh)
}

func BenchmarkThroughput_VeryHigh_44kTo48k(b *testing.B) {
	benchmarkThroughput(b, 44100, 48000, QualityVeryHigh)
}

func BenchmarkThroughput_VeryHigh_48kTo96k(b *testing.B) {
	benchmarkThroughput(b, 48000, 96000, QualityVeryHigh)
}

func benchmarkThroughput(b *testing.B, inputRate, outputRate float64, quality Quality) {
	b.Helper()
	numSamples := int(inputRate) // 1 second of audio

	// Generate test signal
	input := make([]float64, numSamples)
	for i := range input {
		phase := 2.0 * math.Pi * 1000.0 * float64(i) / inputRate
		input[i] = 0.9 * math.Sin(phase)
	}

	// Create resampler
	resampler, err := NewResampler[float64](inputRate, outputRate, quality)
	if err != nil {
		b.Fatalf("failed to create resampler: %v", err)
	}

	// Warm up
	_, _ = resampler.Process(input)
	resampler.Reset()

	b.ResetTimer()
	b.SetBytes(int64(numSamples * 8)) // 8 bytes per float64

	for i := 0; i < b.N; i++ {
		_, err := resampler.Process(input)
		if err != nil {
			b.Fatalf("process failed: %v", err)
		}
	}

	b.StopTimer()
	b.ReportMetric(float64(numSamples*b.N)/b.Elapsed().Seconds()/1e6, "MS/s")
}
