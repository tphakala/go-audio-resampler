package engine

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

const (
	// Path to soxr reference tool (absolute path)
	soxrRefPath = "/home/thakala/src/go-audio-resample/test-reference/test_soxr_reference"

	// Tolerance for comparing outputs
	// These account for filter design differences between our Kaiser-windowed FIR
	// and soxr's Dolph-Chebyshev design. The key metric is correlation, not absolute error.

	// Correlation threshold - signals should have same shape
	correlationThreshold = 0.95 // 0.95+ correlation indicates correct resampling

	// Max error threshold - allows for filter design differences
	// Note: Different filter designs can have ~10% peak differences at high frequencies
	maxErrorThreshold = 0.20 // 20% max error (filter design differences)

	// Stricter thresholds for low frequency signals
	lowFreqMaxError    = 0.15 // 15% for f < 5kHz
	lowFreqCorrelation = 0.99 // 99%+ for low frequencies

	// DC tolerance (should be very tight)
	dcTolerance = 0.01 // 1% for DC signals

	// Standard test signal length
	testSignalSamples = 4000
)

// getSoxrReference runs the soxr reference tool and returns output samples
func getSoxrReference(inputRate, outputRate float64, signalType string, frequency float64) ([]float64, error) {
	args := []string{
		fmt.Sprintf("%.0f", inputRate),
		fmt.Sprintf("%.0f", outputRate),
		signalType,
	}
	if signalType == "sine" {
		args = append(args, fmt.Sprintf("%.0f", frequency))
	}

	cmd := exec.CommandContext(context.Background(), soxrRefPath, args...)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("soxr reference failed: %w", err)
	}

	// Parse output
	// Pre-allocate with estimated capacity (typically ~5000 samples for our test signals)
	samples := make([]float64, 0, 8192)
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "[") {
			continue
		}
		val, err := strconv.ParseFloat(line, 64)
		if err != nil {
			continue
		}
		samples = append(samples, val)
	}

	return samples, nil
}

// generateTestSignal creates a test signal matching the C reference.
// Uses testSignalSamples (4000) as the fixed signal length.
func generateTestSignal(signalType string, sampleRate, frequency float64) []float64 {
	signal := make([]float64, testSignalSamples)

	switch signalType {
	case "dc":
		for i := range signal {
			signal[i] = 1.0
		}
	case "sine":
		for i := range signal {
			phase := 2.0 * math.Pi * frequency * float64(i) / sampleRate
			signal[i] = math.Sin(phase)
		}
	case "impulse":
		signal[testSignalSamples/2] = 1.0
	}

	return signal
}

// findBestOffset finds the offset that maximizes correlation between two signals
func findBestOffset(got, want []float64) int {
	bestOffset := 0
	bestCorr := -2.0

	// Search range based on expected latency differences
	searchRange := 600

	for offset := -searchRange; offset <= searchRange; offset++ {
		corr := computeCorrelationWithSkip(got, want, offset, 200)
		if corr > bestCorr {
			bestCorr = corr
			bestOffset = offset
		}
	}

	return bestOffset
}

// computeCorrelationWithSkip computes Pearson correlation between two signals with a given offset.
// skipSamples specifies how many initial samples to skip (for transient avoidance).
// offset shifts signal 'a' relative to 'b' (positive = a is ahead of b).
// Returns correlation coefficient, or -2 if insufficient data.
func computeCorrelationWithSkip(a, b []float64, offset, skipSamples int) float64 {
	startA := skipSamples
	startB := skipSamples

	if offset > 0 {
		startA += offset
	} else {
		startB -= offset
	}

	n := 1000 // Compare 1000 samples
	if startA+n > len(a) || startB+n > len(b) {
		return -2
	}

	sumA, sumB, sumAB, sumA2, sumB2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := range n {
		va := a[startA+i]
		vb := b[startB+i]
		sumA += va
		sumB += vb
		sumAB += va * vb
		sumA2 += va * va
		sumB2 += vb * vb
	}

	meanA := sumA / float64(n)
	meanB := sumB / float64(n)

	num := sumAB - float64(n)*meanA*meanB
	denA := math.Sqrt(sumA2 - float64(n)*meanA*meanA)
	denB := math.Sqrt(sumB2 - float64(n)*meanB*meanB)

	if denA == 0 || denB == 0 {
		return 0
	}

	return num / (denA * denB)
}

// compareOutputs compares two output arrays with automatic alignment
// Returns max error and correlation after accounting for latency offset
func compareOutputs(got, want []float64) (maxErr, rmsErr, correlation float64, errIdx int) {
	if len(got) < 500 || len(want) < 500 {
		return 0, 0, 0, -1
	}

	// Find best alignment to account for latency differences
	offset := findBestOffset(got, want)

	// Compute aligned comparison
	startGot := 300
	startWant := 300

	if offset > 0 {
		startGot += offset
	} else {
		startWant -= offset
	}

	// Number of samples to compare
	n := 2000
	if startGot+n > len(got) {
		n = len(got) - startGot
	}
	if startWant+n > len(want) {
		n = len(want) - startWant
	}

	if n < 500 {
		return 0, 0, 0, -1
	}

	// Calculate errors on aligned region
	maxErr = 0
	errIdx = -1
	sumSqErr := 0.0
	sumGot := 0.0
	sumWant := 0.0
	sumGotWant := 0.0
	sumGotSq := 0.0
	sumWantSq := 0.0

	for i := 0; i < n; i++ {
		g := got[startGot+i]
		w := want[startWant+i]
		err := math.Abs(g - w)
		if err > maxErr {
			maxErr = err
			errIdx = startGot + i
		}
		sumSqErr += err * err

		sumGot += g
		sumWant += w
		sumGotWant += g * w
		sumGotSq += g * g
		sumWantSq += w * w
	}

	rmsErr = math.Sqrt(sumSqErr / float64(n))

	// Pearson correlation
	meanGot := sumGot / float64(n)
	meanWant := sumWant / float64(n)
	numerator := sumGotWant - float64(n)*meanGot*meanWant
	denomGot := math.Sqrt(sumGotSq - float64(n)*meanGot*meanGot)
	denomWant := math.Sqrt(sumWantSq - float64(n)*meanWant*meanWant)

	if denomGot > 0 && denomWant > 0 {
		correlation = numerator / (denomGot * denomWant)
	}

	return maxErr, rmsErr, correlation, errIdx
}

func TestSoxrComparison_DC(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
	}{
		{44100, 48000}, // CD to DAT
		{48000, 44100}, // DAT to CD
		{44100, 96000}, // 2x upsample
		{96000, 48000}, // 2x downsample
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%.0f_to_%.0f", tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			// Get soxr reference
			soxrOut, err := getSoxrReference(tc.inputRate, tc.outputRate, "dc", 0)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Run our resampler
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			if err != nil {
				t.Fatalf("NewResampler failed: %v", err)
			}

			input := generateTestSignal("dc", tc.inputRate, 0)
			output, err := resampler.Process(input)
			if err != nil {
				t.Fatalf("Process failed: %v", err)
			}
			flush, _ := resampler.Flush()
			output = append(output, flush...)

			// Compare
			maxErr, rmsErr, corr, errIdx := compareOutputs(output, soxrOut)

			t.Logf("Samples: got=%d, want=%d", len(output), len(soxrOut))
			t.Logf("Max error: %.6f at index %d", maxErr, errIdx)
			t.Logf("RMS error: %.6f", rmsErr)
			t.Logf("Correlation: %.6f", corr)

			// DC signals should have very low error
			if maxErr > dcTolerance {
				t.Errorf("DC max error %.6f exceeds tolerance %.6f", maxErr, dcTolerance)
			}
		})
	}
}

func TestSoxrComparison_Sine(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		frequency  float64
	}{
		{44100, 48000, 1000},  // 1 kHz
		{44100, 48000, 10000}, // 10 kHz
		{48000, 44100, 1000},  // downsample
		{44100, 96000, 5000},  // upsample
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%.0fHz_%.0f_to_%.0f", tc.frequency, tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			// Get soxr reference
			soxrOut, err := getSoxrReference(tc.inputRate, tc.outputRate, "sine", tc.frequency)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Run our resampler
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			if err != nil {
				t.Fatalf("NewResampler failed: %v", err)
			}

			input := generateTestSignal("sine", tc.inputRate, tc.frequency)
			output, err := resampler.Process(input)
			if err != nil {
				t.Fatalf("Process failed: %v", err)
			}
			flush, _ := resampler.Flush()
			output = append(output, flush...)

			// Compare
			maxErr, rmsErr, corr, errIdx := compareOutputs(output, soxrOut)

			t.Logf("Samples: got=%d, want=%d", len(output), len(soxrOut))
			t.Logf("Max error: %.6f at index %d", maxErr, errIdx)
			t.Logf("RMS error: %.6f", rmsErr)
			t.Logf("Correlation: %.6f", corr)

			// Determine thresholds based on frequency
			corrThreshold := correlationThreshold
			errThreshold := maxErrorThreshold
			if tc.frequency < 5000 {
				corrThreshold = lowFreqCorrelation
				errThreshold = lowFreqMaxError
			}

			// Correlation should be high for sine waves
			if corr < corrThreshold {
				t.Errorf("Sine correlation %.6f below %.2f", corr, corrThreshold)
			}

			// Max error should be reasonable (allows for filter design differences)
			if maxErr > errThreshold {
				t.Errorf("Sine max error %.6f exceeds tolerance %.2f", maxErr, errThreshold)
			}
		})
	}
}

func TestSoxrComparison_Impulse(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
	}{
		{44100, 48000},
		{48000, 44100},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%.0f_to_%.0f", tc.inputRate, tc.outputRate)
		t.Run(name, func(t *testing.T) {
			// Get soxr reference
			soxrOut, err := getSoxrReference(tc.inputRate, tc.outputRate, "impulse", 0)
			if err != nil {
				t.Skipf("soxr reference not available: %v", err)
			}

			// Run our resampler
			resampler, err := NewResampler[float64](tc.inputRate, tc.outputRate, QualityHigh)
			if err != nil {
				t.Fatalf("NewResampler failed: %v", err)
			}

			input := generateTestSignal("impulse", tc.inputRate, 0)
			output, err := resampler.Process(input)
			if err != nil {
				t.Fatalf("Process failed: %v", err)
			}
			flush, _ := resampler.Flush()
			output = append(output, flush...)

			// Compare correlation (impulse response shape)
			_, _, corr, _ := compareOutputs(output, soxrOut)

			t.Logf("Samples: got=%d, want=%d", len(output), len(soxrOut))
			t.Logf("Correlation: %.6f", corr)

			// Impulse response correlation should be reasonable
			// Note: different filter designs have different impulse responses
			// We use a lower threshold here since impulse responses can vary significantly
			if corr < 0.5 {
				t.Errorf("Impulse correlation %.6f below 0.5 (filter designs differ)", corr)
			}
		})
	}
}
