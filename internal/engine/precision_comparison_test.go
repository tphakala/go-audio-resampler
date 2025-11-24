package engine

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// =============================================================================
// Float32 vs Float64 Precision Comparison Tests
// =============================================================================
//
// These tests compare quality and performance between float32 and float64 paths.
// Run with: go test -v -run=TestPrecision ./internal/engine/
// Benchmarks: GOAMD64=v3 go test -bench=BenchmarkPrecision -benchtime=2s ./internal/engine/

// -----------------------------------------------------------------------------
// Quality Comparison Tests
// -----------------------------------------------------------------------------

// TestPrecisionComparison_DCGain compares DC gain accuracy between float32 and float64.
func TestPrecisionComparison_DCGain(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		quality    Quality
	}{
		{44100, 48000, QualityVeryHigh},
		{48000, 44100, QualityVeryHigh},
		{48000, 32000, QualityVeryHigh},
		{44100, 48000, QualityQuick},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%s/%.0fkTo%.0fk", qualityName(tc.quality), tc.inputRate/1000, tc.outputRate/1000)
		t.Run(name, func(t *testing.T) {
			// Test float64
			resampler64, err := NewResampler[float64](tc.inputRate, tc.outputRate, tc.quality)
			if err != nil {
				t.Fatalf("NewResampler[float64] failed: %v", err)
			}

			// Test float32
			resampler32, err := NewResampler[float32](tc.inputRate, tc.outputRate, tc.quality)
			if err != nil {
				t.Fatalf("NewResampler[float32] failed: %v", err)
			}

			// Generate DC signal
			input64 := make([]float64, 20000)
			input32 := make([]float32, 20000)
			for i := range input64 {
				input64[i] = 1.0
				input32[i] = 1.0
			}

			// Process
			output64, _ := resampler64.Process(input64)
			flush64, _ := resampler64.Flush()
			output64 = append(output64, flush64...)

			output32, _ := resampler32.Process(input32)
			flush32, _ := resampler32.Flush()
			output32 = append(output32, flush32...)

			// Measure DC gain in stable region
			dcGain64 := measureDCGain(output64)
			dcGain32 := measureDCGainF32(output32)

			t.Logf("DC Gain - float64: %.6f, float32: %.6f, diff: %.6f",
				dcGain64, dcGain32, math.Abs(dcGain64-dcGain32))

			// Both should be close to 1.0
			if math.Abs(dcGain64-1.0) > 0.01 {
				t.Errorf("float64 DC gain %.6f deviates from 1.0", dcGain64)
			}
			if math.Abs(dcGain32-1.0) > 0.01 {
				t.Errorf("float32 DC gain %.6f deviates from 1.0", dcGain32)
			}
		})
	}
}

// TestPrecisionComparison_THD compares Total Harmonic Distortion between precisions.
func TestPrecisionComparison_THD(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		quality    Quality
	}{
		{44100, 48000, QualityVeryHigh},
		{48000, 44100, QualityVeryHigh},
		{48000, 32000, QualityVeryHigh},
		{44100, 48000, QualityHigh},
		{44100, 48000, QualityMedium},
		{44100, 48000, QualityQuick},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%s/%.0fkTo%.0fk", qualityName(tc.quality), tc.inputRate/1000, tc.outputRate/1000)
		t.Run(name, func(t *testing.T) {
			testFreq := 1000.0

			// Generate test signal
			numSamples := int(tc.inputRate) // 1 second
			input64 := make([]float64, numSamples)
			input32 := make([]float32, numSamples)
			for i := range numSamples {
				val := math.Sin(2.0 * math.Pi * testFreq * float64(i) / tc.inputRate)
				input64[i] = val
				input32[i] = float32(val)
			}

			// Process float64
			resampler64, _ := NewResampler[float64](tc.inputRate, tc.outputRate, tc.quality)
			output64, _ := resampler64.Process(input64)
			flush64, _ := resampler64.Flush()
			output64 = append(output64, flush64...)

			// Process float32
			resampler32, _ := NewResampler[float32](tc.inputRate, tc.outputRate, tc.quality)
			output32, _ := resampler32.Process(input32)
			flush32, _ := resampler32.Flush()
			output32 = append(output32, flush32...)

			// Measure THD
			thd64 := precisionMeasureTHD(output64, testFreq, tc.outputRate)
			thd32 := precisionMeasureTHDF32(output32, testFreq, tc.outputRate)

			t.Logf("THD @ %.0f Hz - float64: %.2f dB, float32: %.2f dB, diff: %.2f dB",
				testFreq, thd64, thd32, thd32-thd64)

			// Log quality assessment
			if thd32-thd64 > 10 {
				t.Logf("WARNING: float32 THD is %.1f dB worse than float64", thd32-thd64)
			}
		})
	}
}

// TestPrecisionComparison_SNR compares Signal-to-Noise Ratio between precisions.
func TestPrecisionComparison_SNR(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		quality    Quality
	}{
		{44100, 48000, QualityVeryHigh},
		{48000, 44100, QualityVeryHigh},
		{48000, 32000, QualityVeryHigh},
		{44100, 48000, QualityQuick},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%s/%.0fkTo%.0fk", qualityName(tc.quality), tc.inputRate/1000, tc.outputRate/1000)
		t.Run(name, func(t *testing.T) {
			testFreq := 1000.0

			// Generate test signal
			numSamples := int(tc.inputRate)
			input64 := make([]float64, numSamples)
			input32 := make([]float32, numSamples)
			for i := range numSamples {
				val := math.Sin(2.0 * math.Pi * testFreq * float64(i) / tc.inputRate)
				input64[i] = val
				input32[i] = float32(val)
			}

			// Process
			resampler64, _ := NewResampler[float64](tc.inputRate, tc.outputRate, tc.quality)
			output64, _ := resampler64.Process(input64)
			flush64, _ := resampler64.Flush()
			output64 = append(output64, flush64...)

			resampler32, _ := NewResampler[float32](tc.inputRate, tc.outputRate, tc.quality)
			output32, _ := resampler32.Process(input32)
			flush32, _ := resampler32.Flush()
			output32 = append(output32, flush32...)

			// Measure SNR
			snr64 := precisionMeasureSNR(output64, testFreq, tc.outputRate)
			snr32 := precisionMeasureSNRF32(output32, testFreq, tc.outputRate)

			t.Logf("SNR @ %.0f Hz - float64: %.2f dB, float32: %.2f dB, diff: %.2f dB",
				testFreq, snr64, snr32, snr64-snr32)
		})
	}
}

// TestPrecisionComparison_PassbandRipple compares passband ripple between precisions.
func TestPrecisionComparison_PassbandRipple(t *testing.T) {
	testCases := []struct {
		inputRate  float64
		outputRate float64
		quality    Quality
	}{
		{44100, 48000, QualityVeryHigh},
		{48000, 44100, QualityVeryHigh},
		{44100, 48000, QualityQuick},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%s/%.0fkTo%.0fk", qualityName(tc.quality), tc.inputRate/1000, tc.outputRate/1000)
		t.Run(name, func(t *testing.T) {
			ripple64 := precisionMeasurePassbandRipple[float64](tc.inputRate, tc.outputRate, tc.quality)
			ripple32 := precisionMeasurePassbandRipple[float32](tc.inputRate, tc.outputRate, tc.quality)

			t.Logf("Passband Ripple - float64: %.4f dB, float32: %.4f dB, diff: %.4f dB",
				ripple64, ripple32, ripple32-ripple64)
		})
	}
}

// TestPrecisionComparison_Summary provides a comprehensive summary.
func TestPrecisionComparison_Summary(t *testing.T) {
	t.Log("===========================================")
	t.Log("Float32 vs Float64 Precision Summary")
	t.Log("===========================================")
	t.Log("")

	conversions := []struct {
		inputRate  float64
		outputRate float64
	}{
		{44100, 48000},
		{48000, 44100},
		{48000, 32000},
	}

	qualities := []Quality{QualityQuick, QualityMedium, QualityHigh, QualityVeryHigh}

	t.Log("Conversion      | Quality   | THD64 (dB) | THD32 (dB) | SNR64 (dB) | SNR32 (dB)")
	t.Log("----------------|-----------|------------|------------|------------|------------")

	for _, conv := range conversions {
		for _, q := range qualities {
			testFreq := 1000.0
			numSamples := int(conv.inputRate)

			// Generate input
			input64 := make([]float64, numSamples)
			input32 := make([]float32, numSamples)
			for i := range numSamples {
				val := math.Sin(2.0 * math.Pi * testFreq * float64(i) / conv.inputRate)
				input64[i] = val
				input32[i] = float32(val)
			}

			// Process
			resampler64, _ := NewResampler[float64](conv.inputRate, conv.outputRate, q)
			output64, _ := resampler64.Process(input64)
			flush64, _ := resampler64.Flush()
			output64 = append(output64, flush64...)

			resampler32, _ := NewResampler[float32](conv.inputRate, conv.outputRate, q)
			output32, _ := resampler32.Process(input32)
			flush32, _ := resampler32.Flush()
			output32 = append(output32, flush32...)

			// Measure
			thd64 := precisionMeasureTHD(output64, testFreq, conv.outputRate)
			thd32 := precisionMeasureTHDF32(output32, testFreq, conv.outputRate)
			snr64 := precisionMeasureSNR(output64, testFreq, conv.outputRate)
			snr32 := precisionMeasureSNRF32(output32, testFreq, conv.outputRate)

			convName := fmt.Sprintf("%.0fk->%.0fk", conv.inputRate/1000, conv.outputRate/1000)
			t.Logf("%-15s | %-9s | %10.2f | %10.2f | %10.2f | %10.2f",
				convName, qualityName(q), thd64, thd32, snr64, snr32)
		}
	}

	t.Log("")
	t.Log("Notes:")
	t.Log("- THD: More negative = better (less distortion)")
	t.Log("- SNR: More positive = better (cleaner signal)")
	t.Log("- float32 typically has ~6 dB less dynamic range than float64")
}

// -----------------------------------------------------------------------------
// Throughput Benchmarks
// -----------------------------------------------------------------------------

// BenchmarkPrecision_Float64_44kTo48k benchmarks float64 44.1k -> 48k conversion.
func BenchmarkPrecision_Float64_44kTo48k_VeryHigh(b *testing.B) {
	benchPrecision[float64](b, 44100, 48000, QualityVeryHigh)
}

func BenchmarkPrecision_Float32_44kTo48k_VeryHigh(b *testing.B) {
	benchPrecision[float32](b, 44100, 48000, QualityVeryHigh)
}

func BenchmarkPrecision_Float64_48kTo32k_VeryHigh(b *testing.B) {
	benchPrecision[float64](b, 48000, 32000, QualityVeryHigh)
}

func BenchmarkPrecision_Float32_48kTo32k_VeryHigh(b *testing.B) {
	benchPrecision[float32](b, 48000, 32000, QualityVeryHigh)
}

func BenchmarkPrecision_Float64_48kTo44k_VeryHigh(b *testing.B) {
	benchPrecision[float64](b, 48000, 44100, QualityVeryHigh)
}

func BenchmarkPrecision_Float32_48kTo44k_VeryHigh(b *testing.B) {
	benchPrecision[float32](b, 48000, 44100, QualityVeryHigh)
}

func BenchmarkPrecision_Float64_44kTo48k_Quick(b *testing.B) {
	benchPrecision[float64](b, 44100, 48000, QualityQuick)
}

func BenchmarkPrecision_Float32_44kTo48k_Quick(b *testing.B) {
	benchPrecision[float32](b, 44100, 48000, QualityQuick)
}

func BenchmarkPrecision_Float64_48kTo32k_Quick(b *testing.B) {
	benchPrecision[float64](b, 48000, 32000, QualityQuick)
}

func BenchmarkPrecision_Float32_48kTo32k_Quick(b *testing.B) {
	benchPrecision[float32](b, 48000, 32000, QualityQuick)
}

// benchPrecision is the generic benchmark helper.
func benchPrecision[F float32 | float64](b *testing.B, inputRate, outputRate float64, quality Quality) {
	b.Helper()

	resampler, err := NewResampler[F](inputRate, outputRate, quality)
	if err != nil {
		b.Fatal(err)
	}

	// 1 second of input audio
	inputLen := int(inputRate)
	input := make([]F, inputLen)
	for i := range input {
		input[i] = F(math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / inputRate))
	}

	b.ResetTimer()

	var totalSamples int64
	for i := 0; i < b.N; i++ {
		resampler.Reset()
		output, err := resampler.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		totalSamples += int64(len(input))
		_ = output
	}

	b.ReportMetric(float64(totalSamples)/b.Elapsed().Seconds()/1e6, "MS/s")
}

// -----------------------------------------------------------------------------
// Stage-Level Benchmarks
// -----------------------------------------------------------------------------

func BenchmarkPrecision_PolyphaseStage_Float64_VeryHigh(b *testing.B) {
	benchPolyphasePrecision[float64](b, QualityVeryHigh, 48000.0/32000.0)
}

func BenchmarkPrecision_PolyphaseStage_Float32_VeryHigh(b *testing.B) {
	benchPolyphasePrecision[float32](b, QualityVeryHigh, 48000.0/32000.0)
}

func BenchmarkPrecision_DFTStage_Float64_VeryHigh(b *testing.B) {
	benchDFTPrecision[float64](b, QualityVeryHigh)
}

func BenchmarkPrecision_DFTStage_Float32_VeryHigh(b *testing.B) {
	benchDFTPrecision[float32](b, QualityVeryHigh)
}

func benchPolyphasePrecision[F float32 | float64](b *testing.B, quality Quality, ratio float64) {
	b.Helper()

	totalIORatio := 1.0 / ratio
	stage, err := NewPolyphaseStage[F](ratio, totalIORatio, false, quality)
	if err != nil {
		b.Fatal(err)
	}

	inputLen := 48000
	input := make([]F, inputLen)
	for i := range input {
		input[i] = F(math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / 48000.0))
	}

	b.ResetTimer()

	var totalSamples int64
	for i := 0; i < b.N; i++ {
		stage.Reset()
		output, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		totalSamples += int64(len(input))
		_ = output
	}

	b.ReportMetric(float64(totalSamples)/b.Elapsed().Seconds()/1e6, "MS/s")
}

func benchDFTPrecision[F float32 | float64](b *testing.B, quality Quality) {
	b.Helper()

	stage, err := NewDFTStage[F](2, quality)
	if err != nil {
		b.Fatal(err)
	}

	inputLen := 48000
	input := make([]F, inputLen)
	for i := range input {
		input[i] = F(math.Sin(2.0 * math.Pi * 1000.0 * float64(i) / 48000.0))
	}

	b.ResetTimer()

	var totalSamples int64
	for i := 0; i < b.N; i++ {
		stage.Reset()
		output, err := stage.Process(input)
		if err != nil {
			b.Fatal(err)
		}
		totalSamples += int64(len(input))
		_ = output
	}

	b.ReportMetric(float64(totalSamples)/b.Elapsed().Seconds()/1e6, "MS/s")
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

func measureDCGain[F float32 | float64](output []F) float64 {
	if len(output) < 4 {
		return 0
	}
	startIdx := len(output) / 4
	endIdx := 3 * len(output) / 4
	sum := 0.0
	for i := startIdx; i < endIdx; i++ {
		sum += float64(output[i])
	}
	return sum / float64(endIdx-startIdx)
}

func measureDCGainF32(output []float32) float64 {
	if len(output) < 4 {
		return 0
	}
	startIdx := len(output) / 4
	endIdx := 3 * len(output) / 4
	sum := 0.0
	for i := startIdx; i < endIdx; i++ {
		sum += float64(output[i])
	}
	return sum / float64(endIdx-startIdx)
}

func precisionMeasureTHDF32(output []float32, testFreq, sampleRate float64) float64 {
	// Convert to float64 for FFT analysis
	output64 := make([]float64, len(output))
	for i, v := range output {
		output64[i] = float64(v)
	}
	return precisionMeasureTHD(output64, testFreq, sampleRate)
}

func precisionMeasureSNRF32(output []float32, testFreq, sampleRate float64) float64 {
	output64 := make([]float64, len(output))
	for i, v := range output {
		output64[i] = float64(v)
	}
	return precisionMeasureSNR(output64, testFreq, sampleRate)
}

func precisionMeasurePassbandRipple[F float32 | float64](inputRate, outputRate float64, quality Quality) float64 {
	resampler, err := NewResampler[F](inputRate, outputRate, quality)
	if err != nil {
		return 0
	}

	// Test frequencies in passband (up to 80% of output Nyquist)
	maxFreq := outputRate * 0.4
	numFreqs := 20
	gains := make([]float64, numFreqs)

	for i := range numFreqs {
		freq := 100.0 + float64(i)*(maxFreq-100.0)/float64(numFreqs-1)

		// Generate test tone
		numSamples := int(inputRate)
		input := make([]F, numSamples)
		for j := range numSamples {
			input[j] = F(math.Sin(2.0 * math.Pi * freq * float64(j) / inputRate))
		}

		resampler.Reset()
		output, _ := resampler.Process(input)
		flush, _ := resampler.Flush()
		output = append(output, flush...)

		// Measure output amplitude
		gains[i] = measureAmplitude(output)
	}

	// Calculate ripple (max - min in dB)
	maxGain := gains[0]
	minGain := gains[0]
	for _, g := range gains {
		if g > maxGain {
			maxGain = g
		}
		if g < minGain {
			minGain = g
		}
	}

	if minGain <= 0 {
		return 100.0 // Very bad
	}
	return 20 * math.Log10(maxGain/minGain)
}

func measureAmplitude[F float32 | float64](output []F) float64 {
	if len(output) < 100 {
		return 0
	}
	// Use middle portion to avoid edge effects
	start := len(output) / 4
	end := 3 * len(output) / 4

	maxVal := float64(0)
	for i := start; i < end; i++ {
		v := math.Abs(float64(output[i]))
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// precisionMeasureTHD calculates Total Harmonic Distortion using FFT.
func precisionMeasureTHD(output []float64, testFreq, sampleRate float64) float64 {
	if len(output) < 1024 {
		return 0
	}

	// Use a power-of-2 FFT size from the middle of the signal
	fftSize := 8192
	if len(output) < fftSize*2 {
		fftSize = 1024
	}

	start := (len(output) - fftSize) / 2
	segment := output[start : start+fftSize]

	// Apply Hann window
	windowed := make([]complex128, fftSize)
	for i := range fftSize {
		window := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(fftSize-1)))
		windowed[i] = complex(segment[i]*window, 0)
	}

	// Simple DFT (for accuracy, not speed)
	spectrum := make([]complex128, fftSize)
	for k := range fftSize {
		for n := range fftSize {
			angle := -2 * math.Pi * float64(k) * float64(n) / float64(fftSize)
			spectrum[k] += windowed[n] * cmplx.Exp(complex(0, angle))
		}
	}

	// Find fundamental and harmonics
	binWidth := sampleRate / float64(fftSize)
	fundBin := int(math.Round(testFreq / binWidth))

	fundMag := cmplx.Abs(spectrum[fundBin])
	if fundMag < 1e-10 {
		return -200
	}

	// Sum harmonic power (2nd through 5th)
	harmonicPower := 0.0
	for h := 2; h <= 5; h++ {
		harmBin := fundBin * h
		if harmBin < fftSize/2 {
			mag := cmplx.Abs(spectrum[harmBin])
			harmonicPower += mag * mag
		}
	}

	if harmonicPower < 1e-30 {
		return -200
	}

	thd := 10 * math.Log10(harmonicPower/(fundMag*fundMag))
	return thd
}

// precisionMeasureSNR calculates Signal-to-Noise Ratio.
func precisionMeasureSNR(output []float64, testFreq, sampleRate float64) float64 {
	if len(output) < 1024 {
		return 0
	}

	fftSize := 8192
	if len(output) < fftSize*2 {
		fftSize = 1024
	}

	start := (len(output) - fftSize) / 2
	segment := output[start : start+fftSize]

	// Apply Hann window
	windowed := make([]complex128, fftSize)
	for i := range fftSize {
		window := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(fftSize-1)))
		windowed[i] = complex(segment[i]*window, 0)
	}

	// Simple DFT
	spectrum := make([]complex128, fftSize)
	for k := range fftSize {
		for n := range fftSize {
			angle := -2 * math.Pi * float64(k) * float64(n) / float64(fftSize)
			spectrum[k] += windowed[n] * cmplx.Exp(complex(0, angle))
		}
	}

	binWidth := sampleRate / float64(fftSize)
	fundBin := int(math.Round(testFreq / binWidth))

	// Signal power (fundamental + nearby bins)
	signalPower := 0.0
	for b := fundBin - 2; b <= fundBin+2; b++ {
		if b >= 0 && b < fftSize/2 {
			mag := cmplx.Abs(spectrum[b])
			signalPower += mag * mag
		}
	}

	// Noise power (everything else up to Nyquist)
	noisePower := 0.0
	for b := 10; b < fftSize/2; b++ {
		if b < fundBin-5 || b > fundBin+5 {
			mag := cmplx.Abs(spectrum[b])
			noisePower += mag * mag
		}
	}

	if noisePower < 1e-30 {
		return 200
	}

	return 10 * math.Log10(signalPower/noisePower)
}
