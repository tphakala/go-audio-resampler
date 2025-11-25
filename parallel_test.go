package resampler

import (
	"math"
	"testing"
)

// TestProcessMultiParallel tests that parallel processing produces correct results.
func TestProcessMultiParallel(t *testing.T) {
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		channels   = 2
		numSamples = 4410 // 0.1 seconds
		freq       = 440.0
	)

	// Create stereo sine wave input
	input := make([][]float64, channels)
	for ch := range channels {
		input[ch] = make([]float64, numSamples)
		for i := range numSamples {
			// Use different phases for each channel to ensure they're processed independently
			phase := float64(ch) * math.Pi / 4
			input[ch][i] = math.Sin(2*math.Pi*freq*float64(i)/inputRate + phase)
		}
	}

	// Create resamplers with and without parallel processing
	configSeq := &Config{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		Channels:       channels,
		Quality:        QualitySpec{Preset: QualityHigh},
		EnableParallel: false,
	}
	configPar := &Config{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		Channels:       channels,
		Quality:        QualitySpec{Preset: QualityHigh},
		EnableParallel: true,
	}

	resamplerSeq, err := New(configSeq)
	if err != nil {
		t.Fatalf("Failed to create sequential resampler: %v", err)
	}

	resamplerPar, err := New(configPar)
	if err != nil {
		t.Fatalf("Failed to create parallel resampler: %v", err)
	}

	// Process with both
	outputSeq, err := resamplerSeq.ProcessMulti(input)
	if err != nil {
		t.Fatalf("Sequential ProcessMulti failed: %v", err)
	}

	outputPar, err := resamplerPar.ProcessMulti(input)
	if err != nil {
		t.Fatalf("Parallel ProcessMulti failed: %v", err)
	}

	// Verify outputs have same length
	if len(outputSeq) != len(outputPar) {
		t.Fatalf("Channel count mismatch: seq=%d, par=%d", len(outputSeq), len(outputPar))
	}

	for ch := range channels {
		if len(outputSeq[ch]) != len(outputPar[ch]) {
			t.Fatalf("Channel %d length mismatch: seq=%d, par=%d",
				ch, len(outputSeq[ch]), len(outputPar[ch]))
		}

		// Verify outputs are identical (bit-exact)
		for i := range outputSeq[ch] {
			if outputSeq[ch][i] != outputPar[ch][i] {
				t.Errorf("Channel %d sample %d mismatch: seq=%v, par=%v",
					ch, i, outputSeq[ch][i], outputPar[ch][i])
				break // Don't flood with errors
			}
		}
	}
}

// TestProcessMultiChannelIndependence verifies channels are processed independently.
func TestProcessMultiChannelIndependence(t *testing.T) {
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		channels   = 2
		numSamples = 4410
	)

	config := &Config{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		Channels:       channels,
		Quality:        QualitySpec{Preset: QualityHigh},
		EnableParallel: true,
	}

	resampler, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	// Create input where one channel is silence and another is a signal
	input := make([][]float64, channels)
	input[0] = make([]float64, numSamples) // Silent channel (all zeros)
	input[1] = make([]float64, numSamples)
	for i := range numSamples {
		input[1][i] = math.Sin(2 * math.Pi * 440.0 * float64(i) / inputRate)
	}

	output, err := resampler.ProcessMulti(input)
	if err != nil {
		t.Fatalf("ProcessMulti failed: %v", err)
	}

	// Verify channel 0 is still essentially silent (near zero)
	var maxCh0 float64
	for _, v := range output[0] {
		if math.Abs(v) > maxCh0 {
			maxCh0 = math.Abs(v)
		}
	}
	if maxCh0 > 1e-10 {
		t.Errorf("Silent channel has non-zero output: max=%v", maxCh0)
	}

	// Verify channel 1 has signal
	var maxCh1 float64
	for _, v := range output[1] {
		if math.Abs(v) > maxCh1 {
			maxCh1 = math.Abs(v)
		}
	}
	if maxCh1 < 0.9 {
		t.Errorf("Signal channel has too low amplitude: max=%v", maxCh1)
	}
}

// TestProcessMultiMonoFallback verifies mono processing works with parallel enabled.
func TestProcessMultiMonoFallback(t *testing.T) {
	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		channels   = 1
		numSamples = 4410
	)

	config := &Config{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		Channels:       channels,
		Quality:        QualitySpec{Preset: QualityHigh},
		EnableParallel: true, // Should fall back to sequential for mono
	}

	resampler, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create resampler: %v", err)
	}

	input := make([][]float64, channels)
	input[0] = make([]float64, numSamples)
	for i := range numSamples {
		input[0][i] = math.Sin(2 * math.Pi * 440.0 * float64(i) / inputRate)
	}

	output, err := resampler.ProcessMulti(input)
	if err != nil {
		t.Fatalf("ProcessMulti failed: %v", err)
	}

	// Verify output exists and has reasonable length
	// Note: Output may be shorter due to internal buffering (filter latency)
	expectedLen := int(float64(numSamples) * outputRate / inputRate)
	if len(output[0]) < expectedLen/2 || len(output[0]) > expectedLen*2 {
		t.Errorf("Unexpected output length: got=%d, expected~%d", len(output[0]), expectedLen)
	}
}
