package resampler

import (
	"math"
	"testing"
)

func FuzzResamplerNew(f *testing.F) {
	// Common sample rate pairs
	f.Add(44100.0, 48000.0, 1, 0)   // CD to DAT
	f.Add(48000.0, 44100.0, 1, 0)   // DAT to CD
	f.Add(44100.0, 88200.0, 2, 1)   // 2x upsample
	f.Add(48000.0, 16000.0, 1, 2)   // downsample
	f.Add(8000.0, 48000.0, 1, 3)    // telephony to DAT
	f.Add(44100.0, 48000.0, 6, 4)   // 5.1 surround

	// Edge cases
	f.Add(1.0, 256.0, 1, 0)           // extreme upsample (at ratio limit)
	f.Add(256.0, 1.0, 1, 0)           // extreme downsample
	f.Add(0.0, 48000.0, 1, 0)         // zero input rate
	f.Add(44100.0, 0.0, 1, 0)         // zero output rate
	f.Add(-44100.0, 48000.0, 1, 0)    // negative rate
	f.Add(44100.0, 48000.0, 0, 0)     // zero channels
	f.Add(44100.0, 48000.0, 300, 0)   // too many channels

	f.Fuzz(func(t *testing.T, inputRate, outputRate float64, channels int, presetInt int) {
		// Skip values that would cause issues unrelated to what we're testing
		if math.IsNaN(inputRate) || math.IsNaN(outputRate) {
			return
		}
		if math.IsInf(inputRate, 0) || math.IsInf(outputRate, 0) {
			return
		}

		// Map presetInt to valid preset range
		preset := QualityPreset(presetInt % 6)
		if preset < 0 {
			preset = -preset
		}

		config := &Config{
			InputRate:  inputRate,
			OutputRate: outputRate,
			Channels:   channels,
			Quality:    QualitySpec{Preset: preset},
		}

		r, err := New(config)
		if err != nil {
			// Validation rejected it — that's fine
			return
		}

		// If creation succeeded, verify basic properties
		ratio := r.GetRatio()
		if math.IsNaN(ratio) || math.IsInf(ratio, 0) {
			t.Errorf("GetRatio() = %v for rates %v -> %v", ratio, inputRate, outputRate)
		}
		if ratio <= 0 {
			t.Errorf("GetRatio() = %v, want positive", ratio)
		}

		latency := r.GetLatency()
		if latency < 0 {
			t.Errorf("GetLatency() = %v, want non-negative", latency)
		}
	})
}

func FuzzResampleMono(f *testing.F) {
	f.Add(44100.0, 48000.0, 0, 100)  // CD to DAT, 100 samples
	f.Add(48000.0, 44100.0, 1, 1000) // DAT to CD
	f.Add(44100.0, 88200.0, 2, 50)   // 2x upsample
	f.Add(48000.0, 16000.0, 0, 500)  // 3x downsample

	f.Fuzz(func(t *testing.T, inputRate, outputRate float64, presetInt int, numSamples int) {
		// Bound inputs to reasonable ranges to avoid OOM
		if numSamples < 0 || numSamples > 50000 {
			return
		}
		if inputRate <= 0 || outputRate <= 0 {
			return
		}
		if math.IsNaN(inputRate) || math.IsNaN(outputRate) {
			return
		}
		if math.IsInf(inputRate, 0) || math.IsInf(outputRate, 0) {
			return
		}

		ratio := outputRate / inputRate
		if ratio < 1.0/256.0 || ratio > 256.0 {
			return
		}

		preset := QualityPreset(presetInt % 5)
		if preset < 0 {
			preset = -preset
		}

		input := make([]float64, numSamples)
		for i := range input {
			// Generate a simple sine wave
			input[i] = math.Sin(2 * math.Pi * 1000 * float64(i) / inputRate)
		}

		output, err := ResampleMono(input, inputRate, outputRate, preset)
		if err != nil {
			return
		}

		// Verify output properties
		for i, v := range output {
			if math.IsNaN(v) {
				t.Errorf("output[%d] is NaN", i)
				break
			}
			if math.IsInf(v, 0) {
				t.Errorf("output[%d] is Inf", i)
				break
			}
		}

		// Output should not be empty for non-empty input (flush includes latency samples)
		if numSamples > 0 && len(output) == 0 {
			t.Errorf("empty output for %d input samples at ratio %v", numSamples, ratio)
		}
	})
}
