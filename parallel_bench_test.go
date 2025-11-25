package resampler

import (
	"testing"
)

// BenchmarkProcessMultiSequential benchmarks sequential multi-channel processing.
func BenchmarkProcessMultiSequential(b *testing.B) {
	benchmarkProcessMulti(b, false)
}

// BenchmarkProcessMultiParallel benchmarks parallel multi-channel processing.
func BenchmarkProcessMultiParallel(b *testing.B) {
	benchmarkProcessMulti(b, true)
}

func benchmarkProcessMulti(b *testing.B, parallel bool) {
	b.Helper()

	const (
		inputRate  = 44100.0
		outputRate = 48000.0
		channels   = 2     // Stereo
		numSamples = 44100 // 1 second of audio
	)

	config := &Config{
		InputRate:      inputRate,
		OutputRate:     outputRate,
		Channels:       channels,
		Quality:        QualitySpec{Preset: QualityHigh},
		EnableParallel: parallel,
	}

	resampler, err := New(config)
	if err != nil {
		b.Fatalf("Failed to create resampler: %v", err)
	}

	// Create stereo input data
	input := make([][]float64, channels)
	for ch := range channels {
		input[ch] = make([]float64, numSamples)
		for i := range numSamples {
			input[ch][i] = float64(i) / float64(numSamples) // Simple ramp
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := resampler.ProcessMulti(input)
		if err != nil {
			b.Fatalf("ProcessMulti failed: %v", err)
		}
		resampler.Reset()
	}
}

// BenchmarkProcessMultiChannels benchmarks parallel processing with varying channel counts.
func BenchmarkProcessMultiChannels(b *testing.B) {
	channelCounts := []int{1, 2, 4, 6, 8}

	for _, channels := range channelCounts {
		b.Run(channelName(channels), func(b *testing.B) {
			const (
				inputRate  = 44100.0
				outputRate = 48000.0
				numSamples = 44100 // 1 second of audio
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
				b.Fatalf("Failed to create resampler: %v", err)
			}

			// Create multi-channel input data
			input := make([][]float64, channels)
			for ch := range channels {
				input[ch] = make([]float64, numSamples)
				for i := range numSamples {
					input[ch][i] = float64(i) / float64(numSamples)
				}
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := resampler.ProcessMulti(input)
				if err != nil {
					b.Fatalf("ProcessMulti failed: %v", err)
				}
				resampler.Reset()
			}
		})
	}
}

func channelName(channels int) string {
	switch channels {
	case 1:
		return "Mono"
	case 2:
		return "Stereo"
	case 4:
		return "Quad"
	case 6:
		return "5.1"
	case 8:
		return "7.1"
	default:
		return "Custom"
	}
}
