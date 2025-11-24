package main

import (
	"flag"
	"fmt"
	"log"
	"math"

	resampling "github.com/tphakala/go-audio-resampler"
)

func main() {
	// Command-line flags
	var (
		inputRate  = flag.Float64("input-rate", defaultInputRate, "Input sample rate in Hz")
		outputRate = flag.Float64("output-rate", defaultOutputRate, "Output sample rate in Hz")
		channels   = flag.Int("channels", defaultChannels, "Number of audio channels")
		quality    = flag.String("quality", "high", "Quality preset: quick, low, medium, high, veryhigh")
		demo       = flag.Bool("demo", false, "Run a demonstration")
	)
	flag.Parse()

	if *demo {
		runDemo()
		return
	}

	// Parse quality preset
	qualityPreset := parseQuality(*quality)

	// Create resampler configuration
	config := resampling.Config{
		InputRate:  *inputRate,
		OutputRate: *outputRate,
		Channels:   *channels,
		Quality: resampling.QualitySpec{
			Preset: qualityPreset,
		},
		EnableSIMD: true,
	}

	// Create resampler
	resampler, err := resampling.New(&config)
	if err != nil {
		log.Fatalf("Failed to create resampler: %v", err)
	}

	// Get resampler info
	info := resampling.GetInfo(resampler)
	fmt.Printf("Resampler created:\n")
	fmt.Printf("  Algorithm: %s\n", info.Algorithm)
	fmt.Printf("  Ratio: %.6f (%g Hz � %g Hz)\n", resampler.GetRatio(), *inputRate, *outputRate)
	fmt.Printf("  Filter length: %d taps\n", info.FilterLength)
	fmt.Printf("  Phases: %d\n", info.Phases)
	fmt.Printf("  Latency: %d samples\n", info.Latency)
	fmt.Printf("  Memory usage: %.2f KB\n", float64(info.MemoryUsage)/bytesPerKilobyte)
	fmt.Printf("  SIMD: %v (%s)\n", info.SIMDEnabled, info.SIMDType)

	// Example: process a test signal
	fmt.Println("\nProcessing test signal...")
	testSignal := generateTestSignal(testSignalSamples, *inputRate)
	output, err := resampler.Process(testSignal)
	if err != nil {
		log.Fatalf("Processing failed: %v", err)
	}

	fmt.Printf("Input samples: %d\n", len(testSignal))
	fmt.Printf("Output samples: %d\n", len(output))
	fmt.Printf("Expected output: %d\n", int(float64(len(testSignal))*resampler.GetRatio()))
}

func parseQuality(s string) resampling.QualityPreset {
	switch s {
	case "quick":
		return resampling.QualityQuick
	case "low":
		return resampling.QualityLow
	case "medium":
		return resampling.QualityMedium
	case "high":
		return resampling.QualityHigh
	case "veryhigh", "very-high":
		return resampling.QualityVeryHigh
	default:
		return resampling.QualityHigh
	}
}

func generateTestSignal(samples int, sampleRate float64) []float64 {
	signal := make([]float64, samples)

	// Generate a 1kHz sine wave
	frequency := testSignalFrequency
	omega := 2 * math.Pi * frequency / sampleRate

	for i := range signal {
		signal[i] = math.Sin(omega * float64(i))
	}

	return signal
}

func runDemo() {
	fmt.Println("=== Go Audio Resampling Library Demo ===")

	// Demo 1: Different quality levels
	fmt.Println("1. Comparing Quality Levels")
	fmt.Println("----------------------------")

	testRatios := []struct {
		from, to float64
		name     string
	}{
		{sampleRateCD, sampleRateDAT, "CD to DAT"},
		{sampleRateDAT, sampleRateCD, "DAT to CD"},
		{sampleRateCD, sampleRate2xCD, "CD to 2x"},
		{sampleRateHiRes, sampleRateCD, "Hi-res to CD"},
	}

	qualities := []resampling.QualityPreset{
		resampling.QualityQuick,
		resampling.QualityMedium,
		resampling.QualityHigh,
	}

	qualityNames := []string{"Quick", "Medium", "High"}

	for _, ratio := range testRatios {
		fmt.Printf("\n%s (%.0f Hz � %.0f Hz, ratio: %.4f):\n",
			ratio.name, ratio.from, ratio.to, ratio.to/ratio.from)

		for i, q := range qualities {
			config := resampling.Config{
				InputRate:  ratio.from,
				OutputRate: ratio.to,
				Channels:   stereoChannels,
				Quality:    resampling.QualitySpec{Preset: q},
			}

			resampler, err := resampling.New(&config)
			if err != nil {
				fmt.Printf("  %s: Error - %v\n", qualityNames[i], err)
				continue
			}

			info := resampling.GetInfo(resampler)
			fmt.Printf("  %s: %d taps, %d samples latency, %.1f KB memory\n",
				qualityNames[i],
				info.FilterLength,
				info.Latency,
				float64(info.MemoryUsage)/bytesPerKilobyte)
		}
	}

	// Demo 2: Performance characteristics
	fmt.Println("\n2. Performance Characteristics")
	fmt.Println("------------------------------")

	fmt.Println("Processing 1 second of stereo audio (44.1kHz � 48kHz):")

	inputSize := int(sampleRateCD)
	testSignal := generateTestSignal(inputSize, sampleRateCD)

	for i, q := range qualities {
		config := resampling.Config{
			InputRate:  sampleRateCD,
			OutputRate: sampleRateDAT,
			Channels:   stereoChannels,
			Quality:    resampling.QualitySpec{Preset: q},
		}

		resampler, err := resampling.New(&config)
		if err != nil {
			continue
		}

		// Process test signal
		output, _ := resampler.Process(testSignal)

		fmt.Printf("  %s: %d � %d samples\n",
			qualityNames[i], inputSize, len(output))
	}

	// Demo 3: Multi-channel processing
	fmt.Println("\n3. Multi-channel Processing")
	fmt.Println("---------------------------")

	channelCounts := []int{monoChannels, stereoChannels, surround5_1, surround7_1}

	for _, ch := range channelCounts {
		config := resampling.Config{
			InputRate:  sampleRateDAT,
			OutputRate: sampleRateCD,
			Channels:   ch,
			Quality:    resampling.QualitySpec{Preset: resampling.QualityHigh},
		}

		resampler, err := resampling.New(&config)
		if err != nil {
			fmt.Printf("  %d channels: Error - %v\n", ch, err)
			continue
		}

		info := resampling.GetInfo(resampler)
		fmt.Printf("  %d channels: %.1f KB total memory\n",
			ch, float64(info.MemoryUsage)/bytesPerKilobyte)
	}

	fmt.Println("\n=== Demo Complete ===")
}
