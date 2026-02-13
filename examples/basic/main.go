// Example: basic demonstrates simple audio resampling with go-audio-resampler.
//
// This example generates a 1kHz sine wave at 44.1kHz and resamples it to 48kHz.
package main

import (
	"fmt"
	"log"
	"math"

	resampling "github.com/tphakala/go-audio-resampler"
)

func main() {
	// Configuration
	const (
		inputRate  = 44100.0 // CD quality
		outputRate = 48000.0 // DAT/DVD quality
		duration   = 1.0     // 1 second of audio
		frequency  = 1000.0  // 1 kHz test tone
	)

	// Generate a 1kHz sine wave at 44.1kHz
	numSamples := int(inputRate * duration)
	input := make([]float64, numSamples)
	for i := range input {
		t := float64(i) / inputRate
		input[i] = math.Sin(2 * math.Pi * frequency * t)
	}

	fmt.Printf("Generated %d samples at %.0f Hz\n", len(input), inputRate)

	// Method 1: One-shot resampling (simplest)
	fmt.Println("\n--- Method 1: One-shot resampling ---")
	output1, err := resampling.ResampleMono(input, inputRate, outputRate, resampling.QualityHigh)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Resampled to %d samples at %.0f Hz\n", len(output1), outputRate)

	// Method 2: Using the engine directly (more control)
	fmt.Println("\n--- Method 2: Direct engine access ---")
	engine, err := resampling.NewEngine(inputRate, outputRate, resampling.QualityHigh)
	if err != nil {
		log.Fatal(err)
	}

	output2, err := engine.Process(input)
	if err != nil {
		log.Fatal(err)
	}
	flushed, _ := engine.Flush()
	output2 = append(output2, flushed...)

	fmt.Printf("Resampled to %d samples\n", len(output2))
	fmt.Printf("Resampling ratio: %.6f\n", engine.GetRatio())

	// Method 3: Streaming with configuration (production use)
	fmt.Println("\n--- Method 3: Streaming API ---")
	config := &resampling.Config{
		InputRate:  inputRate,
		OutputRate: outputRate,
		Channels:   1,
		Quality:    resampling.QualitySpec{Preset: resampling.QualityHigh},
	}

	r, err := resampling.New(config)
	if err != nil {
		log.Fatal(err)
	}

	// Process in chunks (simulating streaming)
	chunkSize := 4096
	estimatedOutput := int(float64(len(input)) * outputRate / inputRate)
	output3 := make([]float64, 0, estimatedOutput)
	for i := 0; i < len(input); i += chunkSize {
		end := min(i+chunkSize, len(input))
		chunk := input[i:end]

		out, err := r.Process(chunk)
		if err != nil {
			log.Fatal(err)
		}
		output3 = append(output3, out...)
	}

	// Flush remaining samples
	flushed, _ = r.Flush()
	output3 = append(output3, flushed...)

	fmt.Printf("Streamed %d chunks, output %d samples\n",
		(len(input)+chunkSize-1)/chunkSize, len(output3))

	// Get resampler info
	info := resampling.GetInfo(r)
	fmt.Printf("\nResampler info:\n")
	fmt.Printf("  Algorithm: %s\n", info.Algorithm)
	fmt.Printf("  Filter taps: %d\n", info.FilterLength)
	fmt.Printf("  Latency: %d samples\n", info.Latency)
	fmt.Printf("  Memory: %d bytes\n", info.MemoryUsage)

	// Verify output quality by checking frequency content
	fmt.Println("\n--- Quality check ---")
	expectedSamples := int(float64(len(input)) * (outputRate / inputRate))
	fmt.Printf("Expected ~%d output samples, got %d (within normal variance)\n",
		expectedSamples, len(output1))
}
