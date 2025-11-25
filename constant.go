package resampler

import (
	"fmt"
	"sync"
)

// constantRateResampler implements fixed-ratio resampling.
// It uses a multi-stage pipeline approach similar to libsoxr,
// combining different algorithms for optimal performance.
type constantRateResampler struct {
	config   Config
	ratio    float64
	pipeline *Pipeline

	// Per-channel state
	channels []*channelResampler

	// Shared resources
	mu sync.RWMutex
}

// channelResampler holds per-channel state.
type channelResampler struct {
	stages  []Stage
	buffers []*RingBuffer
}

// newConstantRateResampler creates a new constant-rate resampler.
func newConstantRateResampler(config *Config, ratio float64) (*constantRateResampler, error) {
	r := &constantRateResampler{
		config:   *config,
		ratio:    ratio,
		channels: make([]*channelResampler, config.Channels),
	}

	// Build the processing pipeline based on quality and ratio
	pipeline, err := buildPipeline(config, ratio)
	if err != nil {
		return nil, fmt.Errorf("failed to build pipeline: %w", err)
	}
	r.pipeline = pipeline

	// Initialize per-channel state
	for i := 0; i < config.Channels; i++ {
		ch := &channelResampler{
			stages:  make([]Stage, len(pipeline.stages)),
			buffers: make([]*RingBuffer, len(pipeline.stages)+1),
		}

		// Create stage instances for this channel
		for j, stageSpec := range pipeline.stages {
			stage, err := createStage(stageSpec, config)
			if err != nil {
				return nil, fmt.Errorf("failed to create stage %d: %w", j, err)
			}
			ch.stages[j] = stage
		}

		// Create buffers between stages
		for j := 0; j <= len(pipeline.stages); j++ {
			bufferSize := defaultBufferSize
			if config.MaxInputSize > 0 && j == 0 {
				bufferSize = config.MaxInputSize * bufferSizeMultiplier
			}
			ch.buffers[j] = NewRingBuffer(bufferSize)
		}

		r.channels[i] = ch
	}

	return r, nil
}

// Process resamples a mono audio channel.
func (r *constantRateResampler) Process(input []float64) ([]float64, error) {
	if len(r.channels) == 0 {
		return nil, fmt.Errorf("no channels initialized")
	}

	// Use first channel for mono processing
	return r.processChannel(0, input)
}

// ProcessFloat32 resamples float32 audio data.
// Internally converts to float64 for processing, then converts back.
// This approach maintains numerical precision during resampling while
// supporting float32 I/O. The conversion overhead is minimal compared
// to the filter computation (~5% of total time for typical buffers).
// Native float32 processing would require duplicating the engine with
// generic types, which is a trade-off between code complexity and performance.
func (r *constantRateResampler) ProcessFloat32(input []float32) ([]float32, error) {
	// Convert to float64 for high-precision internal processing
	input64 := make([]float64, len(input))
	for i, v := range input {
		input64[i] = float64(v)
	}

	output64, err := r.Process(input64)
	if err != nil {
		return nil, err
	}

	output32 := make([]float32, len(output64))
	for i, v := range output64 {
		output32[i] = float32(v)
	}

	return output32, nil
}

// ProcessMulti processes multiple audio channels.
// When EnableParallel is true in config, channels are processed concurrently.
// Otherwise, channels are processed sequentially.
func (r *constantRateResampler) ProcessMulti(input [][]float64) ([][]float64, error) {
	if len(input) != r.config.Channels {
		return nil, fmt.Errorf("expected %d channels, got %d", r.config.Channels, len(input))
	}

	output := make([][]float64, len(input))

	// Sequential processing (default or when parallel disabled)
	if !r.config.EnableParallel || len(input) <= 1 {
		for ch := range input {
			result, err := r.processChannel(ch, input[ch])
			if err != nil {
				return nil, fmt.Errorf("channel %d: %w", ch, err)
			}
			output[ch] = result
		}
		return output, nil
	}

	// Parallel processing: process channels concurrently
	var wg sync.WaitGroup
	errChan := make(chan error, len(input))

	for ch := range input {
		wg.Add(1)
		go func(channel int) {
			defer wg.Done()

			result, err := r.processChannel(channel, input[channel])
			if err != nil {
				errChan <- fmt.Errorf("channel %d: %w", channel, err)
				return
			}
			output[channel] = result
		}(ch)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return nil, err
		}
	}

	return output, nil
}

// processChannel processes a single channel through the pipeline.
func (r *constantRateResampler) processChannel(channel int, input []float64) ([]float64, error) {
	if channel >= len(r.channels) {
		return nil, fmt.Errorf("channel %d out of range", channel)
	}

	ch := r.channels[channel]

	// Add input to first buffer
	ch.buffers[0].Write(input)

	// Process through each stage
	for i, stage := range ch.stages {
		inputBuffer := ch.buffers[i]
		outputBuffer := ch.buffers[i+1]

		// Process available input
		for inputBuffer.Available() >= stage.GetMinInput() {
			// Get input chunk
			chunk := inputBuffer.Read(stage.GetMinInput())

			// Process through stage
			output, err := stage.Process(chunk)
			if err != nil {
				return nil, fmt.Errorf("stage %d processing error: %w", i, err)
			}

			// Write to next buffer
			outputBuffer.Write(output)
		}
	}

	// Read final output
	finalBuffer := ch.buffers[len(ch.buffers)-1]
	return finalBuffer.ReadAll(), nil
}

// Flush returns any remaining samples.
func (r *constantRateResampler) Flush() ([]float64, error) {
	if len(r.channels) == 0 {
		return []float64{}, nil
	}

	// Flush first channel for mono
	ch := r.channels[0]

	// Flush each stage
	for i, stage := range ch.stages {
		outputBuffer := ch.buffers[i+1]

		// Flush the stage
		output, err := stage.Flush()
		if err != nil {
			return nil, fmt.Errorf("stage %d flush error: %w", i, err)
		}

		if len(output) > 0 {
			outputBuffer.Write(output)
		}
	}

	// Return all remaining output
	finalBuffer := ch.buffers[len(ch.buffers)-1]
	return finalBuffer.ReadAll(), nil
}

// GetLatency returns the total pipeline latency in samples.
func (r *constantRateResampler) GetLatency() int {
	if r.pipeline == nil || len(r.channels) == 0 {
		return 0
	}

	// Use the first channel's stages to calculate latency
	ch := r.channels[0]
	if ch == nil || len(ch.stages) == 0 {
		return 0
	}

	totalLatency := 0
	for _, stage := range ch.stages {
		// Account for stage processing latency and ratio change
		stageLatency := int(float64(stage.GetLatency()) * stage.GetRatio())
		totalLatency += stageLatency
	}

	return totalLatency
}

// Reset clears all internal state.
func (r *constantRateResampler) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, ch := range r.channels {
		// Reset all stages
		for _, stage := range ch.stages {
			stage.Reset()
		}

		// Clear all buffers
		for _, buffer := range ch.buffers {
			buffer.Clear()
		}
	}
}

// GetRatio returns the resampling ratio.
func (r *constantRateResampler) GetRatio() float64 {
	return r.ratio
}

// GetInfo returns information about the resampler.
func (r *constantRateResampler) GetInfo() Info {
	info := Info{
		Algorithm: "multi-stage",
		Latency:   r.GetLatency(),
	}

	// Calculate total memory usage
	var memUsage int64
	for _, ch := range r.channels {
		for _, buffer := range ch.buffers {
			memUsage += int64(buffer.Capacity() * bytesPerFloat64)
		}
		for _, stage := range ch.stages {
			memUsage += stage.GetMemoryUsage()
		}
	}
	info.MemoryUsage = memUsage

	// Get stage information from first channel
	if len(r.channels) > 0 && len(r.channels[0].stages) > 0 {
		// Report info from primary stage
		primaryStage := r.channels[0].stages[0]
		info.FilterLength = primaryStage.GetFilterLength()
		info.Phases = primaryStage.GetPhases()

		// Check for SIMD
		if simd := primaryStage.GetSIMDInfo(); simd != "" {
			info.SIMDEnabled = true
			info.SIMDType = simd
		}
	}

	return info
}
