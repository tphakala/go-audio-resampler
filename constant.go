package resampler

import (
	"fmt"
	"sync"

	pipelinepkg "github.com/tphakala/go-audio-resampler/internal/pipeline"
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

	// Grow-only float64 scratch buffers reused by ProcessFloat32Into to convert
	// float32 input/output without allocating on every call. The engine's
	// growStableLen/appendStable helpers live in internal/engine and are not
	// importable here, so the grow-only sizing is done inline.
	f32in  []float64
	f32out []float64
}

// channelResampler holds per-channel state.
type channelResampler struct {
	stages  []Stage
	buffers []*RingBuffer

	// Pre-allocated scratch buffer reused by ReadInto to avoid allocations.
	readScratch []float64
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
	for i := range config.Channels {
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

// ProcessInto resamples input into the caller-provided output buffer.
// It writes up to len(output) samples and returns the number of valid samples;
// callers should consume output[:n]. The buffer tail beyond n is undefined.
//
// If output is too small, ProcessInto returns ErrBufferTooSmall before any
// processing state is advanced, so callers can retry with a larger buffer.
func (r *constantRateResampler) ProcessInto(input, output []float64) (int, error) {
	if len(r.channels) == 0 {
		return 0, fmt.Errorf("no channels initialized")
	}
	if len(output) < r.EstimateOutput(len(input)) {
		return 0, ErrBufferTooSmall
	}

	return r.processChannelInto(0, input, output)
}

// EstimateOutput returns the maximum number of output samples that
// processing inputLen input samples may produce. Callers should allocate
// output buffers of at least this size for ProcessInto.
func (r *constantRateResampler) EstimateOutput(inputLen int) int {
	return int(float64(inputLen)*r.ratio) + estimateOutputMargin
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

// ProcessFloat32Into resamples float32 input into the caller-provided float32
// output buffer. It is the caller-owned-output, float32 counterpart of
// ProcessInto: it writes up to len(output) samples, returns the count, and
// returns ErrBufferTooSmall before advancing state if output cannot hold
// EstimateOutput(len(input)) samples.
//
// Unlike ProcessFloat32, which allocates an input, intermediate, and output
// slice on every call, this method reuses grow-only internal scratch buffers
// for the float32<->float64 conversion, so it performs zero allocations once
// warm. Processing still runs through the float64 pipeline for precision.
//
// Single-channel only, matching ProcessFloat32 and the float64 ProcessInto.
// It is not safe for concurrent use with itself or the other Process methods.
func (r *constantRateResampler) ProcessFloat32Into(input, output []float32) (int, error) {
	if len(r.channels) == 0 {
		return 0, fmt.Errorf("no channels initialized")
	}
	required := r.EstimateOutput(len(input))
	if len(output) < required {
		return 0, ErrBufferTooSmall // checked before any state is advanced
	}

	// Grow-only float32 -> float64 input scratch.
	if cap(r.f32in) < len(input) {
		r.f32in = make([]float64, len(input))
	} else {
		r.f32in = r.f32in[:len(input)]
	}
	for i, v := range input {
		r.f32in[i] = float64(v)
	}

	// Grow-only float64 output scratch sized to the estimated output bound, not
	// the caller's buffer. processChannelInto never produces more than
	// EstimateOutput samples, so sizing to required keeps the scratch bounded by
	// input length instead of letting an oversized caller buffer grow it without
	// limit for the lifetime of the resampler.
	if cap(r.f32out) < required {
		r.f32out = make([]float64, required)
	} else {
		r.f32out = r.f32out[:required]
	}

	n, err := r.processChannelInto(0, r.f32in, r.f32out)
	if err != nil {
		return 0, err
	}
	for i := range n {
		output[i] = float32(r.f32out[i])
	}
	return n, nil
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

		// Process available input. Read everything at once (matching
		// processChannelInto) rather than GetMinInput()-sized chunks: the stages
		// are streaming-stateful, so chunk granularity does not affect the output
		// (verified bit-identical by TestNewPath_ProcessInto_MatchesProcess), and
		// the larger read trims per-iteration overhead on this allocating path.
		for inputBuffer.Available() >= stage.GetMinInput() {
			chunk := inputBuffer.Read(inputBuffer.Available())

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

// processChannelInto processes a single channel through the pipeline using the
// zero-copy path when available, and writes the output into dst. Returns the
// number of samples written.
func (r *constantRateResampler) processChannelInto(channel int, input, dst []float64) (int, error) {
	if channel >= len(r.channels) {
		return 0, fmt.Errorf("channel %d out of range", channel)
	}

	ch := r.channels[channel]

	ch.buffers[0].Write(input)

	for i, stage := range ch.stages {
		inputBuffer := ch.buffers[i]
		outputBuffer := ch.buffers[i+1]

		zcStage, hasZC := stage.(pipelinepkg.ZeroCopyProcessor)

		for inputBuffer.Available() >= stage.GetMinInput() {
			avail := inputBuffer.Available()
			if cap(ch.readScratch) < avail {
				ch.readScratch = make([]float64, avail)
			} else {
				ch.readScratch = ch.readScratch[:avail]
			}
			n := inputBuffer.ReadInto(ch.readScratch)
			chunk := ch.readScratch[:n]

			var output []float64
			var err error
			if hasZC {
				output, err = zcStage.ProcessZeroCopy(chunk)
			} else {
				output, err = stage.Process(chunk)
			}
			if err != nil {
				return 0, fmt.Errorf("stage %d processing error: %w", i, err)
			}

			outputBuffer.Write(output)
		}
	}

	finalBuffer := ch.buffers[len(ch.buffers)-1]
	if finalBuffer.Available() > len(dst) {
		panic("go-audio-resampler: EstimateOutput underestimated actual output length")
	}
	n := finalBuffer.ReadInto(dst)
	return n, nil
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
	// No locking: doc.go documents that calls on a single instance must be
	// serialized by the caller (standard for stateful streaming DSP), so a mutex
	// here would protect nothing while falsely implying cross-method safety.
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
