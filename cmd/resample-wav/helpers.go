package main

import (
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/tphakala/go-audio-resampler/internal/engine"
)

// wavInputInfo holds validated input file information.
type wavInputInfo struct {
	file         *os.File
	decoder      *wav.Decoder
	rate         int
	channels     int
	bitDepth     int
	totalSamples int64
	format       *audio.Format
}

// openWAVInput opens and validates a WAV file, returning format information.
func openWAVInput(path string, verbose bool) (*wavInputInfo, error) {
	// Open input file
	inputFile, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file: %w", err)
	}

	// Create WAV decoder
	decoder := wav.NewDecoder(inputFile)
	if !decoder.IsValidFile() {
		_ = inputFile.Close()
		return nil, fmt.Errorf("invalid WAV file: %s", path)
	}

	// Read format info
	format := decoder.Format()
	inputRate := format.SampleRate
	channels := format.NumChannels
	bitDepth := int(decoder.BitDepth)

	if verbose {
		log.Printf("Input format: %d Hz, %d channels, %d-bit", inputRate, channels, bitDepth)
	}

	// Get total duration for progress reporting
	duration, err := decoder.Duration()
	if err != nil {
		duration = 0
	}
	totalSamples := int64(duration.Seconds() * float64(inputRate))

	return &wavInputInfo{
		file:         inputFile,
		decoder:      decoder,
		rate:         inputRate,
		channels:     channels,
		bitDepth:     bitDepth,
		totalSamples: totalSamples,
		format:       format,
	}, nil
}

// Close closes the input file.
func (w *wavInputInfo) Close() error {
	return w.file.Close()
}

// createChannelResamplers creates one resampler per channel.
func createChannelResamplers[F Float](
	numChannels int,
	inputRate, targetRate int,
	quality engine.Quality,
) ([]*engine.Resampler[F], error) {
	resamplers := make([]*engine.Resampler[F], numChannels)
	for ch := range numChannels {
		r, err := engine.NewResampler[F](float64(inputRate), float64(targetRate), quality)
		if err != nil {
			return nil, fmt.Errorf("failed to create resampler for channel %d: %w", ch, err)
		}
		resamplers[ch] = r
	}
	return resamplers, nil
}

// wavOutputWriter wraps output file and fast writer.
type wavOutputWriter struct {
	file   *os.File
	writer *fastWAVWriter
}

// createWAVOutput creates output file and writer.
func createWAVOutput(
	path string,
	sampleRate, bitDepth, channels int,
) (*wavOutputWriter, error) {
	// Create output file
	outputFile, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("failed to create output file: %w", err)
	}

	// Create fast WAV writer
	fastWriter, err := newFastWAVWriter(outputFile, sampleRate, bitDepth, channels)
	if err != nil {
		_ = outputFile.Close()
		return nil, fmt.Errorf("failed to create WAV writer: %w", err)
	}

	return &wavOutputWriter{
		file:   outputFile,
		writer: fastWriter,
	}, nil
}

// WriteSamples writes samples to the output file.
func (w *wavOutputWriter) WriteSamples(samples []int) error {
	return w.writer.WriteSamples(samples)
}

// Close closes the output writer and file.
func (w *wavOutputWriter) Close() error {
	if err := w.writer.Close(); err != nil {
		return err
	}
	return w.file.Close()
}

// resampleBuffers holds all preallocated buffers for resampling.
type resampleBuffers[F Float] struct {
	intBuffer         *audio.IntBuffer
	channelBufs       [][]F
	resampledChannels [][]F
	outputIntBuf      []int
	invMaxVal         float64
	maxVal            float64
}

// newResampleBuffers creates and preallocates all processing buffers.
func newResampleBuffers[F Float](
	channels, bitDepth int,
	inputRate, targetRate int,
	format *audio.Format,
) *resampleBuffers[F] {
	// Preallocate buffers for reuse (reduces GC pressure)
	intBuffer := &audio.IntBuffer{
		Data:   make([]int, bufferSize*channels),
		Format: format,
	}

	// Preallocate per-channel input buffers
	channelBufs := make([][]F, channels)
	for ch := range channels {
		channelBufs[ch] = make([]F, bufferSize)
	}

	// Preallocate resampled channel slice (reused each iteration)
	resampledChannels := make([][]F, channels)

	// Preallocate output buffer with estimated size (ratio * input + margin)
	estimatedOutputSize := int(float64(bufferSize)*float64(targetRate)/float64(inputRate)) + outputBufferMargin
	outputIntBuf := make([]int, estimatedOutputSize*channels)

	// Precompute max value for bit depth
	maxVal := getMaxValue(bitDepth)
	invMaxVal := 1.0 / maxVal

	return &resampleBuffers[F]{
		intBuffer:         intBuffer,
		channelBufs:       channelBufs,
		resampledChannels: resampledChannels,
		outputIntBuf:      outputIntBuf,
		invMaxVal:         invMaxVal,
		maxVal:            maxVal,
	}
}

// progressTracker handles progress reporting.
type progressTracker struct {
	totalSamples int64
	lastProgress int
	verbose      bool
}

// newProgressTracker creates a new progress tracker.
func newProgressTracker(totalSamples int64, verbose bool) *progressTracker {
	return &progressTracker{
		totalSamples: totalSamples,
		verbose:      verbose,
	}
}

// reportIfNeeded reports progress if threshold crossed.
func (p *progressTracker) reportIfNeeded(currentSamples int64) {
	if !p.verbose || p.totalSamples == 0 {
		return
	}

	progress := int(float64(currentSamples) / float64(p.totalSamples) * percentScale)
	if progress >= p.lastProgress+progressInterval {
		log.Printf("Progress: %d%%", progress)
		p.lastProgress = progress
	}
}

// resampleChannelData resamples channel buffers using provided resamplers.
// Handles both parallel and sequential modes based on config.
func resampleChannelData[F Float](
	resamplers []*engine.Resampler[F],
	channelBufs [][]F,
	numSamples int,
	parallel bool,
) ([][]F, error) {
	channels := len(resamplers)

	// Parallel processing for multichannel
	if parallel && channels > 1 {
		return resampleParallel(resamplers, channelBufs, numSamples, channels)
	}

	// Sequential processing
	return resampleSequential(resamplers, channelBufs, numSamples, channels)
}

// resampleParallel processes channels concurrently.
func resampleParallel[F Float](
	resamplers []*engine.Resampler[F],
	channelBufs [][]F,
	numSamples, channels int,
) ([][]F, error) {
	resampledChannels := make([][]F, channels)
	var wg sync.WaitGroup
	var processErr error
	var errMu sync.Mutex

	for ch := range channels {
		wg.Add(1)
		go func(channel int) {
			defer wg.Done()
			resampled, err := resamplers[channel].Process(channelBufs[channel][:numSamples])
			if err != nil {
				errMu.Lock()
				if processErr == nil {
					processErr = fmt.Errorf("resampling failed on channel %d: %w", channel, err)
				}
				errMu.Unlock()
				return
			}
			resampledChannels[channel] = resampled
		}(ch)
	}
	wg.Wait()

	if processErr != nil {
		return nil, processErr
	}

	return resampledChannels, nil
}

// resampleSequential processes channels one by one.
func resampleSequential[F Float](
	resamplers []*engine.Resampler[F],
	channelBufs [][]F,
	numSamples, channels int,
) ([][]F, error) {
	resampledChannels := make([][]F, channels)
	for ch := range channels {
		resampled, err := resamplers[ch].Process(channelBufs[ch][:numSamples])
		if err != nil {
			return nil, fmt.Errorf("resampling failed on channel %d: %w", ch, err)
		}
		resampledChannels[ch] = resampled
	}
	return resampledChannels, nil
}

// flushAndPadChannels flushes all resamplers and pads channels to equal length.
func flushAndPadChannels[F Float](
	resamplers []*engine.Resampler[F],
	bitDepth int,
) (outputData []int, flushedSamples int, err error) {
	channels := len(resamplers)
	flushedData := make([][]F, channels)
	maxFlushLen := 0

	// Flush all channels
	for ch := range channels {
		flushed, err := resamplers[ch].Flush()
		if err != nil {
			return nil, 0, fmt.Errorf("failed to flush resampler channel %d: %w", ch, err)
		}
		flushedData[ch] = flushed
		if len(flushed) > maxFlushLen {
			maxFlushLen = len(flushed)
		}
	}

	if maxFlushLen == 0 {
		return nil, 0, nil
	}

	// Pad shorter channels
	for ch := range channels {
		if len(flushedData[ch]) < maxFlushLen {
			padded := make([]F, maxFlushLen)
			copy(padded, flushedData[ch])
			flushedData[ch] = padded
		}
	}

	outputData = interleaveGeneric(flushedData, bitDepth)
	return outputData, maxFlushLen, nil
}
