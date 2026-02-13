// Command resample-wav resamples WAV audio files to a target sample rate.
//
// Usage:
//
//	resample-wav -rate 48 input.wav output.wav
//	resample-wav -rate 16 -quality high input.wav output.wav
//	resample-wav -rate 48 -fast input.wav output.wav          # ~40% faster, float32 precision
//	resample-wav -rate 48 -parallel=false input.wav out.wav   # Disable parallel processing
//
// Parallel processing is enabled by default for stereo/multichannel files,
// providing ~1.7x speedup for stereo and up to 8x for 7.1 surround.
package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/tphakala/go-audio-resampler/internal/engine"
)

const (
	// Buffer size for processing (number of samples per chunk)
	// Larger buffers reduce I/O overhead and improve cache utilization
	bufferSize = 65536

	// Output buffer margin to handle ratio variations
	outputBufferMargin = 1024

	// Channel count constants for fast paths
	monoChannels   = 1
	stereoChannels = 2

	// Sample format constants
	bitsPerSample16 = 16
	bitsPerSample24 = 24
	bitsPerSample32 = 32

	// Conversion constants
	kHzToHz          = 1000
	maxInt16         = 32767.0
	maxInt24         = 8388607.0
	maxInt32         = 2147483647.0
	progressInterval = 10 // Print progress every N%

	// CLI defaults
	defaultRateKHz  = 48.0
	minRequiredArgs = 2
	percentScale    = 100

	// WAV format constants
	wavHeaderSize      = 44 // Total WAV header size in bytes
	wavRiffHeaderSize  = 36 // RIFF header size (file size - 8 = riffHeaderSize + dataSize)
	wavPCMSubchunkSize = 16 // fmt subchunk size for PCM format
	wavFileSizeOffset  = 4  // Byte offset for file size field in header
	wavDataSizeOffset  = 40 // Byte offset for data size field in header

	// Byte sizes for PCM sample formats
	bytesPerSample16 = 2 // 16-bit PCM
	bytesPerSample24 = 3 // 24-bit PCM
	bytesPerSample32 = 4 // 32-bit PCM
	bitsPerByte      = 8 // Bits in a byte

	// Bit shift amounts for 24-bit sample encoding
	bitShift8  = 8
	bitShift16 = 16

	// I/O buffer sizes
	wavWriterBufferSize = 256 * 1024 // 256KB write buffer
	uint32Size          = 4          // Size of uint32 in bytes
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// Parse command line flags
	rateKHz := flag.Float64("rate", defaultRateKHz, "Target sample rate in kHz (e.g., 16, 32, 44.1, 48, 96)")
	quality := flag.String("quality", "high", "Quality preset: low, medium, high")
	fast := flag.Bool("fast", false, "Use float32 precision (~40% faster, sufficient for 16-bit audio)")
	parallel := flag.Bool("parallel", true, "Enable parallel channel processing (faster for stereo/multichannel)")
	verbose := flag.Bool("v", false, "Verbose output")
	cpuprofile := flag.String("cpuprofile", "", "Write CPU profile to file (for PGO)")
	flag.Parse()

	// Validate arguments before setting up profiling
	args := flag.Args()
	if len(args) < minRequiredArgs {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] input.wav output.wav\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s -rate 48 input.wav output.wav      # Resample to 48kHz\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -rate 16 speech.wav speech_16k.wav # Downsample for speech\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -rate 96 music.wav music_hires.wav # Upsample to hi-res\n", os.Args[0])
		return fmt.Errorf("insufficient arguments")
	}

	// Start CPU profiling if requested (for PGO)
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			return fmt.Errorf("could not create CPU profile: %w", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			_ = f.Close()
			return fmt.Errorf("could not start CPU profile: %w", err)
		}
		defer func() {
			pprof.StopCPUProfile()
			_ = f.Close()
		}()
	}

	inputPath := args[0]
	outputPath := args[1]
	targetRate := int(*rateKHz * kHzToHz)

	// Map quality string to engine quality
	engineQuality := parseQuality(*quality)

	if *verbose {
		log.Printf("Input: %s", inputPath)
		log.Printf("Output: %s", outputPath)
		log.Printf("Target rate: %d Hz", targetRate)
		log.Printf("Quality: %s", *quality)
		if *fast {
			log.Printf("Precision: float32 (fast mode)")
		} else {
			log.Printf("Precision: float64 (high precision)")
		}
		if *parallel {
			log.Printf("Parallel: enabled (concurrent channel processing)")
		} else {
			log.Printf("Parallel: disabled (sequential processing)")
		}
	}

	// Process the file
	start := time.Now()
	var stats *resampleStats
	var err error
	if *fast {
		stats, err = resampleWAVFloat32(inputPath, outputPath, targetRate, engineQuality, *verbose, *parallel)
	} else {
		stats, err = resampleWAVFloat64(inputPath, outputPath, targetRate, engineQuality, *verbose, *parallel)
	}
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	// Print summary
	fmt.Printf("Resampled %s -> %s\n", filepath.Base(inputPath), filepath.Base(outputPath))
	fmt.Printf("  %d Hz -> %d Hz (%d channels, %d-bit)\n",
		stats.inputRate, stats.outputRate, stats.channels, stats.bitDepth)
	fmt.Printf("  %d samples -> %d samples\n", stats.inputSamples, stats.outputSamples)
	fmt.Printf("  Duration: %.2fs, Speed: %.1fx realtime\n",
		elapsed.Seconds(),
		float64(stats.inputSamples)/float64(stats.inputRate)/elapsed.Seconds())

	return nil
}

type resampleStats struct {
	inputRate     int
	outputRate    int
	channels      int
	bitDepth      int
	inputSamples  int64
	outputSamples int64
}

func parseQuality(q string) engine.Quality {
	switch strings.ToLower(q) {
	case "low":
		return engine.QualityLow
	case "medium":
		return engine.QualityMedium
	case "high":
		return engine.QualityHigh
	default:
		return engine.QualityHigh
	}
}

// resampleWAVFloat64 resamples using float64 precision (maximum quality).
func resampleWAVFloat64(inputPath, outputPath string, targetRate int, quality engine.Quality, verbose, parallel bool) (*resampleStats, error) {
	return resampleWAVGeneric[float64](inputPath, outputPath, targetRate, quality, verbose, parallel)
}

// resampleWAVFloat32 resamples using float32 precision (~40% faster).
func resampleWAVFloat32(inputPath, outputPath string, targetRate int, quality engine.Quality, verbose, parallel bool) (*resampleStats, error) {
	return resampleWAVGeneric[float32](inputPath, outputPath, targetRate, quality, verbose, parallel)
}

// Float constraint for generic resampling.
type Float interface {
	float32 | float64
}

func resampleWAVGeneric[F Float](inputPath, outputPath string, targetRate int, quality engine.Quality, verbose, parallel bool) (stats *resampleStats, err error) {
	// 1. Open and validate input
	input, err := openWAVInput(inputPath, verbose)
	if err != nil {
		return nil, err
	}
	defer func() { _ = input.Close() }()

	// Check if resampling is needed
	if input.rate == targetRate {
		return nil, fmt.Errorf("input already at target rate %d Hz", targetRate)
	}

	// 2. Create resamplers
	resamplers, err := createChannelResamplers[F](
		input.channels, input.rate, targetRate, quality,
	)
	if err != nil {
		return nil, err
	}

	// 3. Create output writer
	output, err := createWAVOutput(
		outputPath, targetRate, input.bitDepth, input.channels,
	)
	if err != nil {
		return nil, err
	}
	// Close output, capturing close errors on success path (important for WAV header updates)
	defer func() {
		if closeErr := output.Close(); err == nil {
			err = closeErr
		}
	}()

	// 4. Initialize processing buffers
	buffers := newResampleBuffers[F](
		input.channels, input.bitDepth,
		input.rate, targetRate,
		input.format,
	)

	// 5. Initialize tracking
	stats = &resampleStats{
		inputRate:  input.rate,
		outputRate: targetRate,
		channels:   input.channels,
		bitDepth:   input.bitDepth,
	}
	progress := newProgressTracker(input.totalSamples, verbose)

	// 6. Main processing loop
	for {
		// Read chunk
		n, err := input.decoder.PCMBuffer(buffers.intBuffer)
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("failed to read audio data: %w", err)
		}
		if n == 0 {
			break
		}

		// Update tracking
		buffers.intBuffer.Data = buffers.intBuffer.Data[:n*input.channels]
		stats.inputSamples += int64(n)

		// Deinterleave
		deinterleaveInto(
			buffers.intBuffer.Data,
			buffers.channelBufs,
			input.channels, n,
			buffers.invMaxVal,
		)

		// Resample channels (handles parallel/sequential)
		resampledChannels, err := resampleChannelData(
			resamplers,
			buffers.channelBufs,
			n,
			parallel,
		)
		if err != nil {
			return nil, err
		}

		// Interleave and write
		outputLen := interleaveInto(
			resampledChannels,
			buffers.outputIntBuf,
			buffers.maxVal,
		)
		stats.outputSamples += int64(outputLen / input.channels)

		if err := output.WriteSamples(buffers.outputIntBuf[:outputLen]); err != nil {
			return nil, fmt.Errorf("failed to write audio data: %w", err)
		}

		// Progress reporting
		progress.reportIfNeeded(stats.inputSamples)

		// Reset buffer
		buffers.intBuffer.Data = buffers.intBuffer.Data[:cap(buffers.intBuffer.Data)]
	}

	// 7. Flush remaining samples
	flushedData, flushedSamples, err := flushAndPadChannels(resamplers, input.bitDepth)
	if err != nil {
		return nil, err
	}

	if flushedSamples > 0 {
		stats.outputSamples += int64(flushedSamples)
		if err := output.WriteSamples(flushedData); err != nil {
			return nil, fmt.Errorf("failed to write flushed data: %w", err)
		}
	}

	return stats, nil
}

// deinterleaveGeneric converts interleaved int samples to per-channel float slices.
func deinterleaveGeneric[F Float](data []int, channels, bitDepth int) [][]F {
	samplesPerChannel := len(data) / channels
	result := make([][]F, channels)
	for ch := range channels {
		result[ch] = make([]F, samplesPerChannel)
	}

	// Determine max value for normalization
	var maxVal float64
	switch bitDepth {
	case bitsPerSample16:
		maxVal = maxInt16
	case bitsPerSample24:
		maxVal = maxInt24
	case bitsPerSample32:
		maxVal = maxInt32
	default:
		maxVal = maxInt16
	}

	// Deinterleave and normalize to [-1.0, 1.0]
	for i := range samplesPerChannel {
		for ch := range channels {
			result[ch][i] = F(float64(data[i*channels+ch]) / maxVal)
		}
	}

	return result
}

// interleaveGeneric converts per-channel float slices to interleaved int samples.
func interleaveGeneric[F Float](channels [][]F, bitDepth int) []int {
	if len(channels) == 0 || len(channels[0]) == 0 {
		return nil
	}

	numChannels := len(channels)
	samplesPerChannel := len(channels[0])
	result := make([]int, samplesPerChannel*numChannels)

	// Determine max value for denormalization
	var maxVal float64
	switch bitDepth {
	case bitsPerSample16:
		maxVal = maxInt16
	case bitsPerSample24:
		maxVal = maxInt24
	case bitsPerSample32:
		maxVal = maxInt32
	default:
		maxVal = maxInt16
	}

	// Interleave and denormalize from [-1.0, 1.0]
	for i := range samplesPerChannel {
		for ch := range numChannels {
			// Clamp to [-1.0, 1.0] and convert
			sample := float64(channels[ch][i])
			if sample > 1.0 {
				sample = 1.0
			} else if sample < -1.0 {
				sample = -1.0
			}
			result[i*numChannels+ch] = int(sample * maxVal)
		}
	}

	return result
}

// getMaxValue returns the maximum sample value for the given bit depth.
func getMaxValue(bitDepth int) float64 {
	switch bitDepth {
	case bitsPerSample16:
		return maxInt16
	case bitsPerSample24:
		return maxInt24
	case bitsPerSample32:
		return maxInt32
	default:
		return maxInt16
	}
}

// deinterleaveInto converts interleaved int samples into preallocated per-channel buffers.
// This avoids allocations in the hot loop.
func deinterleaveInto[F Float](data []int, channelBufs [][]F, numChannels, samplesPerChannel int, invMaxVal float64) {
	// Fast path for mono
	if numChannels == monoChannels {
		buf := channelBufs[0]
		for i := range samplesPerChannel {
			buf[i] = F(float64(data[i]) * invMaxVal)
		}
		return
	}

	// Fast path for stereo
	if numChannels == stereoChannels {
		buf0, buf1 := channelBufs[0], channelBufs[1]
		for i := range samplesPerChannel {
			idx := i * stereoChannels
			buf0[i] = F(float64(data[idx]) * invMaxVal)
			buf1[i] = F(float64(data[idx+1]) * invMaxVal)
		}
		return
	}

	// General case
	for i := range samplesPerChannel {
		base := i * numChannels
		for ch := range numChannels {
			channelBufs[ch][i] = F(float64(data[base+ch]) * invMaxVal)
		}
	}
}

// interleaveInto converts per-channel float slices into a preallocated int buffer.
// Returns the number of elements written.
func interleaveInto[F Float](channels [][]F, dst []int, maxVal float64) int {
	if len(channels) == 0 || len(channels[0]) == 0 {
		return 0
	}

	numChannels := len(channels)
	samplesPerChannel := len(channels[0])
	totalLen := samplesPerChannel * numChannels

	// Grow destination if needed
	if len(dst) < totalLen {
		return 0 // Caller should handle this
	}

	// Fast path for mono
	if numChannels == monoChannels {
		ch := channels[0]
		for i := range samplesPerChannel {
			sample := float64(ch[i])
			if sample > 1.0 {
				sample = 1.0
			} else if sample < -1.0 {
				sample = -1.0
			}
			dst[i] = int(sample * maxVal)
		}
		return samplesPerChannel
	}

	// Fast path for stereo
	if numChannels == stereoChannels {
		ch0, ch1 := channels[0], channels[1]
		for i := range samplesPerChannel {
			s0, s1 := float64(ch0[i]), float64(ch1[i])
			if s0 > 1.0 {
				s0 = 1.0
			} else if s0 < -1.0 {
				s0 = -1.0
			}
			if s1 > 1.0 {
				s1 = 1.0
			} else if s1 < -1.0 {
				s1 = -1.0
			}
			idx := i * stereoChannels
			dst[idx] = int(s0 * maxVal)
			dst[idx+1] = int(s1 * maxVal)
		}
		return totalLen
	}

	// General case
	for i := range samplesPerChannel {
		base := i * numChannels
		for ch := range numChannels {
			sample := float64(channels[ch][i])
			if sample > 1.0 {
				sample = 1.0
			} else if sample < -1.0 {
				sample = -1.0
			}
			dst[base+ch] = int(sample * maxVal)
		}
	}

	return totalLen
}

// fastWAVWriter writes PCM data directly without per-sample allocations.
// This is much faster than go-audio/wav for large files.
type fastWAVWriter struct {
	w          *bufio.Writer
	f          *os.File
	sampleRate int
	bitDepth   int
	channels   int
	dataSize   uint32
	byteBuf    []byte // Preallocated buffer for encoding
}

// newFastWAVWriter creates a new fast WAV writer.
func newFastWAVWriter(f *os.File, sampleRate, bitDepth, channels int) (*fastWAVWriter, error) {
	w := &fastWAVWriter{
		w:          bufio.NewWriterSize(f, wavWriterBufferSize),
		f:          f,
		sampleRate: sampleRate,
		bitDepth:   bitDepth,
		channels:   channels,
		byteBuf:    make([]byte, bufferSize*channels*(bitDepth/bitsPerByte)),
	}

	// Write WAV header (44 bytes) with placeholder sizes
	if err := w.writeHeader(); err != nil {
		return nil, err
	}

	return w, nil
}

func (w *fastWAVWriter) writeHeader() error {
	byteRate := w.sampleRate * w.channels * (w.bitDepth / bitsPerByte)
	blockAlign := w.channels * (w.bitDepth / bitsPerByte)

	header := make([]byte, wavHeaderSize)

	// RIFF header
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], 0) // Placeholder for file size - 8
	copy(header[8:12], "WAVE")

	// fmt subchunk
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], wavPCMSubchunkSize)   // Subchunk1Size (16 for PCM)
	binary.LittleEndian.PutUint16(header[20:22], 1)                    // AudioFormat (1 = PCM)
	binary.LittleEndian.PutUint16(header[22:24], uint16(w.channels))   // NumChannels
	binary.LittleEndian.PutUint32(header[24:28], uint32(w.sampleRate)) // SampleRate
	binary.LittleEndian.PutUint32(header[28:32], uint32(byteRate))     // ByteRate
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))   // BlockAlign
	binary.LittleEndian.PutUint16(header[34:36], uint16(w.bitDepth))   // BitsPerSample

	// data subchunk
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], 0) // Placeholder for data size

	_, err := w.w.Write(header)
	return err
}

// WriteSamples16 writes 16-bit PCM samples directly from int slice.
func (w *fastWAVWriter) WriteSamples16(samples []int) error {
	n := len(samples)
	needed := n * bytesPerSample16
	if len(w.byteBuf) < needed {
		w.byteBuf = make([]byte, needed)
	}

	buf := w.byteBuf[:needed]
	for i, s := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(int16(s)))
	}

	written, err := w.w.Write(buf)
	w.dataSize += uint32(written)
	return err
}

// WriteSamples24 writes 24-bit PCM samples directly from int slice.
func (w *fastWAVWriter) WriteSamples24(samples []int) error {
	n := len(samples)
	needed := n * bytesPerSample24
	if len(w.byteBuf) < needed {
		w.byteBuf = make([]byte, needed)
	}

	buf := w.byteBuf[:needed]
	for i, s := range samples {
		buf[i*bytesPerSample24] = byte(s)
		buf[i*bytesPerSample24+1] = byte(s >> bitShift8)
		buf[i*bytesPerSample24+2] = byte(s >> bitShift16)
	}

	written, err := w.w.Write(buf)
	w.dataSize += uint32(written)
	return err
}

// WriteSamples32 writes 32-bit PCM samples directly from int slice.
func (w *fastWAVWriter) WriteSamples32(samples []int) error {
	n := len(samples)
	needed := n * bytesPerSample32
	if len(w.byteBuf) < needed {
		w.byteBuf = make([]byte, needed)
	}

	buf := w.byteBuf[:needed]
	for i, s := range samples {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(int32(s)))
	}

	written, err := w.w.Write(buf)
	w.dataSize += uint32(written)
	return err
}

// WriteSamples writes samples using the appropriate bit depth.
func (w *fastWAVWriter) WriteSamples(samples []int) error {
	switch w.bitDepth {
	case bitsPerSample16:
		return w.WriteSamples16(samples)
	case bitsPerSample24:
		return w.WriteSamples24(samples)
	case bitsPerSample32:
		return w.WriteSamples32(samples)
	default:
		return w.WriteSamples16(samples)
	}
}

// Close flushes the buffer and updates the WAV header with final sizes.
func (w *fastWAVWriter) Close() error {
	// Flush buffered data
	if err := w.w.Flush(); err != nil {
		return err
	}

	// Update header with actual sizes
	// File size at offset 4: total file size - 8
	// Data size at offset 40: actual data size
	fileSize := wavRiffHeaderSize + w.dataSize

	// Seek to file size field and update
	if _, err := w.f.Seek(wavFileSizeOffset, io.SeekStart); err != nil {
		return err
	}
	sizeBytes := make([]byte, uint32Size)
	binary.LittleEndian.PutUint32(sizeBytes, fileSize)
	if _, err := w.f.Write(sizeBytes); err != nil {
		return err
	}

	// Seek to data size field and update
	if _, err := w.f.Seek(wavDataSizeOffset, io.SeekStart); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(sizeBytes, w.dataSize)
	if _, err := w.f.Write(sizeBytes); err != nil {
		return err
	}

	return nil
}
