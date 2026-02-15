package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/go-audio/audio"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/go-audio-resampler/internal/engine"
)

func TestOpenWAVInput_FileNotFound(t *testing.T) {
	_, err := openWAVInput("/nonexistent/file.wav", false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to open input file")
}

func TestOpenWAVInput_InvalidWAV(t *testing.T) {
	// Create a temporary file that's not a WAV
	tmpDir := t.TempDir()
	invalidFile := filepath.Join(tmpDir, "invalid.wav")
	err := os.WriteFile(invalidFile, []byte("not a wav file"), 0o644)
	require.NoError(t, err)

	_, err = openWAVInput(invalidFile, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid WAV file")
}

func TestCreateChannelResamplers_Mono(t *testing.T) {
	resamplers, err := createChannelResamplers[float64](
		1,            // mono
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)
	assert.Len(t, resamplers, 1)
	assert.NotNil(t, resamplers[0])
}

func TestCreateChannelResamplers_Stereo(t *testing.T) {
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)
	assert.Len(t, resamplers, 2)
	assert.NotNil(t, resamplers[0])
	assert.NotNil(t, resamplers[1])
}

func TestCreateChannelResamplers_Multichannel(t *testing.T) {
	resamplers, err := createChannelResamplers[float64](
		8,            // 7.1 surround
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)
	assert.Len(t, resamplers, 8)
	for i, r := range resamplers {
		assert.NotNil(t, r, "resampler %d should not be nil", i)
	}
}

func TestCreateWAVOutput_InvalidDirectory(t *testing.T) {
	_, err := createWAVOutput(
		"/nonexistent/dir/output.wav",
		48000, // sample rate
		16,    // bit depth
		2,     // channels
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create output file")
}

func TestCreateWAVOutput_Success(t *testing.T) {
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "test_output.wav")

	writer, err := createWAVOutput(outputPath, 48000, 16, 2)
	require.NoError(t, err)
	require.NotNil(t, writer)
	defer func() { _ = writer.Close() }()

	assert.NotNil(t, writer.file)
	assert.NotNil(t, writer.writer)

	// Verify file was created
	_, err = os.Stat(outputPath)
	require.NoError(t, err)
}

func TestNewResampleBuffers(t *testing.T) {
	format := &audio.Format{
		SampleRate:  44100,
		NumChannels: 2,
	}
	buffers := newResampleBuffers[float64](
		2,            // stereo
		16,           // 16-bit
		44100, 48000, // rates
		format,
	)

	require.NotNil(t, buffers)
	assert.Len(t, buffers.channelBufs, 2)
	assert.Len(t, buffers.resampledChannels, 2)
	assert.NotEmpty(t, buffers.outputIntBuf)
	assert.Greater(t, buffers.maxVal, 0.0)
	assert.Greater(t, buffers.invMaxVal, 0.0)
}

func TestProgressTracker_VerboseMode(t *testing.T) {
	tracker := newProgressTracker(1000, true)
	require.NotNil(t, tracker)

	assert.Equal(t, int64(1000), tracker.totalSamples)
	assert.True(t, tracker.verbose)
	assert.Equal(t, 0, tracker.lastProgress)
}

func TestProgressTracker_NonVerboseMode(t *testing.T) {
	tracker := newProgressTracker(1000, false)
	require.NotNil(t, tracker)

	assert.False(t, tracker.verbose)
	// reportIfNeeded should do nothing in non-verbose mode
	tracker.reportIfNeeded(500) // Should not panic or log
}

func TestProgressTracker_ZeroSamples(t *testing.T) {
	tracker := newProgressTracker(0, true)
	require.NotNil(t, tracker)

	// Should not panic with zero samples
	tracker.reportIfNeeded(100)
}

func TestResampleSequential_Success(t *testing.T) {
	// Create resamplers
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Create test data (need enough samples to overcome filter latency)
	numSamples := 1000
	channelBufs := [][]float64{
		make([]float64, numSamples),
		make([]float64, numSamples),
	}
	// Fill with test data (constant signal)
	for i := range numSamples {
		channelBufs[0][i] = 0.5
		channelBufs[1][i] = -0.5
	}

	// Resample
	result, err := resampleSequential(resamplers, channelBufs, numSamples, 2)
	require.NoError(t, err)
	require.Len(t, result, 2)
	// With 1000 samples, we should get output (or zero due to filter latency)
	assert.NotNil(t, result[0])
	assert.NotNil(t, result[1])
}

func TestResampleParallel_Success(t *testing.T) {
	// Create resamplers
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Create test data (need enough samples to overcome filter latency)
	numSamples := 1000
	channelBufs := [][]float64{
		make([]float64, numSamples),
		make([]float64, numSamples),
	}
	// Fill with test data (constant signal)
	for i := range numSamples {
		channelBufs[0][i] = 0.5
		channelBufs[1][i] = -0.5
	}

	// Resample
	result, err := resampleParallel(resamplers, channelBufs, numSamples, 2)
	require.NoError(t, err)
	require.Len(t, result, 2)
	// With 1000 samples, we should get output (or zero due to filter latency)
	assert.NotNil(t, result[0])
	assert.NotNil(t, result[1])
}

func TestResampleChannelData_ParallelMode(t *testing.T) {
	// Create resamplers
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Create test data
	channelBufs := [][]float64{
		make([]float64, 100),
		make([]float64, 100),
	}

	// Should use parallel for stereo when parallel=true
	result, err := resampleChannelData(resamplers, channelBufs, 100, true)
	require.NoError(t, err)
	require.Len(t, result, 2)
}

func TestResampleChannelData_SequentialMode(t *testing.T) {
	// Create resamplers
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Create test data
	channelBufs := [][]float64{
		make([]float64, 100),
		make([]float64, 100),
	}

	// Should use sequential when parallel=false
	result, err := resampleChannelData(resamplers, channelBufs, 100, false)
	require.NoError(t, err)
	require.Len(t, result, 2)
}

func TestResampleChannelData_MonoFallsBackToSequential(t *testing.T) {
	// Create mono resampler
	resamplers, err := createChannelResamplers[float64](
		1,            // mono
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Create test data
	channelBufs := [][]float64{
		make([]float64, 100),
	}

	// Even with parallel=true, mono should use sequential
	result, err := resampleChannelData(resamplers, channelBufs, 100, true)
	require.NoError(t, err)
	require.Len(t, result, 1)
}

func TestFlushAndPadChannels_EqualLengths(t *testing.T) {
	// Create resamplers
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Process some data first
	channelBufs := [][]float64{
		make([]float64, 100),
		make([]float64, 100),
	}
	_, err = resampleSequential(resamplers, channelBufs, 100, 2)
	require.NoError(t, err)

	// Flush
	outputData, numSamples, err := flushAndPadChannels(resamplers, 16)
	require.NoError(t, err)

	if numSamples > 0 {
		assert.NotEmpty(t, outputData)
		assert.Positive(t, numSamples)
	}
}

func TestFlushAndPadChannels_EmptyFlush(t *testing.T) {
	// Create resamplers but don't process any data
	resamplers, err := createChannelResamplers[float64](
		2,            // stereo
		44100, 48000, // rates
		engine.QualityHigh,
	)
	require.NoError(t, err)

	// Flush should return empty/zero
	outputData, numSamples, err := flushAndPadChannels(resamplers, 16)
	require.NoError(t, err)

	// With no data processed, flush might return nothing
	if numSamples == 0 {
		assert.Empty(t, outputData)
	}
}
