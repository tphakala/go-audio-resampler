// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/go-audio-resampler/internal/engine"
)

func createTempWAV(t *testing.T, filename string, sampleRate, numChannels, bitDepth, numSamples int) {
	t.Helper()
	f, err := os.Create(filename)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	enc := wav.NewEncoder(f, sampleRate, bitDepth, numChannels, 1)

	buf := &audio.IntBuffer{
		Format: &audio.Format{
			SampleRate:  sampleRate,
			NumChannels: numChannels,
		},
		SourceBitDepth: bitDepth,
		Data:           make([]int, numSamples*numChannels),
	}

	// Create simple pattern
	maxVal := getMaxValue(bitDepth)
	for i := range numSamples {
		for ch := range numChannels {
			// Alternating signal
			val := float64(i%2) * 0.5 * maxVal
			buf.Data[i*numChannels+ch] = int(val)
		}
	}

	err = enc.Write(buf)
	require.NoError(t, err)
	err = enc.Close()
	require.NoError(t, err)
}

func TestDeinterleaveInterleaveGeneric(t *testing.T) {
	bitDepths := []int{16, 24, 32}
	channelsList := []int{1, 2, 3}

	for _, bd := range bitDepths {
		for _, chs := range channelsList {
			maxVal := getMaxValue(bd)
			inputData := []int{
				int(0.1 * maxVal), int(-0.2 * maxVal), int(0.3 * maxVal),
				int(-0.4 * maxVal), int(0.5 * maxVal), int(-0.6 * maxVal),
			}
			// adjust length to match channels
			inputData = inputData[:(len(inputData)/chs)*chs]

			// Float64
			floatData64 := deinterleaveGeneric[float64](inputData, chs, bd)
			require.Len(t, floatData64, chs)
			outputData64 := interleaveGeneric(floatData64, bd)
			assert.Equal(t, inputData, outputData64)

			// Float32
			floatData32 := deinterleaveGeneric[float32](inputData, chs, bd)
			require.Len(t, floatData32, chs)
			outputData32 := interleaveGeneric(floatData32, bd)

			for idx := range inputData {
				diff := inputData[idx] - outputData32[idx]
				if diff < 0 {
					diff = -diff
				}
				maxDiff := 1
				if bd == 32 {
					maxDiff = 256
				}
				assert.LessOrEqual(t, diff, maxDiff, "value mismatch for bd=%d at index %d", bd, idx)
			}
		}
	}

	// Empty channels slice
	assert.Nil(t, interleaveGeneric[float64](nil, 16))
	assert.Nil(t, interleaveGeneric([][]float64{}, 16))
}

func TestGetMaxValue(t *testing.T) {
	assert.InDelta(t, maxInt16, getMaxValue(16), 1e-9)
	assert.InDelta(t, maxInt24, getMaxValue(24), 1e-9)
	assert.InDelta(t, maxInt32, getMaxValue(32), 1e-9)
	assert.InDelta(t, maxInt16, getMaxValue(99), 1e-9) // default
}

func TestParseQuality(t *testing.T) {
	assert.Equal(t, engine.QualityLow, parseQuality("low"))
	assert.Equal(t, engine.QualityMedium, parseQuality("medium"))
	assert.Equal(t, engine.QualityHigh, parseQuality("high"))
	assert.Equal(t, engine.QualityHigh, parseQuality("invalid"))
	assert.Equal(t, engine.QualityHigh, parseQuality("HIGH"))
}

func TestDeinterleaveInterleaveInto(t *testing.T) {
	// Mono
	{
		data := []int{100, -200, 300}
		invMax := 1.0 / 1000.0
		chBufs := [][]float64{make([]float64, 3)}
		deinterleaveInto(data, chBufs, 1, 3, invMax)
		assert.InDeltaSlice(t, []float64{0.1, -0.2, 0.3}, chBufs[0], 1e-9)

		dst := make([]int, 3)
		n := interleaveInto(chBufs, dst, 1000.0)
		assert.Equal(t, 3, n)
		assert.Equal(t, data, dst)
	}

	// Stereo
	{
		data := []int{100, 500, -200, -600, 300, 700}
		invMax := 1.0 / 1000.0
		chBufs := [][]float64{make([]float64, 3), make([]float64, 3)}
		deinterleaveInto(data, chBufs, 2, 3, invMax)
		assert.InDeltaSlice(t, []float64{0.1, -0.2, 0.3}, chBufs[0], 1e-9)
		assert.InDeltaSlice(t, []float64{0.5, -0.6, 0.7}, chBufs[1], 1e-9)

		dst := make([]int, 6)
		n := interleaveInto(chBufs, dst, 1000.0)
		assert.Equal(t, 6, n)
		assert.Equal(t, data, dst)
	}

	// Multichannel (3 channels)
	{
		data := []int{100, 500, 900, -200, -600, -1000, 300, 700, 1100}
		invMax := 1.0 / 1000.0
		chBufs := [][]float64{make([]float64, 3), make([]float64, 3), make([]float64, 3)}
		deinterleaveInto(data, chBufs, 3, 3, invMax)
		assert.InDeltaSlice(t, []float64{0.1, -0.2, 0.3}, chBufs[0], 1e-9)
		assert.InDeltaSlice(t, []float64{0.5, -0.6, 0.7}, chBufs[1], 1e-9)
		assert.InDeltaSlice(t, []float64{0.9, -1.0, 1.1}, chBufs[2], 1e-9) // note 1.1 clamp will occur in interleave

		dst := make([]int, 9)
		n := interleaveInto(chBufs, dst, 1000.0)
		assert.Equal(t, 9, n)
		// 1.1 in chBufs[2][2] clamps to 1.0 * 1000 = 1000 instead of 1100
		expectedData := []int{100, 500, 900, -200, -600, -1000, 300, 700, 1000}
		assert.Equal(t, expectedData, dst)
	}

	// InterleaveInto empty channels
	assert.Equal(t, 0, interleaveInto[float64](nil, nil, 1.0))
	assert.Equal(t, 0, interleaveInto([][]float64{}, nil, 1.0))

	// InterleaveInto dst too small
	assert.Equal(t, 0, interleaveInto([][]float64{{0.5}}, make([]int, 0), 1000.0))
}

func TestFastWAVWriter(t *testing.T) {
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "fast_writer_test.wav")

	f, err := os.Create(outputPath)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	writer, err := newFastWAVWriter(f, 44100, 24, 2)
	require.NoError(t, err)

	// write sample data (24-bit needs WriteSamples24)
	samples := []int{10000, -20000, 30000, -40000}
	err = writer.WriteSamples(samples)
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)
	_ = f.Close()

	// Verify WAV file format
	rf, err := os.Open(outputPath)
	require.NoError(t, err)
	defer func() { _ = rf.Close() }()

	dec := wav.NewDecoder(rf)
	require.True(t, dec.IsValidFile())
	assert.Equal(t, uint32(44100), dec.SampleRate)
	assert.Equal(t, uint16(24), dec.BitDepth)
	assert.Equal(t, uint16(2), dec.NumChans)
}

func TestFastWAVWriter_32Bit(t *testing.T) {
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "fast_writer_32_test.wav")

	f, err := os.Create(outputPath)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	writer, err := newFastWAVWriter(f, 48000, 32, 1)
	require.NoError(t, err)

	samples := []int{1000000, -2000000}
	err = writer.WriteSamples(samples)
	require.NoError(t, err)

	err = writer.Close()
	require.NoError(t, err)
	_ = f.Close()

	// Verify
	rf, err := os.Open(outputPath)
	require.NoError(t, err)
	defer func() { _ = rf.Close() }()

	dec := wav.NewDecoder(rf)
	require.True(t, dec.IsValidFile())
	assert.Equal(t, uint32(48000), dec.SampleRate)
	assert.Equal(t, uint16(32), dec.BitDepth)
}

func TestNewFastWAVWriter_Validation(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name      string
		bitDepth  int
		channels  int
		wantError bool
	}{
		{"valid 16-bit stereo", 16, 2, false},
		{"valid 24-bit mono", 24, 1, false},
		{"valid 32-bit stereo", 32, 2, false},
		{"unsupported 8-bit", 8, 2, true},
		{"unsupported 20-bit", 20, 2, true},
		{"zero bit depth", 0, 2, true},
		{"zero channels", 16, 0, true},
		{"negative channels", 16, -1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := os.Create(filepath.Join(tmpDir, "validate.wav"))
			require.NoError(t, err)
			defer func() { _ = f.Close() }()

			w, err := newFastWAVWriter(f, 44100, tt.bitDepth, tt.channels)
			if tt.wantError {
				require.Error(t, err)
				assert.Nil(t, w)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, w)
			}
		})
	}
}

func TestWriteSamples_UnsupportedBitDepth(t *testing.T) {
	// Construct a writer with an unsupported bit depth directly to exercise the
	// defensive default branch; newFastWAVWriter normally rejects this upfront.
	w := &fastWAVWriter{bitDepth: 8}
	err := w.WriteSamples([]int{1, 2, 3})
	require.Error(t, err)
}

func TestResampleWAVGeneric_AlreadyAtTargetRate(t *testing.T) {
	tmpDir := t.TempDir()
	inputPath := filepath.Join(tmpDir, "input_same.wav")
	outputPath := filepath.Join(tmpDir, "output_same.wav")

	createTempWAV(t, inputPath, 44100, 2, 16, 100)

	_, err := resampleWAVFloat64(inputPath, outputPath, 44100, engine.QualityHigh, false, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "input already at target rate")
}

func TestResampleWAVGeneric_Success(t *testing.T) {
	tmpDir := t.TempDir()
	inputPath := filepath.Join(tmpDir, "input.wav")
	outputPath := filepath.Join(tmpDir, "output.wav")

	// Create 1000 samples of 44100Hz stereo 16-bit audio
	createTempWAV(t, inputPath, 44100, 2, 16, 1000)

	// Resample to 48000Hz (Float64, Parallel)
	stats, err := resampleWAVFloat64(inputPath, outputPath, 48000, engine.QualityHigh, true, true)
	require.NoError(t, err)
	require.NotNil(t, stats)

	assert.Equal(t, 44100, stats.inputRate)
	assert.Equal(t, 48000, stats.outputRate)
	assert.Equal(t, 2, stats.channels)
	assert.Equal(t, 16, stats.bitDepth)
	assert.Equal(t, int64(1000), stats.inputSamples)
	assert.Greater(t, stats.outputSamples, int64(1000))

	// Resample to 22050Hz (Float32, Sequential)
	outputPathF32 := filepath.Join(tmpDir, "output_f32.wav")
	stats32, err := resampleWAVFloat32(inputPath, outputPathF32, 22050, engine.QualityMedium, true, false)
	require.NoError(t, err)
	require.NotNil(t, stats32)
	assert.Equal(t, 22050, stats32.outputRate)
}

func TestRun_CLI(t *testing.T) {
	tmpDir := t.TempDir()
	inputPath := filepath.Join(tmpDir, "cli_input.wav")
	outputPath := filepath.Join(tmpDir, "cli_output.wav")

	createTempWAV(t, inputPath, 44100, 1, 16, 500)

	// Save original args/flags
	origArgs := os.Args
	defer func() {
		os.Args = origArgs
	}()

	// 1. Test insufficient arguments
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{origArgs[0]}
	err := run()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient arguments")

	// 2. Test successful execution
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	cpuProfilePath := filepath.Join(tmpDir, "cpu.prof")
	os.Args = []string{
		origArgs[0],
		"-rate", "32.0",
		"-quality", "medium",
		"-v",
		"-cpuprofile", cpuProfilePath,
		inputPath,
		outputPath,
	}

	err = run()
	require.NoError(t, err)

	// Verify output
	_, err = os.Stat(outputPath)
	require.NoError(t, err)

	// Verify CPU profile created
	_, err = os.Stat(cpuProfilePath)
	require.NoError(t, err)
}
