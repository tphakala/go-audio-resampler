// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCubicAndLinearStageGetters(t *testing.T) {
	// CubicStage float64
	cs := NewCubicStage[float64](1.5)
	assert.InDelta(t, 1.5, cs.GetRatio(), 1e-9)
	assert.Equal(t, cubicLatencySamples, cs.GetLatency())
	assert.Equal(t, 1, cs.GetMinInput())
	assert.Equal(t, int64(cubicMemoryUsage), cs.GetMemoryUsage())
	assert.Equal(t, 4, cs.GetFilterLength()) // Hermite cubic interpolation uses 4 points
	assert.Equal(t, 0, cs.GetPhases())
	assert.Empty(t, cs.GetSIMDInfo())

	// LinearStage
	ls := NewLinearStage(1.5)
	assert.InDelta(t, 1.5, ls.GetRatio(), 1e-9)
	assert.Equal(t, linearLatencySamples, ls.GetLatency())
	assert.Equal(t, 1, ls.GetMinInput())
	assert.Equal(t, int64(linearMemoryUsage), ls.GetMemoryUsage())
	assert.Equal(t, 2, ls.GetFilterLength()) // Linear interpolation uses 2 points
	assert.Equal(t, 0, ls.GetPhases())
	assert.Empty(t, ls.GetSIMDInfo())

	// LinearStage Flush
	flushed, err := ls.Flush()
	require.NoError(t, err)
	assert.Empty(t, flushed)
}

func TestStageAdapterGetters(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	sa := NewStageAdapter(r)
	assert.InDelta(t, 48000.0/44100.0, sa.GetRatio(), 1e-9)
	assert.Positive(t, sa.GetLatency())
	assert.Equal(t, 1, sa.GetMinInput())
	assert.Positive(t, sa.GetMemoryUsage())
	assert.Positive(t, sa.GetFilterLength())
	assert.Positive(t, sa.GetPhases())
	assert.NotNil(t, sa.GetSIMDInfo())

	// DFT pre-stage but no polyphase
	r2, err := NewResampler[float64](22050, 44100, QualityHigh)
	require.NoError(t, err)
	sa2 := NewStageAdapter(r2)
	assert.Positive(t, sa2.GetLatency())
	assert.Positive(t, sa2.GetMemoryUsage())
	assert.Positive(t, sa2.GetFilterLength())
	assert.Equal(t, 0, sa2.GetPhases())
}

func TestResamplerProcessZeroCopy(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	// Make input larger than filter latency (1000 samples)
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 0.1
	}

	output, err := r.ProcessZeroCopy(input)
	require.NoError(t, err)
	assert.NotEmpty(t, output)

	// Cubic stage ProcessZeroCopy
	rc, err := NewResampler[float64](44100, 48000, QualityQuick)
	require.NoError(t, err)
	outputC, err := rc.ProcessZeroCopy(input)
	require.NoError(t, err)
	assert.NotEmpty(t, outputC)
}

func TestResamplerLargeInputs_DFTChunked(t *testing.T) {
	// Float64 upsampling to trigger DFTStage chunked path
	r64, err := NewResampler[float64](22050, 44100, QualityHigh)
	require.NoError(t, err)

	largeInput64 := make([]float64, 5000)
	for i := range largeInput64 {
		largeInput64[i] = 0.05
	}

	out64, err := r64.Process(largeInput64)
	require.NoError(t, err)
	// 5000 input samples - 166 tapsPerPhase + 1 = 4835 processable frames
	// 4835 * 2 upsampling factor = 9670 output samples
	assert.Len(t, out64, 9670)

	// Float32 upsampling to trigger DFTStage chunked path
	r32, err := NewResampler[float32](22050, 44100, QualityHigh)
	require.NoError(t, err)

	largeInput32 := make([]float32, 5000)
	for i := range largeInput32 {
		largeInput32[i] = 0.05
	}

	out32, err := r32.Process(largeInput32)
	require.NoError(t, err)
	assert.Len(t, out32, 9670)

	// DFT Decimation stage large input
	rDecim, err := NewResampler[float64](44100, 22050, QualityHigh)
	require.NoError(t, err)

	outDecim, err := rDecim.Process(largeInput64)
	require.NoError(t, err)
	// (5000 input - 750 tapsPerPhase + 1) / 2 = 2125 output samples
	assert.Len(t, outDecim, 2125)
}
